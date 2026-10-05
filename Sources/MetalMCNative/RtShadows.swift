import Foundation
import Metal
import simd

// Ray-traced sun shadows (METALMC_EXP=rtshadows, prototype). The LOD's geometry, which includes the terrain vanilla
// draws near the player (level 0 is its blocks), goes into one primitive acceleration structure per drawn tile, built
// in the background as tiles appear. Each frame an instance structure over the tiles the LOD drew is built, and after
// the level is drawn a compute pass casts one ray toward the sun per 2 x 2 pixels from the depth buffer. Surfaces
// facing away from the sun count as shadowed, so it also gives the world directional sunlight. A fullscreen pass then
// darkens the shadowed pixels before the temporal anti-aliasing, which smooths the half-resolution result and, with
// the ray jittered across the sun's disk every frame, softens the shadows' edges.
//
// Lit mode (METALMC_EXP=lit, Lit.swift): the kernel stores the raw visibility instead (1 lit, 0 shadowed, toward 1 with
// distance where the haze takes over), toward the sun, or toward the moon while the sun is down, and nothing darkens the
// color: the relight multiplies the sun's (or moon's) light by it, and the anti-aliasing doesn't apply it again.
//
// With the GI cache (METALMC_EXP=lit,gi, Gi.swift): each tile's structure has one geometry per quad range of its node's
// buffer (the opaque faces, then the tile-edge skirts: giTileMesh; the same triangles), and the instance structure comes
// with a table of each instance's node buffer address and ranges (GiTile), so a bounce ray reads what it hit from the
// quad itself: no per-triangle data in the structures. After the shadow rays the cache runs its frame (GiCache
// encodeFrame) on the same depth, projection, origin and instance structure, and leaves its half-resolution light for
// the relight like the visibility.

/// On with `shadows=true` in config/metalmc.properties (mmc_set_rt_shadows at startup) or METALMC_EXP=rtshadows.
nonisolated(unsafe) var lodRtShadows = experiments.contains("rtshadows")

@_cdecl("mmc_set_rt_shadows")
public func mmc_set_rt_shadows(_ on: Int32) {
    if on != 0 { lodRtShadows = true }
}

private let rtShadowSource = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct ShadowParams {
    float4x4 invViewProj;   // inverse of the (jittered) projection * view rotation the level was drawn with
    float4 sun;             // xyz: direction to the sun, w: the cloud layer's bottom, camera-relative
    float4 sizes;           // full width, full height, traced width, traced height
    float4 disk;            // xy: this frame's offset across the sun's disk (tangent plane), z: max ray length, w: 1 = no rays (debug)
    float4 camOffset;       // xyz: the camera's position relative to the instance structure's origin
    float4 shade;           // x: strength, y-z: the distance range over which shadows fade out
    uint4 sample;           // x: pixels per traced sample along each axis (2 or 4), y-z: this frame's pixel within the block
};

// What the kernel stores: the factor the pixel's color is multiplied by (1 lit, 1 - strength in full shadow, less far
// away where the haze takes over).
static half shadeOf(constant ShadowParams& p, float3 pos, float lit) {
    float s = p.shade.x * (1.0 - smoothstep(p.shade.y, p.shade.z, length(pos)));
    return half(mix(1.0 - s, 1.0, lit));
}

static float3 relAt(constant ShadowParams& p, uint2 q, float z) {
    float2 uv = (float2(q) + 0.5) / p.sizes.xy;
    float4 h = p.invViewProj * float4(uv * 2.0 - 1.0, z, 1.0);
    return h.xyz / h.w;
}

kernel void rt_shadow(instance_acceleration_structure accel [[buffer(0)]],
                      constant ShadowParams& p [[buffer(1)]],
                      depth2d<float, access::read> depth [[texture(0)]],
                      texture2d<half, access::write> out [[texture(1)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(p.sizes.z) || gid.y >= uint(p.sizes.w)) return;
    uint2 full = uint2(p.sizes.xy);
    // One ray per scale x scale block, from a different pixel of it each frame (the anti-aliasing accumulates them).
    uint2 fp = min(gid * p.sample.x + p.sample.yz, full - 1);
    float d = depth.read(fp);
    if (d <= 0.0) { out.write(half4(1.0), gid); return; }   // sky: reverse-Z puts the far plane at 0
    float3 pos = relAt(p, fp, d);
    // Vanilla's clouds (4 blocks thick) write depth but get no shadows: their undersides face away from the sun.
    if (pos.y > p.sun.w - 0.1 && pos.y < p.sun.w + 4.1) { out.write(half4(1.0), gid); return; }
    // The surface's normal from the neighbors, taking on each axis the one on the same surface (closest depth), then
    // snapped to the nearest axis: nearly everything in this world is axis-aligned.
    uint2 rx = uint2(min(fp.x + 1, full.x - 1), fp.y), lx = uint2(fp.x > 0 ? fp.x - 1 : 0, fp.y);
    uint2 dy = uint2(fp.x, min(fp.y + 1, full.y - 1)), uy = uint2(fp.x, fp.y > 0 ? fp.y - 1 : 0);
    float drx = depth.read(rx), dlx = depth.read(lx), ddy = depth.read(dy), duy = depth.read(uy);
    float3 ex = abs(drx - d) < abs(dlx - d) ? relAt(p, rx, drx) - pos : pos - relAt(p, lx, dlx);
    float3 ey = abs(ddy - d) < abs(duy - d) ? relAt(p, dy, ddy) - pos : pos - relAt(p, uy, duy);
    float3 n = cross(ey, ex);
    if (dot(n, -pos) < 0.0) n = -n;
    float3 an = abs(n);
    n = an.x > an.y && an.x > an.z ? float3(sign(n.x), 0, 0) : (an.y > an.z ? float3(0, sign(n.y), 0) : float3(0, 0, sign(n.z)));
    // Clearly facing away from the sun: in shadow. Faces edge-on to it (the sides of blocks at noon) get a ray like the
    // rest, so they only darken where something is actually in the way; vanilla's face shading already dims them.
    float ndl = dot(n, p.sun.xyz);
    if (ndl < -0.05) { out.write(half4(shadeOf(p, pos, 0.0)), gid); return; }
    // A ray toward a point on the sun's disk, from just off the surface (farther off with distance: depth precision).
    float3 t1 = normalize(cross(p.sun.xyz, float3(0, 0, 1)));
    float3 t2 = cross(p.sun.xyz, t1);
    if (p.disk.w > 0.5) { out.write(half4(1.0), gid); return; }
    ray r;
    r.origin = pos + n * (0.03 + length(pos) * 0.0008) + p.camOffset.xyz;
    r.direction = normalize(p.sun.xyz + t1 * p.disk.x + t2 * p.disk.y);
    r.min_distance = 0.0;
    r.max_distance = p.disk.z;
    intersector<instancing> isect;
    isect.accept_any_intersection(true);
    isect.assume_geometry_type(geometry_type::triangle);
    auto hit = isect.intersect(r, accel, 0xFF);
    out.write(half4(shadeOf(p, pos, hit.type == intersection_type::none ? 1.0 : 0.0)), gid);
}

struct ApplyOut { float4 pos [[position]]; };
vertex ApplyOut rt_shadow_vs(uint vid [[vertex_id]]) {
    ApplyOut o;
    float2 c = float2((vid << 1) & 2, vid & 2);
    o.pos = float4(c * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// Multiplies the color target by the shade stored for the pixel's 2 x 2 block (without anti-aliasing; with it, the
// anti-aliasing does this as it loads the color).
fragment half4 rt_shadow_fs(ApplyOut in [[stage_in]], constant uint& scale [[buffer(0)]], texture2d<half, access::read> lit [[texture(1)]]) {
    uint2 hc = min(uint2(in.pos.xy) / scale, uint2(lit.get_width() - 1, lit.get_height() - 1));
    return half4(half3(lit.read(hc).r), 1.0h);
}
"""

private struct ShadowParams {
    var invViewProj: simd_float4x4
    var sun: SIMD4<Float>
    var sizes: SIMD4<Float>
    var disk: SIMD4<Float>
    var camOffset: SIMD4<Float> = .zero
    var shade: SIMD4<Float> = .zero
    var sample: SIMD4<UInt32> = .zero
}

/// Pixels per traced shadow sample along each axis: 4 (quarter resolution, about 0.5 ms at the panel's resolution; the
/// anti-aliasing accumulates the rotating samples, and on the real-terrain tours it looked the same as half resolution)
/// or 2 (METALMC_SHADOWSCALE=2, about 1 ms).
let rtShadowScale = Int(ProcessInfo.processInfo.environment["METALMC_SHADOWSCALE"] ?? "") == 2 ? 2 : 4

private struct TileKey: Hashable { let node: ObjectIdentifier; let tile: Int }
/// A tile's structure; a class so marking it used needs no dictionary write.
private final class BlasEntry {
    let node: LodMeshNode   // kept alive so its identifier can't be reused while the entry exists
    let accel: MTLAccelerationStructure
    var lastUsed: UInt64
    /// With the GI cache: the structure's geometries' quad ranges in the node's buffer (giTileMesh).
    var giRanges: [(first: Int, count: Int)] = []
    init(_ node: LodMeshNode, _ accel: MTLAccelerationStructure, _ lastUsed: UInt64) { self.node = node; self.accel = accel; self.lastUsed = lastUsed }
}

final class RtShadows: @unchecked Sendable {
    static let shared = RtShadows()
    private let lock = NSLock()
    private var blas: [TileKey: BlasEntry] = [:]
    private var building: Set<TileKey> = []
    private var buildQueue: MTLCommandQueue?
    private let cpuQueue = DispatchQueue(label: "metalmc.rtshadows", qos: .utility)
    private var library: MTLLibrary?
    private var kernel: MTLComputePipelineState?
    private var applyPipes: [UInt: MTLRenderPipelineState] = [:]
    private var litTexture: MTLTexture?
    private var tlasBuffer: (accel: MTLAccelerationStructure, size: Int)?
    private var scratch: MTLBuffer?
    private var frame: UInt64 = 0
    var chosen: [(LodMeshNode, UInt16)] = []   // the tiles the LOD drew this frame (set by mmc_lod_draw)
    // Shadows traced this frame and left for the anti-aliasing to apply (it reads every pixel's color anyway).
    private var deferred: (lit: MTLTexture, params: SIMD4<Float>, width: Int, height: Int)?
    private var dummy: MTLTexture?
    // The instance structure holds the tiles relative to a fixed origin, moved when the camera gets 1 km away, so it
    // is only rebuilt when the set of tiles changes, not every frame; rays start at their camera-relative position
    // plus the camera's offset from that origin.
    private var origin = SIMD3<Double>(0, 0, 0)
    private var instanceKey: [ObjectIdentifier] = []
    private var tlasAccels: [MTLAccelerationStructure] = []   // the structures the current instance structure references
    /// With the GI cache: the current instance structure's GiTile table and the node buffers it points into.
    private var tlasGi: (table: MTLBuffer, buffers: [MTLBuffer])?
    /// With the GI cache: its half-resolution light and code words this frame (one RG32Uint texel each, gi_resolve), and
    /// the sun and sky light it took from the atmosphere (GiCache.envThisFrame), for the relight (takeLitGi).
    private var giOut: MTLTexture?
    private var litGiOut: (gi: MTLTexture, env: MTLBuffer?, width: Int, height: Int)?
    private var giRuns = 0
    /// Offline timing (mmc_debug_gi_frame): 0 skips the cache's frame.
    var giOn = true

    /// Lit mode with the GI cache: this frame's light from it (gi_resolve's output, half the frame's size) and, if it
    /// made them from the atmosphere, the relight's sun and sky light (lit_env's output).
    func takeLitGi(width: Int, height: Int) -> (gi: MTLTexture, env: MTLBuffer?)? {
        defer { litGiOut = nil }
        guard let g = litGiOut, g.width == width, g.height == height else { return nil }
        return (g.gi, g.env)
    }

    func takeDeferred(width: Int, height: Int) -> (lit: MTLTexture, params: SIMD4<Float>)? {
        defer { deferred = nil }
        guard let d = deferred, d.width == width, d.height == height else { return nil }
        return (d.lit, d.params)
    }

    /// Lit mode: this frame's raw visibility for the relight (Lit.swift): the texture, pixels per sample along each axis,
    /// and whether it was traced toward the moon.
    private var litVisibility: (texture: MTLTexture, scale: Int, moon: Bool, width: Int, height: Int)?

    func takeLitVisibility(width: Int, height: Int) -> (texture: MTLTexture, scale: Int, moon: Bool)? {
        defer { litVisibility = nil }
        guard let v = litVisibility, v.width == width, v.height == height else { return nil }
        return (v.texture, v.scale, v.moon)
    }

    func dummyLit() -> MTLTexture {
        if let dummy { return dummy }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: 1, height: 1, mipmapped: false)
        d.usage = .shaderRead
        let t = ctx.device.makeTexture(descriptor: d)!
        dummy = t
        return t
    }

    private func ensurePipelines(format: MTLPixelFormat) -> Bool {
        if library == nil {
            // Lab mode (ShaderLab.swift): after an edit, forget the kernel and the pipelines; the next frame rebuilds them.
            do { library = try ShaderLab.library("rt_shadows", rtShadowSource) { [self] _ in library = nil; kernel = nil; applyPipes = [:] } } catch {
                log("rt shadows: library failed: \(error)")
                return false
            }
            if let f = library?.makeFunction(name: "rt_shadow") { kernel = try? ctx.device.makeComputePipelineState(function: f) }
            if buildQueue == nil { buildQueue = ctx.device.makeCommandQueue() }
        }
        if applyPipes[format.rawValue] == nil, let lib = library {
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = lib.makeFunction(name: "rt_shadow_vs")
            d.fragmentFunction = lib.makeFunction(name: "rt_shadow_fs")
            d.colorAttachments[0].pixelFormat = format
            let a = d.colorAttachments[0]!
            a.isBlendingEnabled = true
            a.rgbBlendOperation = .add
            a.sourceRGBBlendFactor = .destinationColor   // result = shade * destination
            a.destinationRGBBlendFactor = .zero
            a.alphaBlendOperation = .add
            a.sourceAlphaBlendFactor = .zero
            a.destinationAlphaBlendFactor = .one
            applyPipes[format.rawValue] = try? ctx.device.makeRenderPipelineState(descriptor: d)
        }
        return kernel != nil && applyPipes[format.rawValue] != nil
    }

    /// Queues acceleration-structure builds for drawn tiles that have none (nearest first, a few at a time).
    private func scheduleBuilds(camX: Double, camZ: Double) {
        var wanted: [(TileKey, LodMeshNode, Int, Double)] = []
        for (n, mask) in chosen {
            let id = ObjectIdentifier(n)
            let tileSize = Double(lodTileVoxels << n.level)
            for t in 0..<(lodTilesPerSide * lodTilesPerSide) where mask & (1 << UInt16(t)) != 0 {
                let key = TileKey(node: id, tile: t)
                if blas[key] != nil || building.contains(key) { continue }
                let cxT = Double(n.x0) + (Double(t % lodTilesPerSide) + 0.5) * tileSize
                let czT = Double(n.z0) + (Double(t / lodTilesPerSide) + 0.5) * tileSize
                wanted.append((key, n, t, (cxT - camX) * (cxT - camX) + (czT - camZ) * (czT - camZ)))
            }
        }
        if wanted.isEmpty { return }
        wanted.sort { $0.3 < $1.3 }
        for (key, n, t, _) in wanted.prefix(max(0, 8 - building.count)) {
            building.insert(key)
            cpuQueue.async { [self] in build(key, n, t) }
        }
    }

    /// One tile's quads (water aside: it doesn't block the sun) as triangles in node-local blocks, then its structure.
    private func build(_ key: TileKey, _ n: LodMeshNode, _ t: Int) {
        let q = n.buffer.contents().bindMemory(to: UInt32.self, capacity: 2 * n.quadCount)
        let scale = Float(1 << n.level)
        let corners: [[SIMD3<Float>]] = [
            [SIMD3(1, 0, 0), SIMD3(1, 1, 0), SIMD3(1, 1, 1), SIMD3(1, 0, 1)],
            [SIMD3(0, 0, 1), SIMD3(0, 1, 1), SIMD3(0, 1, 0), SIMD3(0, 0, 0)],
            [SIMD3(0, 1, 0), SIMD3(0, 1, 1), SIMD3(1, 1, 1), SIMD3(1, 1, 0)],
            [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 0, 1), SIMD3(0, 0, 1)],
            [SIMD3(1, 0, 1), SIMD3(1, 1, 1), SIMD3(0, 1, 1), SIMD3(0, 0, 1)],
            [SIMD3(0, 0, 0), SIMD3(0, 1, 0), SIMD3(1, 1, 0), SIMD3(1, 0, 0)],
        ]
        var verts: [Float] = [], idx: [UInt32] = []
        var giRanges: [(first: Int, count: Int)] = []
        if litGi {
            // The same quads, one geometry per quad range of the node's buffer: the GI cache's bounce rays find the quad
            // they hit from the geometry and primitive index (zero copy).
            (verts, idx, giRanges) = giTileMesh(quads: q, start: n.start, tile: t, level: n.level)
        } else {
            for k in 0..<lodBucketsPerTile where k < 6 || k >= 12 {
                let a = n.start[lodBucketIndex(t, k, 0)], b = n.start[lodBucketIndex(t, k, lodSubtilesPerTile)]
                for i in a..<b {
                    let w0 = q[2 * i], w1 = q[2 * i + 1]
                    let face = Int((w0 >> 25) & 7)
                    if face > 5 { continue }
                    let local = SIMD3(Float(w0 & 255), Float((w0 >> 16) & 511), Float((w0 >> 8) & 255))
                    let qw = Float(((w1 >> 8) & 255) + 1), qh = Float(((w1 >> 16) & 255) + 1)
                    let ext: SIMD3<Float> = face < 2 ? SIMD3(1, qw, qh) : (face < 4 ? SIMD3(qw, 1, qh) : SIMD3(qw, qh, 1))
                    let base = UInt32(verts.count / 3)
                    for c in corners[face] {
                        let p = (local + c * ext) * scale
                        verts += [p.x, p.y, p.z]
                    }
                    idx += [base, base + 1, base + 2, base, base + 2, base + 3]
                }
            }
        }
        let dev = ctx.device
        guard !idx.isEmpty, let queue = buildQueue,
              let vb = dev.makeBuffer(bytes: verts, length: verts.count * 4, options: .storageModeShared),
              let ib = dev.makeBuffer(bytes: idx, length: idx.count * 4, options: .storageModeShared) else {
            lock.lock(); building.remove(key); lock.unlock()
            return
        }
        let g = MTLAccelerationStructureTriangleGeometryDescriptor()
        g.vertexBuffer = vb
        g.vertexStride = 12
        g.vertexFormat = .float3
        g.indexBuffer = ib
        g.indexType = .uint32
        g.triangleCount = idx.count / 3
        g.opaque = true
        let d = MTLPrimitiveAccelerationStructureDescriptor()
        d.geometryDescriptors = litGi ? giGeometries(vb: vb, ib: ib, triangles: idx.count / 3, ranges: giRanges) : [g]
        let sizes = dev.accelerationStructureSizes(descriptor: d)
        guard let accel = dev.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = dev.makeBuffer(length: max(sizes.buildScratchBufferSize, 16), options: .storageModePrivate),
              let sizeBuf = dev.makeBuffer(length: 8, options: .storageModeShared),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeAccelerationStructureCommandEncoder() else {
            lock.lock(); building.remove(key); lock.unlock()
            return
        }
        enc.build(accelerationStructure: accel, descriptor: d, scratchBuffer: scratch, scratchBufferOffset: 0)
        enc.writeCompactedSize(accelerationStructure: accel, buffer: sizeBuf, offset: 0, sizeDataType: .ulong)
        enc.endEncoding()
        let ranges = giRanges
        cb.addCompletedHandler { [self] _ in
            // Then a compacted copy (about half the memory, same speed), from the build queue's own thread.
            cpuQueue.async { [self] in
                var final = accel
                let compacted = Int(sizeBuf.contents().load(as: UInt64.self))
                if compacted > 0, let small = dev.makeAccelerationStructure(size: compacted),
                   let ccb = queue.makeCommandBuffer(), let cenc = ccb.makeAccelerationStructureCommandEncoder() {
                    cenc.copyAndCompact(sourceAccelerationStructure: accel, destinationAccelerationStructure: small)
                    cenc.endEncoding()
                    ccb.commit()
                    ccb.waitUntilCompleted()
                    final = small
                }
                lock.lock()
                building.remove(key)
                let entry = BlasEntry(n, final, frame)
                entry.giRanges = ranges
                blas[key] = entry
                lock.unlock()
            }
        }
        cb.commit()
    }

    /// Encodes the shadow passes into the current command buffer. `p`: 32 floats, projection (as drawn, jittered)
    /// then view rotation. Returns false if nothing was done.
    func apply(color: MTLTexture, depth: MTLTexture, p: UnsafePointer<Float>, cam: SIMD3<Double>, sunAngle: Float, strength: Float,
               cloudHeight: Float, deferToTaa: Bool) -> Bool {
        guard ctx.device.supportsRaytracing, ctx.pass == nil, ensurePipelines(format: color.pixelFormat),
              let kernel, let applyPipe = applyPipes[color.pixelFormat.rawValue] else { return false }
        frame += 1
        lock.lock()
        if frame % 8 == 0 { scheduleBuilds(camX: cam.x, camZ: cam.z) }
        if simd_length(cam - origin) > 1024 { origin = (cam / 256).rounded(.down) * 256; instanceKey = [] }
        let t0 = DispatchTime.now().uptimeNanoseconds
        // Instances: the drawn tiles whose structures are ready. Rebuilt only when the drawn tiles, the ready structures or
        // the origin changed (the common case in flight is none of them).
        var chosenKey = [Int](); chosenKey.reserveCapacity(chosen.count * 2)
        for (n, mask) in chosen { chosenKey.append(ObjectIdentifier(n).hashValue); chosenKey.append(Int(mask)) }
        chosenKey.append(blas.count)
        if chosenKey == lastChosenKey, let cached = cachedInstances {
            lock.unlock()
            return trace(color: color, depth: depth, p: p, cam: cam, sunAngle: sunAngle, strength: strength, cloudHeight: cloudHeight,
                         deferToTaa: deferToTaa, instances: cached.instances, accels: cached.accels, gi: cached.gi, cpuStart: t0)
        }
        lastChosenKey = chosenKey
        var accels: [MTLAccelerationStructure] = []
        var instances: [MTLAccelerationStructureInstanceDescriptor] = []
        // With the GI cache: each instance's GiTile (its node's buffer and quad ranges) and the node buffers.
        var giTiles: [SIMD4<UInt32>] = [], giBuffers: [MTLBuffer] = []
        for (n, mask) in chosen {
            let id = ObjectIdentifier(n)
            let at = SIMD3(Float(Double(n.x0) - origin.x), Float(Double(lodWorldMinY) - origin.y), Float(Double(n.z0) - origin.z))
            var nodeUsed = false
            for t in 0..<(lodTilesPerSide * lodTilesPerSide) where mask & (1 << UInt16(t)) != 0 {
                guard let e = blas[TileKey(node: id, tile: t)] else { continue }
                e.lastUsed = frame
                if litGi {
                    giTiles.append(giTileEntry(n.buffer, e.giRanges))
                    if !nodeUsed { giBuffers.append(n.buffer); nodeUsed = true }
                }
                let ai = UInt32(accels.count)
                accels.append(e.accel)
                var inst = MTLAccelerationStructureInstanceDescriptor()
                inst.transformationMatrix = MTLPackedFloat4x3(columns: (MTLPackedFloat3Make(1, 0, 0), MTLPackedFloat3Make(0, 1, 0),
                                                                        MTLPackedFloat3Make(0, 0, 1), MTLPackedFloat3Make(at.x, at.y, at.z)))
                inst.options = .opaque
                inst.mask = 0xFF
                inst.intersectionFunctionTableOffset = 0
                inst.accelerationStructureIndex = ai
                instances.append(inst)
            }
        }
        // Structures unused for 10 s are released.
        if frame % 600 == 0 { blas = blas.filter { frame - $0.value.lastUsed < 1200 } }
        lock.unlock()
        cachedInstances = (instances, accels, (giTiles, giBuffers))
        return trace(color: color, depth: depth, p: p, cam: cam, sunAngle: sunAngle, strength: strength, cloudHeight: cloudHeight,
                     deferToTaa: deferToTaa, instances: instances, accels: accels, gi: (giTiles, giBuffers), cpuStart: t0)
    }

    /// A 4 x 4 visiting order that spreads consecutive frames' samples apart (a Bayer order).
    private static let pattern: [SIMD2<UInt32>] = [0, 10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 14, 6, 12].map { SIMD2(UInt32($0 % 4), UInt32($0 / 4)) }
    private var lastChosenKey: [Int] = []
    private var cachedInstances: (instances: [MTLAccelerationStructureInstanceDescriptor], accels: [MTLAccelerationStructure],
                                  gi: (tiles: [SIMD4<UInt32>], buffers: [MTLBuffer]))?
    private var rebuilds = 0

    private func trace(color: MTLTexture, depth: MTLTexture, p: UnsafePointer<Float>, cam: SIMD3<Double>, sunAngle: Float, strength: Float,
                       cloudHeight: Float, deferToTaa: Bool, instances: [MTLAccelerationStructureInstanceDescriptor],
                       accels: [MTLAccelerationStructure], gi: (tiles: [SIMD4<UInt32>], buffers: [MTLBuffer]), cpuStart: UInt64) -> Bool {
        guard let kernel, let applyPipe = applyPipes[color.pixelFormat.rawValue] else { return false }
        if instances.isEmpty { return false }
        // Rebuild only when the set of tile structures (or the origin) changed.
        let key = accels.map { ObjectIdentifier($0) }
        let rebuild = key != instanceKey
        if rebuild { rebuilds += 1 }
        let dev = ctx.device
        // The instance buffer and descriptor are only needed to rebuild (buffer allocation and the size query each cost
        // CPU time every frame otherwise).
        var td: MTLInstanceAccelerationStructureDescriptor?
        if rebuild || tlasBuffer == nil {
            guard let instBuf = dev.makeBuffer(bytes: instances, length: instances.count * MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride,
                                               options: .storageModeShared) else { return false }
            let d = MTLInstanceAccelerationStructureDescriptor()
            d.instancedAccelerationStructures = accels
            d.instanceCount = instances.count
            d.instanceDescriptorBuffer = instBuf
            d.instanceDescriptorType = .default
            let sizes = dev.accelerationStructureSizes(descriptor: d)
            if tlasBuffer == nil || tlasBuffer!.size < sizes.accelerationStructureSize {
                guard let a = dev.makeAccelerationStructure(size: sizes.accelerationStructureSize * 2) else { return false }
                tlasBuffer = (a, sizes.accelerationStructureSize * 2)
            }
            if scratch == nil || scratch!.length < sizes.buildScratchBufferSize {
                scratch = dev.makeBuffer(length: max(sizes.buildScratchBufferSize * 2, 256), options: .storageModePrivate)
            }
            td = d
        }
        guard let tlas = tlasBuffer?.accel else { return false }
        let sc = rtShadowScale
        let w = color.width, h = color.height, hw = (w + sc - 1) / sc, hh = (h + sc - 1) / sc
        if litTexture == nil || litTexture!.width != hw || litTexture!.height != hh {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: hw, height: hh, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            litTexture = dev.makeTexture(descriptor: d)
        }
        guard let lit = litTexture else { return false }
        func mat(_ o: Int) -> simd_float4x4 {
            simd_float4x4(SIMD4(p[o], p[o + 1], p[o + 2], p[o + 3]), SIMD4(p[o + 4], p[o + 5], p[o + 6], p[o + 7]),
                          SIMD4(p[o + 8], p[o + 9], p[o + 10], p[o + 11]), SIMD4(p[o + 12], p[o + 13], p[o + 14], p[o + 15]))
        }
        // Vanilla's sun: rotated -90 degrees about y, then by the sun angle about x, from straight up.
        var sun = SIMD3<Float>(-sin(sunAngle), cos(sunAngle), 0)
        // Lit mode: toward the moon (opposite the sun) while the sun is down, and the raw visibility (strength 1).
        let towardMoon = litEnabled && sun.y < -0.02
        if towardMoon { sun = -sun }
        // A point on the sun's disk (radius about 0.6 degrees), a different one every frame (golden-angle spiral).
        let k = Float(frame % 64), rad = 0.0105 * (k / 64).squareRoot(), ang = k * 2.39996
        var params = ShadowParams(invViewProj: (mat(0) * mat(16)).inverse, sun: SIMD4(sun, cloudHeight - Float(cam.y)),
                                  sizes: SIMD4(Float(w), Float(h), Float(hw), Float(hh)),
                                  disk: SIMD4(rad * cos(ang), rad * sin(ang), 4000, experiments.contains("shnoray") ? 1 : 0),
                                  camOffset: SIMD4(Float(cam.x - origin.x), Float(cam.y - origin.y), Float(cam.z - origin.z), 0),
                                  shade: SIMD4(litEnabled ? 1 : strength, 6000, 20000, 0),
                                  sample: SIMD4(UInt32(sc), UInt32(Self.pattern[Int(frame % 16)].x % UInt32(sc)),
                                                UInt32(Self.pattern[Int(frame % 16)].y % UInt32(sc)), 0))
        ctx.endBlit()
        let cb = ctx.ensureCB()
        if let td, let scratch {
            guard let aenc = cb.makeAccelerationStructureCommandEncoder() else { return false }
            aenc.build(accelerationStructure: tlas, descriptor: td, scratchBuffer: scratch, scratchBufferOffset: 0)
            aenc.endEncoding()
            instanceKey = key
            tlasAccels = accels
            if litGi {
                tlasGi = gi.tiles.count == instances.count
                    ? dev.makeBuffer(bytes: gi.tiles, length: gi.tiles.count * 16, options: .storageModeShared).map { ($0, gi.buffers) } : nil
            }
        }
        guard let enc = cb.makeComputeCommandEncoder(descriptor: profComputePass("ray-traced shadows")) else { return false }
        enc.setComputePipelineState(kernel)
        enc.setAccelerationStructure(tlas, bufferIndex: 0)
        enc.useResources(tlasAccels, usage: .read)
        enc.setBytes(&params, length: MemoryLayout<ShadowParams>.stride, index: 1)
        enc.setTexture(depth, index: 0)
        enc.setTexture(lit, index: 1)
        enc.dispatchThreads(MTLSize(width: hw, height: hh, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        // Lit mode with the GI cache: its frame, on the same depth, projection, origin and instance structure.
        if litGi && giOn {
            encodeGi(cb: cb, depth: depth, invViewProj: params.invViewProj, cam: cam, sunAngle: sunAngle, cloudHeight: cloudHeight, tlas: tlas,
                     width: w, height: h)
        }
        if frame % 240 == 0 {
            log(String(format: "rt shadows: %d tiles, %d instance-structure rebuilds in the last 240 frames, %.3f ms CPU", instances.count, rebuilds,
                       Double(DispatchTime.now().uptimeNanoseconds - cpuStart) / 1e6))
            rebuilds = 0
        }
        if litEnabled {
            litVisibility = (lit, sc, towardMoon, w, h)
            return true
        }
        if deferToTaa {
            deferred = (lit, SIMD4(1, Float(sc), 0, 0), w, h)
            return true
        }
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = color
        d.colorAttachments[0].loadAction = .load
        d.colorAttachments[0].storeAction = .store
        guard let renc = cb.makeRenderCommandEncoder(descriptor: d) else { return false }
        renc.setRenderPipelineState(applyPipe)
        var scale = UInt32(sc)
        renc.setFragmentBytes(&scale, length: 4, index: 0)
        renc.setFragmentTexture(lit, index: 1)
        renc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        renc.endEncoding()
        if frame % 240 == 0 {
            lock.lock(); let n = blas.count, b = building.count; lock.unlock()
            log("rt shadows: \(instances.count) tiles traced, \(n) structures, \(b) building")
        }
        return true
    }

    /// Lit mode with the GI cache (METALMC_EXP=lit,gi): its frame after the shadow rays (GiCache.encodeFrame), with the
    /// light the relight had last frame (the atmosphere's or the daylight curve's), its output left for the relight.
    private func encodeGi(cb: MTLCommandBuffer, depth: MTLTexture, invViewProj: simd_float4x4, cam: SIMD3<Double>, sunAngle: Float,
                          cloudHeight: Float, tlas: MTLAccelerationStructure, width w: Int, height h: Int) {
        guard let cache = GiCache.shared, let gi = tlasGi else { return }
        let hw = (w + 1) / 2, hh = (h + 1) / 2
        if giOut == nil || giOut!.width != hw || giOut!.height != hh {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Uint, width: hw, height: hh, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            giOut = ctx.device.makeTexture(descriptor: d)
            giOut?.label = "MetalMC GI cache light"
        }
        guard let out = giOut else { return }
        cache.tiles = gi
        let sun = SIMD3<Float>(-sin(sunAngle), cos(sunAngle), 0)
        let daylight = litDaylightEnv(sunAngle: sunAngle)
        let light: GiLightSource = Lit.shared.lastAtmosphere ? .atmosphere(fallback: daylight) : .daylight(daylight)
        guard cache.encodeFrame(cb, depth: depth, out: out, invViewProj: invViewProj, cam: cam, origin: origin, accel: tlas,
                                accels: tlasAccels, sunDir: sun, sunUp: sun.y > -0.05 ? 1 : 0, cloudHeight: cloudHeight, light: light) else { return }
        litGiOut = (out, cache.envThisFrame, w, h)
        giRuns += 1
        if giRuns % 1200 == 0 {
            let c = cache.counters.contents().bindMemory(to: UInt32.self, capacity: 20)
            log("gi: \(giRuns) frames, \(gi.table.length / 16) tiles; last read: \(c[19]) visible cells seen, \(c[1]) updated, \(c[11]) rays so far")
        }
    }
}

/// Offline timing with the GI cache (METALMC_EXP=lit,gi): 0 skips its frame in RtShadows.trace (the relight then has no
/// cache light: its sky term as without it), 1 (default) runs it.
@_cdecl("mmc_debug_gi_frame")
public func mmc_debug_gi_frame(_ on: Int32) {
    RtShadows.shared.giOn = on != 0
}

/// Shadows for the level just drawn into `color` (see RtShadows). `p`: projection as drawn (jittered), then view
/// rotation, 32 floats. `sunAngle`: vanilla's sun angle in radians; `strength`: 0 turns them off (in lit mode the
/// visibility is traced anyway, toward the moon at night, and the strength isn't used).
@_cdecl("mmc_shadows_apply")
public func mmc_shadows_apply(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>, _ cam: UnsafePointer<Double>,
                              _ sunAngle: Float, _ strength: Float, _ cloudHeight: Float, _ taa: Int32) -> Int32 {
    guard lodRtShadows, strength > 0 || litEnabled else { return 0 }
    let taaEnabledThisFrame = taa != 0
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    return RtShadows.shared.apply(color: color, depth: depth, p: p, cam: SIMD3(cam[0], cam[1], cam[2]), sunAngle: sunAngle,
                                  strength: strength, cloudHeight: cloudHeight, deferToTaa: taaEnabledThisFrame) ? 1 : 0
}
