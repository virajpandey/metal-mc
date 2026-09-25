import Foundation
import Metal
import MetalMCCore
import simd

// Far-terrain LOD drawn inside Minecraft's main world pass, after solid terrain (Java hook:
// metalmc.lod.mixin.LevelRendererLodMixin). It shares the game's projection (with the far plane pushed
// out) and reverse-Z depth, so it depth-tests correctly against vanilla. It reproduces vanilla's fog
// exactly, so it fades into the same sky and fog colors.

let lodShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct LodUniforms {
    float4x4 proj;
    float4x4 view;
    float4 fogColor;
    float envStart, envEnd, rdStart, rdEnd;
    float discardRadius;   // horizontal distance inside which vanilla chunks are drawn instead
    float sky;             // daylight factor 0..1
    float2 pad;
};
// Per draw: section origin relative to the camera (xyz) and voxel size in blocks (w).
struct Xform { float4 offsetScale; };
struct VOut {
    float4 pos [[position]];
    float3 color;
    float3 rel;
};

// Unit-cube corners per face, counter-clockwise seen from outside. Face order: +X -X +Y -Y +Z -Z.
constant float3 kCorners[6][4] = {
    { float3(1,0,0), float3(1,1,0), float3(1,1,1), float3(1,0,1) },
    { float3(0,0,1), float3(0,1,1), float3(0,1,0), float3(0,0,0) },
    { float3(0,1,0), float3(0,1,1), float3(1,1,1), float3(1,1,0) },
    { float3(0,0,0), float3(1,0,0), float3(1,0,1), float3(0,0,1) },
    { float3(1,0,1), float3(1,1,1), float3(0,1,1), float3(0,0,1) },
    { float3(0,0,0), float3(0,1,0), float3(1,1,0), float3(1,0,0) },
};
// Directional shading close to vanilla's face shading (top 1.0, bottom 0.5, X 0.6, Z 0.8).
constant float kShade[6] = { 0.6, 0.6, 1.0, 0.5, 0.8, 0.8 };

static float3 extentScale(uint face, float w, float h) {
    if (face < 2) return float3(1, w, h);
    if (face < 4) return float3(w, 1, h);
    return float3(w, h, 1);
}

// Quads are 8 bytes: word0 = x | z << 8 | y << 16 | face << 24 (voxel coordinates within the node),
// word1 = material | (w - 1) << 8 | (h - 1) << 16 (greedy extents along the face's u and v axes).
vertex VOut lod_vs(uint vid [[vertex_id]], uint draw [[base_instance]],
                   const device uint2* quads [[buffer(18)]],
                   constant LodUniforms& u [[buffer(19)]],
                   const device Xform* xforms [[buffer(20)]],
                   constant float4* colors [[buffer(21)]]) {
    uint2 q = quads[vid >> 2];
    uint corner = vid & 3;
    uint face = (q.x >> 24) & 7;
    float3 local = float3(q.x & 255, (q.x >> 16) & 255, (q.x >> 8) & 255);
    float w = float(((q.y >> 8) & 255) + 1), h = float(((q.y >> 16) & 255) + 1);
    float3 ext = extentScale(face, w, h);
    float4 xs = xforms[draw].offsetScale;
    float3 rel = xs.xyz + (local + kCorners[face][corner] * ext) * xs.w;
    // Drop whole quads inside the range vanilla draws: decide by the quad's center so all four corners
    // agree, and collapse the quad to a point (no fragment discard, which would disable hidden-surface removal).
    float3 center = xs.xyz + (local + 0.5 * (kCorners[face][0] + kCorners[face][2]) * ext) * xs.w;
    VOut o;
    if (length(center.xz) < u.discardRadius) {
        o.pos = float4(0.0, 0.0, 0.0, 1.0);
        o.color = float3(0.0);
        o.rel = float3(0.0);
        return o;
    }
    float4 clip = u.proj * (u.view * float4(rel, 1.0));
    clip.y = -clip.y;   // same vertical flip as every translated Minecraft shader (flip_vert_y)
    o.pos = clip;
    float3 base = colors[q.y & 255].rgb;
    o.color = base * kShade[face] * mix(0.2, 1.0, u.sky);
    o.rel = rel;
    return o;
}

static float linearFog(float d, float s, float e) {
    if (d <= s) return 0.0;
    if (d >= e) return 1.0;
    return (d - s) / (e - s);
}

fragment float4 lod_fs(VOut in [[stage_in]], constant LodUniforms& u [[buffer(19)]]) {
    float horiz = length(in.rel.xz);
    float spherical = length(in.rel);
    float cylindrical = max(horiz, abs(in.rel.y));
    float fog = max(linearFog(spherical, u.envStart, u.envEnd), linearFog(cylindrical, u.rdStart, u.rdEnd));
    return float4(mix(in.color, u.fogColor.rgb, fog * u.fogColor.a), 1.0);
}

// Occlusion test. After the LOD is drawn, every candidate tile's bounding box is rasterized in the same
// pass with depth testing and no writes. At that point the depth buffer holds vanilla's solid terrain and
// the LOD. Fragments that survive mark the tile visible; the CPU reads the marks once the frame completes.
struct BoxOut {
    float4 pos [[position]];
    uint slot [[flat]];
};
// 12 triangles over the 8 box corners (bit 0 = x, bit 1 = y, bit 2 = z), counter-clockwise seen from
// outside like the LOD quads, so back-face culling keeps the faces toward the camera.
constant ushort kBoxIndex[36] = {
    0, 4, 6, 0, 6, 2,   1, 3, 7, 1, 7, 5,   0, 1, 5, 0, 5, 4,
    2, 6, 7, 2, 7, 3,   0, 2, 3, 0, 3, 1,   4, 5, 7, 4, 7, 6,
};
vertex BoxOut lod_box_vs(uint vid [[vertex_id]], uint iid [[instance_id]],
                         constant LodUniforms& u [[buffer(19)]],
                         const device float4* boxes [[buffer(22)]]) {
    float3 lo = boxes[2 * iid].xyz, hi = boxes[2 * iid + 1].xyz;
    uint c = kBoxIndex[vid];
    float3 p = float3((c & 1) ? hi.x : lo.x, (c & 2) ? hi.y : lo.y, (c & 4) ? hi.z : lo.z);
    float4 clip = u.proj * (u.view * float4(p, 1.0));
    clip.y = -clip.y;
    BoxOut o;
    o.pos = clip;
    o.slot = iid;
    return o;
}
[[early_fragment_tests]]
fragment void lod_box_fs(BoxOut in [[stage_in]], device uint* visible [[buffer(23)]]) {
    visible[in.slot] = 1;
}
"""

/// Uniform block layout shared with the shader (must match LodUniforms).
struct LodUniforms {
    var proj: simd_float4x4
    var view: simd_float4x4
    var fogColor: SIMD4<Float>
    var envStart: Float, envEnd: Float, rdStart: Float, rdEnd: Float
    var discardRadius: Float
    var sky: Float
    var pad: SIMD2<Float> = .zero
}

/// One frame's occlusion test: the boxes drawn and the GPU-written visibility marks, one per slot.
final class LodVisSet {
    static let capacity = 8192
    let marks: MTLBuffer        // UInt32 per slot, written by lod_box_fs
    let boxes: MTLBuffer        // two float4 per slot (lo, hi), camera-relative
    var slots: [(LodMeshNode, Int)] = []
    var frame: UInt64 = 0
    var pending = false         // encoded, not yet harvested (render thread)
    var done = false            // command buffer completed (set on the completion thread, under visLock)

    init?() {
        guard let m = ctx.device.makeBuffer(length: Self.capacity * 4, options: [.storageModeShared]),
              let b = ctx.device.makeBuffer(length: Self.capacity * 32, options: [.storageModeShared]) else { return nil }
        m.label = "MetalMC LOD visibility"
        b.label = "MetalMC LOD boxes"
        marks = m
        boxes = b
    }
}

/// METALMC_EXP=noocc turns off LOD occlusion culling (for A/B comparisons).
let lodOcclusion = !experiments.contains("noocc")
/// METALMC_EXP=occnocull tests boxes but never skips tiles (measures the test's own cost).
let lodOccNoCull = experiments.contains("occnocull")

final class LodRenderer: @unchecked Sendable {
    static let shared = LodRenderer()

    let lock = NSLock()
    var world: LodWorld?
    var colorBuffer: MTLBuffer?
    var indexBuffer: MTLBuffer?        // shared pattern 4q + {0,1,2,0,2,3}
    var indexQuads = 0
    var pipelines: [String: MTLRenderPipelineState] = [:]
    var library: MTLLibrary?
    // Occlusion culling (render thread, except LodVisSet.done).
    let visLock = NSLock()
    var visSets: [LodVisSet] = []
    var frame: UInt64 = 0
    var latestResult: UInt64 = 0       // newest frame whose occlusion results have been read
    var lastCamera = SIMD3<Double>(repeating: .nan)
    var jumpFrame: UInt64 = 0          // last frame the camera jumped; results tested before it are stale

    func pipeline(colorFormats: [MTLPixelFormat], depth: MTLPixelFormat, box: Bool = false) -> MTLRenderPipelineState? {
        let key = colorFormats.map { String($0.rawValue) }.joined(separator: ",") + "/\(depth.rawValue)" + (box ? "/box" : "")
        if let p = pipelines[key] { return p }
        do {
            if library == nil { library = try ctx.device.makeLibrary(source: lodShaderSource, options: nil) }
            let d = MTLRenderPipelineDescriptor()
            d.label = box ? "MetalMC LOD occlusion boxes" : "MetalMC LOD"
            d.vertexFunction = library!.makeFunction(name: box ? "lod_box_vs" : "lod_vs")
            d.fragmentFunction = library!.makeFunction(name: box ? "lod_box_fs" : "lod_fs")
            for (i, f) in colorFormats.enumerated() {
                d.colorAttachments[i].pixelFormat = f
                // Only the main color target gets LOD color; extra targets (OIT) are left untouched.
                // The occlusion boxes write no color at all.
                if i > 0 || box { d.colorAttachments[i].writeMask = [] }
            }
            d.depthAttachmentPixelFormat = depth
            let p = try ctx.device.makeRenderPipelineState(descriptor: d)
            pipelines[key] = p
            return p
        } catch {
            log("LOD pipeline failed: \(error)")
            return nil
        }
    }

    /// Reads the marks of every completed occlusion test (oldest first) into the nodes' tile state, then
    /// returns a free set for this frame (nil if all are still in flight).
    func harvestAndAcquire() -> LodVisSet? {
        visLock.lock()
        let ready = visSets.filter { $0.pending && $0.done }.sorted { $0.frame < $1.frame }
        visLock.unlock()
        for s in ready {
            let marks = s.marks.contents().bindMemory(to: UInt32.self, capacity: LodVisSet.capacity)
            for (i, (node, t)) in s.slots.enumerated() {
                node.tileTested[t] = s.frame
                if marks[i] != 0 { node.tileVisible[t] = s.frame }
            }
            latestResult = max(latestResult, s.frame)
            s.slots.removeAll(keepingCapacity: true)
            visLock.lock(); s.pending = false; s.done = false; visLock.unlock()
        }
        if let free = visSets.first(where: { !$0.pending }) { return free }
        if visSets.count < 4, let s = LodVisSet() { visSets.append(s); return s }
        return nil
    }

    func ensureIndexBuffer(quads: Int) {
        if indexQuads >= quads, indexBuffer != nil { return }
        var n = max(1024, indexQuads)
        while n < quads { n *= 2 }
        var idx = [UInt32](repeating: 0, count: n * 6)
        for q in 0..<n {
            let b = UInt32(4 * q)
            idx[6 * q] = b; idx[6 * q + 1] = b + 1; idx[6 * q + 2] = b + 2
            idx[6 * q + 3] = b; idx[6 * q + 4] = b + 2; idx[6 * q + 5] = b + 3
        }
        indexBuffer = ctx.device.makeBuffer(bytes: idx, length: idx.count * 4, options: [.storageModeShared])
        indexQuads = n
    }

    /// Material colors from the texture averages in LodColors.swift, indexed by material id.
    func ensureColors() {
        if colorBuffer != nil { return }
        let c = lodColorTable()
        colorBuffer = ctx.device.makeBuffer(bytes: c, length: c.count * 16, options: [.storageModeShared])
    }

    /// Quadtree selection. A node splits when the camera is closer than `splitFactor` x the child size:
    /// existing children are visited, and for each missing child the parent draws just the 2 x 2 tiles
    /// covering that quarter (missing children are either empty or past the finest level's range).
    /// Returns nodes with a 16-bit mask of the tiles to draw.
    static func select(_ meshes: [LodNodeKey: LodMeshNode], maxLevel: Int, camX: Double, camZ: Double, splitFactor: Double) -> [(LodMeshNode, UInt16)] {
        var out: [(LodMeshNode, UInt16)] = []
        // Tile masks for each quarter (tiles t = tz * 4 + tx; quarter (qx, qz) covers tx, tz in 2q ..< 2q + 2).
        func quarterMask(_ qx: Int, _ qz: Int) -> UInt16 {
            var m: UInt16 = 0
            for tz in (2 * qz)..<(2 * qz + 2) { for tx in (2 * qx)..<(2 * qx + 2) { m |= 1 << UInt16(tz * 4 + tx) } }
            return m
        }
        func visit(_ level: Int, _ nx: Int, _ nz: Int) {
            let node = meshes[LodNodeKey(level: level, x: nx, z: nz)]
            let size = Double(lodNodeVoxels << level)
            let x0 = Double(nx) * size, z0 = Double(nz) * size
            let dx = max(x0 - camX, 0, camX - (x0 + size)), dz = max(z0 - camZ, 0, camZ - (z0 + size))
            let dist = (dx * dx + dz * dz).squareRoot()
            if level > 1 && dist < splitFactor * size / 2 {
                var parentMask: UInt16 = 0
                for qz in 0...1 {
                    for qx in 0...1 {
                        let cx = 2 * nx + qx, cz = 2 * nz + qz
                        if meshes[LodNodeKey(level: level - 1, x: cx, z: cz)] != nil || hasDescendant(level - 1, cx, cz) {
                            visit(level - 1, cx, cz)
                        } else {
                            parentMask |= quarterMask(qx, qz)
                        }
                    }
                }
                if let node, parentMask != 0 { out.append((node, parentMask)) }
                return
            }
            if let node { out.append((node, 0xFFFF)) }
        }
        // A missing node may still have finer descendants (never happens today, but keep the recursion honest).
        func hasDescendant(_ level: Int, _ nx: Int, _ nz: Int) -> Bool { false }
        for k in meshes.keys where k.level == maxLevel { visit(maxLevel, k.x, k.z) }
        return out
    }
}

@_cdecl("mmc_lod_open")
public func mmc_lod_open(_ worldDir: UnsafePointer<CChar>, _ far: Int32, _ centerX: Int32, _ centerZ: Int32) -> Int32 {
    let r = LodRenderer.shared
    let dir = String(cString: worldDir)
    guard let regionDir = Anvil.regionDirectory(URL(fileURLWithPath: dir)) else {
        log("LOD: no region directory under \(dir)")
        return 0
    }
    var maxLevel = 1
    while (lodNodeVoxels << maxLevel) < Int(far) && maxLevel < 8 { maxLevel += 1 }
    let w = LodWorld(regionDir: regionDir, maxLevel: maxLevel, fineRadius: 1536, centerX: Int(centerX), centerZ: Int(centerZ))
    r.lock.lock()
    r.world?.stop()
    r.world = w
    r.lock.unlock()
    log("LOD: streaming \(regionDir.path) far=\(far) levels 1...\(maxLevel) around (\(centerX), \(centerZ))")
    w.start()
    return 1
}

/// Stops streaming and drops the LOD (the player left the world).
@_cdecl("mmc_lod_close")
public func mmc_lod_close() {
    let r = LodRenderer.shared
    r.lock.lock()
    r.world?.stop()
    r.world = nil
    r.lock.unlock()
}

/// Tells the LOD where the player is, so the finest level follows them.
@_cdecl("mmc_lod_center")
public func mmc_lod_center(_ x: Int32, _ z: Int32, _ vanillaRadius: Int32) {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    w?.setCenter(x: Int(x), z: Int(z), vanillaRadius: Int(vanillaRadius))
}

/// out: state (0 none, 1 building, 2 has nodes), nodes, quads.
@_cdecl("mmc_lod_status")
public func mmc_lod_status(_ out: UnsafeMutablePointer<Int64>) {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { out[0] = 0; out[1] = 0; out[2] = 0; return }
    let snap = w.snapshot()
    out[0] = snap.meshes.isEmpty ? 1 : 2
    out[1] = Int64(snap.meshes.count)
    out[2] = Int64(snap.meshes.values.reduce(0) { $0 + $1.quadCount })
}

/// Draws the LOD into the open render pass. p: proj[16], view[16] (column-major), fog color[4],
/// envStart, envEnd, rdStart, rdEnd, discardRadius, sky. cam: camera position (world, doubles).
/// Returns the number of draws. Leaves ctx.pipe nil so Java re-applies Minecraft's pipeline state.
@_cdecl("mmc_lod_draw")
public func mmc_lod_draw(_ p: UnsafePointer<Float>, _ cam: UnsafePointer<Double>) -> Int32 {
    let t0 = DispatchTime.now().uptimeNanoseconds
    defer { ctx.statLodNanos += DispatchTime.now().uptimeNanoseconds - t0 }
    let r = LodRenderer.shared
    guard let enc = ctx.pass, !ctx.scissorEmpty else { return 0 }
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return 0 }
    let snap = w.snapshot()
    guard !snap.meshes.isEmpty else { return 0 }
    guard let pipe = r.pipeline(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat) else { return 0 }
    r.ensureColors()
    let cx = cam[0], cy = cam[1], cz = cam[2]
    let chosen = LodRenderer.select(snap.meshes, maxLevel: w.maxLevel, camX: cx, camZ: cz, splitFactor: 2.0)
    guard !chosen.isEmpty else { return 0 }
    r.ensureIndexBuffer(quads: chosen.map { $0.0.quadCount }.max() ?? 1)
    guard let ib = r.indexBuffer else { return 0 }

    func mat(_ o: Int) -> simd_float4x4 {
        simd_float4x4(SIMD4(p[o], p[o + 1], p[o + 2], p[o + 3]), SIMD4(p[o + 4], p[o + 5], p[o + 6], p[o + 7]),
                      SIMD4(p[o + 8], p[o + 9], p[o + 10], p[o + 11]), SIMD4(p[o + 12], p[o + 13], p[o + 14], p[o + 15]))
    }
    var u = LodUniforms(proj: mat(0), view: mat(16), fogColor: SIMD4(p[32], p[33], p[34], p[35]),
                        envStart: p[36], envEnd: p[37], rdStart: p[38], rdEnd: p[39], discardRadius: p[40], sky: p[41])

    // Frustum side planes from proj * view (camera-relative). Points behind the camera fail them too.
    let m = u.proj * u.view
    let r0 = SIMD4(m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x)
    let r1 = SIMD4(m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y)
    let r3 = SIMD4(m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w)
    let planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1]
    func visible(_ lo: SIMD3<Float>, _ hi: SIMD3<Float>) -> Bool {
        for pl in planes {
            let v = SIMD3(pl.x >= 0 ? hi.x : lo.x, pl.y >= 0 ? hi.y : lo.y, pl.z >= 0 ? hi.z : lo.z)
            if pl.x * v.x + pl.y * v.y + pl.z * v.z + pl.w < 0 { return false }
        }
        return true
    }

    // Occlusion: tiles whose box was hidden in the newest completed test are skipped. Every candidate
    // tile's box is tested again this frame, so a tile that comes into view reappears within 2-3 frames.
    r.frame += 1
    let vis = lodOcclusion ? r.harvestAndAcquire() : nil
    let camera = SIMD3(cx, cy, cz)
    let jumped = simd_distance(camera, r.lastCamera)
    if !(jumped < 16) { r.jumpFrame = r.frame }   // teleports, respawns (and NaN on the first frame)
    r.lastCamera = camera
    let latest = r.latestResult >= r.jumpFrame ? r.latestResult : 0
    let boxOut = vis.map { $0.boxes.contents().bindMemory(to: SIMD4<Float>.self, capacity: 2 * LodVisSet.capacity) }
    let markOut = vis.map { $0.marks.contents().bindMemory(to: UInt32.self, capacity: LodVisSet.capacity) }

    struct Draw { var node: LodMeshNode; var slot: Int; var first: Int; var count: Int }
    var draws: [Draw] = []
    var xforms = [SIMD4<Float>]()
    xforms.reserveCapacity(chosen.count)
    let discard = u.discardRadius
    for (n, tileMask) in chosen {
        let voxel = Float(1 << n.level)
        let nodeLo = SIMD3(Float(Double(n.x0) - cx), Float(Double(lodWorldMinY) - cy), Float(Double(n.z0) - cz))
        let nodeHi = nodeLo + SIMD3(Float(n.size), Float(lodWorldHeight), Float(n.size))
        if !visible(nodeLo, nodeHi) { continue }
        var slot = -1
        let tileSize = Float(lodTileVoxels) * voxel
        for t in 0..<(lodTilesPerSide * lodTilesPerSide) where tileMask & (1 << UInt16(t)) != 0 {
            let yMin = n.tileY[2 * t], yMax = n.tileY[2 * t + 1]
            if yMin > yMax { continue }
            let tx = t % lodTilesPerSide, tz = t / lodTilesPerSide
            let lo = SIMD3(nodeLo.x + Float(tx) * tileSize, nodeLo.y + Float(yMin) * voxel, nodeLo.z + Float(tz) * tileSize)
            let hi = SIMD3(lo.x + tileSize, nodeLo.y + Float(yMax) * voxel, lo.z + tileSize)
            // Tiles entirely inside the range vanilla draws: every point of the tile's footprint is closer
            // than the discard radius (its farthest corner is).
            let fx = max(abs(lo.x), abs(hi.x)), fz = max(abs(lo.z), abs(hi.z))
            if fx * fx + fz * fz < discard * discard { continue }
            if !visible(lo, hi) { continue }
            // Only box faces toward the camera are rasterized, so a camera inside a box would see nothing
            // of it: such tiles are left untested, which keeps them drawn.
            let inside = lo.x - voxel < 0 && hi.x + voxel > 0 && lo.y - voxel < 0 && hi.y + voxel > 0
                && lo.z - voxel < 0 && hi.z + voxel > 0
            if !inside, let vis, let boxOut, let markOut, vis.slots.count < LodVisSet.capacity {
                // Box grown by one voxel so the tile's own surfaces never hide it.
                let i = vis.slots.count
                boxOut[2 * i] = SIMD4(lo - voxel, 0)
                boxOut[2 * i + 1] = SIMD4(hi + voxel, 0)
                markOut[i] = 0
                vis.slots.append((n, t))
            }
            if latest > 0 && !lodOccNoCull && n.tileTested[t] == latest && n.tileVisible[t] != latest { continue }
            // Face buckets that can face the camera (camera past the tile's nearest plane on that axis).
            let faceVisible = [0 > lo.x, 0 < hi.x, 0 > lo.y, 0 < hi.y, 0 > lo.z, 0 < hi.z]
            if slot < 0 {
                slot = xforms.count
                xforms.append(SIMD4(nodeLo.x, nodeLo.y, nodeLo.z, voxel))
            }
            let base = t * 6
            var f = 0
            while f < 6 {
                if !faceVisible[f] || n.start[base + f + 1] == n.start[base + f] { f += 1; continue }
                var e = f + 1
                while e < 6 && (faceVisible[e] || n.start[base + e + 1] == n.start[base + e]) { e += 1 }
                draws.append(Draw(node: n, slot: slot, first: n.start[base + f], count: n.start[base + e] - n.start[base + f]))
                f = e
            }
        }
    }
    let testBoxes = vis.map { !$0.slots.isEmpty } ?? false
    guard !draws.isEmpty || testBoxes else { return 0 }

    enc.setRenderPipelineState(pipe)
    enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: true))
    enc.setCullMode(.back)
    enc.setTriangleFillMode(.fill)
    enc.setDepthBias(0, slopeScale: 0, clamp: 0)
    enc.setVertexBytes(&u, length: MemoryLayout<LodUniforms>.stride, index: 19)
    enc.setFragmentBytes(&u, length: MemoryLayout<LodUniforms>.stride, index: 19)
    if xforms.count * 16 <= 4096 {
        enc.setVertexBytes(xforms, length: xforms.count * 16, index: 20)
    } else if let xb = ctx.device.makeBuffer(bytes: xforms, length: xforms.count * 16, options: [.storageModeShared]) {
        enc.setVertexBuffer(xb, offset: 0, index: 20)
    }
    enc.setVertexBuffer(r.colorBuffer!, offset: 0, index: 21)
    var bound: ObjectIdentifier?
    ctx.statLodDraws += draws.count
    for d in draws {
        ctx.statLodQuads += d.count
        let id = ObjectIdentifier(d.node)
        if bound != id {
            enc.setVertexBuffer(d.node.buffer, offset: 0, index: 18)
            bound = id
        }
        enc.drawIndexedPrimitives(type: .triangle, indexCount: d.count * 6, indexType: .uint32, indexBuffer: ib,
                                  indexBufferOffset: 0, instanceCount: 1, baseVertex: 4 * d.first, baseInstance: d.slot)
    }
    if let vis, testBoxes, let boxPipe = r.pipeline(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat, box: true),
       let cb = ctx.cb {
        enc.setRenderPipelineState(boxPipe)
        enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: false))
        enc.setCullMode(.back)   // front faces only: half the fragments of drawing every face
        enc.setVertexBuffer(vis.boxes, offset: 0, index: 22)
        enc.setFragmentBuffer(vis.marks, offset: 0, index: 23)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 36, instanceCount: vis.slots.count)
        vis.frame = r.frame
        vis.pending = true
        cb.addCompletedHandler { _ in
            r.visLock.lock(); vis.done = true; r.visLock.unlock()
        }
    } else {
        vis?.slots.removeAll(keepingCapacity: true)
    }
    // Minecraft's pipeline, depth, cull and bias state must be re-applied by the next setPipeline.
    ctx.pipe = nil
    ctx.boundPipeState = nil
    return Int32(draws.count)
}
