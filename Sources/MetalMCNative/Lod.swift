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

final class LodRenderer: @unchecked Sendable {
    static let shared = LodRenderer()

    let lock = NSLock()
    var world: LodWorld?
    var colorBuffer: MTLBuffer?
    var indexBuffer: MTLBuffer?        // shared pattern 4q + {0,1,2,0,2,3}
    var indexQuads = 0
    var pipelines: [String: MTLRenderPipelineState] = [:]
    var library: MTLLibrary?

    func pipeline(colorFormats: [MTLPixelFormat], depth: MTLPixelFormat) -> MTLRenderPipelineState? {
        let key = colorFormats.map { String($0.rawValue) }.joined(separator: ",") + "/\(depth.rawValue)"
        if let p = pipelines[key] { return p }
        do {
            if library == nil { library = try ctx.device.makeLibrary(source: lodShaderSource, options: nil) }
            let d = MTLRenderPipelineDescriptor()
            d.label = "MetalMC LOD"
            d.vertexFunction = library!.makeFunction(name: "lod_vs")
            d.fragmentFunction = library!.makeFunction(name: "lod_fs")
            for (i, f) in colorFormats.enumerated() {
                d.colorAttachments[i].pixelFormat = f
                // Only the main color target gets LOD color; extra targets (OIT) are left untouched.
                if i > 0 { d.colorAttachments[i].writeMask = [] }
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

    /// Quadtree selection: a node is split into its four children when the camera is closer than
    /// `splitFactor` x the child size and all four children exist; otherwise the node itself is drawn.
    static func select(_ meshes: [LodNodeKey: LodMeshNode], maxLevel: Int, camX: Double, camZ: Double, splitFactor: Double) -> [LodMeshNode] {
        var out: [LodMeshNode] = []
        func visit(_ level: Int, _ nx: Int, _ nz: Int) {
            guard let node = meshes[LodNodeKey(level: level, x: nx, z: nz)] else { return }
            let size = Double(lodNodeVoxels << level)
            let x0 = Double(nx) * size, z0 = Double(nz) * size
            let dx = max(x0 - camX, 0, camX - (x0 + size)), dz = max(z0 - camZ, 0, camZ - (z0 + size))
            let dist = (dx * dx + dz * dz).squareRoot()
            if level > 1, dist < splitFactor * size / 2,
               meshes[LodNodeKey(level: level - 1, x: 2 * nx, z: 2 * nz)] != nil,
               meshes[LodNodeKey(level: level - 1, x: 2 * nx + 1, z: 2 * nz)] != nil,
               meshes[LodNodeKey(level: level - 1, x: 2 * nx, z: 2 * nz + 1)] != nil,
               meshes[LodNodeKey(level: level - 1, x: 2 * nx + 1, z: 2 * nz + 1)] != nil {
                visit(level - 1, 2 * nx, 2 * nz); visit(level - 1, 2 * nx + 1, 2 * nz)
                visit(level - 1, 2 * nx, 2 * nz + 1); visit(level - 1, 2 * nx + 1, 2 * nz + 1)
                return
            }
            out.append(node)
        }
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
    r.ensureIndexBuffer(quads: chosen.map(\.quadCount).max() ?? 1)
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

    struct Draw { var node: LodMeshNode; var slot: Int; var first: Int; var count: Int }
    var draws: [Draw] = []
    var xforms = [SIMD4<Float>]()
    xforms.reserveCapacity(chosen.count)
    for n in chosen {
        let lo = SIMD3(Float(Double(n.x0) - cx), Float(Double(lodWorldMinY) - cy), Float(Double(n.z0) - cz))
        let hi = lo + SIMD3(Float(n.size), Float(lodWorldHeight), Float(n.size))
        if !visible(lo, hi) { continue }
        // Face buckets that can face the camera (conservative: the camera is past the node's nearest plane).
        let s = Float(n.size), top = Float(lodWorldHeight)
        let faceVisible = [-lo.x > 0, -lo.x < s, -lo.y > 0, -lo.y < top, -lo.z > 0, -lo.z < s]
        let slot = xforms.count
        xforms.append(SIMD4(lo.x, lo.y, lo.z, Float(1 << n.level)))
        var f = 0
        while f < 6 {
            if !faceVisible[f] || n.faceStart[f + 1] == n.faceStart[f] { f += 1; continue }
            var e = f + 1
            while e < 6 && (faceVisible[e] || n.faceStart[e + 1] == n.faceStart[e]) { e += 1 }
            draws.append(Draw(node: n, slot: slot, first: n.faceStart[f], count: n.faceStart[e] - n.faceStart[f]))
            f = e
        }
    }
    guard !draws.isEmpty else { return 0 }

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
    for d in draws {
        let id = ObjectIdentifier(d.node)
        if bound != id {
            enc.setVertexBuffer(d.node.buffer, offset: 0, index: 18)
            bound = id
        }
        enc.drawIndexedPrimitives(type: .triangle, indexCount: d.count * 6, indexType: .uint32, indexBuffer: ib,
                                  indexBufferOffset: 0, instanceCount: 1, baseVertex: 4 * d.first, baseInstance: d.slot)
    }
    // Minecraft's pipeline, depth, cull and bias state must be re-applied by the next setPipeline.
    ctx.pipe = nil
    ctx.boundPipeState = nil
    return Int32(draws.count)
}
