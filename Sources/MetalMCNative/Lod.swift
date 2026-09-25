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

/// One drawable LOD node: its range in the shared quad buffer and its placement.
struct LodDrawNode {
    var level: Int
    var x0: Int, z0: Int        // world block coordinates of the node corner (y starts at the world bottom)
    var firstQuad: Int
    var quadCount: Int
    var faceStart: [Int]        // 7 prefix offsets (relative to firstQuad) of the face buckets +X -X +Y -Y +Z -Z
    var size: Int { lodNodeVoxels << level }
}

final class LodRenderer: @unchecked Sendable {
    static let shared = LodRenderer()

    let lock = NSLock()
    var state = 0                      // 0 idle, 1 building, 2 ready, 3 failed
    var message = ""
    var nodes: [LodDrawNode] = []
    var byLevel: [Int: [Int64: Int]] = [:]   // level -> node key -> index into nodes
    var maxLevel = 0
    var quadBuffer: MTLBuffer?
    var colorBuffer: MTLBuffer?
    var indexBuffer: MTLBuffer?        // shared pattern 4q + {0,1,2,0,2,3}
    var indexQuads = 0
    var pipelines: [String: MTLRenderPipelineState] = [:]
    var library: MTLLibrary?
    var totalQuads = 0

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

    /// Material colors (linear-ish sRGB, alpha unused) indexed by material id.
    func ensureColors() {
        if colorBuffer != nil { return }
        var c = [SIMD4<Float>](repeating: SIMD4(1, 0, 1, 1), count: 256)
        for m in Mat.allCases { c[Int(m.rawValue)] = m.color }
        colorBuffer = ctx.device.makeBuffer(bytes: c, length: c.count * 16, options: [.storageModeShared])
    }

    func install(built: [LodNode]) {
        var quads: [UInt32] = []
        var list: [LodDrawNode] = []
        var index: [Int: [Int64: Int]] = [:]
        var top = 0
        for n in built where !n.quads.isEmpty {
            let size = n.size
            index[n.level, default: [:]][LodBuild.key(Int((Double(n.x0) / Double(size)).rounded(.down)), Int((Double(n.z0) / Double(size)).rounded(.down)))] = list.count
            var starts = [0]
            for c in n.faceCounts { starts.append(starts.last! + c) }
            list.append(LodDrawNode(level: n.level, x0: n.x0, z0: n.z0, firstQuad: quads.count / 2, quadCount: n.quads.count / 2, faceStart: starts))
            quads.append(contentsOf: n.quads)
            top = max(top, n.level)
        }
        let buf = ctx.device.makeBuffer(bytes: quads, length: max(16, quads.count * 4), options: [.storageModeShared])
        lock.lock()
        quadBuffer = buf
        nodes = list
        byLevel = index
        maxLevel = top
        totalQuads = quads.count / 2
        state = 2
        lock.unlock()
    }

    /// Quadtree selection: a node is split into its four children when the camera is closer than
    /// `splitFactor` x the child size and all four children exist; otherwise the node itself is drawn.
    func select(camX: Double, camZ: Double, splitFactor: Double) -> [Int] {
        var out: [Int] = []
        func visit(_ level: Int, _ nx: Int, _ nz: Int) {
            guard let idx = byLevel[level]?[LodBuild.key(nx, nz)] else { return }
            let size = Double(lodNodeVoxels << level)
            let x0 = Double(nx) * size, z0 = Double(nz) * size
            let dx = max(x0 - camX, 0, camX - (x0 + size)), dz = max(z0 - camZ, 0, camZ - (z0 + size))
            let dist = (dx * dx + dz * dz).squareRoot()
            if level > 1, dist < splitFactor * size / 2, let children = byLevel[level - 1],
               children[LodBuild.key(2 * nx, 2 * nz)] != nil, children[LodBuild.key(2 * nx + 1, 2 * nz)] != nil,
               children[LodBuild.key(2 * nx, 2 * nz + 1)] != nil, children[LodBuild.key(2 * nx + 1, 2 * nz + 1)] != nil {
                visit(level - 1, 2 * nx, 2 * nz); visit(level - 1, 2 * nx + 1, 2 * nz)
                visit(level - 1, 2 * nx, 2 * nz + 1); visit(level - 1, 2 * nx + 1, 2 * nz + 1)
                return
            }
            out.append(idx)
        }
        for k in (byLevel[maxLevel] ?? [:]).keys {
            let (nx, nz) = LodBuild.unkey(k)
            visit(maxLevel, nx, nz)
        }
        return out
    }
}

@_cdecl("mmc_lod_open")
public func mmc_lod_open(_ worldDir: UnsafePointer<CChar>, _ far: Int32, _ centerX: Int32, _ centerZ: Int32) -> Int32 {
    let r = LodRenderer.shared
    r.lock.lock()
    if r.state == 1 { r.lock.unlock(); return 0 }
    r.state = 1
    r.lock.unlock()
    let dir = String(cString: worldDir)
    let farBlocks = Int(far), cx = Int(centerX), cz = Int(centerZ)
    DispatchQueue.global(qos: .utility).async {
        let t0 = Date()
        // Levels up to the one whose nodes are at least `far` across; level 1 is meshed out to 1.5 km.
        var maxLevel = 1
        while (lodNodeVoxels << maxLevel) < farBlocks && maxLevel < 8 { maxLevel += 1 }
        log("LOD: building \(dir) far=\(farBlocks) levels 1...\(maxLevel) around (\(cx), \(cz))")
        let built = LodBuild.build(worldDir: dir, maxLevel: maxLevel, centerX: cx, centerZ: cz, fineRadius: 1536)
        r.install(built: built)
        log("LOD: ready in \(String(format: "%.1f", Date().timeIntervalSince(t0))) s: \(r.nodes.count) nodes, \(r.totalQuads) quads, \(r.totalQuads * 8 / 1_000_000) MB")
    }
    return 1
}

/// out: state, sections, quads.
@_cdecl("mmc_lod_status")
public func mmc_lod_status(_ out: UnsafeMutablePointer<Int64>) {
    let r = LodRenderer.shared
    r.lock.lock(); defer { r.lock.unlock() }
    out[0] = Int64(r.state)
    out[1] = Int64(r.nodes.count)
    out[2] = Int64(r.totalQuads)
}

/// Draws the LOD into the open render pass. p: proj[16], view[16] (column-major), fog color[4],
/// envStart, envEnd, rdStart, rdEnd, discardRadius, sky. cam: camera position (world, doubles).
/// Returns the number of draws. Leaves ctx.pipe nil so Java re-applies Minecraft's pipeline state.
@_cdecl("mmc_lod_draw")
public func mmc_lod_draw(_ p: UnsafePointer<Float>, _ cam: UnsafePointer<Double>) -> Int32 {
    let r = LodRenderer.shared
    guard let enc = ctx.pass, !ctx.scissorEmpty else { return 0 }
    r.lock.lock()
    let ready = r.state == 2
    let quads = r.quadBuffer
    r.lock.unlock()
    guard ready, let quads else { return 0 }
    guard let pipe = r.pipeline(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat) else { return 0 }
    r.ensureColors()
    let cx = cam[0], cy = cam[1], cz = cam[2]
    let chosen = r.select(camX: cx, camZ: cz, splitFactor: 2.0)
    guard !chosen.isEmpty else { return 0 }
    r.ensureIndexBuffer(quads: chosen.map { r.nodes[$0].quadCount }.max() ?? 1)
    guard let ib = r.indexBuffer else { return 0 }

    func mat(_ o: Int) -> simd_float4x4 {
        simd_float4x4(SIMD4(p[o], p[o + 1], p[o + 2], p[o + 3]), SIMD4(p[o + 4], p[o + 5], p[o + 6], p[o + 7]),
                      SIMD4(p[o + 8], p[o + 9], p[o + 10], p[o + 11]), SIMD4(p[o + 12], p[o + 13], p[o + 14], p[o + 15]))
    }
    var u = LodUniforms(proj: mat(0), view: mat(16), fogColor: SIMD4(p[32], p[33], p[34], p[35]),
                        envStart: p[36], envEnd: p[37], rdStart: p[38], rdEnd: p[39], discardRadius: p[40], sky: p[41])
    // Frustum planes from proj * view (camera-relative space, reverse-Z: keep only the four side planes
    // and the near plane; the far plane is beyond the LOD).
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

    struct Draw { var node: Int; var first: Int; var count: Int }
    var draws: [Draw] = []
    var xforms = [SIMD4<Float>]()
    xforms.reserveCapacity(chosen.count)
    for i in chosen {
        let n = r.nodes[i]
        let lo = SIMD3(Float(Double(n.x0) - cx), Float(Double(lodWorldMinY) - cy), Float(Double(n.z0) - cz))
        let hi = lo + SIMD3(Float(n.size), Float(lodWorldHeight), Float(n.size))
        if !visible(lo, hi) { continue }
        // Face buckets that can face the camera (conservative: the camera is past the node's nearest plane).
        let camInX = -lo.x, camInY = -lo.y, camInZ = -lo.z   // camera position relative to the node corner
        let s = Float(n.size), top = Float(lodWorldHeight)
        let faceVisible = [camInX > 0, camInX < s, camInY > 0, camInY < top, camInZ > 0, camInZ < s]
        let slot = xforms.count
        xforms.append(SIMD4(lo.x, lo.y, lo.z, Float(1 << n.level)))
        var f = 0
        while f < 6 {
            if !faceVisible[f] || n.faceStart[f + 1] == n.faceStart[f] { f += 1; continue }
            var e = f + 1
            while e < 6 && (faceVisible[e] || n.faceStart[e + 1] == n.faceStart[e]) { e += 1 }
            draws.append(Draw(node: slot, first: n.firstQuad + n.faceStart[f], count: n.faceStart[e] - n.faceStart[f]))
            f = e
        }
    }
    guard !draws.isEmpty else { return 0 }

    enc.setRenderPipelineState(pipe)
    enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: true))
    enc.setCullMode(.back)
    enc.setTriangleFillMode(.fill)
    enc.setDepthBias(0, slopeScale: 0, clamp: 0)
    enc.setVertexBuffer(quads, offset: 0, index: 18)
    enc.setVertexBytes(&u, length: MemoryLayout<LodUniforms>.stride, index: 19)
    enc.setFragmentBytes(&u, length: MemoryLayout<LodUniforms>.stride, index: 19)
    if xforms.count * 16 <= 4096 {
        enc.setVertexBytes(xforms, length: xforms.count * 16, index: 20)
    } else if let xb = ctx.device.makeBuffer(bytes: xforms, length: xforms.count * 16, options: [.storageModeShared]) {
        enc.setVertexBuffer(xb, offset: 0, index: 20)
    }
    enc.setVertexBuffer(r.colorBuffer!, offset: 0, index: 21)
    for d in draws {
        enc.drawIndexedPrimitives(type: .triangle, indexCount: d.count * 6, indexType: .uint32, indexBuffer: ib,
                                  indexBufferOffset: 0, instanceCount: 1, baseVertex: 4 * d.first, baseInstance: d.node)
    }
    // Minecraft's pipeline, depth, cull and bias state must be re-applied by the next setPipeline.
    ctx.pipe = nil
    ctx.boundPipeState = nil
    return Int32(draws.count)
}
