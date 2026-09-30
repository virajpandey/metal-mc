import Foundation
import Metal
import MetalMCCore
import simd

// Far field (prototype, METALMC_FARFIELD=<level>): LOD levels from <level> up are drawn by a per-pixel height-field
// ray march instead of quads (docs/far-field-design.md). The quads' cost is vertex invocations, and at those levels
// 95% of them draw nothing (terrain seen at grazing angles); a ray march costs per pixel and has exact visibility.
//
// Data: every LOD node at those levels also keeps one word per column (LodBuild.farColumns). Per level there's a
// ring: a W x W window of that level's columns around the camera (one slice of a 2D array texture), and a max
// pyramid of column tops over it. A ring is rewritten on the GPU when the camera moves 64 of its cells or a node in
// its window changes. A bitmap of the area drawn by quads (levels below the far field's, 64-block cells) is rebuilt every frame and
// tested where a ray reaches a column: columns in it count as empty, so rays only find terrain the quads don't draw.
//
// Drawing: right after the LOD's opaque quads, a shell around the camera at the horizontal distance of the nearest tile
// the far field draws (a 64-sided prism from the world bottom up, and its floor). Every hit lies beyond it, so rays
// start there and the early depth test skips pixels with anything drawn nearer than the shell. Each pixel marches ring 0 until its ray leaves ring 0's
// window, then ring 1, and so on (each ring's cells are about 3.5 px wide where its window ends). A hit writes the
// column's color (top or side face, vanilla's lightmap, water over its floor, fog) and its depth.

/// The finest LOD level drawn by the far field (METALMC_FARFIELD); 0: off.
let lodFarFieldLevel = max(0, Int(ProcessInfo.processInfo.environment["METALMC_FARFIELD"] ?? "") ?? 0)
/// Debug (METALMC_EXP=ffsteps): color hits by the number of march steps (green few, red many), misses dark blue.
let farFieldSteps = experiments.contains("ffsteps")
let farFieldWidth = 1024        // cells per ring side
let farFieldMips = 11           // max pyramid levels: 1024 ... 1
let farFieldCoverCells = 256    // coverage bitmap side, 64-block cells (16 km)

extension LodBuild {
    /// One word per column of a node's grid for the far field: bits 0-8 the height of the top solid voxel's top
    /// (voxels above the world bottom, 0 for none), 9-15 voxels of water above it, 16-23 the top voxel's material,
    /// 24-31 the water's material if there's water, else the material under the top voxel.
    static func farColumns(_ g: LodGrid) -> [UInt32] {
        let n = lodNodeVoxels, layer = n * n
        var out = [UInt32](repeating: 0, count: layer)
        g.v.withUnsafeBufferPointer { v in
            for i in 0..<layer {
                var y = g.height - 1
                while y >= 0 && v[y * layer + i] == 0 { y -= 1 }
                if y < 0 { continue }
                var waterTop = -1
                var waterMat: UInt8 = 0
                if lodIsWater(v[y * layer + i]) {
                    waterTop = y
                    waterMat = v[y * layer + i]
                    while y >= 0 && (lodIsWater(v[y * layer + i]) || v[y * layer + i] == 0) { y -= 1 }
                }
                let top: UInt8 = y >= 0 ? v[y * layer + i] : 0
                var sub = top
                if y > 0, v[(y - 1) * layer + i] != 0, !lodIsWater(v[(y - 1) * layer + i]) { sub = v[(y - 1) * layer + i] }
                let depth = waterTop >= 0 ? min(127, waterTop - y) : 0
                out[i] = UInt32(y + 1) | UInt32(depth) << 9 | UInt32(top) << 16 | UInt32(waterTop >= 0 ? waterMat : sub) << 24
            }
        }
        return out
    }
}

/// Uniforms of the march (must match FarUniforms in the shader).
struct FarUniforms {
    var invViewProj: simd_float4x4          // inverse of the jittered projection * view rotation (camera-relative)
    var viewport: SIMD4<Float>               // width, height, depth of the full-screen triangle, rings
    var cam: SIMD4<Float>                    // x: camera height above the world bottom (blocks); y: water alpha;
                                             // z: shell radius (blocks); w: shell top relative to the camera
    var ring: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
                                             // per ring: camera in ring-local cells (xy), cell size in blocks (z)
    var ringCover: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
                                             // per ring: its texel 0 in blocks from the coverage bitmap's corner (xy)
}

let farFieldShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct LodUniforms {
    float4x4 proj;
    float4x4 view;
    float4 fogColor;
    float envStart, envEnd, rdStart, rdEnd;
    float discardRadius;
    float sky;
    float alpha;
    float lightmapOn;
    float4 camFrac;
    float4 camInSection;
    int4 seamInfo;
};
struct FarUniforms {
    float4x4 invViewProj;
    float4 viewport;
    float4 cam;
    float4 ring[8];
    float4 ringCover[8];
};
struct FillParams { int2 nodeCell; int2 ringOrigin; uint ring; uint level; };

constant int W = \(farFieldWidth);
constant int TOP = \(farFieldMips - 1);
constant int COVER = \(farFieldCoverCells);

kernel void ff_clear(uint2 gid [[thread_position_in_grid]], constant uint& slice [[buffer(0)]],
                     texture2d_array<uint, access::write> data [[texture(0)]],
                     texture2d_array<ushort, access::write> heights [[texture(1)]]) {
    if (gid.x >= uint(W) || gid.y >= uint(W)) return;
    data.write(uint4(0), gid, slice);
    heights.write(ushort4(0), gid, slice);
}

// One node's columns into its level's ring.
kernel void ff_fill(uint2 gid [[thread_position_in_grid]], constant FillParams& p [[buffer(0)]],
                    const device uint* cols [[buffer(1)]],
                    texture2d_array<uint, access::write> data [[texture(0)]],
                    texture2d_array<ushort, access::write> heights [[texture(1)]]) {
    if (gid.x >= 256u || gid.y >= 256u) return;
    int2 cell = p.nodeCell + int2(gid);
    int2 local = cell - p.ringOrigin;
    if (local.x < 0 || local.y < 0 || local.x >= W || local.y >= W) return;
    uint c = cols[gid.y * 256u + gid.x];
    data.write(uint4(c), uint2(local), p.ring);
    uint top = ((c & 511u) + ((c >> 9) & 127u)) << p.level;
    heights.write(ushort4(ushort(c == 0u ? 0u : top)), uint2(local), p.ring);
}

kernel void ff_mip(uint2 gid [[thread_position_in_grid]], constant uint& slice [[buffer(0)]],
                   texture2d_array<ushort, access::read> src [[texture(0)]],
                   texture2d_array<ushort, access::write> dst [[texture(1)]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 s = gid * 2u;
    ushort a = max(src.read(s, slice).r, src.read(s + uint2(1, 0), slice).r);
    ushort b = max(src.read(s + uint2(0, 1), slice).r, src.read(s + uint2(1, 1), slice).r);
    dst.write(ushort4(max(a, b)), gid, slice);
}

// The shell: 64 wall segments (6 vertices each) inscribed in the circle of radius cam.z, then the floor (3 each).
vertex float4 ff_vs(uint vid [[vertex_id]], constant LodUniforms& u [[buffer(19)]], constant FarUniforms& f [[buffer(25)]]) {
    const uint wallCorner[6] = { 0, 1, 1, 0, 1, 0 }, wallTop[6] = { 0, 0, 1, 0, 1, 1 };
    float yb = -f.cam.x, yt = f.cam.w;
    float3 rel;
    if (vid < 384u) {
        uint seg = vid / 6u, k = vid % 6u;
        float a = float(seg + wallCorner[k]) * (2.0 * M_PI_F / 64.0);
        rel = float3(f.cam.z * cos(a), wallTop[k] ? yt : yb, f.cam.z * sin(a));
    } else {
        uint seg = (vid - 384u) / 3u, k = (vid - 384u) % 3u;
        float a = float(seg + (k == 2u ? 1u : 0u)) * (2.0 * M_PI_F / 64.0);
        rel = k == 0u ? float3(0.0, yb, 0.0) : float3(f.cam.z * cos(a), yb, f.cam.z * sin(a));
    }
    float4 clip = u.proj * (u.view * float4(rel, 1.0));
    clip.y = -clip.y;
    return clip;
}

constant float kShade[6] = { 0.6, 0.6, 1.0, 0.5, 0.8, 0.8 };

static float3 lodLight(constant LodUniforms& u, texture2d<float> lightmap, sampler s, float skyLevel) {
    if (u.lightmapOn > 0.5) return lightmap.sample(s, float2(0.5 / 16.0, (skyLevel + 0.5) / 16.0), level(0)).rgb;
    return float3(mix(0.2, 1.0, u.sky) * (0.04 + 0.96 * skyLevel / 15.0));
}
static float linearFog(float d, float s, float e) {
    if (d <= s) return 0.0;
    if (d >= e) return 1.0;
    return (d - s) / (e - s);
}

struct FFOut { float4 color [[color(0)]]; float depth [[depth(less)]]; };
struct LodSpriteGPU { float4 top; float4 side; float4 luma; };

// Texture detail like the LOD's (lodShade): the texel's luma relative to the texture's mean luma, from a mip picked for
// the pixel's footprint on the face (the march has no smooth derivatives at column edges, so no gradients).
static float detail(constant LodUniforms& u, constant LodSpriteGPU* sprites, texture2d<float> atlas, sampler as, uint m,
                    int face, float3 rel, float mip) {
    if (u.camFrac.w < 0.5) return 1.0;
    uint mat = (m >= \(lodWaterBase)u && m < \(lodWaterBase + 32)u) ? \(Mat.water.rawValue)u
             : ((m >= \(lodLeavesBase)u && m < \(lodLeavesBase + 32)u) ? \(Mat.leaves.rawValue)u
             : ((m >= \(lodGrassBase)u && m < \(lodGrassBase + 32)u) ? \(Mat.grass.rawValue)u : m));
    float3 wp = rel + u.camFrac.xyz;
    float2 bc = face < 2 ? float2(wp.z, -wp.y) : (face < 4 ? wp.xz : float2(wp.x, -wp.y));
    bool top = face == 2 || face == 3;
    float4 rect = top ? sprites[mat].top : sprites[mat].side;
    float4 t = atlas.sample(as, rect.xy + fract(bc) * (rect.zw - rect.xy), level(mip));
    float luma = dot(t.rgb, float3(0.2126, 0.7152, 0.0722));
    float mean = top ? sprites[mat].luma.x : sprites[mat].luma.y;
    return clamp(mix(mat == \(Mat.water.rawValue)u ? 1.0 : 0.75, luma / max(mean, 0.02), t.a), 0.0, 2.0);
}

fragment FFOut ff_fs(float4 pos [[position]], constant LodUniforms& u [[buffer(19)]], constant FarUniforms& f [[buffer(25)]],
                     constant float4* colors [[buffer(26)]],
                     texture2d_array<uint, access::read> data [[texture(27)]],
                     texture2d_array<ushort, access::read> heights [[texture(28)]],
                     texture2d<float> lightmap [[texture(29)]], sampler ls [[sampler(14)]],
                     constant LodSpriteGPU* sprites [[buffer(20)]], texture2d<float> atlas [[texture(30)]],
                     sampler atlasSampler [[sampler(15)]], const device uchar* cover [[buffer(28)]]) {
    float2 uv = pos.xy / f.viewport.xy;
    float4 hp = f.invViewProj * float4(uv * 2.0 - 1.0, 1.0, 1.0);
    float3 dir = normalize(hp.xyz / hp.w);
    uint rings = uint(f.viewport.w);
    float t = f.cam.z / max(length(dir.xz), 1e-6) * 0.999;   // nothing to find inside the shell
    int steps = 0;
    bool hit = false;
    float tHit = 0.0;
    int face = 2;
    uint col = 0;
    float s = 1.0;
    for (uint r = 0; r < rings && !hit; r++) {
        float4 rc = f.ring[r];
        s = rc.z;
        float3 o = float3(rc.x, f.cam.x, rc.y);
        float3 d = float3(dir.x / s, dir.y, dir.z / s);
        float2 inv = 1.0 / d.xz;
        float2 ta = (0.0 - o.xz) * inv, tb = (float(W) - o.xz) * inv;
        float2 tmn = min(ta, tb), tmx = max(ta, tb);
        float tEnter = max(max(tmn.x, tmn.y), t);
        float tLeave = min(tmx.x, tmx.y);
        if (!(tEnter < tLeave)) continue;
        int lastAxis = tmn.x > tmn.y ? 0 : 1;
        int l = TOP;   // rays over everything in the ring (the sky) leave it in one step
        float tc = tEnter;
        for (int i = 0; i < 192; i++) {
            steps++;
            float3 p = o + d * tc;
            float cs = float(1 << l);
            float2 cell = clamp(floor(p.xz / cs), 0.0, float(W >> l) - 1.0);
            float hmax = float(heights.read(uint2(cell), r, l).r);
            float2 nb = (cell + select(float2(0.0), float2(1.0), d.xz > 0.0)) * cs;
            float2 tt = (nb - o.xz) * inv;
            float tExit = min(min(tt.x, tt.y), tLeave);
            float yA = o.y + d.y * tc, yB = o.y + d.y * tExit;
            if (hmax <= 0.0 || min(yA, yB) > hmax) {
                if (tExit >= tLeave) break;
                lastAxis = tt.x < tt.y ? 0 : 1;
                tc = tExit + max(tExit * 1e-5, 1e-3);
                l = min(l + 1, TOP);
            } else if (l > 0) {
                l--;
            } else {
                // A column in the quads' area: they draw it, so it's empty here.
                float2 cb = floor((f.ringCover[r].xy + cell * s) / 64.0);
                if (all(cb >= 0.0) && all(cb < float(COVER)) && cover[int(cb.y) * COVER + int(cb.x)] != 0) {
                    if (tExit >= tLeave) break;
                    lastAxis = tt.x < tt.y ? 0 : 1;
                    tc = tExit + max(tExit * 1e-5, 1e-3);
                    continue;
                }
                col = data.read(uint2(cell), r).r;
                // Water surfaces sit 10/9 block below the voxel's top, as the LOD draws them (kWaterSurfaceDrop).
                float top = float((col & 511u) + ((col >> 9) & 127u)) * s - (((col >> 9) & 127u) != 0u ? 10.0 / 9.0 : 0.0);
                if (yA <= top) {
                    tHit = tc;
                    face = lastAxis == 0 ? (d.x > 0.0 ? 1 : 0) : (d.z > 0.0 ? 5 : 4);
                } else {
                    tHit = (top - o.y) / d.y;
                    face = 2;
                }
                hit = true;
                break;
            }
        }
        t = tLeave;
    }
    FFOut out;
    if (!hit) {
        if (\(farFieldSteps ? "true" : "false")) { out.color = float4(0.0, 0.0, 0.25, 1.0); out.depth = 0.0; return out; }
        discard_fragment();
        return out;
    }
    float3 rel = dir * tHit;
    float y = f.cam.x + rel.y;
    uint solid = col & 511u, depthVox = (col >> 9) & 127u;
    float solidTop = float(solid) * s;
    uint topMat = (col >> 16) & 255u, lowMat = col >> 24;
    float sky = float(u.seamInfo.z);
    float3 light = lodLight(u, lightmap, ls, sky);
    // About one texel per pixel: the pixel's footprint on the face, in blocks, times 16 texels per block.
    float3 n = face == 2 ? float3(0.0, 1.0, 0.0) : (face < 2 ? float3(1.0, 0.0, 0.0) : float3(0.0, 0.0, 1.0));
    float footprint = tHit * 2.0 / (f.viewport.y * u.proj[1][1]) / max(abs(dot(dir, n)), 0.05);
    float mip = max(0.0, log2(footprint * 16.0));
    float3 color;
    if (depthVox > 0u && face == 2) {
        // Water over its floor, like the LOD's translucent water over its meshed floor (lit for the water above it).
        float depthBlocks = min(15.0, float(depthVox) * s - 1.0);
        float3 floorColor = colors[topMat * 3u].rgb * detail(u, sprites, atlas, atlasSampler, topMat, 2, rel, mip)
                          * lodLight(u, lightmap, ls, max(0.0, sky - depthBlocks));
        float a = f.cam.y;
        color = colors[lowMat * 3u].rgb * light * a + floorColor * (1.0 - a);
    } else if (depthVox > 0u) {
        // The side of a water column. Rays that pass under the quads' water reach the first far-field column from the
        // side, below its surface: that's terrain seen through the water (the quads' water surface is drawn over it),
        // so it's the floor (or the solid side), lit for the water above it, not another water surface.
        float waterTop = float(solid + depthVox) * s - 10.0 / 9.0;
        bool side = y < solidTop;
        float depthBlocks = min(15.0, waterTop - (side ? y : solidTop));
        color = colors[topMat * 3u + (side ? 1u : 0u)].rgb * (side ? kShade[face] : 1.0)
              * detail(u, sprites, atlas, atlasSampler, topMat, side ? face : 2, rel, mip) * lodLight(u, lightmap, ls, max(0.0, sky - depthBlocks));
    } else {
        uint mat = (face == 2 || y >= solidTop - s) ? topMat : lowMat;
        color = colors[mat * 3u + (face == 2 ? 0u : 1u)].rgb * kShade[face] * light
              * detail(u, sprites, atlas, atlasSampler, mat, face, rel, mip);
    }
    if (\(farFieldSteps ? "true" : "false")) color = mix(float3(0.0, 1.0, 0.0), float3(1.0, 0.0, 0.0), saturate(float(steps) / 128.0));
    float horiz = length(rel.xz);
    float fog = max(linearFog(length(rel), u.envStart, u.envEnd), linearFog(max(horiz, abs(rel.y)), u.rdStart, u.rdEnd));
    out.color = float4(mix(color, u.fogColor.rgb, fog * u.fogColor.a), 1.0);
    float4 clip = u.proj * (u.view * float4(rel, 1.0));
    out.depth = min(clip.z / clip.w, pos.z);
    return out;
}
"""

/// Fill parameters (must match FillParams in the shader).
struct FarFillParams {
    var nodeCell: SIMD2<Int32>
    var ringOrigin: SIMD2<Int32>
    var ring: UInt32
    var level: UInt32
}

final class FarField: @unchecked Sendable {
    static let shared = FarField()

    private var library: MTLLibrary?
    private var clearPipe: MTLComputePipelineState?
    private var fillPipe: MTLComputePipelineState?
    private var mipPipe: MTLComputePipelineState?
    private var drawPipes: [String: MTLRenderPipelineState] = [:]
    private var compiling = false
    private let lock = NSLock()

    private var data: MTLTexture?          // R32Uint, W x W x rings: packed columns (LodBuild.farColumns)
    private var heights: MTLTexture?       // R16Uint with mips, W x W x rings: column tops in blocks above the world bottom
    private var mipViews: [MTLTexture] = []
    private var rings = 0
    private var origins: [SIMD2<Int>] = []  // per ring: world cell (of its level) at texel 0
    private var coverOrigin = SIMD2<Int>(0, 0)
    private var cover = [UInt8](repeating: 0, count: farFieldCoverCells * farFieldCoverCells)
    private var ringKeys: [[Int]] = []     // per ring: its window and the nodes it was filled from
    private var coverBuffer: MTLBuffer?    // the bitmap the current frame draws with (a new one whenever it changes)
    private var lastCover: [UInt8] = []
    private(set) var ready = false
    var fills = 0

    /// Compiles the shaders in the background; nil until they're ready.
    private func ensurePipelines() -> Bool {
        lock.lock(); defer { lock.unlock() }
        if clearPipe != nil { return true }
        if compiling { return false }
        compiling = true
        DispatchQueue.global(qos: .userInitiated).async { [self] in
            do {
                let lib = try ctx.device.makeLibrary(source: farFieldShaderSource, options: nil)
                let c = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "ff_clear")!)
                let f = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "ff_fill")!)
                let m = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "ff_mip")!)
                lock.lock(); library = lib; clearPipe = c; fillPipe = f; mipPipe = m; lock.unlock()
                log("far field: shaders compiled")
            } catch {
                log("far field: shader compile failed: \(error)")
            }
        }
        return false
    }

    private func drawPipe(colorFormats: [MTLPixelFormat], depth: MTLPixelFormat) -> MTLRenderPipelineState? {
        let key = colorFormats.map { String($0.rawValue) }.joined(separator: ",") + "/\(depth.rawValue)"
        lock.lock(); defer { lock.unlock() }
        if let p = drawPipes[key] { return p }
        guard let library else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.label = "MetalMC far field"
        d.vertexFunction = library.makeFunction(name: "ff_vs")
        d.fragmentFunction = library.makeFunction(name: "ff_fs")
        for (i, f) in colorFormats.enumerated() {
            d.colorAttachments[i].pixelFormat = f
            if i > 0 { d.colorAttachments[i].writeMask = [] }
        }
        d.depthAttachmentPixelFormat = depth
        do {
            let p = try ctx.device.makeRenderPipelineState(descriptor: d)
            drawPipes[key] = p
            return p
        } catch {
            log("far field: pipeline failed: \(error)")
            return nil
        }
    }

    private func ensureTextures(rings n: Int) -> Bool {
        if n == rings, data != nil { return true }
        let d = MTLTextureDescriptor()
        d.textureType = .type2DArray
        d.width = farFieldWidth
        d.height = farFieldWidth
        d.arrayLength = n
        d.pixelFormat = .r32Uint
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        guard let dt = ctx.device.makeTexture(descriptor: d) else { return false }
        d.pixelFormat = .r16Uint
        d.mipmapLevelCount = farFieldMips
        guard let ht = ctx.device.makeTexture(descriptor: d) else { return false }
        dt.label = "MetalMC far field columns"
        ht.label = "MetalMC far field heights"
        mipViews = (0..<farFieldMips).compactMap { ht.makeTextureView(pixelFormat: .r16Uint, textureType: .type2DArray, levels: $0..<($0 + 1), slices: 0..<n) }
        data = dt
        heights = ht
        rings = n
        ringKeys = []
        return mipViews.count == farFieldMips
    }

    @inline(__always) private static func floorDiv(_ a: Int, _ b: Int) -> Int { a >= 0 ? a / b : -((-a + b - 1) / b) }

    /// Before the LOD draws: rewrites the rings if the camera moved far enough, the nodes changed or the area drawn by
    /// quads changed. Returns true if the far field draws this frame (then the LOD skips its levels' quads).
    func prepare(meshes: [LodNodeKey: LodMeshNode], generation: Int, chosen: [(LodMeshNode, UInt16)], maxLevel: Int,
                 cx: Double, cz: Double) -> Bool {
        let k = lodFarFieldLevel
        guard k > 0, maxLevel >= k, ensurePipelines(), ensureTextures(rings: min(8, maxLevel - k + 1)) else { return false }
        camX = cx
        camZ = cz
        // Ring windows: the camera's cell, rounded down to 64 cells, minus half the width.
        origins = (0..<rings).map { r in
            let s = 1 << (k + r)
            let ccx = Self.floorDiv(Int(cx.rounded(.down)), s), ccz = Self.floorDiv(Int(cz.rounded(.down)), s)
            return SIMD2(Self.floorDiv(ccx, 64) * 64 - farFieldWidth / 2, Self.floorDiv(ccz, 64) * 64 - farFieldWidth / 2)
        }
        // The quads' area: chosen tiles below the far field's level, in 64-block cells.
        let half = farFieldCoverCells / 2
        coverOrigin = SIMD2(Self.floorDiv(Int(cx.rounded(.down)), 1024) * 16 - half, Self.floorDiv(Int(cz.rounded(.down)), 1024) * 16 - half)
        for i in 0..<cover.count { cover[i] = 0 }
        for (n, mask) in chosen where n.level < k {
            let tileBlocks = lodTileVoxels << n.level, span = tileBlocks / 64
            for t in 0..<16 where mask & (1 << UInt16(t)) != 0 {
                let bx = Self.floorDiv(n.x0 + (t % 4) * tileBlocks, 64) - coverOrigin.x
                let bz = Self.floorDiv(n.z0 + (t / 4) * tileBlocks, 64) - coverOrigin.y
                for z in max(0, bz)..<min(farFieldCoverCells, bz + span) {
                    for x in max(0, bx)..<min(farFieldCoverCells, bx + span) { cover[z * farFieldCoverCells + x] = 1 }
                }
            }
        }
        if cover != lastCover || coverBuffer == nil {
            guard let b = ctx.device.makeBuffer(bytes: cover, length: cover.count, options: [.storageModeShared]) else { return false }
            coverBuffer = b
            lastCover = cover
        }
        // Each ring's nodes (a rebuilt node is a new object), and the rings whose window or nodes changed.
        var perRing = [[(LodMeshNode, SIMD2<Int>)]](repeating: [], count: rings)
        var keys = origins.map { [$0.x, $0.y] }
        for (key, node) in meshes {
            let r = key.level - k
            guard r >= 0, r < rings, node.columns != nil else { continue }
            let cell = SIMD2(node.x0 >> key.level, node.z0 >> key.level)
            let o = origins[r]
            if cell.x + 256 <= o.x || cell.y + 256 <= o.y || cell.x >= o.x + farFieldWidth || cell.y >= o.y + farFieldWidth { continue }
            perRing[r].append((node, cell))
            keys[r].append(ObjectIdentifier(node).hashValue)
        }
        for r in 0..<rings { keys[r] = Array(keys[r][..<2]) + keys[r][2...].sorted() }
        let stale = (0..<rings).filter { ringKeys.count != rings || ringKeys[$0] != keys[$0] }
        if stale.isEmpty { return ready }
        guard let data, let clearPipe, let fillPipe, let mipPipe,
              let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        cb.label = "MetalMC far field fill"
        var nodes = 0
        for r in stale {
            var slice = UInt32(r)
            enc.setComputePipelineState(clearPipe)
            enc.setBytes(&slice, length: 4, index: 0)
            enc.setTexture(data, index: 0)
            enc.setTexture(mipViews[0], index: 1)
            enc.dispatchThreads(MTLSize(width: farFieldWidth, height: farFieldWidth, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.setComputePipelineState(fillPipe)
            let o = origins[r]
            for (node, cell) in perRing[r] {
                var p = FarFillParams(nodeCell: SIMD2(Int32(cell.x), Int32(cell.y)), ringOrigin: SIMD2(Int32(o.x), Int32(o.y)),
                                      ring: UInt32(r), level: UInt32(k + r))
                enc.setBytes(&p, length: MemoryLayout<FarFillParams>.stride, index: 0)
                enc.setBuffer(node.columns, offset: 0, index: 1)
                enc.dispatchThreads(MTLSize(width: 256, height: 256, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                nodes += 1
            }
            enc.setComputePipelineState(mipPipe)
            enc.setBytes(&slice, length: 4, index: 0)
            for m in 1..<farFieldMips {
                enc.setTexture(mipViews[m - 1], index: 0)
                enc.setTexture(mipViews[m], index: 1)
                let w = max(1, farFieldWidth >> m)
                enc.dispatchThreads(MTLSize(width: w, height: w, depth: 1), threadsPerThreadgroup: MTLSize(width: min(16, w), height: min(16, w), depth: 1))
            }
        }
        enc.endEncoding()
        // Committed now, ahead of the frame's own command buffer (committed at the end of the frame), so this frame's
        // draw reads the new rings.
        cb.commit()
        ringKeys = keys
        fills += 1
        if fills % 20 == 1 { log("far field: fill \(fills): rings \(stale) of \(rings) from level \(k), \(nodes) nodes") }
        ready = true
        return true
    }

    /// Draws the march inside the LOD's pass (after its opaque quads). `u` is the LOD's uniforms for this frame.
    func draw(_ enc: MTLRenderCommandEncoder, u: LodUniforms, colors: MTLBuffer, lightmap: MTLTexture, lightSampler: MTLSamplerState?,
              cy: Double, nearest: Double) {
        guard ready, let data, let heights, let coverBuffer, let pipe = drawPipe(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat) else { return }
        let k = lodFarFieldLevel
        let vp = u.proj * u.view
        var f = FarUniforms(invViewProj: vp.inverse,
                            viewport: SIMD4(Float(ctx.passWidth), Float(ctx.passHeight), 0, Float(rings)),
                            cam: SIMD4(Float(cy - Double(lodWorldMinY)), lodWaterAlpha, Float(nearest),
                                       Float(max(Double(lodWorldHeight) - (cy - Double(lodWorldMinY)), 0) + 1)),
                            ring: (.zero, .zero, .zero, .zero, .zero, .zero, .zero, .zero),
                            ringCover: (.zero, .zero, .zero, .zero, .zero, .zero, .zero, .zero))
        withUnsafeMutableBytes(of: &f.ring) { raw in
            let rp = raw.bindMemory(to: SIMD4<Float>.self)
            for r in 0..<rings {
                let s = Double(1 << (k + r))
                rp[r] = SIMD4(Float(camX / s - Double(origins[r].x)), Float(camZ / s - Double(origins[r].y)), Float(s), 0)
            }
        }
        withUnsafeMutableBytes(of: &f.ringCover) { raw in
            let rp = raw.bindMemory(to: SIMD4<Float>.self)
            for r in 0..<rings {
                let s = 1 << (k + r)
                rp[r] = SIMD4(Float(origins[r].x * s - coverOrigin.x * 64), Float(origins[r].y * s - coverOrigin.y * 64), 0, 0)
            }
        }
        var uu = u
        enc.setRenderPipelineState(pipe)
        enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: true))
        enc.setCullMode(.none)
        enc.setVertexBytes(&f, length: MemoryLayout<FarUniforms>.stride, index: 25)
        enc.setFragmentBytes(&f, length: MemoryLayout<FarUniforms>.stride, index: 25)
        enc.setVertexBytes(&uu, length: MemoryLayout<LodUniforms>.stride, index: 19)
        enc.setFragmentBytes(&uu, length: MemoryLayout<LodUniforms>.stride, index: 19)
        enc.setFragmentBuffer(colors, offset: 0, index: 26)
        enc.setFragmentTexture(data, index: 27)
        enc.setFragmentTexture(heights, index: 28)
        enc.setFragmentTexture(lightmap, index: 29)
        enc.setFragmentSamplerState(lightSampler, index: 14)
        enc.setFragmentBuffer(coverBuffer, offset: 0, index: 28)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 64 * 9)
        enc.setCullMode(.back)
    }
    var camX = 0.0, camZ = 0.0
}
