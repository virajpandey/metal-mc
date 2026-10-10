import Foundation
import Metal
import simd

// Shadows from block lights (part of colored block light, METALMC_EXP=lit,coloredlight with rtshadows; strength
// METALMC_CLSHADOW, off by default: -PclShadow=0.75 to try them). The light volume (ColoredLight.swift) spreads each color as vanilla spreads block
// light, so its light goes around a pillar and fills the space behind it; a torch casts no shadow. Here, after the sun's
// shadow rays and before the relight, one ray per 4 x 4 pixels (a different pixel of the block each frame, like the sun's)
// goes from the surface toward the light its light comes from, through the sun shadows' acceleration structures
// (RtShadows.swift: the LOD's blocks, the near terrain included); where something is in the way the relight takes the
// colored light down by the strength. The light is found from the volume itself: in the cell the face looks into, a
// bucket is picked by its share of the cell's light (a different pick each frame where colors mix; the anti-aliasing
// averages them), then the bucket's light is climbed, each step to the brightest of the six neighbors, until no neighbor
// is brighter: vanilla's rule makes light fall at least a level a cell away from a light, so that cell is the light (at
// most 15 steps). The ray ends where it enters the light's cell (an opaque light, glowstone, is in the structure), aimed at
// a point in the middle of the cell that moves every frame (soft edges); a face turned away from the light is in shadow.
// Each sample is blended into a history of their own (CLS_BLEND of it, about three frames' worth: the surface's place last
// frame from the camera's motion, its texels whose distance matches): rays from a wall toward a light on it flip between
// hit and miss from frame to frame, a grain the anti-aliasing alone doesn't settle.

/// The shadows' strength (METALMC_CLSHADOW, 0: none; 0.75 is a good start): the share of the colored light a shadowed
/// pixel loses. Off by default until their noise next to their lights is gone (docs/lighting-design.md).
let clShadowStrength: Float = max(0, min(1, Float(ProcessInfo.processInfo.environment["METALMC_CLSHADOW"] ?? "") ?? 0))

private let clShadowSource = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

#define CL_SX \(clSizeX)
#define CL_SY \(clSizeY)
#define CL_SZ \(clSizeZ)
#define CL_BX \(clBricksX)
#define CL_BY \(clBricksY)

struct ClShadowParams {
    float4x4 invViewProj;   // inverse of the (jittered) projection * view rotation the level was drawn with
    float4 sizes;           // full width, full height, traced width, traced height
    float4 camTex;          // xyz: the camera in the volume's (toroidal) cell coordinates
    float4 camVol;          // xyz: the camera relative to the volume's min corner
    float4 camOffset;       // xyz: the camera relative to the instance structure's origin
    uint4 sample;           // x: pixels per sample along each axis, y-z: this frame's pixel within the block, w: frame
    float4 weight[2];       // per bucket: its color's luminance
    float4 curve[4];        // vanilla's block light (linear luminance) at levels 0-15
    float4x4 prevViewProj;  // last frame's projection * view rotation (the history)
    float4 camDelta;        // xyz: this frame's camera minus last frame's, w: 1 to start the history over
};

// The history's share of each frame's sample: a few frames of their own (rays from a wall toward a light on it flip between
// hit and miss from frame to frame, which the anti-aliasing alone doesn't settle).
#ifndef CLS_BLEND
#define CLS_BLEND 0.3
#endif

static uint clsIndex(int3 t) {
    uint3 u = uint3(t & int3(CL_SX - 1, CL_SY - 1, CL_SZ - 1));
    return (((u.z >> 3) * CL_BY + (u.y >> 3)) * CL_BX + (u.x >> 3)) * 512u + ((u.z & 7u) * 8u + (u.y & 7u)) * 8u + (u.x & 7u);
}

static float3 clsRel(constant ClShadowParams& p, uint2 q, float z) {
    float2 uv = (float2(q) + 0.5) / p.sizes.xy;
    float4 h = p.invViewProj * float4(uv * 2.0 - 1.0, z, 1.0);
    return h.xyz / h.w;
}

static float clsRand(uint2 q, uint f, uint k) {
    uint h = (q.x * 0x8da6b343u) ^ (q.y * 0xd8163841u) ^ ((f * 4u + k) * 0xcb1ab31fu);
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return float(h >> 8) * (1.0 / 16777216.0);
}

kernel void cl_shadow(instance_acceleration_structure accel [[buffer(0)]],
                      constant ClShadowParams& p [[buffer(1)]],
                      device const uint* light [[buffer(2)]],
                      depth2d<float, access::read> depth [[texture(0)]],
                      texture2d<half, access::write> out [[texture(1)]],
                      texture2d<half, access::read> history [[texture(2)]],
                      uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(p.sizes.z) || gid.y >= uint(p.sizes.w)) return;
    uint2 full = uint2(p.sizes.xy);
    uint2 fp = min(gid * p.sample.x + p.sample.yz, full - 1);
    float d = depth.read(fp);
    // This frame's sample: 1 where the light reaches the surface (or there's nothing to shadow), 0 where it doesn't.
    float v = 1.0;
    float3 pos = float3(0.0);
    bool surface = d > 0.0;
    do {
        if (!surface) break;
        pos = clsRel(p, fp, d);
        // The surface's normal as the sun's shadows take it: from the neighbors on the same surface, snapped to an axis.
        uint2 rx = uint2(min(fp.x + 1, full.x - 1), fp.y), lx = uint2(fp.x > 0 ? fp.x - 1 : 0, fp.y);
        uint2 dy = uint2(fp.x, min(fp.y + 1, full.y - 1)), uy = uint2(fp.x, fp.y > 0 ? fp.y - 1 : 0);
        float drx = depth.read(rx), dlx = depth.read(lx), ddy = depth.read(dy), duy = depth.read(uy);
        float3 ex = abs(drx - d) < abs(dlx - d) ? clsRel(p, rx, drx) - pos : pos - clsRel(p, lx, dlx);
        float3 ey = abs(ddy - d) < abs(duy - d) ? clsRel(p, dy, ddy) - pos : pos - clsRel(p, uy, duy);
        float3 n = cross(ey, ex);
        if (dot(n, -pos) < 0.0) n = -n;
        float3 an = abs(n);
        n = an.x > an.y && an.x > an.z ? float3(sign(n.x), 0, 0) : (an.y > an.z ? float3(0, sign(n.y), 0) : float3(0, 0, sign(n.z)));
        // The cell the face looks into, relative to the volume's min corner, and the corner's toroidal coordinates.
        int3 sz = int3(CL_SX, CL_SY, CL_SZ);
        int3 c = int3(floor(p.camVol.xyz + pos + n * 0.5));
        if (any(c < int3(1)) || any(c >= sz - 1)) break;
        int3 org = int3(round(p.camTex.xyz - p.camVol.xyz));
        uint L0 = light[clsIndex(c + org)];
        // A bucket, by its share of the cell's light.
        float w[8];
        float W = 0.0;
        for (uint k = 0u; k < 8u; k++) {
            uint l = (L0 >> (4u * k)) & 15u;
            w[k] = l > 0u ? p.curve[l >> 2u][l & 3u] * p.weight[k >> 2u][k & 3u] : 0.0;
            W += w[k];
        }
        if (W <= 1e-6) break;
        float u = clsRand(fp, p.sample.w, 0u) * W;
        uint b = 0u;
        for (uint k = 0u; k < 8u; k++) {
            if (w[k] <= 0.0) continue;
            b = k;
            if (u < w[k]) break;
            u -= w[k];
        }
        // Up the bucket's light to the light it comes from.
        int3 s = c;
        uint ls = (L0 >> (4u * b)) & 15u;
        for (int step = 0; step < 15; step++) {
            int3 best = s;
            uint bl = ls;
            for (int a = 0; a < 6; a++) {
                int3 t = s;
                t[a >> 1] += (a & 1) != 0 ? 1 : -1;
                if (any(t < int3(0)) || any(t >= sz)) continue;
                uint lt = (light[clsIndex(t + org)] >> (4u * b)) & 15u;
                if (lt > bl) { bl = lt; best = t; }
            }
            if (bl == ls) break;
            s = best;
            ls = bl;
        }
        // The ray, from just off the surface toward a point in the middle of the light's cell, stopped where it enters the
        // cell. (Farther off the surface than the sun's rays: these often run along the wall they start from, toward a
        // light on it.)
        float3 o = pos + n * (0.1 + length(pos) * 0.001);
        float3 lo = float3(s) - p.camVol.xyz;
        float3 jitter = float3(clsRand(fp, p.sample.w, 1u), clsRand(fp, p.sample.w, 2u), clsRand(fp, p.sample.w, 3u)) - 0.5;
        float3 dir = lo + 0.5 + jitter * 0.3 - o;
        float dist = length(dir);
        if (dist < 1e-3) break;
        dir /= dist;
        if (dot(dir, n) <= 0.0) { v = 0.0; break; }
        float3 inv = 1.0 / dir;
        float3 t0 = (lo - o) * inv, t1 = (lo + 1.0 - o) * inv;
        float enter = max(max(min(t0.x, t1.x), min(t0.y, t1.y)), min(t0.z, t1.z));
        float maxD = min(enter, dist) - 0.02;
        if (maxD <= 0.02) break;
        ray r;
        r.origin = o + p.camOffset.xyz;
        r.direction = dir;
        r.min_distance = 0.0;
        r.max_distance = maxD;
        intersector<instancing> isect;
        isect.accept_any_intersection(true);
        isect.assume_geometry_type(geometry_type::triangle);
        auto hit = isect.intersect(r, accel, 0xFF);
        v = hit.type == intersection_type::none ? 1.0 : 0.0;
    } while (false);
    // The history: where this surface was last frame (from the camera's motion), its texels whose distance matches,
    // bilinear, with CLS_BLEND of this frame's sample on top. Out: x the visibility, y the distance (blocks) of the last
    // sample, w 1 for a surface.
    float res = v;
    float len = length(pos);
    if (surface && p.camDelta.w < 0.5) {
        float3 prev = pos + p.camDelta.xyz;
        float4 pc = p.prevViewProj * float4(prev, 1.0);
        if (pc.w > 1e-6) {
            float2 puv = pc.xy / pc.w * 0.5 + 0.5;
            if (all(puv > 0.0) && all(puv < 1.0)) {
                float expect = length(prev);
                float2 hc = puv * p.sizes.zw - 0.5;
                int2 i0 = int2(floor(hc));
                float2 fr = hc - float2(i0);
                int2 hs = int2(history.get_width(), history.get_height());
                float acc = 0.0, wsum = 0.0;
                for (int k = 0; k < 4; k++) {
                    int2 o = int2(k & 1, k >> 1);
                    half4 hv = history.read(uint2(clamp(i0 + o, int2(0), hs - 1)));
                    float wb = (o.x == 1 ? fr.x : 1.0 - fr.x) * (o.y == 1 ? fr.y : 1.0 - fr.y);
                    float wt = hv.w > 0.5h && abs(float(hv.y) - expect) < 0.1 * max(expect, 1.0) ? wb : 0.0;
                    acc += float(hv.x) * wt;
                    wsum += wt;
                }
                if (wsum > 0.05) res = mix(acc / wsum, v, CLS_BLEND);
            }
        }
    }
    out.write(half4(half(res), half(min(len, 60000.0)), 0.0h, surface ? 1.0h : 0.0h), gid);
}
"""

private struct ClShadowParamsGPU {
    var invViewProj = matrix_identity_float4x4
    var sizes = SIMD4<Float>.zero
    var camTex = SIMD4<Float>.zero
    var camVol = SIMD4<Float>.zero
    var camOffset = SIMD4<Float>.zero
    var sample = SIMD4<UInt32>.zero
    var weight0 = SIMD4<Float>.zero, weight1 = SIMD4<Float>.zero
    var curve0 = SIMD4<Float>.zero, curve1 = SIMD4<Float>.zero, curve2 = SIMD4<Float>.zero, curve3 = SIMD4<Float>.zero
    var prevViewProj = matrix_identity_float4x4
    var camDelta = SIMD4<Float>.zero
}

/// Offline A/B (tools/litflow.swift, LITFLOW_CLSHADOWTEST=1): the shadows' history on (1) or off (0: each frame's samples
/// alone, as before it). Returns 1 with coloredlight.
@_cdecl("mmc_debug_cl_shadow_history")
public func mmc_debug_cl_shadow_history(_ on: Int32) -> Int32 {
    guard clEnabled else { return 0 }
    ClShadows.shared.historyOn = on != 0
    return 1
}

/// Offline compile check of the shadow kernel: 1 if it builds (errors to the log).
@_cdecl("mmc_debug_cl_shadow_compile")
public func mmc_debug_cl_shadow_compile() -> Int32 {
    do {
        let lib = try ctx.device.makeLibrary(source: clShadowSource, options: nil)
        guard let f = lib.makeFunction(name: "cl_shadow") else { return 0 }
        _ = try ctx.device.makeComputePipelineState(function: f)
        return 1
    } catch {
        log("coloredlight: shadow kernel failed: \(error)")
        return 0
    }
}

final class ClShadows: @unchecked Sendable {
    static let shared = ClShadows()

    private var library: MTLLibrary?
    private var pipe: MTLComputePipelineState?
    private var failed = false
    /// The shadows with their history (RGBA16Float: the visibility, the last sample's distance, 1 for a surface),
    /// ping-ponged: read one, write the other.
    private var history: [MTLTexture] = []
    private var current = 0
    private var prevViewProj = matrix_identity_float4x4
    private var prevCamOffset = SIMD3<Float>(repeating: .nan)
    private var lastNanos: UInt64 = 0
    private var dummy: MTLTexture?
    private var frame: UInt64 = 0
    /// This frame's shadows (set by trace, before the relight that reads them): the texture and its scale.
    private var traced: (texture: MTLTexture, scale: Int)?
    private(set) var traces = 0
    /// Offline A/B (mmc_debug_cl_shadow_history): false starts the history over every frame (each frame's samples alone).
    var historyOn = true

    /// A 4 x 4 visiting order that spreads consecutive frames' samples apart (as the sun's shadows').
    private static let pattern: [SIMD2<UInt32>] = [0, 10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 14, 6, 12].map { SIMD2(UInt32($0 % 4), UInt32($0 / 4)) }

    /// ClFrame.shadow for the relight: strength, pixels per sample, 1 if traced this frame.
    var frameShadow: SIMD4<Float> {
        guard let t = traced else { return .zero }
        return SIMD4(clShadowStrength, Float(t.scale), 1, 0)
    }

    /// The texture the relight binds (this frame's shadows, or a 1 x 1 stand-in it doesn't read).
    var texture: MTLTexture {
        if let t = traced { return t.texture }
        if let dummy { return dummy }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: 1, height: 1, mipmapped: false)
        d.usage = .shaderRead
        let t = ctx.device.makeTexture(descriptor: d)!
        dummy = t
        return t
    }

    private func ensure() -> Bool {
        if failed { return false }
        if pipe != nil { return true }
        do {
            // Lab mode (ShaderLab.swift): after an edit of colored_light_shadows.metal the kernel is built again.
            let lib = try ShaderLab.library("colored_light_shadows", clShadowSource) { [self] _ in library = nil; pipe = nil; failed = false }
            guard let f = lib.makeFunction(name: "cl_shadow") else { failed = true; return false }
            pipe = try ctx.device.makeComputePipelineState(function: f)
            library = lib
            return true
        } catch {
            log("coloredlight: shadow kernel failed: \(error)")
            failed = true
            return false
        }
    }

    /// Lit.relight, before it relights (or leaves the relight for the anti-aliasing): this frame's rays toward the block
    /// lights, on the sun shadows' structures (RtShadows ran before), the frame's depth and projection.
    func trace(cb: MTLCommandBuffer, depth: MTLTexture, invViewProj: simd_float4x4, width: Int, height: Int) {
        traced = nil
        guard clShadowStrength > 0, ctx.device.supportsRaytracing, let st = RtShadows.shared.takeStructure(),
              let inputs = ColoredLight.shared.shadowInputs, ensure(), let pipe else { return }
        let sc = rtShadowScale
        let hw = (width + sc - 1) / sc, hh = (height + sc - 1) / sc
        var reset = false
        if history.count != 2 || history[0].width != hw || history[0].height != hh {
            history = (0..<2).compactMap { _ in
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: hw, height: hh, mipmapped: false)
                d.usage = [.shaderRead, .shaderWrite]
                d.storageMode = .private
                let t = ctx.device.makeTexture(descriptor: d)
                t?.label = "MetalMC colored light shadows"
                return t
            }
            guard history.count == 2 else { history = []; return }
            reset = true
        }
        // A new history after a jump (16 blocks in a frame, or the instance structure's origin moved) or a gap.
        let now = DispatchTime.now().uptimeNanoseconds
        let delta = st.camOffset - prevCamOffset
        if !(simd_length(delta) < 16) || now &- lastNanos > 100_000_000 || !historyOn { reset = true }
        lastNanos = now
        let out = history[1 - current]
        frame += 1
        let f = inputs.frame
        let lum = clBucketColors.map { simd_dot($0, SIMD3<Float>(0.2126, 0.7152, 0.0722)) }
        var p = ClShadowParamsGPU()
        p.invViewProj = invViewProj
        p.sizes = SIMD4(Float(width), Float(height), Float(hw), Float(hh))
        p.camTex = f.camTex
        p.camVol = f.camVol
        p.camOffset = SIMD4(st.camOffset, 0)
        let at = Self.pattern[Int(frame % 16)]
        p.sample = SIMD4(UInt32(sc), at.x % UInt32(sc), at.y % UInt32(sc), UInt32(truncatingIfNeeded: frame))
        p.weight0 = SIMD4(lum[0], lum[1], lum[2], lum[3])
        p.weight1 = SIMD4(lum[4], lum[5], lum[6], lum[7])
        p.curve0 = f.curve0; p.curve1 = f.curve1; p.curve2 = f.curve2; p.curve3 = f.curve3
        p.prevViewProj = prevViewProj
        p.camDelta = reset ? SIMD4(0, 0, 0, 1) : SIMD4(delta, 0)
        guard let enc = cb.makeComputeCommandEncoder(descriptor: profComputePass("colored light shadows")) else { return }
        enc.label = "MetalMC colored light shadows"
        enc.setComputePipelineState(pipe)
        enc.setAccelerationStructure(st.tlas, bufferIndex: 0)
        enc.useResources(st.accels, usage: .read)
        enc.setBytes(&p, length: MemoryLayout<ClShadowParamsGPU>.stride, index: 1)
        enc.setBuffer(inputs.light, offset: 0, index: 2)
        enc.setTexture(depth, index: 0)
        enc.setTexture(out, index: 1)
        enc.setTexture(history[current], index: 2)
        enc.dispatchThreads(MTLSize(width: hw, height: hh, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        traced = (out, sc)
        current = 1 - current
        prevViewProj = invViewProj.inverse
        prevCamOffset = st.camOffset
        traces += 1
        if traces == 1 { log("coloredlight: shadows from block lights, strength \(clShadowStrength), one ray per \(sc) x \(sc) pixels, with a history of their own") }
    }
}
