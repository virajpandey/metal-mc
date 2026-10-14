import Foundation
import Metal
import simd

// The traced sun visibility's history (flicker fix, 2026-10-13; lit mode, RtShadows.swift). The shadow rays go one per
// 4 x 4 pixels, from a different pixel of each block every frame (a 16-frame pattern), and the anti-aliasing was meant
// to average them. It can't: the relight read one texel for the whole block, so where a shadow's edge crosses a block
// (every edge of every object, leaves) all 16 pixels flipped between lit and shadowed together from frame to frame,
// and the resolve's neighborhood clip, seeing the whole neighborhood flip, let it through. The lab's flicker meter
// (Flicker.swift) showed it as most of the forest's flicker.
//
// So the visibility keeps a history at its own resolution: each texel's sample is reprojected through its surface point
// to the last frame's texels (the four around it, those whose sample was at about the same distance: no history across
// a disocclusion) and blended in at an eighth (more as the camera moves), so a texel converges to its block's mean over
// the pattern: the shadow filtered over its footprint, which is what the 16 frames were for. Out: the visibility's r, g and b as
// they came (r the sun's or the moon's visibility, g the light through leaves with leaflight, b one minus the clouds'
// transmittance with clouds), filtered, and in a the sample's distance (blocks; -1 the sky) for the relight's depth-aware
// upsample (litVisUpsample in Lit.swift) and the next frame's check.

private let visFilterSource = """
#include <metal_stdlib>
using namespace metal;

struct VisFilterParams {
    float4x4 invViewProj;   // clip to camera-relative (this frame's, jittered)
    float4x4 prevViewProj;  // the last frame's camera-relative to clip
    float4 camDelta;        // xyz: this frame's camera minus the last's, w: 1 if the history holds the last frame
    uint4 size;             // x, y: the frame's size; z, w: the visibility's
    uint4 sample;           // x: pixels per texel, y, z: this frame's pixel of each block
    float4 blend;           // x: this frame's weight with the camera still
};

kernel void vis_filter(texture2d<float, access::read> raw [[texture(0)]], depth2d<float, access::read> depth [[texture(1)]],
                       texture2d<float, access::read> prev [[texture(2)]], texture2d<float, access::write> out [[texture(3)]],
                       constant VisFilterParams& p [[buffer(0)]], uint2 t [[thread_position_in_grid]]) {
    if (t.x >= p.size.z || t.y >= p.size.w) return;
    uint sc = p.sample.x;
    uint2 q = min(t * sc + p.sample.yz, p.size.xy - 1u);
    float4 v = raw.read(t);
    float d = depth.read(q);
    if (d <= 0.0) {   // the sky: nothing to keep
        out.write(float4(v.rgb, -1.0), t);
        return;
    }
    float2 ndc = (float2(q) + 0.5) / float2(p.size.xy) * 2.0 - 1.0;
    float4 h = p.invViewProj * float4(ndc, d, 1.0);
    float3 rel = h.xyz / h.w;
    float dist = length(rel);
    float3 o = v.rgb;
    if (p.camDelta.w > 0.5) {
        float3 prel = rel + p.camDelta.xyz;
        float4 pc = p.prevViewProj * float4(prel, 1.0);
        if (pc.w > 1e-4) {
            float2 pq = (pc.xy / pc.w * 0.5 + 0.5) * float2(p.size.xy);
            float pdist = length(prel);
            float2 tc = pq / float(sc) - 0.5;
            int2 t0 = int2(floor(tc));
            float2 fr = tc - float2(t0);
            float3 acc = 0.0;
            float wsum = 0.0;
            for (int k = 0; k < 4; k++) {
                int2 tk = t0 + int2(k & 1, k >> 1);
                if (any(tk < int2(0)) || tk.x >= int(p.size.z) || tk.y >= int(p.size.w)) continue;
                float4 hk = prev.read(uint2(tk));
                // The same surface: its sample at about this distance (the block's pixels differ by up to its width).
                if (hk.a < 0.0 || abs(hk.a - pdist) > 0.04 * pdist + 0.25) continue;
                float wk = ((k & 1) != 0 ? fr.x : 1.0 - fr.x) * ((k >> 1) != 0 ? fr.y : 1.0 - fr.y) + 1e-3;
                acc += hk.rgb * wk;
                wsum += wk;
            }
            if (wsum > 0.05) {
                float motion = length(pq - (float2(q) + 0.5));
                o = mix(acc / wsum, v.rgb, mix(p.blend.x, 0.5, saturate(motion / 12.0)));
            }
        }
    }
    out.write(float4(o, dist), t);
}
"""

final class VisFilter: @unchecked Sendable {
    static let shared = VisFilter()
    private var pipe: MTLComputePipelineState?
    private var failed = false
    private var a: MTLTexture?
    private var b: MTLTexture?
    private var prevViewProj: simd_float4x4?
    private var prevCam = SIMD3<Double>.zero
    private var prevMoon = false
    private var lastFrame = -2

    /// METALMC_VISBLEND: the current frame's weight with the camera still (1: no history, the raw rays as before).
    private let blend: Float = Float(ProcessInfo.processInfo.environment["METALMC_VISBLEND"] ?? "") ?? 0.125

    private struct Params {
        var invViewProj: simd_float4x4
        var prevViewProj: simd_float4x4
        var camDelta: SIMD4<Float>
        var size: SIMD4<UInt32>
        var sample: SIMD4<UInt32>
        var blend: SIMD4<Float>
    }

    private func ensure() -> MTLComputePipelineState? {
        if let pipe { return pipe }
        if failed { return nil }
        do {
            let lib = try ShaderLab.library("vis_filter", visFilterSource) { [self] _ in pipe = nil; failed = false }
            guard let f = lib.makeFunction(name: "vis_filter") else { throw NSError(domain: "metalmc", code: 1) }
            pipe = try ctx.device.makeComputePipelineState(function: f)
        } catch {
            log("vis filter: the kernel failed: \(error)")
            failed = true
        }
        return pipe
    }

    /// This frame's visibility (`raw`, one sample per `scale` x `scale` pixels, at pixel `sample` of each block) blended
    /// into its history; returns the filtered texture (rgba16Float: the raw one's r, g, b, and the distance), or nil
    /// to use the raw one. `frame`: RtShadows' frame counter (a gap means the history is stale).
    func filter(cb: MTLCommandBuffer, raw: MTLTexture, depth: MTLTexture, invViewProj: simd_float4x4, cam: SIMD3<Double>,
                moon: Bool, scale: Int, sample: SIMD2<UInt32>, frame: Int) -> MTLTexture? {
        guard blend < 1, let pipe = ensure() else { return nil }
        let w = raw.width, h = raw.height
        if a == nil || a!.width != w || a!.height != h {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            a = ctx.device.makeTexture(descriptor: d)
            b = ctx.device.makeTexture(descriptor: d)
            a?.label = "MetalMC sun visibility (filtered)"
            b?.label = "MetalMC sun visibility (filtered)"
            prevViewProj = nil
        }
        guard let out = a, let prev = b, let enc = cb.makeComputeCommandEncoder() else { return nil }
        let valid = prevViewProj != nil && frame == lastFrame + 1 && moon == prevMoon
        var p = Params(invViewProj: invViewProj, prevViewProj: prevViewProj ?? matrix_identity_float4x4,
                       camDelta: SIMD4(Float(cam.x - prevCam.x), Float(cam.y - prevCam.y), Float(cam.z - prevCam.z), valid ? 1 : 0),
                       size: SIMD4(UInt32(depth.width), UInt32(depth.height), UInt32(w), UInt32(h)),
                       sample: SIMD4(UInt32(scale), sample.x, sample.y, 0), blend: SIMD4(blend, 0, 0, 0))
        enc.label = "MetalMC sun visibility filter"
        enc.setComputePipelineState(pipe)
        enc.setTexture(raw, index: 0)
        enc.setTexture(depth, index: 1)
        enc.setTexture(prev, index: 2)
        enc.setTexture(out, index: 3)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 0)
        enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        a = prev
        b = out
        prevViewProj = invViewProj.inverse
        prevCam = cam
        prevMoon = moon
        lastFrame = frame
        return out
    }
}
