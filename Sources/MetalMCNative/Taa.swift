import Foundation
import Metal
import simd

// Temporal anti-aliasing (config taa, off by default). Each frame the level is drawn with a sub-pixel jitter of its
// projection (GameRendererLodMixin, an 8-step Halton sequence). After the level and before the hand and the HUD, one
// compute pass blends the frame into an accumulated history: the history is reprojected per pixel from depth and the
// camera, clipped to the current 3 x 3 neighborhood's color range (in YCoCg) so moving things and newly revealed
// surfaces don't smear, and mixed with the current pixel (10% current when still, more when moving).
//
// Reprojection assumes a static world, since only the camera's motion is known. Entities that move rely on the clip.
// The depth used is the nearest in the 3 x 3 neighborhood, so the edges of near objects move with the object rather
// than with the background behind them.
//
// The history is RGB10A2: at a 10% blend an 8-bit history stops converging up to 5 levels from its target, and a
// 16-bit float one costs 0.3 ms more per frame at the panel's resolution. Each 16 x 16 threadgroup loads its pixels
// and a 1-pixel border once into threadgroup memory. A draw then writes the new history back into the frame (the frame
// isn't writable from shaders, and the resolve reads each pixel's neighbors), sharpened a little: averaging jittered
// frames is a box filter over each pixel, which softens texture detail.
//
// Minecraft's shaders are translated with a vertical flip (flip_vert_y), so texture row 0 is the bottom of
// vanilla's clip space: normalized device y = 2 * v - 1 for texture coordinate v, in both frames.
//
// An earlier version used MetalFX's temporal scaler at 1:1. It cost 9 ms per frame at the panel's resolution.

private let taaShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct TaaParams {
    float4x4 invCur;     // inverse of this frame's projection * view rotation (unjittered)
    float4x4 prev;       // the previous frame's projection * view rotation (unjittered)
    float4 camDelta;     // xyz: this frame's camera position minus the previous one's
    float2 size;         // texture size in pixels
    float reset;         // 1: the history is unusable (first frame, a jump, a gap)
    float blend;         // weight of the current frame when the view is still
};

static float3 toYCoCg(float3 c) {
    return float3(0.25 * c.r + 0.5 * c.g + 0.25 * c.b, 0.5 * c.r - 0.5 * c.b, -0.25 * c.r + 0.5 * c.g - 0.25 * c.b);
}

static float3 fromYCoCg(float3 c) {
    return float3(c.x + c.y - c.z, c.x + c.z, c.x - c.y - c.z);
}

// Catmull-Rom filtered history in 5 bilinear taps (the 4 corner taps have tiny weights and are dropped). Sharper
// than one bilinear tap, so the image doesn't soften while the camera moves.
static float3 historySample(texture2d<float, access::sample> h, float2 uv, float2 size) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 pos = uv * size;
    float2 t1 = floor(pos - 0.5) + 0.5;
    float2 f = pos - t1;
    float2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    float2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    float2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    float2 w3 = f * f * (-0.5 + 0.5 * f);
    float2 w12 = w1 + w2;
    float2 t12 = (t1 + w2 / w12) / size;
    float2 t0 = (t1 - 1.0) / size;
    float2 t3 = (t1 + 2.0) / size;
    float3 r = h.sample(s, float2(t12.x, t0.y)).rgb * (w12.x * w0.y)
             + h.sample(s, float2(t0.x, t12.y)).rgb * (w0.x * w12.y)
             + h.sample(s, t12).rgb * (w12.x * w12.y)
             + h.sample(s, float2(t3.x, t12.y)).rgb * (w3.x * w12.y)
             + h.sample(s, float2(t12.x, t3.y)).rgb * (w12.x * w3.y);
    float w = w12.x * w0.y + w0.x * w12.y + w12.x * w12.y + w3.x * w12.y + w12.x * w3.y;
    return max(r / w, 0.0);
}

// Moves h toward the box's center until it's inside the box.
static float3 clipToBox(float3 lo, float3 hi, float3 h) {
    float3 c = 0.5 * (hi + lo), e = 0.5 * (hi - lo) + 1e-4;
    float3 v = h - c;
    float3 a = abs(v / e);
    float m = max(a.x, max(a.y, a.z));
    return m > 1.0 ? c + v / m : h;
}

kernel void taa_resolve(texture2d<float, access::read> color [[texture(0)]],
                        depth2d<float, access::read> depth [[texture(1)]],
                        texture2d<float, access::sample> history [[texture(2)]],
                        texture2d<float, access::write> nextHistory [[texture(3)]],
                        constant TaaParams& p [[buffer(0)]],
                        uint2 gid [[thread_position_in_grid]],
                        uint2 lid [[thread_position_in_threadgroup]],
                        uint2 tgid [[threadgroup_position_in_grid]]) {
    // The threadgroup's 16 x 16 pixels and a 1-pixel border, loaded once: color in YCoCg (alpha in w) and depth.
    threadgroup half4 tile[18 * 18];
    threadgroup float dtile[18 * 18];
    int2 size = int2(p.size);
    int2 base = int2(tgid) * 16 - 1;
    for (uint i = lid.y * 16 + lid.x; i < 18 * 18; i += 256) {
        uint2 q = uint2(clamp(base + int2(i % 18, i / 18), int2(0), size - 1));
        float4 c = color.read(q);
        tile[i] = half4(half3(toYCoCg(c.rgb)), half(c.a));
        dtile[i] = depth.read(q);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(gid.x) >= size.x || int(gid.y) >= size.y) return;
    uint c0 = (lid.y + 1) * 18 + lid.x + 1;
    half4 center = tile[c0];
    half3 cur = center.xyz;
    half3 lo = cur, hi = cur, m1 = cur, m2 = cur * cur;
    float nearest = dtile[c0];   // reverse-Z: larger is nearer, 0 is the sky
    for (int dy = -1; dy <= 1; dy++) {
        for (int dx = -1; dx <= 1; dx++) {
            if (dx == 0 && dy == 0) continue;
            uint k = uint(int(c0) + dy * 18 + dx);
            half3 c = tile[k].xyz;
            lo = min(lo, c);
            hi = max(hi, c);
            m1 += c;
            m2 += c * c;
            nearest = max(nearest, dtile[k]);
        }
    }
    half3 result = cur;
    if (p.reset == 0.0) {
        float2 uv = (float2(gid) + 0.5) / p.size;
        float2 ndc = uv * 2.0 - 1.0;
        float4 prevClip;
        if (nearest <= 0.0) {
            // Sky: only the camera's rotation moves it.
            float4 dir = p.invCur * float4(ndc, 0.0, 1.0);
            prevClip = p.prev * float4(dir.xyz, 0.0);
        } else {
            float4 rel = p.invCur * float4(ndc, nearest, 1.0);
            rel.xyz /= rel.w;
            prevClip = p.prev * float4(rel.xyz + p.camDelta.xyz, 1.0);
        }
        float2 prevUV = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
        if (prevClip.w > 0.0 && all(prevUV > 0.0) && all(prevUV < 1.0)) {
            float2 motion = (prevUV - uv) * p.size;
            // Still: the history pixel itself (Catmull-Rom at a pixel center is that pixel).
            float3 hs = dot(motion, motion) < 1e-6 ? history.read(gid).rgb : historySample(history, prevUV, p.size);
            half3 h = half3(toYCoCg(hs));
            half3 mu = m1 / 9.0h, sigma = sqrt(max(m2 / 9.0h - mu * mu, 0.0h));
            h = half3(clipToBox(float3(max(lo, mu - 1.25h * sigma)), float3(min(hi, mu + 1.25h * sigma)), float3(h)));
            result = mix(h, cur, half(mix(p.blend, 0.3, saturate(length(motion) / 16.0))));
        }
    }
    nextHistory.write(float4(saturate(fromYCoCg(float3(result))), 1.0), gid);
}

struct CopyVOut {
    float4 pos [[position]];
};

vertex CopyVOut taa_copy_vs(uint vid [[vertex_id]]) {
    CopyVOut o;
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// The new history into the frame, with contrast-adaptive sharpening (after AMD's CAS): each pixel is pushed away from
// its 4 neighbors, less where they already differ a lot, so edges don't ring.
fragment float4 taa_copy_fs(CopyVOut in [[stage_in]], texture2d<float, access::read> h [[texture(0)]],
                            constant float& sharpen [[buffer(0)]]) {
    int2 size = int2(h.get_width(), h.get_height());
    int2 g = int2(in.pos.xy);
    float3 c = h.read(uint2(g)).rgb;
    if (sharpen <= 0.0) return float4(c, 1.0);
    float3 n = h.read(uint2(clamp(g + int2(0, -1), int2(0), size - 1))).rgb;
    float3 s = h.read(uint2(clamp(g + int2(0, 1), int2(0), size - 1))).rgb;
    float3 w = h.read(uint2(clamp(g + int2(-1, 0), int2(0), size - 1))).rgb;
    float3 e = h.read(uint2(clamp(g + int2(1, 0), int2(0), size - 1))).rgb;
    float3 mn = min(c, min(min(n, s), min(w, e))), mx = max(c, max(max(n, s), max(w, e)));
    float3 amp = sqrt(saturate(min(mn, 2.0 - mx) / max(mx, 1e-4)));
    float3 wt = -amp * (0.2 * sharpen);
    return float4(saturate((c + (n + s + w + e) * wt) / (1.0 + 4.0 * wt)), 1.0);
}
"""

private struct TaaParams {
    var invCur: simd_float4x4
    var prev: simd_float4x4
    var camDelta: SIMD4<Float>
    var size: SIMD2<Float>
    var reset: Float
    var blend: Float
}

/// METALMC_TAABLEND: the current frame's weight when the view is still (default 0.1).
private let taaBlend: Float = Float(ProcessInfo.processInfo.environment["METALMC_TAABLEND"] ?? "") ?? 0.1
/// METALMC_TAASHARPEN: sharpening of the result, 0 (none) to 1 (default 0.5).
private let taaSharpen: Float = Float(ProcessInfo.processInfo.environment["METALMC_TAASHARPEN"] ?? "") ?? 0.5

final class Taa: @unchecked Sendable {
    static let shared = Taa()

    var history: [MTLTexture] = []   // two, ping-ponged: read one, write the other
    var current = 0
    var key = ""
    var library: MTLLibrary?
    var pipe: MTLComputePipelineState?
    var copyPipes: [UInt: MTLRenderPipelineState] = [:]   // by the frame's pixel format
    var valid = false                // the current history holds a frame
    var prevViewProj = matrix_identity_float4x4
    var prevCam = SIMD3<Double>(repeating: .nan)
    var lastApply: UInt64 = 0        // uptime of the last frame anti-aliased; after a gap (menus, loading) the history restarts
    var failed = false

    func ensure(color: MTLTexture) -> Bool {
        if failed { return false }
        let key = "\(color.width)x\(color.height)/\(color.pixelFormat.rawValue)"
        if key == self.key, !history.isEmpty { return true }
        do {
            if library == nil {
                let lib = try ctx.device.makeLibrary(source: taaShaderSource, options: nil)
                pipe = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "taa_resolve")!)
                library = lib
            }
            if copyPipes[color.pixelFormat.rawValue] == nil, let lib = library {
                let d = MTLRenderPipelineDescriptor()
                d.label = "MetalMC TAA copy"
                d.vertexFunction = lib.makeFunction(name: "taa_copy_vs")
                d.fragmentFunction = lib.makeFunction(name: "taa_copy_fs")
                d.colorAttachments[0].pixelFormat = color.pixelFormat
                copyPipes[color.pixelFormat.rawValue] = try ctx.device.makeRenderPipelineState(descriptor: d)
            }
        } catch {
            log("TAA: shaders failed: \(error)")
            failed = true
            return false
        }
        func texture(_ format: MTLPixelFormat, _ usage: MTLTextureUsage) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: color.width, height: color.height, mipmapped: false)
            d.usage = usage
            d.storageMode = .private
            return ctx.device.makeTexture(descriptor: d)
        }
        guard let h0 = texture(.rgb10a2Unorm, [.shaderRead, .shaderWrite]),
              let h1 = texture(.rgb10a2Unorm, [.shaderRead, .shaderWrite]) else { return false }
        history = [h0, h1]
        valid = false
        self.key = key
        log("TAA: resolve at \(key)")
        return true
    }
}

/// Anti-aliases the level just drawn into `color` (a TextureBox handle) using `depth`. p: this frame's unjittered
/// projection[16] and view rotation[16] (column-major); cam: camera position (world, doubles); jitter: the sub-pixel
/// offset the frame was drawn with, in pixels (unused: the history is the average of the jittered frames). Returns 1
/// if it ran.
@_cdecl("mmc_taa_apply")
public func mmc_taa_apply(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>, _ cam: UnsafePointer<Double>,
                          _ jitterX: Float, _ jitterY: Float, _ reset: Int32) -> Int32 {
    let t = Taa.shared
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    guard ctx.pass == nil, t.ensure(color: color), let pipe = t.pipe, let copyPipe = t.copyPipes[color.pixelFormat.rawValue] else { return 0 }
    func mat(_ o: Int) -> simd_float4x4 {
        simd_float4x4(SIMD4(p[o], p[o + 1], p[o + 2], p[o + 3]), SIMD4(p[o + 4], p[o + 5], p[o + 6], p[o + 7]),
                      SIMD4(p[o + 8], p[o + 9], p[o + 10], p[o + 11]), SIMD4(p[o + 12], p[o + 13], p[o + 14], p[o + 15]))
    }
    let viewProj = mat(0) * mat(16)
    let camera = SIMD3(cam[0], cam[1], cam[2])
    let delta = camera - t.prevCam
    // A new history after a jump (teleport, respawn, dimension change), a gap of frames without anti-aliasing
    // (a menu, the loading screen) or on request.
    let now = DispatchTime.now().uptimeNanoseconds
    let restart = !t.valid || !(simd_length(delta) < 16) || now - t.lastApply > 100_000_000 || reset != 0
    t.lastApply = now
    ctx.endBlit()
    let cb = ctx.ensureCB()
    guard let enc = cb.makeComputeCommandEncoder() else { return 0 }
    var params = TaaParams(invCur: viewProj.inverse, prev: t.prevViewProj,
                           camDelta: restart ? .zero : SIMD4(Float(delta.x), Float(delta.y), Float(delta.z), 0),
                           size: SIMD2(Float(color.width), Float(color.height)), reset: restart ? 1 : 0, blend: taaBlend)
    enc.setComputePipelineState(pipe)
    enc.setTexture(color, index: 0)
    enc.setTexture(depth, index: 1)
    enc.setTexture(t.history[t.current], index: 2)
    enc.setTexture(t.history[1 - t.current], index: 3)
    enc.setBytes(&params, length: MemoryLayout<TaaParams>.stride, index: 0)
    // Whole threadgroups: every thread helps load the tile, including those past the edge.
    enc.dispatchThreadgroups(MTLSize(width: (color.width + 15) / 16, height: (color.height + 15) / 16, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    enc.endEncoding()
    let d = MTLRenderPassDescriptor()
    d.colorAttachments[0].texture = color
    d.colorAttachments[0].loadAction = .dontCare
    d.colorAttachments[0].storeAction = .store
    guard let copy = cb.makeRenderCommandEncoder(descriptor: d) else { return 0 }
    copy.setRenderPipelineState(copyPipe)
    copy.setFragmentTexture(t.history[1 - t.current], index: 0)
    var sharpen = taaSharpen
    copy.setFragmentBytes(&sharpen, length: 4, index: 0)
    copy.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    copy.endEncoding()
    t.current = 1 - t.current
    t.valid = true
    t.prevViewProj = viewProj
    t.prevCam = camera
    return 1
}
