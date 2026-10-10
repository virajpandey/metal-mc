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
// 16-bit float one costs 0.3 ms more per frame at the panel's resolution. (With HDR's float frame, METALMC_EXP=hdr, it
// is RGBA16Float, since the frame holds values above 1.) Each threadgroup loads its 32 x 32 pixels and a 1-pixel border
// once into threadgroup memory (taa_resolve). A draw then writes the new history back into the frame (the frame
// isn't writable from shaders, and the resolve reads each pixel's neighbors), sharpened a little: averaging jittered
// frames is a box filter over each pixel, which softens texture detail.
//
// Minecraft's shaders are translated with a vertical flip (flip_vert_y), so texture row 0 is the bottom of
// vanilla's clip space: normalized device y = 2 * v - 1 for texture coordinate v, in both frames.
//
// An earlier version used MetalFX's temporal scaler at 1:1. It cost 9 ms per frame at the panel's resolution.

// Lit hook (Lit.swift): the relight as the resolve loads each pixel; with water (METALMC_EXP=lit,water) its reflections
// after it (this frame's waves' tile and sky map are textures 13 and 14, which Lit.bindDeferred binds; without our sky
// litWaterPixel leaves water alone).
private let taaRelightLoad = "        if (taaLit) c.rgb = litRelightPixel(c.rgb, q, d, litGbuf.read(q).rg, litVis, litLm, litFrame, litEnv\(litGi ? ", litGiIrr, litGiCode" : "")\(clEnabled ? ", litClRGB, litClAux, litClFrame, litClSh" : ""));"
private let taaWaterLoad = """
        if (taaLit) {
            uint2 g = litGbuf.read(q).rg;
            c.rgb = litRelightPixel(c.rgb, q, d, g, litVis, litLm, litFrame, litEnv\(litGi ? ", litGiIrr, litGiCode" : "")\(clEnabled ? ", litClRGB, litClAux, litClFrame, litClSh" : ""));
            c.rgb = litWaterWet(c.rgb, q, d, g, litFrame, litWaterSky);
            c.rgb = litWaterPixel(c.rgb, q, d, g, litVis, litFrame, litEnv, litWaves, litWavesDetail, litWaterSky, litLm, litWaterTrace, litWaterTraceAS, color, litGbuf, depth\(litGi ? ", litGiCode" : ""));
            c.rgb = litWaterFog(c.rgb, q, d, litFrame, litEnv, litLm);
        }
"""

// Sky hook (Sky.swift): the atmosphere's shared functions, for the aerial perspective the resolve can apply as it loads.
// Lit hook (Lit.swift, METALMC_EXP=lit only): the relight's, likewise.
private let taaShaderSource = skyShaderHeader + (litEnabled ? "\n#define LIT_MODE 1\n" + litShaderHeader + (litGi ? giUpsampleHeader : "") + litRelightHeader : "") + """

// With the sky on (METALMC_EXP=sky), a second variant of the resolve applies the aerial perspective, the render
// distance's fade into the sky and the tone curve to each pixel as it loads it (skyLevelColor), after the shadows'
// shade, so that costs no full-screen pass of its own. The default variant has none of it.
constant bool taaSky [[function_constant(0)]];
#if LIT_MODE
// Lit mode: variants that relight the terrain (litRelightPixel) as each pixel is loaded, before the sky's aerial perspective.
constant bool taaLit [[function_constant(1)]];
#endif

struct TaaParams {
    float4x4 invCur;     // inverse of this frame's projection * view rotation (unjittered)
    float4x4 prev;       // the previous frame's projection * view rotation (unjittered)
    float4 camDelta;     // xyz: this frame's camera position minus the previous one's
    float2 size;         // texture size in pixels
    float reset;         // 1: the history is unusable (first frame, a jump, a gap)
    float blend;         // weight of the current frame when the view is still
    float4 shadow;       // ray-traced shadows (RtShadows) folded in: x strength (0: none), y-z fade distance range
    float4 range;        // x: the largest value the frame holds (1; more for a float frame with HDR, Hdr.swift)
};

// Ray-traced shadows (RtShadows) store, per p.shadow.y x p.shadow.y pixels, the factor to multiply the color by (strength
// and distance fade included); p.shadow.x is 0 when there are none this frame.
static half shadowShade(texture2d<half, access::read> lit, uint2 q, constant TaaParams& p) {
    if (p.shadow.x <= 0.0) return 1.0h;
    return lit.read(min(q / uint(p.shadow.y), uint2(lit.get_width() - 1, lit.get_height() - 1))).r;
}

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

// taa_resolve's tile: TAA_T x TAA_T pixels per threadgroup (16 or 32, a preprocessor macro of the library), TAA_A with
// its 1-pixel border. The variants whose load is heavy (the sky's aerial perspective, lit mode's relight) take 32: their
// border's share of the loads is 13% against 27%. The plain resolve takes 16: for it the bigger tile's threadgroup memory
// (14 KB against 4) costs more than the border saves (0.85 against 0.74 ms at the panel's resolution, measured).
#ifndef TAA_T
#define TAA_T 32
#endif
#define TAA_A (TAA_T + 2)

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
                        texture2d<half, access::read> lit [[texture(4)]],
                        constant TaaParams& p [[buffer(0)]],
                        constant SkyFrame& sky [[buffer(1), function_constant(taaSky)]],
                        texture3d<float> apScatter [[texture(5), function_constant(taaSky)]],
                        texture3d<float> apTrans [[texture(6), function_constant(taaSky)]],
                        texture2d<float> skyView [[texture(7), function_constant(taaSky)]],
#if LIT_MODE
                        constant LitFrame& litFrame [[buffer(2), function_constant(taaLit)]],
                        constant float4* litEnv [[buffer(3), function_constant(taaLit)]],
                        texture2d<uint, access::read> litGbuf [[texture(8), function_constant(taaLit)]],
                        texture2d<half, access::read> litVis [[texture(9), function_constant(taaLit)]],
                        texture2d<float> litLm [[texture(10), function_constant(taaLit)]],\(litGi ? "\n                        texture2d<float> litGiIrr [[texture(11), function_constant(taaLit)]],\n                        texture2d<uint> litGiCode [[texture(12), function_constant(taaLit)]]," : "")\(litWater ? "\n                        texture2d<float> litWaves [[texture(13), function_constant(taaLit)]],\n                        texture2d<float> litWaterSky [[texture(14), function_constant(taaLit)]],\n                        texture2d<float> litWaterTrace [[texture(18), function_constant(taaLit)]],\n                        texture2d<float> litWavesDetail [[texture(19), function_constant(taaLit)]],\n                        texture2d<uint> litWaterTraceAS [[texture(20), function_constant(taaLit)]]," : "")\(clEnabled ? "\n                        texture3d<float> litClRGB [[texture(15), function_constant(taaLit)]],\n                        texture3d<float> litClAux [[texture(16), function_constant(taaLit)]],\n                        constant ClFrame& litClFrame [[buffer(4), function_constant(taaLit)]],\n                        texture2d<half> litClSh [[texture(17), function_constant(taaLit)]]," : "")
#endif
                        uint2 lid [[thread_position_in_threadgroup]],
                        uint2 tgid [[threadgroup_position_in_grid]]) {
    // The threadgroup's 32 x 32 pixels and a 1-pixel border, loaded once: color in YCoCg (alpha in w) and depth. The
    // border is loaded by both threadgroups beside it, and each load is the heavy part (in lit mode the relight, the
    // shadows' shade, the aerial perspective): 13% more loads than pixels here, against 27% for a 16 x 16 tile (which
    // also left most of its threads idle for the last few). 16 x 16 threads, each resolving 4 pixels.
    threadgroup half4 tile[TAA_A * TAA_A];
    threadgroup float dtile[TAA_A * TAA_A];
    int2 size = int2(p.size);
    int2 base = int2(tgid) * TAA_T - 1;
    for (uint i = lid.y * 16 + lid.x; i < TAA_A * TAA_A; i += 256) {
        uint2 q = uint2(clamp(base + int2(i % TAA_A, i / TAA_A), int2(0), size - 1));
        float4 c = color.read(q);
        float d = depth.read(q);
#if LIT_MODE
\(litWater ? taaWaterLoad : taaRelightLoad)
#endif
        c.rgb *= float(shadowShade(lit, q, p));
        if (taaSky) {
            float haze;
            c.rgb = skyLevelColor(c.rgb, q, d, sky, apScatter, apTrans, skyView, haze);
        }
        tile[i] = half4(half3(toYCoCg(c.rgb)), half(c.a));
        dtile[i] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The tile's 16 x 16 pieces in turn (one, or four for a 32 x 32 tile), so a SIMD group's pixels stay side by side.
    for (uint piece = 0; piece < (TAA_T / 16) * (TAA_T / 16); piece++) {
        uint2 lp = lid + uint2((piece % (TAA_T / 16)) * 16u, (piece / (TAA_T / 16)) * 16u);
        uint2 gid = tgid * TAA_T + lp;
        if (int(gid.x) >= size.x || int(gid.y) >= size.y) continue;
        uint c0 = (lp.y + 1) * TAA_A + lp.x + 1;
        half4 center = tile[c0];
        half3 cur = center.xyz;
        half3 lo = cur, hi = cur, m1 = cur, m2 = cur * cur;
        float nearest = dtile[c0];   // reverse-Z: larger is nearer, 0 is the sky
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                if (dx == 0 && dy == 0) continue;
                uint k = uint(int(c0) + dy * TAA_A + dx);
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
        nextHistory.write(float4(clamp(fromYCoCg(float3(result)), 0.0, p.range.x), 1.0), gid);
    }
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
// its 4 neighbors, less where they already differ a lot, so edges don't ring. sc: x sharpening, y the frame's largest
// value (above 1 only for HDR's float frame, whose highlights past 2 CAS leaves unsharpened), z dither, w the frame.
// The dither (the sky's, Sky.swift, on an 8-bit frame, and on HDR's packed RG11B10Float frame, whose 6-bit mantissas
// band too; 0 on RGBA16Float) keeps the sky's smooth gradients from banding as the history is written back: the sky's
// own dither is averaged away in the history.
static float3 taaDither(float3 c, int2 g, float4 sc) {
    if (sc.z == 0.0) return c;
    float2 p = float2(g) + 5.588238 * sc.w;
    float n = fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715)))) - 0.5;
    // sc.z < 0: the packed RG11B10Float frame, whose steps are relative to the value (see skyDither).
    if (sc.z < 0.0) return c * (1.0 + n * float3(1.0 / 64.0, 1.0 / 64.0, 1.0 / 32.0));
    return c + n * sc.z;
}

fragment float4 taa_copy_fs(CopyVOut in [[stage_in]], texture2d<float, access::read> h [[texture(0)]],
                            constant float4& sc [[buffer(0)]]) {
    int2 size = int2(h.get_width(), h.get_height());
    int2 g = int2(in.pos.xy);
    float3 c = h.read(uint2(g)).rgb;
    float sharpen = sc.x;
    if (sharpen <= 0.0) return float4(taaDither(c, g, sc), 1.0);
    float3 n = h.read(uint2(clamp(g + int2(0, -1), int2(0), size - 1))).rgb;
    float3 s = h.read(uint2(clamp(g + int2(0, 1), int2(0), size - 1))).rgb;
    float3 w = h.read(uint2(clamp(g + int2(-1, 0), int2(0), size - 1))).rgb;
    float3 e = h.read(uint2(clamp(g + int2(1, 0), int2(0), size - 1))).rgb;
    float3 mn = min(c, min(min(n, s), min(w, e))), mx = max(c, max(max(n, s), max(w, e)));
    float3 amp = sqrt(saturate(min(mn, 2.0 - mx) / max(mx, 1e-4)));
    float3 wt = -amp * (0.2 * sharpen);
    return float4(taaDither(clamp((c + (n + s + w + e) * wt) / (1.0 + 4.0 * wt), 0.0, sc.y), g, sc), 1.0);
}
"""

private struct TaaParams {
    var invCur: simd_float4x4
    var prev: simd_float4x4
    var camDelta: SIMD4<Float>
    var size: SIMD2<Float>
    var reset: Float
    var blend: Float
    var shadow: SIMD4<Float> = .zero
    var range = SIMD4<Float>(1, 0, 0, 0)
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
    var skyPipe: MTLComputePipelineState?   // the resolve with the sky's aerial perspective (Sky.swift), built on first use
    var skyFailed = false
    var frames = 0                   // frames anti-aliased, for the sky's dither
    var copyPipes: [UInt: MTLRenderPipelineState] = [:]   // by the frame's pixel format
    var foldedCopyPipes: [String: MTLRenderPipelineState] = [:]   // the copy inside another pass (pass folding), by formats

    /// The copy for a pass with these depth and stencil attachment formats (.invalid for none): the copy's own pass has
    /// none, the hand's pass it's folded into (pass folding, Backend.swift) has the frame's depth.
    func copyPipe(color: MTLPixelFormat, depth: MTLPixelFormat, stencil: MTLPixelFormat) -> MTLRenderPipelineState? {
        if depth == .invalid && stencil == .invalid { return copyPipes[color.rawValue] }
        let k = "\(color.rawValue)/\(depth.rawValue)/\(stencil.rawValue)"
        if let p = foldedCopyPipes[k] { return p }
        guard let lib = library else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.label = "MetalMC TAA copy (folded)"
        d.vertexFunction = lib.makeFunction(name: "taa_copy_vs")
        d.fragmentFunction = lib.makeFunction(name: "taa_copy_fs")
        d.colorAttachments[0].pixelFormat = color
        d.depthAttachmentPixelFormat = depth
        d.stencilAttachmentPixelFormat = stencil
        guard let p = try? ctx.device.makeRenderPipelineState(descriptor: d) else { return nil }
        foldedCopyPipes[k] = p
        return p
    }
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
                // Lab mode (ShaderLab.swift): after an edit, forget both libraries and their pipelines; the next frame
                // builds them again (and restarts the history).
                let lib = try ShaderLab.library("taa", taaShaderSource) { [self] _ in
                    library = nil; pipe = nil; skyPipe = nil; litPipes = [:]; copyPipes = [:]; foldedCopyPipes = [:]
                    skyFailed = false; litFailed = false; failed = false; self.key = ""
                }
                // The plain resolve from a library with 16 x 16 tiles (TAA_T); the variants with a heavy load use 32.
                let small = MTLCompileOptions()
                small.preprocessorMacros = ["TAA_T": NSNumber(value: 16)]
                let plainLib = try ShaderLab.library("taa", taaShaderSource, options: small)
                pipe = try ctx.device.makeComputePipelineState(function: resolveFunction(plainLib, sky: false))
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
        // HDR hook (Hdr.swift): a float frame holds values above 1 (the sky's highlights), which RGB10A2 would clip.
        let historyFormat: MTLPixelFormat = isFloatFrame(color.pixelFormat) ? color.pixelFormat : .rgb10a2Unorm
        guard let h0 = texture(historyFormat, [.shaderRead, .shaderWrite]),
              let h1 = texture(historyFormat, [.shaderRead, .shaderWrite]) else { return false }
        history = [h0, h1]
        valid = false
        self.key = key
        log("TAA: resolve at \(key)")
        return true
    }

    /// The resolve, with or without the sky's aerial perspective in its load (the taaSky function constant), and in lit
    /// mode with or without the relight (taaLit).
    func resolveFunction(_ lib: MTLLibrary, sky: Bool, lit: Bool = false) throws -> MTLFunction {
        let values = MTLFunctionConstantValues()
        var on = sky
        values.setConstantValue(&on, type: .bool, index: 0)
        if litEnabled {
            var relight = lit
            values.setConstantValue(&relight, type: .bool, index: 1)
        }
        return try lib.makeFunction(name: "taa_resolve", constantValues: values)
    }

    /// Lit hook (Lit.swift): the resolve that relights the terrain as it loads each pixel, with or without the sky's aerial
    /// perspective after it (nil if it failed to build: the plain resolve runs, without the relight).
    private var litPipes: [Bool: MTLComputePipelineState] = [:]
    private var litFailed = false
    func litResolve(sky: Bool) -> MTLComputePipelineState? {
        if let p = litPipes[sky] { return p }
        guard let library, !litFailed else { return nil }
        do {
            litPipes[sky] = try ctx.device.makeComputePipelineState(function: resolveFunction(library, sky: sky, lit: true))
        } catch {
            log("TAA: lit resolve failed: \(error)")
            litFailed = true
        }
        return litPipes[sky]
    }

    /// Sky hook (Sky.swift): the resolve variant that applies the aerial perspective as it loads (nil if it failed to
    /// build: the plain resolve runs, without the sky's haze).
    func skyResolve() -> MTLComputePipelineState? {
        if let skyPipe { return skyPipe }
        guard let library, !skyFailed else { return nil }
        do {
            skyPipe = try ctx.device.makeComputePipelineState(function: resolveFunction(library, sky: true))
        } catch {
            log("TAA: sky resolve failed: \(error)")
            skyFailed = true
        }
        return skyPipe
    }
}

/// Anti-aliases the level just drawn into `color` (a TextureBox handle) using `depth`. p: this frame's unjittered
/// projection[16] and view rotation[16] (column-major); cam: camera position (world, doubles); jitter: the sub-pixel
/// offset the frame was drawn with, in pixels (unused: the history is the average of the jittered frames). Returns 1
/// if it ran.
/// Debug: the resolve's shader source (with METALMC_EXP=lit,gi in the environment, its lit variants' too), for an offline
/// compile check (tools/shadercheck.py). Returns its length.
@_cdecl("mmc_debug_taa_shader_source")
public func mmc_debug_taa_shader_source(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let bytes = Array(taaShaderSource.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}

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
    guard let enc = cb.makeComputeCommandEncoder(descriptor: profComputePass("anti-aliasing resolve (+ aerial perspective, relight)")) else { return 0 }
    var params = TaaParams(invCur: viewProj.inverse, prev: t.prevViewProj,
                           camDelta: restart ? .zero : SIMD4(Float(delta.x), Float(delta.y), Float(delta.z), 0),
                           size: SIMD2(Float(color.width), Float(color.height)), reset: restart ? 1 : 0, blend: taaBlend)
    // Ray-traced shadows traced this frame are applied here, as the color is loaded (RtShadows defers to the TAA).
    let shadows = RtShadows.shared.takeDeferred(width: color.width, height: color.height)
    if let sh = shadows { params.shadow = sh.params }
    // HDR hook (Hdr.swift): a float frame's values aren't clamped to 1.
    let frameMax: Float = isFloatFrame(color.pixelFormat) ? 65504 : 1
    params.range.x = frameMax
    // Lit hook (Lit.swift): the relight left for the resolve, applied as the color is loaded, before the sky's work.
    let relight = litEnabled && Lit.shared.hasDeferred(width: color.width, height: color.height)
    // Sky hook (Sky.swift): the level through the air (aerial perspective, the render distance's fade, the tone curve),
    // applied as the color is loaded, after the shadows' shade; on an 8-bit frame, dithered as it's written back.
    var skyDither: Float = 0
    var tile = 16   // the plain resolve's tile; the variants with a heavy load use 32 (TAA_T in taa_resolve)
    if let sky = Sky.shared.takeDeferredAerial(width: color.width, height: color.height),
       let skyPipe = relight ? t.litResolve(sky: true) : t.skyResolve() {
        var frame = sky.frame
        enc.setComputePipelineState(skyPipe)
        enc.setBytes(&frame, length: MemoryLayout<SkyFrameGPU>.stride, index: 1)
        enc.setTexture(sky.apScatter, index: 5)
        enc.setTexture(sky.apTrans, index: 6)
        enc.setTexture(sky.skyView, index: 7)
        skyDither = color.pixelFormat == .rgba16Float ? 0 : (color.pixelFormat == .rg11b10Float ? -1 : 1.0 / 255)
        tile = 32
    } else if relight, let litPipe = t.litResolve(sky: false) {
        enc.setComputePipelineState(litPipe)
        tile = 32
    } else {
        enc.setComputePipelineState(pipe)
    }
    if relight { Lit.shared.bindDeferred(enc) }
    enc.setTexture(color, index: 0)
    enc.setTexture(depth, index: 1)
    enc.setTexture(t.history[t.current], index: 2)
    enc.setTexture(t.history[1 - t.current], index: 3)
    enc.setTexture(shadows?.lit ?? RtShadows.shared.dummyLit(), index: 4)
    enc.setBytes(&params, length: MemoryLayout<TaaParams>.stride, index: 0)
    // Whole threadgroups of 16 x 16 threads, each over a tile of 16 x 16 pixels (the plain resolve) or 32 x 32 (TAA_T):
    // every thread helps load the tile, including those past the edge.
    enc.dispatchThreadgroups(MTLSize(width: (color.width + tile - 1) / tile, height: (color.height + tile - 1) / tile, depth: 1),
                             threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    enc.endEncoding()
    // The new history into the frame: a draw that waits to be the first of the next pass on the frame (the hand's) with
    // pass folding, else a pass of its own now (Backend.swift: runPending encodes it).
    let newHistory = t.history[1 - t.current]
    let sharpen = SIMD4<Float>(taaSharpen, frameMax, skyDither, Float(t.frames % 64))
    let colorFormat = color.pixelFormat
    ctx.pendingCopy = PendingCopy(texture: color) { enc, depthFormat, stencilFormat in
        guard let p = t.copyPipe(color: colorFormat, depth: depthFormat, stencil: stencilFormat) ?? (depthFormat == .invalid ? copyPipe : nil) else {
            log("TAA: no copy pipeline for depth \(depthFormat.rawValue), stencil \(stencilFormat.rawValue)")
            return
        }
        enc.setRenderPipelineState(p)
        enc.setFragmentTexture(newHistory, index: 0)
        var sc = sharpen
        enc.setFragmentBytes(&sc, length: 16, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
    }
    if !passFolding { ctx.runPending(cb) }
    t.current = 1 - t.current
    t.frames += 1
    t.valid = true
    t.prevViewProj = viewProj
    t.prevCam = camera
    return 1
}
