import Foundation
import Metal
import simd

// Temporal upscaling (TAAU): the level drawn at a fraction of the panel's resolution on each axis (RenderScale.swift:
// METALMC_SCALE, or its governor) and resolved by the anti-aliasing into its history at the panel's resolution. The
// hand, the HUD and post-processing stay at the panel's resolution (the Java side draws the level into a smaller target
// of its own, metalmc.render.RenderScale, and the rest into vanilla's main target).
//
// The resolve is the anti-aliasing's (Taa.swift: reprojected history, Catmull-Rom history sample, variance clip in YCoCg),
// with two changes for samples that no longer sit on the output's pixel grid:
// - Each output pixel takes this frame's samples near it: the 3 x 3 rendered texels around the nearest one, each weighted
//   by exp(-k d^2) for its sample's distance d from the output pixel's center in output pixels (k: METALMC_TAAU_K, 3 by
//   default, a Gaussian of 0.41 pixels). A sample's position is its texel's center minus the frame's jitter.
// - The history's share follows how much of this frame landed near the pixel: the current frame's weight is
//   W / (W + (1 / b - 1) E[W]), W the sum of the weights, E[W] its mean over all jitters (the sample density times the
//   Gaussian's integral) and b the anti-aliasing's blend (0.1 still, more in motion). An output pixel with a sample at its
//   center takes more than b of it; one with none near takes little. On average it's b, as without upscaling.
// The jitter covers 16 positions (a Halton 2,3 sequence, TemporalAA.java) instead of 8 when upscaling: each rendered texel
// spreads over more than one output pixel.
//
// Each threadgroup loads a 32 x 32 tile of the rendered frame and its 1-texel border (the load's heavy part, the relight,
// the water and the aerial perspective, runs once per rendered texel, 13% border), then resolves every output pixel whose
// nearest sample this frame is one of the tile's texels: the tiles partition the output exactly, whatever the scale.

/// METALMC_TAAU_K: the current frame's reconstruction: each sample's weight is exp(-k d^2), d its distance from the output
/// pixel's center in output pixels (default 3: a Gaussian of 0.41 pixels; larger is sharper and slower to converge).
let taauSharpness: Float = Float(ProcessInfo.processInfo.environment["METALMC_TAAU_K"] ?? "") ?? 3.0
/// METALMC_TAAU_SHARPEN: the copy's sharpening when upscaling, 0-1 (default: METALMC_TAASHARPEN's, 0.5).
let taauSharpen: Float = Float(ProcessInfo.processInfo.environment["METALMC_TAAU_SHARPEN"] ?? "")
    ?? Float(ProcessInfo.processInfo.environment["METALMC_TAASHARPEN"] ?? "") ?? 0.5
/// METALMC_TAABLEND, as the anti-aliasing reads it (the current frame's weight when still, 0.1).
private let taauBlend: Float = Float(ProcessInfo.processInfo.environment["METALMC_TAABLEND"] ?? "") ?? 0.1
/// Debug: -1 flips the jitter's sign (a wrong sign shows as a softer, swimming image; METALMC_TAAU_JSIGN).
private let taauJitterSign: Float = Float(ProcessInfo.processInfo.environment["METALMC_TAAU_JSIGN"] ?? "") ?? 1

private let taauKernelSource = """

// ---------------------------------------------------------------------------------------- temporal upscaling (TAAU)
// TaaUpscale.swift: the level rendered at a fraction of the output's resolution, resolved into the output-sized history.

struct TaauParams {
    float4 map;      // xy: the frame's jitter in rendered pixels (texel c's sample sits at c + 0.5 - jitter, unjittered);
                     // zw: rendered size / output size
    float4 inSize;   // xy: the rendered frame's size in pixels
    float4 recon;    // x: k (a sample's weight is exp(-k d^2), d in output pixels); y: the mean sum of the 3 x 3 weights
};

#define TAAU_T 32
#define TAAU_A (TAAU_T + 2)

kernel void taa_upscale(texture2d<float, access::read> color [[texture(0)]],
                        depth2d<float, access::read> depth [[texture(1)]],
                        texture2d<float, access::sample> history [[texture(2)]],
                        texture2d<float, access::write> nextHistory [[texture(3)]],
                        texture2d<half, access::read> lit [[texture(4)]],
                        constant TaaParams& p [[buffer(0)]],
                        constant TaauParams& u [[buffer(10)]],
                        constant SkyFrame& sky [[buffer(1), function_constant(taaSky)]],
                        texture3d<float> apScatter [[texture(5), function_constant(taaSky)]],
                        texture3d<float> apTrans [[texture(6), function_constant(taaSky)]],
                        texture2d<float> skyView [[texture(7), function_constant(taaSky)]],\(cloudsEnabled ? "\n                        constant CloudFrame& clouds [[buffer(5), function_constant(taaSky)]],\n                        texture2d<float> cloudTex [[texture(21), function_constant(taaSky)]],\n                        texture2d<float> cloudDepth [[texture(22), function_constant(taaSky)]]," : "")
#if LIT_MODE
                        constant LitFrame& litFrame [[buffer(2), function_constant(taaLit)]],
                        constant float4* litEnv [[buffer(3), function_constant(taaLit)]],
                        texture2d<uint, access::read> litGbuf [[texture(8), function_constant(taaLit)]],
                        texture2d<half, access::read> litVis [[texture(9), function_constant(taaLit)]],
                        texture2d<float> litLm [[texture(10), function_constant(taaLit)]],\(litGi ? "\n                        texture2d<float> litGiIrr [[texture(11), function_constant(taaLit)]],\n                        texture2d<uint> litGiCode [[texture(12), function_constant(taaLit)]]," : "")\(litWater ? "\n                        texture2d<float> litWaves [[texture(13), function_constant(taaLit)]],\n                        texture2d<float> litWaterSky [[texture(14), function_constant(taaLit)]],\n                        texture2d<float> litWaterTrace [[texture(18), function_constant(taaLit)]],\n                        texture2d<float> litWavesDetail [[texture(19), function_constant(taaLit)]],\n                        texture2d<uint> litWaterTraceAS [[texture(20), function_constant(taaLit)]]," : "")\(clEnabled ? "\n                        texture3d<float> litClRGB [[texture(15), function_constant(taaLit)]],\n                        texture3d<float> litClAux [[texture(16), function_constant(taaLit)]],\n                        constant ClFrame& litClFrame [[buffer(4), function_constant(taaLit)]],\n                        texture2d<half> litClSh [[texture(17), function_constant(taaLit)]]," : "")
#endif
                        uint2 lid [[thread_position_in_threadgroup]],
                        uint2 tgid [[threadgroup_position_in_grid]]) {
    // The tile: the rendered frame's 32 x 32 texels and a 1-texel border, loaded once (as taa_resolve loads its pixels:
    // the relight, the water, the shadows' shade, the aerial perspective), in YCoCg with depth.
    threadgroup half4 tile[TAAU_A * TAAU_A];
    threadgroup float dtile[TAAU_A * TAAU_A];
    int2 size = int2(u.inSize.xy);
    int2 t0 = int2(tgid) * TAAU_T;
    int2 base = t0 - 1;
    for (uint i = lid.y * 16 + lid.x; i < TAAU_A * TAAU_A; i += 256) {
        uint2 q = uint2(clamp(base + int2(i % TAAU_A, i / TAAU_A), int2(0), size - 1));
        float4 c = color.read(q);
        float d = depth.read(q);
#if LIT_MODE
\(litWater ? taaWaterLoad : taaRelightLoad)
#endif
        c.rgb *= float(shadowShade(lit, q, p));
        if (taaSky) {
            float haze;
            c.rgb = skyLevelColor(c.rgb, q, d, sky, apScatter, apTrans, skyView, haze);\(cloudsEnabled ? "\n            c.rgb = cloudsComposite(c.rgb, q, d, sky, clouds, cloudTex, cloudDepth);" : "")
        }
        tile[i] = half4(half3(toYCoCg(c.rgb)), half(c.a));
        dtile[i] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // The output pixels whose nearest sample this frame is one of the tile's texels. Output pixel o's center is at
    // P = (o + 0.5) r in rendered pixels (r: rendered / output size), and its nearest sample is texel floor(P + jitter),
    // clamped to the frame. The range holds them all and one more on each side, which the test below sorts out.
    float2 j = u.map.xy, r = u.map.zw;
    float2 toOut = 1.0 / r;
    int2 osize = int2(p.size);
    int2 o0 = int2(ceil((float2(t0) - j) * toOut - 0.5)) - 1;
    int2 o1 = int2(floor((float2(t0 + TAAU_T) - j) * toOut - 0.5)) + 1;
    o0 = select(max(o0, int2(0)), int2(0), t0 == int2(0));
    o1 = select(min(o1, osize - 1), osize - 1, t0 + TAAU_T >= size);
    int2 span = max(o1 - o0 + 1, int2(0));
    int n = span.x * span.y;
    for (int k = int(lid.y * 16 + lid.x); k < n; k += 256) {
        int2 o = o0 + int2(k % span.x, k / span.x);
        float2 P = (float2(o) + 0.5) * r;
        int2 c = clamp(int2(floor(P + j)), int2(0), size - 1);
        if (any(c < t0) || any(c >= t0 + TAAU_T)) continue;
        int c0 = (c.y - base.y) * TAAU_A + (c.x - base.x);
        // The 3 x 3 texels around the nearest sample: the clip box's statistics, the nearest depth (for the reprojection,
        // reverse-Z: larger is nearer), and this frame's reconstruction at the output pixel.
        float2 ds = (float2(c) + 0.5 - j - P) * toOut;
        half3 lo = half3(HALF_MAX), hi = half3(-HALF_MAX), m1 = half3(0.0h), m2 = half3(0.0h);
        float3 sum = float3(0.0);
        float wsum = 0.0, nearest = 0.0;
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int k2 = c0 + dy * TAAU_A + dx;
                half3 s = tile[k2].xyz;
                lo = min(lo, s);
                hi = max(hi, s);
                m1 += s;
                m2 += s * s;
                nearest = max(nearest, dtile[k2]);
                float2 dd = ds + float2(float(dx), float(dy)) * toOut;
                float w = exp(-u.recon.x * dot(dd, dd));
                sum += float3(s) * w;
                wsum += w;
            }
        }
        half3 cur = half3(sum / max(wsum, 1e-6));
        half3 result = cur;
        if (p.reset == 0.0) {
            float2 uv = (float2(o) + 0.5) / p.size;
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
                float3 hs = dot(motion, motion) < 1e-6 ? history.read(uint2(o)).rgb : historySample(history, prevUV, p.size);
                half3 h = half3(toYCoCg(hs));
                half3 mu = m1 / 9.0h, sigma = sqrt(max(m2 / 9.0h - mu * mu, 0.0h));
                h = half3(clipToBox(float3(max(lo, mu - 1.25h * sigma)), float3(min(hi, mu + 1.25h * sigma)), float3(h)));
                float b = mix(p.blend, 0.3, saturate(length(motion) / 16.0));
                float a = wsum / (wsum + (1.0 / b - 1.0) * u.recon.y);
                result = mix(h, cur, half(a));
            }
        }
        nextHistory.write(float4(clamp(fromYCoCg(float3(result)), 0.0, p.range.x), 1.0), uint2(o));
    }
}
"""

/// Taa.swift's TaaParams, field for field.
private struct TaauResolveParams {
    var invCur: simd_float4x4
    var prev: simd_float4x4
    var camDelta: SIMD4<Float>
    var size: SIMD2<Float>
    var reset: Float
    var blend: Float
    var shadow: SIMD4<Float> = .zero
    var range = SIMD4<Float>(1, 0, 0, 0)
}

private struct TaauParams {
    var map: SIMD4<Float>
    var inSize: SIMD4<Float>
    var recon: SIMD4<Float>
}

final class TaaUpscale: @unchecked Sendable {
    static let shared = TaaUpscale()

    private var library: MTLLibrary?
    private var pipes: [Int: MTLComputePipelineState] = [:]   // by variant: bit 0 the sky's aerial perspective, bit 1 the relight
    private var failed = false
    private(set) var frames = 0
    private var lastKey = ""
    /// The reconstruction's sharpness and the copy's sharpening: METALMC_TAAU_K and METALMC_TAAU_SHARPEN, or in lab mode
    /// <shader dir>/taau.txt ("K SHARPEN", read every 240 frames) to tune them in one session.
    private(set) var k = taauSharpness, sharpen = taauSharpen
    private var labText: String?

    private func labOverride() {
        guard let dir = ShaderLab.dir, frames % 240 == 0 else { return }
        let text = (try? String(contentsOfFile: dir + "/taau.txt", encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text != labText else { return }
        labText = text
        let v = (text ?? "").split(whereSeparator: { $0 == " " || $0 == "," }).compactMap { Float($0) }
        k = v.count > 0 && v[0] > 0 ? v[0] : taauSharpness
        sharpen = v.count > 1 && v[1] >= 0 ? v[1] : taauSharpen
        log(String(format: "TAAU: lab override %@: reconstruction k %.2f, sharpen %.2f", text ?? "removed", k, sharpen))
    }

    /// The upscaling resolve with or without the sky's aerial perspective and lit mode's relight in its load (nil if it
    /// failed to build).
    func pipe(sky: Bool, lit: Bool) -> MTLComputePipelineState? {
        let key = (sky ? 1 : 0) | (lit ? 2 : 0)
        if let p = pipes[key] { return p }
        if failed { return nil }
        do {
            if library == nil {
                // Lab mode (ShaderLab.swift): after an edit of taau.metal (or a header it includes), build it all again.
                library = try ShaderLab.library("taau", taaShaderSource + taauKernelSource) { [self] _ in
                    library = nil; pipes = [:]; failed = false
                }
            }
            let values = MTLFunctionConstantValues()
            var s = sky
            values.setConstantValue(&s, type: .bool, index: 0)
            if litEnabled {
                var l = lit
                values.setConstantValue(&l, type: .bool, index: 1)
            }
            let f = try library!.makeFunction(name: "taa_upscale", constantValues: values)
            pipes[key] = try ctx.device.makeComputePipelineState(function: f)
        } catch {
            log("TAAU: shaders failed: \(error)")
            failed = true
            return nil
        }
        return pipes[key]
    }

    /// The mean of the 3 x 3 reconstruction weights over all jitters: the samples per output pixel (r.x r.y) times the
    /// Gaussian's integral (pi / k).
    static func expectedWeight(rx: Float, ry: Float, k: Float) -> Float { rx * ry * Float.pi / k }

    /// Resolves the level drawn at a fraction of the output's size (`color`, `depth`) into the anti-aliasing's history at
    /// `out`'s size, and leaves the history's copy into `out` waiting for the next pass on it (pass folding), as the
    /// anti-aliasing does. p, cam, reset: as mmc_taa_apply's; jitter: the frame's, in rendered pixels.
    func apply(color: MTLTexture, depth: MTLTexture, out: MTLTexture, p: UnsafePointer<Float>, cam: UnsafePointer<Double>,
               jitter: SIMD2<Float>, reset: Bool) -> Bool {
        let t = Taa.shared
        guard ctx.pass == nil, t.ensure(color: out), let copyPipe = t.copyPipes[out.pixelFormat.rawValue] else { return false }
        labOverride()
        func mat(_ o: Int) -> simd_float4x4 {
            simd_float4x4(SIMD4(p[o], p[o + 1], p[o + 2], p[o + 3]), SIMD4(p[o + 4], p[o + 5], p[o + 6], p[o + 7]),
                          SIMD4(p[o + 8], p[o + 9], p[o + 10], p[o + 11]), SIMD4(p[o + 12], p[o + 13], p[o + 14], p[o + 15]))
        }
        let viewProj = mat(0) * mat(16)
        let camera = SIMD3(cam[0], cam[1], cam[2])
        let delta = camera - t.prevCam
        // A new history after a jump, a gap of frames without anti-aliasing, or on request (as mmc_taa_apply).
        let now = DispatchTime.now().uptimeNanoseconds
        let restart = !t.valid || !(simd_length(delta) < 16) || now - t.lastApply > 100_000_000 || reset
        // The relight, the shadows and the aerial perspective left for the resolve, at the rendered size.
        let relight = litEnabled && Lit.shared.hasDeferred(width: color.width, height: color.height)
        let sky = Sky.shared.takeDeferredAerial(width: color.width, height: color.height)
        guard let pipe = pipe(sky: sky != nil, lit: relight) else { return false }
        t.lastApply = now
        ctx.endBlit()
        let cb = ctx.ensureCB()
        let label = "anti-aliasing upscale \(color.width)x\(color.height) to \(out.width)x\(out.height) (+ aerial perspective, relight)"
        guard let enc = cb.makeComputeCommandEncoder(descriptor: profComputePass(label)) else { return false }
        var params = TaauResolveParams(invCur: viewProj.inverse, prev: t.prevViewProj,
                                       camDelta: restart ? .zero : SIMD4(Float(delta.x), Float(delta.y), Float(delta.z), 0),
                                       size: SIMD2(Float(out.width), Float(out.height)), reset: restart ? 1 : 0, blend: taauBlend)
        let shadows = RtShadows.shared.takeDeferred(width: color.width, height: color.height)
        if let sh = shadows { params.shadow = sh.params }
        let frameMax: Float = isFloatFrame(out.pixelFormat) ? 65504 : 1
        params.range.x = frameMax
        var skyDither: Float = 0
        enc.setComputePipelineState(pipe)
        if let sky {
            var frame = sky.frame
            enc.setBytes(&frame, length: MemoryLayout<SkyFrameGPU>.stride, index: 1)
            enc.setTexture(sky.apScatter, index: 5)
            enc.setTexture(sky.apTrans, index: 6)
            enc.setTexture(sky.skyView, index: 7)
            // Clouds hook (Clouds.swift): their composite in the load, at the level's size (their march follows it).
            if cloudsEnabled { Clouds.shared.bindTaa(enc, width: color.width, height: color.height) }
            skyDither = out.pixelFormat == .rgba16Float ? 0 : (out.pixelFormat == .rg11b10Float ? -1 : 1.0 / 255)
        }
        if relight { Lit.shared.bindDeferred(enc) }
        enc.setTexture(color, index: 0)
        enc.setTexture(depth, index: 1)
        enc.setTexture(t.history[t.current], index: 2)
        enc.setTexture(t.history[1 - t.current], index: 3)
        enc.setTexture(shadows?.lit ?? RtShadows.shared.dummyLit(), index: 4)
        enc.setBytes(&params, length: MemoryLayout<TaauResolveParams>.stride, index: 0)
        let rx = Float(color.width) / Float(out.width), ry = Float(color.height) / Float(out.height)
        var up = TaauParams(map: SIMD4(jitter.x * taauJitterSign, jitter.y * taauJitterSign, rx, ry),
                            inSize: SIMD4(Float(color.width), Float(color.height), 0, 0),
                            recon: SIMD4(k, TaaUpscale.expectedWeight(rx: rx, ry: ry, k: k), 0, 0))
        enc.setBytes(&up, length: MemoryLayout<TaauParams>.stride, index: 10)
        // Whole threadgroups of 16 x 16 threads over the rendered frame's 32 x 32 tiles (each resolves the output pixels
        // whose nearest sample is in its tile).
        enc.dispatchThreadgroups(MTLSize(width: (color.width + 31) / 32, height: (color.height + 31) / 32, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        // The new history into the frame, as the anti-aliasing's copy (sharpened, dithered), folded into the next pass on it.
        let newHistory = t.history[1 - t.current]
        let sharpen = SIMD4<Float>(self.sharpen, frameMax, skyDither, Float(t.frames % 64))
        let colorFormat = out.pixelFormat
        ctx.pendingCopy = PendingCopy(texture: out) { enc, depthFormat, stencilFormat in
            guard let p = t.copyPipe(color: colorFormat, depth: depthFormat, stencil: stencilFormat) ?? (depthFormat == .invalid ? copyPipe : nil) else {
                log("TAAU: no copy pipeline for depth \(depthFormat.rawValue), stencil \(stencilFormat.rawValue)")
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
        frames += 1
        let key = "\(color.width)x\(color.height)>\(out.width)x\(out.height)"
        if key != lastKey {
            lastKey = key
            log(String(format: "TAAU: %@ (scale %.3f x %.3f), reconstruction k %.2f, mean weight %.3f, sharpen %.2f",
                       key, rx, ry, k, up.recon.y, self.sharpen))
        }
        return true
    }
}

/// The anti-aliasing for a level drawn into `color` and `depth` at a fraction of `out`'s size (the panel's): resolves it
/// into the history at `out`'s size and copies that into `out`. The same as mmc_taa_apply when the sizes match and color
/// is out. p (32 floats), cam, reset as mmc_taa_apply's; jitterX/Y: the frame's jitter in rendered pixels. Returns 1 if it
/// ran (0: nothing was written to `out`).
// MARK: - Offline test (tools/taautest.swift)

/// The test image: a zone plate (frequency rising from the corner, half a cycle a pixel of the output at half its width)
/// and a dark bar with a hard, slightly tilted edge, at (x, y) in output pixels of a w-wide output.
private func taauTestImage(_ x: Double, _ y: Double, _ w: Double) -> Double {
    let zone = 0.5 + 0.5 * cos(Double.pi * (x * x + y * y) / w)
    let bar = abs((x - 0.62 * w) * 0.995 + (y - 0.3 * w) * 0.1) < 0.04 * w ? 0.1 : 1.0
    return zone * bar
}

private func taauHalton(_ index: Int, _ base: Int) -> Float {
    var f: Float = 1, r: Float = 0, i = index
    while i > 0 { f /= Float(base); r += f * Float(i % base); i /= base }
    return r
}

/// Offline test of the upscaling resolve on a still view (no game, no window): `frames` frames of the test image drawn at
/// `scale` of a w x h output with the 16-step jitter (mode 0: TAAU, through TaaUpscale.apply and the copy, as the game runs
/// them), or at the output's size with the 8-step jitter through the plain anti-aliasing (mode 1, mmc_taa_apply), or
/// drawn once at `scale` without jitter and filtered up bilinearly (mode 2: no temporal upscaling), or mode 3: compile
/// every variant of the resolve and return. jsign -1 tells the resolve the jitter's sign is the other way (a wrong sign
/// should score worse). Writes the result's luminance (w x h floats) to `out` and returns its RMS error against the image
/// box-filtered over each output pixel (8 x 8 samples); errors in `stats`: [all, the band the drawn frame can hold (its
/// Nyquist), the band past it]. Negative on failure.
@_cdecl("mmc_debug_taau_test")
public func mmc_debug_taau_test(_ w: Int32, _ h: Int32, _ scale: Float, _ frames: Int32, _ jsign: Float, _ mode: Int32,
                                _ out: UnsafeMutablePointer<Float>, _ stats: UnsafeMutablePointer<Double>) -> Double {
    let W = Int(w), H = Int(h)
    if mode == 3 {
        var failed = 0
        for sky in [false, true] {
            for lit in [false, true] where TaaUpscale.shared.pipe(sky: sky, lit: lit) == nil { failed += 1 }
        }
        return Double(failed)
    }
    let inW = mode == 1 ? W : max(1, Int((Float(W) * scale).rounded())), inH = mode == 1 ? H : max(1, Int((Float(H) * scale).rounded()))
    let rx = Double(inW) / Double(W), ry = Double(inH) / Double(H)
    var result = [Float](repeating: 0, count: W * H)
    if mode == 2 {
        // Drawn once without jitter (texel centers), filtered up bilinearly at each output pixel's center.
        var img = [Double](repeating: 0, count: inW * inH)
        for cy in 0..<inH { for cx in 0..<inW { img[cy * inW + cx] = taauTestImage((Double(cx) + 0.5) / rx, (Double(cy) + 0.5) / ry, Double(W)) } }
        for oy in 0..<H {
            for ox in 0..<W {
                let px = (Double(ox) + 0.5) * rx - 0.5, py = (Double(oy) + 0.5) * ry - 0.5
                let x0 = Int(floor(px)), y0 = Int(floor(py)), fx = px - floor(px), fy = py - floor(py)
                func at(_ x: Int, _ y: Int) -> Double { img[min(max(y, 0), inH - 1) * inW + min(max(x, 0), inW - 1)] }
                let v = (at(x0, y0) * (1 - fx) + at(x0 + 1, y0) * fx) * (1 - fy) + (at(x0, y0 + 1) * (1 - fx) + at(x0 + 1, y0 + 1) * fx) * fy
                result[oy * W + ox] = Float(v)
            }
        }
    } else {
        let dev = ctx.device
        func tex(_ f: MTLPixelFormat, _ tw: Int, _ th: Int, _ usage: MTLTextureUsage, _ mode: MTLStorageMode) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: tw, height: th, mipmapped: false)
            d.usage = usage
            d.storageMode = mode
            return dev.makeTexture(descriptor: d)
        }
        guard let color = tex(.rgba8Unorm, inW, inH, [.shaderRead, .renderTarget], .shared),
              let depth = tex(.depth32Float, inW, inH, [.shaderRead, .renderTarget], .private),
              let outT = mode == 1 ? color : tex(.rgba8Unorm, W, H, [.shaderRead, .renderTarget], .private),
              let readback = dev.makeBuffer(length: W * H * 4, options: .storageModeShared) else { return -1 }
        // A plane at depth 0.5 under identity matrices: every pixel reprojects onto itself (a still view).
        if let cb = ctx.queue.makeCommandBuffer() {
            let d = MTLRenderPassDescriptor()
            d.depthAttachment.texture = depth
            d.depthAttachment.loadAction = .clear
            d.depthAttachment.clearDepth = 0.5
            d.depthAttachment.storeAction = .store
            cb.makeRenderCommandEncoder(descriptor: d)?.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }
        var p = [Float](repeating: 0, count: 32)
        for i in 0..<4 { p[i * 5] = 1; p[16 + i * 5] = 1 }
        let cam: [Double] = [0, 0, 0]
        let colorH = makeHandle(TextureBox(color)), depthH = makeHandle(TextureBox(depth))
        defer { mmc_handle_release(colorH); mmc_handle_release(depthH) }
        var pixels = [UInt8](repeating: 0, count: inW * inH * 4)
        let phases = mode == 1 ? 8 : 16
        for frame in 0..<Int(frames) {
            let i = frame % phases + 1
            let jx = taauHalton(i, 2) - 0.5, jy = taauHalton(i, 3) - 0.5
            // Texel c shows the image at c + 0.5 - jitter (rendered pixels), as a jittered projection draws it.
            for cy in 0..<inH {
                for cx in 0..<inW {
                    let v = taauTestImage((Double(cx) + 0.5 - Double(jx)) / rx, (Double(cy) + 0.5 - Double(jy)) / ry, Double(W))
                    let b = UInt8(max(0, min(255, (v * 255).rounded())))
                    let o = (cy * inW + cx) * 4
                    pixels[o] = b; pixels[o + 1] = b; pixels[o + 2] = b; pixels[o + 3] = 255
                }
            }
            color.replace(region: MTLRegionMake2D(0, 0, inW, inH), mipmapLevel: 0, withBytes: pixels, bytesPerRow: inW * 4)
            let ok: Bool
            if mode == 1 {
                ok = p.withUnsafeBufferPointer { pp in cam.withUnsafeBufferPointer { cc in
                    mmc_taa_apply(colorH, depthH, pp.baseAddress!, cc.baseAddress!, jx, jy, frame == 0 ? 1 : 0) == 1 } }
            } else {
                ok = p.withUnsafeBufferPointer { pp in cam.withUnsafeBufferPointer { cc in
                    TaaUpscale.shared.apply(color: color, depth: depth, out: outT, p: pp.baseAddress!, cam: cc.baseAddress!,
                                            jitter: SIMD2(jx * jsign, jy * jsign), reset: frame == 0) } }
            }
            guard ok else { return -2 }
            let cb = ctx.ensureCBNoFlush()
            ctx.runPending(cb)
            ctx.endBlit()
            if frame == Int(frames) - 1, let blit = cb.makeBlitCommandEncoder() {
                blit.copy(from: outT, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: W, height: H, depth: 1),
                          to: readback, destinationOffset: 0, destinationBytesPerRow: W * 4, destinationBytesPerImage: W * H * 4)
                blit.endEncoding()
            }
            cb.commit()
            cb.waitUntilCompleted()
            ctx.cb = nil
        }
        let bytes = readback.contents().bindMemory(to: UInt8.self, capacity: W * H * 4)
        for k in 0..<(W * H) { result[k] = Float(bytes[k * 4]) / 255 }
    }
    // The reference: the image box-filtered over each output pixel (8 x 8 samples); errors over all, under the drawn
    // frame's Nyquist (the zone plate's frequency radius / w under half the scale) and past it.
    var e = [0.0, 0.0, 0.0], n = [0, 0, 0]
    for oy in 0..<H {
        for ox in 0..<W {
            var ref = 0.0
            for sy in 0..<8 { for sx in 0..<8 { ref += taauTestImage(Double(ox) + (Double(sx) + 0.5) / 8, Double(oy) + (Double(sy) + 0.5) / 8, Double(W)) } }
            ref /= 64
            let v = Double(result[oy * W + ox])
            out[oy * W + ox] = Float(v)
            let d2 = (v - ref) * (v - ref)
            // The zone plate's local frequency is radius / w cycles a pixel; the drawn frame holds up to scale / 2.
            let band = (Double(ox * ox + oy * oy)).squareRoot() / Double(W) < min(rx, ry) / 2 ? 1 : 2
            e[0] += d2; n[0] += 1
            e[band] += d2; n[band] += 1
        }
    }
    for k in 0..<3 { stats[k] = n[k] > 0 ? (e[k] / Double(n[k])).squareRoot() : 0 }
    return stats[0]
}

@_cdecl("mmc_taa_apply_scaled")
public func mmc_taa_apply_scaled(_ colorHandle: Int64, _ depthHandle: Int64, _ outHandle: Int64, _ p: UnsafePointer<Float>,
                                 _ cam: UnsafePointer<Double>, _ jitterX: Float, _ jitterY: Float, _ reset: Int32) -> Int32 {
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    let out = (from(outHandle) as TextureBox).texture
    if (color.parent ?? color) === (out.parent ?? out) {
        return mmc_taa_apply(colorHandle, depthHandle, p, cam, jitterX, jitterY, reset)
    }
    return TaaUpscale.shared.apply(color: color, depth: depth, out: out, p: p, cam: cam, jitter: SIMD2(jitterX, jitterY),
                                   reset: reset != 0) ? 1 : 0
}
