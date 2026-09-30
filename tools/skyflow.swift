// Offline run of the sky and HDR frame as the game calls it (no game, no window): through libMetalMCNative's C entry
// points, with METALMC_EXP=sky,hdr, it encodes the sky pass, a vanilla-style pipeline (declaring an RGBA8 target)
// drawing "terrain" into the RGBA16Float main target, the aerial perspective (its own pass, and inside the
// anti-aliasing), an HDR screenshot copy, and a present into an offscreen EDR CAMetalLayer. It checks the results and
// writes the frame as a PNG. With "sdr" last it runs METALMC_EXP=sky alone: an RGBA8 main target, the SDR tone curve,
// an 8-bit layer.
// usage: swift build -c release; swiftc -O tools/skyflow.swift -o .build/skyflow
//        .build/skyflow .build/release/libMetalMCNative.dylib <output dir> [sdr]
import CoreGraphics
import Foundation
import ImageIO
import QuartzCore
import simd

let args = CommandLine.arguments
let sdr = args.count >= 4 && args[3] == "sdr"
setenv("METALMC_EXP", sdr ? "sky" : "sky,hdr", 1)
guard args.count >= 3, let lib = dlopen(args[1], RTLD_NOW) else { print("usage: skyflow <dylib> <output dir> [sdr]"); exit(1) }
let outDir = URL(fileURLWithPath: args[2])
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
func fn<T>(_ name: String, _: T.Type) -> T {
    guard let p = dlsym(lib, name) else { print("missing \(name)"); exit(1) }
    return unsafeBitCast(p, to: T.self)
}
let ctxInit = fn("mmc_ctx_init", (@convention(c) () -> Int32).self)
let textureCreate = fn("mmc_texture_create", (@convention(c) (Int32, Int32, Int32, Int32, Int32, Int32) -> Int64).self)
let bufferCreate = fn("mmc_buffer_create", (@convention(c) (Int64) -> Int64).self)
let bufferContents = fn("mmc_buffer_contents", (@convention(c) (Int64) -> Int64).self)
let pipelineCreate = fn("mmc_pipeline_create", (@convention(c) (UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>?,
                                                                 UnsafePointer<CChar>?, UnsafePointer<Int32>, Int32, UnsafeMutablePointer<CChar>, Int32) -> Int64).self)
let passBegin = fn("mmc_pass_begin", (@convention(c) (UnsafePointer<Int64>, Int32, Int32, UnsafePointer<Float>, Int64, Int32, Float,
                                                       Int32, Int32, Int32, Int32) -> Int32).self)
let passEnd = fn("mmc_pass_end", (@convention(c) () -> Void).self)
let setPipeline = fn("mmc_rp_set_pipeline", (@convention(c) (Int64) -> Int32).self)
let draw = fn("mmc_rp_draw", (@convention(c) (Int32, Int32, Int32, Int32) -> Void).self)
let copyToBuffer = fn("mmc_copy_texture_to_buffer", (@convention(c) (Int64, Int32, Int32, Int32, Int32, Int32, Int64, Int64, Int32) -> Void).self)
let submit = fn("mmc_submit", (@convention(c) (Int64) -> Void).self)
let waitSubmit = fn("mmc_wait_submit", (@convention(c) (Int64, Int64) -> Int32).self)
let skyEnabled = fn("mmc_sky_enabled", (@convention(c) () -> Int32).self)
let hdrEnabled = fn("mmc_hdr_enabled", (@convention(c) () -> Int32).self)
let skyPrepare = fn("mmc_sky_prepare", (@convention(c) (UnsafePointer<Float>) -> Int32).self)
let skyViewFn = fn("mmc_sky_view", (@convention(c) (Int64, UnsafePointer<Float>) -> Int32).self)
let skyDraw = fn("mmc_sky_draw", (@convention(c) (UnsafePointer<Float>) -> Int32).self)
let skyAerial = fn("mmc_sky_aerial", (@convention(c) (Int64, Int64, UnsafePointer<Float>, Int32) -> Int32).self)
let taaApply = fn("mmc_taa_apply", (@convention(c) (Int64, Int64, UnsafePointer<Float>, UnsafePointer<Double>, Float, Float, Int32) -> Int32).self)
let hdrSnapshot = fn("mmc_hdr_snapshot", (@convention(c) (Int64, Int64) -> Int32).self)
let surfaceCreate = fn("mmc_surface2_create", (@convention(c) (Int64) -> Int64).self)
let surfaceConfigure = fn("mmc_surface2_configure", (@convention(c) (Int64, Int32, Int32, Int32) -> Void).self)
let surfaceBlit = fn("mmc_surface2_blit", (@convention(c) (Int64, Int64) -> Void).self)
let surfacePresent = fn("mmc_surface2_present", (@convention(c) (Int64) -> Void).self)

// Submit indices start past 1: the backend counts submit 1 as already complete.
func check(_ ok: Bool, _ what: String) {
    print((ok ? "ok    " : "FAIL  ") + what)
    if !ok { exit(1) }
}
check(ctxInit() == 1, "context")
check(skyEnabled() == 1 && hdrEnabled() == (sdr ? 0 : 1), "METALMC_EXP=\(sdr ? "sky" : "sky,hdr") seen by the library")

let W: Int32 = 1728, H: Int32 = 1117   // half the panel, to keep the readback small
let color = textureCreate(sdr ? 70 : 115, W, H, 1, 1, 1)   // RGBA8, or RGBA16Float: what MainTargetMixin makes it with HDR
let depth = textureCreate(252, W, H, 1, 1, 1)   // Depth32Float
let snap = textureCreate(70, W, H, 1, 1, 1)     // RGBA8, the screenshot copy
check(color != 0 && depth != 0 && snap != 0, "textures")

// A pipeline like vanilla's terrain (translated MSL, entry main0, an RGBA8 color target declared): a quad over the
// bottom of the screen, below the horizon, from 20 blocks away at the bottom edge to 20 km at the top of the quad,
// in a mid gray-green.
let vs = """
#include <metal_stdlib>
using namespace metal;
struct O { float4 p [[position]]; };
vertex O main0(uint vid [[vertex_id]]) {
    float2 c[6] = { float2(-1, -1), float2(1, -1), float2(1, -0.02), float2(-1, -1), float2(1, -0.02), float2(-1, -0.02) };
    float2 q = c[vid];
    float dist = mix(20.0, 20000.0, pow((q.y + 1.0) / 0.98, 6.0));
    // Vanilla's shaders are translated with SPIRV-Cross's flip_vert_y: texture row 0 is the GL bottom row.
    O o; o.p = float4(q.x, -q.y, 0.05 / dist, 1.0); return o;
}
"""
let fs = """
#include <metal_stdlib>
using namespace metal;
fragment float4 main0() { return float4(0.40, 0.52, 0.30, 1.0); }
"""
var params: [Int32] = [3, 0, 0, 1, 7, 1, 0, 0, 0, 0, 0, 1, 70, 15, 0, 0, 0, 1, 0, 1, 0, 0, 0]
var err = [CChar](repeating: 0, count: 512)
let pipe = pipelineCreate("terrain", vs, "main0", fs, "main0", &params, Int32(params.count), &err, 512)
check(pipe != 0, "vanilla-style RGBA8 pipeline created (with its RGBA16Float variant warmed): \(String(cString: err))")

// The camera: 150 blocks up (87 above sea level), looking west toward the sun 10 degrees up, 70-degree vertical view.
let sunElevation: Float = 10 * .pi / 180
let sunAngle = .pi / 2 - sunElevation   // vanilla's sun is (-sin a, cos a, 0)
let f = 1 / tan(35 * Float.pi / 180), aspect = Float(W) / Float(H)
let proj = simd_float4x4(SIMD4(f / aspect, 0, 0, 0), SIMD4(0, f, 0, 0), SIMD4(0, 0, 0, -1), SIMD4(0, 0, 0.05, 0))
let view = simd_float4x4(SIMD4(0, 0, 1, 0), SIMD4(0, 1, 0, 0), SIMD4(-1, 0, 0, 0), SIMD4(0, 0, 0, 1))   // forward = -x
var mats = [Float](repeating: 0, count: 32)
for c in 0..<4 { for r in 0..<4 { mats[c * 4 + r] = proj[c][r]; mats[16 + c * 4 + r] = view[c][r] } }

func frame(_ index: Int64, taa: Bool) -> [Float] {
    // GameRenderer.renderLevel's start: the tables (no pass open).
    var prep: [Float] = [sunAngle, 1, 87, 3000, 3200]
    check(skyPrepare(&prep) == 1, "frame \(index): sky tables prepared")
    // SkyRenderer.render's start (before its pass opens): the quarter-resolution sky, on the first frame only, so the
    // second one works out every pixel (the fallback).
    if !taa { check(skyViewFn(color, &mats) == 1, "frame \(index): quarter-resolution sky") }
    // Vanilla's sky pass: the sky drawn in place of the sky disc.
    var h = color
    var clear: [Float] = [0, 0, 0, 0]
    check(passBegin(&h, 1, 1, &clear, depth, 1, 0, 0, 0, W, H) == 1, "frame \(index): sky pass")
    check(skyDraw(&mats) == 1, "frame \(index): sky drawn")
    passEnd()
    // The main pass: vanilla-style terrain into the float target (the backend substitutes the pass's format).
    check(passBegin(&h, 1, 0, &clear, depth, 0, 0, 0, 0, W, H) == 1, "frame \(index): main pass")
    check(setPipeline(pipe) == 1, "frame \(index): RGBA8-declared pipeline accepted in the \(sdr ? "RGBA8" : "RGBA16Float") pass")
    draw(6, 1, 0, 0)
    passEnd()
    // After the level: aerial perspective (its own pass, or left for the anti-aliasing).
    check(skyAerial(color, depth, &mats, taa ? 1 : 0) == 1, "frame \(index): aerial perspective \(taa ? "left for the anti-aliasing" : "as its own pass")")
    if taa {
        var cam: [Double] = [0, 150, 0]
        check(taaApply(color, depth, &mats, &cam, 0, 0, 0) == 1, "frame \(index): anti-aliasing with the aerial perspective in its load")
    }
    // Read the frame back.
    let bpp: Int32 = sdr ? 4 : 8
    let buf = bufferCreate(Int64(W) * Int64(H) * Int64(bpp))
    copyToBuffer(color, 0, 0, 0, W, H, buf, 0, W * bpp)
    submit(index)
    check(waitSubmit(index, 5_000_000_000) == 1, "frame \(index): GPU done")
    let raw = UnsafeRawPointer(bitPattern: Int(bufferContents(buf)))!
    let n = Int(W * H * 4)
    if sdr { return (0..<n).map { Float(raw.assumingMemoryBound(to: UInt8.self)[$0]) / 255 } }
    return (0..<n).map { Float(raw.assumingMemoryBound(to: Float16.self)[$0]) }
}

func stats(_ px: [Float], _ label: String) {
    var nan = 0, over1 = 0, maxV: Float = 0
    for i in 0..<px.count where i % 4 < 3 {
        let v = Float(px[i])
        if !v.isFinite { nan += 1 } else { maxV = max(maxV, v); if v > 1 { over1 += 1 } }
    }
    func at(_ x: Int, _ y: Int) -> String {
        let i = (y * Int(W) + x) * 4
        return String(format: "(%.3f, %.3f, %.3f)", Float(px[i]), Float(px[i + 1]), Float(px[i + 2]))
    }
    // Texture row 0 is the bottom of the screen: near terrain at the bottom, far terrain just under the horizon.
    print("      \(label): NaN/Inf \(nan), channels above 1 (the sun, headroom 1 offline) \(over1), brightest \(String(format: "%.3f", maxV)); "
          + "near terrain \(at(Int(W) / 2, 5)), far terrain \(at(Int(W) / 2, Int(H) / 2 - 20)), sky above \(at(Int(W) / 4, Int(H) - 10))")
    check(nan == 0, "\(label): no NaN")
}

let a = frame(10, taa: false)
stats(a, "aerial perspective pass")
// The near terrain keeps vanilla's color (0.40, 0.52, 0.30 encoded) within a step; far terrain has taken on the haze,
// which toward a low sun is its warm light (away from it, the sky's blue).
let near = (5 * Int(W) + Int(W) / 2) * 4, far = ((Int(H) / 2 - 20) * Int(W) + Int(W) / 2) * 4
check(abs(Float(a[near]) - 0.40) < 0.01 && abs(Float(a[near + 1]) - 0.52) < 0.01 && abs(Float(a[near + 2]) - 0.30) < 0.01,
      "near terrain keeps its own color")
check(Float(a[far]) > Float(a[near]) + 0.05 && Float(a[far + 2]) > Float(a[near + 2]) + 0.05, "far terrain hazed (brighter, toward the sunlit haze)")
let b = frame(11, taa: true)
stats(b, "inside the anti-aliasing")
// Frame 10's sky came from the quarter-resolution one, frame 11's was worked out per pixel: over the sky (rows above
// the horizon, clear of the terrain's anti-aliased edge) they should agree to well under an 8-bit step, bar the
// sharpening at the sun's rim.
var maxDiff: Float = 0, meanDiff: Double = 0, n = 0
for y in (Int(H) / 2 + 10)..<Int(H) {
    for x in 0..<Int(W) {
        let i = (y * Int(W) + x) * 4
        let d = abs(Float(a[i + 1]) - Float(b[i + 1]))
        maxDiff = max(maxDiff, d)
        meanDiff += Double(d)
        n += 1
    }
}
print(String(format: "      sky, quarter resolution filtered up vs every pixel (green): mean difference %.5f, largest %.4f", meanDiff / Double(n), maxDiff))
// (On an 8-bit target both are dithered, with different noise each frame: allow a step.)
check(meanDiff / Double(n) < (sdr ? 1.0 : 0.5) / 255, "quarter-resolution sky matches the per-pixel one on average to \(sdr ? "an 8-bit step" : "half an 8-bit step")")

// The frame as an 8-bit image: with HDR, the screenshot copy (rolled into SDR); without, the target itself.
if !sdr { check(hdrSnapshot(color, snap) == 1, "HDR screenshot copy encoded") }
let sbuf = bufferCreate(Int64(W * H * 4))
copyToBuffer(sdr ? color : snap, 0, 0, 0, W, H, sbuf, 0, W * 4)
submit(12)
check(waitSubmit(12, 5_000_000_000) == 1, "8-bit frame read back")
let sp = UnsafeRawPointer(bitPattern: Int(bufferContents(sbuf)))!.assumingMemoryBound(to: UInt8.self)
var rgba = [UInt8](repeating: 0, count: Int(W * H * 4))
for y in 0..<Int(H) {   // flip: texture row 0 is the bottom
    for x in 0..<Int(W) * 4 { rgba[y * Int(W) * 4 + x] = sp[(Int(H) - 1 - y) * Int(W) * 4 + x] }
}
for i in stride(from: 3, to: rgba.count, by: 4) { rgba[i] = 255 }
let provider = CGDataProvider(data: Data(rgba) as CFData)!
let img = CGImage(width: Int(W), height: Int(H), bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: Int(W) * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
let dest = CGImageDestinationCreateWithURL(outDir.appendingPathComponent(sdr ? "frame-sdr.png" : "frame-hdr-sdr-copy.png") as CFURL, "public.png" as CFString, 1, nil)!
CGImageDestinationAddImage(dest, img, nil)
CGImageDestinationFinalize(dest)
print("      wrote \(sdr ? "frame-sdr.png" : "frame-hdr-sdr-copy.png")")

// The present into a layer that isn't on any screen (nothing is shown): EDR with HDR, 8-bit without.
let layer = CAMetalLayer()
let surface = surfaceCreate(Int64(Int(bitPattern: Unmanaged.passUnretained(layer).toOpaque())))
check(surface != 0, "surface over an offscreen CAMetalLayer")
check(sdr ? layer.pixelFormat == .bgra8Unorm && !layer.wantsExtendedDynamicRangeContent : layer.pixelFormat == .rgba16Float && layer.wantsExtendedDynamicRangeContent,
      sdr ? "layer is 8-bit SDR as before" : "layer is RGBA16Float with EDR on")
print("      layer color space: \(layer.colorspace.map { ($0.name as String?) ?? "?" } ?? "nil")")
surfaceConfigure(surface, W, H, 1)
surfaceBlit(surface, color)
submit(13)
check(waitSubmit(13, 5_000_000_000) == 1, sdr ? "SDR present done" : "HDR present encoded and done (decode to linear light into the float drawable)")
surfacePresent(surface)
