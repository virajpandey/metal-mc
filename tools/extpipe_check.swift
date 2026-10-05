// Offline check of an external pipeline (METALMC_EXTPIPE, Sources/MetalMCNative/ExtPipe.swift) through the library's C
// entry points as the game calls them, no game and no window: loads and compiles the description, then runs frames with
// an empty G-buffer (nothing drawn: the sky everywhere) through every deferred, composite and final pass into a main
// target, and writes that target as a PNG. It checks the description loads, every program compiles and every pass's
// pipeline state builds, the passes run without GPU errors, and shows what the pipeline makes of an empty level (its sky).
// usage: swift build -c release --product MetalMCNative; swiftc -O tools/extpipe_check.swift -o .build/extpipe_check
//        .build/extpipe_check .build/release/libMetalMCNative.dylib <pipeline dir> <out.png> [width height frames sunAngle]
//   sunAngle: OptiFine's (0 sunrise, 0.25 noon, 0.5 sunset); the camera looks west, 10 degrees up, at y 80.
import CoreGraphics
import Foundation
import ImageIO
import simd

let args = CommandLine.arguments
guard args.count >= 4 else { print("usage: extpipe_check <dylib> <pipeline dir> <out.png> [width height frames sunAngle]"); exit(1) }
setenv("METALMC_EXTPIPE", args[2], 1)
let W = args.count > 4 ? Int32(args[4])! : 1728
let H = args.count > 5 ? Int32(args[5])! : 1117
let frames = args.count > 6 ? Int(args[6])! : 8
let sunAngle = args.count > 7 ? Float(args[7])! : 0.25
guard let lib = dlopen(args[1], RTLD_NOW) else { print("dlopen failed"); exit(1) }
func fn<T>(_ name: String, _: T.Type) -> T {
    guard let p = dlsym(lib, name) else { print("missing \(name)"); exit(1) }
    return unsafeBitCast(p, to: T.self)
}
let ctxInit = fn("mmc_ctx_init", (@convention(c) () -> Int32).self)
let textureCreate = fn("mmc_texture_create", (@convention(c) (Int32, Int32, Int32, Int32, Int32, Int32) -> Int64).self)
let bufferCreate = fn("mmc_buffer_create", (@convention(c) (Int64) -> Int64).self)
let bufferContents = fn("mmc_buffer_contents", (@convention(c) (Int64) -> Int64).self)
let extEnabled = fn("mmc_ext_enabled", (@convention(c) () -> Int32).self)
let extFrameBegin = fn("mmc_ext_frame_begin", (@convention(c) (UnsafePointer<Float>, Int32, Int32, Int32, Int64) -> Int32).self)
let extRedirect = fn("mmc_ext_redirect", (@convention(c) (Int32) -> Void).self)
let extTranslucent = fn("mmc_ext_translucent", (@convention(c) () -> Int32).self)
let extDebugView = fn("mmc_ext_debug_view", (@convention(c) (UnsafePointer<CChar>, Float) -> Void).self)
let passBegin = fn("mmc_pass_begin", (@convention(c) (UnsafePointer<Int64>, Int32, Int32, UnsafePointer<Float>, Int64, Int32, Float, Int32, Int32, Int32, Int32) -> Int32).self)
let passEnd = fn("mmc_pass_end", (@convention(c) () -> Void).self)
let copyTexToBuf = fn("mmc_copy_texture_to_buffer", (@convention(c) (Int64, Int32, Int32, Int32, Int32, Int32, Int64, Int64, Int32) -> Void).self)
let submit = fn("mmc_submit", (@convention(c) (Int64) -> Void).self)
let waitSubmit = fn("mmc_wait_submit", (@convention(c) (Int64, Int64) -> Int32).self)

guard ctxInit() == 1 else { print("mmc_ctx_init failed"); exit(1) }
let t0 = Date()
guard extEnabled() == 1 else { print("FAIL: the pipeline didn't load (see the [metalmc-native] lines above)"); exit(2) }
print(String(format: "loaded and compiled in %.2f s", Date().timeIntervalSince(t0)))

// The standard uniforms, in ExtPipe.swift's extStdLayout order.
let layout: [(String, Int)] = [
    ("gbufferModelView", 16), ("gbufferModelViewInverse", 16), ("gbufferPreviousModelView", 16), ("gbufferPreviousProjection", 16),
    ("gbufferProjection", 16), ("gbufferProjectionInverse", 16), ("shadowModelView", 16), ("shadowModelViewInverse", 16),
    ("shadowProjection", 16), ("shadowProjectionInverse", 16), ("cameraPosition", 3), ("previousCameraPosition", 3),
    ("sunPosition", 3), ("moonPosition", 3), ("upPosition", 3), ("shadowLightPosition", 3), ("skyColor", 3), ("fogColor", 3),
    ("eyeBrightness", 2), ("eyeBrightnessSmooth", 2), ("aspectRatio", 1), ("blindness", 1), ("darknessFactor", 1), ("far", 1),
    ("near", 1), ("fogMode", 1), ("fogStart", 1), ("fogEnd", 1), ("fogDensity", 1), ("frameCounter", 1), ("frameTime", 1),
    ("frameTimeCounter", 1), ("heldBlockLightValue", 1), ("heldBlockLightValue2", 1), ("heldItemId", 1), ("heldItemId2", 1),
    ("isEyeInWater", 1), ("moonPhase", 1), ("nightVision", 1), ("rainStrength", 1), ("sunAngle", 1), ("shadowAngle", 1),
    ("viewHeight", 1), ("viewWidth", 1), ("wetness", 1), ("worldTime", 1), ("worldDay", 1), ("screenBrightness", 1),
    ("eyeAltitude", 1), ("centerDepthSmooth", 1), ("hideGUI", 1), ("thunderStrength", 1), ("playerMood", 1),
]
var offsets: [String: Int] = [:]
var count = 0
for (n, c) in layout { offsets[n] = count; count += c }
var std = [Float](repeating: 0, count: count)
func put(_ n: String, _ v: [Float]) { for (i, x) in v.enumerated() { std[offsets[n]! + i] = x } }
func flat(_ m: simd_float4x4) -> [Float] { (0..<4).flatMap { c in (0..<4).map { r in m[c][r] } } }
func rotX(_ a: Float) -> simd_float4x4 { simd_float4x4(rows: [SIMD4(1, 0, 0, 0), SIMD4(0, cos(a), -sin(a), 0), SIMD4(0, sin(a), cos(a), 0), SIMD4(0, 0, 0, 1)]) }
func rotY(_ a: Float) -> simd_float4x4 { simd_float4x4(rows: [SIMD4(cos(a), 0, sin(a), 0), SIMD4(0, 1, 0, 0), SIMD4(-sin(a), 0, cos(a), 0), SIMD4(0, 0, 0, 1)]) }
func rotZ(_ a: Float) -> simd_float4x4 { simd_float4x4(rows: [SIMD4(cos(a), -sin(a), 0, 0), SIMD4(sin(a), cos(a), 0, 0), SIMD4(0, 0, 1, 0), SIMD4(0, 0, 0, 1)]) }
let rad = Float.pi / 180
// Looking west (yaw 90: vanilla's view rotation is rotX(pitch) * rotY(yaw + 180)), 10 degrees up.
let view = rotX(-10 * rad) * rotY((90 + 180) * rad)
let fov: Float = 70 * rad, aspect = Float(W) / Float(H), n: Float = 0.05, f: Float = 768
let t = 1 / tan(fov / 2)
let proj = simd_float4x4(rows: [SIMD4(t / aspect, 0, 0, 0), SIMD4(0, t, 0, 0), SIMD4(0, 0, -(f + n) / (f - n), -2 * f * n / (f - n)), SIMD4(0, 0, -1, 0)])
put("gbufferModelView", flat(view)); put("gbufferModelViewInverse", flat(view.inverse))
put("gbufferPreviousModelView", flat(view)); put("gbufferPreviousProjection", flat(proj))
put("gbufferProjection", flat(proj)); put("gbufferProjectionInverse", flat(proj.inverse))
let skyAngle = sunAngle >= 0.25 ? sunAngle - 0.25 : sunAngle + 0.75
let extConst = fn("mmc_ext_const", (@convention(c) (UnsafePointer<CChar>, Float) -> Float).self)
let sunPath: Float = extConst("sunPathRotation", 0)
let cel = view * rotY(-90 * rad) * rotZ(sunPath * rad) * rotX(skyAngle * 360 * rad)
let sun = cel * SIMD4<Float>(0, 100, 0, 0), moon = cel * SIMD4<Float>(0, -100, 0, 0), up = view * SIMD4<Float>(0, 100, 0, 0)
put("sunPosition", [sun.x, sun.y, sun.z]); put("moonPosition", [moon.x, moon.y, moon.z]); put("upPosition", [up.x, up.y, up.z])
put("shadowLightPosition", sunAngle <= 0.5 ? [sun.x, sun.y, sun.z] : [moon.x, moon.y, moon.z])
let shadowAngle = sunAngle <= 0.5 ? sunAngle : sunAngle - 0.5
let shadowSky = shadowAngle < 0.25 ? shadowAngle + 0.75 : shadowAngle - 0.25
var sv = matrix_identity_float4x4
sv.columns.3 = SIMD4(0, 0, -100, 1)
sv = sv * rotX(90 * rad) * rotZ(-shadowSky * 360 * rad) * rotX(sunPath * rad)
let sd: Float = extConst("shadowDistance", 160), sn: Float = 0.05, sf: Float = 256
let sp = simd_float4x4(rows: [SIMD4(1 / sd, 0, 0, 0), SIMD4(0, 1 / sd, 0, 0), SIMD4(0, 0, 2 / (sn - sf), -(sf + sn) / (sf - sn)), SIMD4(0, 0, 0, 1)])
put("shadowModelView", flat(sv)); put("shadowModelViewInverse", flat(sv.inverse))
put("shadowProjection", flat(sp)); put("shadowProjectionInverse", flat(sp.inverse))
put("cameraPosition", [8.5, 80, 8.5]); put("previousCameraPosition", [8.5, 80, 8.5])
put("fogColor", [0.7, 0.8, 1.0]); put("skyColor", [0.47, 0.65, 1.0])
put("eyeBrightness", [0, 240]); put("eyeBrightnessSmooth", [0, 240])
put("aspectRatio", [aspect]); put("far", [192]); put("near", [n])
put("sunAngle", [sunAngle]); put("shadowAngle", [shadowAngle])
put("viewWidth", [Float(W)]); put("viewHeight", [Float(H)]); put("screenBrightness", [0.5]); put("eyeAltitude", [80])
put("worldTime", [Float(Int((sunAngle - 0.25) * 24000 + 30000) % 24000)])

// The main target (RGBA8, a render target) and a readback buffer.
let main = textureCreate(70, W, H, 1, 1, 1)
let rowBytes = Int(W) * 4
let readback = bufferCreate(Int64(rowBytes * Int(H)))
var submitIndex: Int64 = 2
var colors: [Int64] = [main]
let clear: [Float] = [0, 0, 0, 0]
let tf = Date()
func frame(_ frame: Int, readback doRead: Bool) {
    put("frameCounter", [Float(frame)]); put("frameTimeCounter", [Float(frame) / 60]); put("frameTime", [1.0 / 60])
    guard extFrameBegin(std, Int32(count), W, H, 0) == 1 else { print("FAIL: frame \(frame) not started"); exit(3) }
    extRedirect(2)
    guard passBegin(&colors, 1, 0, clear, 0, 0, 0, 0, 0, W, H) == 1 else { print("FAIL: G-buffer pass didn't open"); exit(3) }
    _ = extTranslucent()   // the deferred passes
    passEnd()              // composite and final
    if doRead { copyTexToBuf(main, 0, 0, 0, W, H, readback, 0, Int32(rowBytes)) }
    submit(submitIndex)
    _ = waitSubmit(submitIndex, -1)
    submitIndex += 1
}
for f in 0..<frames { frame(f, readback: f == frames - 1) }
print(String(format: "%d frames in %.2f s (%.1f ms a frame, waiting for each)", frames, Date().timeIntervalSince(tf), Date().timeIntervalSince(tf) * 1000 / Double(frames)))

// Row 0 of the target is the GL bottom row: flip for the PNG.
func writePNG(_ path: String) {
let src = UnsafeRawPointer(bitPattern: Int(bufferContents(readback)))!.assumingMemoryBound(to: UInt8.self)
var px = [UInt8](repeating: 0, count: rowBytes * Int(H))
var sum = SIMD3<Double>(0, 0, 0)
for y in 0..<Int(H) {
    for x in 0..<rowBytes {
        let v = src[(Int(H) - 1 - y) * rowBytes + x]
        px[y * rowBytes + x] = x % 4 == 3 ? 255 : v
        if x % 4 < 3 { sum[x % 4] += Double(v) }
    }
}
let np = Double(W) * Double(H)
print(String(format: "mean color %.1f %.1f %.1f", sum.x / np, sum.y / np, sum.z / np))
let cs = CGColorSpace(name: CGColorSpace.sRGB)!
let ctx = CGContext(data: &px, width: Int(W), height: Int(H), bitsPerComponent: 8, bytesPerRow: rowBytes, space: cs,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
let img = ctx.makeImage()!
let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil)!
CGImageDestinationAddImage(dest, img, nil)
CGImageDestinationFinalize(dest)
print("wrote \(path)")
}
writePNG(args[3])
// EXTCHECK_VIEWS=<target>[:scale],...: one more frame per view, with that target on the screen (ExtPipe's debug view).
for v in (ProcessInfo.processInfo.environment["EXTCHECK_VIEWS"] ?? "").split(separator: ",") {
    let parts = v.split(separator: ":")
    extDebugView(String(parts[0]), parts.count > 1 ? Float(parts[1]) ?? 1 : 1)
    frame(frames, readback: true)
    print("view \(v):", terminator: " ")
    writePNG(args[3].replacingOccurrences(of: ".png", with: "-\(parts[0]).png"))
}
