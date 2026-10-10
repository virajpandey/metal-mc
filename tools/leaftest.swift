// Offline checks of foliage (METALMC_EXP=leaflight and wave, Sources/MetalMCNative/Foliage.swift) through
// libMetalMCNative's C entry points, as the game calls them, no game and no window (tools/litflow.swift's way).
// usage: swift build -c release --product MetalMCNative; swiftc -O tools/leaftest.swift -o .build/leaftest
//   .build/leaftest <dylib> - - compile
//       compiles every shader foliage changes, as the library builds them under METALMC_EXP (default the full look with
//       leaflight and wave): the LOD's quads (vertex, fragment with the G-buffer), the near chunks' (solid and cutout),
//       the ray-traced shadows' kernel, lit mode's relight pass, the anti-aliasing's resolve (lit, sky).
//   .build/leaftest <dylib> <world dir> <output dir> [scene ...]
//       opens the world's LOD around each scene's camera (tools/bench/scenes.json's forest, noon_overview, mountain_view;
//       the LOD's level 0 stands in for the near terrain), and per scene writes the relit frame (<scene>.png, the mean of
//       16 frames: the shadows' rays take one pixel of each 4 x 4 block a frame), the foliage classes the G-buffer holds
//       (<scene>-classes.png: leaves green, plants yellow, other terrain gray), the traced texel (<scene>-traced.png: red
//       the direct sun, green the light through leaves), and counts. LEAFTEST_TIME=1 adds the shadows' pass timed at the
//       panel's resolution (3456 x 2234) with the sun through leaves and with leaves opaque (mmc_debug_foliage_rt_time),
//       and the relight's own pass (mmc_debug_lit_time). METALMC_EXP sets the switches (default lit,rtshadows,sky,gi,
//       leaflight,wave); run once without leaflight for the "before" pictures. LEAFTEST_POST=1 adds the post chain.
//       LEAFTEST_SIZE=w,h sets the pictures' size (1728,1117).
import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

setbuf(stdout, nil)
let args = CommandLine.arguments
guard args.count >= 4 else { print("usage: leaftest <dylib> <world dir> <output dir> [scene ...|compile]"); exit(1) }
let compileOnly = args.count >= 5 && args[4] == "compile"
let defaultExp = compileOnly ? "lit,nearchunks,rtshadows,sky,gi,water,coloredlight,post,leaflight,wave" : "lit,rtshadows,sky,gi,leaflight,wave"
let exp = ProcessInfo.processInfo.environment["METALMC_EXP"] ?? defaultExp
setenv("METALMC_EXP", exp, 1)
let expSet = Set(exp.split(separator: ",").map(String.init))
guard let lib = dlopen(args[1], RTLD_NOW) else { print("dlopen failed"); exit(1) }
func fn<T>(_ name: String, _: T.Type) -> T {
    guard let p = dlsym(lib, name) else { print("missing \(name)"); exit(1) }
    return unsafeBitCast(p, to: T.self)
}
func check(_ ok: Bool, _ what: String) {
    print((ok ? "ok    " : "FAIL  ") + what)
    if !ok { exit(1) }
}
let ctxInit = fn("mmc_ctx_init", (@convention(c) () -> Int32).self)
check(ctxInit() == 1, "context (METALMC_EXP=\(exp))")
let foliageEnabled = fn("mmc_foliage_enabled", (@convention(c) () -> Int32).self)()
print("      foliage switches: leaflight \(foliageEnabled & 1 != 0), wave \(foliageEnabled & 2 != 0)")

// MARK: - compile

if compileOnly {
    let device = MTLCreateSystemDefaultDevice()!
    func source(_ name: String, _ which: Int32? = nil) -> String? {
        guard let p = dlsym(lib, name) else { return nil }
        var buf = [CChar](repeating: 0, count: 1 << 22)
        let n: Int32
        if let which {
            let f = unsafeBitCast(p, to: (@convention(c) (Int32, UnsafeMutablePointer<CChar>, Int32) -> Int32).self)
            n = f(which, &buf, Int32(buf.count))
        } else {
            let f = unsafeBitCast(p, to: (@convention(c) (UnsafeMutablePointer<CChar>, Int32) -> Int32).self)
            n = f(&buf, Int32(buf.count))
        }
        return n > 0 && Int(n) < buf.count ? String(cString: buf) : nil
    }
    var failures = 0
    func attempt(_ what: String, _ body: () throws -> Void) {
        do { try body(); print("ok    \(what)") } catch { print("FAIL  \(what): \(error)"); failures += 1 }
    }
    let lit = expSet.contains("lit")
    let colorFormats: [MTLPixelFormat] = lit ? [.rg11b10Float, .rg32Uint] : [.rgba8Unorm]
    if let src = source("mmc_debug_foliage_shader_source", 0) {
        attempt("the LOD's quads: lod_vs, lod_fs (and seam, fade)\(lit ? " with the G-buffer" : "")") {
            let l = try device.makeLibrary(source: src, options: nil)
            for fs in ["lod_fs", "lod_fs_seam", "lod_fs_fade"] {
                guard let f = l.makeFunction(name: fs) else { continue }
                let d = MTLRenderPipelineDescriptor()
                d.vertexFunction = l.makeFunction(name: "lod_vs")
                d.fragmentFunction = f
                for (i, cf) in colorFormats.enumerated() { d.colorAttachments[i].pixelFormat = cf }
                d.depthAttachmentPixelFormat = .depth32Float
                _ = try device.makeRenderPipelineState(descriptor: d)
            }
            print("      the LOD's source: \(src.contains("lodFoliageLeaf") ? "foliage leaf test" : "no foliage"), \(src.contains("LOD_WAVE") ? "wave" : "no wave")")
        }
    } else { print("FAIL  no LOD source"); failures += 1 }
    if expSet.contains("nearchunks"), let src = source("mmc_debug_foliage_shader_source", 1) {
        attempt("the near chunks: near_vs, near_fs solid and cutout\(lit ? " with the G-buffer" : "")") {
            let opts = MTLCompileOptions()
            opts.languageVersion = .version3_0
            opts.preserveInvariance = true
            let l = try device.makeLibrary(source: src, options: opts)
            for cutout in [false, true] {
                let c = MTLFunctionConstantValues()
                var cut = cutout
                c.setConstantValue(&cut, type: .bool, index: 0)
                let d = MTLRenderPipelineDescriptor()
                d.vertexFunction = l.makeFunction(name: "near_vs")
                d.fragmentFunction = try l.makeFunction(name: "near_fs", constantValues: c)
                for (i, cf) in colorFormats.enumerated() { d.colorAttachments[i].pixelFormat = cf }
                d.depthAttachmentPixelFormat = .depth32Float
                _ = try device.makeRenderPipelineState(descriptor: d)
            }
            print("      the near chunks' source: \(src.contains("NEAR_FOLIAGE") ? "foliage classes" : "no foliage"), \(src.contains("NEAR_WAVE") ? "wave" : "no wave")")
        }
    }
    if let src = source("mmc_debug_foliage_shader_source", 2) {
        attempt("the ray-traced shadows' kernel\(src.contains("rtThroughLeaves") ? " (the sun through leaves)" : "")") {
            let l = try device.makeLibrary(source: src, options: nil)
            _ = try device.makeComputePipelineState(function: l.makeFunction(name: "rt_shadow")!)
        }
    }
    if lit, let src = source("mmc_debug_lit_shader_source") {
        attempt("lit mode's relight pass (RG11B10Float, RGBA8)\(src.contains("litFoliageLight") ? " with the foliage term" : "")") {
            let l = try device.makeLibrary(source: src, options: nil)
            for format in [MTLPixelFormat.rg11b10Float, .rgba8Unorm] {
                let d = MTLRenderPipelineDescriptor()
                d.vertexFunction = l.makeFunction(name: "lit_fullscreen_vs")
                d.fragmentFunction = l.makeFunction(name: "lit_relight_fs")
                d.colorAttachments[0].pixelFormat = format
                _ = try device.makeRenderPipelineState(descriptor: d)
            }
        }
    }
    if let src = source("mmc_debug_taa_shader_source") {
        for tile in [32, 16] {
            attempt("the anti-aliasing's resolve, \(tile) x \(tile) tiles, lit and sky variants") {
                let opts = MTLCompileOptions()
                opts.preprocessorMacros = ["TAA_T": NSNumber(value: tile)]
                let l = try device.makeLibrary(source: src, options: opts)
                for (skyV, litV) in [(false, false), (true, false), (false, true), (true, true)] where lit || !litV {
                    let v = MTLFunctionConstantValues()
                    var s = skyV, li = litV
                    v.setConstantValue(&s, type: .bool, index: 0)
                    if lit { v.setConstantValue(&li, type: .bool, index: 1) }
                    _ = try device.makeComputePipelineState(function: try l.makeFunction(name: "taa_resolve", constantValues: v))
                }
            }
        }
    }
    print(failures == 0 ? "done: every variant compiles" : "FAIL  \(failures) failed")
    exit(failures == 0 ? 0 : 1)
}

// MARK: - pictures

let textureCreate = fn("mmc_texture_create", (@convention(c) (Int32, Int32, Int32, Int32, Int32, Int32) -> Int64).self)
let bufferCreate = fn("mmc_buffer_create", (@convention(c) (Int64) -> Int64).self)
let bufferContents = fn("mmc_buffer_contents", (@convention(c) (Int64) -> Int64).self)
let release = fn("mmc_handle_release", (@convention(c) (Int64) -> Void).self)
let passBegin = fn("mmc_pass_begin", (@convention(c) (UnsafePointer<Int64>, Int32, Int32, UnsafePointer<Float>, Int64, Int32, Float,
                                                       Int32, Int32, Int32, Int32) -> Int32).self)
let passEnd = fn("mmc_pass_end", (@convention(c) () -> Void).self)
let copyToBuffer = fn("mmc_copy_texture_to_buffer", (@convention(c) (Int64, Int32, Int32, Int32, Int32, Int32, Int64, Int64, Int32) -> Void).self)
let copyBufferToTexture = fn("mmc_copy_buffer_to_texture", (@convention(c) (Int64, Int64, Int32, Int64, Int32, Int32, Int32, Int32, Int32, Int32) -> Void).self)
let submit = fn("mmc_submit", (@convention(c) (Int64) -> Void).self)
let waitSubmit = fn("mmc_wait_submit", (@convention(c) (Int64, Int64) -> Int32).self)
let lodOpen = fn("mmc_lod_open", (@convention(c) (UnsafePointer<CChar>, Int32, Int32, Int32) -> Int32).self)
let lodStatus = fn("mmc_lod_status", (@convention(c) (UnsafeMutablePointer<Int64>) -> Void).self)
let lodDraw = fn("mmc_lod_draw", (@convention(c) (UnsafePointer<Float>, UnsafePointer<Double>) -> Int32).self)
let lodSetLightmap = fn("mmc_lod_set_lightmap", (@convention(c) (Int64) -> Void).self)
let shadowsApply = fn("mmc_shadows_apply", (@convention(c) (Int64, Int64, UnsafePointer<Float>, UnsafePointer<Double>, Float, Float, Float, Int32) -> Int32).self)
let skyPrepare = fn("mmc_sky_prepare", (@convention(c) (UnsafePointer<Float>) -> Int32).self)
let skyDraw = fn("mmc_sky_draw", (@convention(c) (UnsafePointer<Float>) -> Int32).self)
let skyAerial = fn("mmc_sky_aerial", (@convention(c) (Int64, Int64, UnsafePointer<Float>, Int32) -> Int32).self)
let hdrSnapshot = fn("mmc_hdr_snapshot", (@convention(c) (Int64, Int64) -> Int32).self)
let litLevelPass = fn("mmc_lit_level_pass", (@convention(c) () -> Void).self)
let litRelight = fn("mmc_lit_relight", (@convention(c) (Int64, Int64, UnsafePointer<Float>, Int64, Int32) -> Int32).self)
let litSetView = fn("mmc_lit_set_view", (@convention(c) (Int32) -> Void).self)
let litCopyGbuffer = fn("mmc_lit_copy_gbuffer", (@convention(c) (Int64, Int64) -> Int32).self)
let litTime = fn("mmc_debug_lit_time", (@convention(c) (Int64, Int64, UnsafePointer<Float>, Int64, Int32, UnsafeMutablePointer<Double>) -> Void).self)
let foliageTime = fn("mmc_debug_foliage_time", (@convention(c) (Double) -> Void).self)
let foliageRtTime = fn("mmc_debug_foliage_rt_time", (@convention(c) (Int64, Int64, UnsafePointer<Float>, UnsafePointer<Double>, Float, Int32, UnsafeMutablePointer<Double>) -> Void).self)
let postOn = ProcessInfo.processInfo.environment["LEAFTEST_POST"] == "1" && expSet.contains("post")
let postApply = postOn ? fn("mmc_post_apply", (@convention(c) (Int64, Int64, UnsafePointer<Float>, UnsafePointer<Double>, Int32) -> Int32).self) : nil
let sky = expSet.contains("sky")
let floatTarget = postOn || expSet.contains("hdr")
foliageTime(100)

struct Scene { let name: String; let pos: SIMD3<Double>; let yaw: Double; let pitch: Double; let time: Double }
// tools/bench/scenes.json's poses (the feet; the eye is 1.62 higher).
let allScenes: [Scene] = [
    Scene(name: "forest", pos: SIMD3(-12, 92, 372), yaw: 125, pitch: 18, time: 3000),
    Scene(name: "noon_overview", pos: SIMD3(8, 165, 8), yaw: 281.5, pitch: 12, time: 6000),
    Scene(name: "mountain_view", pos: SIMD3(1064, 172, 230), yaw: 90, pitch: 6, time: 7000),
]
let wanted = args.count > 4 ? Array(args[4...]) : ["forest"]
let scenes = allScenes.filter { wanted.contains($0.name) }
let outDir = URL(fileURLWithPath: args[3])
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
let tag = expSet.contains("leaflight") ? "after" : "before"

/// Vanilla's sun angle for a day time (DimensionType.timeOfDay x 2 pi).
func sunAngle(_ time: Double) -> Float {
    let d0 = (time / 24000 - 0.25) - (time / 24000 - 0.25).rounded(.down)
    let d1 = 0.5 - cos(d0 * .pi) / 2
    return Float((d0 * 2 + d1) / 3 * 2 * .pi)
}

// Vanilla's lightmap at a sky factor (litflow's).
func lightmapPixels(skyFactor: Float) -> [UInt8] {
    var px = [UInt8](repeating: 255, count: 16 * 16 * 4)
    func br(_ x: Float) -> Float { x / (4 - 3 * x) }
    for sy in 0..<16 {
        for bx in 0..<16 {
            let bl = Float(bx) / 15, sl = Float(sy) / 15
            var c = SIMD3<Float>(repeating: 1) * br(sl) * skyFactor
            let m = 0.9 * (2 * bl - 1) * (2 * bl - 1)
            c += (SIMD3<Float>(1.0, 0.85, 0.65) * (1 - m) + SIMD3<Float>(repeating: 1) * m) * br(bl)
            c = simd_clamp(c, .zero, SIMD3(repeating: 1))
            let mx = max(c.x, max(c.y, c.z)), inv = 1 - mx
            let ng = mx > 0 ? c * ((1 - inv * inv * inv * inv) / mx) : c
            c = c * 0.5 + ng * 0.5
            for k in 0..<3 { px[(sy * 16 + bx) * 4 + k] = UInt8((c[k] * 255).rounded()) }
        }
    }
    return px
}
let lightmap = textureCreate(70, 16, 16, 1, 1, 0)
let lmStaging = bufferCreate(16 * 16 * 4)
var submitIndex: Int64 = 10
do {
    let px = lightmapPixels(skyFactor: 1)
    UnsafeMutableRawPointer(bitPattern: Int(bufferContents(lmStaging)))!.copyMemory(from: px, byteCount: px.count)
    copyBufferToTexture(lmStaging, 0, 64, lightmap, 0, 0, 0, 0, 16, 16)
    submit(submitIndex); _ = waitSubmit(submitIndex, 5_000_000_000); submitIndex += 1
}
lodSetLightmap(lightmap)

var cam = SIMD3<Double>(0, 0, 0)
var yaw: Float = 0, pitch: Float = 0
func matrices(width: Int32, height: Int32) -> [Float] {
    let fwd = SIMD3<Float>(-sin(yaw) * cos(pitch), -sin(pitch), cos(yaw) * cos(pitch))
    let right = simd_normalize(simd_cross(fwd, SIMD3(0, 1, 0))), up = simd_cross(right, fwd)
    let view = simd_float4x4(rows: [SIMD4(right, 0), SIMD4(up, 0), SIMD4(-fwd, 0), SIMD4(0, 0, 0, 1)])
    let f = 1 / tan(35 * Float.pi / 180), aspect = Float(width) / Float(height)
    let proj = simd_float4x4(SIMD4(f / aspect, 0, 0, 0), SIMD4(0, f, 0, 0), SIMD4(0, 0, 0, -1), SIMD4(0, 0, 0.05, 0))
    var m = [Float](repeating: 0, count: 32)
    for c in 0..<4 { for r in 0..<4 { m[c * 4 + r] = proj[c][r]; m[16 + c * 4 + r] = view[c][r] } }
    return m
}
struct Targets { let color: Int64, depth: Int64, w: Int32, h: Int32 }
func targets(_ w: Int32, _ h: Int32) -> Targets {
    Targets(color: textureCreate(floatTarget ? 92 : 70, w, h, 1, 1, 1), depth: textureCreate(252, w, h, 1, 1, 1), w: w, h: h)
}

/// One frame as the game draws it: the sky, the level with the G-buffer, the shadows, the relight in its own pass, the
/// aerial perspective, (post).
func frame(_ t: Targets, sun: Float, readback: ((Int64) -> Void)? = nil) {
    var mats = matrices(width: t.w, height: t.h)
    if sky { var prep: [Float] = [sun, 1, Float(cam.y) - 63, 0, 0]; _ = skyPrepare(&prep) }
    var h = t.color
    var clear: [Float] = [0.62, 0.75, 0.95, 1]
    _ = passBegin(&h, 1, 1, &clear, t.depth, 1, 0, 0, 0, t.w, t.h)
    if sky { _ = skyDraw(&mats) }
    passEnd()
    litLevelPass()
    _ = passBegin(&h, 1, 0, &clear, t.depth, 0, 0, 0, 0, t.w, t.h)
    var p = mats + [0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, 0, 1]
    var c = [cam.x, cam.y, cam.z]
    _ = lodDraw(&p, &c)
    passEnd()
    _ = shadowsApply(t.color, t.depth, &mats, &c, sun, 0.42, 192, 0)
    var lp = mats + [sun, 0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, sky ? 1 : 0]
    _ = litRelight(t.color, t.depth, &lp, lightmap, 0)
    if sky { _ = skyAerial(t.color, t.depth, &mats, 0) }
    if let postApply { var pp = mats + [sun]; var c3 = [cam.x, cam.y, cam.z]; _ = postApply(t.color, t.depth, &pp, &c3, 0) }
    readback?(t.color)
    submit(submitIndex)
    if waitSubmit(submitIndex, 10_000_000_000) != 1 { check(false, "frame \(submitIndex) done") }
    submitIndex += 1
}

/// The color as 8-bit sRGB, the mean of `frames` frames, and the last frame's G-buffer.
func capture(_ t: Targets, sun: Float, view: Int32 = 0, frames: Int = 16) -> (color: [UInt8], gbuf: [UInt32]) {
    let n = Int(t.w) * Int(t.h)
    let cbuf = bufferCreate(Int64(n * 4)), gbuf = bufferCreate(Int64(n * 8))
    let snap = floatTarget ? textureCreate(70, t.w, t.h, 1, 1, 1) : 0
    litSetView(view)
    var sum = [Float](repeating: 0, count: n * 4)
    var gotG = false
    for k in 0..<frames {
        frame(t, sun: sun) { color in
            if floatTarget { _ = hdrSnapshot(color, snap) }
            copyToBuffer(floatTarget ? snap : color, 0, 0, 0, t.w, t.h, cbuf, 0, t.w * 4)
            if k == frames - 1 { gotG = litCopyGbuffer(gbuf, 0) == 1 }
        }
        let cp = UnsafeRawPointer(bitPattern: Int(bufferContents(cbuf)))!.assumingMemoryBound(to: UInt8.self)
        for i in 0..<(n * 4) { sum[i] += Float(cp[i]) }
    }
    litSetView(0)
    let color = sum.map { UInt8(min(255, ($0 / Float(frames)).rounded())) }
    var g: [UInt32] = []
    if gotG {
        let gp = UnsafeRawPointer(bitPattern: Int(bufferContents(gbuf)))!
        g = Array(UnsafeBufferPointer(start: gp.assumingMemoryBound(to: UInt32.self), count: n * 2))
    }
    release(cbuf); release(gbuf)
    if snap != 0 { release(snap) }
    return (color, g)
}

func writePNG(_ px: [UInt8], _ w: Int, _ h: Int, _ name: String) {
    var rgba = [UInt8](repeating: 255, count: w * h * 4)
    for y in 0..<h { for x in 0..<w { for k in 0..<3 { rgba[(y * w + x) * 4 + k] = px[((h - 1 - y) * w + x) * 4 + k] } } }
    let provider = CGDataProvider(data: Data(rgba) as CFData)!
    let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider, decode: nil,
                      shouldInterpolate: false, intent: .defaultIntent)!
    let dest = CGImageDestinationCreateWithURL(outDir.appendingPathComponent(name) as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
    print("      wrote \(name)")
}

let sizeSpec = (ProcessInfo.processInfo.environment["LEAFTEST_SIZE"] ?? "1728,1117").split(separator: ",").map { Int32($0)! }
for s in scenes {
    cam = s.pos + SIMD3(0, 1.62, 0)
    yaw = Float(s.yaw) * .pi / 180
    pitch = Float(s.pitch) * .pi / 180
    check(lodOpen(args[2], 2048, Int32(cam.x), Int32(cam.z)) == 1, "\(s.name): LOD opened on \(args[2])")
    var st = [Int64](repeating: 0, count: 4)
    var lastQuads: Int64 = -1, stable = 0
    let t0 = Date()
    while stable < 24 && Date().timeIntervalSince(t0) < 600 {
        lodStatus(&st)
        stable = st[0] == 2 && st[3] == 1 && st[2] == lastQuads ? stable + 1 : 0
        lastQuads = st[2]
        Thread.sleep(forTimeInterval: 0.25)
    }
    check(st[0] == 2, "\(s.name): LOD built: \(st[1]) nodes, \(st[2]) quads in \(Int(Date().timeIntervalSince(t0))) s")
    let sun = sunAngle(s.time)
    let t = targets(sizeSpec[0], sizeSpec[1])
    // Warm-up: the shadows' tile structures build in the background, nearest first, as tiles are drawn.
    for _ in 0..<240 { frame(t, sun: sun) }
    Thread.sleep(forTimeInterval: 3)
    for _ in 0..<120 { frame(t, sun: sun) }
    let (color, g) = capture(t, sun: sun)
    writePNG(color, Int(t.w), Int(t.h), "\(s.name)-\(tag).png")
    if !g.isEmpty {
        // The G-buffer's classes over lit terrain (face code 1-7), the albedo's blue byte's low bits with leaflight.
        var lit = 0, leaves = 0, plants = 0, other3 = 0
        var pic = [UInt8](repeating: 0, count: Int(t.w) * Int(t.h) * 4)
        for i in 0..<(Int(t.w) * Int(t.h)) {
            let x = g[2 * i]
            if x >> 29 == 0 { continue }
            lit += 1
            let cls = expSet.contains("leaflight") ? (x >> 16) & 3 : 0
            if cls == 1 { leaves += 1 } else if cls == 2 { plants += 1 } else if cls == 3 { other3 += 1 }
            let c: (UInt8, UInt8, UInt8) = cls == 1 ? (25, 200, 40) : (cls == 2 ? (230, 200, 25) : (64, 64, 64))
            pic[4 * i] = c.0; pic[4 * i + 1] = c.1; pic[4 * i + 2] = c.2
        }
        print(String(format: "      %@: lit terrain %.1f%% of the frame; of it leaves %.1f%%, plants %.1f%%, class 3 %.1f%%", s.name,
                     100 * Double(lit) / Double(t.w * t.h), 100 * Double(leaves) / Double(max(lit, 1)), 100 * Double(plants) / Double(max(lit, 1)),
                     100 * Double(other3) / Double(max(lit, 1))))
        if expSet.contains("leaflight") { writePNG(pic, Int(t.w), Int(t.h), "\(s.name)-classes.png") }
    }
    if expSet.contains("leaflight") {
        writePNG(capture(t, sun: sun, view: 14).color, Int(t.w), Int(t.h), "\(s.name)-traced.png")
        writePNG(capture(t, sun: sun, view: 13, frames: 1).color, Int(t.w), Int(t.h), "\(s.name)-view13.png")
    }
    writePNG(capture(t, sun: sun, view: 6).color, Int(t.w), Int(t.h), "\(s.name)-\(tag)-sunvis.png")
    release(t.color); release(t.depth)
    if ProcessInfo.processInfo.environment["LEAFTEST_TIME"] == "1" {
        let big = targets(3456, 2234)
        for _ in 0..<60 { frame(big, sun: sun) }
        var mats = matrices(width: big.w, height: big.h)
        var c = [cam.x, cam.y, cam.z]
        if expSet.contains("leaflight") {
            var out = [Double](repeating: 0, count: 4)
            foliageRtTime(big.color, big.depth, &mats, &c, sun, 60, &out)
            print(String(format: "      %@ at 3456 x 2234: the shadows' pass with the sun through leaves %.3f ms (fastest %.3f), leaves opaque %.3f (fastest %.3f): +%.3f",
                         s.name, out[0], out[2], out[1], out[3], out[0] - out[1]))
        }
        // The level pass alone (the LOD and the far field with the G-buffer), each in a submit of its own: median and
        // fastest GPU ms (with wave, the LOD's sway is in it).
        let gpuTimes = fn("mmc_gpu_times_take", (@convention(c) (UnsafeMutablePointer<Double>, Int32) -> Int32).self)
        var tbuf = [Double](repeating: 0, count: 256)
        _ = gpuTimes(&tbuf, 256)
        var levelTimes: [Double] = []
        for _ in 0..<60 {
            var h = big.color
            var clear: [Float] = [0.62, 0.75, 0.95, 1]
            litLevelPass()
            _ = passBegin(&h, 1, 1, &clear, big.depth, 1, 0, 0, 0, big.w, big.h)
            var p = mats + [0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, 0, 1]
            _ = lodDraw(&p, &c)
            passEnd()
            submit(submitIndex)
            _ = waitSubmit(submitIndex, 10_000_000_000)
            submitIndex += 1
            let k = gpuTimes(&tbuf, 256)
            if k > 0 { levelTimes.append(tbuf[Int(k) - 1] * 1000) }
        }
        levelTimes.sort()
        if !levelTimes.isEmpty {
            print(String(format: "      %@ at 3456 x 2234: the level pass %.3f ms (fastest %.3f) [%@%@]", s.name, levelTimes[levelTimes.count / 2],
                         levelTimes[0], tag, expSet.contains("wave") ? ", wave" : ""))
        }
        // The relight's own pass on this frame's G-buffer (median, fastest).
        frame(big, sun: sun)
        var lp = mats + [sun, 0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, sky ? 1 : 0]
        var o2 = [Double](repeating: 0, count: 2)
        litTime(big.color, big.depth, &lp, lightmap, 60, &o2)
        print(String(format: "      %@ at 3456 x 2234: the relight's own pass %.3f ms (fastest %.3f) [%@]", s.name, o2[0], o2[1], tag))
        release(big.color); release(big.depth)
    }
}
print("done")
