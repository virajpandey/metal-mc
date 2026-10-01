// Offline run of lit mode (METALMC_EXP=lit, Sources/MetalMCNative/Lit.swift) as the game calls it, no game and no window:
// through libMetalMCNative's C entry points it opens a fixture world's LOD, then per frame draws the sky pass, the level's
// main pass with the G-buffer (mmc_lit_level_pass, the LOD and far field through mmc_lod_draw, a vanilla-style "entity"
// that writes depth and a vanilla-style translucent band that doesn't, like rain), the ray-traced shadows' raw
// visibility, the relight (its own pass, or in the anti-aliasing's resolve), and with the sky its aerial perspective.
// It checks the G-buffer against the depth buffer and the overlay test, writes PNGs of the forward and relit frames at
// four times of day and of the G-buffer's debug views, and times the relight and the G-buffer's cost in the main pass at
// the panel's resolution.
// usage: swift build -c release --product MetalMCNative; swiftc -O tools/litflow.swift -o .build/litflow
//        .build/litflow .build/release/libMetalMCNative.dylib <world dir> <output dir> [sdr|sky|hdr|off] [time]
//   sdr: METALMC_EXP=lit,rtshadows (vanilla daylight curve); sky: + sky (the atmosphere's light); hdr: + sky,hdr (float
//   target); off: rtshadows only (no lit mode: the baseline main-pass time). "time" adds the timing runs
//   (LITFLOW_TIMEONLY=1: only those). METALMC_FARFIELD=<level> turns the far field on as in the game;
//   LITFLOW_VIEW=x,y,z,yaw,pitch moves the camera (default 8,150,8,100,22).
import CoreGraphics
import Foundation
import ImageIO
import simd

let args = CommandLine.arguments
guard args.count >= 4 else { print("usage: litflow <dylib> <world dir> <output dir> [sdr|sky|hdr|off] [time]"); exit(1) }
let mode = args.count >= 5 ? args[4] : "sdr"
let timing = args.count >= 6 && args[5] == "time"
let exp = mode == "off" ? "rtshadows" : (mode == "sky" ? "lit,rtshadows,sky" : (mode == "hdr" ? "lit,rtshadows,sky,hdr" : "lit,rtshadows"))
setenv("METALMC_EXP", exp, 1)
let lit = mode != "off", sky = mode == "sky" || mode == "hdr", hdr = mode == "hdr"
guard let lib = dlopen(args[1], RTLD_NOW) else { print("dlopen failed"); exit(1) }
let outDir = URL(fileURLWithPath: args[3])
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
func fn<T>(_ name: String, _: T.Type) -> T {
    guard let p = dlsym(lib, name) else { print("missing \(name)"); exit(1) }
    return unsafeBitCast(p, to: T.self)
}
let ctxInit = fn("mmc_ctx_init", (@convention(c) () -> Int32).self)
let textureCreate = fn("mmc_texture_create", (@convention(c) (Int32, Int32, Int32, Int32, Int32, Int32) -> Int64).self)
let bufferCreate = fn("mmc_buffer_create", (@convention(c) (Int64) -> Int64).self)
let bufferContents = fn("mmc_buffer_contents", (@convention(c) (Int64) -> Int64).self)
let release = fn("mmc_handle_release", (@convention(c) (Int64) -> Void).self)
let pipelineCreate = fn("mmc_pipeline_create", (@convention(c) (UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>, UnsafePointer<CChar>?,
                                                                 UnsafePointer<CChar>?, UnsafePointer<Int32>, Int32, UnsafeMutablePointer<CChar>, Int32) -> Int64).self)
let passBegin = fn("mmc_pass_begin", (@convention(c) (UnsafePointer<Int64>, Int32, Int32, UnsafePointer<Float>, Int64, Int32, Float,
                                                       Int32, Int32, Int32, Int32) -> Int32).self)
let passEnd = fn("mmc_pass_end", (@convention(c) () -> Void).self)
let setPipeline = fn("mmc_rp_set_pipeline", (@convention(c) (Int64) -> Int32).self)
let draw = fn("mmc_rp_draw", (@convention(c) (Int32, Int32, Int32, Int32) -> Void).self)
let copyToBuffer = fn("mmc_copy_texture_to_buffer", (@convention(c) (Int64, Int32, Int32, Int32, Int32, Int32, Int64, Int64, Int32) -> Void).self)
let copyBufferToTexture = fn("mmc_copy_buffer_to_texture", (@convention(c) (Int64, Int64, Int32, Int64, Int32, Int32, Int32, Int32, Int32, Int32) -> Void).self)
let submit = fn("mmc_submit", (@convention(c) (Int64) -> Void).self)
let waitSubmit = fn("mmc_wait_submit", (@convention(c) (Int64, Int64) -> Int32).self)
let gpuTimesTake = fn("mmc_gpu_times_take", (@convention(c) (UnsafeMutablePointer<Double>, Int32) -> Int32).self)
let lodOpen = fn("mmc_lod_open", (@convention(c) (UnsafePointer<CChar>, Int32, Int32, Int32) -> Int32).self)
let lodStatus = fn("mmc_lod_status", (@convention(c) (UnsafeMutablePointer<Int64>) -> Void).self)
let lodDraw = fn("mmc_lod_draw", (@convention(c) (UnsafePointer<Float>, UnsafePointer<Double>) -> Int32).self)
let lodSetLightmap = fn("mmc_lod_set_lightmap", (@convention(c) (Int64) -> Void).self)
let shadowsApply = fn("mmc_shadows_apply", (@convention(c) (Int64, Int64, UnsafePointer<Float>, UnsafePointer<Double>, Float, Float, Float, Int32) -> Int32).self)
let skyPrepare = fn("mmc_sky_prepare", (@convention(c) (UnsafePointer<Float>) -> Int32).self)
let skyDraw = fn("mmc_sky_draw", (@convention(c) (UnsafePointer<Float>) -> Int32).self)
let skyAerial = fn("mmc_sky_aerial", (@convention(c) (Int64, Int64, UnsafePointer<Float>, Int32) -> Int32).self)
let hdrSnapshot = fn("mmc_hdr_snapshot", (@convention(c) (Int64, Int64) -> Int32).self)

func check(_ ok: Bool, _ what: String) {
    print((ok ? "ok    " : "FAIL  ") + what)
    if !ok { exit(1) }
}
check(ctxInit() == 1, "context (METALMC_EXP=\(exp))")

// Lit entry points (absent from a build without Lit.swift: "off" doesn't need them).
typealias LitRelightF = @convention(c) (Int64, Int64, UnsafePointer<Float>, Int64, Int32) -> Int32
let litLevelPass = lit ? fn("mmc_lit_level_pass", (@convention(c) () -> Void).self) : nil
let litRelight = lit ? fn("mmc_lit_relight", LitRelightF.self) : nil
let litSetView = lit ? fn("mmc_lit_set_view", (@convention(c) (Int32) -> Void).self) : nil
let litSetNoAttach = lit ? fn("mmc_lit_set_no_attach", (@convention(c) (Int32) -> Void).self) : nil
let litCopyGbuffer = lit ? fn("mmc_lit_copy_gbuffer", (@convention(c) (Int64, Int64) -> Int32).self) : nil
let litTime = lit ? fn("mmc_debug_lit_time", (@convention(c) (Int64, Int64, UnsafePointer<Float>, Int64, Int32, UnsafeMutablePointer<Double>) -> Void).self) : nil
let litTimeTaa = lit ? fn("mmc_debug_lit_time_taa", (@convention(c) (Int64, Int64, UnsafePointer<Float>, Int64, Int32, UnsafeMutablePointer<Double>) -> Void).self) : nil
let taaApply = fn("mmc_taa_apply", (@convention(c) (Int64, Int64, UnsafePointer<Float>, UnsafePointer<Double>, Float, Float, Int32) -> Int32).self)
if lit { check(fn("mmc_lit_enabled", (@convention(c) () -> Int32).self)() == 1, "lit mode seen by the library") }

// The world's LOD around the camera.
let viewSpec = (ProcessInfo.processInfo.environment["LITFLOW_VIEW"] ?? "8,150,8,100,22").split(separator: ",").map { Double($0)! }
let cam = SIMD3<Double>(viewSpec[0], viewSpec[1], viewSpec[2])
let yaw = Float(viewSpec[3]) * .pi / 180, pitch = Float(viewSpec[4]) * .pi / 180
check(lodOpen(args[2], 2048, Int32(cam.x), Int32(cam.z)) == 1, "LOD opened on \(args[2])")
var st = [Int64](repeating: 0, count: 4)
var lastQuads: Int64 = -1, stable = 0
let t0 = Date()
while stable < 24 && Date().timeIntervalSince(t0) < 600 {
    lodStatus(&st)
    stable = st[0] == 2 && st[3] == 1 && st[2] == lastQuads ? stable + 1 : 0
    lastQuads = st[2]
    Thread.sleep(forTimeInterval: 0.25)
}
check(st[0] == 2, "LOD built: \(st[1]) nodes, \(st[2]) quads in \(Int(Date().timeIntervalSince(t0))) s")

// Vanilla's lightmap (lightmap.fsh): block light along x, sky light along y, at a sky factor (1 at noon, about 0.2 at
// midnight), the default brightness (0.5), no ambient (the overworld's).
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
func setLightmap(skyFactor: Float) {
    let px = lightmapPixels(skyFactor: skyFactor)
    UnsafeMutableRawPointer(bitPattern: Int(bufferContents(lmStaging)))!.copyMemory(from: px, byteCount: px.count)
    copyBufferToTexture(lmStaging, 0, 64, lightmap, 0, 0, 0, 0, 16, 16)
    submit(submitIndex); _ = waitSubmit(submitIndex, 5_000_000_000); submitIndex += 1
}
lodSetLightmap(lightmap)

// Vanilla-style pipelines (translated MSL, entry main0, an RGBA8 target declared): an "entity" (a quad 12 blocks in front
// of the camera, writing depth, in the middle of the screen) and a translucent band over the bottom fifth that tests
// depth but doesn't write it (like rain or block cracks).
let vsEntity = """
#include <metal_stdlib>
using namespace metal;
struct O { float4 p [[position]]; };
vertex O main0(uint vid [[vertex_id]]) {
    float2 c[6] = { float2(-0.15, -0.2), float2(0.15, -0.2), float2(0.15, 0.2), float2(-0.15, -0.2), float2(0.15, 0.2), float2(-0.15, 0.2) };
    O o; o.p = float4(c[vid].x, -c[vid].y, 0.05 / 12.0, 1.0); return o;
}
"""
let fsEntity = "#include <metal_stdlib>\nusing namespace metal;\nfragment float4 main0() { return float4(0.85, 0.1, 0.6, 1.0); }\n"
let vsBand = """
#include <metal_stdlib>
using namespace metal;
struct O { float4 p [[position]]; };
vertex O main0(uint vid [[vertex_id]]) {
    float2 c[6] = { float2(-1, -1), float2(1, -1), float2(1, -0.6), float2(-1, -1), float2(1, -0.6), float2(-1, -0.6) };
    O o; o.p = float4(c[vid].x, -c[vid].y, 1.0, 1.0); return o;
}
"""
let fsBand = "#include <metal_stdlib>\nusing namespace metal;\nfragment float4 main0() { return float4(0.75, 0.8, 0.9, 0.35); }\n"
var err = [CChar](repeating: 0, count: 512)
// topology, cull, wire, has depth, compare, write, bias, slope, masks, precreate, 1 target: format, mask, blend, ops, factors.
var entityParams: [Int32] = [3, 0, 0, 1, 6, 1, 0, 0, 0, 0, 0, 1, 70, 15, 0, 0, 0, 1, 0, 1, 0, 0, 0]
var bandParams: [Int32] = [3, 0, 0, 1, 7, 0, 0, 0, 0, 0, 0, 1, 70, 15, 1, 0, 0, 4, 5, 1, 5, 0, 0]
let entityPipe = pipelineCreate("entity", vsEntity, "main0", fsEntity, "main0", &entityParams, Int32(entityParams.count), &err, 512)
let bandPipe = pipelineCreate("band", vsBand, "main0", fsBand, "main0", &bandParams, Int32(bandParams.count), &err, 512)
check(entityPipe != 0 && bandPipe != 0, "vanilla-style pipelines created \(String(cString: err))")

// The camera: Minecraft's yaw and pitch, a view rotation looking down -z, an infinite reverse-Z projection like the game's.
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
    Targets(color: textureCreate(hdr ? 115 : 70, w, h, 1, 1, 1), depth: textureCreate(252, w, h, 1, 1, 1), w: w, h: h)
}

/// One frame as the game draws it. sunAngle: vanilla's (0 noon, pi midnight, -pi/2 sunrise in the east). Returns whether
/// the shadows traced and the relight ran.
@discardableResult
func frame(_ t: Targets, sunAngle: Float, relight: Bool, extras: Bool = true, taa: Bool = false, readback: ((Int64) -> Void)? = nil) -> (Bool, Bool) {
    var mats = matrices(width: t.w, height: t.h)
    // GameRenderer.renderLevel's start: the sky's tables (camera 87 m above sea level).
    if sky { var prep: [Float] = [sunAngle, 1, Float(cam.y) - 63, 0, 0]; _ = skyPrepare(&prep) }
    var h = t.color
    var clear: [Float] = [0.62, 0.75, 0.95, 1]
    _ = passBegin(&h, 1, 1, &clear, t.depth, 1, 0, 0, 0, t.w, t.h)
    if sky { _ = skyDraw(&mats) }
    passEnd()
    // The level's main pass, with the G-buffer.
    litLevelPass?()
    _ = passBegin(&h, 1, 0, &clear, t.depth, 0, 0, 0, 0, t.w, t.h)
    // mmc_lod_draw's p: projection, view, fog color, env start/end, render-distance start/end, discard radius, daylight.
    var p = mats + [0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, 0, 1]
    var c = [cam.x, cam.y, cam.z]
    _ = lodDraw(&p, &c)
    if extras {
        _ = setPipeline(entityPipe); draw(6, 1, 0, 0)
        _ = setPipeline(bandPipe); draw(6, 1, 0, 0)
    }
    passEnd()
    let traced = shadowsApply(t.color, t.depth, &mats, &c, sunAngle, 0.42, 192, 0) == 1
    var ran = false
    if relight, let litRelight {
        // Lit.relight's p: matrices, sun angle, fog color (strength 0: none), fog distances, our sky drew.
        var lp = mats + [sunAngle, 0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, sky ? 1 : 0]
        ran = litRelight(t.color, t.depth, &lp, lightmap, taa ? 1 : 0) == 1
    }
    if sky { _ = skyAerial(t.color, t.depth, &mats, taa ? 1 : 0) }
    if taa { var c2 = [cam.x, cam.y, cam.z]; _ = taaApply(t.color, t.depth, &mats, &c2, 0, 0, 0) }
    readback?(t.color)
    submit(submitIndex)
    if waitSubmit(submitIndex, 10_000_000_000) != 1 { check(false, "frame \(submitIndex) done") }
    submitIndex += 1
    return (traced, ran)
}

// A readback of the color (as 8-bit sRGB: with HDR through the screenshot copy), the depth and the G-buffer. The color is
// the mean of `frames` frames: the shadows' rays take one pixel of each 4 x 4 block per frame, in a 16-frame cycle, which
// the anti-aliasing accumulates in the game.
struct Readback { var color: [UInt8] = []; var depth: [Float] = []; var gbuf: [UInt32] = [] }
func capture(_ t: Targets, sunAngle: Float, relight: Bool, view: Int32 = 0, extras: Bool = true, frames: Int = 16, taa: Bool = false) -> Readback {
    let n = Int(t.w) * Int(t.h)
    let cbuf = bufferCreate(Int64(n * 4)), dbuf = bufferCreate(Int64(n * 4)), gbuf = bufferCreate(Int64(n * 8))
    let snap = hdr ? textureCreate(70, t.w, t.h, 1, 1, 1) : 0
    litSetView?(view)
    var gotG = false
    var sum = [Float](repeating: 0, count: n * 4)
    for k in 0..<frames {
        frame(t, sunAngle: sunAngle, relight: relight, extras: extras, taa: taa) { color in
            if hdr { _ = hdrSnapshot(color, snap) }
            copyToBuffer(hdr ? snap : color, 0, 0, 0, t.w, t.h, cbuf, 0, t.w * 4)
            if k == frames - 1 {
                copyToBuffer(t.depth, 0, 0, 0, t.w, t.h, dbuf, 0, t.w * 4)
                gotG = litCopyGbuffer?(gbuf, 0) == 1
            }
        }
        let cp = UnsafeRawPointer(bitPattern: Int(bufferContents(cbuf)))!.assumingMemoryBound(to: UInt8.self)
        for i in 0..<(n * 4) { sum[i] += Float(cp[i]) }
    }
    litSetView?(0)
    var r = Readback()
    r.color = sum.map { UInt8(min(255, ($0 / Float(frames)).rounded())) }
    let dp = UnsafeRawPointer(bitPattern: Int(bufferContents(dbuf)))!
    r.depth = Array(UnsafeBufferPointer(start: dp.assumingMemoryBound(to: Float.self), count: n))
    if gotG {
        let gp = UnsafeRawPointer(bitPattern: Int(bufferContents(gbuf)))!
        r.gbuf = Array(UnsafeBufferPointer(start: gp.assumingMemoryBound(to: UInt32.self), count: n * 2))
    }
    release(cbuf); release(dbuf); release(gbuf)
    if snap != 0 { release(snap) }
    return r
}

func writePNG(_ px: [UInt8], _ w: Int, _ h: Int, _ name: String) {
    var rgba = [UInt8](repeating: 255, count: w * h * 4)
    for y in 0..<h {   // flip: texture row 0 is the bottom
        for x in 0..<w { for k in 0..<3 { rgba[(y * w + x) * 4 + k] = px[((h - 1 - y) * w + x) * 4 + k] } }
    }
    let provider = CGDataProvider(data: Data(rgba) as CFData)!
    let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                      bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider, decode: nil,
                      shouldInterpolate: false, intent: .defaultIntent)!
    let dest = CGImageDestinationCreateWithURL(outDir.appendingPathComponent(name) as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
    print("      wrote \(name)")
}

// Warm-up at a small size: shaders compile in the background, the far field's rings fill, and the shadows' tile
// structures build (nearest first, about one a frame) until the traced count stops changing.
let small = targets(432, 280)
let noon: Float = 0, morning: Float = -0.9, dusk: Float = 1.43, midnight: Float = .pi
setLightmap(skyFactor: 1)
var tracedFrames = 0
for i in 0..<720 {
    let (traced, _) = frame(small, sunAngle: morning, relight: lit, extras: false)
    if traced { tracedFrames += 1 }
    if i % 120 == 119 { print("      warm-up frame \(i + 1): shadows traced in \(tracedFrames) frames") }
}
check(tracedFrames > 0, "ray-traced shadows ran during the warm-up")

let W: Int32 = 1728, H: Int32 = 1117   // half the panel each way, for the pictures
let half = targets(W, H)
let tag = mode
// Lit.swift's depth key and its match (bits 4-19 of the depth, within one step).
func depthMatches(_ depth: Float, _ key: UInt32) -> Bool {
    let d = ((depth.bitPattern >> 4) &- key) & 0xFFFF
    return d <= 1 || d == 0xFFFF
}
func lumOf(_ px: [UInt8], _ i: Int) -> Double { 0.2126 * Double(px[i]) + 0.7152 * Double(px[i + 1]) + 0.0722 * Double(px[i + 2]) }

// The forward frame (no relight) and the relit one, morning sun in the east (LITFLOW_TIMEONLY=1 skips to the timing).
let timeOnly = ProcessInfo.processInfo.environment["LITFLOW_TIMEONLY"] == "1"
let fwd = timeOnly ? Readback() : capture(half, sunAngle: morning, relight: false, frames: 1)
if !timeOnly { writePNG(fwd.color, Int(W), Int(H), "\(tag)-forward-morning.png") }
if lit && !timeOnly {
    let litM = capture(half, sunAngle: morning, relight: true)
    writePNG(litM.color, Int(W), Int(H), "\(tag)-lit-morning.png")
    // The G-buffer against the depth buffer: terrain pixels whose key matches the final depth (relit), those under the
    // entity quad (whose depth must not match), and elsewhere (water surfaces over meshed floors).
    let n = Int(W) * Int(H)
    var marked = 0, matched = 0, entityMarked = 0, entityMatched = 0, waterLike = 0, nearMiss = 0, nearMissMax = 0
    var nearHist = [Int](repeating: 0, count: 7)   // 1, 2, 3, 4, 5-8, 9-16, 17-64 steps
    var faces = [Int](repeating: 0, count: 8)
    var mismatchMask = [UInt8](repeating: 0, count: n * 4)
    // The entity quad's pixels, 3 in from its edges (pixels on its rim whose centers it misses still show terrain).
    let ex0 = Int(Float(W) * 0.425) + 3, ex1 = Int(Float(W) * 0.575) - 3, ey0 = Int(Float(H) * 0.4) + 3, ey1 = Int(Float(H) * 0.6) - 3
    for i in 0..<n {
        let gx = litM.gbuf[2 * i], gy = litM.gbuf[2 * i + 1]
        let code = Int(gx >> 29)
        faces[code] += 1
        if code == 0 { continue }
        marked += 1
        let b = litM.depth[i].bitPattern
        let x = i % Int(W), y = i / Int(W)
        let inEntity = x >= ex0 && x < ex1 && y >= ey0 && y < ey1
        if inEntity { entityMarked += 1 }
        if depthMatches(litM.depth[i], gy & 0xFFFF) {
            matched += 1
            if inEntity { entityMatched += 1 }
        } else if !inEntity {
            waterLike += 1
            // A near miss: the depth a few steps of its last bit away would have matched (precision, not occlusion).
            var near = 0
            for k in 1...64 {
                for s in [b &+ UInt32(k), b &- UInt32(k)] where depthMatches(Float(bitPattern: s), gy & 0xFFFF) { near = k }
                if near > 0 { break }
            }
            if near > 0 {
                nearMiss += 1
                nearMissMax = max(nearMissMax, near)
                nearHist[near <= 4 ? near - 1 : (near <= 8 ? 4 : (near <= 16 ? 5 : 6))] += 1
                mismatchMask[i * 4] = 255; mismatchMask[i * 4 + 1] = 255
            } else {
                mismatchMask[i * 4] = 255; mismatchMask[i * 4 + 2] = 255
            }
        } else {
            mismatchMask[i * 4 + 1] = 160
        }
    }
    print(String(format: "      G-buffer: %.1f%% of pixels lit terrain, %.1f%% of those still showing (relit); faces +X %d -X %d +Y %d -Y %d +Z %d -Z %d other %d",
                 100 * Double(marked) / Double(n), 100 * Double(matched) / Double(max(marked, 1)),
                 faces[1], faces[2], faces[3], faces[4], faces[5], faces[6], faces[7]))
    print("      under the entity quad: \(entityMarked) terrain pixels behind it, \(entityMatched) taken for terrain; elsewhere \(waterLike) mismatching (water drawn over its floor), \(nearMiss) of them within \(nearMissMax) steps of the depth's last bit (yellow in the mask)")
    print("      near misses by steps of the depth's last bit (1, 2, 3, 4, 5-8, 9-16, 17-64): \(nearHist)")
    writePNG(mismatchMask, Int(W), Int(H), "\(tag)-depth-mismatch.png")
    check(marked > n / 5, "the LOD wrote the G-buffer")
    // A 16-bit key matched within one step takes about three depth changes in 65,536 for a match.
    check(entityMarked > 1000 && entityMatched <= 2 + 8 * entityMarked / 65536, "terrain behind the entity is left alone (its depth doesn't match, bar key collisions)")
    // Debug views.
    for (v, name) in [(1, "albedo"), (2, "face"), (3, "levels"), (4, "ao"), (5, "light"), (6, "visibility"), (7, "overlay")] {
        let r = capture(half, sunAngle: morning, relight: true, view: Int32(v), frames: v == 5 || v == 6 ? 16 : 1)
        writePNG(r.color, Int(W), Int(H), "\(tag)-view-\(name).png")
        if v == 7 {
            // The overlay test: plain terrain (green) everywhere but under the translucent band (red), bottom fifth.
            var plainTop = 0, overlayTop = 0, plainBand = 0, overlayBand = 0
            for i in 0..<n {
                let red = r.color[4 * i], green = r.color[4 * i + 1]
                let y = i / Int(W)
                let inBand = y < Int(Float(H) * 0.2) - 2   // row 0 is the bottom
                let outside = y > Int(Float(H) * 0.2) + 2
                if green > 150 && red < 50 { if inBand { plainBand += 1 } else if outside { plainTop += 1 } }
                if red > 150 && green < 50 { if inBand { overlayBand += 1 } else if outside { overlayTop += 1 } }
            }
            print(String(format: "      overlay test: above the band %.2f%% of relit pixels flagged as overlaid; in the band %.1f%% flagged",
                         100 * Double(overlayTop) / Double(max(plainTop + overlayTop, 1)), 100 * Double(overlayBand) / Double(max(plainBand + overlayBand, 1))))
            check(Double(overlayTop) < 0.02 * Double(plainTop + overlayTop), "plain terrain passes the overlay test (under 2% flagged)")
            check(Double(overlayBand) > 0.9 * Double(plainBand + overlayBand), "the band (no depth write) is caught as an overlay")
        }
    }
    // Brightness against the forward frame: sunlit tops, sides, and the frame overall (8-bit luma, relit / forward).
    var sumF = 0.0, sumL = 0.0
    for i in stride(from: 0, to: n * 4, by: 4) { sumF += lumOf(fwd.color, i); sumL += lumOf(litM.color, i) }
    print(String(format: "      mean luma: forward %.1f, relit %.1f (morning)", sumF / Double(n), sumL / Double(n)))
    // The relight inside the anti-aliasing's resolve (its taaLit variant) against its own pass: 16 frames each, the
    // resolve's history against the mean of the pass's frames, over the frame (the resolve's sharpening and its 10% blend
    // account for a few levels).
    let litT = capture(half, sunAngle: morning, relight: true, frames: 24, taa: true)
    writePNG(litT.color, Int(W), Int(H), "\(tag)-lit-morning-taa.png")
    var diff = 0.0, big = 0
    for i in 0..<n {
        let d = abs(lumOf(litT.color, 4 * i) - lumOf(litM.color, 4 * i))
        diff += d
        if d > 24 { big += 1 }
    }
    print(String(format: "      relight in the anti-aliasing's resolve vs its own pass: mean luma difference %.2f, %.2f%% of pixels over 24 levels",
                 diff / Double(n), 100 * Double(big) / Double(n)))
    check(diff / Double(n) < 4, "the relight in the anti-aliasing's resolve matches its own pass (mean difference under 4 levels)")
    // Times of day.
    for (name, angle, skyFactor) in [("noon", noon, Float(1)), ("dusk", dusk, Float(0.55)), ("midnight", midnight, Float(0.2))] {
        setLightmap(skyFactor: skyFactor)
        for _ in 0..<16 { frame(small, sunAngle: angle, relight: true, extras: false) }   // the shadows' accumulation isn't kept, but the sky tables are
        let f = capture(half, sunAngle: angle, relight: false, frames: 1)
        let l = capture(half, sunAngle: angle, relight: true)
        writePNG(f.color, Int(W), Int(H), "\(tag)-forward-\(name).png")
        writePNG(l.color, Int(W), Int(H), "\(tag)-lit-\(name).png")
        // Tops (+Y faces in the G-buffer, relit) against the same pixels forward.
        var tf = 0.0, tl = 0.0, tn = 0, sf = 0.0, sl = 0.0, sn = 0
        for i in 0..<n {
            let gx = l.gbuf[2 * i], gy = l.gbuf[2 * i + 1]
            let code = gx >> 29
            guard code != 0, depthMatches(l.depth[i], gy & 0xFFFF), i / Int(W) > Int(Float(H) * 0.2) + 2 else { continue }
            if code == 3 { tf += lumOf(f.color, 4 * i); tl += lumOf(l.color, 4 * i); tn += 1 }
            else if code != 4 && code != 7 { sf += lumOf(f.color, 4 * i); sl += lumOf(l.color, 4 * i); sn += 1 }
        }
        print(String(format: "      %@: tops (%d px) mean luma forward %.1f relit %.1f; sides (%d px) forward %.1f relit %.1f",
                     name, tn, tf / Double(max(tn, 1)), tl / Double(max(tn, 1)), sn, sf / Double(max(sn, 1)), sl / Double(max(sn, 1))))
    }
    setLightmap(skyFactor: 1)
}

// Timing at the panel's resolution (3456 x 2234): the relight alone, and the main pass with and without the G-buffer.
if timing {
    let full = targets(3456, 2234)
    for _ in 0..<8 { frame(full, sunAngle: morning, relight: lit, extras: false) }
    // Other work on the machine shares the GPU and stretches some command buffers: medians and the fastest (closer to a
    // pass's own cost) are both given, and A/B pairs alternate.
    var times = [Double](repeating: 0, count: 4096)
    func mainPassTime(attach: Bool) -> Double {
        _ = gpuTimesTake(&times, 4096)
        let mats = matrices(width: full.w, height: full.h)
        var h = full.color
        var clear: [Float] = [0.62, 0.75, 0.95, 1]
        litSetNoAttach?(attach ? 0 : 1)
        _ = passBegin(&h, 1, 1, &clear, full.depth, 1, 0, 0, 0, full.w, full.h)
        passEnd()
        litLevelPass?()
        _ = passBegin(&h, 1, 0, &clear, full.depth, 0, 0, 0, 0, full.w, full.h)
        var p = mats + [0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, 0, 1]
        var c = [cam.x, cam.y, cam.z]
        _ = lodDraw(&p, &c)
        passEnd()
        submit(submitIndex); _ = waitSubmit(submitIndex, 10_000_000_000); submitIndex += 1
        litSetNoAttach?(0)
        let k = gpuTimesTake(&times, 4096)
        return k > 0 ? times[Int(k) - 1] * 1000 : -1
    }
    func stats(_ s: [Double]) -> (Double, Double) {
        let t = s.sorted()
        return t.isEmpty ? (-1, -1) : (t[t.count / 2], t[0])
    }
    for round in 0..<2 {
        var with: [Double] = [], without: [Double] = []
        for _ in 0..<60 {
            if lit { with.append(mainPassTime(attach: true)) }
            without.append(mainPassTime(attach: false))
        }
        let (wm, wf) = stats(with), (om, of) = stats(without)
        if lit {
            print(String(format: "      round %d: sky clear + main pass (LOD) at 3456 x 2234: with the G-buffer %.3f ms (fastest %.3f), without %.3f (fastest %.3f): +%.3f (fastest +%.3f), 60 each",
                         round, wm, wf, om, of, wm - om, wf - of))
        } else {
            print(String(format: "      round %d: sky clear + main pass (LOD) at 3456 x 2234, no lit mode: %.3f ms (fastest %.3f), 60 runs", round, om, of))
        }
    }
    if lit, let litTime {
        frame(full, sunAngle: morning, relight: false, extras: false)
        let mats = matrices(width: full.w, height: full.h)
        // Leave the G-buffer of one more main pass for the timed relights.
        litLevelPass?()
        var h = full.color
        var clear: [Float] = [0.62, 0.75, 0.95, 1]
        _ = passBegin(&h, 1, 0, &clear, full.depth, 0, 0, 0, 0, full.w, full.h)
        var p = mats + [0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, 0, 1]
        var c = [cam.x, cam.y, cam.z]
        _ = lodDraw(&p, &c)
        passEnd()
        submit(submitIndex); _ = waitSubmit(submitIndex, 10_000_000_000); submitIndex += 1
        var lp = mats + [morning, 0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, sky ? 1 : 0]
        var two = [Double](repeating: 0, count: 2)
        for round in 0..<2 {
            litTime(full.color, full.depth, &lp, lightmap, 60, &two)
            print(String(format: "      round %d: relight pass at 3456 x 2234 (%@ target): %.3f ms (fastest %.3f), 60 runs", round, hdr ? "RGBA16Float" : "RGBA8", two[0], two[1]))
        }
        if let litTimeTaa {
            var out = [Double](repeating: 0, count: 4)
            for round in 0..<2 {
                litTimeTaa(full.color, full.depth, &lp, lightmap, 60, &out)
                print(String(format: "      round %d: anti-aliasing resolve at 3456 x 2234 (%@ target): with the relight in its load %.3f ms (fastest %.3f), without %.3f (fastest %.3f): +%.3f (fastest +%.3f), 60 each",
                             round, hdr ? "RGBA16Float" : "RGBA8", out[0], out[2], out[1], out[3], out[0] - out[1], out[2] - out[3]))
            }
        }
    }
}
print("done")
