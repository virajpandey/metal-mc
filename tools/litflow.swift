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
//   LITFLOW_GI=1 adds the GI cache (METALMC_EXP=...,gi, Gi.swift): pictures and numbers of lit mode with and without the
//   cache's light in the same frames (mmc_debug_lit_gi), morning, noon and dusk, and with "time" the frame with and
//   without the cache's frame (mmc_debug_gi_frame). Pictures are named <mode>gi-... then.
//   LITFLOW_WATER=1 adds water (METALMC_EXP=...,water: its reflections, Lit.swift) and runs only the water section: per
//   time of day (LITFLOW_WATERTIMES, default noon,dusk,sunset; also morning, low (sun 7 degrees up in the east), midnight)
//   the relit frame with the reflections off and on in the same build (mmc_debug_lit_water; with a library without it only
//   "off", for before/after against an older build), the waves held at a time that steps 1/120 s a frame, the water
//   pixels' numbers, the debug views (9: which pixels reflect, 10: the reflections alone) and with the anti-aliasing; with
//   "time" the reflections' cost in the resolve and in the relight's own pass (without LITFLOW_WATER, the same passes as
//   the baseline). LITFLOW_NAME names the view in the pictures (<mode>-water-<name>-<time>-off.png, -on.png, -on-taa.png,
//   -reflections.png, -changed-dry.png if anything but water changed; <mode>-water-<name>-view-water.png).
//   litflow <dylib> - - compile: compiles every shader variant of the anti-aliasing's resolve, the far field and lit mode's
//   own pass under METALMC_EXP=$LITFLOW_EXP (default lit,rtshadows,sky,water), then exits (no world, no GPU work).
import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

let args = CommandLine.arguments
guard args.count >= 4 else { print("usage: litflow <dylib> <world dir> <output dir> [sdr|sky|hdr|off|compile] [time]"); exit(1) }
let mode = args.count >= 5 ? args[4] : "sdr"
let timing = args.count >= 6 && args[5] == "time"
let giOn = ProcessInfo.processInfo.environment["LITFLOW_GI"] == "1" && mode != "off"
let waterOn = ProcessInfo.processInfo.environment["LITFLOW_WATER"] == "1" && mode != "off"
let exp = mode == "compile" ? (ProcessInfo.processInfo.environment["LITFLOW_EXP"] ?? "lit,rtshadows,sky,water")
    : (mode == "off" ? "rtshadows" : (mode == "sky" ? "lit,rtshadows,sky" : (mode == "hdr" ? "lit,rtshadows,sky,hdr" : "lit,rtshadows")))
        + (giOn ? ",gi" : "") + (waterOn ? ",water" : "")
setenv("METALMC_EXP", exp, 1)
let lit = mode != "off", sky = mode == "sky" || mode == "hdr", hdr = mode == "hdr"
guard let lib = dlopen(args[1], RTLD_NOW) else { print("dlopen failed"); exit(1) }
let outDir = URL(fileURLWithPath: args[3])
if mode != "compile" { try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true) }
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

// Mode "compile": every variant of the shaders lit mode touches compiles with the device's compiler, as the library would
// build them under this METALMC_EXP: the anti-aliasing's resolve (both tile sizes; plain, sky, lit, lit and sky), the far
// field (its march for the level pass's targets, its compute kernels) and lit mode's own pass (each target format) and
// sun and sky kernel.
if mode == "compile" {
    let device = MTLCreateSystemDefaultDevice()!
    func source(_ name: String) -> String? {
        guard let p = dlsym(lib, name) else { return nil }
        let f = unsafeBitCast(p, to: (@convention(c) (UnsafeMutablePointer<CChar>, Int32) -> Int32).self)
        var buf = [CChar](repeating: 0, count: 1 << 21)
        let n = f(&buf, Int32(buf.count))
        return n > 0 && Int(n) < buf.count ? String(cString: buf) : nil
    }
    let litOn = exp.split(separator: ",").contains("lit")
    var failures = 0
    func attempt(_ what: String, _ body: () throws -> Void) {
        do { try body(); print("ok    \(what)") } catch { print("FAIL  \(what): \(error)"); failures += 1 }
    }
    if let src = source("mmc_debug_taa_shader_source") {
        for tile in [32, 16] {
            let opts = MTLCompileOptions()
            opts.preprocessorMacros = ["TAA_T": NSNumber(value: tile)]
            for (skyV, litV) in [(false, false), (true, false), (false, true), (true, true)] where litOn || !litV {
                attempt("anti-aliasing resolve, \(tile) x \(tile) tiles\(skyV ? ", sky" : "")\(litV ? ", lit" : "")") {
                    let l = try device.makeLibrary(source: src, options: opts)
                    let v = MTLFunctionConstantValues()
                    var s = skyV, li = litV
                    v.setConstantValue(&s, type: .bool, index: 0)
                    if litOn { v.setConstantValue(&li, type: .bool, index: 1) }
                    _ = try device.makeComputePipelineState(function: try l.makeFunction(name: "taa_resolve", constantValues: v))
                }
            }
        }
    } else { print("FAIL  no anti-aliasing source"); failures += 1 }
    if let src = source("mmc_debug_far_shader_source") {
        attempt("far field: march (level pass targets\(litOn ? " with the G-buffer" : "")) and kernels") {
            let l = try device.makeLibrary(source: src, options: nil)
            for k in ["ff_clear", "ff_fill", "ff_mip", "ff_profile"] { _ = try device.makeComputePipelineState(function: l.makeFunction(name: k)!) }
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = l.makeFunction(name: "ff_vs")
            d.fragmentFunction = l.makeFunction(name: "ff_fs")
            d.colorAttachments[0].pixelFormat = .rgba8Unorm
            if litOn { d.colorAttachments[1].pixelFormat = .rg32Uint }
            d.depthAttachmentPixelFormat = .depth32Float
            _ = try device.makeRenderPipelineState(descriptor: d)
        }
    } else { print("FAIL  no far field source"); failures += 1 }
    if litOn {
        if let src = source("mmc_debug_lit_shader_source") {
            for format in [MTLPixelFormat.rgba8Unorm, .rgba16Float, .rg11b10Float] {
                attempt("lit mode's relight pass (target format \(format.rawValue)), sun and sky kernel\(src.contains("lit_water_waves") ? ", water's waves and sky map kernels" : "")") {
                    let l = try device.makeLibrary(source: src, options: nil)
                    _ = try device.makeComputePipelineState(function: l.makeFunction(name: "lit_env")!)
                    for k in ["lit_water_waves", "lit_water_sky"] {
                        if let w = l.makeFunction(name: k) { _ = try device.makeComputePipelineState(function: w) }
                    }
                    let d = MTLRenderPipelineDescriptor()
                    d.vertexFunction = l.makeFunction(name: "lit_fullscreen_vs")
                    d.fragmentFunction = l.makeFunction(name: "lit_relight_fs")
                    d.colorAttachments[0].pixelFormat = format
                    _ = try device.makeRenderPipelineState(descriptor: d)
                }
            }
        } else { print("      (no lit mode source in this library: an older build)") }
    }
    print(failures == 0 ? "done: every variant compiles" : "FAIL  \(failures) failed")
    exit(failures == 0 ? 0 : 1)
}

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
// The GI cache's switches (LITFLOW_GI=1): the relight's use of its light, and its frame in RtShadows.
let litGiSwitch = giOn ? fn("mmc_debug_lit_gi", (@convention(c) (Int32) -> Int32).self) : nil
let giFrameSwitch = giOn ? fn("mmc_debug_gi_frame", (@convention(c) (Int32) -> Void).self) : nil
if let litGiSwitch { check(litGiSwitch(1) == 1, "the GI cache is wired into lit mode (lit,gi)") }
// Water's switches (LITFLOW_WATER=1; absent from a library from before water, which then draws the "off" pictures): the
// reflections on or off, and the waves' time, which frame() steps by 1/120 s a frame so pictures repeat.
let litWaterSwitch = waterOn ? dlsym(lib, "mmc_debug_lit_water").map { unsafeBitCast($0, to: (@convention(c) (Int32, Double) -> Int32).self) } : nil
// The resolve and the relight's own pass timed with the reflections on and off (without water both are the same passes:
// the baseline for water's whole cost).
let litWaterTime = lit ? dlsym(lib, "mmc_debug_lit_water_time").map {
    unsafeBitCast($0, to: (@convention(c) (Int64, Int64, UnsafePointer<Float>, Int64, Int32, UnsafeMutablePointer<Double>) -> Int32).self) } : nil
if waterOn { print(litWaterSwitch.map { $0(1, 100) == 1 ? "ok    water is on (lit,water)" : "FAIL  water isn't on" } ?? "      no water in this library: its pictures are the reflections off") }
var reflectWater = true
var framesDrawn = 0

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
    // Water: the reflections on or off, the waves 1/120 s on from the last frame (as at 120 Hz).
    if let litWaterSwitch { _ = litWaterSwitch(reflectWater ? 1 : 0, 100 + Double(framesDrawn) / 120) }
    framesDrawn += 1
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
let tag = mode + (giOn ? "gi" : "")
// Lit.swift's depth key and its match (bits 4-19 of the depth, within one step).
func depthMatches(_ depth: Float, _ key: UInt32) -> Bool {
    let d = ((depth.bitPattern >> 4) &- key) & 0xFFFF
    return d <= 1 || d == 0xFFFF
}
func lumOf(_ px: [UInt8], _ i: Int) -> Double { 0.2126 * Double(px[i]) + 0.7152 * Double(px[i + 1]) + 0.0722 * Double(px[i + 2]) }

// The forward frame (no relight) and the relit one, morning sun in the east (LITFLOW_TIMEONLY=1 skips to the timing;
// LITFLOW_GIONLY=1 to the GI cache's comparison).
let timeOnly = ProcessInfo.processInfo.environment["LITFLOW_TIMEONLY"] == "1"
let giOnly = giOn && ProcessInfo.processInfo.environment["LITFLOW_GIONLY"] == "1"
let fwd = timeOnly || giOnly || waterOn ? Readback() : capture(half, sunAngle: morning, relight: false, frames: 1)
if !timeOnly && !giOnly && !waterOn { writePNG(fwd.color, Int(W), Int(H), "\(tag)-forward-morning.png") }
if lit && !timeOnly && !giOnly && !waterOn {
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

// Water (LITFLOW_WATER=1): per time of day the relit frame (no extras) with the reflections off and on in the same build,
// with the anti-aliasing, the reflections alone (debug view 10), and the water pixels' numbers: how many the G-buffer
// flags with a matching depth (the far field's water now; the LOD's quads' once its water pipeline writes the flag), and
// their mean 8-bit luma off and on, by the upper and lower half of the frame (far and near water).
if waterOn && !timeOnly {
    let n = Int(W) * Int(H)
    let name = ProcessInfo.processInfo.environment["LITFLOW_NAME"] ?? "view"
    // Sun angles (vanilla's) and the lightmap's daylight for them: sunset has the sun 2 degrees up in the west, low 7 up in
    // the east.
    let suns: [String: (Float, Float)] = ["noon": (noon, 1), "morning": (morning, 1), "dusk": (dusk, 0.55), "sunset": (1.53, 0.4),
                                          "low": (-1.45, 0.6), "midnight": (midnight, 0.2)]
    let list = (ProcessInfo.processInfo.environment["LITFLOW_WATERTIMES"] ?? "noon,dusk,sunset").split(separator: ",").map(String.init)
    func waterPixels(_ r: Readback) -> [Int] {
        guard !r.gbuf.isEmpty else { return [] }
        return (0..<n).filter { (r.gbuf[2 * $0] >> 28) == 1 && depthMatches(r.depth[$0], r.gbuf[2 * $0 + 1] & 0xFFFF) }
    }
    func meanLuma(_ px: [UInt8], _ idx: [Int]) -> Double { idx.isEmpty ? 0 : idx.reduce(0.0) { $0 + lumOf(px, 4 * $1) } / Double(idx.count) }
    for t in list {
        guard let (angle, skyFactor) = suns[t] else { print("      unknown time \(t)"); continue }
        setLightmap(skyFactor: skyFactor)
        for _ in 0..<16 { frame(small, sunAngle: angle, relight: true, extras: false) }   // the sky's tables for this sun
        reflectWater = false
        let off = capture(half, sunAngle: angle, relight: true, extras: false)
        writePNG(off.color, Int(W), Int(H), "\(tag)-water-\(name)-\(t)-off.png")
        guard litWaterSwitch != nil else { continue }
        // 48 frames more, so the "on" frames have the same places in every 64-frame cycle (the shadows' sun disk samples,
        // the sky's dither) as the "off" ones: then only water may differ between the two.
        for _ in 0..<48 { frame(small, sunAngle: angle, relight: true, extras: false) }
        reflectWater = true
        let on = capture(half, sunAngle: angle, relight: true, extras: false)
        writePNG(on.color, Int(W), Int(H), "\(tag)-water-\(name)-\(t)-on.png")
        let refl = capture(half, sunAngle: angle, relight: true, view: 10, extras: false)
        writePNG(refl.color, Int(W), Int(H), "\(tag)-water-\(name)-\(t)-reflections.png")
        let onT = capture(half, sunAngle: angle, relight: true, extras: false, frames: 24, taa: true)
        writePNG(onT.color, Int(W), Int(H), "\(tag)-water-\(name)-\(t)-on-taa.png")
        let water = waterPixels(on)
        let far = water.filter { $0 / Int(W) >= Int(H) / 2 }, near = water.filter { $0 / Int(W) < Int(H) / 2 }   // row 0 is the bottom
        // Pixels that changed and aren't water: none but where something else changed between the two captures. The
        // shadows' tile structures still build in the background after the warm-up (nearest first, about one a frame, and
        // the far field's view has thousands), so far terrain can take a new shadow in between: a few hundred pixels of
        // it, measured, all far land near the horizon. Marked in a picture (white on the dimmed "on" frame) when there are
        // any; a change that touched anything but water would show over most of the frame.
        var isWater = [Bool](repeating: false, count: n)
        for i in water { isWater[i] = true }
        var changed = 0, changedDry = 0
        var dryMask = on.color.map { $0 / 4 }
        for i in 0..<n where on.color[4 * i] != off.color[4 * i] || on.color[4 * i + 1] != off.color[4 * i + 1] || on.color[4 * i + 2] != off.color[4 * i + 2] {
            changed += 1
            if !isWater[i] { changedDry += 1; for k in 0..<3 { dryMask[4 * i + k] = 255 } }
        }
        print(String(format: "      %@ %@: water %.1f%% of the frame (upper half %d px, lower %d); mean luma off -> on: all %.1f -> %.1f, upper %.1f -> %.1f, lower %.1f -> %.1f; %d pixels changed, %d of them not water",
                     name, t, 100 * Double(water.count) / Double(n), far.count, near.count, meanLuma(off.color, water), meanLuma(on.color, water),
                     meanLuma(off.color, far), meanLuma(on.color, far), meanLuma(off.color, near), meanLuma(on.color, near), changed, changedDry))
        if changedDry > 0 { writePNG(dryMask, Int(W), Int(H), "\(tag)-water-\(name)-\(t)-changed-dry.png") }
        check(water.count > 0, "the G-buffer flags water (\(name), \(t))")
        check(changedDry * 200 < n, "the reflections change water pixels only (but for under 0.5% of the frame, far shadows: \(name), \(t))")
    }
    if litWaterSwitch != nil {
        let cover = capture(half, sunAngle: suns[list.first ?? "noon"]?.0 ?? noon, relight: true, view: 9, extras: false, frames: 1)
        writePNG(cover.color, Int(W), Int(H), "\(tag)-water-\(name)-view-water.png")
    }
    reflectWater = true
    setLightmap(skyFactor: 1)
}

// The GI cache (LITFLOW_GI=1): lit mode with and without the cache's light in place of its sky term, in the same frames
// (the cache keeps running either way; mmc_debug_lit_gi picks the relight's sky term). Per time of day: the pictures
// (means of 16 frames, like the others), the mean 8-bit luma of the relit terrain by kind (shaded: faces the sun doesn't
// light, turned away from it or edge-on, which only the sky term lights; sunlit tops), how much of the relit terrain the
// cache covered (debug view 8), and the frame-to-frame change of single frames on the shaded faces (the cache's noise:
// without it their light doesn't change between frames).
if let litGiSwitch, !timeOnly, !waterOn {
    let n = Int(W) * Int(H)
    let normals: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)]
    // Relit pixels above the translucent band (none is drawn here: extras off) whose G-buffer key matches, by kind.
    func kinds(_ r: Readback, sunAngle: Float) -> (shaded: [Int], tops: [Int], all: [Int]) {
        let sun = SIMD3<Float>(-sin(sunAngle), cos(sunAngle), 0)
        var shaded: [Int] = [], tops: [Int] = [], all: [Int] = []
        for i in 0..<n {
            let gx = r.gbuf[2 * i], gy = r.gbuf[2 * i + 1]
            let code = Int(gx >> 29)
            guard code != 0, code != 7, depthMatches(r.depth[i], gy & 0xFFFF) else { continue }
            // Light sources (block light 15) are full bright either way.
            if (gy >> 24) >= 240 { continue }
            all.append(i)
            let ndl = simd_dot(normals[code - 1], sun)
            if ndl < 0.05 { shaded.append(i) } else if code == 3 { tops.append(i) }
        }
        return (shaded, tops, all)
    }
    func meanLuma(_ px: [UInt8], _ idx: [Int]) -> Double { idx.isEmpty ? 0 : idx.reduce(0.0) { $0 + lumOf(px, 4 * $1) } / Double(idx.count) }
    // Mean absolute luma change between two frames over idx, relative to their mean luma.
    func change(_ a: [UInt8], _ b: [UInt8], _ idx: [Int]) -> Double {
        var d = 0.0, s = 0.0
        for i in idx { d += abs(lumOf(a, 4 * i) - lumOf(b, 4 * i)); s += lumOf(a, 4 * i) }
        return s > 0 ? d / s : 0
    }
    for (name, angle, skyFactor) in [("morning", morning, Float(1)), ("noon", noon, Float(1)), ("dusk", dusk, Float(0.55))] {
        setLightmap(skyFactor: skyFactor)
        // The cache follows the new sun over its history (64 updates): 160 frames at the pictures' size.
        _ = litGiSwitch(1)
        for _ in 0..<160 { frame(half, sunAngle: angle, relight: true, extras: false) }
        _ = litGiSwitch(0)
        let without = capture(half, sunAngle: angle, relight: true, extras: false)
        let w1 = capture(half, sunAngle: angle, relight: true, extras: false, frames: 1)
        let w2 = capture(half, sunAngle: angle, relight: true, extras: false, frames: 1)
        _ = litGiSwitch(1)
        let with = capture(half, sunAngle: angle, relight: true, extras: false)
        let g1 = capture(half, sunAngle: angle, relight: true, extras: false, frames: 1)
        let g2 = capture(half, sunAngle: angle, relight: true, extras: false, frames: 1)
        let cover = capture(half, sunAngle: angle, relight: true, view: 8, extras: false, frames: 1)
        writePNG(without.color, Int(W), Int(H), "\(tag)-lit-\(name).png")
        writePNG(with.color, Int(W), Int(H), "\(tag)-lit-gi-\(name).png")
        writePNG(cover.color, Int(W), Int(H), "\(tag)-view-gi-coverage-\(name).png")
        let k = kinds(with, sunAngle: angle)
        var green = 0, red = 0
        for i in k.all {
            if cover.color[4 * i + 1] > 150 && cover.color[4 * i] < 50 { green += 1 } else if cover.color[4 * i] > 150 { red += 1 }
        }
        print(String(format: "      %@: shaded faces (%d px) mean luma lit %.1f, lit+gi %.1f; sunlit tops (%d px) %.1f, %.1f; all relit terrain %.1f, %.1f",
                     name, k.shaded.count, meanLuma(without.color, k.shaded), meanLuma(with.color, k.shaded), k.tops.count,
                     meanLuma(without.color, k.tops), meanLuma(with.color, k.tops), meanLuma(without.color, k.all), meanLuma(with.color, k.all)))
        print(String(format: "      %@: the cache's light on %.1f%% of relit terrain (%.1f%% without data); frame-to-frame change of shaded faces: lit %.2f%%, lit+gi %.2f%%",
                     name, 100 * Double(green) / Double(max(k.all.count, 1)), 100 * Double(red) / Double(max(k.all.count, 1)),
                     100 * change(w1.color, w2.color, k.shaded), 100 * change(g1.color, g2.color, k.shaded)))
    }
    // A jump in the time of day (/time set): noon, settled, then dusk at once. The shaded faces' mean luma per frame after
    // the jump against where it settles 192 frames later, and the relight without the cache (its sky term follows at
    // once). The cells hold light per unit of the frame's daylight, so the change of brightness and color shows at once;
    // what lags is the new sun direction's pattern of bounced light, over the cells' history.
    setLightmap(skyFactor: 1)
    for _ in 0..<160 { frame(half, sunAngle: noon, relight: true, extras: false) }
    setLightmap(skyFactor: 0.55)
    var trace: [(Int, Double)] = []
    var ks: [Int] = []
    for f in 0..<64 {
        let r = capture(half, sunAngle: dusk, relight: true, extras: false, frames: 1)
        if f == 0 { ks = kinds(r, sunAngle: dusk).shaded }
        if [0, 1, 2, 4, 8, 16, 32, 63].contains(f) { trace.append((f, meanLuma(r.color, ks))) }
    }
    for _ in 0..<128 { frame(half, sunAngle: dusk, relight: true, extras: false) }
    let settled = meanLuma(capture(half, sunAngle: dusk, relight: true, extras: false, frames: 1).color, ks)
    _ = litGiSwitch(0)
    let skyTerm = meanLuma(capture(half, sunAngle: dusk, relight: true, extras: false, frames: 1).color, ks)
    _ = litGiSwitch(1)
    print("      noon -> dusk at once: shaded faces' mean luma by frame after the jump " + trace.map { String(format: "%d: %.1f", $0.0, $0.1) }.joined(separator: ", ")
          + String(format: "; settled %.1f (the sky term without the cache: %.1f)", settled, skyTerm))
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
        // Water: the reflections' cost on this frame's G-buffer, on and off alternating, in the anti-aliasing's resolve
        // (with the relight and the aerial perspective in its load, as the game runs them with the sky) and in the relight's
        // own pass. Without water the same passes once, for the baseline.
        if let litWaterTime, sky {
            var out = [Double](repeating: 0, count: 8)
            for round in 0..<2 {
                guard litWaterTime(full.color, full.depth, &lp, lightmap, 60, &out) == 1 else { print("FAIL  water timing"); break }
                if waterOn {
                    print(String(format: "      round %d: water at 3456 x 2234 (%@ target): the anti-aliasing resolve with the reflections %.3f ms (fastest %.3f), without %.3f (fastest %.3f): +%.3f (fastest +%.3f); the relight's own pass %.3f (fastest %.3f) against %.3f (fastest %.3f): +%.3f (fastest +%.3f); 60 each",
                                 round, hdr ? "RGBA16Float" : "RGBA8", out[0], out[2], out[1], out[3], out[0] - out[1], out[2] - out[3],
                                 out[4], out[6], out[5], out[7], out[4] - out[5], out[6] - out[7]))
                } else {
                    print(String(format: "      round %d: no water, at 3456 x 2234 (%@ target): the anti-aliasing resolve with the relight and the aerial perspective %.3f ms (fastest %.3f); the relight's own pass %.3f (fastest %.3f); 120 each",
                                 round, hdr ? "RGBA16Float" : "RGBA8", (out[0] + out[1]) / 2, min(out[2], out[3]), (out[4] + out[5]) / 2, min(out[6], out[7])))
                }
            }
        }
    }
    if let giFrameSwitch {
        // The GI cache: whole frames at the panel's resolution (sky clear, main pass, shadow rays, the cache's frame, the
        // relight in its own pass or in the anti-aliasing's resolve) with and without the cache's frame (without it the
        // relight has no cache light, so its upsample is skipped too), alternating, after 16 frames to settle.
        for taa in [false, true] {
            for _ in 0..<16 { frame(full, sunAngle: morning, relight: true, extras: false, taa: taa) }
            var with: [Double] = [], without: [Double] = []
            for k in 0..<120 {
                let on = k % 2 == 0
                giFrameSwitch(on ? 1 : 0)
                _ = gpuTimesTake(&times, 4096)
                frame(full, sunAngle: morning, relight: true, extras: false, taa: taa)
                let c = Int(gpuTimesTake(&times, 4096))
                if c > 0 { if on { with.append(times[c - 1] * 1000) } else { without.append(times[c - 1] * 1000) } }
            }
            giFrameSwitch(1)
            let (wm, wf) = stats(with), (om, of) = stats(without)
            print(String(format: "      frame at 3456 x 2234 (%@, relight %@): with the GI cache %.3f ms (fastest %.3f), without %.3f (fastest %.3f): +%.3f (fastest +%.3f), 60 each",
                         hdr ? "RGBA16Float" : "RGBA8", taa ? "in the anti-aliasing's resolve" : "its own pass", wm, wf, om, of, wm - om, wf - of))
        }
    }
}
print("done")
