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
//   "cull" instead of "time": the LOD's quad culling (METALMC_EXP=quadcull, Lod.swift) checked and timed: the main pass
//   bit for bit with and without it in several views, and its time at the panel's resolution (see the "cull" section
//   below; LITFLOW_VIEWS=x,y,z,yaw,pitch;... sets the views, LITFLOW_CULLREF=<file> checks the unculled picture against
//   another build's).
//   LITFLOW_GATE=<dir>: take the GPU lock only after the LOD's build (see there).
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
//   LITFLOW_CL=1 adds colored block light (METALMC_EXP=...,coloredlight, ColoredLight.swift) and runs only its section:
//   the blocks around the camera from the world's region files (as the game's Java side sends them), the test gallery
//   (LITFLOW_CLSCENE, default mod/src/client/resources/metalmc/coloredlight_scene.txt; "none" for the world as it is)
//   applied to them and to the LOD, then per view (LITFLOW_CLVIEWS=name:x,y,z,yaw,pitch;...; default the gallery from
//   above and low over its mixing area, and the default view's real terrain) and time of day (LITFLOW_CLTIMES, default midnight,noon) the relit frame with
//   vanilla's block light and with the colored light (cl-<view>-<time>-vanilla.png, -colored.png, -colored-taa.png),
//   the light alone and the sanity check's view (-light.png, -check.png), its numbers, the flood fill checked against
//   a CPU reference, a block change, and with "time" its cost (the volume's work from scratch, still, moving, after an
//   edit; the relight and the anti-aliasing's resolve with and without it). Set LITFLOW_VIEW to the first view (the LOD
//   is opened around it).
import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

let args = CommandLine.arguments
guard args.count >= 4 else { print("usage: litflow <dylib> <world dir> <output dir> [sdr|sky|hdr|off|compile] [time|cull]"); exit(1) }
let mode = args.count >= 5 ? args[4] : "sdr"
let timing = args.count >= 6 && args[5] == "time"
let cullRun = args.count >= 6 && args[5] == "cull"
let giOn = ProcessInfo.processInfo.environment["LITFLOW_GI"] == "1" && mode != "off"
let waterOn = ProcessInfo.processInfo.environment["LITFLOW_WATER"] == "1" && mode != "off"
// LITFLOW_POST=1 (with sky or hdr): the post chain (METALMC_EXP=...,post, Post.swift) after the anti-aliasing; runs only the
// post section (pictures of each effect at several times of day, the exposure's numbers; with "time" each stage's cost).
let postOn = ProcessInfo.processInfo.environment["LITFLOW_POST"] == "1" && (mode == "sky" || mode == "hdr")
let clOn = ProcessInfo.processInfo.environment["LITFLOW_CL"] == "1" && mode != "off" && mode != "compile"
// LITFLOW_CLOUDS=1 (with sky or hdr): the volumetric clouds (METALMC_EXP=...,clouds, Clouds.swift) before the shadows; runs
// only the clouds section (with LITFLOW_POST=1 too, through the post chain as in the game).
let cloudsOn = ProcessInfo.processInfo.environment["LITFLOW_CLOUDS"] == "1" && (mode == "sky" || mode == "hdr")
let exp = mode == "compile" ? (ProcessInfo.processInfo.environment["LITFLOW_EXP"] ?? "lit,rtshadows,sky,water")
    : (mode == "off" ? "rtshadows" : (mode == "sky" ? "lit,rtshadows,sky" : (mode == "hdr" ? "lit,rtshadows,sky,hdr" : "lit,rtshadows")))
        + (giOn ? ",gi" : "") + (waterOn ? ",water" : "") + (clOn ? ",coloredlight" : "") + (postOn ? ",post" : "") + (cloudsOn ? ",clouds" : "")
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
    // The post chain (METALMC_EXP=...,post): its kernels and its copy into the frame for the float frames.
    if exp.split(separator: ",").contains("post"), let src = source("mmc_debug_post_shader_source") {
        attempt("post chain: kernels, copy into RG11B10Float and RGBA16Float frames") {
            let l = try device.makeLibrary(source: src, options: nil)
            for k in ["post_bloom_first", "post_bloom_down", "post_bloom_up", "post_histogram", "post_exposure", "post_shaft_mask",
                      "post_shaft_blur", "post_shaft_color", "post_composite"] {
                _ = try device.makeComputePipelineState(function: l.makeFunction(name: k)!)
            }
            for format in [MTLPixelFormat.rg11b10Float, .rgba16Float] {
                let d = MTLRenderPipelineDescriptor()
                d.vertexFunction = l.makeFunction(name: "post_copy_vs")
                d.fragmentFunction = l.makeFunction(name: "post_copy_fs")
                d.colorAttachments[0].pixelFormat = format
                d.depthAttachmentPixelFormat = .depth32Float
                _ = try device.makeRenderPipelineState(descriptor: d)
            }
        }
    }
    if let src = source("mmc_debug_cl_shader_source") {
        attempt("colored light's kernels (upload, edit, list, flood fill, resolve, clear)") {
            let l = try device.makeLibrary(source: src, options: nil)
            for k in ["cl_upload", "cl_edit", "cl_list_begin", "cl_list", "cl_propagate", "cl_resolve", "cl_clear"] {
                let p = try device.makeComputePipelineState(function: l.makeFunction(name: k)!)
                if k == "cl_propagate" || k == "cl_resolve" { check(p.maxTotalThreadsPerThreadgroup >= 512, "\(k) takes 512 threads a threadgroup (\(p.maxTotalThreadsPerThreadgroup))") }
            }
        }
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
/// The colored light's debug views are read without the sky's aerial perspective (by day its haze is over every pixel).
var skipAerial = false

// The world's LOD around the camera.
let viewSpec = (ProcessInfo.processInfo.environment["LITFLOW_VIEW"] ?? "8,150,8,100,22").split(separator: ",").map { Double($0)! }
// (vars: the colored light's section moves the camera between its views.)
var cam = SIMD3<Double>(viewSpec[0], viewSpec[1], viewSpec[2])
var yaw = Float(viewSpec[3]) * .pi / 180, pitch = Float(viewSpec[4]) * .pi / 180
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
// Colored block light (LITFLOW_CL=1): the blocks around the camera into the light's store (the volume reaches 128 blocks
// out: 10 chunks either way), then the test scene into the store and the LOD, and the LOD's rebuild of what it touched.
typealias ClFrameF = @convention(c) (Double, Double, Double) -> Void
let clFrameC: ClFrameF? = clOn ? fn("mmc_cl_frame", ClFrameF.self) : nil
if clOn {
    let load = fn("mmc_debug_cl_load", (@convention(c) (UnsafePointer<CChar>, Int32, Int32, Int32, Int32) -> Int32).self)
    let ccx = Int32((cam.x / 16).rounded(.down)), ccz = Int32((cam.z / 16).rounded(.down))
    let tl = Date()
    // With "time" far enough east for the moving camera's 240 blocks too.
    let reach: Int32 = timing ? 26 : 10
    let n = load(args[2], ccx - 10, ccz - 10, ccx + reach, ccz + 10)
    check(n > 0, "colored light: \(n) chunks around the camera read from the region files in \(String(format: "%.1f", Date().timeIntervalSince(tl))) s")
    let scenePath = ProcessInfo.processInfo.environment["LITFLOW_CLSCENE"]
        ?? URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("mod/src/client/resources/metalmc/coloredlight_scene.txt").path
    if scenePath != "none" {
        let scene = fn("mmc_debug_cl_scene", (@convention(c) (UnsafePointer<CChar>, UnsafePointer<CChar>) -> Int32).self)
        lodStatus(&st)
        let before = st[2]
        let placed = scene(scenePath, args[2])
        check(placed > 0, "colored light: the scene's \(placed) blocks placed (\(scenePath))")
        // The LOD picks the chunks up on its next pass (every 2 s) and rebuilds their nodes: wait for the quad count to
        // change and settle.
        let tw = Date()
        var changedAt: Date? = nil, last = before
        while Date().timeIntervalSince(tw) < 120 {
            Thread.sleep(forTimeInterval: 0.5)
            lodStatus(&st)
            if st[2] != last { last = st[2]; changedAt = Date() }
            if let c = changedAt, Date().timeIntervalSince(c) > 6 { break }
        }
        print("      the LOD rebuilt with the scene: \(before) -> \(st[2]) quads in \(Int(Date().timeIntervalSince(tw))) s")
    }
}
// LITFLOW_GATE=<dir>: the LOD's build (about 100 s of CPU at background priority, no GPU work) runs before the GPU lock is
// taken: litflow writes <dir>/ready and waits for <dir>/go, which a wrapper writes once it holds the lock
// (tools/bench/gpuwait.sh), so the lock is held only for the GPU work.
if let gate = ProcessInfo.processInfo.environment["LITFLOW_GATE"] {
    FileManager.default.createFile(atPath: gate + "/ready", contents: nil)
    while !FileManager.default.fileExists(atPath: gate + "/go") { Thread.sleep(forTimeInterval: 0.2) }
    print("ok    GPU lock taken")
}

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
    // With post the game's main target is RG11B10Float (92), as with HDR's packed default.
    Targets(color: textureCreate(hdr ? 115 : (postOn ? 92 : 70), w, h, 1, 1, 1), depth: textureCreate(252, w, h, 1, 1, 1), w: w, h: h)
}
// The clouds' entry point (LITFLOW_CLOUDS=1), off for a frame with cloudsSkip, and the rain (1 clear) they're made for.
let cloudsFrame = cloudsOn ? fn("mmc_clouds_frame", (@convention(c) (Int64, Int64, UnsafePointer<Float>, UnsafePointer<Double>) -> Int32).self) : nil
var cloudsSkip = false, cloudsRainBrightness: Float = 1
// The post chain's entry points (LITFLOW_POST=1).
let postApply = postOn ? fn("mmc_post_apply", (@convention(c) (Int64, Int64, UnsafePointer<Float>, UnsafePointer<Double>, Int32) -> Int32).self) : nil
var postRan = false, postSkip = false

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
    // Clouds (LITFLOW_CLOUDS=1): this frame's clouds before the shadows, as GameRendererLodMixin calls them (cloudsSkip: off).
    if let cloudsFrame, !cloudsSkip {
        var cp = mats + [sunAngle, cloudsRainBrightness, 0, 63]
        _ = cloudsFrame(t.color, t.depth, &cp, &c)
    }
    let traced = shadowsApply(t.color, t.depth, &mats, &c, sunAngle, 0.42, 192, 0) == 1
    var ran = false
    // Water: the reflections on or off, the waves 1/120 s on from the last frame (as at 120 Hz).
    if let litWaterSwitch { _ = litWaterSwitch(reflectWater ? 1 : 0, 100 + Double(framesDrawn) / 120) }
    framesDrawn += 1
    // Colored block light: the volume's work for this camera, before the relight (as GameRendererLodMixin calls it).
    if relight, let clFrameC { clFrameC(cam.x, cam.y, cam.z) }
    if relight, let litRelight {
        // Lit.relight's p: matrices, sun angle, fog color (strength 0: none), fog distances, our sky drew.
        var lp = mats + [sunAngle, 0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, sky ? 1 : 0]
        ran = litRelight(t.color, t.depth, &lp, lightmap, taa ? 1 : 0) == 1
    }
    if sky && !skipAerial { _ = skyAerial(t.color, t.depth, &mats, taa ? 1 : 0) }
    if taa { var c2 = [cam.x, cam.y, cam.z]; _ = taaApply(t.color, t.depth, &mats, &c2, 0, 0, 0) }
    if let postApply, !postSkip { var pp = mats + [sunAngle]; var c3 = [cam.x, cam.y, cam.z]; postRan = postApply(t.color, t.depth, &pp, &c3, taa ? 1 : 0) == 1 }
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
    let floatTarget = hdr || postOn
    let snap = floatTarget ? textureCreate(70, t.w, t.h, 1, 1, 1) : 0
    litSetView?(view)
    var gotG = false
    var sum = [Float](repeating: 0, count: n * 4)
    for k in 0..<frames {
        frame(t, sunAngle: sunAngle, relight: relight, extras: extras, taa: taa) { color in
            if floatTarget { _ = hdrSnapshot(color, snap) }
            copyToBuffer(floatTarget ? snap : color, 0, 0, 0, t.w, t.h, cbuf, 0, t.w * 4)
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

// "cull": the LOD's quad culling (METALMC_EXP=quadcull). Per view, the level's main pass as the timing runs draw it (the
// sky's clear, then the LOD and the far field through mmc_lod_draw) without and with culling, switched in one process
// (mmc_debug_lod_paths). The check: color, depth and (lit mode) the G-buffer must match bit for bit at the panel's
// resolution, at three sub-pixel shifts of the projection (as the anti-aliasing's jitter moves it), and with vanilla
// sections around the camera cut out (the seam variants). The timing: the variants alternate, 40 frames each, and then
// run back to back with two frames in flight; the culling pass is a command buffer of its own, timed apart and as the
// span from its start to the frame's end.
if cullRun {
    setbuf(stdout, nil)   // in order with the library's log lines (the traced frames' stage times)
    let setVanilla = fn("mmc_lod_set_vanilla", (@convention(c) (UnsafePointer<Int64>, Int32) -> Void).self)
    let traceFrames = fn("mmc_trace_frames", (@convention(c) (Int32) -> Void).self)
    // A library without quad culling (the build before it) only renders the reference (LITFLOW_CULLREF, below).
    let refOnly = dlsym(lib, "mmc_debug_lod_paths") == nil
    typealias PathsF = @convention(c) (Int32, Int32, Int32) -> Int32
    typealias CullLastF = @convention(c) (UnsafeMutablePointer<Double>) -> Int32
    typealias PipelinesF = @convention(c) () -> Int32
    let pathsC: PathsF? = refOnly ? nil : fn("mmc_debug_lod_paths", PathsF.self)
    let cullLastC: CullLastF? = refOnly ? nil : fn("mmc_debug_lod_quadcull_last", CullLastF.self)
    let pipelinesC: PipelinesF? = refOnly ? nil : fn("mmc_debug_lod_pipelines", PipelinesF.self)
    func debugPaths(_ mesh: Int32, _ cull: Int32, _ flags: Int32) -> Int32 { pathsC?(mesh, cull, flags) ?? 1 }
    func debugCullLast(_ out: UnsafeMutablePointer<Double>) -> Int32 { cullLastC?(out) ?? 0 }
    func debugPipelines() -> Int32 { pipelinesC?() ?? 0 }
    struct View { let name: String; let pos: SIMD3<Double>; let yaw: Float; let pitch: Float }
    var views: [View] = []
    if let spec = ProcessInfo.processInfo.environment["LITFLOW_VIEWS"] {
        for (i, v) in spec.split(separator: ";").enumerated() {
            let n = v.split(separator: ",").map { Double($0)! }
            views.append(View(name: "view\(i)", pos: SIMD3(n[0], n[1], n[2]), yaw: Float(n[3]), pitch: Float(n[4])))
        }
    } else {
        // The default view and the other three directions at y 150 looking 22 degrees down (the real-terrain flight's
        // height), two lower and flatter, and one high.
        views = [View(name: "y150-yaw100", pos: SIMD3(8, 150, 8), yaw: 100, pitch: 22), View(name: "y150-yaw10", pos: SIMD3(8, 150, 8), yaw: 10, pitch: 22),
                 View(name: "y150-yaw190", pos: SIMD3(8, 150, 8), yaw: 190, pitch: 22), View(name: "y150-yaw280", pos: SIMD3(8, 150, 8), yaw: 280, pitch: 22),
                 View(name: "y110-yaw100", pos: SIMD3(8, 110, 8), yaw: 100, pitch: 8), View(name: "y110-yaw280", pos: SIMD3(8, 110, 8), yaw: 280, pitch: 8),
                 View(name: "y260-yaw100", pos: SIMD3(8, 260, 8), yaw: 100, pitch: 35)]
    }
    // The camera of `matrices`, per view, with the projection shifted by `jitter` pixels (as TemporalAA.jitter does).
    func viewMatrices(_ t: Targets, _ v: View, jitter: SIMD2<Float>) -> [Float] {
        let yw = v.yaw * .pi / 180, pt = v.pitch * .pi / 180
        let fwd = SIMD3<Float>(-sin(yw) * cos(pt), -sin(pt), cos(yw) * cos(pt))
        let right = simd_normalize(simd_cross(fwd, SIMD3(0, 1, 0))), up = simd_cross(right, fwd)
        let view = simd_float4x4(rows: [SIMD4(right, 0), SIMD4(up, 0), SIMD4(-fwd, 0), SIMD4(0, 0, 0, 1)])
        let f = 1 / tan(35 * Float.pi / 180), aspect = Float(t.w) / Float(t.h)
        let jx = 2 * jitter.x / Float(t.w), jy = 2 * jitter.y / Float(t.h)
        let proj = simd_float4x4(SIMD4(f / aspect, 0, 0, 0), SIMD4(0, f, 0, 0), SIMD4(-jx, -jy, 0, -1), SIMD4(0, 0, 0.05, 0))
        var m = [Float](repeating: 0, count: 32)
        for c in 0..<4 { for r in 0..<4 { m[c * 4 + r] = proj[c][r]; m[16 + c * 4 + r] = view[c][r] } }
        return m
    }
    // Vanilla's sections for the seam variants: every section within 5 of the camera's (SectionPos.asLong keys), and the
    // matching half-size of vanilla's area for mmc_lod_draw's discard radius.
    func vanillaKeys(_ v: View) -> [Int64] {
        let cx = Int64((v.pos.x / 16).rounded(.down)), cz = Int64((v.pos.z / 16).rounded(.down))
        var keys: [Int64] = []
        for x in (cx - 5)...(cx + 5) { for z in (cz - 5)...(cz + 5) { for y in Int64(-4)...19 {
            keys.append(((x & 0x3FFFFF) << 42) | ((z & 0x3FFFFF) << 20) | (y & 0xFFFFF))
        } } }
        return keys
    }
    struct Capture { var color: [UInt32] = []; var depth: [UInt32] = []; var gbuf: [UInt32] = [] }
    let n = 3456 * 2234
    let full = targets(3456, 2234)
    let cbuf = bufferCreate(Int64(n * 4)), dbuf = bufferCreate(Int64(n * 4)), gbuf = bufferCreate(Int64(n * 8))
    /// One main pass of view `v` (submitted and waited for); with `read`, its color, depth and G-buffer.
    @discardableResult
    func mainPass(_ t: Targets, _ v: View, jitter: SIMD2<Float> = .zero, seam: Bool = false, read: Bool = false) -> Capture {
        let keys = seam ? vanillaKeys(v) : [0]
        keys.withUnsafeBufferPointer { setVanilla($0.baseAddress!, seam ? Int32(keys.count) : 0) }
        var h = t.color
        var clear: [Float] = [0.62, 0.75, 0.95, 1]
        _ = passBegin(&h, 1, 1, &clear, t.depth, 1, 0, 0, 0, t.w, t.h)
        passEnd()
        litLevelPass?()
        _ = passBegin(&h, 1, 0, &clear, t.depth, 0, 0, 0, 0, t.w, t.h)
        var p = viewMatrices(t, v, jitter: jitter) + [0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, seam ? 96 : 0, 1]
        var c = [v.pos.x, v.pos.y, v.pos.z]
        _ = lodDraw(&p, &c)
        passEnd()
        var gotG = false
        if read {
            copyToBuffer(t.color, 0, 0, 0, t.w, t.h, cbuf, 0, t.w * 4)
            copyToBuffer(t.depth, 0, 0, 0, t.w, t.h, dbuf, 0, t.w * 4)
            gotG = litCopyGbuffer?(gbuf, 0) == 1
        }
        submit(submitIndex)
        if waitSubmit(submitIndex, 10_000_000_000) != 1 { check(false, "frame \(submitIndex) done") }
        submitIndex += 1
        var r = Capture()
        if read {
            let k = Int(t.w) * Int(t.h)
            r.color = Array(UnsafeBufferPointer(start: UnsafeRawPointer(bitPattern: Int(bufferContents(cbuf)))!.assumingMemoryBound(to: UInt32.self), count: k))
            r.depth = Array(UnsafeBufferPointer(start: UnsafeRawPointer(bitPattern: Int(bufferContents(dbuf)))!.assumingMemoryBound(to: UInt32.self), count: k))
            if gotG {
                r.gbuf = Array(UnsafeBufferPointer(start: UnsafeRawPointer(bitPattern: Int(bufferContents(gbuf)))!.assumingMemoryBound(to: UInt32.self), count: 2 * k))
            }
        }
        return r
    }
    // Warm-up at a small size: the LOD's pipelines compile in the background and the far field's rings fill.
    let small = targets(432, 280)
    for _ in 0..<120 { mainPass(small, views[0]) }
    mainPass(full, views[0])
    let failed = debugPipelines()
    if !refOnly {
        check(failed == 0, "every LOD pipeline variant compiles (\(lit ? "lit" : "no lit") mode: vertex and mesh paths; plain, seam, water, fade, boxes) and the culling pass: \(failed) failed")
    }
    // The default path against another build's (LITFLOW_CULLREF=<file>): per view and case, a hash of the color and depth
    // after the same frames from the same start, written by the first run (e.g. with the library before quad culling) and
    // checked by the next. The rest of the run doesn't come before it, so both see the same frames.
    let cases: [(jitter: SIMD2<Float>, seam: Bool)] = [(SIMD2(0, 0), false), (SIMD2(0.25, -0.1667), false), (SIMD2(-0.375, 0.2778), true),
                                                       (SIMD2(0, 0), true)]
    if let refPath = ProcessInfo.processInfo.environment["LITFLOW_CULLREF"] {
        func hash(_ a: [UInt32]) -> UInt64 { a.reduce(UInt64(0xcbf2_9ce4_8422_2325)) { ($0 ^ UInt64($1)) &* 0x100_0000_01b3 } }
        // Settled first: two seconds of frames (the far field fills its rings over frames, the LOD's last nodes land),
        // then the first view until two captures 8 frames apart agree.
        let t0 = Date()
        while Date().timeIntervalSince(t0) < 2 { mainPass(full, views[0]) }
        var last: UInt64 = 0
        for _ in 0..<20 {
            for _ in 0..<7 { mainPass(full, views[0]) }
            let h = hash(mainPass(full, views[0], read: true).color)
            if h == last { break }
            last = h
        }
        var lines: [String] = []
        for v in views {
            for c in cases {
                for _ in 0..<12 { mainPass(full, v, jitter: c.jitter, seam: c.seam) }
                let cap = mainPass(full, v, jitter: c.jitter, seam: c.seam, read: true)
                for _ in 0..<3 { mainPass(full, v, jitter: c.jitter, seam: c.seam) }
                let again = mainPass(full, v, jitter: c.jitter, seam: c.seam, read: true)
                // Pixels the LOD or the far field drew (not the clear color): a blank capture would match anything.
                let drawn = cap.depth.reduce(0) { $0 + ($1 != 0 ? 1 : 0) }
                let stable = hash(cap.color) == hash(again.color) && hash(cap.depth) == hash(again.depth)
                lines.append("\(v.name) \(c.jitter.x) \(c.jitter.y) \(c.seam) \(String(hash(cap.color), radix: 16)) \(String(hash(cap.depth), radix: 16)) drawn \(drawn)\(stable ? "" : " unsettled")")
            }
        }
        if let old = try? String(contentsOfFile: refPath, encoding: .utf8) {
            let before = old.split(separator: "\n").map(String.init)
            let same = before == lines
            for (a, b) in zip(before, lines) where a != b { print("      reference: was \(a), now \(b)") }
            print((same ? "ok    " : "FAIL  ") + "the default path draws the same as the reference build (\(lines.count) views and cases, color and depth hashes, \(refPath))")
        } else {
            try? lines.joined(separator: "\n").write(toFile: refPath, atomically: true, encoding: .utf8)
            print("ok    reference written: \(lines.count) views and cases (\(refPath))")
        }
    }
    if refOnly { print("done"); exit(0) }
    // flags: 2 has the culling pass keep every quad (the cost of the pass and the indirect draws alone).
    let paths: [(name: String, mesh: Int32, cull: Int32, flags: Int32)] = [("unculled", 0, 0, 0), ("culled", 0, 1, 0), ("keep-all", 0, 1, 2)]
    func use(_ k: Int) { _ = debugPaths(paths[k].mesh, paths[k].cull, paths[k].flags) }
    // Pixels that differ between two captures: color, depth, G-buffer; and a picture of where (color red, depth green,
    // G-buffer blue).
    func differ(_ a: Capture, _ b: Capture, picture: String?) -> (Int, Int, Int) {
        var dc = 0, dd = 0, dg = 0
        var px: [UInt8] = picture == nil ? [] : [UInt8](repeating: 0, count: n * 4)
        for i in 0..<n {
            let c = a.color[i] != b.color[i], d = a.depth[i] != b.depth[i]
            let g = !a.gbuf.isEmpty && (a.gbuf[2 * i] != b.gbuf[2 * i] || a.gbuf[2 * i + 1] != b.gbuf[2 * i + 1])
            if c { dc += 1 }
            if d { dd += 1 }
            if g { dg += 1 }
            if picture != nil && (c || d || g) { px[4 * i] = c ? 255 : 0; px[4 * i + 1] = d ? 255 : 0; px[4 * i + 2] = g ? 255 : 0 }
        }
        if let picture, dc + dd + dg > 0 { writePNG(px, 3456, 2234, picture) }
        return (dc, dd, dg)
    }
    var allExact = true
    var stats = [Double](repeating: 0, count: 8)
    for v in views {
        print("      view \(v.name): (\(v.pos.x), \(v.pos.y), \(v.pos.z)) yaw \(v.yaw) pitch \(v.pitch)")
        for c in cases {
            // The view settles first (the occlusion test's results come from earlier frames, and a change in the picture
            // takes a few frames to go through them). Then unculled, culled and unculled again: the two unculled ones must
            // match (the view had settled) for the comparison to count.
            use(0)
            for _ in 0..<10 { mainPass(full, v, jitter: c.jitter, seam: c.seam) }
            func cap(_ k: Int) -> Capture {
                use(k)
                for _ in 0..<2 { mainPass(full, v, jitter: c.jitter, seam: c.seam) }
                return mainPass(full, v, jitter: c.jitter, seam: c.seam, read: true)
            }
            let c0 = cap(0), c1 = cap(1)
            let kept = debugCullLast(&stats) == 1 ? String(format: ", kept %.1f%% of %.0f quads", 100 * stats[2] / max(stats[1], 1), stats[1]) : ""
            let d = differ(c0, c1, picture: "cull-diff-\(v.name)-\(Int(c.jitter.x * 1000))\(c.seam ? "-seam" : "").png")
            let s = differ(c0, cap(0), picture: nil)
            let settled = s.0 + s.1 + s.2 == 0
            if settled && d.0 + d.1 + d.2 > 0 { allExact = false }
            print("      \(v.name) jitter (\(c.jitter.x), \(c.jitter.y))\(c.seam ? " seam" : ""): culled against unculled: \(d.0) color, \(d.1) depth, \(d.2) G-buffer pixels differ\(kept)"
                  + (settled ? "" : " (NOT SETTLED: \(s.0), \(s.1), \(s.2) differ between the unculled ones)"))
        }
    }
    use(0)
    // Reported, not fatal: the timing still runs.
    print((allExact ? "ok    " : "FAIL  ") + "culling leaves the main pass bit for bit the same (color, depth\(lit ? ", G-buffer" : "")) in every view, jitter and seam case")
    // Timing: the variants alternate, frame by frame, 40 each per view, after the view settled.
    func stat(_ s: [Double]) -> (Double, Double) {
        let t = s.sorted()
        return t.isEmpty ? (-1, -1) : (t[t.count / 2], t[0])
    }
    var times = [Double](repeating: 0, count: 4096)
    /// Frames back to back with two in flight, as the game runs: wall time per frame (ms), GPU-bound.
    func throughput(_ v: View, frames: Int) -> Double {
        var h = full.color
        var clear: [Float] = [0.62, 0.75, 0.95, 1]
        let none: [Int64] = [0]
        none.withUnsafeBufferPointer { setVanilla($0.baseAddress!, 0) }
        let t0 = Date()
        for k in 0..<frames {
            _ = passBegin(&h, 1, 1, &clear, full.depth, 1, 0, 0, 0, full.w, full.h)
            passEnd()
            litLevelPass?()
            _ = passBegin(&h, 1, 0, &clear, full.depth, 0, 0, 0, 0, full.w, full.h)
            var p = viewMatrices(full, v, jitter: .zero) + [0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, 0, 1]
            var c = [v.pos.x, v.pos.y, v.pos.z]
            _ = lodDraw(&p, &c)
            passEnd()
            submit(submitIndex)
            if k >= 1 { _ = waitSubmit(submitIndex - 1, 10_000_000_000) }
            submitIndex += 1
        }
        _ = waitSubmit(submitIndex - 1, 10_000_000_000)
        _ = gpuTimesTake(&times, 4096)
        return Date().timeIntervalSince(t0) * 1000 / Double(frames)
    }
    for v in views {
        use(0)
        for _ in 0..<6 { mainPass(full, v) }
        // The frame's own GPU time (its command buffer), and with culling the span from the culling pass's start to the
        // frame's end: the culling pass runs as the frame's first passes start, and its draws wait for it.
        var main = [[Double]](repeating: [], count: paths.count), pre = [[Double]](repeating: [], count: paths.count)
        var span = [[Double]](repeating: [], count: paths.count)
        var kept = 0.0, total = 0.0, jobs = 0.0
        for k in 0..<(40 * paths.count) {
            let i = k % paths.count
            use(i)
            _ = gpuTimesTake(&times, 4096)
            mainPass(full, v)
            let c = Int(gpuTimesTake(&times, 4096))
            if c > 0 { main[i].append(times[c - 1] * 1000) }
            if debugCullLast(&stats) == 1 {
                pre[i].append(stats[0])
                if stats[4] > 0 { span[i].append(stats[4]) }
                if i == 1 { kept = stats[2]; total = stats[1]; jobs = stats[3] }
            }
        }
        // Throughput: 3 rounds of 60 frames per path, alternating.
        var tput = [[Double]](repeating: [], count: paths.count)
        for _ in 0..<3 {
            for (i, _) in paths.enumerated() {
                use(i)
                _ = throughput(v, frames: 8)
                tput[i].append(throughput(v, frames: 60))
            }
        }
        let s = main.map { stat($0) }, sp = pre.map { stat($0) }, ss = span.map { stat($0) }
        print(String(format: "      %@ (%@), at 3456 x 2234; culling kept %.0f of %.0f quads (%.1f%%) in %.0f jobs. Sky clear + main pass, median (fastest) of 40 frames each, alternating; with culling, the span from its pass's start; then frames back to back (two in flight), ms per frame, median of 3 x 60:",
                     v.name, lit ? "lit" : "no lit", kept, total, 100 * kept / max(total, 1), jobs))
        for (i, pth) in paths.enumerated() {
            print(String(format: "        %-18@ %.3f (%.3f)", pth.name, s[i].0, s[i].1)
                  + (pre[i].isEmpty ? "" : String(format: ", culling pass %.3f (%.3f), span %.3f (%.3f)", sp[i].0, sp[i].1, ss[i].0, ss[i].1))
                  + String(format: "; back to back %.3f", stat(tput[i]).0))
        }
        // Stage times (the library logs a traced frame's passes: total, vertex, fragment).
        for (i, pth) in paths.enumerated() {
            use(i)
            mainPass(full, v)
            print("        traced: \(pth.name)")
            traceFrames(1)
            mainPass(full, v)
            _ = debugCullLast(&stats)
        }
        use(0)
    }
    print("done")
    exit(0)
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

// Clouds (LITFLOW_CLOUDS=1): per time of day (LITFLOW_CLOUDTIMES, default noon,sunset,morning) the relit frame with the
// anti-aliasing (and post with LITFLOW_POST=1) without and with the clouds, 48 frames each (the clouds' history and the
// anti-aliasing's settle; METALMC_CLOUDVIEW=4 in the environment shows the cloud shadows alone), the mean luma of the sky
// (top fifth) and of the terrain (bottom 55%); with "time", whole frames at 3456 x 2234 with and without the clouds,
// alternating. Then exits.
if cloudsOn {
    func meanLuma(_ px: [UInt8], rows: Range<Int>) -> Double {
        var s = 0.0
        for y in rows { for x in 0..<Int(W) { s += lumOf(px, (y * Int(W) + x) * 4) } }
        return s / Double(rows.count * Int(W))
    }
    let name = ProcessInfo.processInfo.environment["LITFLOW_NAME"] ?? "view"
    let suns: [String: (Float, Float)] = ["noon": (noon, 1), "morning": (morning, 1), "dusk": (dusk, 0.55), "sunset": (1.49, 0.45),
                                          "midnight": (midnight, 0.2)]
    let list = (ProcessInfo.processInfo.environment["LITFLOW_CLOUDTIMES"] ?? "noon,sunset,morning").split(separator: ",").map(String.init)
    if ProcessInfo.processInfo.environment["LITFLOW_TIMEONLY"] != "1" {
        for t in list {
            guard let (angle, skyFactor) = suns[t] else { print("      unknown time \(t)"); continue }
            setLightmap(skyFactor: skyFactor)
            for (label, skip) in [("off", true), ("on", false)] {
                cloudsSkip = skip
                for _ in 0..<16 { frame(small, sunAngle: angle, relight: true, extras: false) }   // the sky's tables for this sun
                let r = capture(half, sunAngle: angle, relight: true, extras: false, frames: 48, taa: true)
                writePNG(r.color, Int(W), Int(H), "clouds-\(name)-\(t)-\(label).png")
                // The readback's rows are bottom first: the picture's top fifth is its last rows.
                let h = Int(H)
                print(String(format: "      %@ %@ clouds %@: mean luma, sky (top fifth) %.1f, terrain (bottom 55%%) %.1f, frame %.1f", name, t, label,
                             meanLuma(r.color, rows: (h - h / 5)..<h), meanLuma(r.color, rows: 0..<(h * 55 / 100)), meanLuma(r.color, rows: 0..<h)))
            }
        }
        cloudsSkip = false
    }
    if timing {
        let full = targets(3456, 2234)
        setLightmap(skyFactor: 1)
        for _ in 0..<16 { frame(full, sunAngle: noon, relight: true, extras: false, taa: true) }
        var times = [Double](repeating: 0, count: 4096)
        var with: [Double] = [], without: [Double] = []
        for k in 0..<160 {
            cloudsSkip = k % 2 == 1
            _ = gpuTimesTake(&times, 4096)
            frame(full, sunAngle: noon, relight: true, extras: false, taa: true)
            let c = Int(gpuTimesTake(&times, 4096))
            if c > 0 && k >= 8 { if cloudsSkip { without.append(times[c - 1] * 1000) } else { with.append(times[c - 1] * 1000) } }
        }
        cloudsSkip = false
        let ws = with.sorted(), os = without.sorted()
        if !ws.isEmpty && !os.isEmpty {
            print(String(format: "      frame at 3456 x 2234 (noon, %@): with the clouds %.3f ms (fastest %.3f), without %.3f (fastest %.3f): +%.3f (fastest +%.3f), %d each",
                         name, ws[ws.count / 2], ws[0], os[os.count / 2], os[0], ws[ws.count / 2] - os[os.count / 2], ws[0] - os[0], ws.count))
        }
    }
    print("done")
    exit(0)
}

// Post (LITFLOW_POST=1): per time of day (LITFLOW_POSTTIMES, default noon,dusk,sunset,midnight), the relit frame through the
// post chain with the anti-aliasing, each effect added in turn in the same process (mmc_debug_post): "before" (the legacy
// curve, no effects: the look before post, through the same float frame), "agx" (the tone curve alone), "bloom", "full"
// (bloom, light shafts, eye adaptation), and the full chain with the other curves; the mean 8-bit luma of each and the
// exposure's state. The exposure snaps to its target at each picture's first frame and follows at 1/60 s a frame. With
// "time": each stage's cost at the panel's resolution, and whole frames with and without post. Then exits.
if postOn {
    let postDebug = fn("mmc_debug_post", (@convention(c) (Int32, Int32, Double, Int32) -> Int32).self)
    let postExposure = fn("mmc_debug_post_exposure", (@convention(c) (UnsafeMutablePointer<Float>) -> Int32).self)
    let postTime = fn("mmc_debug_post_time", (@convention(c) (Int64, Int64, UnsafePointer<Float>, Int32, UnsafeMutablePointer<Double>) -> Int32).self)
    check(postDebug(-1, -1, 1.0 / 60, 1) == 1, "post is on (\(exp))")
    let n = Int(W) * Int(H)
    func meanLuma(_ px: [UInt8]) -> Double {
        var s = 0.0
        for i in stride(from: 0, to: n * 4, by: 4) { s += lumOf(px, i) }
        return s / Double(n)
    }
    // Pixels at 8-bit white (all three channels 255): clipped highlights.
    func clipped(_ px: [UInt8]) -> Double {
        var c = 0
        for i in stride(from: 0, to: n * 4, by: 4) where px[i] == 255 && px[i + 1] == 255 && px[i + 2] == 255 { c += 1 }
        return 100 * Double(c) / Double(n)
    }
    let name = ProcessInfo.processInfo.environment["LITFLOW_NAME"] ?? "view"
    let suns: [String: (Float, Float)] = ["noon": (noon, 1), "morning": (morning, 1), "dusk": (dusk, 0.55), "sunset": (1.53, 0.4),
                                          "low": (-1.45, 0.6), "midnight": (midnight, 0.2)]
    // The tone curves (mmc_debug_post_curve): gray from 2^-10 to 2^14 in sixteenths of a stop, and a saturated orange, at
    // headroom 1 (SDR), 2, 4, 8: monotonic, never past the headroom, and with headroom the SDR curve below its knee.
    if let curveF = dlsym(lib, "mmc_debug_post_curve").map({ unsafeBitCast($0, to: (@convention(c) (UnsafePointer<Float>, UnsafeMutablePointer<Float>, Int32, Float, Int32) -> Int32).self) }) {
        let steps = 24 * 16 + 1
        var input = [Float](repeating: 0, count: steps * 4 * 2)
        for k in 0..<steps {
            let x = powf(2, -10 + Float(k) / 16)
            input[4 * k] = x; input[4 * k + 1] = x; input[4 * k + 2] = x
            input[4 * (steps + k)] = x; input[4 * (steps + k) + 1] = 0.35 * x; input[4 * (steps + k) + 2] = 0.06 * x
        }
        for curve in Int32(0)...3 {
            var sdr = [Float](repeating: 0, count: input.count)
            var line: [String] = []
            for h: Float in [1, 2, 4, 8] {
                var out = [Float](repeating: 0, count: input.count)
                guard curveF(input, &out, Int32(steps * 2), h, curve) == 1 else { print("FAIL  curve check"); break }
                if h == 1 { sdr = out }
                var mono = true, under = true, belowKnee = 0.0
                for k in 0..<(steps * 2) {
                    let y = max(out[4 * k], max(out[4 * k + 1], out[4 * k + 2]))
                    if y > h * 1.0005 { under = false }
                    if k % steps > 0 && out[4 * k + 1] < out[4 * (k - 1) + 1] - 1e-5 { mono = false }
                    let x = input[4 * k + 1]
                    if h > 1 && x < 0.5 { belowKnee = max(belowKnee, Double(abs(out[4 * k + 1] - sdr[4 * k + 1]))) }
                }
                func at(_ x: Float) -> Float { out[4 * Int(((log2(x) + 10) * 16).rounded()) + 1] }
                line.append(String(format: "H %.0f: 0.18 -> %.3f, 1 -> %.3f, 4 -> %.3f, 64 -> %.3f, 4096 -> %.2f%@%@%@", h, at(0.18), at(1), at(4), at(64), at(4096),
                                   mono ? "" : " NOT MONOTONIC", under ? "" : " PAST THE HEADROOM",
                                   h > 1 ? String(format: " (below 0.5 within %.4f of SDR)", belowKnee) : ""))
            }
            print("      tone curve \(["legacy", "agx", "aces", "gt"][Int(curve)]): " + line.joined(separator: "; "))
        }
    }
    let list = (ProcessInfo.processInfo.environment["LITFLOW_POSTTIMES"] ?? "noon,dusk,sunset,midnight").split(separator: ",").map(String.init)
    if ProcessInfo.processInfo.environment["LITFLOW_TIMEONLY"] != "1" {
        for t in list {
            guard let (angle, skyFactor) = suns[t] else { print("      unknown time \(t)"); continue }
            setLightmap(skyFactor: skyFactor)
            for _ in 0..<16 { frame(small, sunAngle: angle, relight: true, extras: false) }   // the sky's tables for this sun
            for (label, effects, curve) in [("before", Int32(0), Int32(0)), ("agx", 0, 1), ("bloom", 1, 1), ("bloom-shafts", 3, 1),
                                            ("full", 7, 1), ("full-aces", 7, 2), ("full-gt", 7, 3)] {
                _ = postDebug(effects, curve, 1.0 / 60, 1)
                let r = capture(half, sunAngle: angle, relight: true, extras: false, frames: 24, taa: true)
                writePNG(r.color, Int(W), Int(H), "post-\(name)-\(t)-\(label).png")
                var e = [Float](repeating: 0, count: 8)
                _ = postExposure(&e)
                print(String(format: "      %@ %@ %@: mean luma %.1f, %.2f%% clipped white; exposure %+.2f stops (target %+.2f), metered log2 luminance %.2f",
                             name, t, label, meanLuma(r.color), clipped(r.color), e[0], e[1], e[2]))
            }
            check(postRan, "the post chain ran (\(t))")
        }
        // Eye adaptation over time: from noon's exposure into midnight at once, the exposure per frame at 120 Hz (what the
        // smoothing does), then back.
        setLightmap(skyFactor: 1)
        _ = postDebug(7, 1, 1.0 / 120, 1)
        for _ in 0..<8 { frame(small, sunAngle: noon, relight: true, extras: false, taa: true) }
        for (label, angle, sf) in [("noon -> midnight", midnight, Float(0.2)), ("midnight -> noon", noon, Float(1))] {
            setLightmap(skyFactor: sf)
            var trace: [String] = []
            for k in 0..<720 {
                frame(small, sunAngle: angle, relight: true, extras: false, taa: true)
                var e = [Float](repeating: 0, count: 8)
                _ = postExposure(&e)
                if [0, 15, 30, 60, 120, 240, 480, 719].contains(k) { trace.append(String(format: "%.2f s %+.2f", Double(k + 1) / 120, e[0])) }
            }
            print("      adaptation \(label) at 120 Hz (stops): " + trace.joined(separator: ", "))
        }
    }
    if timing {
        let full = targets(3456, 2234)
        setLightmap(skyFactor: 0.4)
        _ = postDebug(7, 1, 1.0 / 120, 1)
        for _ in 0..<8 { frame(full, sunAngle: 1.53, relight: true, extras: false, taa: true) }
        var out = [Double](repeating: 0, count: 10)
        var pp = matrices(width: full.w, height: full.h) + [Float(1.53)]
        for round in 0..<2 {
            guard postTime(full.color, full.depth, &pp, 40, &out) == 1 else { print("FAIL  post timing"); break }
            print(String(format: "      round %d: post at 3456 x 2234 (sunset, the sun on screen), median (fastest) of 40: bloom %.3f (%.3f) ms, exposure %.3f (%.3f), light shafts %.3f (%.3f), composite %.3f (%.3f), copy into the frame %.3f (%.3f)",
                         round, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7], out[8], out[9]))
        }
        // Whole frames (sky, main pass, shadows, relight and aerial perspective in the anti-aliasing's resolve) with and
        // without post, alternating, after 8 to settle.
        var times = [Double](repeating: 0, count: 4096)
        var with: [Double] = [], without: [Double] = []
        for k in 0..<120 {
            postSkip = k % 2 == 1
            _ = gpuTimesTake(&times, 4096)
            frame(full, sunAngle: 1.53, relight: true, extras: false, taa: true)
            let c = Int(gpuTimesTake(&times, 4096))
            if c > 0 { if postSkip { without.append(times[c - 1] * 1000) } else { with.append(times[c - 1] * 1000) } }
        }
        postSkip = false
        let ws = with.sorted(), os = without.sorted()
        if !ws.isEmpty && !os.isEmpty {
            print(String(format: "      frame at 3456 x 2234 (sunset): with post %.3f ms (fastest %.3f), without %.3f (fastest %.3f): +%.3f (fastest +%.3f), 60 each",
                         ws[ws.count / 2], ws[0], os[os.count / 2], os[0], ws[ws.count / 2] - os[os.count / 2], ws[0] - os[0]))
        }
    }
    print("done")
    exit(0)
}

// Colored block light (LITFLOW_CL=1): its own section, then exit.
if clOn {
    let n = Int(W) * Int(H)
    let clSwitch = fn("mmc_debug_cl_on", (@convention(c) (Int32, Double) -> Int32).self)
    let clBench = fn("mmc_debug_cl_bench", (@convention(c) (Double, Double, Double, Int32, UnsafeMutablePointer<Double>) -> Int32).self)
    let clVerify = fn("mmc_debug_cl_verify", (@convention(c) (UnsafeMutablePointer<Int64>) -> Int32).self)
    let clInvalidate = fn("mmc_debug_cl_invalidate", (@convention(c) () -> Void).self)
    let clBlock = fn("mmc_cl_block", (@convention(c) (Int32, Int32, Int32, Int32, Int32) -> Void).self)
    let clClassify = fn("mmc_cl_classify", (@convention(c) (UnsafePointer<CChar>) -> Int32).self)
    // A fixed clock for the fire's flicker (the pictures repeat), which also keeps the frame's volume valid for the timings.
    check(clSwitch(1, 100) == 1, "colored light is on (lit,coloredlight)")
    struct ClView { let name: String; let pos: SIMD3<Double>; let yaw: Float; let pitch: Float }
    let views: [ClView] = (ProcessInfo.processInfo.environment["LITFLOW_CLVIEWS"] ?? "above:0.5,227.6,2.5,0,47;low:0.5,207.1,15.5,0,24;world:8,150,8,100,22")
        .split(separator: ";").map { s in
            let parts = s.split(separator: ":"), v = parts[1].split(separator: ",").map { Double($0)! }
            return ClView(name: String(parts[0]), pos: SIMD3(v[0], v[1], v[2]), yaw: Float(v[3]), pitch: Float(v[4]))
        }
    func use(_ v: ClView) { cam = v.pos; yaw = v.yaw * .pi / 180; pitch = v.pitch * .pi / 180 }
    let suns: [String: (Float, Float)] = ["midnight": (midnight, 0.2), "noon": (noon, 1), "dusk": (dusk, 0.55), "morning": (morning, 1)]
    let times = (ProcessInfo.processInfo.environment["LITFLOW_CLTIMES"] ?? "midnight,noon").split(separator: ",").map(String.init)
    var bench = [Double](repeating: 0, count: 2 * 512)
    /// Frames of the volume's work alone until its flood fill has nothing left (up to `limit`): how many it took.
    func settle(limit: Int = 64) -> Int {
        for k in 0..<limit {
            _ = clBench(cam.x, cam.y, cam.z, 1, &bench)
            if bench[1] == 0 { return k }
        }
        return limit
    }
    func verify(_ what: String) {
        var out = [Int64](repeating: 0, count: 6)
        guard clVerify(&out) == 1 else { check(false, "verify ran"); return }
        print("      \(what): flood fill against the CPU's: \(out[1]) of \(out[0]) cells differ (largest \(out[4]) levels), \(out[5]) brighter than the CPU's; \(out[3]) cells lit; \(out[2]) cells' codes differ from the store's")
        // Reported, not fatal: the pictures still come.
        print((out[1] == 0 && out[2] == 0 ? "ok    " : "FAIL  ") + "the volume's light is vanilla's flood fill per color, exactly (\(what))")
    }
    // Natural scenes in the loaded area: light sources next to open cells, by 32 x 32 column cell and kind (warm and fire
    // lights: villages; lava; others), the top few of each with their mean height, to point a view or the tour at.
    if let emittersC = dlsym(lib, "mmc_debug_cl_emitters").map({ unsafeBitCast($0, to: (@convention(c) (Int32, Int32, Int32, Int32, Int32, Int32, UnsafeMutablePointer<Int32>, UnsafeMutablePointer<Int32>, Int32) -> Int32).self) }) {
        let maxN: Int32 = 400_000
        var e = [Int32](repeating: 0, count: 4 * Int(maxN)), o = [Int32](repeating: 0, count: Int(maxN))
        let cx = Int32(cam.x), cz = Int32(cam.z)
        let total = Int(emittersC(cx - 160, -64, cz - 160, cx + 400, 319, cz + 160, &e, &o, maxN))
        var cells: [Int64: (warm: Int, lava: Int, other: Int, ySum: Int, n: Int)] = [:]
        for k in 0..<min(total, Int(maxN)) where o[k] > 0 {
            let x = Int(e[4 * k]), y = Int(e[4 * k + 1]), z = Int(e[4 * k + 2]), code = Int(e[4 * k + 3])
            if y > 0 && y < 200 && abs(x) < 26 && z >= 12 && z <= 66 { continue }   // (the scene, if it's there)
            if y >= 190 { continue }
            let cls = (code >> 8) & 63
            let key = Int64((x >> 5) + 100_000) << 32 | Int64((z >> 5) + 100_000)
            var c = cells[key] ?? (0, 0, 0, 0, 0)
            if [2, 3, 4, 5, 6, 7, 12, 29].contains(cls) { c.warm += 1 } else if cls == 10 || cls == 11 { c.lava += 1 } else { c.other += 1 }
            c.ySum += y; c.n += 1
            cells[key] = c
        }
        func show(_ name: String, _ score: ((warm: Int, lava: Int, other: Int, ySum: Int, n: Int)) -> Int) {
            let top = cells.sorted { score($0.value) > score($1.value) }.prefix(5).filter { score($0.value) > 0 }
            print("      natural light sources, most \(name) (32 x 32 column cells: x, z, mean y, warm/lava/other): "
                  + top.map { "(\((Int($0.key >> 32) - 100_000) * 32 + 16), \((Int($0.key & 0xFFFF_FFFF) - 100_000) * 32 + 16), y \($0.value.ySum / max($0.value.n, 1)): \($0.value.warm)/\($0.value.lava)/\($0.value.other))" }.joined(separator: " "))
        }
        print("      \(total) light sources in the loaded area")
        show("warm (villages)", { $0.warm })
        show("lava", { $0.lava })
        show("other", { $0.other })
        // A camera in a cave over lava: lava with air above it, grouped by 32 x 32 column cell; from the biggest groups'
        // centers, an open spot 3-8 blocks above the lava and 6-14 away, mostly open around, with an open line of sight to
        // the lava. Printed as a pose (feet position for the tour: eye minus 1.62).
        if let openC = dlsym(lib, "mmc_debug_cl_open").map({ unsafeBitCast($0, to: (@convention(c) (Int32, Int32, Int32) -> Int32).self) }) {
            func open(_ x: Int, _ y: Int, _ z: Int) -> Bool { openC(Int32(x), Int32(y), Int32(z)) == 1 }
            var groups: [Int64: [SIMD3<Int>]] = [:]
            for k in 0..<min(total, Int(maxN)) where ((Int(e[4 * k + 3]) >> 8) & 63) == 10 {
                let x = Int(e[4 * k]), y = Int(e[4 * k + 1]), z = Int(e[4 * k + 2])
                guard y < 180, open(x, y + 1, z) else { continue }
                groups[Int64((x >> 5) + 100_000) << 32 | Int64((z >> 5) + 100_000), default: []].append(SIMD3(x, y, z))
            }
            var found = 0
            for (_, cells) in groups.sorted(by: { $0.value.count > $1.value.count }).prefix(8) where found < 3 {
                let c = cells.reduce(SIMD3<Int>.zero, &+) / SIMD3(repeating: cells.count)
                let surface = cells.map { $0.y }.max() ?? c.y
                let target = SIMD3<Double>(Double(c.x) + 0.5, Double(surface) + 1.0, Double(c.z) + 0.5)
                var best: (SIMD3<Double>, Int)? = nil
                for dist in stride(from: 14, through: 6, by: -2) {
                    for dy in [5, 4, 6, 3, 7, 8] {
                        for a in 0..<16 {
                            let ang = Double(a) * .pi / 8
                            let p = SIMD3<Int>(c.x + Int((Double(dist) * cos(ang)).rounded()), surface + dy, c.z + Int((Double(dist) * sin(ang)).rounded()))
                            guard open(p.x, p.y, p.z), open(p.x, p.y - 1, p.z) else { continue }
                            var around = 0
                            for ox in -1...1 { for oy in -1...1 { for oz in -1...1 where open(p.x + ox, p.y + oy, p.z + oz) { around += 1 } } }
                            guard around >= 22 else { continue }
                            let eye = SIMD3<Double>(Double(p.x) + 0.5, Double(p.y) + 0.5, Double(p.z) + 0.5)
                            let steps = Int(simd_length(target - eye) * 2)
                            var clear = true
                            for s in 1..<max(steps, 2) {
                                let q = eye + (target - eye) * (Double(s) / Double(max(steps, 2)))
                                if !open(Int(q.x.rounded(.down)), Int(q.y.rounded(.down)), Int(q.z.rounded(.down))) { clear = false; break }
                            }
                            if clear && around > (best?.1 ?? 0) { best = (eye, around) }
                        }
                    }
                    if best != nil { break }
                }
                guard let b = best else { continue }
                let eye = b.0
                let d = target - eye
                let yawDeg = atan2(-d.x, d.z) * 180 / .pi, pitchDeg = -atan2(d.y, (d.x * d.x + d.z * d.z).squareRoot()) * 180 / .pi
                print(String(format: "      cave over lava (%d cells with air above, around %d %d %d): eye %.1f,%.1f,%.1f yaw %.0f pitch %.0f (tour feet y %.2f)",
                             cells.count, c.x, surface, c.z, eye.x, eye.y, eye.z, yawDeg, pitchDeg, eye.y - 1.62))
                found += 1
            }
        }
    }
    for v in views {
        use(v)
        print("      view \(v.name): (\(v.pos.x), \(v.pos.y), \(v.pos.z)) yaw \(v.yaw) pitch \(v.pitch)")
        print("      settled in \(settle()) frames of the volume's work")
        verify("view \(v.name)")
        for t in times {
            guard let (angle, skyFactor) = suns[t] else { continue }
            setLightmap(skyFactor: skyFactor)
            for _ in 0..<16 { frame(small, sunAngle: angle, relight: true, extras: false) }
            _ = clSwitch(0, 100)
            let off = capture(half, sunAngle: angle, relight: true, extras: false)
            // 48 frames more, so the "on" frames have the same places in every 64-frame cycle (the shadows' disk samples,
            // the sky's dither) as the "off" ones: then only block light may differ between the two.
            for _ in 0..<48 { frame(small, sunAngle: angle, relight: true, extras: false) }
            _ = clSwitch(1, 100)
            let on = capture(half, sunAngle: angle, relight: true, extras: false)
            let onT = capture(half, sunAngle: angle, relight: true, extras: false, frames: 24, taa: true)
            // Vanilla's again, at the same place in the cycle (16 + 24 frames since the colored capture began, 24 more):
            // far terrain whose shadow structures are still building changes a few hundred pixels between any two
            // captures, which is the yardstick for the check below.
            for _ in 0..<24 { frame(small, sunAngle: angle, relight: true, extras: false) }
            _ = clSwitch(0, 100)
            let off2 = capture(half, sunAngle: angle, relight: true, extras: false)
            _ = clSwitch(1, 100)
            writePNG(off.color, Int(W), Int(H), "cl-\(v.name)-\(t)-vanilla.png")
            writePNG(on.color, Int(W), Int(H), "cl-\(v.name)-\(t)-colored.png")
            // Side by side, vanilla's block light left and the colored light right.
            var pair = [UInt8](repeating: 255, count: 2 * n * 4)
            for y in 0..<Int(H) {
                for x in 0..<Int(W) {
                    for k in 0..<4 {
                        pair[(y * 2 * Int(W) + x) * 4 + k] = off.color[(y * Int(W) + x) * 4 + k]
                        pair[(y * 2 * Int(W) + Int(W) + x) * 4 + k] = on.color[(y * Int(W) + x) * 4 + k]
                    }
                }
            }
            writePNG(pair, 2 * Int(W), Int(H), "cl-\(v.name)-\(t)-pair.png")
            writePNG(onT.color, Int(W), Int(H), "cl-\(v.name)-\(t)-colored-taa.png")
            skipAerial = true
            let lightView = capture(half, sunAngle: angle, relight: true, view: 11, extras: false, frames: 1)
            let checkView = capture(half, sunAngle: angle, relight: true, view: 12, extras: false, frames: 1)
            skipAerial = false
            writePNG(lightView.color, Int(W), Int(H), "cl-\(v.name)-\(t)-light.png")
            writePNG(checkView.color, Int(W), Int(H), "cl-\(v.name)-\(t)-check.png")
            // Relit terrain (not light sources) by vanilla's block light level at the pixel, and what the check did there:
            // red the volume scaled down to vanilla's level, green colored light, blue a share of vanilla's light made up.
            var lit = 0, dark = 0, applied = 0, clamped = 0, filled = 0, outside = 0, darkLit = 0, darkClamped = 0, darkChanged = 0
            var darkDrift = 0
            var fillSum = 0.0, lumOff = 0.0, lumOn = 0.0, rgbOff = SIMD3<Double>.zero, rgbOn = SIMD3<Double>.zero
            for i in 0..<n {
                let gx = on.gbuf[2 * i], gy = on.gbuf[2 * i + 1]
                guard gx >> 29 != 0, depthMatches(on.depth[i], gy & 0xFFFF) else { continue }
                let level = Double(gy >> 24) / 16
                if level >= 15 { continue }
                let r = checkView.color[4 * i], g = checkView.color[4 * i + 1], b = checkView.color[4 * i + 2]
                let isOutside = r == g && r > 150 && r < 200 && b < 20
                let isClamped = r > 200, isApplied = g > 150 && !isOutside
                if level > 0 {
                    lit += 1
                    if isOutside { outside += 1 }
                    if isApplied { applied += 1 }
                    if isClamped { clamped += 1 }
                    if b > 10 { filled += 1; fillSum += Double(b) / 230 }
                    lumOff += lumOf(off.color, 4 * i); lumOn += lumOf(on.color, 4 * i)
                    rgbOff += SIMD3(Double(off.color[4 * i]), Double(off.color[4 * i + 1]), Double(off.color[4 * i + 2]))
                    rgbOn += SIMD3(Double(on.color[4 * i]), Double(on.color[4 * i + 1]), Double(on.color[4 * i + 2]))
                } else {
                    dark += 1
                    if isClamped { darkClamped += 1 }
                    if lumOf(lightView.color, 4 * i) > 3 { darkLit += 1 }
                    // The frame as drawn: where vanilla has no block light the colored frame is the vanilla one (the
                    // relight returns vanilla's light there), so it may differ from vanilla's no more than two vanilla
                    // frames differ from each other; a glow would show at thousands of pixels (the volume has light at
                    // tens of thousands of them, darkClamped).
                    func differs(_ a: [UInt8], _ b: [UInt8]) -> Bool {
                        a[4 * i] != b[4 * i] || a[4 * i + 1] != b[4 * i + 1] || a[4 * i + 2] != b[4 * i + 2]
                    }
                    if differs(on.color, off.color) { darkChanged += 1 }
                    if differs(off.color, off2.color) { darkDrift += 1 }
                }
            }
            let l = Double(max(lit, 1))
            print(String(format: "      %@ %@: %d relit pixels with block light, %d without; with: colored %.1f%%, scaled down to vanilla's level %.1f%%, vanilla's made up %.1f%% (mean share %.2f), outside the volume %.1f%%; mean luma %.1f -> %.1f, mean RGB (%.0f, %.0f, %.0f) -> (%.0f, %.0f, %.0f)",
                         v.name, t, lit, dark, 100 * Double(applied) / l, 100 * Double(clamped) / l, 100 * Double(filled) / l,
                         filled > 0 ? fillSum / Double(filled) : 0, 100 * Double(outside) / l, lumOff / l, lumOn / l,
                         rgbOff.x / l, rgbOff.y / l, rgbOff.z / l, rgbOn.x / l, rgbOn.y / l, rgbOn.z / l))
            print("      \(v.name) \(t): where vanilla has no block light (\(dark) pixels), the volume had light at \(darkClamped), which the check took down to the curve at its slack (\(darkLit) of them over 3 levels of luma in the light view, which samples there); the frame as drawn differs from vanilla's at \(darkChanged); vanilla's frames before and after it differ from each other at \(darkDrift)")
            print((darkChanged <= darkDrift ? "ok    " : "FAIL  ") + "nothing glows where vanilla's block light is 0: the colored frame differs from vanilla's there no more than vanilla's own frames do (\(v.name), \(t))")
        }
    }
    // A block change: a soul torch placed in the mixing area, then broken (as the Java side sends them), the frames each
    // takes to settle, and the check against the CPU after both.
    use(views[0])
    setLightmap(skyFactor: 0.2)
    _ = settle()
    let soulCode = Int32(0 | 10 << 4) | (clClassify("minecraft:soul_torch") << 8)
    let ex: Int32 = -12, ey: Int32 = 200, ez: Int32 = 32
    clBlock(0, ex, ey, ez, soulCode)
    var settleTrace: [Int] = []
    for _ in 0..<16 { _ = clBench(cam.x, cam.y, cam.z, 1, &bench); settleTrace.append(Int(bench[1])) }
    print("      soul torch placed at (\(ex), \(ey), \(ez)): bricks the flood fill ran over, frame by frame: \(settleTrace)")
    verify("after placing a soul torch")
    let placed = capture(half, sunAngle: midnight, relight: true, extras: false)
    writePNG(placed.color, Int(W), Int(H), "cl-\(views[0].name)-midnight-edit-placed.png")
    clBlock(0, ex, ey, ez, 0)
    settleTrace = []
    for _ in 0..<16 { _ = clBench(cam.x, cam.y, cam.z, 1, &bench); settleTrace.append(Int(bench[1])) }
    print("      and broken: \(settleTrace)")
    verify("after breaking it")
    if timing {
        // The volume's work at the panel's resolution doesn't depend on it; the relight's does.
        func stat(_ s: [Double]) -> (Double, Double) { let t = s.sorted(); return t.isEmpty ? (-1, -1) : (t[t.count / 2], t[0]) }
        func series(_ k: Int) -> String { (0..<k).map { String(format: "%.3f/%d", bench[2 * $0], Int(bench[2 * $0 + 1])) }.joined(separator: " ") }
        clInvalidate()
        _ = clBench(cam.x, cam.y, cam.z, 12, &bench)
        print("      from scratch (every section uploaded, light from nothing), GPU ms/bricks by frame: \(series(12))")
        _ = settle()
        _ = clBench(cam.x, cam.y, cam.z, 60, &bench)
        print(String(format: "      still: median %.3f ms (fastest %.3f) a frame, %d bricks", stat((0..<60).map { bench[2 * $0] }).0,
                     stat((0..<60).map { bench[2 * $0] }).1, Int(bench[1])))
        // Moving as the real-terrain flight does (20 blocks a second at 120 Hz), then 6 times faster: the volume moves a
        // section at a time, and the sections that enter it are uploaded and lit.
        for (speed, label) in [(1.0 / 6, "20 blocks/s"), (1.0, "120 blocks/s")] {
            var ms: [Double] = [], bricks: [Double] = []
            var p = cam
            for _ in 0..<240 {
                p.x += speed
                _ = clBench(p.x, p.y, p.z, 1, &bench)
                ms.append(bench[0]); bricks.append(bench[1])
            }
            let (med, fast) = stat(ms)
            print(String(format: "      moving at %@ for 240 frames: median %.3f ms (fastest %.3f, slowest %.3f), mean %.3f; flood fill in %d frames, %.0f bricks on average there",
                         label, med, fast, ms.max() ?? 0, ms.reduce(0, +) / Double(ms.count), bricks.filter { $0 > 0 }.count,
                         bricks.reduce(0, +) / Double(max(bricks.filter { $0 > 0 }.count, 1))))
        }
        _ = settle()
        clBlock(0, ex, ey, ez, soulCode)
        _ = clBench(cam.x, cam.y, cam.z, 8, &bench)
        print("      a torch placed, by frame: \(series(8))")
        clBlock(0, ex, ey, ez, 0)
        _ = settle()
        // The relight with and without the colored light, at the panel's resolution, in its own pass and in the
        // anti-aliasing's resolve (the code is in either way: "without" is the volume switched off).
        let full = targets(3456, 2234)
        setLightmap(skyFactor: 0.2)
        for _ in 0..<8 { frame(full, sunAngle: midnight, relight: true, extras: false) }
        frame(full, sunAngle: midnight, relight: false, extras: false)
        let mats = matrices(width: full.w, height: full.h)
        litLevelPass?()
        var h = full.color
        var clear: [Float] = [0.62, 0.75, 0.95, 1]
        _ = passBegin(&h, 1, 0, &clear, full.depth, 0, 0, 0, 0, full.w, full.h)
        var p = mats + [0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, 0, 1]
        var c = [cam.x, cam.y, cam.z]
        _ = lodDraw(&p, &c)
        passEnd()
        submit(submitIndex); _ = waitSubmit(submitIndex, 10_000_000_000); submitIndex += 1
        var lp = mats + [midnight, 0.62, 0.75, 0.95, 0, 1e9, 2e9, 1e9, 2e9, sky ? 1 : 0]
        var two = [Double](repeating: 0, count: 2), four = [Double](repeating: 0, count: 4)
        for round in 0..<2 {
            for on in [Int32(1), 0] {
                _ = clSwitch(on, 100)
                litTime?(full.color, full.depth, &lp, lightmap, 60, &two)
                litTimeTaa?(full.color, full.depth, &lp, lightmap, 60, &four)
                print(String(format: "      round %d, colored light %@: relight pass %.3f ms (fastest %.3f); anti-aliasing resolve with the relight %.3f (fastest %.3f), without %.3f (fastest %.3f)",
                             round, on == 1 ? "on " : "off", two[0], two[1], four[0], four[2], four[1], four[3]))
            }
        }
        _ = clSwitch(1, 100)
    }
    print("done")
    exit(0)
}

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
