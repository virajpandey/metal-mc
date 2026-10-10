// Offline check of the volumetric clouds (Clouds.swift, METALMC_EXP=sky,clouds), no game: renders views through
// libMetalMCNative's mmc_debug_clouds_render (the sky's tables, the clouds' frames as the game makes them, accumulated
// over frames with the camera still, then the sky or a flat grass plain at sea level with the clouds' shadows, the clouds
// composited as the anti-aliasing's resolve does, AgX), writes PNGs, and times the clouds' frame at the panel's resolution.
// usage: swift build -c release --product MetalMCNative; swiftc -O tools/cloudtest.swift -o .build/cloudtest
//        .build/cloudtest .build/release/libMetalMCNative.dylib <output dir> [views...|time]
//   views: names from the list below (default all); "time": only the timing at 3456 x 2234.
//   CLOUDTEST_W, CLOUDTEST_H: picture size (1152 x 745); CLOUDTEST_FRAMES: frames accumulated (48); CLOUDTEST_COVER:
//   coverage for every view (default each view's, nan: the setting's); CLOUDTEST_VIEW: the debug view (METALMC_CLOUDVIEW's
//   numbers); CLOUDTEST_TIME: the wind's time (s).
import CoreGraphics
import Foundation
import ImageIO

setenv("METALMC_EXP", ProcessInfo.processInfo.environment["CLOUDTEST_EXP"] ?? "sky,clouds,post", 1)
let args = CommandLine.arguments
guard args.count >= 3, let lib = dlopen(args[1], RTLD_NOW) else {
    print("usage: cloudtest <libMetalMCNative.dylib> <output dir> [views...|time]")
    exit(1)
}
typealias RenderFn = @convention(c) (UnsafePointer<Double>, Int32, Int32, UnsafeMutablePointer<UInt8>, UnsafeMutablePointer<Double>) -> Int32
func fn<T>(_ name: String, _: T.Type) -> T {
    guard let p = dlsym(lib, name) else { print("missing \(name)"); exit(1) }
    return unsafeBitCast(p, to: T.self)
}
let ctxInit = fn("mmc_ctx_init", (@convention(c) () -> Int32).self)
let render = fn("mmc_debug_clouds_render", RenderFn.self)
guard ctxInit() == 1 else { print("context failed"); exit(1) }
let outDir = URL(fileURLWithPath: args[2])
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
let env = ProcessInfo.processInfo.environment
let W = Int(env["CLOUDTEST_W"] ?? "") ?? 1152, H = Int(env["CLOUDTEST_H"] ?? "") ?? 745
let frames = Double(env["CLOUDTEST_FRAMES"] ?? "") ?? 48
let coverAll = Double(env["CLOUDTEST_COVER"] ?? "")
let viewAll = Double(env["CLOUDTEST_VIEW"] ?? "")
let timeAll = Double(env["CLOUDTEST_TIME"] ?? "")

func writePNG(_ rgba: [UInt8], _ w: Int, _ h: Int, _ url: URL) {
    let provider = CGDataProvider(data: Data(rgba) as CFData)!
    let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                      provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
}

/// Vanilla's sun angle for a sun elevation (degrees): its direction is (-sin a, cos a, 0), west for a > 0.
func sunAngle(_ elevation: Double) -> Double { (90 - elevation) * .pi / 180 }

struct View {
    let name: String
    var elev: Double, rain: Double = 1, pos: (Double, Double, Double), yaw: Double, pitch: Double, fov: Double = 70
    var ground: Bool = true, exposure: Double = 1, cover: Double = .nan, view: Double = 0, time: Double = 1000
}
// Yaw: Minecraft's (0 south, 90 west: toward the afternoon sun, 270 east), pitch > 0 looks down.
let all: [View] = [
    View(name: "noon", elev: 70, pos: (8, 166, 8), yaw: 281.5, pitch: -8),
    View(name: "noon-up", elev: 70, pos: (8, 166, 8), yaw: 200, pitch: -55, fov: 80),
    View(name: "afternoon-sun", elev: 35, pos: (8, 166, 8), yaw: 90, pitch: -8),
    View(name: "afternoon-away", elev: 35, pos: (8, 166, 8), yaw: 270, pitch: -8),
    View(name: "golden", elev: 8, pos: (-640, 80, 4), yaw: 90, pitch: -6),
    View(name: "sunset", elev: 4.6, pos: (-640, 80, 4), yaw: 90, pitch: -3, exposure: 1.4),
    View(name: "sunset-away", elev: 4.6, pos: (-640, 80, 4), yaw: 270, pitch: -6, exposure: 1.4),
    View(name: "dusk", elev: -3, pos: (-640, 80, 4), yaw: 90, pitch: -6, exposure: 2),
    View(name: "night", elev: -45, pos: (64, 76, 168), yaw: 134, pitch: -10, exposure: 3),
    View(name: "rain", elev: 60, rain: 0, pos: (20, 72, 178), yaw: 160, pitch: -10),
    View(name: "overcast", elev: 60, pos: (20, 72, 178), yaw: 160, pitch: -10, cover: 1.6),
    View(name: "clear", elev: 60, pos: (20, 72, 178), yaw: 160, pitch: -10, cover: -0.3),
    View(name: "shadows", elev: 50, pos: (8, 480, 8), yaw: 120, pitch: 40, fov: 70),
    View(name: "above", elev: 40, pos: (8, 3600, 8), yaw: 100, pitch: 25),
    View(name: "inside", elev: 40, pos: (8, 1900, 8), yaw: 100, pitch: 0),
    View(name: "mountain", elev: 60, pos: (1064, 174, 230), yaw: 90, pitch: -4),
]
let wanted = Set(args.dropFirst(3))
let timing = wanted.contains("time")
var times = [Double](repeating: 0, count: 4)
if !timing {
    for v in all where wanted.isEmpty || wanted.contains(v.name) {
        var rgba = [UInt8](repeating: 0, count: W * H * 4)
        let cover = coverAll ?? v.cover
        let p: [Double] = [sunAngle(v.elev), v.rain, v.pos.0, v.pos.1, v.pos.2, v.yaw, v.pitch, v.fov, timeAll ?? v.time, frames,
                           v.ground ? 1 : 0, v.exposure, cover, viewAll ?? v.view]
        let t0 = Date()
        guard render(p, Int32(W), Int32(H), &rgba, &times) == 1 else { print("\(v.name): render failed"); exit(1) }
        writePNG(rgba, W, H, outDir.appendingPathComponent("\(v.name).png"))
        // Mean 8-bit luma of the top fifth (the sky and clouds) and the whole frame.
        var top = 0.0, allL = 0.0
        for y in 0..<H {
            for x in 0..<W {
                let i = (y * W + x) * 4
                let l = 0.2126 * Double(rgba[i]) + 0.7152 * Double(rgba[i + 1]) + 0.0722 * Double(rgba[i + 2])
                allL += l
                if y < H / 5 { top += l }
            }
        }
        print(String(format: "%@: luma top fifth %.1f, frame %.1f; clouds' last frame %.3f ms GPU at %dx%d, view %.3f ms (%.1f s)",
                     v.name, top / Double(W * H / 5), allL / Double(W * H), times[0], W, H, times[1], Date().timeIntervalSince(t0)))
    }
} else {
    // The clouds' frame (sky light, shadow map, the march at half resolution) at the panel's resolution: looking up into
    // scattered cumulus at noon (most of the screen sky), toward the horizon, and from above the layer. Median of 30.
    for (name, v) in [("up", all[1]), ("horizon", all[0]), ("sunset", all[5]), ("above", all[13]), ("overcast", all[10])] {
        var ms: [Double] = [], shadowMs: [Double] = [], marchMs: [Double] = []
        var rgba = [UInt8](repeating: 0, count: 3456 * 2234 * 4)
        for k in 0..<4 {
            let p: [Double] = [sunAngle(v.elev), v.rain, v.pos.0, v.pos.1, v.pos.2, v.yaw, v.pitch, v.fov, v.time + Double(k), 16,
                               v.ground ? 1 : 0, v.exposure, coverAll ?? v.cover, 0]
            guard render(p, 3456, 2234, &rgba, &times) == 1 else { print("render failed"); exit(1) }
            ms.append(times[0]); shadowMs.append(times[2]); marchMs.append(times[3])
        }
        // Each render's frames: the last one's time; repeat for a median.
        for k in 0..<26 {
            let p: [Double] = [sunAngle(v.elev), v.rain, v.pos.0, v.pos.1, v.pos.2, v.yaw, v.pitch, v.fov, v.time + Double(k), 16,
                               v.ground ? 1 : 0, v.exposure, coverAll ?? v.cover, 0]
            guard render(p, 3456, 2234, &rgba, &times) == 1 else { print("render failed"); exit(1) }
            ms.append(times[0]); shadowMs.append(times[2]); marchMs.append(times[3])
        }
        ms.sort(); shadowMs.sort(); marchMs.sort()
        print(String(format: "%@: the clouds' frame at 3456x2234: median %.3f ms, fastest %.3f, slowest %.3f (GPU, %d runs); sky light and shadow map median %.3f (fastest %.3f), march %.3f (%.3f)", name,
                     ms[ms.count / 2], ms[0], ms[ms.count - 1], ms.count, shadowMs[shadowMs.count / 2], shadowMs[0], marchMs[marchMs.count / 2], marchMs[0]))
    }
}
