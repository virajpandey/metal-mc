// Offline check of the sky, aerial perspective and tone curve (Sky.swift, Hdr.swift), no game: renders views through
// libMetalMCNative's debug entry points at several times of day and in rain, writes PNGs, and prints NaN counts,
// colors, table and per-frame pass timings on this GPU, and the tone curve at several headrooms.
// usage: swift build -c release; swiftc -O tools/skytest.swift -o .build/skytest
//        .build/skytest .build/release/libMetalMCNative.dylib <output dir>
import CoreGraphics
import Foundation
import ImageIO

typealias RenderFn = @convention(c) (Float, Float, Double, Float, UnsafePointer<Float>, Int32, Int32,
                                     UnsafeMutablePointer<UInt8>, UnsafeMutablePointer<Float>, UnsafeMutablePointer<Double>) -> Int32
typealias TimeFn = @convention(c) (Int32, Int32, Int32, UnsafeMutablePointer<Double>) -> Int32
typealias CurveFn = @convention(c) (UnsafePointer<Float>, UnsafeMutablePointer<Float>, Int32, Float) -> Int32

let args = CommandLine.arguments
guard args.count >= 3, let lib = dlopen(args[1], RTLD_NOW) else {
    print("usage: skytest <libMetalMCNative.dylib> <output dir>; dlopen: \(String(cString: dlerror()))")
    exit(1)
}
let outDir = URL(fileURLWithPath: args[2])
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
let render = unsafeBitCast(dlsym(lib, "mmc_debug_sky_render"), to: RenderFn.self)
let timePasses = unsafeBitCast(dlsym(lib, "mmc_debug_sky_time"), to: TimeFn.self)
let curve = unsafeBitCast(dlsym(lib, "mmc_debug_hdr_curve"), to: CurveFn.self)

func writePNG(_ rgba: [UInt8], _ w: Int, _ h: Int, _ url: URL) {
    let provider = CGDataProvider(data: Data(rgba) as CFData)!
    let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4,
                      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                      provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)!
    let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)!
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
}

let deg = Float.pi / 180
/// Vanilla's sun angle for a sun elevation: its direction is (-sin a, cos a, 0), so the elevation is 90 degrees - a.
func sunAngle(elevation: Float) -> Float { (90 - elevation) * deg }

struct Case { let name: String; let elevation: Float; let rainBrightness: Float }
let cases = [
    Case(name: "noon", elevation: 90, rainBrightness: 1),
    Case(name: "afternoon", elevation: 35, rainBrightness: 1),
    Case(name: "golden", elevation: 8, rainBrightness: 1),
    Case(name: "sunset", elevation: 1, rainBrightness: 1),
    Case(name: "twilight", elevation: -4, rainBrightness: 1),
    Case(name: "bluehour", elevation: -9, rainBrightness: 1),
    Case(name: "night", elevation: -30, rainBrightness: 1),
    Case(name: "rain", elevation: 35, rainBrightness: 0),
]
struct View { let name: String; let window: [Float]; let w: Int; let h: Int }
let views = [
    // The whole sphere: azimuth from the sun -180..180, elevation 90..-90.
    View(name: "pano", window: [-180 * deg, 180 * deg, 90 * deg, -90 * deg], w: 1536, h: 768),
    // Toward the sun and away from it, near the horizon (the ridges are at -150, -90, -30, 30 and 90 degrees).
    View(name: "sun", window: [-60 * deg, 60 * deg, 25 * deg, -6 * deg], w: 1600, h: 420),
    View(name: "away", window: [120 * deg, 240 * deg, 25 * deg, -6 * deg], w: 1600, h: 420),
]
let camY = 150.0

func lum(_ p: UnsafePointer<Float>) -> Float { 0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2] }

for c in cases {
    for v in views {
        var rgba = [UInt8](repeating: 0, count: v.w * v.h * 4)
        var lin = [Float](repeating: 0, count: v.w * v.h * 4)
        var times = [Double](repeating: 0, count: 5)
        let ok = v.window.withUnsafeBufferPointer { win in
            render(sunAngle(elevation: c.elevation), c.rainBrightness, camY, 1, win.baseAddress!, Int32(v.w), Int32(v.h), &rgba, &lin, &times)
        }
        guard ok == 1 else { print("\(c.name) \(v.name): render failed"); exit(1) }
        writePNG(rgba, v.w, v.h, outDir.appendingPathComponent("\(c.name)-\(v.name).png"))
        var nan = 0, maxL: Float = 0
        for i in 0..<(v.w * v.h * 4) where i % 4 < 3 {
            if !lin[i].isFinite { nan += 1 } else { maxL = max(maxL, lin[i]) }
        }
        var line = "\(c.name) \(v.name): NaN/Inf \(nan), brightest channel \(String(format: "%.2f", maxL))"
        if v.name == "pano" {
            // Sky colors (what an 8-bit target holds): the zenith, and the horizon toward and away from the sun.
            func px(_ x: Int, _ y: Int) -> String {
                let i = (y * v.w + x) * 4
                return "(\(rgba[i]), \(rgba[i + 1]), \(rgba[i + 2]))"
            }
            let horizonRow = v.h / 2 - 2   // just above the horizon
            line += "; zenith \(px(v.w / 2, 4)), horizon at the sun \(px(v.w / 2, horizonRow)), opposite \(px(2, horizonRow))"
            // Kinks: the sky's second difference between rows, in 8-bit steps of its sRGB-encoded luminance, up the
            // column 160 degrees from the sun, clear of the ridges (a lookup table's rows showing would stand out against
            // the smooth rest), and an image of it everywhere (x 64) to see where any are.
            func enc(_ x: Float) -> Float { 255 * (x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055) }
            var worst: Float = 0, worstRow = 0
            var kinks = [UInt8](repeating: 0, count: v.w * v.h * 4)
            lin.withUnsafeBufferPointer { b in
                let p = b.baseAddress!
                for y in 1..<(v.h - 1) {
                    for x in 0..<v.w {
                        let a = enc(lum(p + ((y - 1) * v.w + x) * 4)), m = enc(lum(p + (y * v.w + x) * 4)), n = enc(lum(p + ((y + 1) * v.w + x) * 4))
                        let d2 = abs(a - 2 * m + n)
                        let g = UInt8(min(255, d2 * 64))
                        kinks[(y * v.w + x) * 4] = g; kinks[(y * v.w + x) * 4 + 1] = g; kinks[(y * v.w + x) * 4 + 2] = g
                        if x == Int(0.944 * Float(v.w)) && y >= 20 && y < v.h / 2 - 3 && d2 > worst { worst = d2; worstRow = y }
                    }
                }
            }
            writePNG(kinks, v.w, v.h, outDir.appendingPathComponent("\(c.name)-\(v.name)-kinks.png"))
            line += String(format: "; largest 2nd difference up the sky %.3f steps (row %d)", worst, worstRow)
        }
        print(line)
    }
}

// Table timings: rebuild everything 20 times for the afternoon and take the medians.
var runs: [[Double]] = [[], [], [], [], []]
for _ in 0..<20 {
    var rgba = [UInt8](repeating: 0, count: 64 * 32 * 4)
    var lin = [Float](repeating: 0, count: 64 * 32 * 4)
    var times = [Double](repeating: 0, count: 5)
    let win: [Float] = [-180 * deg, 180 * deg, 90 * deg, -90 * deg]
    _ = render(sunAngle(elevation: 35), 1, camY, 1, win, 64, 32, &rgba, &lin, &times)
    for k in 0..<5 { runs[k].append(times[k]) }
}
func median(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
print(String(format: "tables (GPU ms, median of 20): transmittance %.3f, multiple scattering %.3f, sky view %.3f, aerial perspective %.3f",
             median(runs[0]), median(runs[1]), median(runs[2]), median(runs[3])))

// Per-frame passes at the panel's native size.
var pass = [Double](repeating: 0, count: 6)
if timePasses(3456, 2234, 30, &pass) == 1 {
    print(String(format: "per frame at 3456x2234 (GPU ms, median of 30): sky view + aerial perspective rebuild %.3f; sky pass working out every pixel %.3f, "
                 + "from the quarter-resolution sky %.3f; aerial perspective as its own pass %.3f; anti-aliasing %.3f, with the aerial perspective in its load %.3f",
                 pass[0], pass[1], pass[3], pass[2], pass[4], pass[5]))
} else {
    print("pass timing failed")
}

// The tone curve at several headrooms: identity below the knee, monotonic, never past the headroom.
let inputs: [Float] = [0, 0.1, 0.5, 0.85, 0.9, 1, 1.2, 1.5, 2, 3, 5, 10, 30, 100, 1000, 30000]
for H: Float in [1, 1.5, 3, 8, 16] {
    var inp = [Float](), out = [Float](repeating: 0, count: inputs.count * 4)
    for x in inputs { inp += [x, x, x, 0] }
    guard curve(inp, &out, Int32(inputs.count), H) == 1 else { print("curve failed"); exit(1) }
    var monotonic = true, bounded = true
    for i in 0..<inputs.count {
        if i > 0 && out[4 * i] < out[4 * (i - 1)] - 1e-5 { monotonic = false }
        if out[4 * i] > H + 1e-4 { bounded = false }
    }
    let shown = zip(inputs, stride(from: 0, to: out.count, by: 4).map { out[$0] }).map { String(format: "%g->%.3f", $0, $1) }
    print("tone curve H=\(H): monotonic \(monotonic), within headroom \(bounded): " + shown.joined(separator: " "))
}
// A saturated color above the knee keeps its hue at first and whitens as it gets brighter.
var sat = [Float](), satOut = [Float](repeating: 0, count: 12)
for s: Float in [1, 4, 40] { sat += [s, 0.4 * s, 0.1 * s, 0] }
_ = curve(sat, &satOut, 3, 4)
print("orange 1/4/40 at H=4: " + (0..<3).map { String(format: "(%.2f, %.2f, %.2f)", satOut[4 * $0], satOut[4 * $0 + 1], satOut[4 * $0 + 2]) }.joined(separator: " "))
