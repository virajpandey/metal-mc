// Offline test of the temporal upscaler (TAAU, Sources/MetalMCNative/TaaUpscale.swift), no game and no window: a still
// view of a test image (a zone plate whose frequency rises past what the rendered frame can hold, and a dark bar with a
// hard edge) drawn with the jitter and resolved by the real kernels, scored against the image box-filtered over each
// output pixel. Compares TAAU at several scales (and with the jitter's sign flipped, which must score worse), the plain
// anti-aliasing at the output's resolution, and a bilinear upscale without jitter. Writes PNGs of each.
// usage: swift build -c release --product MetalMCNative; swiftc -O tools/taautest.swift -o .build/taautest
//        METALMC_EXP=lit,nearchunks,rtshadows,sky,gi,water,coloredlight bash tools/bench/gpuwait.sh \
//          .build/taautest .build/release/libMetalMCNative.dylib <output dir> [size]
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
guard args.count >= 3 else { print("usage: taautest <dylib> <output dir> [size]"); exit(1) }
guard let lib = dlopen(args[1], RTLD_NOW) else { print("dlopen failed: \(String(cString: dlerror()))"); exit(1) }
let outDir = URL(fileURLWithPath: args[2])
try? FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
let size = args.count > 3 ? Int(args[3]) ?? 512 : 512

typealias TestF = @convention(c) (Int32, Int32, Float, Int32, Float, Int32, UnsafeMutablePointer<Float>, UnsafeMutablePointer<Double>) -> Double
guard let sym = dlsym(lib, "mmc_debug_taau_test") else { print("missing mmc_debug_taau_test"); exit(1) }
let test = unsafeBitCast(sym, to: TestF.self)

func writePNG(_ v: [Float], _ w: Int, _ h: Int, _ name: String) {
    var bytes = [UInt8](repeating: 0, count: w * h)
    // Row 0 of the result is the image's bottom row (y up); PNG rows go down.
    for y in 0..<h { for x in 0..<w { bytes[(h - 1 - y) * w + x] = UInt8(max(0, min(255, (v[y * w + x] * 255).rounded()))) } }
    guard let provider = CGDataProvider(data: Data(bytes) as CFData),
          let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 8, bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(),
                            bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(outDir.appendingPathComponent(name) as CFURL, UTType.png.identifier as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
}

var failures = 0
func check(_ ok: Bool, _ what: String) {
    print((ok ? "ok    " : "FAIL  ") + what)
    if !ok { failures += 1 }
}

var buf = [Float](repeating: 0, count: size * size)
var stats = [Double](repeating: 0, count: 3)
let compileFailed = test(Int32(size), Int32(size), 1, 0, 1, 3, &buf, &stats)
check(compileFailed == 0, "every variant of the upscaling resolve compiles under METALMC_EXP=\(ProcessInfo.processInfo.environment["METALMC_EXP"] ?? "") (\(Int(compileFailed)) failed)")

func run(_ label: String, scale: Float, frames: Int, jsign: Float = 1, mode: Int32) -> [Double] {
    let rms = test(Int32(size), Int32(size), scale, Int32(frames), jsign, mode, &buf, &stats)
    print(String(format: "%-34@ RMS error %.4f   (where the drawn frame can hold the image: %.4f, past it: %.4f)", label as NSString, rms, stats[1], stats[2]))
    writePNG(buf, size, size, "taau-\(label).png")
    return [rms, stats[1], stats[2]]
}

let native = run("native-taa", scale: 1, frames: 32, mode: 1)
let bilinear = run("bilinear-0.75", scale: 0.75, frames: 1, mode: 2)
let t75 = run("taau-0.75", scale: 0.75, frames: 32, mode: 0)
let flipped = run("taau-0.75-flipped-jitter", scale: 0.75, frames: 32, jsign: -1, mode: 0)
let t67 = run("taau-0.667", scale: 2.0 / 3.0, frames: 32, mode: 0)
let t60 = run("taau-0.6", scale: 0.6, frames: 32, mode: 0)
let t75short = run("taau-0.75-8frames", scale: 0.75, frames: 8, mode: 0)
check(t75[0] > 0 && t75[0] < bilinear[0], "TAAU at 0.75 beats a bilinear upscale (\(String(format: "%.4f against %.4f", t75[0], bilinear[0])))")
check(t75[0] < flipped[0], "the jitter's sign matters: the right one scores better (\(String(format: "%.4f against %.4f", t75[0], flipped[0])))")
check(t75[2] < bilinear[2], "TAAU holds detail past the drawn frame's Nyquist better than a bilinear upscale (\(String(format: "%.4f against %.4f", t75[2], bilinear[2])))")
// Where the drawn frame can hold the image (past it, the zone plate soon passes even the output's Nyquist, and every
// method's error there is aliasing).
check(t60[1] > t67[1] && t67[1] > t75[1], "error grows as the scale drops, where the drawn frame can hold the image (0.75 \(String(format: "%.4f", t75[1])), 0.667 \(String(format: "%.4f", t67[1])), 0.6 \(String(format: "%.4f", t60[1])))")
print(String(format: "native TAA %.4f, TAAU 0.75 after 8 frames %.4f, after 32 %.4f", native[0], t75short[0], t75[0]))
print(failures == 0 ? "done: all checks pass" : "FAIL  \(failures) checks failed")
exit(failures == 0 ? 0 : 1)
