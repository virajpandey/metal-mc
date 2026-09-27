// Fidelity score for the far-terrain LOD. Compares three screenshots of the same pose, taken with fog off
// (-Pfidelity=1): A = vanilla at render distance 32 (the reference: real terrain out to 512 blocks),
// B = vanilla at render distance 12 (what vanilla draws without LOD), C = render distance 12 + LOD.
// The band is where A and B differ: terrain between 192 and 512 blocks that the LOD has to supply.
// Inside the band it reports the mean color error of C against A and the share of holes (band pixels
// where C still shows what B shows, i.e. no terrain). Images are box-downsampled 4x first, so the score
// measures shapes, colors and lighting rather than texel-level texture differences. It also reports the
// error at 16x (regions of 16 x 16 screen pixels: overall color and shading, insensitive to the exact
// position of edges, which LOD voxels can't match block for block) and the mean signed color bias.
//
// usage: fidscore <screenshot dir> <labelA> <labelB> <labelC> [step...]
//        (files are <label>-tour-NN-<step>.png; default: every step found for labelA)
// build: swiftc -O tools/fidscore.swift -o <out>
import CoreGraphics
import Foundation
import ImageIO

let args = CommandLine.arguments
guard args.count >= 5 else { print("usage: fidscore <dir> <labelA> <labelB> <labelC> [step...]"); exit(2) }
let dir = args[1], labelA = args[2], labelB = args[3], labelC = args[4]
let scale = 4

/// Loads a PNG and box-downsamples it by `scale`, returning RGB bytes.
func load(_ path: String) -> (w: Int, h: Int, px: [UInt8])? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let W = img.width, H = img.height
    var full = [UInt8](repeating: 0, count: W * H * 4)
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: &full, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: cs,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return nil }
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: W, height: H))
    let w = W / scale, h = H / scale
    var out = [UInt8](repeating: 0, count: w * h * 3)
    full.withUnsafeBufferPointer { f in
        for y in 0..<h {
            for x in 0..<w {
                var r = 0, g = 0, b = 0
                for dy in 0..<scale {
                    var i = ((y * scale + dy) * W + x * scale) * 4
                    for _ in 0..<scale { r += Int(f[i]); g += Int(f[i + 1]); b += Int(f[i + 2]); i += 4 }
                }
                let n = scale * scale, o = (y * w + x) * 3
                out[o] = UInt8(r / n); out[o + 1] = UInt8(g / n); out[o + 2] = UInt8(b / n)
            }
        }
    }
    return (w, h, out)
}

// FIDSCORE_HEATMAP=<dir>: writes <dir>/<step>.png, the 4x-downsampled band error (brighter = worse,
// red = LOD brighter than the reference, blue = darker; outside the band dimmed).
let heatDir = ProcessInfo.processInfo.environment["FIDSCORE_HEATMAP"]
func writeHeatmap(_ path: String, w: Int, h: Int, rgba: [UInt8]) {
    var data = rgba
    let cs = CGColorSpaceCreateDeviceRGB()
    guard let ctx = CGContext(data: &data, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
                              bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue), let img = ctx.makeImage(),
          let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil) else { return }
    CGImageDestinationAddImage(dest, img, nil)
    CGImageDestinationFinalize(dest)
}

var steps: [String] = Array(args.dropFirst(5))
if steps.isEmpty {
    let files = (try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []
    steps = files.filter { $0.hasPrefix("\(labelA)-tour-") && $0.hasSuffix(".png") }
        .map { String($0.dropFirst("\(labelA)-".count).dropLast(4)) }.sorted()
}

/// Further 4x box downsample of a 4x image, plus a coverage map of band pixels.
func coarse(_ img: (w: Int, h: Int, px: [UInt8])) -> (w: Int, h: Int, px: [Double]) {
    let w = img.w / 4, h = img.h / 4
    var out = [Double](repeating: 0, count: w * h * 3)
    for y in 0..<h {
        for x in 0..<w {
            for k in 0..<3 {
                var s = 0
                for dy in 0..<4 { for dx in 0..<4 { s += Int(img.px[((y * 4 + dy) * img.w + x * 4 + dx) * 3 + k]) } }
                out[(y * w + x) * 3 + k] = Double(s) / 16
            }
        }
    }
    return (w, h, out)
}

var totBand = 0, totHoles = 0, totBig = 0
var totErr = 0.0
var totCoarseErr = 0.0, totCoarseN = 0
var bias = [0.0, 0.0, 0.0]
print(String(format: "%-26@ %8@ %8@ %8@ %8@ %8@", "step" as NSString, "band%" as NSString, "error" as NSString, "err16x" as NSString, "holes%" as NSString, "big%" as NSString))
for step in steps {
    guard let a = load("\(dir)/\(labelA)-\(step).png"), let b = load("\(dir)/\(labelB)-\(step).png"),
          let c = load("\(dir)/\(labelC)-\(step).png"), a.w == b.w, a.w == c.w, a.h == b.h, a.h == c.h else {
        print("\(step): missing or mismatched images"); continue
    }
    var band = 0, holes = 0, big = 0
    var err = 0.0
    var inBand = [Bool](repeating: false, count: a.w * a.h)
    for i in 0..<(a.w * a.h) {
        let o = i * 3
        var dab = 0, dcb = 0, dca = 0, sum = 0
        for k in 0..<3 {
            let va = Int(a.px[o + k]), vb = Int(b.px[o + k]), vc = Int(c.px[o + k])
            dab = max(dab, abs(va - vb)); dcb = max(dcb, abs(vc - vb))
            let d = abs(vc - va); dca = max(dca, d); sum += d
        }
        if dab <= 24 { continue }
        band += 1
        inBand[i] = true
        err += Double(sum) / 3
        for k in 0..<3 { bias[k] += Double(Int(c.px[o + k]) - Int(a.px[o + k])) }
        if dcb < 12 { holes += 1 }
        if dca > 48 { big += 1 }
    }
    if let heatDir {
        var rgba = [UInt8](repeating: 0, count: a.w * a.h * 4)
        for i in 0..<(a.w * a.h) {
            let o = i * 3
            if inBand[i] && (0..<3).allSatisfy({ abs(Int(c.px[o + $0]) - Int(b.px[o + $0])) < 12 }) {
                rgba[i * 4] = 255; rgba[i * 4 + 1] = 255   // hole: yellow
            } else if inBand[i] {
                let d = (Int(c.px[o]) + Int(c.px[o + 1]) + Int(c.px[o + 2])) - (Int(a.px[o]) + Int(a.px[o + 1]) + Int(a.px[o + 2]))
                let m = min(255, abs(d) * 2)
                rgba[i * 4] = UInt8(d > 0 ? m : 0); rgba[i * 4 + 2] = UInt8(d < 0 ? m : 0)
                rgba[i * 4 + 1] = UInt8(min(255, (abs(Int(c.px[o]) - Int(a.px[o])) + abs(Int(c.px[o + 1]) - Int(a.px[o + 1])) + abs(Int(c.px[o + 2]) - Int(a.px[o + 2]))) / 2))
            } else {
                for k in 0..<3 { rgba[i * 4 + k] = a.px[o + k] / 4 }
            }
        }
        writeHeatmap("\(heatDir)/\(step).png", w: a.w, h: a.h, rgba: rgba)
    }
    // 16x: blocks of 4 x 4 downsampled pixels that are entirely inside the band.
    let ca = coarse(a), cc = coarse(c)
    var cErr = 0.0, cN = 0
    for y in 0..<ca.h {
        for x in 0..<ca.w {
            var all = true
            for dy in 0..<4 where all { for dx in 0..<4 where !inBand[(y * 4 + dy) * a.w + x * 4 + dx] { all = false } }
            if !all { continue }
            let o = (y * ca.w + x) * 3
            cErr += (abs(cc.px[o] - ca.px[o]) + abs(cc.px[o + 1] - ca.px[o + 1]) + abs(cc.px[o + 2] - ca.px[o + 2])) / 3
            cN += 1
        }
    }
    totCoarseErr += cErr; totCoarseN += cN
    let n = a.w * a.h
    print(String(format: "%-26@ %7.1f%% %8.2f %8.2f %7.2f%% %7.2f%%", step as NSString, 100.0 * Double(band) / Double(n),
                 band > 0 ? err / Double(band) : 0, cN > 0 ? cErr / Double(cN) : 0, band > 0 ? 100.0 * Double(holes) / Double(band) : 0,
                 band > 0 ? 100.0 * Double(big) / Double(band) : 0))
    totBand += band; totHoles += holes; totBig += big; totErr += err
}
let nb = Double(max(1, totBand))
print(String(format: "SCORE %@: error %.2f  err16x %.2f  holes %.2f%%  big %.2f%%  bias RGB %+.1f %+.1f %+.1f  (band pixels %d)",
             labelC as NSString, totErr / nb, totCoarseN > 0 ? totCoarseErr / Double(totCoarseN) : 0,
             100.0 * Double(totHoles) / nb, 100.0 * Double(totBig) / nb, bias[0] / nb, bias[1] / nb, bias[2] / nb, totBand))
