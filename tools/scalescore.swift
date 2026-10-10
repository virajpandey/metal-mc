// How the level drawn at a render scale and upscaled (TAAU, Sources/MetalMCNative/TaaUpscale.swift) holds up against
// the same view at the panel's resolution: per pair of screenshots of one pose (8-bit luma),
//   - PSNR and RMS error of the scaled shot against the native one, over the whole frame and over the native shot's
//     fine detail (pixels whose 3 x 3 gradient is in its top 10%: foliage edges, the LOD's texture, thin lines);
//   - sharpness: the mean squared gradient (horizontal and vertical neighbor differences) of the scaled shot over the
//     native one's (1: as sharp; under 1 softer; over 1 sharper or noisier);
//   - with a second shot of each (the same still view a little later), shimmer: the RMS difference between a scale's two
//     shots, over the whole frame and over the fine detail (temporal instability the anti-aliasing didn't settle).
// usage: swiftc -O tools/scalescore.swift -o .build/scalescore
//        .build/scalescore <native.png> <scaled.png> [<native-later.png> <scaled-later.png>]
import CoreGraphics
import Foundation
import ImageIO

func luma(_ path: String) -> (Int, Int, [Float]) {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { print("can't read \(path)"); exit(1) }
    let w = img.width, h = img.height
    var px = [UInt8](repeating: 0, count: w * h * 4)
    let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    var y = [Float](repeating: 0, count: w * h)
    for i in 0..<(w * h) { y[i] = 0.2126 * Float(px[4 * i]) + 0.7152 * Float(px[4 * i + 1]) + 0.0722 * Float(px[4 * i + 2]) }
    return (w, h, y)
}

let a = CommandLine.arguments
guard a.count >= 3 else { print("usage: scalescore <native.png> <scaled.png> [<native-later.png> <scaled-later.png>]"); exit(1) }
let (w, h, ref) = luma(a[1])
let (w2, h2, test) = luma(a[2])
guard w == w2, h == h2 else { print("size mismatch: \(w)x\(h) vs \(w2)x\(h2)"); exit(1) }

func gradient(_ y: [Float]) -> [Float] {
    var g = [Float](repeating: 0, count: w * h)
    for j in 1..<(h - 1) {
        for i in 1..<(w - 1) {
            let k = j * w + i
            let dx = y[k + 1] - y[k - 1], dy = y[k + w] - y[k - w]
            g[k] = dx * dx + dy * dy
        }
    }
    return g
}
let gref = gradient(ref), gtest = gradient(test)
let threshold = gref.sorted()[Int(Double(gref.count) * 0.9)]
func rms(_ x: [Float], _ y: [Float], detail: Bool) -> Double {
    var s = 0.0, n = 0
    for k in 0..<(w * h) where !detail || gref[k] > threshold {
        let d = Double(x[k] - y[k])
        s += d * d
        n += 1
    }
    return n > 0 ? (s / Double(n)).squareRoot() : 0
}
func psnr(_ e: Double) -> Double { e > 0 ? 20 * log10(255 / e) : 99 }
let eAll = rms(ref, test, detail: false), eDetail = rms(ref, test, detail: true)
let sharp = gtest.reduce(0.0) { $0 + Double($1) } / max(1e-9, gref.reduce(0.0) { $0 + Double($1) })
print(String(format: "against native: PSNR %.2f dB (RMS %.2f), fine detail PSNR %.2f dB (RMS %.2f); sharpness %.3f",
             psnr(eAll), eAll, psnr(eDetail), eDetail, sharp))
if a.count >= 5 {
    let (_, _, refLater) = luma(a[3])
    let (_, _, testLater) = luma(a[4])
    print(String(format: "shimmer (RMS between two shots of the still view): native %.2f (fine detail %.2f), scaled %.2f (fine detail %.2f)",
                 rms(ref, refLater, detail: false), rms(ref, refLater, detail: true),
                 rms(test, testLater, detail: false), rms(test, testLater, detail: true)))
}
