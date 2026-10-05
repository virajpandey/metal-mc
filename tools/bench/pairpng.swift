// Side-by-side PNGs for before/after comparisons: each pair of inputs scaled to a common height and put next to each
// other, left then right, optionally cropped first.
// usage: swiftc -O tools/bench/pairpng.swift -o bench_out/pairpng
//        bench_out/pairpng <left.png> <right.png> <out.png> [height (default: the left's)] [crop x,y,w,h in the inputs' pixels]
import CoreGraphics
import Foundation
import ImageIO

let a = CommandLine.arguments
guard a.count >= 4 else { print("usage: pairpng <left.png> <right.png> <out.png> [height] [x,y,w,h]"); exit(1) }
func load(_ p: String) -> CGImage {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: p) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { print("can't read \(p)"); exit(1) }
    return img
}
var left = load(a[1]), right = load(a[2])
if a.count >= 6 {
    let c = a[5].split(separator: ",").map { Int($0)! }
    let r = CGRect(x: c[0], y: c[1], width: c[2], height: c[3])
    left = left.cropping(to: r) ?? left
    right = right.cropping(to: r) ?? right
}
let h = a.count >= 5 ? Int(a[4])! : left.height
let wl = left.width * h / left.height, wr = right.width * h / right.height
let gap = 8
guard let ctx = CGContext(data: nil, width: wl + gap + wr, height: h, bitsPerComponent: 8, bytesPerRow: 0,
                          space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { exit(1) }
ctx.interpolationQuality = .high
ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
ctx.fill(CGRect(x: 0, y: 0, width: wl + gap + wr, height: h))
ctx.draw(left, in: CGRect(x: 0, y: 0, width: wl, height: h))
ctx.draw(right, in: CGRect(x: wl + gap, y: 0, width: wr, height: h))
let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: a[3]) as CFURL, "public.png" as CFString, 1, nil)!
CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
CGImageDestinationFinalize(dest)
print("wrote \(a[3]) (\(wl + gap + wr) x \(h))")
