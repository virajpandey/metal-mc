// Difference of two same-size screenshots: prints the share of changed pixels and the mean change, and writes a
// heatmap (changed pixels bright on the dimmed first image; red = second brighter, blue = second darker).
// usage: imgdiff <a.png> <b.png> <out.png> [threshold, default 8]
// build: swiftc -O tools/bench/imgdiff.swift -o bench_out/imgdiff
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

func load(_ path: String) -> (Int, Int, [UInt8]) {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { fatalError("can't read \(path)") }
    let w = img.width, h = img.height
    var px = [UInt8](repeating: 0, count: w * h * 4)
    let ctx = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    ctx.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
    return (w, h, px)
}

let args = CommandLine.arguments
guard args.count >= 4 else { print("usage: imgdiff <a.png> <b.png> <out.png> [threshold]"); exit(2) }
let (w, h, a) = load(args[1])
let (w2, h2, b) = load(args[2])
guard w == w2, h == h2 else { print("size mismatch: \(w)x\(h) vs \(w2)x\(h2)"); exit(1) }
let threshold = args.count > 4 ? Int(args[4]) ?? 8 : 8
var out = [UInt8](repeating: 0, count: w * h * 4)
var changed = 0, brighter = 0, total = 0.0
for i in 0..<(w * h) {
    let o = 4 * i
    let la = Int(a[o]) + Int(a[o + 1]) + Int(a[o + 2]), lb = Int(b[o]) + Int(b[o + 1]) + Int(b[o + 2])
    let d = abs(Int(a[o]) - Int(b[o])) + abs(Int(a[o + 1]) - Int(b[o + 1])) + abs(Int(a[o + 2]) - Int(b[o + 2]))
    total += Double(d) / 3
    let dim = UInt8(la / 12)
    if d / 3 >= threshold {
        changed += 1
        if lb > la { brighter += 1 }
        let v = UInt8(min(255, 80 + d))
        out[o] = lb > la ? v : 0; out[o + 1] = 0; out[o + 2] = lb > la ? 0 : v
    } else {
        out[o] = dim; out[o + 1] = dim; out[o + 2] = dim
    }
    out[o + 3] = 255
}
print(String(format: "changed %.3f%% of pixels (%d, %d brighter in b), mean change %.3f", 100 * Double(changed) / Double(w * h), changed, brighter, total / Double(w * h)))
let ctx = CGContext(data: &out, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                    space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: args[3]) as CFURL, UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(dest, ctx.makeImage()!, nil)
CGImageDestinationFinalize(dest)
