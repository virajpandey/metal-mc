// The flicker meter's numbers by class (docs/lab-mode.md, "The flicker meter"): lab mode's `burst LABEL` writes
// LABEL-flicker.bin (Sources/MetalMCNative/Flicker.swift), `mask LABEL` the class mask of the same view.
//
//   swiftc -O tools/flickstat.swift -o bench_out/flickstat
//   bench_out/flickstat LABEL-flicker.bin [LABEL-mask.png] [--json out.json] [--hot N]
//
// Per class: its share of the frame, mean d1 (any change from frame to frame) and d2 (change that reverses: flicker), d2's
// 99th percentile, the share of its pixels with d2 over 2 and over 6 levels. Then the N (8) 64 x 64 tiles with the most
// flicker (mean d2), with their place, for cropping the kept frames (ffmpeg -vf crop=64:64:X:Y).
import CoreGraphics
import Foundation
import ImageIO

let classes: [(name: String, rgb: (UInt8, UInt8, UInt8))] = [
    ("sky", (60, 120, 220)), ("water", (0, 200, 200)), ("vegetation", (40, 170, 40)), ("sunlit", (255, 215, 60)),
    ("shade", (120, 70, 180)), ("blocklit", (255, 120, 20)), ("far", (150, 150, 150)), ("unlit", (220, 30, 90))]

var args = Array(CommandLine.arguments.dropFirst())
var jsonOut: String?, hotN = 8
if let i = args.firstIndex(of: "--json") { jsonOut = args[i + 1]; args.removeSubrange(i...i + 1) }
if let i = args.firstIndex(of: "--hot") { hotN = Int(args[i + 1]) ?? 8; args.removeSubrange(i...i + 1) }
guard let binPath = args.first, let bin = FileManager.default.contents(atPath: binPath), bin.count > 16,
      String(data: bin.prefix(4), encoding: .utf8) == "MMCF" else {
    FileHandle.standardError.write("usage: flickstat LABEL-flicker.bin [LABEL-mask.png] [--json out.json] [--hot N]\n".data(using: .utf8)!)
    exit(2)
}
func u32(_ o: Int) -> Int { bin.withUnsafeBytes { Int(UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: o, as: UInt32.self))) } }
let W = u32(4), H = u32(8), frames = u32(12)
guard bin.count >= 16 + W * H * 8 else { print("short file"); exit(1) }
var d1 = [Float](repeating: 0, count: W * H), d2 = [Float](repeating: 0, count: W * H)
bin.withUnsafeBytes { raw in
    let px = raw.baseAddress!.advanced(by: 16).assumingMemoryBound(to: Float16.self)
    for i in 0..<(W * H) { d1[i] = Float(px[4 * i + 2]); d2[i] = Float(px[4 * i + 3]) }
}

// The mask's classes (all "frame" without one).
var cls = [UInt8](repeating: 255, count: W * H)
if args.count > 1 {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: args[1]) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil), img.width == W, img.height == H else {
        print("the mask isn't \(W)x\(H): \(args[1])"); exit(1)
    }
    var rgba = [UInt8](repeating: 0, count: W * H * 4)
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    rgba.withUnsafeMutableBytes { b in
        let ctx = CGContext(data: b.baseAddress, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: cs,
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        ctx.interpolationQuality = .none
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: W, height: H))
    }
    for i in 0..<(W * H) {
        let r = rgba[4 * i], g = rgba[4 * i + 1], b = rgba[4 * i + 2]
        var best = 0, bd = Int.max
        for (k, c) in classes.enumerated() {
            let dd = abs(Int(r) - Int(c.rgb.0)) + abs(Int(g) - Int(c.rgb.1)) + abs(Int(b) - Int(c.rgb.2))
            if dd < bd { bd = dd; best = k }
        }
        cls[i] = UInt8(best)
    }
}

struct Row { var name: String; var share = 0.0, d1 = 0.0, d2 = 0.0, p99 = 0.0, over2 = 0.0, over6 = 0.0 }
func row(_ name: String, _ pick: (Int) -> Bool) -> Row {
    var r = Row(name: name), n = 0
    var hist = [Int](repeating: 0, count: 1024)
    var o2 = 0, o6 = 0
    for i in 0..<(W * H) where pick(i) {
        n += 1; r.d1 += Double(d1[i]); r.d2 += Double(d2[i])
        if d2[i] > 2 { o2 += 1 }
        if d2[i] > 6 { o6 += 1 }
        hist[min(Int(d2[i] * 4), 1023)] += 1
    }
    guard n > 0 else { return r }
    var seen = 0
    for (k, c) in hist.enumerated() { seen += c; if seen * 100 >= n * 99 { r.p99 = Double(k) / 4; break } }
    r.share = 100 * Double(n) / Double(W * H); r.d1 /= Double(n); r.d2 /= Double(n)
    r.over2 = 100 * Double(o2) / Double(n); r.over6 = 100 * Double(o6) / Double(n)
    return r
}
var rows = [row("frame") { _ in true }]
if args.count > 1 { for (k, c) in classes.enumerated() { let r = row(c.name) { cls[$0] == UInt8(k) }; if r.share > 0 { rows.append(r) } } }

print("\(binPath): \(W)x\(H), \(frames) frames")
print("class        share     d1      d2   d2 p99  d2>2 %  d2>6 %")
for r in rows {
    print(String(format: "%-11@ %5.1f%%  %6.3f  %6.3f  %6.2f  %6.2f  %6.2f", r.name as NSString, r.share, r.d1, r.d2, r.p99, r.over2, r.over6))
}

// The hottest tiles.
let T = 64
var tiles: [(x: Int, y: Int, d2: Double, cls: String)] = []
for ty in stride(from: 0, to: H - T + 1, by: T) {
    for tx in stride(from: 0, to: W - T + 1, by: T) {
        var s = 0.0
        var counts = [Int](repeating: 0, count: classes.count + 1)
        for y in ty..<(ty + T) { for x in tx..<(tx + T) { let i = y * W + x; s += Double(d2[i]); counts[min(Int(cls[i]), classes.count)] += 1 } }
        let top = counts.enumerated().max { $0.1 < $1.1 }!.0
        tiles.append((tx, ty, s / Double(T * T), top < classes.count ? classes[top].name : "frame"))
    }
}
tiles.sort { $0.d2 > $1.d2 }
print("hottest \(T)x\(T) tiles (x, y from the top left; mean d2; mostly):")
for t in tiles.prefix(hotN) { print(String(format: "  %4d %4d  %6.2f  %@", t.x, t.y, t.d2, t.cls as NSString)) }

if let jsonOut {
    let obj: [String: Any] = ["file": binPath, "width": W, "height": H, "frames": frames,
                              "classes": rows.map { ["name": $0.name, "share": $0.share, "d1": $0.d1, "d2": $0.d2, "d2_p99": $0.p99,
                                                     "d2_over2": $0.over2, "d2_over6": $0.over6] },
                              "hot": tiles.prefix(hotN).map { ["x": $0.x, "y": $0.y, "d2": $0.d2, "class": $0.cls] }]
    if let data = try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted, .sortedKeys]) {
        FileManager.default.createFile(atPath: jsonOut, contents: data)
    }
}
