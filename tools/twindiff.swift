// Twin diff: how far our frame is from its twin by another renderer (SEUS through the external pipeline: same engine,
// same camera, same geometry), class by class, from the class mask lab mode's `mask` command makes from our G-buffer
// (docs/lab-mode.md, "The class mask"; docs/lighting-design.md, "The grade").
//
// For each class (sky, water, vegetation, sunlit, shade, blocklit, far, unlit) and both frames: the class's share of the
// frame; OKLab mean L, a, b and chroma; L at the 10th, 50th and 90th percentiles; local contrast (mean |L - blur(L)|).
// Then the deltas (ours - twin) and the score, lower is better:
//
//   per class   dE = sqrt(dL^2 + da^2 + db^2 + dC^2)            the class means' OKLab distance, mean chroma as a 4th axis
//               P  = (|dL10| + |dL50| + |dL90|) / 3             the shape of the lightness distribution
//               K  = |dLC|                                      local contrast
//               s  = dE + 0.5 P + 2 K
//   total       100 x sum over classes of weight x share x s    (weights: sky 0.25, unlit 0.25, the rest 1)
//
// The frames are compared at half resolution (2 x 2 boxes averaged in linear light); a box whose mask pixels don't
// mostly (3 of 4) agree on a class is left out, so the edges between classes (where a sub-pixel offset between the twins
// would count) don't. Lightness is OKLab L (0-1): 0.01 is about one step of a 100-step gray ramp.
//
// usage: twindiff <ours.png> <twin.png> <mask.png> [--json out.json] [--board out.png] [--name NAME]
//                 [--weights sky=0.25,unlit=0.25,...] [--quiet]
//   --board writes ours | twin over mask | the lightness difference (blue: ours darker, red: ours brighter; full color at
//   0.25 of L), each panel 1152 wide; PNG, or JPEG for a .jpg path. Boards hold the twin's pixels: keep them out of the
//   repository.
// build: swiftc -O tools/twindiff.swift -o bench_out/twindiff
import CoreGraphics
import Foundation
import ImageIO

let classNames = ["sky", "water", "vegetation", "sunlit", "shade", "blocklit", "far", "unlit"]
/// The mask's colors (LabMask.swift's labMaskClasses, same order).
let classColors: [(Int, Int, Int)] = [(60, 120, 220), (0, 200, 200), (40, 170, 40), (255, 215, 60), (120, 70, 180),
                                       (255, 120, 20), (150, 150, 150), (220, 30, 90)]
var weights: [Double] = [0.25, 1, 1, 1, 1, 1, 1, 0.25]
let wPercentile = 0.5, wContrast = 2.0

// MARK: - Arguments

var positional: [String] = []
var jsonPath: String?, boardPath: String?, name: String?
var quiet = false
var argv = Array(CommandLine.arguments.dropFirst())
while !argv.isEmpty {
    let a = argv.removeFirst()
    switch a {
    case "--json": jsonPath = argv.removeFirst()
    case "--board": boardPath = argv.removeFirst()
    case "--name": name = argv.removeFirst()
    case "--quiet": quiet = true
    case "--weights":
        for kv in argv.removeFirst().split(separator: ",") {
            let p = kv.split(separator: "=")
            if p.count == 2, let i = classNames.firstIndex(of: String(p[0])), let v = Double(p[1]) { weights[i] = v }
        }
    default: positional.append(a)
    }
}
guard positional.count == 3 else {
    FileHandle.standardError.write("usage: twindiff <ours.png> <twin.png> <mask.png> [--json out.json] [--board out.png] [--name NAME] [--weights sky=0.25,...] [--quiet]\n".data(using: .utf8)!)
    exit(2)
}

// MARK: - Images

struct Image { let w: Int; let h: Int; var px: [UInt8] }   // RGBA, 8 bits, rows top-down

/// Loads a PNG's 8-bit values as stored: drawn into a context in the image's own color space, so nothing is converted.
func load(_ path: String) -> Image {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
          let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
        FileHandle.standardError.write("can't read \(path)\n".data(using: .utf8)!)
        exit(1)
    }
    let w = img.width, h = img.height
    let space = img.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
    var px = [UInt8](repeating: 0, count: w * h * 4)
    let ok = px.withUnsafeMutableBytes { buf -> Bool in
        guard let c = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                                bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return false }
        c.setBlendMode(.copy)
        c.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return true
    }
    if !ok { FileHandle.standardError.write("can't decode \(path)\n".data(using: .utf8)!); exit(1) }
    return Image(w: w, h: h, px: px)
}

let ours = load(positional[0]), twin = load(positional[1]), mask = load(positional[2])
guard ours.w == twin.w, ours.h == twin.h, ours.w == mask.w, ours.h == mask.h else {
    FileHandle.standardError.write("sizes differ: ours \(ours.w)x\(ours.h), twin \(twin.w)x\(twin.h), mask \(mask.w)x\(mask.h)\n".data(using: .utf8)!)
    exit(1)
}

// MARK: - Half resolution, OKLab

let toLinear: [Float] = (0..<256).map { i in
    let c = Float(i) / 255
    return c <= 0.04045 ? c / 12.92 : powf((c + 0.055) / 1.055, 2.4)
}

/// sRGB-linear to OKLab (Ottosson).
@inline(__always) func oklab(_ r: Float, _ g: Float, _ b: Float) -> (Float, Float, Float) {
    let l = cbrtf(0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b)
    let m = cbrtf(0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b)
    let s = cbrtf(0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)
}

let W = ours.w / 2, H = ours.h / 2, N = W * H

/// Half-resolution OKLab planes (2 x 2 boxes averaged in linear light).
func halfLab(_ im: Image) -> (L: [Float], a: [Float], b: [Float]) {
    var L = [Float](repeating: 0, count: N), A = L, B = L
    im.px.withUnsafeBufferPointer { p in
        for y in 0..<H {
            for x in 0..<W {
                var r: Float = 0, g: Float = 0, b: Float = 0
                for dy in 0..<2 {
                    var i = ((2 * y + dy) * im.w + 2 * x) * 4
                    for _ in 0..<2 {
                        r += toLinear[Int(p[i])]; g += toLinear[Int(p[i + 1])]; b += toLinear[Int(p[i + 2])]
                        i += 4
                    }
                }
                let o = oklab(r * 0.25, g * 0.25, b * 0.25)
                let k = y * W + x
                L[k] = o.0; A[k] = o.1; B[k] = o.2
            }
        }
    }
    return (L, A, B)
}

/// The mask's class per half-resolution texel: the class 3 or 4 of its pixels have, else -1 (an edge between classes).
func halfClasses(_ im: Image) -> [Int8] {
    var out = [Int8](repeating: -1, count: N)
    func code(_ p: UnsafeBufferPointer<UInt8>, _ i: Int) -> Int {
        let r = Int(p[i]), g = Int(p[i + 1]), b = Int(p[i + 2])
        var best = 0, bestD = Int.max
        for (k, c) in classColors.enumerated() {
            let d = (r - c.0) * (r - c.0) + (g - c.1) * (g - c.1) + (b - c.2) * (b - c.2)
            if d < bestD { bestD = d; best = k }
        }
        return best
    }
    im.px.withUnsafeBufferPointer { p in
        for y in 0..<H {
            for x in 0..<W {
                var count = [Int](repeating: 0, count: classColors.count)
                for dy in 0..<2 { for dx in 0..<2 { count[code(p, ((2 * y + dy) * im.w + 2 * x + dx) * 4)] += 1 } }
                if let m = count.indices.max(by: { count[$0] < count[$1] }), count[m] >= 3 { out[y * W + x] = Int8(m) }
            }
        }
    }
    return out
}

/// A box blur of radius r along both axes, twice (close to a Gaussian of sigma about 1.2 r).
func blur(_ src: [Float], r: Int) -> [Float] {
    var a = src, b = src
    for _ in 0..<2 {
        // Rows.
        for y in 0..<H {
            var s: Float = 0
            let base = y * W
            for x in -r...r { s += a[base + min(max(x, 0), W - 1)] }
            for x in 0..<W {
                b[base + x] = s / Float(2 * r + 1)
                s += a[base + min(x + r + 1, W - 1)] - a[base + max(x - r, 0)]
            }
        }
        // Columns.
        for x in 0..<W {
            var s: Float = 0
            for y in -r...r { s += b[min(max(y, 0), H - 1) * W + x] }
            for y in 0..<H {
                a[y * W + x] = s / Float(2 * r + 1)
                s += b[min(y + r + 1, H - 1) * W + x] - b[max(y - r, 0) * W + x]
            }
        }
    }
    return a
}

let classes = halfClasses(mask)
let lo = halfLab(ours), lt = halfLab(twin)
let blurO = blur(lo.L, r: 2), blurT = blur(lt.L, r: 2)

// MARK: - Statistics

struct Stats {
    var n = 0
    var L = 0.0, a = 0.0, b = 0.0, C = 0.0, lc = 0.0
    var p10 = 0.0, p50 = 0.0, p90 = 0.0
    var dict: [String: Double] {
        ["L": L, "a": a, "b": b, "C": C, "p10": p10, "p50": p50, "p90": p90, "lc": lc]
    }
}

let bins = 1000
func stats(_ lab: (L: [Float], a: [Float], b: [Float]), _ bl: [Float], _ cls: Int) -> Stats {
    var s = Stats()
    var hist = [Int](repeating: 0, count: bins)
    var sL = 0.0, sa = 0.0, sb = 0.0, sC = 0.0, slc = 0.0
    for k in 0..<N where Int(classes[k]) == cls {
        let L = lab.L[k], a = lab.a[k], b = lab.b[k]
        sL += Double(L); sa += Double(a); sb += Double(b); sC += Double((a * a + b * b).squareRoot())
        slc += Double(abs(L - bl[k]))
        hist[min(max(Int(L * Float(bins)), 0), bins - 1)] += 1
        s.n += 1
    }
    guard s.n > 0 else { return s }
    let n = Double(s.n)
    s.L = sL / n; s.a = sa / n; s.b = sb / n; s.C = sC / n; s.lc = slc / n
    func pct(_ q: Double) -> Double {
        let target = q * n
        var acc = 0.0
        for i in 0..<bins {
            let next = acc + Double(hist[i])
            if next >= target { return (Double(i) + (Double(hist[i]) > 0 ? (target - acc) / Double(hist[i]) : 0.5)) / Double(bins) }
            acc = next
        }
        return 1
    }
    s.p10 = pct(0.1); s.p50 = pct(0.5); s.p90 = pct(0.9)
    return s
}

/// The census's numbers (docs/lab-mode.md, "Twins"), as ffmpeg's signalstats gives them on a PNG (BT.601, limited range):
/// the mean luma (YAVG), its 10th percentile (YLOW) and the mean saturation (SATAVG) of a band of rows.
struct Region { var y = 0.0, ylow = 0.0, sat = 0.0 }
func region(_ im: Image, _ y0: Int, _ y1: Int) -> Region {
    var hist = [Int](repeating: 0, count: 256)
    var sy = 0.0, ss = 0.0
    im.px.withUnsafeBufferPointer { p in
        for y in y0..<y1 {
            var i = y * im.w * 4
            for _ in 0..<im.w {
                let r = Double(p[i]), g = Double(p[i + 1]), b = Double(p[i + 2])
                let Y = 16 + (219.0 / 255) * (0.299 * r + 0.587 * g + 0.114 * b)
                let u = (224.0 / 255) * (-0.168736 * r - 0.331264 * g + 0.5 * b)
                let v = (224.0 / 255) * (0.5 * r - 0.418688 * g - 0.081312 * b)
                sy += Y
                ss += (u * u + v * v).squareRoot()
                hist[min(Int(Y.rounded()), 255)] += 1
                i += 4
            }
        }
    }
    let n = (y1 - y0) * im.w
    var acc = 0, low = 0
    for (k, c) in hist.enumerated() {
        acc += c
        if acc * 10 >= n { low = k; break }
    }
    return Region(y: sy / Double(n), ylow: Double(low), sat: ss / Double(n))
}
/// Terrain: the bottom 55% of the frame; sky: the top 20%.
func regions(_ im: Image) -> (terrain: Region, sky: Region) {
    (region(im, Int(Double(im.h) * 0.45), im.h), region(im, 0, Int(Double(im.h) * 0.2)))
}

var total = 0.0
var classified = 0
for c in classes where c >= 0 { classified += 1 }
var rows: [[String: Any]] = []
var table = "class       share  | ours: L     a      b     C   | p10  p50  p90 | lc    | twin: L     a      b     C   | p10  p50  p90 | lc    |      dE      P      K  points\n"
for (ci, cname) in classNames.enumerated() {
    let so = stats(lo, blurO, ci), st = stats(lt, blurT, ci)
    let share = classified > 0 ? Double(so.n) / Double(classified) : 0
    var row: [String: Any] = ["class": cname, "share": share, "weight": weights[ci], "ours": so.dict, "twin": st.dict]
    if so.n > 0 {
        let dL = so.L - st.L, da = so.a - st.a, db = so.b - st.b, dC = so.C - st.C
        let dE = (dL * dL + da * da + db * db + dC * dC).squareRoot()
        let P = (abs(so.p10 - st.p10) + abs(so.p50 - st.p50) + abs(so.p90 - st.p90)) / 3
        let K = abs(so.lc - st.lc)
        let s = dE + wPercentile * P + wContrast * K
        let contribution = 100 * weights[ci] * share * s
        total += contribution
        row["delta"] = ["L": dL, "a": da, "b": db, "C": dC, "p10": so.p10 - st.p10, "p50": so.p50 - st.p50, "p90": so.p90 - st.p90, "lc": so.lc - st.lc]
        row["dE"] = dE; row["P"] = P; row["K"] = K; row["score"] = s; row["contribution"] = contribution
        table += String(format: "%-11@ %5.1f%% |       %.3f %+.3f %+.3f %.3f | %.2f %.2f %.2f | %.3f |       %.3f %+.3f %+.3f %.3f | %.2f %.2f %.2f | %.3f | %7.4f %6.3f %6.3f %6.2f\n",
                        cname as NSString, 100 * share, so.L, so.a, so.b, so.C, so.p10, so.p50, so.p90, so.lc,
                        st.L, st.a, st.b, st.C, st.p10, st.p50, st.p90, st.lc, dE, P, K, contribution)
    }
    rows.append(row)
}
let ro = regions(ours), rt = regions(twin)
table += String(format: "total %.2f  (terrain Y/p10/sat ours %.1f/%.0f/%.1f twin %.1f/%.0f/%.1f, sky Y ours %.1f twin %.1f; %.1f%% of texels on class edges)\n",
                total, ro.terrain.y, ro.terrain.ylow, ro.terrain.sat, rt.terrain.y, rt.terrain.ylow, rt.terrain.sat, ro.sky.y, rt.sky.y,
                100 * Double(N - classified) / Double(N))
if !quiet {
    if let name { print("== \(name)") }
    print(table, terminator: "")
}

if let jsonPath {
    let doc: [String: Any] = ["name": name ?? "", "ours": positional[0], "twin": positional[1], "mask": positional[2],
                              "size": [ours.w, ours.h], "total": total, "classes": rows,
                              "regions": ["terrain_y_ours": ro.terrain.y, "terrain_y_twin": rt.terrain.y, "sky_y_ours": ro.sky.y, "sky_y_twin": rt.sky.y,
                                          "terrain_p10_ours": ro.terrain.ylow, "terrain_p10_twin": rt.terrain.ylow,
                                          "terrain_sat_ours": ro.terrain.sat, "terrain_sat_twin": rt.terrain.sat,
                                          "sky_sat_ours": ro.sky.sat, "sky_sat_twin": rt.sky.sat],
                              "formula": "100 * sum(weight * share * (dE + \(wPercentile) P + \(wContrast) K))"]
    if let data = try? JSONSerialization.data(withJSONObject: doc, options: [.prettyPrinted, .sortedKeys]) {
        try? data.write(to: URL(fileURLWithPath: jsonPath))
    }
}

// MARK: - Board

if let boardPath {
    let pw = 1152, ph = pw * ours.h / ours.w, gap = 8
    let bw = 2 * pw + gap, bh = 2 * ph + gap
    guard let ctx = CGContext(data: nil, width: bw, height: bh, bitsPerComponent: 8, bytesPerRow: 0,
                              space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { exit(1) }
    ctx.interpolationQuality = .high
    ctx.setFillColor(CGColor(red: 1, green: 1, blue: 1, alpha: 1))
    ctx.fill(CGRect(x: 0, y: 0, width: bw, height: bh))
    func cg(_ im: Image) -> CGImage? {
        guard let p = CGDataProvider(data: Data(im.px) as CFData) else { return nil }
        return CGImage(width: im.w, height: im.h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: im.w * 4,
                       space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue),
                       provider: p, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }
    // The lightness difference, blurred (ours - twin): red where ours is brighter, blue where darker, gray on class edges.
    var diff = Image(w: W, h: H, px: [UInt8](repeating: 0, count: N * 4))
    let dO = blur(lo.L, r: 3), dT = blur(lt.L, r: 3)
    for k in 0..<N {
        let d = Double(dO[k] - dT[k]) / 0.25
        let m = min(abs(d), 1)
        let base = 40.0
        var r = base, g = base, b = base
        if d > 0 { r = base + (255 - base) * m; g = base * (1 - m); b = base * (1 - m) } else { b = base + (255 - base) * m; r = base * (1 - m); g = base * (1 - m) * 0.5 + 60 * m }
        if classes[k] < 0 { r = 90; g = 90; b = 90 }
        diff.px[4 * k] = UInt8(r); diff.px[4 * k + 1] = UInt8(g); diff.px[4 * k + 2] = UInt8(b); diff.px[4 * k + 3] = 255
    }
    let panels: [(Image, Int, Int)] = [(ours, 0, ph + gap), (twin, pw + gap, ph + gap), (mask, 0, 0), (diff, pw + gap, 0)]
    for (im, x, y) in panels {
        if let i = cg(im) { ctx.draw(i, in: CGRect(x: x, y: y, width: pw, height: ph)) }
    }
    // PNG, or JPEG (quality 0.9) for a .jpg path.
    let jpeg = boardPath.lowercased().hasSuffix(".jpg") || boardPath.lowercased().hasSuffix(".jpeg")
    if let out = ctx.makeImage(),
       let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: boardPath) as CFURL, (jpeg ? "public.jpeg" : "public.png") as CFString, 1, nil) {
        CGImageDestinationAddImage(dest, out, jpeg ? [kCGImageDestinationLossyCompressionQuality: 0.9] as CFDictionary : nil)
        CGImageDestinationFinalize(dest)
    }
}
