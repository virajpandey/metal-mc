import Foundation
import Metal
import simd

// Near chunks in our own compact format (METALMC_EXP=nearchunks, off by default; docs/near-chunks-design.md). The
// first step of the rewrite's swing 1: vanilla's section compiler keeps meshing (it knows every block model, fluid and
// tint), but its solid and cutout layers are repacked on the mesh worker into 32-byte quad records in one GPU arena, and
// the terrain pass draws them with a shader equivalent to vanilla's terrain shader instead of vanilla's draws.
// Translucent terrain stays vanilla's: it needs per-quad sorting.
//
// Vanilla's BLOCK vertex is 28 bytes (float position, RGBA8 color, float UV, 16-bit block and sky light), so a face is
// 112 bytes. Almost every face in normal terrain is an axis-aligned rectangle on the 1/16-block grid with a rectangular
// texture mapping, and its color is a per-block tint times a per-corner gray level (ambient occlusion times directional
// shade). Those faces fit one 32-byte record exactly. Everything else (plants with random offsets, cross models,
// rotated elements, fluid tops, tints the solver can't split) is a "generic" quad: four 16-byte vertices, the first in
// its record and the other three after the section's records (80 bytes in all). A layer with a quad that neither form
// can hold isn't repacked, and vanilla draws it as before.
//
// Record (8 words), axis-aligned form (bit 0 of word 0 clear):
//   w0: bit 0 kind (0) | 1-2 plane axis (0 x, 1 y, 2 z) | 3-4 corner of vertex 0 | 5 corners run backwards |
//       6 UV layout | 7-15 origin x | 16-24 origin y (1/16 blocks, +16)
//   w1: 0-8 origin z | 9-17 extent along the first in-plane axis | 18-26 extent along the second (1/16 blocks)
//   w2: UV of vertex 0 (u | v << 16, low 16 bits of u * 65536) | w3: UV of vertex 2
//   w4: tint r | g << 8 | b << 16 | gray of vertex 0 << 24
//   w5: gray of vertices 1, 2, 3 | bit 16 of vertex 0's u, v and vertex 2's u, v << 24..27 (so a UV of 1.0 is exact)
//   w6: light of vertices 0 and 1 (block | sky << 8, 16 bits each) | w7: vertices 2 and 3
// Generic form: w0 = 1 | (offset of vertices 1-3, in words, from this record) << 1, w1-w4 vertex 0; vertices 1-3 follow
// all the records (80 bytes a quad in all). Each generic vertex is 4 words:
//   v0: x (18 bits) | y low 14 bits << 18 | v1: y high 4 bits | z << 4 | r << 22 | bit 16 of u << 30 | of v << 31
//   v2: g | b << 8 | block << 16 | sky << 24 | v3: u | v << 16 (low 16 bits). Positions are (p + 8) * 8192 (1/8192 block
//   steps over -8..24).
//
// What is exact: colors (the 8-bit values), light, the vertex order (so the triangle split and the interpolation of
// light and ambient occlusion are vanilla's), positions on the 1/16 grid (the axis-aligned form), and UVs on the
// 1/65536 grid, which holds every whole texel of a power-of-two atlas up to 65536 wide and every 1/16 texel up to 4096
// wide (vanilla's model UVs are whole or half texels). Rounded: generic positions, to 1/8192 of a block (at most 6.1e-5
// block off), and other UVs (flowing fluids' rotated ones) to within 7.6e-6. docs/near-chunks-design.md has the
// measured numbers.
//
// The arena is a list of 64 MiB slabs of 32-byte slots with a first-fit free list per slab. Mesh workers encode into a
// scratch array, allocate under the lock, copy without it, then publish the entry. A freed range is reused only once
// the GPU has finished every submit that could still draw it (the latest submit with a near-chunk draw when it was
// freed), so a worker never overwrites memory an in-flight frame reads.

let nearChunksEnabled = experiments.contains("nearchunks")

// MARK: - Codec

struct NearCodecStats {
    var quads = 0
    var aligned = 0       // axis-aligned records
    var generic = 0       // generic records
    var tinted = 0        // aligned records with a tint other than white
    var tintMisses = 0    // aligned shapes whose colors didn't split into tint x gray (stored generic)
}

enum NearCodec {
    static let recordWords = 8
    static let genericScale: Float = 8192
    static let genericBias: Float = 8
    static let genericMax = (1 << 18) - 1
    static let uvScale: Float = 65536

    /// Gray levels vanilla's block lighter produces for whole faces, brightest first: the mean of four 0.2/1.0 shade
    /// values (summed in any order, in float, as BlockModelLighter does) times 255, floored, then scaled by a cardinal
    /// shade (ARGB.scaleRGB), plus the flat path's gray(shade). The tint solver tries these first.
    static let likelyGray: [Int32] = {
        var levels = Set<Int32>()
        let shades: [Float] = [1.0, 0.8, 0.6, 0.5, 0.9]
        for bits in 0..<16 {
            let s = (0..<4).map { (bits >> $0) & 1 == 1 ? Float(0.2) : Float(1.0) }
            let mean = (s[0] + s[1] + s[2] + s[3]) * Float(0.25)
            let g0 = Int32((mean * Float(255)).rounded(.down))
            for shade in shades {
                levels.insert(min(255, max(0, Int32(Float(g0) * shade))))
            }
        }
        for shade in shades { levels.insert(Int32((shade * Float(255)).rounded(.down))) }
        return levels.sorted(by: >)
    }()

    /// Splits four colors into a tint T and gray levels g with vanilla's ARGB.multiply(gray(g), T) = floor(T * g / 255)
    /// per channel reproducing each exactly. Returns (T packed r | g << 8 | b << 16, gray levels), or nil.
    static func solveTint(_ r: SIMD4<Int32>, _ g: SIMD4<Int32>, _ b: SIMD4<Int32>) -> (UInt32, SIMD4<Int32>)? {
        if r == g && g == b { return (0xFFFFFF, r) }   // gray: a white tint
        let lum = r &+ g &+ b
        var m = 0
        for k in 1..<4 where lum[k] > lum[m] { m = k }
        let cm = SIMD3(r[m], g[m], b[m])
        let cmax = max(cm.x, max(cm.y, cm.z))

        // Tint candidates come from the brightest vertex at gray level gm; every other vertex then needs one gray level
        // that fits all three channels.
        func attempt(_ gm: Int32, combos: Int32) -> (UInt32, SIMD4<Int32>)? {
            guard gm > 0, gm >= cmax else { return nil }
            var lo = SIMD3<Int32>(repeating: 0), hi = SIMD3<Int32>(repeating: 0)
            for c in 0..<3 {
                lo[c] = (255 * cm[c] + gm - 1) / gm
                hi[c] = min(255, (255 * cm[c] + 254) / gm)
                if lo[c] > hi[c] { return nil }
            }
            // The ranges are one or two values wide for bright vertices; cap the search for dim ones.
            let n = SIMD3(min(hi.x - lo.x + 1, combos), min(hi.y - lo.y + 1, combos), min(hi.z - lo.z + 1, combos))
            for i0 in 0..<n.x {
                for i1 in 0..<n.y {
                    for i2 in 0..<n.z {
                        let t = SIMD3(lo.x + i0, lo.y + i1, lo.z + i2)
                        var gs = SIMD4<Int32>(repeating: 0)
                        gs[m] = gm
                        var ok = true
                        for k in 0..<4 where k != m {
                            var glo: Int32 = 0, ghi: Int32 = 255
                            let v = SIMD3(r[k], g[k], b[k])
                            for c in 0..<3 {
                                let tc = t[c]
                                if tc == 0 {
                                    if v[c] != 0 { ok = false; break }
                                    continue
                                }
                                glo = max(glo, (255 * v[c] + tc - 1) / tc)
                                ghi = min(ghi, (255 * v[c] + 254) / tc)
                            }
                            if !ok || glo > ghi { ok = false; break }
                            gs[k] = glo
                        }
                        if ok { return (UInt32(t.x) | UInt32(t.y) << 8 | UInt32(t.z) << 16, gs) }
                    }
                }
            }
            return nil
        }
        for gm in likelyGray where gm >= cmax {
            if let s = attempt(gm, combos: 4) { return s }
        }
        // Partial faces weight their corners' ambient occlusion by the face's extent, so any level can occur.
        var gm: Int32 = 255
        while gm >= max(1, cmax) {
            if let s = attempt(gm, combos: 2) { return s }
            gm -= 1
        }
        return nil
    }

    // The range check happens in float, before converting (NaN fails it too).
    @inline(__always) static func quantizeGeneric(_ p: Float) -> UInt32? {
        let q = ((p + genericBias) * genericScale).rounded()
        guard q >= 0, q <= Float(genericMax) else { return nil }
        return UInt32(q)
    }

    /// Encodes `vertexCount` vertices of vanilla's BLOCK format (28 bytes each, 4 per quad) into records followed by
    /// the generic vertices. Returns false if some quad can't be represented (the layer then stays vanilla's).
    static func encode(_ src: UnsafeRawPointer, vertexCount: Int, into out: inout [UInt32], stats: inout NearCodecStats) -> Bool {
        guard vertexCount > 0, vertexCount % 4 == 0 else { return false }
        let n = vertexCount / 4
        // Room for the worst case (every quad generic); trimmed at the end.
        out.removeAll(keepingCapacity: true)
        out.append(contentsOf: repeatElement(0, count: n * (recordWords + 12) + recordWords))
        var local = NearCodecStats()
        local.quads = n
        var extWords = 0
        // One quad's fields, lane k of each group of four: positions x y z, then r g b, block, sky, u, v (UVs * 65536).
        // Scalar loads and checks: Swift's SIMD rounding and mask reductions cost more than the whole encode.
        let pos = UnsafeMutablePointer<Float>.allocate(capacity: 12)
        let val = UnsafeMutablePointer<UInt32>.allocate(capacity: 28)
        defer { pos.deallocate(); val.deallocate() }
        let ok = out.withUnsafeMutableBufferPointer { buf -> Bool in
            let records = buf.baseAddress!, ext = records + n * recordWords
            for q in 0..<n {
                for k in 0..<4 {
                    let p = src + (q * 4 + k) * 28
                    pos[k] = p.loadUnaligned(fromByteOffset: 0, as: Float.self)
                    pos[4 + k] = p.loadUnaligned(fromByteOffset: 4, as: Float.self)
                    pos[8 + k] = p.loadUnaligned(fromByteOffset: 8, as: Float.self)
                    val[k] = UInt32(p.load(fromByteOffset: 12, as: UInt8.self))
                    val[4 + k] = UInt32(p.load(fromByteOffset: 13, as: UInt8.self))
                    val[8 + k] = UInt32(p.load(fromByteOffset: 14, as: UInt8.self))
                    guard p.load(fromByteOffset: 15, as: UInt8.self) == 255 else { return false }   // terrain alpha is opaque
                    let block = p.loadUnaligned(fromByteOffset: 24, as: Int16.self), sky = p.loadUnaligned(fromByteOffset: 26, as: Int16.self)
                    guard block >= 0, block <= 255, sky >= 0, sky <= 255 else { return false }
                    val[12 + k] = UInt32(block)
                    val[16 + k] = UInt32(sky)
                    // UVs keep a 17th bit so 1.0 is exact. Checked in float before converting (NaN fails too); adding a half
                    // and truncating rounds to nearest for these non-negative values.
                    let u = p.loadUnaligned(fromByteOffset: 16, as: Float.self) * uvScale + 0.5
                    let v = p.loadUnaligned(fromByteOffset: 20, as: Float.self) * uvScale + 0.5
                    guard u >= 0.5, u < 65537, v >= 0.5, v < 65537 else { return false }
                    val[20 + k] = UInt32(u)
                    val[24 + k] = UInt32(v)
                }
                let rec = records + q * recordWords
                if aligned(pos, val, &local, into: rec) {
                    local.aligned += 1
                    continue
                }
                // Generic: vertex 0 in the record, vertices 1-3 after all the records (their offset from this record, in
                // words, fits in 31 bits).
                rec[0] = 1 | UInt32((n - q) * recordWords + extWords) << 1
                for k in 0..<4 {
                    guard let px = quantizeGeneric(pos[k]), let py = quantizeGeneric(pos[4 + k]), let pz = quantizeGeneric(pos[8 + k]) else { return false }
                    let w = k == 0 ? rec + 1 : ext + extWords + (k - 1) * 4
                    let qu = val[20 + k], qv = val[24 + k]
                    w[0] = px | (py & 0x3FFF) << 18
                    w[1] = py >> 14 | pz << 4 | val[k] << 22 | (qu >> 16) << 30 | (qv >> 16) << 31
                    w[2] = val[4 + k] | val[8 + k] << 8 | val[12 + k] << 16 | val[16 + k] << 24
                    w[3] = (qu & 0xFFFF) | (qv & 0xFFFF) << 16
                }
                extWords += 12
                local.generic += 1
            }
            return true
        }
        guard ok else { return false }
        // Whole 32-byte slots: the arena allocates in records.
        let words = (n * recordWords + extWords + recordWords - 1) / recordWords * recordWords
        out.removeLast(out.count - words)
        stats.quads += local.quads
        stats.aligned += local.aligned
        stats.generic += local.generic
        stats.tinted += local.tinted
        stats.tintMisses += local.tintMisses
        return true
    }

    /// Writes the axis-aligned record for a quad (fields as `encode` gathers them) and returns true, or returns false if
    /// it isn't a rectangle on the 1/16 grid with a rectangular UV mapping and a tint x gray color.
    private static func aligned(_ pos: UnsafePointer<Float>, _ val: UnsafePointer<UInt32>, _ stats: inout NearCodecStats,
                                into rec: UnsafeMutablePointer<UInt32>) -> Bool {
        // Positions in 1/16 blocks, whole and within the record's 9 bits (-1 .. 30.9375 blocks; NaN fails too).
        var grid = (Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0), Int32(0))
        let ok = withUnsafeMutableBytes(of: &grid) { raw -> Bool in
            let g = raw.baseAddress!.assumingMemoryBound(to: Int32.self)
            for i in 0..<12 {
                let v = pos[i] * 16
                guard v >= -16, v <= 495 else { return false }
                let iv = Int32(v)
                guard Float(iv) == v else { return false }
                g[i] = iv
            }
            return true
        }
        guard ok else { return false }
        let x = SIMD4(grid.0, grid.1, grid.2, grid.3), y = SIMD4(grid.4, grid.5, grid.6, grid.7), z = SIMD4(grid.8, grid.9, grid.10, grid.11)
        func flat(_ a: SIMD4<Int32>) -> Bool { a[0] == a[1] && a[0] == a[2] && a[0] == a[3] }
        let axis: UInt32
        let a1: SIMD4<Int32>, a2: SIMD4<Int32>
        switch (flat(x), flat(y), flat(z)) {
        case (true, false, false): axis = 0; a1 = y; a2 = z
        case (false, true, false): axis = 1; a1 = x; a2 = z
        case (false, false, true): axis = 2; a1 = x; a2 = y
        default: return false
        }
        let lo1 = min(min(a1[0], a1[1]), min(a1[2], a1[3])), hi1 = max(max(a1[0], a1[1]), max(a1[2], a1[3]))
        let lo2 = min(min(a2[0], a2[1]), min(a2[2], a2[3])), hi2 = max(max(a2[0], a2[1]), max(a2[2], a2[3]))
        guard lo1 < hi1, lo2 < hi2 else { return false }
        // Corners in cycle order (0,0) (1,0) (1,1) (0,1); the four vertices must walk it one way or the other.
        var corner: (Int32, Int32, Int32, Int32) = (0, 0, 0, 0)
        for k in 0..<4 {
            let s: Int32 = a1[k] == hi1 ? 1 : (a1[k] == lo1 ? 0 : -1)
            let t: Int32 = a2[k] == hi2 ? 1 : (a2[k] == lo2 ? 0 : -1)
            guard s >= 0, t >= 0 else { return false }
            let c = t == 0 ? s : 3 - s
            switch k {
            case 0: corner.0 = c
            case 1: corner.1 = c
            case 2: corner.2 = c
            default: corner.3 = c
            }
        }
        let start = corner.0
        let reverse: UInt32
        if corner.1 == (start + 1) & 3, corner.2 == (start + 2) & 3, corner.3 == (start + 3) & 3 {
            reverse = 0
        } else if corner.1 == (start + 3) & 3, corner.2 == (start + 2) & 3, corner.3 == (start + 1) & 3 {
            reverse = 1
        } else {
            return false
        }
        // UVs: vertices 0 and 2 are opposite corners; 1 and 3 take one coordinate from each.
        let qu = val + 20, qv = val + 24
        let layout: UInt32
        if qu[1] == qu[2] && qv[1] == qv[0] && qu[3] == qu[0] && qv[3] == qv[2] {
            layout = 0
        } else if qu[1] == qu[0] && qv[1] == qv[2] && qu[3] == qu[2] && qv[3] == qv[0] {
            layout = 1
        } else {
            return false
        }
        let tint: UInt32, gray: SIMD4<Int32>
        let r = SIMD4<Int32>(Int32(val[0]), Int32(val[1]), Int32(val[2]), Int32(val[3]))
        let g = SIMD4<Int32>(Int32(val[4]), Int32(val[5]), Int32(val[6]), Int32(val[7]))
        let b = SIMD4<Int32>(Int32(val[8]), Int32(val[9]), Int32(val[10]), Int32(val[11]))
        if let solved = solveTint(r, g, b) {
            (tint, gray) = solved
        } else {
            stats.tintMisses += 1
            return false
        }
        if tint != 0xFFFFFF { stats.tinted += 1 }
        let ox = axis == 0 ? x[0] : lo1, oy = axis == 1 ? y[0] : (axis == 0 ? lo1 : lo2), oz = axis == 2 ? z[0] : lo2
        let bl = val + 12, sk = val + 16
        rec[0] = axis << 1 | UInt32(start) << 3 | reverse << 5 | layout << 6 | UInt32(ox + 16) << 7 | UInt32(oy + 16) << 16
        rec[1] = UInt32(oz + 16) | UInt32(hi1 - lo1) << 9 | UInt32(hi2 - lo2) << 18
        rec[2] = (qu[0] & 0xFFFF) | (qv[0] & 0xFFFF) << 16
        rec[3] = (qu[2] & 0xFFFF) | (qv[2] & 0xFFFF) << 16
        rec[4] = tint | UInt32(gray[0]) << 24
        rec[5] = UInt32(gray[1]) | UInt32(gray[2]) << 8 | UInt32(gray[3]) << 16
            | (qu[0] >> 16) << 24 | (qv[0] >> 16) << 25 | (qu[2] >> 16) << 26 | (qv[2] >> 16) << 27
        rec[6] = bl[0] | sk[0] << 8 | (bl[1] | sk[1] << 8) << 16
        rec[7] = bl[2] | sk[2] << 8 | (bl[3] | sk[3] << 8) << 16
        return true
    }

    /// Decodes vertex `k` of record `slot` on the CPU, the same way near_vs does: position, UV, color (0-255), light.
    static func decode(_ w: UnsafePointer<UInt32>, slot: Int, k: Int) -> (SIMD3<Float>, SIMD2<Float>, SIMD3<UInt32>, SIMD2<UInt32>) {
        let s = w + slot * recordWords
        let w0 = s[0]
        if w0 & 1 != 0 {
            let p = k == 0 ? s + 1 : s + Int(w0 >> 1) + (k - 1) * 4
            let px = p[0] & 0x3FFFF, py = (p[0] >> 18) | (p[1] & 0xF) << 14, pz = (p[1] >> 4) & 0x3FFFF
            let pos = SIMD3<Float>(Float(px), Float(py), Float(pz)) * (1.0 / genericScale) - genericBias
            let rgb = SIMD3<UInt32>((p[1] >> 22) & 0xFF, p[2] & 0xFF, (p[2] >> 8) & 0xFF)
            let light = SIMD2<UInt32>((p[2] >> 16) & 0xFF, p[2] >> 24)
            let uq = (p[3] & 0xFFFF) | ((p[1] >> 30) & 1) << 16, vq = (p[3] >> 16) | ((p[1] >> 31) & 1) << 16
            return (pos, SIMD2<Float>(Float(uq), Float(vq)) * (1.0 / uvScale), rgb, light)
        }
        let w1 = s[1], w5 = s[5]
        let axis = (w0 >> 1) & 3, start = (w0 >> 3) & 3, reverse = (w0 >> 5) & 1, layout = (w0 >> 6) & 1
        let idx = reverse != 0 ? (start &+ 4 &- UInt32(k)) & 3 : (start &+ UInt32(k)) & 3
        let su: Int32 = idx == 1 || idx == 2 ? 1 : 0, sv: Int32 = idx >= 2 ? 1 : 0
        var p = SIMD3<Int32>(Int32((w0 >> 7) & 0x1FF), Int32((w0 >> 16) & 0x1FF), Int32(w1 & 0x1FF)) &- 16
        let e1 = Int32((w1 >> 9) & 0x1FF), e2 = Int32((w1 >> 18) & 0x1FF)
        switch axis {
        case 0: p.y += su * e1; p.z += sv * e2
        case 1: p.x += su * e1; p.z += sv * e2
        default: p.x += su * e1; p.y += sv * e2
        }
        let a = SIMD2<UInt32>(s[2] & 0xFFFF | ((w5 >> 24) & 1) << 16, s[2] >> 16 | ((w5 >> 25) & 1) << 16)
        let c = SIMD2<UInt32>(s[3] & 0xFFFF | ((w5 >> 26) & 1) << 16, s[3] >> 16 | ((w5 >> 27) & 1) << 16)
        let uvq = k == 0 ? a : (k == 2 ? c : ((k == 1) == (layout != 0) ? SIMD2(a.x, c.y) : SIMD2(c.x, a.y)))
        let tint = SIMD3<UInt32>(s[4] & 0xFF, (s[4] >> 8) & 0xFF, (s[4] >> 16) & 0xFF)
        let gray = k == 0 ? s[4] >> 24 : (w5 >> UInt32(8 * (k - 1))) & 0xFF
        let lw = s[6 + k / 2] >> UInt32(16 * (k & 1))
        return (SIMD3<Float>(p) * (1.0 / 16.0), SIMD2<Float>(Float(uvq.x), Float(uvq.y)) * (1.0 / uvScale),
                tint &* gray / 255, SIMD2<UInt32>(lw & 0xFF, (lw >> 8) & 0xFF))
    }
}

// MARK: - Arena

final class NearArena: @unchecked Sendable {
    static let shared = NearArena()
    static let slabSlots = 1 << 21          // 64 MiB of 32-byte records
    static let slotBytes = 32

    struct Range {
        var start: Int
        var count: Int
    }

    /// One MTLBuffer of slots with a first-fit free list (sorted by start, neighbors merged).
    final class Slab {
        let buffer: MTLBuffer
        var free: [Range]
        var used = 0

        init?(slots: Int) {
            guard let b = ctx.device.makeBuffer(length: slots * NearArena.slotBytes, options: [.storageModeShared]) else { return nil }
            b.label = "MetalMC near chunks"
            buffer = b
            free = [Range(start: 0, count: slots)]
        }

        func allocate(_ n: Int) -> Int? {
            guard let i = free.firstIndex(where: { $0.count >= n }) else { return nil }
            let start = free[i].start
            if free[i].count == n { free.remove(at: i) } else { free[i].start += n; free[i].count -= n }
            used += n
            return start
        }

        func release(_ start: Int, _ n: Int) {
            used -= n
            // First free range after `start`.
            var lo = 0, hi = free.count
            while lo < hi {
                let mid = (lo + hi) / 2
                if free[mid].start < start { lo = mid + 1 } else { hi = mid }
            }
            let mergesBefore = lo > 0 && free[lo - 1].start + free[lo - 1].count == start
            let mergesAfter = lo < free.count && start + n == free[lo].start
            switch (mergesBefore, mergesAfter) {
            case (true, true):
                free[lo - 1].count += n + free[lo].count
                free.remove(at: lo)
            case (true, false):
                free[lo - 1].count += n
            case (false, true):
                free[lo].start = start
                free[lo].count += n
            case (false, false):
                free.insert(Range(start: start, count: n), at: lo)
            }
        }
    }

    struct Entry {
        var slab: Int32 = 0
        var start: Int32 = 0
        var slots: Int32 = 0
        var quads: Int32 = 0
        var gen: UInt8 = 0
        var live = false
    }

    let lock = NSLock()
    var slabs: [Slab] = []
    var entries: [Entry] = []
    var freeEntries: [Int] = []
    var pending: [(slab: Int, range: Range, tag: Int64)] = []
    /// The newest submit with a near-chunk draw (render thread, under the lock).
    var lastDrawSubmit: Int64 = 0

    // Counters: repacked layers, layers left to vanilla, and what the live entries hold.
    var stats = NearCodecStats()
    var layersPacked = 0, layersRejected = 0
    var liveQuads = 0, liveSlots = 0

    /// Moves freed ranges whose last possible draw has completed on the GPU back to their slabs. Under the lock.
    func reclaim() {
        guard !pending.isEmpty else { return }
        ctx.cond.lock()
        let done = ctx.completed
        ctx.cond.unlock()
        var keep: [(slab: Int, range: Range, tag: Int64)] = []
        for p in pending {
            if p.tag <= done { slabs[p.slab].release(p.range.start, p.range.count) } else { keep.append(p) }
        }
        pending = keep
    }

    /// Allocates `n` slots. Under the lock.
    func allocate(_ n: Int) -> (Int, Int)? {
        reclaim()
        for (i, s) in slabs.enumerated() {
            if let start = s.allocate(n) { return (i, start) }
        }
        guard n <= Self.slabSlots, let s = Slab(slots: Self.slabSlots) else { return nil }
        slabs.append(s)
        log("near chunks: arena slab \(slabs.count) (\((slabs.count * Self.slabSlots * Self.slotBytes) >> 20) MiB in all)")
        return s.allocate(n).map { (slabs.count - 1, $0) }
    }

    /// Repacks one section layer (any thread). Returns its entry id, or 0 if it stays vanilla's.
    func add(_ src: UnsafeRawPointer, vertexCount: Int) -> UInt32 {
        var words: [UInt32] = []
        var st = NearCodecStats()
        guard NearCodec.encode(src, vertexCount: vertexCount, into: &words, stats: &st) else {
            lock.lock(); layersRejected += 1; lock.unlock()
            return 0
        }
        let slots = words.count / NearCodec.recordWords
        lock.lock()
        guard let (slab, start) = allocate(slots) else {
            layersRejected += 1
            lock.unlock()
            return 0
        }
        let buffer = slabs[slab].buffer
        lock.unlock()
        // The range is ours alone until the entry is published.
        words.withUnsafeBytes { buffer.contents().advanced(by: start * Self.slotBytes).copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        lock.lock()
        defer { lock.unlock() }
        let index: Int
        if let i = freeEntries.popLast() { index = i } else {
            guard entries.count < 0xFF_FFFE else { slabs[slab].release(start, slots); layersRejected += 1; return 0 }
            entries.append(Entry())
            index = entries.count - 1
        }
        var e = entries[index]
        e.slab = Int32(slab)
        e.start = Int32(start)
        e.slots = Int32(slots)
        e.quads = Int32(vertexCount / 4)
        e.gen = (e.gen &+ 1) & 0x7F   // ids stay positive as Java ints
        e.live = true
        entries[index] = e
        layersPacked += 1
        stats.quads += st.quads
        stats.aligned += st.aligned
        stats.generic += st.generic
        stats.tinted += st.tinted
        stats.tintMisses += st.tintMisses
        liveQuads += st.quads
        liveSlots += slots
        return UInt32(index + 1) | UInt32(e.gen) << 24
    }

    /// The live entry for an id. Under the lock.
    @inline(__always) func entry(_ id: UInt32) -> Entry? {
        let index = Int(id & 0xFF_FFFF) - 1
        guard index >= 0, index < entries.count else { return nil }
        let e = entries[index]
        return e.live && e.gen == UInt8(id >> 24) ? e : nil
    }

    /// Frees an entry (any thread). Its slots are reused once the GPU is past every draw that may use them.
    func free(_ id: UInt32) {
        lock.lock()
        defer { lock.unlock() }
        guard let e = entry(id) else { return }
        let index = Int(id & 0xFF_FFFF) - 1
        entries[index].live = false
        freeEntries.append(index)
        pending.append((Int(e.slab), Range(start: Int(e.start), count: Int(e.slots)), lastDrawSubmit))
        liveQuads -= Int(e.quads)
        liveSlots -= Int(e.slots)
    }
}

// MARK: - Shaders

// Minecraft's uniform blocks as std140 lays them out (Globals, Fog, TerrainUniform, Projection), and its per-section
// instanced stream (ChunkPosition, ChunkVisibility), bound straight from the buffers vanilla filled for its own draws.
// The math follows core/terrain.vsh and terrain.fsh (fog.glsl, sample_lightmap.glsl, texture_sampling.glsl) operation
// for operation, with the vertical flip every translated Minecraft vertex shader gets (flip_vert_y).
let nearShaderSource = """
#include <metal_stdlib>
using namespace metal;

// Lit mode (METALMC_EXP=lit, Lit.swift): the near chunks also write the terrain G-buffer.
#define LIT_MODE \(litEnabled ? 1 : 0)
\(litShaderHeader)

struct NearGlobals {
    packed_int3 cameraBlockPos;
    float glintAlpha;
    packed_float3 cameraOffset;
    float gameTime;
    float2 screenSize;
    int menuBlurRadius;
    int useRgss;
};

struct NearFog {
    float4 color;
    float environmentalStart;
    float environmentalEnd;
    float renderDistanceStart;
    float renderDistanceEnd;
    float skyEnd;
    float cloudsEnd;
};

struct NearTerrain {
    float4x4 modelView;
    int2 textureSize;
};

struct NearSection {
    packed_int3 position;
    float visibility;
};

struct NearCorner {
    float3 pos;
    float2 uv;
    uint3 rgb;
    uint2 light;
};

constant bool nearCutout [[function_constant(0)]];

// Vertex k of record `slot` (see NearChunks.swift for the layout).
static NearCorner nearDecode(device const uint* q, uint slot, uint k) {
    device const uint* s = q + slot * 8;
    uint w0 = s[0];
    NearCorner c;
    if ((w0 & 1u) != 0) {
        device const uint* v = k == 0u ? s + 1 : s + (w0 >> 1) + (k - 1u) * 4u;
        uint a = v[0], b = v[1], d = v[2], e = v[3];
        uint3 p = uint3(a & 0x3FFFFu, (a >> 18) | ((b & 0xFu) << 14), (b >> 4) & 0x3FFFFu);
        c.pos = float3(p) * (1.0 / 8192.0) - 8.0;
        c.rgb = uint3((b >> 22) & 0xFFu, d & 0xFFu, (d >> 8) & 0xFFu);
        c.light = uint2((d >> 16) & 0xFFu, d >> 24);
        uint2 uvq = uint2(e & 0xFFFFu, e >> 16) | ((uint2(b >> 30, b >> 31) & 1u) << 16);
        c.uv = float2(uvq) * (1.0 / 65536.0);
        return c;
    }
    uint w1 = s[1];
    uint axis = (w0 >> 1) & 3u, start = (w0 >> 3) & 3u, reverse = (w0 >> 5) & 1u, layout = (w0 >> 6) & 1u;
    uint idx = (reverse != 0u ? start + 4u - k : start + k) & 3u;
    int su = (idx == 1u || idx == 2u) ? 1 : 0;
    int sv = idx >= 2u ? 1 : 0;
    int3 p = int3(int((w0 >> 7) & 0x1FFu), int((w0 >> 16) & 0x1FFu), int(w1 & 0x1FFu)) - 16;
    int e1 = int((w1 >> 9) & 0x1FFu), e2 = int((w1 >> 18) & 0x1FFu);
    if (axis == 0u) { p.y += su * e1; p.z += sv * e2; }
    else if (axis == 1u) { p.x += su * e1; p.z += sv * e2; }
    else { p.x += su * e1; p.y += sv * e2; }
    c.pos = float3(p) * (1.0 / 16.0);
    uint a = s[2], cc = s[3], w5 = s[5];
    uint2 uvA = uint2(a & 0xFFFFu, a >> 16) | ((uint2(w5 >> 24, w5 >> 25) & 1u) << 16);
    uint2 uvC = uint2(cc & 0xFFFFu, cc >> 16) | ((uint2(w5 >> 26, w5 >> 27) & 1u) << 16);
    uint2 uvq;
    if (k == 0u) uvq = uvA;
    else if (k == 2u) uvq = uvC;
    else uvq = ((k == 1u) == (layout != 0u)) ? uint2(uvA.x, uvC.y) : uint2(uvC.x, uvA.y);
    c.uv = float2(uvq) * (1.0 / 65536.0);
    uint w4 = s[4];
    uint3 tint = uint3(w4 & 0xFFu, (w4 >> 8) & 0xFFu, (w4 >> 16) & 0xFFu);
    uint gray = k == 0u ? (w4 >> 24) : ((w5 >> (8u * (k - 1u))) & 0xFFu);
    c.rgb = tint * gray / 255u;
    uint lw = s[6 + (k >> 1)] >> (16u * (k & 1u));
    c.light = uint2(lw & 0xFFu, (lw >> 8) & 0xFFu);
    return c;
}

// Invariant, so the solid and cutout pipelines put a face at exactly the same depth: the grass side overlay (cutout) sits on
// its dirt side (solid) and passes the depth test only at equal depth.
struct NearVertex {
    float4 position [[position, invariant]];
    float sphericalVertexDistance;
    float cylindricalVertexDistance;
    float4 vertexColor;
    float2 texCoord0;
    float chunkVisibility;
#if LIT_MODE
    // Lit mode: vertexColor is unlit (rgb: the vertex color, a: its gray level, 1 for generic quads); the light levels
    // (block, sky) are interpolated like vanilla's light coordinates and the lightmap is sampled per pixel; the face.
    float2 lightLevels;
    uint litFace [[flat]];
#endif
};

#if LIT_MODE
// Vanilla's face shade (top 1, bottom 0.5, x 0.6, z 0.8): the gray levels have it, the G-buffer's AO doesn't.
constant float kNearShade[6] = { 0.6, 0.6, 1.0, 0.5, 0.8, 0.8 };

// Vertex k's gray level (0-255) for an axis-aligned record, 255 for a generic one (its colors aren't split).
static uint nearGray(device const uint* q, uint slot, uint k) {
    device const uint* s = q + slot * 8;
    if ((s[0] & 1u) != 0u) return 255u;
    return k == 0u ? (s[4] >> 24) : ((s[5] >> (8u * (k - 1u))) & 0xFFu);
}

// The face for the G-buffer (0-5, the LOD's numbering, or 7): an axis-aligned record's plane axis, facing the camera
// (back faces are culled, so what's drawn faces it); a generic quad's from its corners, if it's axis-aligned.
static uint nearFace(device const uint* q, uint slot, float3 rel) {
    device const uint* s = q + slot * 8;
    uint axis;
    if ((s[0] & 1u) == 0u) {
        axis = (s[0] >> 1) & 3u;
    } else {
        float3 a = nearDecode(q, slot, 0u).pos, b = nearDecode(q, slot, 1u).pos, c = nearDecode(q, slot, 2u).pos;
        float3 n = cross(b - a, c - a);
        float3 an = abs(n);
        float m = max(an.x, max(an.y, an.z));
        if (!(m > 0.0) || m < 0.999 * length(n)) return 7u;
        axis = an.x == m ? 0u : (an.y == m ? 1u : 2u);
    }
    return axis * 2u + (rel[axis] > 0.0 ? 1u : 0u);
}
#endif

vertex NearVertex near_vs(uint vid [[vertex_id]], uint section [[base_instance]],
                          device const uint* quads [[buffer(16)]],
                          device const NearSection* sections [[buffer(17)]],
                          constant float4x4& projMat [[buffer(18)]],
                          constant NearTerrain& terrain [[buffer(19)]],
                          constant NearGlobals& globals [[buffer(20)]],
                          texture2d<float> lightmap [[texture(16)]], sampler lightmapSampler [[sampler(12)]]) {
    NearCorner c = nearDecode(quads, vid >> 2, vid & 3u);
    NearSection sec = sections[section];
    float3 pos = c.pos + float3(int3(sec.position) - int3(globals.cameraBlockPos)) + float3(globals.cameraOffset);
    NearVertex o;
    o.position = (projMat * terrain.modelView) * float4(pos, 1.0);
    o.position.y = -o.position.y;
    o.sphericalVertexDistance = length(pos);
    o.cylindricalVertexDistance = max(length(pos.xz), abs(pos.y));
    float2 lightUv = clamp((float2(c.light) / 256.0) + 0.5 / 16.0, float2(0.5 / 16.0), float2(15.5 / 16.0));
    // The same unorm conversion as the vertex fetch of vanilla's RGBA8 color (a division by 255 may round differently).
    float4 color = unpack_unorm4x8_to_float(c.rgb.x | (c.rgb.y << 8) | (c.rgb.z << 16) | 0xFF000000u);
#if LIT_MODE
    (void)lightUv;
    o.vertexColor = float4(color.rgb, float(nearGray(quads, vid >> 2, vid & 3u)) / 255.0);
    o.lightLevels = float2(c.light) / 16.0;
    o.litFace = nearFace(quads, vid >> 2, pos);
#else
    o.vertexColor = color * lightmap.sample(lightmapSampler, lightUv, level(0.0));
#endif
    o.texCoord0 = c.uv;
    float dist = length(pos);
    o.chunkVisibility = mix(1.0, sec.visibility, clamp((dist - 16.0) / 16.0, 0.0, 1.0));
    return o;
}

static float4 sampleNearest(texture2d<float> source, sampler s, float2 uv, float2 pixelSize, float2 du, float2 dv, float2 texelScreenSize) {
    float2 uvTexelCoords = uv / pixelSize;
    float2 texelCenter = round(uvTexelCoords) - 0.5f;
    float2 texelOffset = uvTexelCoords - texelCenter;
    texelOffset = (texelOffset - 0.5f) * pixelSize / texelScreenSize + 0.5f;
    texelOffset = clamp(texelOffset, 0.0f, 1.0f);
    uv = (texelCenter + texelOffset) * pixelSize;
    return source.sample(s, uv, gradient2d(du, dv));
}

static float4 sampleRGSS(texture2d<float> source, sampler s, float2 uv, float2 pixelSize, float2 du, float2 dv) {
    float2 texelScreenSize = sqrt(du * du + dv * dv);
    float maxTexelSize = max(texelScreenSize.x, texelScreenSize.y);
    float minPixelSize = min(pixelSize.x, pixelSize.y);
    float transitionStart = minPixelSize * 1.0;
    float transitionEnd = minPixelSize * 2.0;
    float blendFactor = smoothstep(transitionStart, transitionEnd, maxTexelSize);
    float duLength = length(du);
    float dvLength = length(dv);
    float minDerivative = min(duLength, dvLength);
    float maxDerivative = max(duLength, dvLength);
    float effectiveDerivative = sqrt(minDerivative * maxDerivative);
    float mipLevelExact = max(0.0, log2(effectiveDerivative / minPixelSize));
    float mipLevelLow = floor(mipLevelExact);
    float mipLevelHigh = mipLevelLow + 1.0;
    float mipBlend = fract(mipLevelExact);
    const float2 offsets[4] = { float2(0.125, 0.375), float2(-0.125, -0.375), float2(0.375, -0.125), float2(-0.375, 0.125) };
    float4 rgssColorLow = float4(0.0);
    float4 rgssColorHigh = float4(0.0);
    for (int i = 0; i < 4; ++i) {
        float2 sampleUV = uv + offsets[i] * pixelSize;
        rgssColorLow += source.sample(s, sampleUV, level(mipLevelLow));
        rgssColorHigh += source.sample(s, sampleUV, level(mipLevelHigh));
    }
    rgssColorLow *= 0.25;
    rgssColorHigh *= 0.25;
    float4 rgssColor = mix(rgssColorLow, rgssColorHigh, mipBlend);
    float4 nearestColor = sampleNearest(source, s, uv, pixelSize, du, dv, texelScreenSize);
    return mix(nearestColor, rgssColor, blendFactor);
}

static float linearFogValue(float vertexDistance, float fogStart, float fogEnd) {
    if (vertexDistance <= fogStart) {
        return 0.0;
    } else if (vertexDistance >= fogEnd) {
        return 1.0;
    }
    return (vertexDistance - fogStart) / (fogEnd - fogStart);
}

#if LIT_MODE
// Lit mode: the color as without it (the lightmap sampled per pixel at the interpolated light coordinates instead of per
// vertex), and the G-buffer: the texel times the tint (the vertex color over its gray level), the gray level over the
// face's shade as AO, the face, the light levels.
struct NearOut { float4 color [[color(0)]]; uint2 gbuf [[color(1)]]; };

fragment NearOut near_fs(NearVertex in [[stage_in]],
                         constant NearTerrain& terrain [[buffer(19)]],
                         constant NearGlobals& globals [[buffer(20)]],
                         constant NearFog& fog [[buffer(21)]],
                         texture2d<float> atlas [[texture(17)]], sampler atlasSampler [[sampler(13)]],
                         texture2d<float> lightmap [[texture(16)]], sampler lightmapSampler [[sampler(12)]]) {
    float2 pixelSize = 1.0f / float2(terrain.textureSize);
    float2 du = dfdx(in.texCoord0);
    float2 dv = dfdy(in.texCoord0);
    float4 sampled = globals.useRgss == 1 ? sampleRGSS(atlas, atlasSampler, in.texCoord0, pixelSize, du, dv)
                                          : sampleNearest(atlas, atlasSampler, in.texCoord0, pixelSize, du, dv, sqrt(du * du + dv * dv));
    float2 lightUv = clamp(in.lightLevels / 16.0 + 0.5 / 16.0, float2(0.5 / 16.0), float2(15.5 / 16.0));
    float4 color = sampled * float4(in.vertexColor.rgb * lightmap.sample(lightmapSampler, lightUv, level(0.0)).rgb, 1.0);
    color = mix(fog.color * float4(1, 1, 1, color.a), color, in.chunkVisibility);
    if (nearCutout && color.a < 0.5) {
        discard_fragment();
    }
    float fogValue = max(linearFogValue(in.sphericalVertexDistance, fog.environmentalStart, fog.environmentalEnd),
                         linearFogValue(in.cylindricalVertexDistance, fog.renderDistanceStart, fog.renderDistanceEnd));
    NearOut out;
    out.color = float4(mix(color.rgb, fog.color.rgb, fogValue * fog.color.a), color.a);
    float gray = max(in.vertexColor.a, 1.0 / 255.0);
    float shade = in.litFace < 6u ? kNearShade[in.litFace] : 1.0;
    out.gbuf = litPack(sampled.rgb * saturate(in.vertexColor.rgb / gray), gray / shade, in.litFace, in.position.z,
                       in.lightLevels.y, in.lightLevels.x);
    return out;
}
#else
fragment float4 near_fs(NearVertex in [[stage_in]],
                        constant NearTerrain& terrain [[buffer(19)]],
                        constant NearGlobals& globals [[buffer(20)]],
                        constant NearFog& fog [[buffer(21)]],
                        texture2d<float> atlas [[texture(17)]], sampler atlasSampler [[sampler(13)]]) {
    float2 pixelSize = 1.0f / float2(terrain.textureSize);
    float2 du = dfdx(in.texCoord0);
    float2 dv = dfdy(in.texCoord0);
    float4 sampled = globals.useRgss == 1 ? sampleRGSS(atlas, atlasSampler, in.texCoord0, pixelSize, du, dv)
                                          : sampleNearest(atlas, atlasSampler, in.texCoord0, pixelSize, du, dv, sqrt(du * du + dv * dv));
    float4 color = sampled * in.vertexColor;
    color = mix(fog.color * float4(1, 1, 1, color.a), color, in.chunkVisibility);
    if (nearCutout && color.a < 0.5) {
        discard_fragment();
    }
    float fogValue = max(linearFogValue(in.sphericalVertexDistance, fog.environmentalStart, fog.environmentalEnd),
                         linearFogValue(in.cylindricalVertexDistance, fog.renderDistanceStart, fog.renderDistanceEnd));
    return float4(mix(color.rgb, fog.color.rgb, fogValue * fog.color.a), color.a);
}
#endif

// Offline check (mmc_near_debug_decode_gpu): every vertex of `quadCount` records through nearDecode.
kernel void near_decode_test(device const uint* quads [[buffer(0)]], device float* out [[buffer(1)]],
                             constant uint& quadCount [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (i >= quadCount * 4u) return;
    NearCorner c = nearDecode(quads, i >> 2, i & 3u);
    device float* o = out + i * 10u;
    o[0] = c.pos.x; o[1] = c.pos.y; o[2] = c.pos.z;
    o[3] = c.uv.x; o[4] = c.uv.y;
    o[5] = float(c.rgb.x); o[6] = float(c.rgb.y); o[7] = float(c.rgb.z);
    o[8] = float(c.light.x); o[9] = float(c.light.y);
}
"""

// MARK: - Renderer

// Binding slots, clear of vanilla's (uniform i is index i, push constants 24, vertex slots 30 and 29) and set through
// ctx's binding caches, which are cleared again after the draw.
private let nearQuadsIndex = 16, nearSectionsIndex = 17, nearProjectionIndex = 18, nearTerrainIndex = 19
private let nearGlobalsIndex = 20, nearFogIndex = 21
private let nearLightmapIndex = 16, nearAtlasIndex = 17
private let nearLightmapSamplerIndex = 12, nearAtlasSamplerIndex = 13

final class NearRenderer: @unchecked Sendable {
    static let shared = NearRenderer()
    let lock = NSLock()
    /// 0 not started, 1 compiling, 2 ready, 3 failed.
    var state = 0
    var library: MTLLibrary?
    var pipelines: [String: MTLRenderPipelineState] = [:]
    var indexBuffer: MTLBuffer?
    var indexQuads = 0
    // Render-thread counters: draw calls, quads, CPU nanoseconds in mmc_near_draw, records whose entry was gone.
    var statDraws = 0, statQuads = 0, statNanos: UInt64 = 0, statMissing = 0
    var frames = 0

    var failed: Bool {
        lock.lock()
        defer { lock.unlock() }
        return state == 3
    }

    /// Compiles the shader library on a background queue (first call only); true once it's ready.
    func ready() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if state == 0 {
            state = 1
            DispatchQueue.global(qos: .userInitiated).async { [self] in
                let t0 = DispatchTime.now().uptimeNanoseconds
                do {
                    let opts = MTLCompileOptions()
                    opts.languageVersion = .version3_0
                    opts.preserveInvariance = true
                    let lib = try ctx.device.makeLibrary(source: nearShaderSource, options: opts)
                    lock.lock(); library = lib; lock.unlock()
                    // The main pass's usual formats, so the first frame doesn't compile on the render thread (in lit mode,
                    // with the G-buffer).
                    let formats: [MTLPixelFormat] = litEnabled ? [hdrOutput ? .rgba16Float : .rgba8Unorm, litGbufferFormat] : [.rgba8Unorm]
                    for cutout in [false, true] { _ = pipeline(cutout: cutout, colorFormats: formats, depth: .depth32Float) }
                    lock.lock(); state = 2; lock.unlock()
                    log("near chunks: shaders compiled in \((DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000) ms")
                } catch {
                    lock.lock(); state = 3; lock.unlock()
                    log("near chunks: shader compile failed, vanilla keeps drawing near terrain: \(error)")
                }
            }
        }
        return state == 2
    }

    func pipeline(cutout: Bool, colorFormats: [MTLPixelFormat], depth: MTLPixelFormat) -> MTLRenderPipelineState? {
        let key = (cutout ? "c/" : "s/") + colorFormats.map { String($0.rawValue) }.joined(separator: ",") + "/\(depth.rawValue)"
        lock.lock()
        if let p = pipelines[key] { lock.unlock(); return p }
        let lib = library
        lock.unlock()
        guard let lib else { return nil }
        do {
            let constants = MTLFunctionConstantValues()
            var cut = cutout
            constants.setConstantValue(&cut, type: .bool, index: 0)
            let d = MTLRenderPipelineDescriptor()
            d.label = cutout ? "near chunks cutout" : "near chunks solid"
            d.vertexFunction = lib.makeFunction(name: "near_vs")
            d.fragmentFunction = try lib.makeFunction(name: "near_fs", constantValues: constants)
            for (i, f) in colorFormats.enumerated() {
                d.colorAttachments[i].pixelFormat = f
                // Vanilla's terrain pipelines write color target 0; leave any others alone, except lit mode's G-buffer.
                d.colorAttachments[i].writeMask = i == 0 || litWritesGbuffer(i, f) ? .all : []
            }
            d.depthAttachmentPixelFormat = depth
            if depth == .depth32Float_stencil8 { d.stencilAttachmentPixelFormat = depth }
            let p = try ctx.device.makeRenderPipelineState(descriptor: d)
            lock.lock(); pipelines[key] = p; lock.unlock()
            return p
        } catch {
            log("near chunks: pipeline \(key) failed: \(error)")
            return nil
        }
    }

    /// A shared sequential quad index buffer (0 1 2 2 3 0 per quad, like vanilla's) for at least `quads` quads.
    func indices(_ quads: Int) -> MTLBuffer? {
        if let b = indexBuffer, indexQuads >= quads { return b }
        var n = max(indexQuads, 4096)
        while n < quads { n *= 2 }
        var idx = [UInt32](repeating: 0, count: n * 6)
        for q in 0..<n {
            let v = UInt32(4 * q)
            idx[6 * q] = v; idx[6 * q + 1] = v + 1; idx[6 * q + 2] = v + 2
            idx[6 * q + 3] = v + 2; idx[6 * q + 4] = v + 3; idx[6 * q + 5] = v
        }
        guard let b = ctx.device.makeBuffer(bytes: idx, length: idx.count * 4, options: [.storageModeShared]) else { return nil }
        b.label = "MetalMC near chunk indices"
        indexBuffer = b
        indexQuads = n
        return b
    }
}

// MARK: - Entry points

/// 1 if near chunks are on (METALMC_EXP=nearchunks) and their shaders are compiled; starts compiling them on the first
/// call. The mod diverts vanilla's solid and cutout draws only then.
@_cdecl("mmc_near_ready")
public func mmc_near_ready() -> Int32 {
    guard nearChunksEnabled else { return 0 }
    return NearRenderer.shared.ready() ? 1 : 0
}

/// Repacks one section layer of vanilla BLOCK vertices (28 bytes each, 4 per quad; any thread, the mesh workers).
/// Returns the entry id (always positive), or 0 if the layer stays vanilla's.
@_cdecl("mmc_near_add")
public func mmc_near_add(_ vertices: UnsafeRawPointer, _ vertexCount: Int32) -> Int32 {
    guard nearChunksEnabled, !NearRenderer.shared.failed else { return 0 }
    return Int32(bitPattern: NearArena.shared.add(vertices, vertexCount: Int(vertexCount)))
}

/// Frees an entry (any thread; the section mesh closed).
@_cdecl("mmc_near_free")
public func mmc_near_free(_ id: Int32) {
    guard id != 0 else { return }
    NearArena.shared.free(UInt32(bitPattern: id))
}

/// Draws `count` records {entry id, first quad, quad count, section index} of one layer (0 solid, 1 cutout) into the open
/// render pass (the main pass, right after vanilla's own draws of that layer). `submit` is the submit this frame
/// records into. res: vanilla's per-section stream (buffer, offset), then its Projection, TerrainUniform, Globals and Fog
/// uniform buffers (buffer, offset each), the block atlas (view, sampler) and the lightmap (view, sampler): 14 values.
/// Returns the records drawn, or -1 if the pipeline isn't available. Leaves ctx.pipe nil so Java re-applies Minecraft's
/// pipeline state.
@_cdecl("mmc_near_draw")
public func mmc_near_draw(_ layer: Int32, _ records: UnsafePointer<Int32>, _ count: Int32, _ submit: Int64,
                          _ res: UnsafePointer<Int64>, _ wireframe: Int32) -> Int32 {
    let t0 = DispatchTime.now().uptimeNanoseconds
    let r = NearRenderer.shared
    defer { r.statNanos += DispatchTime.now().uptimeNanoseconds - t0 }
    guard count > 0 else { return 0 }
    guard let enc = ctx.pass, !ctx.scissorEmpty else { return 0 }
    for i in [0, 2, 4, 6, 8, 10, 11, 12, 13] where res[i] == 0 { return -1 }
    guard let pipe = r.pipeline(cutout: layer == 1, colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat) else { return -1 }
    let arena = NearArena.shared
    var maxQuads = 0
    for i in 0..<Int(count) { maxQuads = max(maxQuads, Int(records[4 * i + 2])) }
    guard let ib = r.indices(maxQuads) else { return -1 }

    enc.setRenderPipelineState(pipe)
    enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: true))
    enc.setCullMode(.back)
    enc.setTriangleFillMode(wireframe != 0 ? .lines : .fill)
    enc.setDepthBias(0, slopeScale: 0, clamp: 0)
    func buffer(_ i: Int) -> MTLBuffer { (from(res[2 * i]) as BufferBox).buffer }
    ctx.bindBuffer(enc, 0, buffer(0), Int(res[1]), nearSectionsIndex)
    ctx.bindBuffer(enc, 0, buffer(1), Int(res[3]), nearProjectionIndex)
    ctx.bindBuffer(enc, 0, buffer(2), Int(res[5]), nearTerrainIndex)
    ctx.bindBuffer(enc, 1, buffer(2), Int(res[5]), nearTerrainIndex)
    ctx.bindBuffer(enc, 0, buffer(3), Int(res[7]), nearGlobalsIndex)
    ctx.bindBuffer(enc, 1, buffer(3), Int(res[7]), nearGlobalsIndex)
    ctx.bindBuffer(enc, 1, buffer(4), Int(res[9]), nearFogIndex)
    ctx.bindTexture(enc, 1, (from(res[10]) as TextureBox).texture, nearAtlasIndex)
    ctx.bindSampler(enc, 1, (from(res[11]) as SamplerBox).state, nearAtlasSamplerIndex)
    ctx.bindTexture(enc, 0, (from(res[12]) as TextureBox).texture, nearLightmapIndex)
    ctx.bindSampler(enc, 0, (from(res[13]) as SamplerBox).state, nearLightmapSamplerIndex)
    if litEnabled {
        // Lit mode: the fragment shader samples the lightmap (per pixel).
        ctx.bindTexture(enc, 1, (from(res[12]) as TextureBox).texture, nearLightmapIndex)
        ctx.bindSampler(enc, 1, (from(res[13]) as SamplerBox).state, nearLightmapSamplerIndex)
    }

    var drawn = 0
    arena.lock.lock()
    arena.lastDrawSubmit = max(arena.lastDrawSubmit, submit)
    for i in 0..<Int(count) {
        let rec = records + 4 * i
        guard let e = arena.entry(UInt32(bitPattern: rec[0])) else { r.statMissing += 1; continue }
        let first = Int(rec[1]), n = min(Int(rec[2]), Int(e.quads) - first)
        guard first >= 0, n > 0 else { continue }
        ctx.bindBuffer(enc, 0, arena.slabs[Int(e.slab)].buffer, 0, nearQuadsIndex)
        enc.drawIndexedPrimitives(type: .triangle, indexCount: n * 6, indexType: .uint32, indexBuffer: ib, indexBufferOffset: 0,
                                  instanceCount: 1, baseVertex: 4 * (Int(e.start) + first), baseInstance: Int(rec[3]))
        drawn += 1
        r.statQuads += n
    }
    arena.lock.unlock()
    r.statDraws += drawn
    ctx.statDraws += drawn

    // Minecraft's pipeline, depth, cull, fill and bias state must be re-applied by the next setPipeline, and nothing
    // may assume our bindings are still there.
    ctx.pipe = nil
    ctx.boundPipeState = nil
    for i in [nearQuadsIndex, nearSectionsIndex, nearProjectionIndex, nearTerrainIndex, nearGlobalsIndex, nearFogIndex] {
        ctx.boundBuffers[0][i] = nil; ctx.boundBuffers[1][i] = nil
        ctx.boundOffsets[0][i] = -1; ctx.boundOffsets[1][i] = -1
    }
    for i in [nearLightmapIndex, nearAtlasIndex] { ctx.boundTextures[0][i] = nil; ctx.boundTextures[1][i] = nil }
    for i in [nearLightmapSamplerIndex, nearAtlasSamplerIndex] { ctx.boundSamplers[0][i] = nil; ctx.boundSamplers[1][i] = nil }

    if layer == 1 {
        r.frames += 1
        if r.frames % 1200 == 0 {
            log(nearStatsLine())
            r.statDraws = 0; r.statQuads = 0; r.statNanos = 0; r.statMissing = 0
        }
    }
    return Int32(drawn)
}

func nearStatsLine() -> String {
    let a = NearArena.shared, r = NearRenderer.shared
    a.lock.lock()
    let s = a.stats, packed = a.layersPacked, rejected = a.layersRejected, liveQuads = a.liveQuads, liveSlots = a.liveSlots
    let slabs = a.slabs.count
    a.lock.unlock()
    let perQuad = liveQuads > 0 ? Double(liveSlots * NearArena.slotBytes) / Double(liveQuads) : 0
    let alignedPct = s.quads > 0 ? 100.0 * Double(s.aligned) / Double(s.quads) : 0
    return String(format: "near chunks: %ld layers repacked, %ld left to vanilla; live %ld quads in %.1f MB (%.1f B/quad vs 112), %ld slabs; "
                  + "%.1f%% aligned, %ld tinted, %ld tint misses; per frame over the last 1200 (solid + cutout): %.0f draws, %.0f quads, "
                  + "%.3f ms CPU, %ld records missing",
                  packed, rejected, liveQuads, Double(liveSlots * NearArena.slotBytes) / 1e6, perQuad, slabs,
                  alignedPct, s.tinted, s.tintMisses, Double(r.statDraws) / 1200, Double(r.statQuads) / 1200,
                  Double(r.statNanos) / 1e6 / 1200, r.statMissing)
}

/// out: layers repacked, layers left to vanilla, quads encoded, aligned, generic, tinted, tint misses, live quads, live
/// bytes, slabs, draws, quads drawn, CPU nanoseconds in mmc_near_draw, missing records (the last four reset).
@_cdecl("mmc_near_stats")
public func mmc_near_stats(_ out: UnsafeMutablePointer<Int64>) {
    let a = NearArena.shared, r = NearRenderer.shared
    a.lock.lock()
    out[0] = Int64(a.layersPacked); out[1] = Int64(a.layersRejected); out[2] = Int64(a.stats.quads)
    out[3] = Int64(a.stats.aligned); out[4] = Int64(a.stats.generic); out[5] = Int64(a.stats.tinted)
    out[6] = Int64(a.stats.tintMisses); out[7] = Int64(a.liveQuads); out[8] = Int64(a.liveSlots * NearArena.slotBytes)
    out[9] = Int64(a.slabs.count)
    a.lock.unlock()
    out[10] = Int64(r.statDraws); out[11] = Int64(r.statQuads); out[12] = Int64(r.statNanos); out[13] = Int64(r.statMissing)
    r.statDraws = 0; r.statQuads = 0; r.statNanos = 0; r.statMissing = 0
}
