import CoreGraphics
import Foundation
import ImageIO
import Metal
import MetalMCCore
import simd

// Offline checks of the far field's columns (tools/fartest.py drives them; docs/far-field-design.md has the results):
// the march's shader source, a region's columns at any level (old voxel tops or cells), an evaluation of the cell
// statistics against level-1 truth from sample cameras, the heights of real columns against the generator's at the
// same places, and a render of the march over an open world to a PNG. None of them runs in the game.

/// Debug: the far field's shader source with its constants filled in (for an offline compile check). Returns its length.
@_cdecl("mmc_debug_far_shader_source")
public func mmc_debug_far_shader_source(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let bytes = Array(farFieldShaderSource.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}

/// Debug: one region's columns at `level`, the region in the low corner of the node (512 >> level cells a side), as the
/// far field gets them: two words per column, 256 x 256. `mode` 0: cells (block precision), 1: the voxels' tops as
/// before (LodBuild.voxelWord), 2: cells through the region's level-2 quadrant (LodQuadrant, as cached) and back.
/// Returns the columns with data.
@_cdecl("mmc_debug_far_region")
public func mmc_debug_far_region(_ path: UnsafePointer<CChar>, _ level: Int32, _ mode: Int32, _ out: UnsafeMutablePointer<UInt32>) -> Int32 {
    guard var g = LodBuild.regionGrid(path: String(cString: path)) else { return 0 }
    if mode == 1 { g.far = [] }
    while g.level < Int(level) {
        var p = LodGrid(level: g.level + 1)
        g.downsample(into: &p, qx: 0, qz: 0)
        g = p
        if mode == 2 && g.level == 2 {
            var q = LodGrid(level: 2)
            LodQuadrant(grid: g).expand(into: &q, qx: 0, qz: 0)
            g = q
        }
    }
    let cols = mode == 1 ? (0..<(lodNodeVoxels * lodNodeVoxels)).flatMap { [LodBuild.voxelWord(g, $0), 0] } : LodBuild.farColumns(g)
    var any: Int32 = 0
    for (i, w) in cols.enumerated() { out[i] = w; if i % 2 == 0 && w != 0 { any += 1 } }
    return any
}

/// The top a column's words draw (blocks above the world bottom): the ground or the water surface (10/9 block under
/// its word's top, as the shader draws it), or the canopy's top over it. 0 for none.
@inline(__always) private func drawnTop(_ w0: UInt32, _ w1: UInt32) -> Float {
    if w0 == 0 { return 0 }
    let depth = (w0 >> 9) & 127
    let ground = Float((w0 & 511) + depth) - (depth != 0 ? 10.0 / 9.0 : 0)
    return max(ground, Float(w1 & 511))
}

/// Height maps of the regions in `dir` at levels 1-6 for the evaluation: [variant][level] -> side x side tops, with the
/// grid's origin at block (x0, z0). Variants: 0 voxel tops (the first far field), then cells with the canopy drawn
/// 1 in a share of cells equal to its cover, 2 where at least half covered, 3 where any; 4-5 cells merged by their
/// maximum instead of their mean (canopy where any, where half covered). Level 1 of variant 3 is the truth.
private struct FarEvalMaps {
    var x0 = 0, z0 = 0, side1 = 0
    var tops: [[[Float]]] = []          // [variant][level]
    var cells: [[[LodFarCell]]] = []    // [0 mean, 1 max][level]: the cells, for the seam check
    var dryShare: [[Double]] = []       // per level: share of columns with data that are dry
}

private func farEvalRegions(_ dir: String) -> [(Int, Int, String)] {
    let files = ((try? FileManager.default.contentsOfDirectory(atPath: dir)) ?? []).filter { $0.hasSuffix(".mca") }
    return files.compactMap { f in
        let p = f.split(separator: ".")
        guard p.count == 4, let x = Int(p[1]), let z = Int(p[2]) else { return nil }
        return (x, z, (dir as NSString).appendingPathComponent(f))
    }
}

private func farEvalBuild(_ dir: String) -> FarEvalMaps {
    let regions = farEvalRegions(dir)
    var m = FarEvalMaps()
    let rx0 = regions.map { $0.0 }.min() ?? 0, rz0 = regions.map { $0.1 }.min() ?? 0
    let rx1 = regions.map { $0.0 }.max() ?? 0, rz1 = regions.map { $0.1 }.max() ?? 0
    let span = max(rx1 - rx0, rz1 - rz0) + 1
    m.x0 = rx0 * 512; m.z0 = rz0 * 512; m.side1 = span * 256
    let variants = 6, levels = 7
    m.tops = (0..<variants).map { _ in (0..<levels).map { l in l == 0 ? [] : [Float](repeating: 0, count: (span * (512 >> l)) * (span * (512 >> l))) } }
    m.cells = (0..<2).map { _ in (0..<levels).map { l in l == 0 ? [] : [LodFarCell](repeating: LodFarCell(), count: (span * (512 >> l)) * (span * (512 >> l))) } }
    let lock = NSLock()
    for pass in 0..<2 {
        lodFarMergeMax = pass == 0 ? 0 : 3
        DispatchQueue.concurrentPerform(iterations: regions.count) { k in
            let (rx, rz, path) = regions[k]
            guard var g = LodBuild.regionGrid(path: path) else { return }
            var vox = g
            vox.far = []
            var out: [(Int, Int, [LodFarCell], [Float], [Float], [Float], [Float])] = []   // level, side, cells, tops of variants
            for l in 1...6 {
                if l > 1 {
                    var p = LodGrid(level: l)
                    g.downsample(into: &p, qx: 0, qz: 0)
                    g = p
                    if pass == 0 {
                        var pv = LodGrid(level: l)
                        vox.downsample(into: &pv, qx: 0, qz: 0)
                        vox = pv
                    }
                }
                let side = 512 >> l, n = lodNodeVoxels
                var cells = [LodFarCell](repeating: LodFarCell(), count: side * side)
                var t = [[Float]](repeating: [Float](repeating: 0, count: side * side), count: 4)
                for z in 0..<side {
                    for x in 0..<side {
                        let i = z * n + x
                        let c = g.far[i].area != 0 ? g.far[i] : LodFarCell.voxels(g, column: i)
                        cells[z * side + x] = c
                        if pass == 0 {
                            t[0][z * side + x] = drawnTop(LodBuild.voxelWord(vox, i), 0)
                            for mode in 0..<3 { let (a, b) = c.words(x: x, z: z, level: l, mode: mode); t[1 + mode][z * side + x] = drawnTop(a, b) }
                        } else {
                            for mode in [2, 1] { let (a, b) = c.words(x: x, z: z, level: l, mode: mode); t[mode == 2 ? 0 : 1][z * side + x] = drawnTop(a, b) }
                        }
                    }
                }
                out.append((l, side, cells, t[0], t[1], t[2], t[3]))
            }
            lock.lock()
            for (l, side, cells, a, b, c, d) in out {
                let ox = (rx - rx0) * side, oz = (rz - rz0) * side, full = span * side
                for z in 0..<side {
                    for x in 0..<side {
                        let j = (oz + z) * full + ox + x, s = z * side + x
                        m.cells[pass][l][j] = cells[s]
                        if pass == 0 {
                            m.tops[0][l][j] = a[s]; m.tops[1][l][j] = b[s]; m.tops[2][l][j] = c[s]; m.tops[3][l][j] = d[s]
                        } else {
                            m.tops[4][l][j] = a[s]; m.tops[5][l][j] = b[s]
                        }
                    }
                }
            }
            lock.unlock()
        }
    }
    lodFarMergeMax = 0
    return m
}

private var farEvalCache: (String, FarEvalMaps)?

/// Debug: the cell statistics against level-1 truth, from cameras around the regions of `dir` (see FarEvalMaps for the
/// variants). For each level 2-6 (the band the far field draws it in: 1024 x 2^(level-1) to 1024 x 2^level blocks),
/// eight cameras around the regions' center at world y 150 and 260 look across them; along each pixel-wide azimuth the
/// band's terrain is sampled per block, and per pixel of elevation (70-degree vertical field of view on 2234 px) the
/// first hit is compared. out[(level - 2) * 6 * 8 + variant * 8 + k]: k 0 mean |silhouette error| (px), 1 mean signed
/// (positive: too tall), 2 mean |depth error| of pixels both hit (relative), 3 share of those more than 5% off, 4 pixels
/// per azimuth hit in one and not the other, 5 mean |top - truth| over the level's cells (blocks), 6 mean signed,
/// 7 share of cells with a canopy.
@_cdecl("mmc_debug_far_eval")
public func mmc_debug_far_eval(_ dirC: UnsafePointer<CChar>, _ out: UnsafeMutablePointer<Double>) {
    let dir = String(cString: dirC)
    let m: FarEvalMaps
    if let c = farEvalCache, c.0 == dir { m = c.1 } else { m = farEvalBuild(dir); farEvalCache = (dir, m) }
    let truth = m.tops[3][1], side1 = m.side1
    let pa = 70.0 * Double.pi / 180 / 2234
    let cx = Double(m.x0) + Double(side1), cz = Double(m.z0) + Double(side1)   // the center (level-1 cells are 2 blocks)
    let extent = Double(side1)   // half the regions' side in blocks
    for l in 2...6 {
        let d0 = 1024.0 * Double(1 << (l - 1)), d1 = 1024.0 * Double(1 << l)
        let dist = max(d0 + 1500, extent * 1.2)
        let sideL = side1 >> (l - 1)
        let cands = (0..<6).map { m.tops[$0][l] }
        // Per camera, per variant: rays, |silhouette|, signed silhouette, depth error, bad pixels, pixels both hit, mismatches.
        var acc = [[Double]](repeating: [Double](repeating: 0, count: 6 * 7), count: 16)
        acc.withUnsafeMutableBufferPointer { accp in
            let accb = accp.baseAddress!
            DispatchQueue.concurrentPerform(iterations: 16) { cam in
                var a7 = [Double](repeating: 0, count: 6 * 7)
                let a = Double(cam % 8) * .pi / 4 + 0.3
                let px = cx + cos(a) * dist, pz = cz + sin(a) * dist
                let camH = Double((cam < 8 ? 150 : 260) - lodWorldMinY)
                // Azimuths across the regions: toward the center, plus or minus the angle the square subtends.
                let toC = atan2(cz - pz, cx - px), half = atan(extent * 1.42 / dist)
                let nAz = Int(2 * half / pa)
                for k in 0..<nAz {
                    let az = toC - half + (Double(k) + 0.5) * pa
                    let dx = cos(az), dz = sin(az)
                    // The band, clipped to the regions' square (outside it nothing is hit in either).
                    var lo = d0, hi = d1
                    for (p, dd, mn, mx) in [(px, dx, Double(m.x0), Double(m.x0) + 2 * extent), (pz, dz, Double(m.z0), Double(m.z0) + 2 * extent)] {
                        if abs(dd) < 1e-9 { if p < mn || p >= mx { hi = lo }; continue }
                        let t0 = (mn - p) / dd, t1 = (mx - p) / dd
                        lo = max(lo, min(t0, t1)); hi = min(hi, max(t0, t1))
                    }
                    if hi <= lo { continue }
                    var r = [Double](repeating: -Double.infinity, count: 7)   // running max elevation: truth, then variants
                    var pix = [Int](repeating: 0, count: 7)
                    var hits = [[Double]](repeating: [], count: 7)          // first distance per pixel of elevation
                    var base = 0, first = true, any = false
                    var d = lo.rounded(.up)
                    var e = [Double](repeating: 0, count: 7)
                    while d < hi {
                        let ix = Int(((px + dx * d - Double(m.x0)) / 2).rounded(.down)), iz = Int(((pz + dz * d - Double(m.z0)) / 2).rounded(.down))
                        var hT = 0.0
                        if ix >= 0 && iz >= 0 && ix < side1 && iz < side1 {
                            hT = Double(truth[iz * side1 + ix])
                            let j = (iz >> (l - 1)) * sideL + (ix >> (l - 1))
                            for v in 0..<6 { e[1 + v] = atan2(Double(cands[v][j]) - camH, d) }
                            any = any || hT > 0
                        } else {
                            for v in 0..<6 { e[1 + v] = atan2(-camH, d) }
                        }
                        e[0] = atan2(hT - camH, d)
                        if first {
                            base = Int((e.min()! / pa).rounded(.down))
                            for v in 0..<7 { pix[v] = base }
                            first = false
                        }
                        for v in 0..<7 {
                            r[v] = max(r[v], e[v])
                            while Double(pix[v]) * pa <= r[v] { if pix[v] > base { hits[v].append(d) }; pix[v] += 1 }
                        }
                        d += 1
                    }
                    if !any { continue }
                    for v in 0..<6 {
                        let o = v * 7, hc = hits[1 + v], ht = hits[0]
                        a7[o] += 1
                        a7[o + 1] += abs(r[1 + v] - r[0]) / pa
                        a7[o + 2] += (r[1 + v] - r[0]) / pa
                        let nb = min(ht.count, hc.count)
                        for i in 0..<nb {
                            let err = abs(hc[i] - ht[i]) / ht[i]
                            a7[o + 3] += err
                            if err > 0.05 { a7[o + 4] += 1 }
                        }
                        a7[o + 5] += Double(nb)
                        a7[o + 6] += Double(abs(ht.count - hc.count))
                    }
                }
                (accb + cam).pointee = a7
            }
        }
        for v in 0..<6 {
            let cand = m.tops[v][l]
            var rays = 0.0, silAbs = 0.0, silSigned = 0.0, depthErr = 0.0, bad = 0.0, both = 0.0, mismatch = 0.0
            for c in acc {
                rays += c[v * 7]; silAbs += c[v * 7 + 1]; silSigned += c[v * 7 + 2]; depthErr += c[v * 7 + 3]
                bad += c[v * 7 + 4]; both += c[v * 7 + 5]; mismatch += c[v * 7 + 6]
            }
            // Heights over the level's cells against the truth's mean over each cell (where both have data).
            var hAbs = 0.0, hSigned = 0.0, hn = 0.0, canopy = 0.0
            let f = 1 << (l - 1)
            for z in 0..<sideL {
                for x in 0..<sideL {
                    let c = cand[z * sideL + x]
                    if c <= 0 { continue }
                    var s = 0.0, k = 0.0
                    for dz in 0..<f { for dx in 0..<f { let t = truth[(z * f + dz) * side1 + x * f + dx]; if t > 0 { s += Double(t); k += 1 } } }
                    if k == 0 { continue }
                    hAbs += abs(Double(c) - s / k); hSigned += Double(c) - s / k; hn += 1
                    let cell = m.cells[v >= 4 ? 1 : 0][l][z * sideL + x]
                    let mode = v == 0 ? -1 : (v <= 3 ? v - 1 : (v == 4 ? 2 : 1))
                    if mode >= 0 && cell.words(x: x, z: z, level: l, mode: mode).1 != 0 { canopy += 1 }
                }
            }
            let o = (l - 2) * 48 + v * 8
            out[o] = silAbs / max(rays, 1); out[o + 1] = silSigned / max(rays, 1)
            out[o + 2] = depthErr / max(both, 1); out[o + 3] = bad / max(both, 1); out[o + 4] = mismatch / max(rays, 1)
            out[o + 5] = hAbs / max(hn, 1); out[o + 6] = hSigned / max(hn, 1); out[o + 7] = canopy / max(hn, 1)
        }
    }
}

/// Debug: real columns against the generator's at the same places, for the level-`level` nodes cached in `farDir`
/// (<game dir>/metalmc/lod/far/<world>): generated columns sample the generator's height at each column's center, so
/// where real and generated terrain meet, their difference is the step the seam shows. out: [0] columns compared,
/// then for the voxel tops (the first far field) and for the cells: mean signed, mean |.|, 95th percentile |.| of
/// (real top - generated top) as drawn, canopy included ([1-3] voxels, [4-6] cells); [7-9] the cells' ground alone
/// against the generator's where both are dry; [10] boundary cells (real next to generated), [11-12] mean |step| there
/// with voxel tops and with cells, [13] mean |step| between the generated columns one and two cells further out (the
/// terrain's own).
@_cdecl("mmc_debug_far_seam")
public func mmc_debug_far_seam(_ dirC: UnsafePointer<CChar>, _ farDirC: UnsafePointer<CChar>, _ level: Int32, _ out: UnsafeMutablePointer<Double>) {
    let dir = String(cString: dirC), l = Int(level)
    guard l >= 2 && l <= 6 else { return }   // the levels FarEvalMaps holds
    let m: FarEvalMaps
    if let c = farEvalCache, c.0 == dir { m = c.1 } else { m = farEvalBuild(dir); farEvalCache = (dir, m) }
    let store = LodFarStore()
    store.cacheDir = URL(fileURLWithPath: String(cString: farDirC))
    let sideL = m.side1 >> (l - 1), s = 1 << l, n = lodNodeVoxels
    let cx0 = m.x0 >> l, cz0 = m.z0 >> l   // world cell of the maps' corner at this level
    var nodes: [LodNodeKey: LodFarColumns?] = [:]
    /// The generator's drawn top and dry ground at world cell (x, z) of this level, or nil if it isn't cached.
    func generated(_ x: Int, _ z: Int) -> (Float, Float?)? {
        let key = LodNodeKey(level: l, x: x >> 8, z: z >> 8)
        if nodes[key] == nil {
            store.lock.lock(); let c = store.load(key); store.lock.unlock()
            nodes[key] = .some(c)
        }
        guard let cols = nodes[key] ?? nil else { return nil }
        let col = (z & 255) * n + (x & 255)
        store.lock.lock(); let surface = store.surfaces[cols.biome[col]] ?? LodFarStore.defaultSurface; store.lock.unlock()
        var slope = 0
        for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
            let nx = (x & 255) + dx, nz = (z & 255) + dz
            if nx < 0 || nz < 0 || nx >= n || nz >= n { continue }
            slope = max(slope, abs(Int(cols.height[nz * n + nx]) - Int(cols.height[col])))
        }
        let c = LodFarStore.generatedCell(height: Int(cols.height[col]), sampled: l, s: surface, steep: slope * 2 > 3 << l, bx: x * s, bz: z * s, L: l)
        let (w0, w1) = c.words(x: x & 255, z: z & 255, level: l)
        return (drawnTop(w0, w1), c.wet < 128 ? Float(c.ground) / Float(LodFarCell.unit) : nil)
    }
    var dv: [Double] = [], dc: [Double] = [], dg: [Double] = []
    var bn = 0.0, stepV = 0.0, stepC = 0.0, stepG = 0.0
    let vox = m.tops[0][l], cells = m.tops[1][l]
    for z in 0..<sideL {
        for x in 0..<sideL {
            let i = z * sideL + x
            guard cells[i] > 0, let (gt, gg) = generated(cx0 + x, cz0 + z) else { continue }
            dv.append(Double(vox[i] - gt)); dc.append(Double(cells[i] - gt))
            let cell = m.cells[0][l][i]
            if let gg, cell.wet < 128 { dg.append(Double(cell.ground) / Double(LodFarCell.unit) - Double(gg)) }
            // Boundary: a real cell whose neighbor has no real data, but generated columns.
            for (ddx, ddz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                let nx = x + ddx, nz = z + ddz
                if nx >= 0 && nz >= 0 && nx < sideL && nz < sideL && cells[nz * sideL + nx] > 0 { continue }
                guard let (g1, _) = generated(cx0 + nx, cz0 + nz), let (g2, _) = generated(cx0 + nx + ddx, cz0 + nz + ddz) else { continue }
                bn += 1
                stepV += Double(abs(vox[i] - g1)); stepC += Double(abs(cells[i] - g1)); stepG += Double(abs(g1 - g2))
            }
        }
    }
    func stats(_ a: [Double]) -> (Double, Double, Double) {
        if a.isEmpty { return (0, 0, 0) }
        let absd = a.map { abs($0) }.sorted()
        return (a.reduce(0, +) / Double(a.count), absd.reduce(0, +) / Double(a.count), absd[min(absd.count - 1, Int(Double(absd.count) * 0.95))])
    }
    out[0] = Double(dc.count)
    (out[1], out[2], out[3]) = stats(dv)
    (out[4], out[5], out[6]) = stats(dc)
    (out[7], out[8], out[9]) = stats(dg)
    out[10] = bn; out[11] = stepV / max(bn, 1); out[12] = stepC / max(bn, 1); out[13] = stepG / max(bn, 1)
}

/// Debug: renders the far field of the open world (mmc_lod_open3) from a camera to a PNG, as the game would draw it past
/// the quads but with every column drawn by the march (no quad area; rays start `shell` blocks out), without texture
/// detail, lightmap or fog. Yaw and pitch as Minecraft's (yaw 0 looks toward +z, 90 toward -x; pitch down positive).
/// Returns 1 once drawn, 0 while the far field isn't ready (call again), -1 on failure.
@_cdecl("mmc_debug_far_render")
public func mmc_debug_far_render(_ camX: Double, _ camY: Double, _ camZ: Double, _ yawDeg: Float, _ pitchDeg: Float, _ fovYDeg: Float,
                                 _ width: Int32, _ height: Int32, _ shell: Float, _ pathC: UnsafePointer<CChar>) -> Int32 {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return -1 }
    let snap = w.snapshot()
    let W = Int(width), H = Int(height)
    ctx.passColorFormats = [.rgba8Unorm]
    ctx.passDepthFormat = .depth32Float
    ctx.passWidth = W
    ctx.passHeight = H
    // Every ring filled for this camera (one per call), and the fills done before drawing.
    var ready = false
    for _ in 0..<12 {
        ready = FarField.shared.prepare(meshes: snap.meshes, generation: snap.generation, chosen: [], maxLevel: w.maxLevel, cx: camX, cz: camZ)
    }
    guard ready else { return 0 }
    r.ensureColors()
    guard let colors = r.colorBuffer else { return -1 }
    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: W, height: H, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]
    td.storageMode = .shared
    let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: W, height: H, mipmapped: false)
    dd.usage = [.renderTarget]
    dd.storageMode = .private
    let ld = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 16, height: 16, mipmapped: false)
    guard let color = ctx.device.makeTexture(descriptor: td), let depth = ctx.device.makeTexture(descriptor: dd),
          let lightmap = ctx.device.makeTexture(descriptor: ld) else { return -1 }
    // Camera: Minecraft's view direction, a view rotation looking down -z, and a reverse-Z projection like the game's.
    let yaw = yawDeg * .pi / 180, pitch = pitchDeg * .pi / 180
    let fwd = SIMD3<Float>(-sin(yaw) * cos(pitch), -sin(pitch), cos(yaw) * cos(pitch))
    let right = simd_normalize(simd_cross(fwd, SIMD3(0, 1, 0))), up = simd_cross(right, fwd)
    let view = simd_float4x4(rows: [SIMD4(right, 0), SIMD4(up, 0), SIMD4(-fwd, 0), SIMD4(0, 0, 0, 1)])
    let f = 1 / tan(fovYDeg * .pi / 360), aspect = Float(W) / Float(H), near: Float = 0.05, far: Float = 300_000
    let proj = simd_float4x4(rows: [SIMD4(f / aspect, 0, 0, 0), SIMD4(0, f, 0, 0),
                                    SIMD4(0, 0, near / (far - near), near * far / (far - near)), SIMD4(0, 0, -1, 0)])
    var u = LodUniforms(proj: proj, view: view, fogColor: SIMD4(0.62, 0.75, 0.95, 0), envStart: 1e9, envEnd: 2e9, rdStart: 1e9, rdEnd: 2e9,
                        discardRadius: 0, sky: 1)
    u.seamInfo = SIMD4(0, 0, 15, 0)
    let rp = MTLRenderPassDescriptor()
    rp.colorAttachments[0].texture = color
    rp.colorAttachments[0].loadAction = .clear
    rp.colorAttachments[0].clearColor = MTLClearColor(red: 0.62, green: 0.75, blue: 0.95, alpha: 1)
    rp.colorAttachments[0].storeAction = .store
    rp.depthAttachment.texture = depth
    rp.depthAttachment.loadAction = .clear
    rp.depthAttachment.clearDepth = 0
    rp.depthAttachment.storeAction = .dontCare
    let sd = MTLSamplerDescriptor()
    guard let sampler = ctx.device.makeSamplerState(descriptor: sd) else { return -1 }
    // FARTEST_REPEAT=n: draw the march n times and log the median GPU time of a draw (the first is a warm-up).
    let repeats = max(1, Int(ProcessInfo.processInfo.environment["FARTEST_REPEAT"] ?? "") ?? 1)
    // With a shader whose FF_STATS is set (mmc_debug_far_set_shader): its per-pixel counters, two uint4 per pixel.
    if farDebugStats {
        guard let b = ctx.device.makeBuffer(length: W * H * 32, options: [.storageModeShared]) else { return -1 }
        memset(b.contents(), 0, W * H * 32)
        FarField.shared.statsBuffer = b
    }
    defer { FarField.shared.statsBuffer = nil }
    // FARTEST_NEAR=<blocks>: the game's nearer geometry stood in for. In the game the far field's shell (at the nearest far
    // tile) is drawn after the quads, and the early depth test skips its pixels wherever they're nearer; with nothing in
    // front, rays toward terrain inside the shell start under it and "hit" at once (7 steps), half the screen of work the
    // game doesn't do. A first draw marches from <blocks> out to fill the depth (not timed); the timed draws, from the
    // shell, test against it, so only pixels whose terrain lies past the shell run the march, as in the game.
    let nearShell = Float(ProcessInfo.processInfo.environment["FARTEST_NEAR"] ?? "") ?? 0
    if nearShell > 0 {
        rp.depthAttachment.storeAction = .store
        guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { return -1 }
        enc.setFragmentBuffer(colors, offset: 0, index: 20)
        enc.setFragmentTexture(lightmap, index: 30)
        enc.setFragmentSamplerState(sampler, index: 15)
        FarField.shared.draw(enc, u: u, colors: colors, lightmap: lightmap, lightSampler: sampler, cy: camY, nearest: Double(nearShell), builtin: true)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if cb.status != .completed { return -1 }
        rp.depthAttachment.loadAction = .load
        if let b = FarField.shared.statsBuffer { memset(b.contents(), 0, W * H * 32) }
    }
    var times: [Double] = [], cpu: [Double] = [], preTimes: [Double] = []
    for _ in 0..<repeats {
        guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { return -1 }
        // Texture detail is off (camFrac.w 0), but its slots get something bound, as the LOD's pass does in the game.
        enc.setFragmentBuffer(colors, offset: 0, index: 20)
        enc.setFragmentTexture(lightmap, index: 30)
        enc.setFragmentSamplerState(sampler, index: 15)
        let c0 = DispatchTime.now().uptimeNanoseconds
        FarField.shared.draw(enc, u: u, colors: colors, lightmap: lightmap, lightSampler: sampler, cy: camY, nearest: Double(shell))
        cpu.append(Double(DispatchTime.now().uptimeNanoseconds - c0) / 1e3)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        if cb.status != .completed { return -1 }
        var ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        // The horizon profile's pass runs in a command buffer of its own, before the draw's.
        if let bcb = FarField.shared.lastProfileCB {
            bcb.waitUntilCompleted()
            preTimes.append((bcb.gpuEndTime - bcb.gpuStartTime) * 1000)
            ms += preTimes.last!
        }
        times.append(ms)
    }
    if repeats > 1 {
        let t = times.dropFirst().sorted()
        let c = cpu.dropFirst().sorted(), b = preTimes.dropFirst().sorted()
        log(String(format: "far field render: march %.3f ms GPU (median of %d, min %.3f)%@; draw's CPU %.0f us", t[t.count / 2], t.count,
                   t.first ?? 0, b.isEmpty ? "" : String(format: ", of which the profile %.3f", b[b.count / 2]), c[c.count / 2]))
        farDebugLastMs = t[t.count / 2]
    }
    // Rows come out bottom-up (the march's shell flips y like the game's GL-style targets).
    var px = [UInt8](repeating: 0, count: W * H * 4)
    color.getBytes(&px, bytesPerRow: W * 4, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0)
    if farFieldSteps {
        // Mean march steps over the pixels that hit (the debug view's red is steps / 128, saturating), and the hit share.
        var sum = 0.0, hits = 0
        for i in 0..<(W * H) where !(px[4 * i] == 0 && px[4 * i + 1] == 0) { sum += Double(px[4 * i]) / 255 * 128; hits += 1 }
        log("far field render: \(hits) of \(W * H) pixels hit, mean steps \(String(format: "%.1f", sum / Double(max(hits, 1))))")
        // What a SIMD group pays: its 32 pixels (taken as 8 x 4 blocks) step together, so each block costs its slowest
        // pixel. Misses count as the steps they took too (their red is 0 here, so they count as 0: a lower bound).
        var blockSum = 0.0, blocks = 0, hist = [Int](repeating: 0, count: 9)
        for by in stride(from: 0, to: H - 3, by: 4) {
            for bx in stride(from: 0, to: W - 7, by: 8) {
                var mx = 0.0, any = false
                for y in by..<(by + 4) {
                    for x in bx..<(bx + 8) {
                        let i = y * W + x
                        if px[4 * i] == 0 && px[4 * i + 1] == 0 { continue }
                        any = true
                        mx = max(mx, Double(px[4 * i]) / 255 * 128)
                    }
                }
                if any { blockSum += mx; blocks += 1; hist[min(8, Int(mx / 16))] += 1 }
            }
        }
        log("far field render: \(blocks) 8x4 blocks with hits, mean of their slowest pixel's steps \(String(format: "%.1f", blockSum / Double(max(blocks, 1)))), by 16s \(hist)")
    }
    if let b = FarField.shared.statsBuffer { farDebugAnalyze(b.contents().bindMemory(to: UInt32.self, capacity: W * H * 8), W, H) }
    var flipped = [UInt8](repeating: 0, count: px.count)
    for y in 0..<H { flipped.replaceSubrange((y * W * 4)..<((y + 1) * W * 4), with: px[((H - 1 - y) * W * 4)..<((H - y) * W * 4)]) }
    guard let provider = CGDataProvider(data: Data(flipped) as CFData),
          let image = CGImage(width: W, height: H, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: W * 4, space: CGColorSpaceCreateDeviceRGB(),
                              bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
                              decode: nil, shouldInterpolate: false, intent: .defaultIntent),
          let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: String(cString: pathC)) as CFURL, "public.png" as CFString, 1, nil)
    else { return -1 }
    CGImageDestinationAddImage(dest, image, nil)
    return CGImageDestinationFinalize(dest) ? 1 : -1
}

/// Debug: the far-field words of the open world's level-`level` nodes at world cells (cx0 + i, cz0 + j), i < w, j < h
/// (two per cell, row by row; 0 where no node has columns). Returns the cells found.
@_cdecl("mmc_debug_far_cells")
public func mmc_debug_far_cells(_ level: Int32, _ cx0: Int32, _ cz0: Int32, _ w: Int32, _ h: Int32, _ out: UnsafeMutablePointer<UInt32>) -> Int32 {
    let r = LodRenderer.shared
    r.lock.lock(); let world = r.world; r.lock.unlock()
    guard let world else { return 0 }
    let snap = world.snapshot()
    var found: Int32 = 0
    for j in 0..<Int(h) {
        for i in 0..<Int(w) {
            let cx = Int(cx0) + i, cz = Int(cz0) + j
            let key = LodNodeKey(level: Int(level), x: cx >> 8, z: cz >> 8)
            let o = 2 * (j * Int(w) + i)
            out[o] = 0; out[o + 1] = 0
            guard let node = snap.meshes[key], let cols = node.columns else { continue }
            let p = cols.contents().bindMemory(to: UInt32.self, capacity: 2 * 65536)
            let c = (cz & 255) * 256 + (cx & 255)
            out[o] = p[2 * c]; out[o + 1] = p[2 * c + 1]
            found += 1
        }
    }
    return found
}

/// Offline: whether the draw's shader keeps per-pixel march counters (mmc_debug_far_set_shader with stats).
private var farDebugStats = false
/// Offline: the last render's median GPU time of the march (FARTEST_REPEAT > 1), ms.
private var farDebugLastMs = 0.0

@_cdecl("mmc_debug_far_last_ms")
public func mmc_debug_far_last_ms() -> Double { farDebugLastMs }

/// Debug: the far field's draw shaders from the source in `path` (a variant of `fartest.py shader`'s output; empty: the
/// built-in source), with FF_STATS set if `stats` (mmc_debug_far_render then logs the march's counters). 1 if it compiled.
@_cdecl("mmc_debug_far_set_shader")
public func mmc_debug_far_set_shader(_ pathC: UnsafePointer<CChar>, _ stats: Int32) -> Int32 {
    let path = String(cString: pathC)
    var src = farFieldShaderSource
    if !path.isEmpty {
        guard let s = try? String(contentsOfFile: path, encoding: .utf8) else { return 0 }
        src = s
    }
    if stats != 0 { src = src.replacingOccurrences(of: "#define FF_STATS 0", with: "#define FF_STATS 1") }
    let ok = FarField.shared.setDrawSource(path.isEmpty && stats == 0 ? nil : src)
    farDebugStats = ok && stats != 0
    return ok ? 1 : 0
}

/// The march's counters (FF_STATS: two uint4 per pixel, see ff_fs) over the image: where the steps go, and what SIMD
/// groups (taken as 8 x 4 pixel blocks of the target) pay. With the rings marched one after the other a group pays each
/// ring's slowest pixel ("now"); one loop over all rings would pay the slowest pixel's total ("one loop").
private func farDebugAnalyze(_ s: UnsafePointer<UInt32>, _ W: Int, _ H: Int) {
    @inline(__always) func ringSteps(_ a: UnsafePointer<UInt32>, _ r: Int) -> Int { Int((a[4 + r / 4] >> UInt32(8 * (r % 4))) & 255) }
    var marched = 0, hits = 0, capped = 0
    var steps = 0.0, hitSteps = 0.0, adv = 0.0, desc = 0.0, dip = 0.0, colMiss = 0.0, wasted = 0.0, rings = 0.0
    var perRing = [Double](repeating: 0, count: 8), hitRing = [Int](repeating: 0, count: 9)
    for i in 0..<(W * H) {
        let a = s + 8 * i
        if a[3] >> 31 == 0 { continue }
        marched += 1
        let st = Double(a[0] & 0xFFFF)
        steps += st
        rings += Double(a[0] >> 16)
        adv += Double(a[1] & 0xFFFF); desc += Double(a[1] >> 16)
        dip += Double(a[2] & 0xFFFF); colMiss += Double(a[2] >> 16)
        wasted += Double((a[3] >> 8) & 0xFFF)
        let hr = Int(a[3] & 0xFF)
        if hr != 15 { hits += 1; hitSteps += st; hitRing[min(hr, 7)] += 1 } else { hitRing[8] += 1 }
        var cap = false
        for r in 0..<8 { let n = ringSteps(a, r); perRing[r] += Double(n); if n >= 192 { cap = true } }
        if cap { capped += 1 }
    }
    let m = Double(max(marched, 1))
    func f1(_ x: Double) -> String { String(format: "%.1f", x) }
    log("far field stats: \(marched) pixels marched, \(hits) hit, \(capped) at a ring's step cap; per marched pixel: steps \(f1(steps / m))"
        + " (hits \(f1(hitSteps / Double(max(hits, 1))))), advances over level 0 \(f1(adv / m)), descents \(f1(desc / m)),"
        + " level-0 tests \(f1(dip / m)) (missed \(f1(colMiss / m))), climbs undone \(f1(wasted / m)), rings \(String(format: "%.2f", rings / m))")
    log("far field stats: steps per ring (per marched pixel) \(perRing.map { f1($0 / m) }), hits by ring \(Array(hitRing.dropLast())), misses \(hitRing[8])")
    var groups = 0, flat = 0.0, nested = 0.0, mean = 0.0, hitKeys = 0.0
    var hist = [Int](repeating: 0, count: 10)
    let bands = 8
    var bandNested = [Double](repeating: 0, count: bands), bandFlat = [Double](repeating: 0, count: bands), bandGroups = [Int](repeating: 0, count: bands)
    var keys = [Int](repeating: 0, count: 32)
    for by in stride(from: 0, to: H - 3, by: 4) {
        for bx in stride(from: 0, to: W - 7, by: 8) {
            var mx = 0, sum = 0, n = 0, nk = 0
            var rmx = [Int](repeating: 0, count: 8)
            for y in by..<(by + 4) {
                for x in bx..<(bx + 8) {
                    let a = s + 8 * (y * W + x)
                    if a[3] >> 31 == 0 { continue }
                    n += 1
                    let st = Int(a[0] & 0xFFFF)
                    mx = max(mx, st); sum += st
                    for r in 0..<8 { rmx[r] = max(rmx[r], ringSteps(a, r)) }
                    // Distinct steps (ring, step in it) at which a lane hits: the hit's column test runs once per such step.
                    let hr = Int(a[3] & 0xFF)
                    if hr != 15 {
                        let k = hr << 16 | Int((a[3] >> 20) & 0x7FF)
                        if !keys[0..<nk].contains(k) { keys[nk] = k; nk += 1 }
                    }
                }
            }
            if n == 0 { continue }
            let nst = rmx.reduce(0, +)
            groups += 1
            flat += Double(mx); nested += Double(nst); mean += Double(sum) / Double(n); hitKeys += Double(nk)
            hist[min(9, nst / 32)] += 1
            let band = (H - 1 - by) * bands / H   // image rows from the top (the target's rows are bottom-up)
            bandNested[band] += Double(nst); bandFlat[band] += Double(mx); bandGroups[band] += 1
        }
    }
    let g = Double(max(groups, 1))
    log("far field stats: \(groups) 8x4 groups marched; per group: mean pixel \(f1(mean / g)), slowest pixel (one loop) \(f1(flat / g)),"
        + " rings one after another (now) \(f1(nested / g)), distinct hit steps \(f1(hitKeys / g)); now by 32s \(hist)")
    let total = max(bandNested.reduce(0, +), 1)
    log("far field stats: by eighths of the image from the top: share of the groups' steps \(bandNested.map { String(format: "%.0f%%", 100 * $0 / total) }),"
        + " per group now \((0..<bands).map { f1(bandNested[$0] / Double(max(bandGroups[$0], 1))) }), one loop \((0..<bands).map { f1(bandFlat[$0] / Double(max(bandGroups[$0], 1))) })")
}

/// Debug: two PNGs of the same size: out[0] pixels that differ at all, [1] by more than 2 levels in a channel, [2] by more
/// than 16, [3] the largest difference, [4-7] the differing pixels' bounding box (x0, y0, x1, y1, rows from the top).
/// Writes `outPath` (unless empty): the first image dimmed, differing pixels marked 5 x 5 (red where the second is
/// brighter, blue where darker) so single pixels show. Returns 1, or 0 if they can't be read or differ in size.
@_cdecl("mmc_debug_far_compare")
public func mmc_debug_far_compare(_ aC: UnsafePointer<CChar>, _ bC: UnsafePointer<CChar>, _ outC: UnsafePointer<CChar>,
                                  _ out: UnsafeMutablePointer<Double>) -> Int32 {
    func load(_ path: String) -> (Int, Int, [UInt8])? {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        let w = img.width, h = img.height
        var px = [UInt8](repeating: 0, count: w * h * 4)
        guard let c = CGContext(data: &px, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
        c.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        return (w, h, px)
    }
    guard let (w, h, a) = load(String(cString: aC)), let (w2, h2, b) = load(String(cString: bC)), w == w2, h == h2 else { return 0 }
    var any = 0, over2 = 0, over16 = 0, largest = 0, x0 = w, y0 = h, x1 = -1, y1 = -1
    var mark = [Int8](repeating: 0, count: w * h)
    for y in 0..<h {
        for x in 0..<w {
            let o = 4 * (y * w + x)
            var dm = 0, sa = 0, sb = 0
            for c in 0..<3 { dm = max(dm, abs(Int(a[o + c]) - Int(b[o + c]))); sa += Int(a[o + c]); sb += Int(b[o + c]) }
            if dm == 0 { continue }
            any += 1
            if dm > 2 { over2 += 1 }
            if dm > 16 { over16 += 1 }
            largest = max(largest, dm)
            x0 = min(x0, x); y0 = min(y0, y); x1 = max(x1, x); y1 = max(y1, y)
            mark[y * w + x] = sb >= sa ? 1 : -1
        }
    }
    out[0] = Double(any); out[1] = Double(over2); out[2] = Double(over16); out[3] = Double(largest)
    out[4] = Double(x0); out[5] = Double(y0); out[6] = Double(x1); out[7] = Double(y1)
    let outPath = String(cString: outC)
    if !outPath.isEmpty {
        var img = [UInt8](repeating: 255, count: w * h * 4)
        for i in 0..<(w * h) {
            let v = UInt8((Int(a[4 * i]) + Int(a[4 * i + 1]) + Int(a[4 * i + 2])) / 9)
            img[4 * i] = v; img[4 * i + 1] = v; img[4 * i + 2] = v
        }
        for y in 0..<h {
            for x in 0..<w where mark[y * w + x] != 0 {
                for yy in max(0, y - 2)...min(h - 1, y + 2) {
                    for xx in max(0, x - 2)...min(w - 1, x + 2) {
                        let o = 4 * (yy * w + xx)
                        img[o] = mark[y * w + x] > 0 ? 255 : 0; img[o + 1] = 0; img[o + 2] = mark[y * w + x] > 0 ? 0 : 255
                    }
                }
            }
        }
        guard let provider = CGDataProvider(data: Data(img) as CFData),
              let image = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: w * 4, space: CGColorSpaceCreateDeviceRGB(),
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
                                  decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: outPath) as CFURL, "public.png" as CFString, 1, nil)
        else { return 0 }
        CGImageDestinationAddImage(dest, image, nil)
        _ = CGImageDestinationFinalize(dest)
    }
    return 1
}
