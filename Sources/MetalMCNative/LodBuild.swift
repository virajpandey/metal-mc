import Foundation
import MetalMCCore

// LOD construction from a world save (clean-room design; Voxy's public descriptions were a reference).
//
// Levels: a level-L voxel covers 2^L x 2^L x 2^L blocks (L >= 1). A node is a square column of
// 256 x 256 voxels by the full world height, so a level-1 node is exactly one region file (512 blocks),
// and a level-(L+1) node merges 2 x 2 level-L nodes at half resolution.
//
// Downsampling keeps the top layer: in each 2 x 2 x 2 group, the highest non-air child wins (ties go to
// the last one visited), so surfaces keep their material from far away.

let lodNodeVoxels = 256
let lodTileVoxels = 64
let lodTilesPerSide = lodNodeVoxels / lodTileVoxels   // 4 x 4 tiles per node
let lodWorldMinY = -64
let lodWorldHeight = 384

/// A node's voxel grid: x fastest, then z, then y. Material ids from MetalMCCore.Mat.
struct LodGrid {
    let level: Int
    let height: Int            // voxels
    var v: [UInt8]

    init(level: Int) {
        self.level = level
        height = lodWorldHeight >> level
        v = [UInt8](repeating: 0, count: lodNodeVoxels * lodNodeVoxels * height)
    }

    @inline(__always) func index(_ x: Int, _ y: Int, _ z: Int) -> Int { (y * lodNodeVoxels + z) * lodNodeVoxels + x }

    /// Fills air (and water) that can't be reached from the sky or the node's sides with stone. Sealed
    /// caves can't be seen from LOD distances, and their walls would otherwise dominate the mesh.
    /// Cave entrances stay open because they connect to reachable air.
    mutating func fillUnreachable() {
        let n = lodNodeVoxels, h = height
        var seen = [Bool](repeating: false, count: v.count)
        var stack: [Int32] = []
        stack.reserveCapacity(1 << 16)
        v.withUnsafeMutableBufferPointer { g in
            seen.withUnsafeMutableBufferPointer { s in
                @inline(__always) func passable(_ i: Int) -> Bool { let m = g[i]; return m == 0 || lodIsWater(m) }
                @inline(__always) func push(_ i: Int) {
                    if !s[i] && passable(i) { s[i] = true; stack.append(Int32(i)) }
                }
                // Seeds: the top layer and the four side walls.
                for z in 0..<n { for x in 0..<n { push(((h - 1) * n + z) * n + x) } }
                for y in 0..<h {
                    for k in 0..<n {
                        push((y * n + 0) * n + k); push((y * n + (n - 1)) * n + k)
                        push((y * n + k) * n + 0); push((y * n + k) * n + (n - 1))
                    }
                }
                while let i32 = stack.popLast() {
                    let i = Int(i32)
                    let x = i % n, z = (i / n) % n, y = i / (n * n)
                    if x > 0 { push(i - 1) }
                    if x < n - 1 { push(i + 1) }
                    if z > 0 { push(i - n) }
                    if z < n - 1 { push(i + n) }
                    if y > 0 { push(i - n * n) }
                    if y < h - 1 { push(i + n * n) }
                }
                let stone = Mat.stone.rawValue
                for i in 0..<g.count where !s[i] && passable(i) { g[i] = stone }
            }
        }
    }

    /// Half-resolution copy of this grid, written into one quadrant (qx, qz in 0...1) of `parent`.
    func downsample(into parent: inout LodGrid, qx: Int, qz: Int) {
        let half = lodNodeVoxels / 2
        let ph = parent.height
        v.withUnsafeBufferPointer { src in
            parent.v.withUnsafeMutableBufferPointer { dst in
                for y in 0..<height {
                    let py = y >> 1
                    if py >= ph { break }
                    for z in 0..<lodNodeVoxels {
                        let pz = qz * half + (z >> 1)
                        let srow = (y * lodNodeVoxels + z) * lodNodeVoxels
                        let drow = (py * lodNodeVoxels + pz) * lodNodeVoxels + qx * half
                        for x in 0..<lodNodeVoxels {
                            let m = src[srow + x]
                            if m != 0 { dst[drow + (x >> 1)] = m }   // ascending y: the top layer wins
                        }
                    }
                }
            }
        }
    }
}

/// A meshed node ready to draw.
struct LodNode {
    let level: Int
    let x0: Int, z0: Int        // world block coordinates of the node's corner
    var quads: [UInt32]         // pairs: word0 = x | z<<8 | y<<16 | face<<24, word1 = mat | (w-1)<<8 | (h-1)<<16
    var counts: [Int]           // quads per (tile, face), tile-major: tile = tz * 4 + tx, face +X -X +Y -Y +Z -Z
    var tileY: [Int]            // per tile: min and max voxel y of its quads (min > max if empty)
    var size: Int { lodNodeVoxels << level }
}

enum LodBuild {
    /// Builds a level-1 grid from one region file (r.X.Z.mca), or nil if it has no fully generated chunks.
    static func regionGrid(path: String) -> LodGrid? {
        guard let data = FileManager.default.contents(atPath: path), data.count >= 8192 else { return nil }
        let r = [UInt8](data)
        var grid = LodGrid(level: 1)
        var any = false
        var cache: [String: UInt8] = [:]
        grid.v.withUnsafeMutableBufferPointer { g in
            for i in 0..<1024 {
                guard Anvil.be32(r, i * 4) != 0, let chunk = try? ChunkScan.decodeChunk(region: r, index: i, cache: &cache),
                      !chunk.sections.isEmpty else { continue }
                any = true
                let lx0 = ((chunk.cx & 31) * 16) >> 1, lz0 = ((chunk.cz & 31) * 16) >> 1
                // Biome tint class per 4 x 4-block cell (index z * 4 + x).
                var tint = [UInt8](repeating: 0, count: 16)
                if chunk.surfaceBiomes.count == 16 { for i in 0..<16 { tint[i] = lodTintIndex(chunk.surfaceBiomes[i]) } }
                for (sy, blocks) in chunk.sections.sorted(by: { $0.sy < $1.sy }) {
                    blocks.withUnsafeBufferPointer { b in
                        for by in 0..<16 {
                            let vy = (sy * 16 + by) >> 1
                            for bz in 0..<16 {
                                let row = (vy * lodNodeVoxels + lz0 + (bz >> 1)) * lodNodeVoxels + lx0
                                let brow = (by << 8) | (bz << 4)
                                let trow = (bz >> 2) * 4
                                for bx in 0..<16 {
                                    let m = b[brow | bx]
                                    if m != 0 { g[row + (bx >> 1)] = lodTinted(m, tint[trow + (bx >> 2)]) }
                                }
                            }
                        }
                    }
                }
            }
        }
        return any ? grid : nil
    }

    /// Greedy mesh of a node grid. Faces toward air are emitted; water only shows faces toward air.
    /// Outside the node counts as air on the sides (skirt walls that hide cracks between levels) and
    /// as solid below the world. Merged quads are capped at `maxMerge` voxels per side, so each quad
    /// stays small enough to be dropped individually where vanilla chunks are drawn.
    static func mesh(_ grid: LodGrid, maxMerge: Int = 16) -> (quads: [UInt32], counts: [Int], tileY: [Int]) {
        let n = lodNodeVoxels, h = grid.height
        let kinds = lodKinds
        let airK = MaterialKind.air.rawValue, waterK = MaterialKind.water.rawValue
        let tiles = lodTilesPerSide * lodTilesPerSide
        var buckets = [[UInt32]](repeating: [], count: tiles * 6)
        var tileY = [Int](repeating: 0, count: tiles * 2)
        for t in 0..<tiles { tileY[2 * t] = Int.max; tileY[2 * t + 1] = Int.min }
        var mask = [UInt8](repeating: 0, count: n * max(n, h))

        grid.v.withUnsafeBufferPointer { g in
            @inline(__always) func at(_ x: Int, _ y: Int, _ z: Int) -> UInt8 {
                if y < 0 { return Mat.stone.rawValue }
                if x < 0 || z < 0 || x >= n || z >= n || y >= h { return 0 }
                return g[(y * n + z) * n + x]
            }
            // Face order +X -X +Y -Y +Z -Z. For each face: normal axis, and the (u, v) axes whose extents
            // the shader scales by (w, h): X faces u = y, v = z; Y faces u = x, v = z; Z faces u = x, v = y.
            for face in 0..<6 {
                let axis = face / 2
                let dir = face % 2 == 0 ? 1 : -1
                // (d, u, v) -> (x, y, z): X faces u = y, v = z; Y faces u = x, v = z; Z faces u = x, v = y.
                let dn = axis == 1 ? h : n
                let du = axis == 0 ? h : n
                let dv = axis == 2 ? h : n
                @inline(__always) func xyz(_ d: Int, _ u: Int, _ v: Int) -> (Int, Int, Int) {
                    switch axis {
                    case 0: return (d, u, v)
                    case 1: return (u, d, v)
                    default: return (u, v, d)
                    }
                }
                let (sx, sy, sz) = axis == 0 ? (dir, 0, 0) : (axis == 1 ? (0, dir, 0) : (0, 0, dir))
                for d in 0..<dn {
                    var anyFace = false
                    for vv in 0..<dv {
                        for uu in 0..<du {
                            let (x, y, z) = xyz(d, uu, vv)
                            let m = at(x, y, z)
                            var f: UInt8 = 0
                            if m != 0 {
                                let nk = kinds[Int(at(x + sx, y + sy, z + sz))]
                                let k = kinds[Int(m)]
                                if k == waterK { if nk == airK { f = m } }
                                else if nk == airK { f = m }   // LOD water is opaque, so faces under water are hidden
                            }
                            mask[vv * du + uu] = f
                            if f != 0 { anyFace = true }
                        }
                    }
                    if !anyFace { continue }
                    // Greedy rectangles over the (u, v) mask.
                    for vv in 0..<dv {
                        var uu = 0
                        while uu < du {
                            let m = mask[vv * du + uu]
                            if m == 0 { uu += 1; continue }
                            // Merges stay inside a tile along x and z (u is y for X faces; v is y for Z faces).
                            let uLimit = axis == 0 ? du : min(du, (uu / lodTileVoxels + 1) * lodTileVoxels)
                            let vLimit = axis == 2 ? dv : min(dv, (vv / lodTileVoxels + 1) * lodTileVoxels)
                            var w = 1
                            while w < maxMerge && uu + w < uLimit && mask[vv * du + uu + w] == m { w += 1 }
                            var ht = 1
                            grow: while ht < maxMerge && vv + ht < vLimit {
                                for k in 0..<w where mask[(vv + ht) * du + uu + k] != m { break grow }
                                ht += 1
                            }
                            for a in 0..<ht { for k in 0..<w { mask[(vv + a) * du + uu + k] = 0 } }
                            let (x, y, z) = xyz(d, uu, vv)
                            let t = (z / lodTileVoxels) * lodTilesPerSide + x / lodTileVoxels
                            buckets[t * 6 + face].append(UInt32(x) | UInt32(z) << 8 | UInt32(y) << 16 | UInt32(face) << 24)
                            buckets[t * 6 + face].append(UInt32(m) | UInt32(w - 1) << 8 | UInt32(ht - 1) << 16)
                            // Vertical extent of this quad in voxels (X and Z faces extend along y by w or h).
                            let top = y + (axis == 0 ? w : (axis == 2 ? ht : 1))
                            tileY[2 * t] = min(tileY[2 * t], y)
                            tileY[2 * t + 1] = max(tileY[2 * t + 1], top)
                            uu += w
                        }
                    }
                }
            }
        }
        var out: [UInt32] = []
        out.reserveCapacity(buckets.reduce(0) { $0 + $1.count })
        var counts = [Int](repeating: 0, count: tiles * 6)
        for (i, b) in buckets.enumerated() {
            out.append(contentsOf: b)
            counts[i] = b.count / 2
        }
        return (out, counts, tileY)
    }

    /// Builds every level from the save's overworld region files. Level-1 nodes are only meshed within
    /// `fineRadius` blocks of (cx, cz), since they're drawn only near the camera; coarser levels cover
    /// the whole save. Progress goes to `log`.
    static func build(worldDir: String, maxLevel: Int, centerX: Int, centerZ: Int, fineRadius: Int) -> [LodNode] {
        guard let dir = Anvil.regionDirectory(URL(fileURLWithPath: worldDir)) else {
            log("LOD: no region directory under \(worldDir)")
            return []
        }
        let files = ((try? FileManager.default.contentsOfDirectory(atPath: dir.path)) ?? []).filter { $0.hasSuffix(".mca") }
        var coords: [(Int, Int, String)] = []
        for f in files {
            let parts = f.split(separator: ".")
            if parts.count == 4, let x = Int(parts[1]), let z = Int(parts[2]) { coords.append((x, z, dir.appendingPathComponent(f).path)) }
        }
        let t0 = Date()
        // Level 1: one grid per region, in parallel.
        var level: [Int64: LodGrid] = [:]
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: coords.count) { i in
            let (x, z, path) = coords[i]
            if let g = regionGrid(path: path) {
                lock.lock(); level[key(x, z)] = g; lock.unlock()
            }
        }
        log("LOD: \(level.count) regions voxelized in \(String(format: "%.1f", Date().timeIntervalSince(t0))) s")

        var nodes: [LodNode] = []
        var lvl = 1
        while !level.isEmpty {
            let size = lodNodeVoxels << lvl
            let entries = Array(level)
            var meshed = [LodNode?](repeating: nil, count: entries.count)
            meshed.withUnsafeMutableBufferPointer { out in
                let outp = out.baseAddress!
                DispatchQueue.concurrentPerform(iterations: entries.count) { i in
                    let (k, g) = entries[i]
                    let (nx, nz) = unkey(k)
                    let x0 = nx * size, z0 = nz * size
                    if lvl == 1 {
                        let dx = Double(max(x0 - centerX, 0, centerX - (x0 + size)))
                        let dz = Double(max(z0 - centerZ, 0, centerZ - (z0 + size)))
                        if (dx * dx + dz * dz).squareRoot() > Double(fineRadius) { return }
                    }
                    var filled = g
                    filled.fillUnreachable()
                    // Level-1 quads stay small so they can be dropped one by one where vanilla draws.
                    let m = mesh(filled, maxMerge: lvl == 1 ? 16 : 64)
                    outp[i] = LodNode(level: lvl, x0: x0, z0: z0, quads: m.quads, counts: m.counts, tileY: m.tileY)
                }
            }
            let made = meshed.compactMap { $0 }
            nodes.append(contentsOf: made)
            log("LOD: level \(lvl): \(made.count) nodes, \(made.reduce(0) { $0 + $1.quads.count / 2 }) quads (\(String(format: "%.1f", Date().timeIntervalSince(t0))) s)")
            if lvl >= maxLevel { break }
            // Next level: 2 x 2 children per parent.
            var parents: [Int64: LodGrid] = [:]
            for (k, g) in level {
                let (x, z) = unkey(k)
                let pk = key(x >> 1, z >> 1)   // arithmetic shift: floor division for negatives
                var p = parents[pk] ?? LodGrid(level: lvl + 1)
                g.downsample(into: &p, qx: x & 1, qz: z & 1)
                parents[pk] = p
            }
            level = parents
            lvl += 1
        }
        return nodes
    }

    @inline(__always) static func key(_ x: Int, _ z: Int) -> Int64 { Int64(Int32(truncatingIfNeeded: x)) << 32 | Int64(UInt32(bitPattern: Int32(truncatingIfNeeded: z))) }
    @inline(__always) static func unkey(_ k: Int64) -> (Int, Int) { (Int(Int32(truncatingIfNeeded: k >> 32)), Int(Int32(truncatingIfNeeded: k))) }
}

/// Debug: block names the material classifier doesn't know in one region file, as "name count" lines.
@_cdecl("mmc_debug_unknown_blocks")
public func mmc_debug_unknown_blocks(_ path: UnsafePointer<CChar>, _ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    guard let data = FileManager.default.contents(atPath: String(cString: path)) else { return 0 }
    let r = [UInt8](data)
    var counts: [String: Int] = [:]
    for i in 0..<1024 where Anvil.be32(r, i * 4) != 0 {
        if let c = try? Anvil.decodeChunk(region: r, index: i) {
            for (k, v) in c.unknown { counts[k, default: 0] += v }
        }
    }
    let text = counts.sorted { $0.value > $1.value }.map { "\($0.key) \($0.value)" }.joined(separator: "\n")
    let bytes = Array(text.utf8.prefix(Int(len) - 1))
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(counts.count)
}

/// Debug: decodes every chunk of a region with both decoders and counts chunks whose results differ.
/// out[0] = chunks compared, out[1] = mismatches, out[2] = old decoder µs, out[3] = new decoder µs.
@_cdecl("mmc_debug_compare_decoders")
public func mmc_debug_compare_decoders(_ path: UnsafePointer<CChar>, _ out: UnsafeMutablePointer<Int64>) {
    guard let data = FileManager.default.contents(atPath: String(cString: path)) else { return }
    let r = [UInt8](data)
    var compared: Int64 = 0, mismatches: Int64 = 0
    var tOld = 0.0, tNew = 0.0
    var cache: [String: UInt8] = [:]
    for i in 0..<1024 where Anvil.be32(r, i * 4) != 0 {
        let t0 = Date()
        let a = try? Anvil.decodeChunk(region: r, index: i)
        let t1 = Date()
        let b = try? ChunkScan.decodeChunk(region: r, index: i, cache: &cache)
        let t2 = Date()
        tOld += t1.timeIntervalSince(t0); tNew += t2.timeIntervalSince(t1)
        compared += 1
        guard let a, let b else { if (a == nil) != (b == nil) { mismatches += 1 }; continue }
        let same = a.cx == b.cx && a.cz == b.cz && a.status == b.status && a.surfaceBiomes == b.surfaceBiomes
            && a.sections.count == b.sections.count
            && zip(a.sections, b.sections).allSatisfy { $0.sy == $1.sy && $0.data == $1.data }
        if !same { mismatches += 1 }
    }
    out[0] = compared; out[1] = mismatches; out[2] = Int64(tOld * 1e6); out[3] = Int64(tNew * 1e6)
}
