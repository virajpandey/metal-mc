import Compression
import Foundation
import MetalMCCore

// Far terrain from the world generator. Where no chunk was ever generated, the LOD would end. In
// single-player the mod samples the generator's own terrain shape and biomes on a coarse grid
// (FarTerrain.java): per column the ground height (a search on the overworld's sloped_cheese density, about
// 1.3 blocks from the real surface) and the surface biome. Levels 3 and up are built from those columns
// wherever a column has no real data, so the horizon shows the world's real mountains, coasts and forests
// before the player has been there. Real chunks always win, column by column.
//
// Generated nodes are cached per world and seed under <game dir>/metalmc/lod/far/ (not in the save), so the
// ~85 us per column is paid once: one file per node, heights and a biome-name palette, LZFSE-compressed.

let lodFarMinLevel = 3
let lodFarDebug = experiments.contains("fardebug")
let lodSeaLevel = 63

/// One node's generated columns: 256 x 256 (one per voxel column at the node's level), x fastest.
struct LodFarColumns {
    var height: [Int16]      // first block above the ground (world y)
    var biome: [UInt16]      // id registered with mmc_lod_far_biome
}

/// How a biome's surface looks from far away.
struct LodFarSurface {
    var top: UInt8           // surface material (tinted)
    var under: UInt8         // the blocks under it
    var canopy: Int          // blocks of tree canopy above the ground (dense forests), 0 for none
    var leaves: UInt8        // canopy material (tinted)
    var frozen: Bool         // water freezes at the surface
    var mountain: Bool       // steep slopes show stone
    var bands = false        // badlands: terracotta in horizontal color bands by height
    var water: UInt8 = lodTinted(Mat.water.rawValue, 0)   // tinted water
}

/// Surface rules approximated per biome name (vanilla's surface rules and tree density, simplified).
func lodFarSurface(_ name: String) -> LodFarSurface {
    let t = lodTintIndex(name)
    let n = name.hasPrefix("minecraft:") ? String(name.dropFirst(10)) : name
    let grass = lodTinted(Mat.grass.rawValue, t), leaves = lodTinted(Mat.leaves.rawValue, t)
    let dirt = Mat.dirt.rawValue, stone = Mat.stone.rawValue, sand = Mat.sand.rawValue
    let snow = Mat.snow.rawValue, gravel = Mat.gravel.rawValue
    var s = LodFarSurface(top: grass, under: dirt, canopy: 0, leaves: leaves, frozen: false, mountain: false,
                          water: lodTinted(Mat.water.rawValue, t))
    switch n {
    case "forest", "flower_forest", "birch_forest", "old_growth_birch_forest", "windswept_forest": s.canopy = 6
    case "dark_forest", "pale_garden": s.canopy = 7
    case "taiga": s.canopy = 8
    case "old_growth_pine_taiga", "old_growth_spruce_taiga": s.canopy = 11
    case "jungle", "bamboo_jungle": s.canopy = 13
    case "cherry_grove": s.canopy = 5; s.leaves = Mat.cherryLeaves.rawValue
    case "mangrove_swamp": s.canopy = 6; s.top = Mat.mud.rawValue; s.under = Mat.mud.rawValue
    case "snowy_taiga", "grove": s.canopy = 7; s.top = snow
    case "snowy_plains", "ice_spikes": s.top = snow; s.frozen = true
    case "snowy_slopes", "frozen_peaks", "jagged_peaks": s.top = snow; s.under = stone; s.frozen = true; s.mountain = true
    case "stony_peaks": s.top = stone; s.under = stone; s.mountain = true
    case "windswept_hills", "windswept_savanna", "meadow": s.mountain = true
    case "windswept_gravelly_hills": s.top = gravel; s.under = gravel; s.mountain = true
    case "stony_shore": s.top = stone; s.under = stone
    case "desert": s.top = sand; s.under = Mat.sandstone.rawValue
    case "beach": s.top = sand; s.under = sand
    case "snowy_beach": s.top = snow; s.under = sand; s.frozen = true
    case "badlands", "eroded_badlands": s.top = Mat.terracotta.rawValue; s.under = Mat.terracotta.rawValue; s.bands = true
    case "wooded_badlands": s.top = Mat.terracotta.rawValue; s.under = Mat.terracotta.rawValue; s.canopy = 4; s.bands = true
    case "warm_ocean", "lukewarm_ocean", "deep_lukewarm_ocean", "beach_ocean": s.top = sand; s.under = sand
    case "ocean", "deep_ocean", "cold_ocean", "deep_cold_ocean", "river": s.top = gravel; s.under = gravel
    case "frozen_ocean", "deep_frozen_ocean", "frozen_river": s.top = gravel; s.under = gravel; s.frozen = true
    default: break
    }
    return s
}

/// Badlands bands: like vanilla's, a fixed sequence of terracotta colors by height (mostly plain terracotta,
/// with orange, yellow, brown, red, white and light gray bands 1-3 blocks thick), here the same everywhere.
let lodBadlandsBands: [UInt8] = {
    let colors: [Mat] = [.orangeTerracotta, .yellowTerracotta, .brownTerracotta, .redTerracotta, .whiteTerracotta, .lightGrayTerracotta]
    var out = [UInt8](repeating: Mat.terracotta.rawValue, count: 192)
    var y = 0, k = 0
    while y < out.count {
        y += 2 + Int(lodHash(k, 7, 3) * 5)
        let thick = 1 + Int(lodHash(k, 11, 4) * 3)
        let c = colors[Int(lodHash(k, 13, 5) * Double(colors.count))]
        for t in 0..<thick where y + t < out.count { out[y + t] = c.rawValue }
        y += thick
        k += 1
    }
    return out
}()

/// Deterministic hash of a block position, in [0, 1).
@inline(__always) func lodHash(_ x: Int, _ z: Int, _ salt: UInt64 = 0) -> Double {
    var h = UInt64(bitPattern: Int64(x)) &* 0x9E37_79B9_7F4A_7C15 ^ UInt64(bitPattern: Int64(z)) &* 0xC2B2_AE3D_27D4_EB4F ^ salt
    h ^= h >> 29
    h &*= 0xBF58_476D_1CE4_E5B9
    h ^= h >> 32
    return Double(h >> 11) / Double(UInt64(1) << 53)
}

final class LodFarStore: @unchecked Sendable {
    let lock = NSLock()
    var surfaces: [UInt16: LodFarSurface] = [:]
    var biomeIds: [String: UInt16] = [:]
    var biomeNames: [UInt16: String] = [:]
    var cacheDir: URL?
    var cached = Set<LodNodeKey>()          // nodes with a cache file not yet loaded
    var columns: [LodNodeKey: LodFarColumns] = [:]
    var dirty = Set<LodNodeKey>()          // received, not yet built into their node
    var requested = Set<LodNodeKey>()      // handed to the generator, not yet received
    var realRegions = Set<Int64>()         // regions with any real chunk
    var fullRegions = Set<Int64>()         // regions with every chunk generated
    var holeRegions = Set<Int64>()         // regions with some chunks (their levels 0-1 fill from generated nodes)

    /// Id for a biome name, registering it (with its surface) on first use. Caller holds `lock`.
    func biomeId(_ name: String) -> UInt16 {
        if let id = biomeIds[name] { return id }
        let id = UInt16(biomeIds.count)
        biomeIds[name] = id
        biomeNames[id] = name
        surfaces[id] = lodFarSurface(name)
        return id
    }

    static func fileName(_ k: LodNodeKey) -> String { "\(k.level).\(k.x).\(k.z).far" }

    /// Writes a node's columns to the cache: "MMCF" 1, then LZFSE of: palette count, names (length-prefixed
    /// UTF-8), heights (Int16) and palette indices (UInt16), 256 x 256 each. Caller holds no lock.
    func save(_ k: LodNodeKey, _ c: LodFarColumns) {
        lock.lock(); let dir = cacheDir; let names = biomeNames; lock.unlock()
        guard let dir else { return }
        var palette: [UInt16: UInt16] = [:]
        var paletteNames: [String] = []
        var body: [UInt8] = []
        body.reserveCapacity(c.height.count * 4 + 1024)
        var idx = [UInt16](repeating: 0, count: c.biome.count)
        for (i, b) in c.biome.enumerated() {
            if let p = palette[b] { idx[i] = p; continue }
            let p = UInt16(paletteNames.count)
            palette[b] = p
            paletteNames.append(names[b] ?? "minecraft:plains")
            idx[i] = p
        }
        body += [UInt8(paletteNames.count & 255), UInt8(paletteNames.count >> 8)]
        for n in paletteNames { let u = Array(n.utf8); body.append(UInt8(u.count)); body += u }
        for h in c.height { body += [UInt8(UInt16(bitPattern: h) & 255), UInt8(UInt16(bitPattern: h) >> 8)] }
        for p in idx { body += [UInt8(p & 255), UInt8(p >> 8)] }
        var packed = [UInt8](repeating: 0, count: body.count + 1024)
        let n = body.withUnsafeBufferPointer { compression_encode_buffer(&packed, packed.count, $0.baseAddress!, $0.count, nil, COMPRESSION_LZFSE) }
        guard n > 0 else { return }
        let len = body.count
        let out = Data(Array("MMCF".utf8) + [1, UInt8(len & 255), UInt8((len >> 8) & 255), UInt8((len >> 16) & 255), UInt8(len >> 24)] + packed[0..<n])
        try? out.write(to: dir.appendingPathComponent(Self.fileName(k)), options: .atomic)
    }

    /// Reads a node from the cache. Caller holds `lock` (biome names are registered).
    func load(_ k: LodNodeKey) -> LodFarColumns? {
        guard let dir = cacheDir, let d = try? Data(contentsOf: dir.appendingPathComponent(Self.fileName(k))), d.count > 9 else { return nil }
        let raw = [UInt8](d)
        guard raw[0...3] == Array("MMCF".utf8)[0...3], raw[4] == 1 else { return nil }
        let len = Int(raw[5]) | Int(raw[6]) << 8 | Int(raw[7]) << 16 | Int(raw[8]) << 24
        var body = [UInt8](repeating: 0, count: len)
        let got = raw.withUnsafeBufferPointer { compression_decode_buffer(&body, len, $0.baseAddress! + 9, $0.count - 9, nil, COMPRESSION_LZFSE) }
        let cols = lodNodeVoxels * lodNodeVoxels
        guard got == len, len >= 2 else { return nil }
        let count = Int(body[0]) | Int(body[1]) << 8
        var i = 2
        var ids: [UInt16] = []
        for _ in 0..<count {
            guard i < len else { return nil }
            let l = Int(body[i]); i += 1
            guard i + l <= len, let name = String(bytes: body[i..<(i + l)], encoding: .utf8) else { return nil }
            ids.append(biomeId(name)); i += l
        }
        guard len - i == cols * 4 else { return nil }
        var height = [Int16](repeating: 0, count: cols), biome = [UInt16](repeating: 0, count: cols)
        for c in 0..<cols { height[c] = Int16(bitPattern: UInt16(body[i + 2 * c]) | UInt16(body[i + 2 * c + 1]) << 8) }
        i += cols * 2
        for c in 0..<cols {
            let p = Int(UInt16(body[i + 2 * c]) | UInt16(body[i + 2 * c + 1]) << 8)
            biome[c] = p < ids.count ? ids[p] : 0
        }
        return LodFarColumns(height: height, biome: biome)
    }

    /// Writes one generated column into grid column `col` (voxels of level `L`, `h` voxels tall): stone, then
    /// the surface's under material, the surface block, and water up to sea level or tree canopy above it.
    /// `height` is the first block above the ground; `steep` puts stone on the surface in mountain biomes.
    static func writeColumn(_ v: UnsafeMutableBufferPointer<UInt8>, col: Int, n: Int, h: Int, L: Int,
                            height: Int, s: LodFarSurface, steep: Bool, bx: Int, bz: Int) {
        let top = height - 1                                  // the ground's top block
        let surfaceMat = s.mountain && steep ? Mat.stone.rawValue : s.top
        let vyTop = min(h - 1, max(0, (top + 64) >> L))
        for y in 0..<vyTop { v[y * n * n + col] = y + 2 >= vyTop ? s.under : Mat.stone.rawValue }
        if s.bands {
            // Terracotta bands down to 40 blocks under the surface (a voxel takes the band at its middle).
            let first = max(0, vyTop - max(1, 40 >> L))
            for y in first...vyTop {
                let block = (y << L) + (1 << L) / 2 - 64
                v[y * n * n + col] = lodBadlandsBands[((block % 192) + 192) % 192]
            }
            return
        }
        // Under water, grass and snow give way to what's under them (lake and river beds).
        let wet = top + 1 < lodSeaLevel
        let vegetation = surfaceMat == Mat.grass.rawValue || surfaceMat == Mat.snow.rawValue
            || (surfaceMat >= lodGrassBase && surfaceMat < lodGrassBase + 32)
        v[vyTop * n * n + col] = wet && vegetation ? s.under : surfaceMat
        if wet {
            let vyWater = min(h - 1, (lodSeaLevel - 1 + 64) >> L)
            if vyWater > vyTop {
                for y in (vyTop + 1)...vyWater { v[y * n * n + col] = s.water }
                if s.frozen { v[vyWater * n * n + col] = Mat.ice.rawValue }
            }
        } else if s.canopy > 0 {
            // Forest canopy isn't a flat sheet: tree height varies by 8-block patch (-2 to +2 blocks), and up close
            // (voxels under 8 blocks) one 4-block cell in six is a clearing.
            if L < 3 && lodHash(bx >> 2, bz >> 2, 1) < 1.0 / 6.0 { return }
            let canopy = s.canopy + Int(lodHash(bx >> 3, bz >> 3, 2) * 5) - 2
            let vyCanopy = min(h - 1, (top + canopy + 64) >> L)
            if vyCanopy > vyTop { for y in (vyTop + 1)...vyCanopy { v[y * n * n + col] = s.leaves } }
            else { v[vyTop * n * n + col] = s.leaves }
        }
    }

    static let defaultSurface = LodFarSurface(top: Mat.grass.rawValue, under: Mat.dirt.rawValue, canopy: 0, leaves: Mat.leaves.rawValue,
                                              frozen: false, mountain: false)

    /// Writes the far columns of `key` into `g` wherever the grid's column has no data (all air). Returns the
    /// number of columns filled. Caller holds no lock.
    func fill(_ g: inout LodGrid, key: LodNodeKey) -> Int {
        lock.lock()
        guard let cols = columns[key] else { lock.unlock(); return 0 }
        let table = surfaces
        lock.unlock()
        let n = lodNodeVoxels, h = g.height, L = g.level
        var filled = 0
        g.v.withUnsafeMutableBufferPointer { v in
            for z in 0..<n {
                for x in 0..<n {
                    let col = z * n + x
                    var empty = true
                    for y in 0..<h where v[y * n * n + col] != 0 { empty = false; break }
                    if !empty { continue }
                    // Steep: more than 1.5 blocks up or down per block to a neighbor, as on cliffs.
                    var slope = 0
                    for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                        let nx = x + dx, nz = z + dz
                        if nx < 0 || nz < 0 || nx >= n || nz >= n { continue }
                        slope = max(slope, abs(Int(cols.height[nz * n + nx]) - Int(cols.height[col])))
                    }
                    Self.writeColumn(v, col: col, n: n, h: h, L: L, height: Int(cols.height[col]),
                                     s: table[cols.biome[col]] ?? Self.defaultSurface, steep: slope * 2 > 3 << L,
                                     bx: key.x * (n << L) + (x << L), bz: key.z * (n << L) + (z << L))
                    filled += 1
                }
            }
        }
        return filled
    }

    /// For grids finer than level 3 (levels 0-2, near the player): fills columns with no data from the finest
    /// generated node that covers them, heights interpolated between its columns. That's where explored terrain
    /// ends; without it the LOD shows a ledge down to nothing there. Returns the number of columns filled.
    func fillFromAncestors(_ g: inout LodGrid, x0: Int, z0: Int, maxLevel: Int) -> Int {
        guard maxLevel >= lodFarMinLevel else { return 0 }
        lock.lock()
        let cols = columns, table = surfaces
        lock.unlock()
        if cols.isEmpty { return 0 }
        let n = lodNodeVoxels, h = g.height, L = g.level
        let voxel = 1 << L
        var filled = 0
        var lastKey: LodNodeKey?
        var last: LodFarColumns?
        g.v.withUnsafeMutableBufferPointer { v in
            for z in 0..<n {
                for x in 0..<n {
                    let col = z * n + x
                    var empty = true
                    for y in 0..<h where v[y * n * n + col] != 0 { empty = false; break }
                    if !empty { continue }
                    let bx = Double(x0 + x * voxel) + Double(voxel) / 2, bz = Double(z0 + z * voxel) + Double(voxel) / 2
                    // The finest generated node covering the column.
                    var found: (LodNodeKey, LodFarColumns)?
                    for lv in lodFarMinLevel...maxLevel {
                        let size = Double(lodNodeVoxels << lv)
                        let k = LodNodeKey(level: lv, x: Int((bx / size).rounded(.down)), z: Int((bz / size).rounded(.down)))
                        if k == lastKey, let last { found = (k, last); break }
                        if let c = cols[k] { found = (k, c); lastKey = k; last = c; break }
                    }
                    guard let (k, c) = found else { continue }
                    let s = Double(1 << k.level)
                    // Column coordinates within the node, centers at half a column.
                    let fx = (bx - Double(k.x * (lodNodeVoxels << k.level))) / s - 0.5
                    let fz = (bz - Double(k.z * (lodNodeVoxels << k.level))) / s - 0.5
                    let ix = min(n - 2, max(0, Int(fx.rounded(.down)))), iz = min(n - 2, max(0, Int(fz.rounded(.down))))
                    let tx = min(1, max(0, fx - Double(ix))), tz = min(1, max(0, fz - Double(iz)))
                    let h00 = Double(c.height[iz * n + ix]), h10 = Double(c.height[iz * n + ix + 1])
                    let h01 = Double(c.height[(iz + 1) * n + ix]), h11 = Double(c.height[(iz + 1) * n + ix + 1])
                    let height = Int(((h00 * (1 - tx) + h10 * tx) * (1 - tz) + (h01 * (1 - tx) + h11 * tx) * tz).rounded())
                    let nearest = (tz < 0.5 ? iz : iz + 1) * n + (tx < 0.5 ? ix : ix + 1)
                    let slope = max(abs(h10 - h00), abs(h01 - h00), abs(h11 - h10), abs(h11 - h01)) / s
                    Self.writeColumn(v, col: col, n: n, h: h, L: L, height: height,
                                     s: table[c.biome[nearest]] ?? Self.defaultSurface, steep: slope > 1.5,
                                     bx: x0 + x * voxel, bz: z0 + z * voxel)
                    filled += 1
                }
            }
        }
        return filled
    }
}

/// Nodes (level, x, z triples in `out`) that should get generated columns, nearest first: levels 3 and up
/// that the quadtree can draw around the player. Returned nodes are marked requested until they're put.
@_cdecl("mmc_lod_far_wanted")
public func mmc_lod_far_wanted(_ out: UnsafeMutablePointer<Int32>, _ max: Int32) -> Int32 {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return 0 }
    let list = w.farWanted(max: Int(max))
    for (i, k) in list.enumerated() {
        out[3 * i] = Int32(k.level); out[3 * i + 1] = Int32(k.x); out[3 * i + 2] = Int32(k.z)
    }
    return Int32(list.count)
}

/// The id mmc_lod_far_put uses for biome `name` (e.g. minecraft:forest), or -1 with no LOD open.
@_cdecl("mmc_lod_far_biome")
public func mmc_lod_far_biome(_ name: UnsafePointer<CChar>) -> Int32 {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return -1 }
    w.far.lock.lock(); defer { w.far.lock.unlock() }
    return Int32(w.far.biomeId(String(cString: name)))
}

/// Where generated nodes are cached for this world and seed (created if missing); "" for no cache.
@_cdecl("mmc_lod_far_cache")
public func mmc_lod_far_cache(_ dir: UnsafePointer<CChar>) {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return }
    let path = String(cString: dir)
    guard !path.isEmpty else { return }
    let url = URL(fileURLWithPath: path)
    try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
    var keys = Set<LodNodeKey>()
    for f in (try? FileManager.default.contentsOfDirectory(atPath: path)) ?? [] where f.hasSuffix(".far") {
        let p = f.dropLast(4).split(separator: ".")
        if p.count == 3, let l = Int(p[0]), let x = Int(p[1]), let z = Int(p[2]) { keys.insert(LodNodeKey(level: l, x: x, z: z)) }
    }
    w.far.lock.lock(); w.far.cacheDir = url; w.far.cached = keys; w.far.lock.unlock()
    log("LOD: far terrain cache \(path): \(keys.count) nodes")
}

/// Generated columns for node (level, x, z): 256 x 256 ground heights (first block above the ground) and
/// biome ids, x fastest. The node is rebuilt with them on the next update pass.
@_cdecl("mmc_lod_far_put")
public func mmc_lod_far_put(_ level: Int32, _ x: Int32, _ z: Int32, _ heights: UnsafePointer<Int16>, _ biomes: UnsafePointer<UInt16>) {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return }
    let n = lodNodeVoxels * lodNodeVoxels
    let key = LodNodeKey(level: Int(level), x: Int(x), z: Int(z))
    let cols = LodFarColumns(height: Array(UnsafeBufferPointer(start: heights, count: n)), biome: Array(UnsafeBufferPointer(start: biomes, count: n)))
    w.far.lock.lock()
    w.far.columns[key] = cols
    w.far.requested.remove(key)
    w.far.dirty.insert(key)
    w.far.lock.unlock()
    w.far.save(key, cols)
}
