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
    case "badlands", "eroded_badlands": s.top = Mat.terracotta.rawValue; s.under = Mat.terracotta.rawValue; s.mountain = true
    case "wooded_badlands": s.top = Mat.terracotta.rawValue; s.under = Mat.terracotta.rawValue; s.canopy = 4
    case "warm_ocean", "lukewarm_ocean", "deep_lukewarm_ocean", "beach_ocean": s.top = sand; s.under = sand
    case "ocean", "deep_ocean", "cold_ocean", "deep_cold_ocean", "river": s.top = gravel; s.under = gravel
    case "frozen_ocean", "deep_frozen_ocean", "frozen_river": s.top = gravel; s.under = gravel; s.frozen = true
    default: break
    }
    return s
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

    func surface(_ id: UInt16) -> LodFarSurface {
        surfaces[id] ?? LodFarSurface(top: Mat.grass.rawValue, under: Mat.dirt.rawValue, canopy: 0, leaves: Mat.leaves.rawValue, frozen: false, mountain: false)
    }

    /// Writes the far columns of `key` into `g` wherever the grid's column has no data (all air). Returns the
    /// number of columns filled. Caller holds no lock.
    func fill(_ g: inout LodGrid, key: LodNodeKey) -> Int {
        lock.lock()
        guard let cols = columns[key] else { lock.unlock(); return 0 }
        let table = surfaces
        lock.unlock()
        let n = lodNodeVoxels, h = g.height, L = g.level
        let stone = Mat.stone.rawValue, ice = Mat.ice.rawValue
        var filled = 0
        g.v.withUnsafeMutableBufferPointer { v in
            for z in 0..<n {
                for x in 0..<n {
                    let col = z * n + x
                    var empty = true
                    for y in 0..<h where v[y * n * n + col] != 0 { empty = false; break }
                    if !empty { continue }
                    let top = Int(cols.height[col]) - 1          // the ground's top block
                    let bid = cols.biome[col]
                    let s = table[bid] ?? LodFarSurface(top: Mat.grass.rawValue, under: Mat.dirt.rawValue, canopy: 0, leaves: Mat.leaves.rawValue, frozen: false, mountain: false)
                    // Steep ground (more than 1.5 blocks up or down per block to a neighbor) shows stone in
                    // mountain biomes, as vanilla's surface rules do on cliffs.
                    var surfaceMat = s.top
                    if s.mountain {
                        var slope = 0
                        for (dx, dz) in [(-1, 0), (1, 0), (0, -1), (0, 1)] {
                            let nx = x + dx, nz = z + dz
                            if nx < 0 || nz < 0 || nx >= n || nz >= n { continue }
                            slope = max(slope, abs(Int(cols.height[nz * n + nx]) - Int(cols.height[col])))
                        }
                        if slope * 2 > 3 << L { surfaceMat = stone }
                    }
                    let vyTop = min(h - 1, max(0, (top + 64) >> L))
                    for y in 0..<vyTop { v[y * n * n + col] = y + 2 >= vyTop ? s.under : stone }
                    // Under water, grass and snow give way to what's under them (lake and river beds).
                    let wet = top + 1 < lodSeaLevel
                    let vegetation = surfaceMat == Mat.grass.rawValue || surfaceMat == Mat.snow.rawValue
                        || (surfaceMat >= lodGrassBase && surfaceMat < lodGrassBase + 32)
                    v[vyTop * n * n + col] = wet && vegetation ? s.under : surfaceMat
                    if wet {
                        let vyWater = min(h - 1, (lodSeaLevel - 1 + 64) >> L)
                        if vyWater > vyTop {
                            for y in (vyTop + 1)...vyWater { v[y * n * n + col] = s.water }
                            if s.frozen { v[vyWater * n * n + col] = ice }
                        }
                    } else if s.canopy > 0 {
                        let vyCanopy = min(h - 1, (top + s.canopy + 64) >> L)
                        if vyCanopy > vyTop { for y in (vyTop + 1)...vyCanopy { v[y * n * n + col] = s.leaves } }
                        else { v[vyTop * n * n + col] = s.leaves }
                    }
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
