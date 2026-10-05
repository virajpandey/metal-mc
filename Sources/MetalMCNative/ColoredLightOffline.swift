import Foundation
import Metal
import MetalMCCore
import simd

// Colored block light offline (ColoredLight.swift; tools/litflow.swift drives it): the blocks around a camera from a
// world's region files, as the game's Java side would send them, a test scene of vanilla commands applied to them (and
// to the LOD, so the scene is drawn), and checks: the flood fill against a reference on the CPU, and its cost.
//
// In the game vanilla says how much light each block gives off and lets through (getLightEmission,
// getLightDampening); here they come from the block states' names and properties: vanilla's emission levels, and
// dampening 15 for full opaque blocks, 1 for water and leaves, 0 for the rest (slabs, stairs and the like let light
// through, as vanilla lets it through their open faces).

enum ClOffline {
    /// Vanilla's light emission for a block state (name without the namespace, properties).
    static func emission(_ n: String, _ props: [String: String]) -> Int {
        let lit = props["lit"] == "true"
        switch n {
        case "lava", "lava_cauldron", "fire", "lantern", "sea_lantern", "glowstone", "jack_o_lantern", "shroomlight", "beacon",
             "conduit", "end_portal", "end_gateway", "ochre_froglight", "verdant_froglight", "pearlescent_froglight":
            return 15
        case "campfire": return lit ? 15 : 0
        case "soul_campfire": return lit ? 10 : 0
        case "torch", "wall_torch", "end_rod", "copper_torch", "copper_wall_torch": return 14
        case "cave_vines", "cave_vines_plant": return props["berries"] == "true" ? 14 : 0
        case "furnace", "blast_furnace", "smoker": return lit ? 13 : 0
        case "nether_portal": return 11
        case "soul_torch", "soul_wall_torch", "soul_lantern", "soul_fire", "crying_obsidian": return 10
        case "redstone_ore", "deepslate_redstone_ore": return lit ? 9 : 0
        case "redstone_torch", "redstone_wall_torch": return props["lit"] == "false" ? 0 : 7
        case "redstone_lamp": return lit ? 15 : 0
        case "glow_lichen", "enchanting_table", "ender_chest": return 7
        case "sculk_catalyst", "vault": return 6
        case "amethyst_cluster": return 5
        case "large_amethyst_bud", "trial_spawner": return 4
        case "magma_block": return 3
        case "medium_amethyst_bud", "firefly_bush": return 2
        case "small_amethyst_bud", "brewing_stand", "brown_mushroom", "dragon_egg", "end_portal_frame", "sculk_sensor",
             "calibrated_sculk_sensor":
            return 1
        case "respawn_anchor": return [0, 3, 7, 11, 15][min(4, max(0, Int(props["charges"] ?? "0") ?? 0))]
        case "sea_pickle":
            return props["waterlogged"] == "true" ? 3 + 3 * min(4, max(1, Int(props["pickles"] ?? "1") ?? 1)) : 0
        case "light": return min(15, max(0, Int(props["level"] ?? "15") ?? 15))
        default: break
        }
        if n.hasSuffix("copper_lantern") { return 15 }
        if n.hasSuffix("candle") { return lit ? 3 * min(4, max(1, Int(props["candles"] ?? "1") ?? 1)) : 0 }
        if n.hasSuffix("candle_cake") { return lit ? 3 : 0 }
        if n.hasSuffix("copper_bulb") {
            guard lit else { return 0 }
            return n.contains("oxidized") ? 4 : (n.contains("weathered") ? 8 : (n.contains("exposed") ? 12 : 15))
        }
        return 0
    }

    private static let clear: Set<String> = [
        "air", "cave_air", "void_air", "structure_void", "light", "barrier", "glass", "glass_pane", "torch", "wall_torch",
        "soul_torch", "soul_wall_torch", "redstone_torch", "redstone_wall_torch", "copper_torch", "copper_wall_torch",
        "lantern", "soul_lantern", "end_rod", "lightning_rod", "chain", "iron_bars", "ladder", "lever", "repeater",
        "comparator", "redstone_wire", "tripwire", "tripwire_hook", "rail", "powered_rail", "detector_rail", "activator_rail",
        "snow", "cactus", "sugar_cane", "bamboo", "kelp", "kelp_plant", "seagrass", "tall_seagrass", "sea_pickle", "lily_pad",
        "short_grass", "tall_grass", "fern", "large_fern", "dead_bush", "vine", "glow_lichen", "sculk_vein", "cobweb",
        "fire", "soul_fire", "campfire", "soul_campfire", "brewing_stand", "enchanting_table", "end_portal_frame", "beacon",
        "conduit", "dragon_egg", "hopper", "cauldron", "water_cauldron", "lava_cauldron", "lectern", "grindstone",
        "stonecutter", "bell", "flower_pot", "decorated_pot", "cake", "scaffolding", "pointed_dripstone", "amethyst_cluster",
        "large_amethyst_bud", "medium_amethyst_bud", "small_amethyst_bud", "cave_vines", "cave_vines_plant", "spore_blossom",
        "hanging_roots", "big_dripleaf", "big_dripleaf_stem", "small_dripleaf", "azalea", "flowering_azalea", "moss_carpet",
        "nether_portal", "end_portal", "end_gateway", "frogspawn", "sculk_sensor", "calibrated_sculk_sensor",
        "sculk_shrieker", "daylight_detector", "brown_mushroom", "red_mushroom", "crimson_fungus", "warped_fungus",
        "crimson_roots", "warped_roots", "nether_sprouts", "twisting_vines", "twisting_vines_plant", "weeping_vines",
        "weeping_vines_plant", "sweet_berry_bush", "wheat", "carrots", "potatoes", "beetroots", "melon_stem",
        "pumpkin_stem", "attached_melon_stem", "attached_pumpkin_stem", "nether_wart", "cocoa", "torchflower",
        "pitcher_plant", "pitcher_crop", "leaf_litter", "firefly_bush", "heavy_core", "chest", "trapped_chest", "ender_chest",
        "anvil", "chipped_anvil", "damaged_anvil", "piston_head", "moving_piston", "bubble_column", "trial_spawner", "vault",
        "candle", "respawn_anchor",
    ]
    private static let clearSuffixes = ["_slab", "_stairs", "_fence", "_fence_gate", "_wall", "_door", "_trapdoor", "_pane",
                                        "_glass", "_sign", "_hanging_sign", "_wall_sign", "_banner", "_wall_banner", "_button",
                                        "_pressure_plate", "_carpet", "_bed", "_sapling", "_flower", "_tulip", "_candle",
                                        "_candle_cake", "_head", "_wall_head", "_skull", "_wall_skull", "_coral", "_coral_fan",
                                        "_coral_wall_fan", "_petals", "_eyeblossom", "_rail", "_shelf", "_copper_bulb",
                                        "_lantern", "_torch", "_wall_torch", "_bars", "_chain", "_grate", "_campfire"]
    private static let flowers: Set<String> = ["dandelion", "poppy", "blue_orchid", "allium", "azure_bluet", "oxeye_daisy",
                                               "cornflower", "lily_of_the_valley", "wither_rose", "sunflower", "lilac",
                                               "rose_bush", "peony", "open_eyeblossom", "closed_eyeblossom", "pink_petals",
                                               "wildflowers", "cactus_flower", "bush", "short_dry_grass", "tall_dry_grass"]

    /// Vanilla's light dampening, by name: 15 for full opaque blocks, 1 for water, leaves and ice, 0 for the rest.
    static func dampening(_ n: String) -> Int {
        if n == "water" || n == "lava" || n.hasSuffix("leaves") || n == "ice" || n == "frosted_ice" || n == "slime_block"
            || n == "honey_block" {
            return 1
        }
        if clear.contains(n) || flowers.contains(n) { return 0 }
        for s in clearSuffixes where n.hasSuffix(s) { return 0 }
        return 15
    }

    /// The cell code of a block state ("minecraft:torch" and its properties).
    static func code(_ fullName: String, _ props: [String: String]) -> UInt16 {
        let n = fullName.hasPrefix("minecraft:") ? String(fullName.dropFirst(10)) : fullName
        let e = emission(n, props)
        return clCode(dampening: dampening(n), emission: e, cls: e > 0 ? clClassify(n) : .none)
    }

    /// "minecraft:torch[facing=north]" (an optional {nbt} after it ignored) as its name and properties.
    static func parseState(_ s: String) -> (String, [String: String]) {
        var str = s
        if let brace = str.firstIndex(of: "{") { str = String(str[..<brace]) }
        guard let open = str.firstIndex(of: "["), let close = str.lastIndex(of: "]") else {
            return (str.hasPrefix("minecraft:") ? str : "minecraft:" + str, [:])
        }
        let name = String(str[..<open])
        var props: [String: String] = [:]
        for kv in str[str.index(after: open)..<close].split(separator: ",") {
            let p = kv.split(separator: "=", maxSplits: 1)
            if p.count == 2 { props[p[0].trimmingCharacters(in: .whitespaces)] = p[1].trimmingCharacters(in: .whitespaces) }
        }
        return (name.hasPrefix("minecraft:") ? name : "minecraft:" + name, props)
    }

    /// A palette entry's name and properties (26.x: {id, properties}, or a plain string; older: {Name, Properties}).
    static func paletteEntry(_ e: NBT) -> (String, [String: String]) {
        guard let name = Anvil.paletteName(e) else { return ("minecraft:air", [:]) }
        var props: [String: String] = [:]
        if let p = e["properties"] ?? e["Properties"] {
            for k in p.keys { if let v = p[k]?.stringValue { props[k] = v } }
        }
        return (name, props)
    }

    /// Chunk `index` of a region file's bytes: its position and its sections' codes (section y -4..19, 4096 codes y, z,
    /// x), all-air sections left out. Nil if it isn't a fully generated chunk.
    static func chunkCodes(region r: [UInt8], index i: Int) -> (cx: Int, cz: Int, sections: [(Int, [UInt16])])? {
        let offset = Int(Anvil.be32(r, i * 4) >> 8) * 4096
        guard offset >= 8192, offset + 5 <= r.count else { return nil }
        let length = Int(Anvil.be32(r, offset))
        let ctype = r[offset + 4]
        let start = offset + 5, end = offset + 4 + length
        guard length > 1, end <= r.count else { return nil }
        let raw: [UInt8]
        switch ctype {
        case 2:
            guard end - start > 6, let d = try? (Data(r[(start + 2)..<(end - 4)]) as NSData).decompressed(using: .zlib) as Data else { return nil }
            raw = [UInt8](d)
        case 3: raw = Array(r[start..<end])
        default: return nil
        }
        guard let root = try? NBTReader.parse(raw), let cx = root["xPos"]?.intValue, let cz = root["zPos"]?.intValue,
              (root["Status"]?.stringValue ?? "").hasSuffix("full"), let secs = root["sections"]?.listValue else { return nil }
        var out: [(Int, [UInt16])] = []
        for s in secs {
            guard let y = s["Y"]?.intValue, y >= ClStore.minSectionY, y <= ClStore.maxSectionY, let states = s["block_states"],
                  let palette = states["palette"]?.listValue, !palette.isEmpty else { continue }
            let codes = palette.map { e -> UInt16 in let (n, p) = paletteEntry(e); return code(n, p) }
            if codes.count == 1 {
                if codes[0] != 0 { out.append((y, [UInt16](repeating: codes[0], count: 4096))) }
                continue
            }
            guard let longs = states["data"]?.longArrayValue, !longs.isEmpty else { continue }
            let bits = max(4, Anvil.bitsNeeded(palette.count)), perLong = 64 / bits
            let mask = (UInt64(1) << UInt64(bits)) - 1
            var data = [UInt16](repeating: 0, count: 4096)
            for idx in 0..<4096 {
                let li = idx / perLong
                if li >= longs.count { break }
                let v = Int((UInt64(bitPattern: longs[li]) >> UInt64((idx % perLong) * bits)) & mask)
                data[idx] = v < codes.count ? codes[v] : 0
            }
            if data.contains(where: { $0 != 0 }) { out.append((y, data)) }
        }
        return (cx, cz, out)
    }

    /// The scene's commands as block placements: "fill x0 y0 z0 x1 y1 z1 <state>" and "setblock x y z <state>" (absolute
    /// coordinates; anything else, and lines starting with #, ignored), in order.
    static func sceneBlocks(_ text: String) -> [(SIMD3<Int>, String, [String: String])] {
        var out: [(SIMD3<Int>, String, [String: String])] = []
        for line in text.split(separator: "\n") {
            let t = line.trimmingCharacters(in: .whitespaces)
            if t.isEmpty || t.hasPrefix("#") { continue }
            let w = t.split(separator: " ", omittingEmptySubsequences: true).map(String.init)
            if w[0] == "fill", w.count >= 8, let x0 = Int(w[1]), let y0 = Int(w[2]), let z0 = Int(w[3]), let x1 = Int(w[4]),
               let y1 = Int(w[5]), let z1 = Int(w[6]) {
                let (n, p) = parseState(w[7])
                for x in min(x0, x1)...max(x0, x1) {
                    for y in min(y0, y1)...max(y0, y1) { for z in min(z0, z1)...max(z0, z1) { out.append((SIMD3(x, y, z), n, p)) } }
                }
            } else if w[0] == "setblock", w.count >= 5, let x = Int(w[1]), let y = Int(w[2]), let z = Int(w[3]) {
                let (n, p) = parseState(w[4...].joined(separator: " "))
                out.append((SIMD3(x, y, z), n, p))
            }
        }
        return out
    }
}

@inline(__always) private func floorDiv(_ a: Int, _ b: Int) -> Int { a >= 0 ? a / b : -((-a + b - 1) / b) }

/// Offline: loads the chunks from (cx0, cz0) to (cx1, cz1) (inclusive) of the world's overworld region files into the
/// colored light's store, as the game's Java side sends them. Returns the number of chunks read.
@_cdecl("mmc_debug_cl_load")
public func mmc_debug_cl_load(_ world: UnsafePointer<CChar>, _ cx0: Int32, _ cz0: Int32, _ cx1: Int32, _ cz1: Int32) -> Int32 {
    let dir = Anvil.regionDirectory(URL(fileURLWithPath: String(cString: world)), dimension: "minecraft:overworld")
    var read: Int32 = 0
    for rx in floorDiv(Int(cx0), 32)...floorDiv(Int(cx1), 32) {
        for rz in floorDiv(Int(cz0), 32)...floorDiv(Int(cz1), 32) {
            guard let data = FileManager.default.contents(atPath: dir.appendingPathComponent("r.\(rx).\(rz).mca").path), data.count >= 8192 else { continue }
            let r = [UInt8](data)
            for i in 0..<1024 where Anvil.be32(r, i * 4) != 0 {
                let cx = rx * 32 + (i & 31), cz = rz * 32 + (i >> 5)
                guard cx >= Int(cx0), cx <= Int(cx1), cz >= Int(cz0), cz <= Int(cz1), let c = ClOffline.chunkCodes(region: r, index: i) else { continue }
                ColoredLight.shared.store.putChunk(cx: c.cx, cz: c.cz, c.sections)
                read += 1
            }
        }
    }
    return read
}

/// Offline: applies a scene of vanilla commands (fill, setblock; ClOffline.sceneBlocks) to the colored light's store and to
/// the LOD (each chunk it touches decoded from the world's region files with the scene's blocks on top, and handed to the
/// LOD as a live chunk, which it then rebuilds in the background). Returns the number of blocks placed, or -1.
@_cdecl("mmc_debug_cl_scene")
public func mmc_debug_cl_scene(_ path: UnsafePointer<CChar>, _ world: UnsafePointer<CChar>) -> Int32 {
    guard let text = try? String(contentsOfFile: String(cString: path), encoding: .utf8) else { return -1 }
    let blocks = ClOffline.sceneBlocks(text)
    let dir = Anvil.regionDirectory(URL(fileURLWithPath: String(cString: world)), dimension: "minecraft:overworld")
    // By chunk, in order (later commands win).
    var byChunk: [Int64: [(SIMD3<Int>, String, [String: String])]] = [:]
    for b in blocks { byChunk[clSectionKey(floorDiv(b.0.x, 16), 0, floorDiv(b.0.z, 16)), default: []].append(b) }
    var regions: [Int64: [UInt8]] = [:]
    var cache: [String: UInt8] = [:]
    let scratch = ChunkScan.Scratch()
    for (_, list) in byChunk {
        let cx = floorDiv(list[0].0.x, 16), cz = floorDiv(list[0].0.z, 16)
        let rk = clSectionKey(floorDiv(cx, 32), 0, floorDiv(cz, 32))
        if regions[rk] == nil {
            regions[rk] = FileManager.default.contents(atPath: dir.appendingPathComponent("r.\(floorDiv(cx, 32)).\(floorDiv(cz, 32)).mca").path).map { [UInt8]($0) } ?? []
        }
        let r = regions[rk]!
        let index = (cz & 31) * 32 + (cx & 31)
        // The colored light's codes, by section.
        var secs: [Int: [UInt16]] = [:]
        var mats = [UInt8](repeating: 0, count: 16 * 16 * lodWorldHeight)
        var tints = [UInt8](repeating: 0, count: 16)
        if r.count >= 8192 && Anvil.be32(r, index * 4) != 0 {
            if let c = ClOffline.chunkCodes(region: r, index: index) { for (sy, codes) in c.sections { secs[sy] = codes } }
            if let chunk = r.withUnsafeBufferPointer({ try? ChunkScan.decodeChunk(region: $0, index: index, cache: &cache, scratch: scratch) }) {
                for (sy, b) in chunk.sections where sy >= 0 && sy < lodWorldHeight / 16 { for k in 0..<4096 { mats[sy * 4096 + k] = b[k] } }
                for k in 0..<16 { tints[k] = chunk.surfaceBiomes.count == 16 ? lodTintIndex(chunk.surfaceBiomes[k]) : 0 }
            }
        }
        for (p, name, props) in list {
            let sy = floorDiv(p.y, 16)
            guard sy >= ClStore.minSectionY, sy <= ClStore.maxSectionY, p.y >= lodWorldMinY, p.y < lodWorldMinY + lodWorldHeight else { continue }
            if secs[sy] == nil { secs[sy] = [UInt16](repeating: 0, count: 4096) }
            secs[sy]![((p.y & 15) * 16 + (p.z & 15)) * 16 + (p.x & 15)] = ClOffline.code(name, props)
            mats[((p.y - lodWorldMinY) * 16 + (p.z & 15)) * 16 + (p.x & 15)] = Materials.classify(name).rawValue
        }
        ColoredLight.shared.store.putChunk(cx: cx, cz: cz, secs.sorted { $0.key < $1.key }.map { ($0.key, $0.value) })
        mmc_lod_ingest(Int32(cx), Int32(cz), mats, tints)
    }
    return Int32(blocks.count)
}

/// Offline A/B: `on` 0 relights with vanilla's block light (the volume keeps running), 1 with the colored light; `time` >=
/// 0 holds the flicker's clock there (seconds), below 0 it follows the system's.
@_cdecl("mmc_debug_cl_on")
public func mmc_debug_cl_on(_ on: Int32, _ time: Double) -> Int32 {
    ColoredLight.shared.shadingOn = on != 0
    ColoredLight.shared.debugTime = time
    return clEnabled ? 1 : 0
}

/// Offline: forgets what the volume holds (every section is uploaded again, its light from nothing), for a timing of the
/// fill from scratch.
@_cdecl("mmc_debug_cl_invalidate")
public func mmc_debug_cl_invalidate() {
    ColoredLight.shared.invalidate()
}

/// Offline timing: `frames` frames of the volume's work alone for a camera at (x, y, z), each in a command buffer of its
/// own, waited for. out[2k]: frame k's GPU milliseconds, out[2k + 1]: the bricks its flood fill ran over.
@_cdecl("mmc_debug_cl_bench")
public func mmc_debug_cl_bench(_ x: Double, _ y: Double, _ z: Double, _ frames: Int32, _ out: UnsafeMutablePointer<Double>) -> Int32 {
    guard clEnabled else { return 0 }
    let c = ColoredLight.shared
    for k in 0..<Int(frames) {
        guard let cb = ctx.queue.makeCommandBuffer() else { return 0 }
        c.debugCB = cb
        c.frame(camera: SIMD3(x, y, z))
        c.debugCB = nil
        cb.commit()
        cb.waitUntilCompleted()
        out[2 * k] = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        // The listing this frame ran (the args buffer, shared, now final), read by the next frame's encode: read it here.
        out[2 * k + 1] = Double(c.debugListed())
    }
    return 1
}

/// Offline check: the volume's light against a flood fill on the CPU over the same codes (the GPU's own, read back), per
/// bucket, as vanilla spreads block light, and the GPU's codes against the store's. Call once the volume has settled.
/// out: [cells compared, cells whose light differs, cells whose code differs from the store's, cells lit (any bucket),
/// the largest level difference, cells where the GPU's brightest bucket is brighter than the CPU's].
@_cdecl("mmc_debug_cl_verify")
public func mmc_debug_cl_verify(_ out: UnsafeMutablePointer<Int64>) -> Int32 {
    let c = ColoredLight.shared
    guard clEnabled, let rb = c.debugReadback() else { return 0 }
    let org = c.debugOrigin
    let sx = clSizeX, sy = clSizeY, sz = clSizeZ, n = sx * sy * sz
    let om = SIMD3(((org.x % sx) + sx) % sx, ((org.y % sy) + sy) % sy, ((org.z % sz) + sz) % sz)
    // Cells in volume-relative order (x fastest); idx maps to the kernels' storage.
    var idx = [Int32](repeating: 0, count: n)
    for z in 0..<sz { for y in 0..<sy { for x in 0..<sx {
        idx[(z * sy + y) * sx + x] = Int32(ColoredLight.cellIndex((x + om.x) & (sx - 1), (y + om.y) & (sy - 1), (z + om.z) & (sz - 1)))
    } } }
    // The store's codes for the same cells.
    var codeDiff: Int64 = 0
    let store = c.store
    store.lock.lock()
    for z in stride(from: 0, to: sz, by: 16) { for y in stride(from: 0, to: sy, by: 16) { for x in stride(from: 0, to: sx, by: 16) {
        let wx = org.x + x, wy = org.y + y, wz = org.z + z
        let s = store.section(floorDiv(wx, 16), floorDiv(wy, 16), floorDiv(wz, 16))
        for ly in 0..<16 { for lz in 0..<16 { for lx in 0..<16 {
            let want = s.codes?[(ly * 16 + lz) * 16 + lx] ?? s.uniform
            if rb.blocks[Int(idx[((z + lz) * sy + y + ly) * sx + x + lx])] != want { codeDiff += 1 }
        } } }
    } } }
    store.lock.unlock()
    // The table the kernels use: class -> buckets.
    func emission(_ code: UInt16) -> [Int] {
        var e = [Int](repeating: 0, count: 8)
        let level = Int(code >> 4) & 15
        guard level > 0, let cls = ClClass(rawValue: UInt8((code >> 8) & 63)) else { return e }
        let (a, b, d) = cls.buckets
        e[a] = level
        if let b, level > d { e[b] = max(e[b], level - d) }
        return e
    }
    var cpu = [[UInt8]](repeating: [UInt8](repeating: 0, count: n), count: 8)
    for k in 0..<8 {
        var queues = [[Int32]](repeating: [], count: 16)
        for i in 0..<n {
            let code = rb.blocks[Int(idx[i])]
            if (code >> 4) & 15 == 0 { continue }
            let l = emission(code)[k]
            if l > 0 { cpu[k][i] = UInt8(l); queues[l].append(Int32(i)) }
        }
        var level = 15
        while level > 1 {
            for i32 in queues[level] where Int(cpu[k][Int(i32)]) == level {
                let i = Int(i32), x = i % sx, y = (i / sx) % sy, z = i / (sx * sy)
                func push(_ j: Int) {
                    let dec = max(1, Int(rb.blocks[Int(idx[j])] & 15))
                    if dec >= 15 { return }
                    let l = level - dec
                    if l > Int(cpu[k][j]) { cpu[k][j] = UInt8(l); queues[l].append(Int32(j)) }
                }
                if x > 0 { push(i - 1) }
                if x < sx - 1 { push(i + 1) }
                if y > 0 { push(i - sx) }
                if y < sy - 1 { push(i + sx) }
                if z > 0 { push(i - sx * sy) }
                if z < sz - 1 { push(i + sx * sy) }
            }
            queues[level].removeAll()
            level -= 1
        }
    }
    var differ: Int64 = 0, lit: Int64 = 0, maxDiff: Int64 = 0, brighter: Int64 = 0
    for i in 0..<n {
        let g = rb.light[Int(idx[i])]
        var any = false, bad = false, gm = 0, cm = 0
        for k in 0..<8 {
            let gl = Int((g >> (4 * UInt32(k))) & 15), cl = Int(cpu[k][i])
            if gl != cl { bad = true; maxDiff = max(maxDiff, Int64(abs(gl - cl))) }
            if gl > 0 || cl > 0 { any = true }
            gm = max(gm, gl); cm = max(cm, cl)
        }
        if bad { differ += 1 }
        if any { lit += 1 }
        if gm > cm { brighter += 1 }
    }
    out[0] = Int64(n); out[1] = differ; out[2] = codeDiff; out[3] = lit; out[4] = maxDiff; out[5] = brighter
    return 1
}

/// Offline: the store's light sources in the box (x0, y0, z0)-(x1, y1, z1) (world blocks, inclusive), up to `max`, as
/// (x, y, z, code) in `out`; and per source, in out2, how many of its 6 neighbors are open (dampening under 15). Returns
/// how many there are in all.
@_cdecl("mmc_debug_cl_emitters")
public func mmc_debug_cl_emitters(_ x0: Int32, _ y0: Int32, _ z0: Int32, _ x1: Int32, _ y1: Int32, _ z1: Int32,
                                  _ out: UnsafeMutablePointer<Int32>, _ out2: UnsafeMutablePointer<Int32>, _ max: Int32) -> Int32 {
    let store = ColoredLight.shared.store
    store.lock.lock(); defer { store.lock.unlock() }
    func code(_ x: Int, _ y: Int, _ z: Int) -> UInt16 {
        let s = store.section(floorDiv(x, 16), floorDiv(y, 16), floorDiv(z, 16))
        return s.codes?[((y & 15) * 16 + (z & 15)) * 16 + (x & 15)] ?? s.uniform
    }
    var n: Int32 = 0
    for sy in floorDiv(Int(y0), 16)...floorDiv(Int(y1), 16) {
        for sz in floorDiv(Int(z0), 16)...floorDiv(Int(z1), 16) {
            for sx in floorDiv(Int(x0), 16)...floorDiv(Int(x1), 16) {
                let s = store.section(sx, sy, sz)
                guard let codes = s.codes else { continue }
                for i in 0..<4096 where (codes[i] >> 4) & 15 != 0 {
                    let x = sx * 16 + (i & 15), z = sz * 16 + ((i >> 4) & 15), y = sy * 16 + (i >> 8)
                    guard x >= Int(x0), x <= Int(x1), y >= Int(y0), y <= Int(y1), z >= Int(z0), z <= Int(z1) else { continue }
                    if n < max {
                        out[4 * Int(n)] = Int32(x); out[4 * Int(n) + 1] = Int32(y); out[4 * Int(n) + 2] = Int32(z); out[4 * Int(n) + 3] = Int32(codes[i])
                        var open: Int32 = 0
                        for (dx, dy, dz) in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)] where code(x + dx, y + dy, z + dz) & 15 < 15 { open += 1 }
                        out2[Int(n)] = open
                    }
                    n += 1
                }
            }
        }
    }
    return n
}

/// Offline: whether block (x, y, z) is open (dampening under 15) in the store: 1, else 0.
@_cdecl("mmc_debug_cl_open")
public func mmc_debug_cl_open(_ x: Int32, _ y: Int32, _ z: Int32) -> Int32 {
    let store = ColoredLight.shared.store
    store.lock.lock(); defer { store.lock.unlock() }
    let s = store.section(floorDiv(Int(x), 16), floorDiv(Int(y), 16), floorDiv(Int(z), 16))
    let c = s.codes?[((Int(y) & 15) * 16 + (Int(z) & 15)) * 16 + (Int(x) & 15)] ?? s.uniform
    return c & 15 < 15 ? 1 : 0
}

/// Debug: the volume's kernels' source, for an offline compile check. Returns its length.
@_cdecl("mmc_debug_cl_shader_source")
public func mmc_debug_cl_shader_source(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let bytes = Array(clKernelSource.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}
