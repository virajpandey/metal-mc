import simd

public enum MaterialKind: UInt8 {
    case air, opaque, water
}

/// Flat-colored render materials. Milestone 1 stand-in for real block models and textures:
/// every block state maps to one of these by name.
public enum Mat: UInt8, CaseIterable {
    case air, stone, dirt, grass, sand, water, unknown
    case deepslate, gravel, log, planks, leaves, cherryLeaves, snow, ice, clay, terracotta, lava
    case cobblestone, bricks, path, farmland, hay, wool, moss, cherryWood, lightStone, granite, sandstone, mud
    case amethyst, pumpkin
    // Landscape materials with their own textures: the stone varieties of mountains, the terracotta bands of
    // badlands, taiga and mushroom-island ground, frozen peaks. The LOD keeps ids below 64.
    case andesite, diorite, tuff, dripstone, coarseDirt, podzol, mycelium, redSand, redSandstone
    case whiteTerracotta, orangeTerracotta, yellowTerracotta, redTerracotta, brownTerracotta, lightGrayTerracotta
    case packedIce, blueIce, obsidian
    // The End: end stone islands, purpur end cities and chorus plants.
    case endStone, purpur, chorus
    // Light sources. Solid ones are drawn (and glow); the others (torches, lanterns, fire, campfires, end rods,
    // glow lichen) are invisible from LOD distances but light what's around them, by their light level.
    case glowstone, seaLantern, shroomlight, jackOLantern, froglight
    case light15, light14, light10, light7
    // 26.3's poplars (dappled forests): leaves with their own colors, not tinted by the biome. Past the LOD's
    // biome-tinted ids (64-159).
    case yellowPoplarLeaves = 160, redPoplarLeaves, orangePoplarLeaves
    // Huge mushrooms (dark forests, mushroom fields) and ocean monuments.
    case redMushroomBlock, brownMushroomBlock, mushroomStem, prismarine, darkPrismarine
    // Magma blocks (ocean floors, the Nether): drawn full-bright like vanilla's (emissive), but they only give off
    // light level 3, where lava gives 15.
    case magma

    public var kind: MaterialKind {
        switch self {
        case .air, .light15, .light14, .light10, .light7: return .air
        case .water: return .water
        default: return .opaque
        }
    }

    public var color: SIMD4<Float> {
        switch self {
        case .air: return SIMD4(0, 0, 0, 0)
        case .stone: return SIMD4(0.50, 0.50, 0.53, 1)
        case .dirt: return SIMD4(0.47, 0.33, 0.21, 1)
        case .grass: return SIMD4(0.37, 0.63, 0.26, 1)
        case .sand: return SIMD4(0.86, 0.80, 0.56, 1)
        case .water: return SIMD4(0.18, 0.38, 0.85, 0.62)
        case .unknown: return SIMD4(0.78, 0.35, 0.78, 1)
        case .deepslate: return SIMD4(0.30, 0.30, 0.33, 1)
        case .gravel: return SIMD4(0.55, 0.53, 0.52, 1)
        case .log: return SIMD4(0.40, 0.30, 0.18, 1)
        case .planks: return SIMD4(0.66, 0.52, 0.32, 1)
        case .leaves: return SIMD4(0.22, 0.45, 0.15, 1)
        case .cherryLeaves: return SIMD4(0.93, 0.66, 0.78, 1)
        case .snow: return SIMD4(0.95, 0.96, 0.98, 1)
        case .ice: return SIMD4(0.62, 0.75, 0.95, 1)
        case .clay: return SIMD4(0.62, 0.64, 0.70, 1)
        case .terracotta: return SIMD4(0.62, 0.38, 0.28, 1)
        case .lava: return SIMD4(0.95, 0.45, 0.10, 1)
        case .cobblestone: return SIMD4(0.45, 0.45, 0.46, 1)
        case .bricks: return SIMD4(0.58, 0.30, 0.25, 1)
        case .path: return SIMD4(0.58, 0.47, 0.28, 1)
        case .farmland: return SIMD4(0.40, 0.26, 0.15, 1)
        case .hay: return SIMD4(0.80, 0.70, 0.25, 1)
        case .wool: return SIMD4(0.90, 0.90, 0.90, 1)
        case .moss: return SIMD4(0.35, 0.45, 0.25, 1)
        case .cherryWood: return SIMD4(0.35, 0.20, 0.22, 1)
        case .lightStone: return SIMD4(0.70, 0.70, 0.70, 1)
        case .granite: return SIMD4(0.60, 0.45, 0.40, 1)
        case .sandstone: return SIMD4(0.85, 0.78, 0.55, 1)
        case .mud: return SIMD4(0.35, 0.30, 0.28, 1)
        case .amethyst: return SIMD4(0.60, 0.45, 0.80, 1)
        case .pumpkin: return SIMD4(0.85, 0.52, 0.12, 1)
        case .andesite: return SIMD4(0.53, 0.53, 0.53, 1)
        case .diorite: return SIMD4(0.74, 0.74, 0.74, 1)
        case .tuff: return SIMD4(0.42, 0.43, 0.38, 1)
        case .dripstone: return SIMD4(0.53, 0.42, 0.36, 1)
        case .coarseDirt: return SIMD4(0.47, 0.33, 0.23, 1)
        case .podzol: return SIMD4(0.36, 0.25, 0.10, 1)
        case .mycelium: return SIMD4(0.44, 0.39, 0.42, 1)
        case .redSand: return SIMD4(0.75, 0.40, 0.13, 1)
        case .redSandstone: return SIMD4(0.72, 0.39, 0.13, 1)
        case .whiteTerracotta: return SIMD4(0.82, 0.70, 0.63, 1)
        case .orangeTerracotta: return SIMD4(0.63, 0.33, 0.14, 1)
        case .yellowTerracotta: return SIMD4(0.73, 0.52, 0.21, 1)
        case .redTerracotta: return SIMD4(0.56, 0.24, 0.18, 1)
        case .brownTerracotta: return SIMD4(0.30, 0.20, 0.14, 1)
        case .lightGrayTerracotta: return SIMD4(0.53, 0.42, 0.38, 1)
        case .packedIce: return SIMD4(0.55, 0.71, 0.97, 1)
        case .blueIce: return SIMD4(0.45, 0.63, 0.99, 1)
        case .obsidian: return SIMD4(0.06, 0.04, 0.10, 1)
        case .endStone: return SIMD4(0.86, 0.87, 0.62, 1)
        case .purpur: return SIMD4(0.66, 0.49, 0.66, 1)
        case .chorus: return SIMD4(0.37, 0.24, 0.37, 1)
        case .glowstone: return SIMD4(0.67, 0.53, 0.33, 1)
        case .seaLantern: return SIMD4(0.67, 0.78, 0.74, 1)
        case .shroomlight: return SIMD4(0.94, 0.58, 0.28, 1)
        case .jackOLantern: return SIMD4(0.84, 0.60, 0.19, 1)
        case .froglight: return SIMD4(0.93, 0.89, 0.72, 1)
        case .light15, .light14, .light10, .light7: return SIMD4(0, 0, 0, 0)
        case .yellowPoplarLeaves: return SIMD4(0.84, 0.52, 0.16, 1)
        case .redPoplarLeaves: return SIMD4(0.60, 0.17, 0.15, 1)
        case .orangePoplarLeaves: return SIMD4(0.74, 0.34, 0.09, 1)
        case .redMushroomBlock: return SIMD4(0.78, 0.18, 0.17, 1)
        case .brownMushroomBlock: return SIMD4(0.58, 0.44, 0.33, 1)
        case .mushroomStem: return SIMD4(0.80, 0.77, 0.70, 1)
        case .prismarine: return SIMD4(0.39, 0.63, 0.58, 1)
        case .darkPrismarine: return SIMD4(0.20, 0.36, 0.30, 1)
        case .magma: return SIMD4(0.55, 0.25, 0.10, 1)
        }
    }
}

public enum Materials {
    /// Indexed by raw value (256 entries: material ids aren't contiguous; ids from 160 on follow the LOD's tinted ids).
    public static let kinds: [MaterialKind] = {
        var k = [MaterialKind](repeating: .opaque, count: 256)
        for m in Mat.allCases { k[Int(m.rawValue)] = m.kind }
        return k
    }()
    public static let colors: [SIMD4<Float>] = {
        var c = [SIMD4<Float>](repeating: SIMD4(0.5, 0.5, 0.5, 1), count: 256)
        for m in Mat.allCases { c[Int(m.rawValue)] = m.color }
        return c
    }()

    /// Thin or decorative blocks (plants, torches, rails, glass, fences, ...) are skipped for now.
    private static let decorative = [
        "torch", "rail", "button", "pressure_plate", "sign", "banner", "sapling", "vine", "lily_pad",
        "carpet", "wheat", "carrots", "potatoes", "beetroots", "sugar_cane", "dandelion", "poppy", "tulip",
        "orchid", "allium", "bluet", "daisy", "cornflower", "lily_of_the_valley", "petals", "leaf_litter",
        "bell", "lantern", "chain", "ladder", "lever", "tripwire", "cobweb", "glass", "iron_bars", "fence",
        "door", "flower", "bush", "pot", "candle", "head", "skull", "scaffolding", "lilac", "rose", "peony",
        "sunflower", "pitcher", "roots", "lichen", "sculk_vein", "pointed_dripstone", "amethyst_cluster",
        "_bud", "coral", "sea_pickle", "twisting", "weeping", "frogspawn", "redstone_wire", "hopper",
        "cactus", "bamboo", "cocoa", "melon_stem", "pumpkin_stem", "nether_wart", "sweet_berry",
        "brewing_stand", "end_rod", "shrub", "eyeblossom", "shelf_mushroom",
    ]

    public static func classify(_ fullName: String) -> Mat {
        let n = fullName.hasPrefix("minecraft:") ? String(fullName.dropFirst(10)) : fullName
        switch n {
        case "air", "cave_air", "void_air", "structure_void", "light", "barrier":
            return .air
        case "torch", "wall_torch", "end_rod": return .light14
        case "soul_torch", "soul_wall_torch", "soul_lantern", "soul_fire", "soul_campfire": return .light10
        case "redstone_torch", "redstone_wall_torch", "glow_lichen": return .light7
        case "lantern", "fire", "campfire", "beacon": return .light15
        case "glowstone": return .glowstone
        case "sea_lantern": return .seaLantern
        case "shroomlight": return .shroomlight
        case "jack_o_lantern": return .jackOLantern
        case "water", "bubble_column", "seagrass", "tall_seagrass", "kelp", "kelp_plant":
            return .water
        case "snow":
            // Snow layers: thin, but they make the ground white, which is what matters from a distance.
            return .snow
        case "short_grass", "tall_grass", "grass", "fern", "large_fern", "brown_mushroom", "red_mushroom",
             "short_dry_grass", "tall_dry_grass", "azalea", "flowering_azalea", "big_dripleaf", "big_dripleaf_stem",
             "small_dripleaf", "spore_blossom", "hanging_roots", "cactus_flower":
            return .air
        default:
            break
        }
        func has(_ s: String) -> Bool { n.contains(s) }
        func any(_ list: [String]) -> Bool { list.contains { n.contains($0) } }

        if has("froglight") { return .froglight }
        if has("end_stone") || n == "end_portal_frame" { return .endStone }
        if has("purpur") { return .purpur }
        if has("chorus") { return .chorus }
        if n == "end_portal" || n == "end_gateway" || n == "dragon_egg" { return .obsidian }
        if has("yellow_poplar_leaves") { return .yellowPoplarLeaves }
        if has("red_poplar_leaves") { return .redPoplarLeaves }
        if has("orange_poplar_leaves") { return .orangePoplarLeaves }
        if has("cherry_leaves") { return .cherryLeaves }
        if n == "red_mushroom_block" { return .redMushroomBlock }
        if n == "brown_mushroom_block" { return .brownMushroomBlock }
        if n == "mushroom_stem" { return .mushroomStem }
        if has("dark_prismarine") { return .darkPrismarine }
        if has("prismarine") { return .prismarine }
        if n == "creaking_heart" { return .log }
        if has("quartz") { return .lightStone }
        if has("leaves") { return .leaves }
        if has("grass_block") { return .grass }
        if any(decorative) { return .air }
        if has("moss") { return .moss }
        if has("amethyst") { return .amethyst }
        if has("magma") { return .magma }
        if has("pumpkin") || has("melon") { return .pumpkin }
        if n.hasPrefix("raw_") || n == "spawner" { return .stone }
        if has("dirt_path") { return .path }
        if has("farmland") { return .farmland }
        if has("mud") { return .mud }
        if has("coarse_dirt") { return .coarseDirt }
        if has("podzol") { return .podzol }
        if has("mycelium") { return .mycelium }
        if has("dirt") { return .dirt }
        if has("red_sandstone") { return .redSandstone }
        if has("sandstone") { return .sandstone }
        if has("red_sand") { return .redSand }
        if has("sand") { return .sand }
        if has("gravel") { return .gravel }
        if has("clay") { return .clay }
        if has("tuff") { return .tuff }
        if any(["deepslate", "basalt", "blackstone"]) { return .deepslate }
        if has("granite") { return .granite }
        if has("diorite") { return .diorite }
        if has("andesite") { return .andesite }
        if has("dripstone") { return .dripstone }
        if has("calcite") { return .lightStone }
        if any(["cobblestone", "stone_brick"]) { return .cobblestone }
        if has("glazed") { return .terracotta }
        if has("light_gray_terracotta") { return .lightGrayTerracotta }
        if has("white_terracotta") { return .whiteTerracotta }
        if has("orange_terracotta") { return .orangeTerracotta }
        if has("yellow_terracotta") { return .yellowTerracotta }
        if has("brown_terracotta") { return .brownTerracotta }
        if has("red_terracotta") { return .redTerracotta }
        if has("terracotta") { return .terracotta }
        if has("obsidian") { return .obsidian }
        if has("brick") { return .bricks }
        if any(["ore", "stone", "bedrock", "obsidian", "furnace", "smoker", "anvil", "cauldron", "grindstone"]) { return .stone }
        if has("cherry") { return .cherryWood }
        if any(["log", "wood", "stem", "hyphae"]) { return .log }
        if any(["planks", "stairs", "slab", "crafting_table", "barrel", "bookshelf", "chest", "lectern",
                "composter", "loom", "cartography", "fletching", "smithing", "beehive", "bee_nest"]) { return .planks }
        if has("snow") { return .snow }
        if has("packed_ice") { return .packedIce }
        if has("blue_ice") { return .blueIce }
        if has("ice") { return .ice }
        if has("lava") { return .lava }
        if has("hay") { return .hay }
        if has("wool") || n.hasSuffix("_bed") { return .wool }
        if has("sculk") { return .deepslate }
        if has("copper") { return .granite }
        if has("netherrack") { return .terracotta }
        if any(["vault", "trial_spawner", "dispenser", "dropper", "observer", "piston", "repeater", "comparator",
                "note_block", "target", "redstone", "lodestone", "respawn_anchor"]) { return .stone }
        return .unknown
    }
}
