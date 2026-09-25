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

    public var kind: MaterialKind {
        switch self {
        case .air: return .air
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
        }
    }
}

public enum Materials {
    public static let kinds: [MaterialKind] = Mat.allCases.map(\.kind)
    public static let colors: [SIMD4<Float>] = Mat.allCases.map(\.color)

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
        "brewing_stand",
    ]

    public static func classify(_ fullName: String) -> Mat {
        let n = fullName.hasPrefix("minecraft:") ? String(fullName.dropFirst(10)) : fullName
        switch n {
        case "air", "cave_air", "void_air", "structure_void", "light", "barrier":
            return .air
        case "water", "bubble_column", "seagrass", "tall_seagrass", "kelp", "kelp_plant":
            return .water
        case "snow", "short_grass", "tall_grass", "grass", "fern", "large_fern", "brown_mushroom", "red_mushroom",
             "short_dry_grass", "tall_dry_grass", "azalea", "flowering_azalea", "big_dripleaf", "big_dripleaf_stem",
             "small_dripleaf", "fire", "soul_fire", "spore_blossom", "hanging_roots", "cactus_flower":
            return .air
        default:
            break
        }
        func has(_ s: String) -> Bool { n.contains(s) }
        func any(_ list: [String]) -> Bool { list.contains { n.contains($0) } }

        if has("cherry_leaves") { return .cherryLeaves }
        if has("leaves") { return .leaves }
        if has("grass_block") { return .grass }
        if any(decorative) { return .air }
        if has("moss") { return .moss }
        if has("amethyst") { return .amethyst }
        if has("magma") { return .lava }
        if has("pumpkin") || has("melon") { return .pumpkin }
        if n.hasPrefix("raw_") || n == "spawner" { return .stone }
        if has("dirt_path") { return .path }
        if has("farmland") { return .farmland }
        if has("mud") { return .mud }
        if any(["dirt", "podzol", "mycelium"]) { return .dirt }
        if has("sandstone") { return .sandstone }
        if has("sand") { return .sand }
        if has("gravel") { return .gravel }
        if has("clay") { return .clay }
        if any(["deepslate", "tuff", "basalt", "blackstone"]) { return .deepslate }
        if has("granite") { return .granite }
        if any(["diorite", "andesite", "calcite", "dripstone"]) { return .lightStone }
        if any(["cobblestone", "stone_brick"]) { return .cobblestone }
        if has("terracotta") { return .terracotta }
        if has("brick") { return .bricks }
        if any(["ore", "stone", "bedrock", "obsidian", "furnace", "smoker", "anvil", "cauldron", "grindstone"]) { return .stone }
        if has("cherry") { return .cherryWood }
        if any(["log", "wood", "stem", "hyphae"]) { return .log }
        if any(["planks", "stairs", "slab", "crafting_table", "barrel", "bookshelf", "chest", "lectern",
                "composter", "loom", "cartography", "fletching", "smithing", "beehive", "bee_nest"]) { return .planks }
        if has("snow") { return .snow }
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
