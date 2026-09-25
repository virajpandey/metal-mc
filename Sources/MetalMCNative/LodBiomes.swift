import Foundation
import MetalMCCore

// Biome tints for LOD grass, leaves and water, so far terrain matches vanilla at the seam in savannas,
// swamps, jungles, snowy biomes and colored oceans. Tinted voxels use material ids 64 + t (grass),
// 96 + t (leaves) and 128 + t (water), where t indexes `lodTints`. They downsample like any material.

struct LodTint {
    let grass: UInt32
    let foliage: UInt32
    let water: UInt32
}

/// Vanilla's biome grass/foliage/water colors (Java edition), grouped into tint classes. Index 0 is the default (plains).
let lodTints: [LodTint] = [
    LodTint(grass: 0x91BD59, foliage: 0x77AB2F, water: 0x3F76E4),   // 0 plains and default
    LodTint(grass: 0x79C05A, foliage: 0x59AE30, water: 0x3F76E4),   // 1 forest
    LodTint(grass: 0x88BB67, foliage: 0x6BA941, water: 0x3F76E4),   // 2 birch forest
    LodTint(grass: 0x507A32, foliage: 0x59AE30, water: 0x3F76E4),   // 3 dark forest
    LodTint(grass: 0x86B783, foliage: 0x68A464, water: 0x3F76E4),   // 4 taiga
    LodTint(grass: 0x80B497, foliage: 0x60A17B, water: 0x3D57D6),   // 5 snowy
    LodTint(grass: 0xBFB755, foliage: 0xAEA42A, water: 0x3F76E4),   // 6 savanna, desert
    LodTint(grass: 0x90814D, foliage: 0x9E814D, water: 0x3F76E4),   // 7 badlands
    LodTint(grass: 0x6A7039, foliage: 0x6A7039, water: 0x617B64),   // 8 swamp
    LodTint(grass: 0x6A7039, foliage: 0x8DB127, water: 0x3A7A6A),   // 9 mangrove swamp
    LodTint(grass: 0x59C93C, foliage: 0x30BB0B, water: 0x3F76E4),   // 10 jungle
    LodTint(grass: 0x83BB6D, foliage: 0x63A948, water: 0x0E4ECF),   // 11 meadow
    LodTint(grass: 0xB6DB61, foliage: 0xB6DB61, water: 0x5DB7EF),   // 12 cherry grove
    LodTint(grass: 0x8AB689, foliage: 0x6DA36B, water: 0x3F76E4),   // 13 windswept hills, stony peaks
    LodTint(grass: 0x55C93F, foliage: 0x2BBB0F, water: 0x3F76E4),   // 14 mushroom fields
    LodTint(grass: 0x91BD59, foliage: 0x77AB2F, water: 0x43D5EE),   // 15 warm ocean
    LodTint(grass: 0x91BD59, foliage: 0x77AB2F, water: 0x45ADF2),   // 16 lukewarm ocean
    LodTint(grass: 0x91BD59, foliage: 0x77AB2F, water: 0x3D57D6),   // 17 cold ocean
    LodTint(grass: 0x80B497, foliage: 0x60A17B, water: 0x3938C9),   // 18 frozen ocean and river
    LodTint(grass: 0x778272, foliage: 0x878D76, water: 0x76889D),   // 19 pale garden
]

private let lodBiomeTable: [String: UInt8] = {
    var t: [String: UInt8] = [:]
    func set(_ index: UInt8, _ names: [String]) { for n in names { t["minecraft:" + n] = index } }
    set(1, ["forest", "flower_forest"])
    set(2, ["birch_forest", "old_growth_birch_forest"])
    set(3, ["dark_forest"])
    set(4, ["taiga", "old_growth_pine_taiga", "old_growth_spruce_taiga"])
    set(5, ["snowy_plains", "snowy_taiga", "ice_spikes", "grove", "snowy_slopes", "frozen_peaks", "jagged_peaks", "snowy_beach"])
    set(6, ["savanna", "savanna_plateau", "windswept_savanna", "desert"])
    set(7, ["badlands", "eroded_badlands", "wooded_badlands"])
    set(8, ["swamp"])
    set(9, ["mangrove_swamp"])
    set(10, ["jungle", "sparse_jungle", "bamboo_jungle"])
    set(11, ["meadow"])
    set(12, ["cherry_grove"])
    set(13, ["windswept_hills", "windswept_gravelly_hills", "windswept_forest", "stony_peaks", "stony_shore"])
    set(14, ["mushroom_fields"])
    set(15, ["warm_ocean"])
    set(16, ["lukewarm_ocean", "deep_lukewarm_ocean"])
    set(17, ["cold_ocean", "deep_cold_ocean"])
    set(18, ["frozen_ocean", "deep_frozen_ocean", "frozen_river"])
    set(19, ["pale_garden"])
    return t
}()

@inline(__always) func lodTintIndex(_ biome: String) -> UInt8 { lodBiomeTable[biome] ?? 0 }

let lodGrassBase: UInt8 = 64, lodLeavesBase: UInt8 = 96, lodWaterBase: UInt8 = 128

/// Untinted texture averages (grass_block_top, oak_leaves, water_still), from tools/lod_colors.py.
let lodGrassGray: Float = 0.579, lodLeavesGray: Float = 0.565, lodWaterGray: Float = 0.695

/// Material kind for every LOD material id, including the tinted variants.
let lodKinds: [UInt8] = {
    var k = [UInt8](repeating: MaterialKind.opaque.rawValue, count: 256)
    for m in Mat.allCases { k[Int(m.rawValue)] = m.kind.rawValue }
    for t in 0..<32 { k[Int(lodWaterBase) + t] = MaterialKind.water.rawValue }
    return k
}()

@inline(__always) func lodIsWater(_ m: UInt8) -> Bool { m == Mat.water.rawValue || (m >= lodWaterBase && m < lodWaterBase + 32) }

/// Tinted variant of a block material for a biome tint class (other materials are unchanged).
@inline(__always) func lodTinted(_ m: UInt8, _ t: UInt8) -> UInt8 {
    if m == Mat.grass.rawValue { return lodGrassBase + t }
    if m == Mat.leaves.rawValue { return lodLeavesBase + t }
    if m == Mat.water.rawValue { return lodWaterBase + t }
    return m
}

func lodColorTable() -> [SIMD4<Float>] {
    var c = [SIMD4<Float>](repeating: SIMD4(0.492, 0.492, 0.492, 1), count: 256)
    for (i, color) in lodMaterialColors.enumerated() { c[i] = color }
    func rgb(_ v: UInt32, _ gray: Float) -> SIMD4<Float> {
        SIMD4(Float((v >> 16) & 255) / 255 * gray, Float((v >> 8) & 255) / 255 * gray, Float(v & 255) / 255 * gray, 1)
    }
    for (t, tint) in lodTints.enumerated() {
        c[Int(lodGrassBase) + t] = rgb(tint.grass, lodGrassGray)
        c[Int(lodLeavesBase) + t] = rgb(tint.foliage, lodLeavesGray)
        c[Int(lodWaterBase) + t] = rgb(tint.water, lodWaterGray)
    }
    return c
}
