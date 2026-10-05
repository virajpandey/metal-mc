import Foundation
import Metal
import simd

// Colored block light (METALMC_EXP=lit,coloredlight, prototype, off by default; docs/lighting-design.md, "Colored block
// light"). Lit mode's relight took block light from vanilla's lightmap: one warm white at vanilla's level. With this on,
// torches are orange, soul fire cyan, redstone red, lava orange-red, sea lanterns and end rods cool white, froglights
// their three colors, amethyst faint purple, and fire flickers.
//
// The light lives in a camera-centred volume of 1-block cells (256 x 128 x 256 blocks: 128 either side, 64 above and
// below), addressed modulo its size so it scrolls with the camera without moving data: only the 16^3 sections that enter
// it are uploaded. Per cell it holds eight light levels, one per color ("bucket": warm, fire, soul,
// red, lava, white, green, purple), 4 bits each in one 32-bit word. Each bucket spreads exactly as vanilla's block light
// does, by a flood fill: a cell's level is its own emission, or its brightest neighbor's less max(1, its light
// dampening); opaque cells hold only their own emission. So every color has vanilla's reach and falloff, light goes
// around corners and is blocked by solid blocks, the brightest bucket of a cell is vanilla's own level (the sanity
// reference below), and lights of different colors add where they overlap.
//
// The flood fill runs on the GPU, in place, a few passes a frame (METALMC_CLITER, 4), only over the 8^3 bricks that
// changed in the last frame and their 26 neighbors: in a still scene nothing runs; a new or broken torch settles in
// three or four frames. Each changed brick is then resolved into two filtered 3D textures: the light's linear color
// (each bucket's color times vanilla's brightness curve at its level; the flickering fire bucket apart), premultiplied by
// whether the cell is open (light reaches it), and the open flag itself, so the relight's trilinear sample can be divided
// by it: light is averaged over the open cells around the sample point only, and a sample half a block in front of a face
// never reaches through a 1-block wall.
//
// Block data: in the game, the mod's Java side (metalmc.light.ColoredLight) sends each loaded chunk's blocks as vanilla
// describes them (getLightDampening, getLightEmission, and a color class from the block's name) and every block change
// after; offline, ColoredLightOffline.swift reads them from region files. The relight (Lit.swift) samples the volume in
// place of vanilla's lightmap and checks it against vanilla's level in the G-buffer: never brighter than vanilla allows
// (nothing glows where vanilla is dark; a stale or leaking cell can't light a cave), and where vanilla has light the
// volume doesn't (outside it, not settled yet, a source it doesn't know), vanilla's own block light makes up the rest.
// Without METALMC_EXP=coloredlight none of it is compiled into a shader or run, and every shader is the same text as
// before.

/// METALMC_EXP=coloredlight (with lit): colored block light.
let clEnabled = litEnabled && experiments.contains("coloredlight")

/// The volume's size in cells (blocks), powers of two, multiples of 16 (sections) and 8 (bricks).
let clSizeX = 256, clSizeY = 128, clSizeZ = 256
let clBricksX = clSizeX / 8, clBricksY = clSizeY / 8, clBricksZ = clSizeZ / 8
let clBricks = clBricksX * clBricksY * clBricksZ
let clSlotsX = clSizeX / 16, clSlotsY = clSizeY / 16, clSlotsZ = clSizeZ / 16
let clSlots = clSlotsX * clSlotsY * clSlotsZ

private func clEnvInt(_ name: String, _ fallback: Int) -> Int { Int(ProcessInfo.processInfo.environment[name] ?? "") ?? fallback }
private func clEnvFloat(_ name: String, _ fallback: Float) -> Float { Float(ProcessInfo.processInfo.environment[name] ?? "") ?? fallback }

/// Blocks of the volume below the camera (METALMC_CLBELOW, 64: centred, so a cave's ceiling 40 blocks up still gets
/// colors; more puts more of the terrain under a high camera in it).
private let clBelow = max(16, min(clSizeY - 16, clEnvInt("METALMC_CLBELOW", 64)))
/// Flood-fill passes a frame (METALMC_CLITER, 4; at most 8: the bricks it runs over are those within one brick of a change).
private let clIterations = max(1, min(8, clEnvInt("METALMC_CLITER", 4)))
/// Sections uploaded a frame at most (nearest first): the first fill of the volume takes 4 frames.
private let clMaxSlotsPerFrame = max(16, clEnvInt("METALMC_CLSLOTS", 512))
/// The colored light's brightness (METALMC_CLGAIN, 1), the fire bucket's flicker (METALMC_CLFLICKER, 0.12 of its light),
/// and how many levels the volume may be off vanilla's before the sanity check steps in (METALMC_CLSLACK, 1).
private let clGain = max(0, clEnvFloat("METALMC_CLGAIN", 1))
private let clFlicker = max(0, min(1, clEnvFloat("METALMC_CLFLICKER", 0.12)))
private let clSlack = max(0, clEnvFloat("METALMC_CLSLACK", 1))

// MARK: - Colors

/// The eight light colors, in the volume's bucket order. Each is a chroma (linear sRGB) shown at a luminance: the warm
/// buckets about as bright as vanilla's torchlight (whose luminance the curve below is), the saturated ones less (a red
/// light reads as bright as a white one of more luminance).
private let clBucketSpecs: [(name: String, chroma: SIMD3<Float>, luminance: Float)] = [
    ("warm", SIMD3(1.00, 0.56, 0.25), 0.95),     // torches, lanterns, glowstone, jack o'lanterns: a candle-warm 1900 K
    ("fire", SIMD3(1.00, 0.46, 0.15), 0.95),     // fire, campfires, candles, lit furnaces: redder, and it flickers
    ("soul", SIMD3(0.18, 0.76, 1.00), 0.70),     // soul fire, soul torches and lanterns: cyan
    ("red", SIMD3(1.00, 0.07, 0.03), 0.40),      // redstone torches and ore
    ("lava", SIMD3(1.00, 0.30, 0.05), 0.80),     // lava, magma: orange-red
    ("white", SIMD3(0.78, 0.90, 1.00), 0.95),    // sea lanterns, end rods, beacons, light blocks: cool white
    ("green", SIMD3(0.36, 1.00, 0.30), 0.75),    // verdant froglights, glow lichen, sea pickles, copper torches
    ("purple", SIMD3(0.62, 0.28, 1.00), 0.55),   // amethyst, crying obsidian, portals, pearlescent froglights
]
let clFireBucket = 1

/// The buckets' colors: chroma scaled to its luminance.
let clBucketColors: [SIMD3<Float>] = clBucketSpecs.map { s in
    let y = s.chroma.x * 0.2126 + s.chroma.y * 0.7152 + s.chroma.z * 0.0722
    return s.chroma * (s.luminance / y)
}

/// Vanilla's block light by level in linear light: its lightmap (lightmap.fsh at the default brightness, no sky light),
/// decoded from sRGB, as luminance. The relight's block light without colors was the lightmap itself, so the colored
/// buckets use the same curve, and vanilla's level in the G-buffer is compared with the volume's on it.
let clCurve: [Float] = (0..<16).map { l in
    func br(_ x: Float) -> Float { x / (4 - 3 * x) }
    func decode(_ c: Float) -> Float { c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4) }
    let bl = Float(l) / 15
    let m = 0.9 * (2 * bl - 1) * (2 * bl - 1)
    var c = (SIMD3<Float>(1.0, 0.85, 0.65) * (1 - m) + SIMD3<Float>(repeating: 1) * m) * br(bl)
    c = simd_clamp(c, .zero, SIMD3(repeating: 1))
    let mx = max(c.x, max(c.y, c.z)), inv = 1 - mx
    let ng = mx > 0 ? c * ((1 - inv * inv * inv * inv) / mx) : c
    c = c * 0.5 + ng * 0.5
    return decode(c.x) * 0.2126 + decode(c.y) * 0.7152 + decode(c.z) * 0.0722
}

// MARK: - Emitter classes

/// What color a light-emitting block gives off. The class only matters for blocks that emit (vanilla's getLightEmission
/// says how much); each maps to a bucket, and some to a second bucket a few levels dimmer, which tints them (pink, teal,
/// yellow) without the buckets they share getting a color of their own.
enum ClClass: UInt8, CaseIterable {
    case none = 0, unknown, torch, lantern, fire, campfire, candle, furnace, soul, redstone, lava, magma, glowstone,
         shroomlight, seaLantern, endRod, beacon, ochreFroglight, verdantFroglight, pearlFroglight, amethyst,
         cryingObsidian, portal, glowLichen, glowBerries, seaPickle, copper, sculk, enchant, jackOLantern, lightBlock,
         redstoneLamp, copperBulb, conduit, endPortal, trial, firefly, brewing

    /// (bucket, second bucket or nil, how many levels below the first the second starts).
    var buckets: (Int, Int?, Int) {
        switch self {
        case .none, .unknown, .torch, .lantern, .glowstone, .jackOLantern, .brewing, .redstoneLamp, .copperBulb: return (0, nil, 0)
        case .glowBerries: return (0, 6, 4)
        case .shroomlight: return (0, 4, 3)
        case .ochreFroglight: return (0, 6, 4)
        case .fire, .campfire, .candle, .furnace, .trial: return (1, nil, 0)
        case .soul, .sculk: return (2, nil, 0)
        case .conduit: return (2, 5, 2)
        case .redstone: return (3, nil, 0)
        case .lava, .magma: return (4, nil, 0)
        case .seaLantern: return (5, 2, 4)
        case .endRod, .lightBlock, .beacon: return (5, nil, 0)
        case .verdantFroglight, .copper: return (6, nil, 0)
        case .glowLichen: return (6, 2, 1)
        case .seaPickle: return (6, 2, 2)
        case .firefly: return (6, 0, 3)
        case .pearlFroglight: return (7, 3, 3)
        case .amethyst, .cryingObsidian, .portal, .enchant, .endPortal: return (7, nil, 0)
        }
    }
}

/// The color class of a block, by its registry name ("minecraft:soul_torch"). Called once per block by the Java side.
func clClassify(_ fullName: String) -> ClClass {
    let n = fullName.hasPrefix("minecraft:") ? String(fullName.dropFirst(10)) : fullName
    if n.hasPrefix("soul_") { return .soul }   // soul torches, lanterns, fire, campfires (soul sand and soil don't emit)
    if n.contains("redstone_torch") || n.contains("redstone_wall_torch") || n.hasSuffix("redstone_ore") { return .redstone }
    if n == "redstone_lamp" { return .redstoneLamp }
    if n.contains("copper_bulb") { return .copperBulb }
    if n.contains("copper_torch") || n.contains("copper_wall_torch") || n.contains("copper_lantern") { return .copper }
    switch n {
    case "torch", "wall_torch": return .torch
    case "lantern": return .lantern
    case "fire": return .fire
    case "campfire": return .campfire
    case "furnace", "blast_furnace", "smoker": return .furnace
    case "magma_block": return .magma
    case "glowstone": return .glowstone
    case "shroomlight": return .shroomlight
    case "sea_lantern": return .seaLantern
    case "end_rod": return .endRod
    case "beacon": return .beacon
    case "ochre_froglight": return .ochreFroglight
    case "verdant_froglight": return .verdantFroglight
    case "pearlescent_froglight": return .pearlFroglight
    case "crying_obsidian": return .cryingObsidian
    case "nether_portal": return .portal
    case "glow_lichen": return .glowLichen
    case "sea_pickle": return .seaPickle
    case "enchanting_table", "ender_chest", "end_portal_frame", "dragon_egg", "respawn_anchor": return .enchant
    case "jack_o_lantern": return .jackOLantern
    case "light": return .lightBlock
    case "conduit": return .conduit
    case "end_portal", "end_gateway": return .endPortal
    case "trial_spawner", "vault": return .trial
    case "firefly_bush": return .firefly
    case "brewing_stand", "brown_mushroom": return .brewing
    default: break
    }
    if n.contains("lava") { return .lava }
    if n.hasSuffix("candle") || n.hasSuffix("candle_cake") { return .candle }
    if n.contains("amethyst") { return .amethyst }
    if n.hasPrefix("cave_vines") { return .glowBerries }
    if n.hasPrefix("sculk") || n.hasPrefix("calibrated_sculk") { return .sculk }
    return .unknown
}

/// A block's cell code: light dampening (bits 0-3, vanilla's 0-15; 15 is opaque), emission (4-7) and color class (8-13).
@inline(__always) func clCode(dampening: Int, emission: Int, cls: ClClass) -> UInt16 {
    UInt16(min(15, max(0, dampening))) | UInt16(min(15, max(0, emission))) << 4 | (emission > 0 ? UInt16(cls.rawValue & 63) << 8 : 0)
}

/// The GPU's class table: bucket (bits 0-3), second bucket (4-7, 15 for none), the second's levels below (8-11).
private let clClassTable: [UInt32] = (0..<64).map { i in
    guard let c = ClClass(rawValue: UInt8(i)) else { return 15 << 4 }
    let (a, b, d) = c.buckets
    return UInt32(a) | UInt32(b ?? 15) << 4 | UInt32(d) << 8
}

// MARK: - The store (the blocks around the player, by section)

/// One section's codes (4096, laid out y, z, x), or one code for all of it.
struct ClSection {
    var codes: [UInt16]?
    var uniform: UInt16
    var version: UInt32
}

@inline(__always) func clSectionKey(_ sx: Int, _ sy: Int, _ sz: Int) -> Int64 {
    (Int64(sx) & 0x3FFFFF) << 42 | (Int64(sz) & 0x3FFFFF) << 20 | (Int64(sy) & 0xFFFFF)
}

@inline(__always) private func floorDiv(_ a: Int, _ b: Int) -> Int { a >= 0 ? a / b : -((-a + b - 1) / b) }

/// Sections of the world the volume may need: written by the Java side's worker thread (chunks) and render thread
/// (block changes), or offline; read by the render thread when it fills the volume.
final class ClStore: @unchecked Sendable {
    let lock = NSLock()
    private(set) var sections: [Int64: ClSection] = [:]
    /// Chunk columns' sections (to replace a whole column when a chunk arrives again).
    private var columns: [Int64: [Int]] = [:]
    private var nextVersion: UInt32 = 1
    /// Bumped by every change: the volume rechecks its sections only when it moved.
    private(set) var stamp: UInt64 = 0
    /// Block changes since the last frame: (x, y, z, code).
    var edits: [(Int, Int, Int, UInt16)] = []
    var generation: Int32 = 0

    /// Section y range of the world (the overworld's -64..319).
    static let minSectionY = -4, maxSectionY = 19

    func reset(generation g: Int32) {
        lock.lock(); defer { lock.unlock() }
        sections.removeAll()
        columns.removeAll()
        edits.removeAll()
        generation = g
        stamp &+= 1
    }

    /// Replaces chunk (cx, cz): `list` holds its non-air sections (section y, 4096 codes y, z, x). Sections not in it are
    /// air.
    func putChunk(cx: Int, cz: Int, _ list: [(Int, [UInt16])]) {
        lock.lock(); defer { lock.unlock() }
        let ck = clSectionKey(cx, 0, cz)
        for sy in columns[ck] ?? [] { sections[clSectionKey(cx, sy, cz)] = nil }
        var ys: [Int] = []
        for (sy, codes) in list {
            let first = codes[0]
            let uniform = !codes.contains { $0 != first }
            if uniform && first == 0 { continue }
            sections[clSectionKey(cx, sy, cz)] = ClSection(codes: uniform ? nil : codes, uniform: first, version: nextVersion)
            nextVersion &+= 1
            ys.append(sy)
        }
        // Sections that were there and aren't now become air: a new version for them too (a missing section is version 0).
        columns[ck] = ys
        stamp &+= 1
    }

    /// One block changed (the section's version stays, and a section made by the change keeps the version of the air it
    /// was: the volume applies the change in place, without resetting the section's light).
    func edit(x: Int, y: Int, z: Int, code: UInt16) {
        lock.lock(); defer { lock.unlock() }
        let sx = floorDiv(x, 16), sy = floorDiv(y, 16), sz = floorDiv(z, 16)
        let k = clSectionKey(sx, sy, sz)
        var s = sections[k] ?? ClSection(codes: nil, uniform: 0, version: 0)
        if sections[k] == nil {
            let ck = clSectionKey(sx, 0, sz)
            if !(columns[ck] ?? []).contains(sy) { columns[ck, default: []].append(sy) }
        }
        var codes = s.codes ?? [UInt16](repeating: s.uniform, count: 4096)
        codes[((y & 15) * 16 + (z & 15)) * 16 + (x & 15)] = code
        s.codes = codes
        sections[k] = s
        edits.append((x, y, z, code))
        stamp &+= 1
    }

    /// Section (sx, sy, sz) as stored: missing ones are air above the world's bottom and solid below it.
    func section(_ sx: Int, _ sy: Int, _ sz: Int) -> ClSection {
        if sy < ClStore.minSectionY { return ClSection(codes: nil, uniform: 15, version: 0) }
        return sections[clSectionKey(sx, sy, sz)] ?? ClSection(codes: nil, uniform: 0, version: 0)
    }

    var sectionCount: Int { lock.lock(); defer { lock.unlock() }; return sections.count }
}

// MARK: - GPU

/// Mirrors ClParams in the kernels.
private struct ClParamsGPU {
    var orgMod = SIMD4<Int32>.zero   // xyz: the volume's min corner, modulo its size (toroidal cell coordinates)
    var stamp = SIMD4<UInt32>.zero   // x: this pass's stamp, y: listing threshold, z: resolve threshold, w: count
}

/// Mirrors ClSlot: a section to upload (tex: its first cell, toroidal; info: mode 0 codes / 1 uniform / 2 stamp only,
/// offset of its codes in the staging buffer, the uniform code).
private struct ClSlotGPU {
    var tex = SIMD4<Int32>.zero
    var info = SIMD4<UInt32>.zero
}

/// Mirrors ClFrame in the relight's header.
struct ClFrameGPU {
    var camTex = SIMD4<Float>.zero   // xyz: the camera in the volume's (toroidal) texture space, blocks; w: 1 if valid
    var camVol = SIMD4<Float>.zero   // xyz: the camera relative to the volume's min corner, blocks; w: fade width at its edges
    var size = SIMD4<Float>.zero     // xyz: the volume's size, cells; w: time (s) for the flicker
    var tune = SIMD4<Float>.zero     // x: gain, y: flicker amplitude, z: sanity slack (levels), w: 1 to fill from vanilla
    var fire = SIMD4<Float>.zero     // rgb: the fire bucket's color
    var curve0 = SIMD4<Float>.zero, curve1 = SIMD4<Float>.zero, curve2 = SIMD4<Float>.zero, curve3 = SIMD4<Float>.zero
}

/// The volume's kernels.
let clKernelSource = """
#include <metal_stdlib>
using namespace metal;

#define CL_SX \(clSizeX)
#define CL_SY \(clSizeY)
#define CL_SZ \(clSizeZ)
#define CL_BX \(clBricksX)
#define CL_BY \(clBricksY)
#define CL_BZ \(clBricksZ)
#define CL_BRICKS \(clBricks)
#define CL_FIRE \(clFireBucket)u

struct ClParams {
    int4 orgMod;     // xyz: the volume's min corner in its own (toroidal) cell coordinates
    uint4 stamp;     // x: this pass's stamp, y: listing threshold, z: resolve threshold, w: count (slots, edits)
};
struct ClSlot { int4 tex; uint4 info; };

// Cells are stored brick by brick (8 x 8 x 8, 2 KB of light), so a brick's pass reads and writes whole lines.
static uint clBrickIndex(uint3 b) { return (b.z * CL_BY + b.y) * CL_BX + b.x; }
static uint3 clBrickCoord(uint b) { return uint3(b % CL_BX, (b / CL_BX) % CL_BY, b / (CL_BX * CL_BY)); }
static uint clIndex(uint3 u) { return clBrickIndex(u >> 3) * 512u + ((u.z & 7u) * 8u + (u.y & 7u)) * 8u + (u.x & 7u); }
static uint3 clWrap(int3 u) { return uint3(u & int3(CL_SX - 1, CL_SY - 1, CL_SZ - 1)); }

// A cell's own light: its emission level in its class's bucket, and in the second bucket some levels less.
static uint clEmission(uint code, constant uint* table) {
    uint level = (code >> 4) & 15u;
    if (level == 0u) return 0u;
    uint e = table[(code >> 8) & 63u];
    uint a = e & 7u, b = (e >> 4) & 15u, d = (e >> 8) & 15u;
    uint out = level << (4u * a);
    if (b < 8u && level > d) out |= (level - d) << (4u * b);
    return out;
}

// Sections into the volume: their codes (or one code for all), the light reset to each cell's own emission (the cells
// held another part of the world, or nothing), and their bricks stamped so the flood fill runs over them. One thread a
// cell; mode 2 only stamps (bricks next to the volume's new edge after it moved).
kernel void cl_upload(device const ushort* staging [[buffer(0)]], device const ClSlot* slots [[buffer(1)]],
                      device ushort* blocks [[buffer(2)]], device uint* light [[buffer(3)]], device uint* changed [[buffer(4)]],
                      constant ClParams& p [[buffer(5)]], constant uint* table [[buffer(6)]],
                      uint3 gid [[thread_position_in_grid]]) {
    uint s = gid.z >> 4;
    if (s >= p.stamp.w) return;
    ClSlot slot = slots[s];
    uint3 l = uint3(gid.x, gid.y, gid.z & 15u);
    uint3 u = uint3(slot.tex.xyz) + l;
    if (all((l & 7u) == 0u)) changed[clBrickIndex(u >> 3)] = p.stamp.x;
    if (slot.info.x == 2u) return;
    uint code = slot.info.x == 1u ? slot.info.z : uint(staging[slot.info.y + (l.y * 16u + l.z) * 16u + l.x]);
    uint i = clIndex(u);
    blocks[i] = ushort(code);
    light[i] = clEmission(code, table);
}

// Single block changes: the cell's code, its brick stamped. Its light is left for the flood fill (a light that was
// removed fades a level a pass; one placed starts at its emission).
kernel void cl_edit(device const uint4* edits [[buffer(0)]], device ushort* blocks [[buffer(2)]], device uint* light [[buffer(3)]],
                    device uint* changed [[buffer(4)]], constant ClParams& p [[buffer(5)]], constant uint* table [[buffer(6)]],
                    uint t [[thread_position_in_grid]]) {
    if (t >= p.stamp.w) return;
    uint4 e = edits[t];
    uint i = clIndex(e.xyz);
    blocks[i] = ushort(e.w);
    if ((e.w & 15u) >= 15u) light[i] = clEmission(e.w, table);   // now opaque: dark at once but for its own light
    changed[clBrickIndex(e.xyz >> 3)] = p.stamp.x;
}

kernel void cl_list_begin(device uint* args [[buffer(8)]]) {
    args[0] = 0u; args[1] = 1u; args[2] = 1u;
}

// The bricks the flood fill runs over this frame: those whose brick or any of its 26 neighbors changed since the
// threshold (a frame's passes move light at most 8 cells, one brick). Indirect dispatch arguments in args.
kernel void cl_list(device const uint* changed [[buffer(4)]], constant ClParams& p [[buffer(5)]], device uint* list [[buffer(7)]],
                    device atomic_uint* args [[buffer(8)]], uint b [[thread_position_in_grid]]) {
    if (b >= uint(CL_BRICKS)) return;
    int3 c = int3(clBrickCoord(b));
    uint m = 0u;
    for (int dz = -1; dz <= 1; dz++) {
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uint3 n = uint3((c + int3(dx, dy, dz)) & int3(CL_BX - 1, CL_BY - 1, CL_BZ - 1));
                m = max(m, changed[clBrickIndex(n)]);
            }
        }
    }
    if (m >= p.stamp.y) list[atomic_fetch_add_explicit(&args[0], 1u, memory_order_relaxed)] = b;
}

// One flood-fill pass over a listed brick, in place (a threadgroup a brick, a thread a cell): per bucket, the brightest
// of the six neighbors inside the volume less max(1, the cell's dampening), or the cell's own emission if brighter;
// opaque cells hold only their own. The eight 4-bit levels are handled as two sets of four bytes. A brick whose cells
// changed is stamped.
kernel void cl_propagate(device const ushort* blocks [[buffer(2)]], device uint* light [[buffer(3)]], device uint* changed [[buffer(4)]],
                         constant ClParams& p [[buffer(5)]], constant uint* table [[buffer(6)]], device const uint* list [[buffer(7)]],
                         uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]],
                         uint li [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint tgChanged;
    if (li == 0u) atomic_store_explicit(&tgChanged, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint b = list[tg.x];
    uint3 u = clBrickCoord(b) * 8u + lid;
    uint i = clIndex(u);
    uint code = uint(blocks[i]);
    uint dec = max(1u, code & 15u);
    uint em = clEmission(code, table);
    uint cur = light[i];
    uint L = em;
    if (dec < 15u) {
        // The cell relative to the volume's min corner: neighbors past the volume's edge are dark (the other side of the
        // toroidal addressing is another part of the world).
        int3 r = (int3(u) - p.orgMod.xyz) & int3(CL_SX - 1, CL_SY - 1, CL_SZ - 1);
        uchar4 lo = uchar4(0), hi = uchar4(0);
        if (r.x > 0) { uint v = light[clIndex(clWrap(int3(u) + int3(-1, 0, 0)))]; lo = max(lo, as_type<uchar4>(v & 0x0F0F0F0Fu)); hi = max(hi, as_type<uchar4>((v >> 4) & 0x0F0F0F0Fu)); }
        if (r.x < CL_SX - 1) { uint v = light[clIndex(clWrap(int3(u) + int3(1, 0, 0)))]; lo = max(lo, as_type<uchar4>(v & 0x0F0F0F0Fu)); hi = max(hi, as_type<uchar4>((v >> 4) & 0x0F0F0F0Fu)); }
        if (r.y > 0) { uint v = light[clIndex(clWrap(int3(u) + int3(0, -1, 0)))]; lo = max(lo, as_type<uchar4>(v & 0x0F0F0F0Fu)); hi = max(hi, as_type<uchar4>((v >> 4) & 0x0F0F0F0Fu)); }
        if (r.y < CL_SY - 1) { uint v = light[clIndex(clWrap(int3(u) + int3(0, 1, 0)))]; lo = max(lo, as_type<uchar4>(v & 0x0F0F0F0Fu)); hi = max(hi, as_type<uchar4>((v >> 4) & 0x0F0F0F0Fu)); }
        if (r.z > 0) { uint v = light[clIndex(clWrap(int3(u) + int3(0, 0, -1)))]; lo = max(lo, as_type<uchar4>(v & 0x0F0F0F0Fu)); hi = max(hi, as_type<uchar4>((v >> 4) & 0x0F0F0F0Fu)); }
        if (r.z < CL_SZ - 1) { uint v = light[clIndex(clWrap(int3(u) + int3(0, 0, 1)))]; lo = max(lo, as_type<uchar4>(v & 0x0F0F0F0Fu)); hi = max(hi, as_type<uchar4>((v >> 4) & 0x0F0F0F0Fu)); }
        uchar4 d = uchar4(uchar(dec));
        lo = max(subsat(lo, d), as_type<uchar4>(em & 0x0F0F0F0Fu));
        hi = max(subsat(hi, d), as_type<uchar4>((em >> 4) & 0x0F0F0F0Fu));
        L = as_type<uint>(lo) | (as_type<uint>(hi) << 4);
    }
    if (L != cur) {
        light[i] = L;
        atomic_store_explicit(&tgChanged, 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (li == 0u && atomic_load_explicit(&tgChanged, memory_order_relaxed) != 0u) changed[b] = p.stamp.x;
}

// The listed bricks that changed this frame into the textures the relight samples: per cell the light's color without
// the fire bucket (each bucket's color times the curve at its level) and whether the cell is open, the fire bucket's
// light and vanilla's equivalent (the curve at the brightest bucket's level), all premultiplied by open.
kernel void cl_resolve(device const ushort* blocks [[buffer(2)]], device const uint* light [[buffer(3)]],
                       device const uint* changed [[buffer(4)]], constant ClParams& p [[buffer(5)]],
                       device const uint* list [[buffer(7)]], constant float4* color [[buffer(9)]], constant float* curve [[buffer(10)]],
                       texture3d<half, access::write> rgbOut [[texture(0)]], texture3d<half, access::write> auxOut [[texture(1)]],
                       uint3 tg [[threadgroup_position_in_grid]], uint3 lid [[thread_position_in_threadgroup]]) {
    uint b = list[tg.x];
    if (changed[b] < p.stamp.z) return;
    uint3 u = clBrickCoord(b) * 8u + lid;
    uint i = clIndex(u);
    uint L = light[i];
    float open = (uint(blocks[i]) & 15u) < 15u ? 1.0 : 0.0;
    float3 rgb = float3(0.0);
    float fire = 0.0;
    uint top = 0u;
    for (uint k = 0u; k < 8u; k++) {
        uint l = (L >> (4u * k)) & 15u;
        top = max(top, l);
        float c = curve[l];
        if (k == CL_FIRE) fire = c; else rgb += color[k].rgb * c;
    }
    rgbOut.write(half4(half3(rgb * open), half(open)), u);
    auxOut.write(half4(half(fire * open), half(curve[top] * open), 0.0h, 0.0h), u);
}

kernel void cl_clear(texture3d<half, access::write> a [[texture(0)]], texture3d<half, access::write> b [[texture(1)]],
                     uint3 gid [[thread_position_in_grid]]) {
    if (any(gid >= uint3(CL_SX, CL_SY, CL_SZ))) return;
    a.write(half4(0.0h), gid);
    b.write(half4(0.0h), gid);
}
"""

/// The relight's part (spliced into Lit.swift's litRelightHeader with clEnabled only): the frame's parameters and the
/// colored block light of a pixel.
let clRelightHeader = """

// Colored block light (METALMC_EXP=coloredlight, ColoredLight.swift): the light volume's frame.
struct ClFrame {
    float4 camTex;   // xyz: the camera in the volume's (toroidal) texture space, blocks; w: 1 if the volume is valid
    float4 camVol;   // xyz: the camera relative to the volume's min corner, blocks; w: fade width at its edges
    float4 size;     // xyz: the volume's size, cells; w: time (s) for the flicker
    float4 tune;     // x: gain, y: flicker amplitude, z: sanity slack (levels), w: 1 to fill from vanilla
    float4 fire;     // rgb: the fire bucket's color
    float4 curve[4]; // vanilla's block light (linear luminance) at levels 0-15
};

// The curve at a fractional level.
static float clCurve(constant ClFrame& c, float level) {
    float l = clamp(level, 0.0, 15.0);
    uint i = min(uint(l), 14u);
    float a = c.curve[i >> 2][i & 3u], b = c.curve[(i + 1u) >> 2][(i + 1u) & 3u];
    return mix(a, b, l - float(i));
}

// Fire's flicker: two octaves of smoothed noise in time.
static float clHash(float x) {
    uint h = uint(int(x)) * 0x9E3779B9u;
    h ^= h >> 15; h *= 0x85EBCA6Bu; h ^= h >> 13;
    return float(h & 0xFFFFu) / 65535.0;
}
static float clNoise(float x) {
    float i = floor(x), f = x - i;
    return mix(clHash(i), clHash(i + 1.0), f * f * (3.0 - 2.0 * f));
}

// The pixel's block light, colored (linear, per unit of albedo, before AO). vanilla: the lightmap's block light at the
// pixel's level (with its ambient), what the relight had without colors; ambient: the lightmap at level 0; level:
// vanilla's level from the G-buffer (smooth lighting's, fractional); rel: the pixel's position relative to the camera; fi,
// n: its face. The volume is sampled half a block in front of the face (in the cell the face looks into; plants in their
// own), trilinear, divided by the share of open cells, so solid cells don't count and a 1-block wall's far side never
// does. The sanity check against vanilla's level: the volume's brightest bucket (vanilla's equivalent) may not pass the
// curve at vanilla's level plus the slack (else it's scaled down: a leak or a stale cell can't light a cave), and where it
// falls short of the curve at vanilla's level less the slack (outside the volume, not settled, a source it doesn't know),
// vanilla's own block light makes up the share it misses. Within 16 blocks of the volume's edge it fades to vanilla's.
// dbg (debug view 12): red where the check scaled it down, green where colored light applied, blue the share vanilla
// made up, yellow outside the volume, gray nothing open around. Where vanilla's level is 0 the check would leave at most
// the curve at the slack (level 1: 0.3% of full light), so the volume isn't sampled there (most terrain by day), but in
// the debug views (view: the relight's), which show what the check removed.
static float3 clBlockLight(float3 vanilla, float3 ambient, float level, float3 rel, uint fi, float3 n, constant ClFrame& c,
                           texture3d<float> rgbVol, texture3d<float> auxVol, thread float4& dbg, uint view) {
    if (c.camTex.w <= 0.0 || (level < 0.01 && view == 0u)) return vanilla;
    float3 off = fi < 6u ? n * 0.5 : float3(0.0);
    float3 q = c.camVol.xyz + rel + off;
    float3 e = min(q, c.size.xyz - q);
    float w = saturate((min(e.x, min(e.y, e.z)) - 1.0) / max(c.camVol.w, 1.0));
    if (w <= 0.0) { dbg = float4(0.7, 0.7, 0.0, 1.0); return vanilla; }
    constexpr sampler s(filter::linear, address::repeat);
    float3 uvw = (c.camTex.xyz + rel + off) / c.size.xyz;
    float4 a = rgbVol.sample(s, uvw);
    float2 x = auxVol.sample(s, uvw).xy;
    if (a.w < 0.02) { dbg = float4(0.35, 0.35, 0.35, 1.0); return vanilla; }
    float inv = 1.0 / a.w;
    float flicker = 1.0 + c.tune.y * ((clNoise(c.size.w * 7.0) - 0.5) * 1.2 + (clNoise(c.size.w * 1.9 + 31.0) - 0.5) * 0.8);
    float3 rgb = (a.rgb + c.fire.rgb * (x.x * flicker)) * inv;
    float m = x.y * inv;
    float hi = clCurve(c, level + c.tune.z), lo = clCurve(c, level - c.tune.z);
    float k = m > hi ? hi / max(m, 1e-6) : 1.0;
    float fill = c.tune.w > 0.0 && lo > 1e-5 ? saturate(1.0 - m * k / lo) : 0.0;
    float3 colored = ambient + rgb * (k * c.tune.x) + max(vanilla - ambient, float3(0.0)) * fill;
    dbg = float4(k < 0.98 ? 0.9 : 0.0, m * k > 1e-4 ? 0.8 : 0.0, fill * 0.9, 1.0);
    return mix(vanilla, colored, w);
}
"""

final class ColoredLight: @unchecked Sendable {
    static let shared = ColoredLight()

    let store = ClStore()

    private var library: MTLLibrary?
    private var failed = false
    private var uploadPipe: MTLComputePipelineState?
    private var editPipe: MTLComputePipelineState?
    private var listBeginPipe: MTLComputePipelineState?
    private var listPipe: MTLComputePipelineState?
    private var propagatePipe: MTLComputePipelineState?
    private var resolvePipe: MTLComputePipelineState?
    private var clearPipe: MTLComputePipelineState?
    private var blocks: MTLBuffer?
    private var light: MTLBuffer?
    private var changed: MTLBuffer?
    private var list: MTLBuffer?
    private var args: MTLBuffer?
    private var colorBuffer: MTLBuffer?
    private var curveBuffer: MTLBuffer?
    private(set) var rgb: MTLTexture?
    private(set) var aux: MTLTexture?
    private var dummy3D: MTLTexture?
    private var cleared = false

    /// The volume's min corner (world blocks), and what each slot (section of the volume) holds: its section key and the
    /// store's version of it then (Int64.min: nothing yet).
    private var org = SIMD3<Int>(Int.min, Int.min, Int.min)
    private var slotKey = [Int64](repeating: Int64.min, count: clSlots)
    private var slotVersion = [UInt32](repeating: 0, count: clSlots)
    private var checkedStamp: UInt64 = .max
    private var pendingSlots = true
    /// The flood fill's stamps: each frame stamps its uploads with `stamp`, its passes with the next ones.
    private var stamp: UInt32 = 1
    private var lastFrameFirst: UInt32 = 0
    /// This frame's relight parameters, and when they were made (they hold for 100 ms: the relight checks).
    private var frameParams = ClFrameGPU()
    private var frameTime: UInt64 = 0
    private var frames = 0
    private var stats = (uploads: 0, edits: 0, framesWithWork: 0, bricks: 0)
    /// Bricks the last listing ran the flood fill over (read back late; offline after a wait it's exact).
    private(set) var lastListed = 0
    /// Offline: the shading switch (the volume keeps running), and the clock for the flicker (below 0: the system's).
    var shadingOn = true
    var debugTime: Double = -1
    /// Offline timing: encode the volume's work into its own command buffer (committed by the caller).
    var debugCB: MTLCommandBuffer?

    private func ensure() -> Bool {
        if failed { return false }
        if library != nil { return true }
        do {
            let lib = try ctx.device.makeLibrary(source: clKernelSource, options: nil)
            func pipe(_ name: String) throws -> MTLComputePipelineState {
                try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
            }
            uploadPipe = try pipe("cl_upload")
            editPipe = try pipe("cl_edit")
            listBeginPipe = try pipe("cl_list_begin")
            listPipe = try pipe("cl_list")
            propagatePipe = try pipe("cl_propagate")
            resolvePipe = try pipe("cl_resolve")
            clearPipe = try pipe("cl_clear")
            library = lib
        } catch {
            log("coloredlight: shaders failed: \(error)")
            failed = true
            return false
        }
        let cells = clSizeX * clSizeY * clSizeZ
        blocks = ctx.device.makeBuffer(length: cells * 2, options: .storageModePrivate)
        light = ctx.device.makeBuffer(length: cells * 4, options: .storageModePrivate)
        changed = ctx.device.makeBuffer(length: clBricks * 4, options: .storageModePrivate)
        list = ctx.device.makeBuffer(length: clBricks * 4, options: .storageModePrivate)
        // Shared: the CPU reads the last listing's count for the stats (a frame or two late).
        args = ctx.device.makeBuffer(length: 16, options: .storageModeShared)
        var colors = clBucketColors.map { SIMD4<Float>($0, 0) }
        colorBuffer = ctx.device.makeBuffer(bytes: &colors, length: colors.count * 16, options: .storageModeShared)
        var curve = clCurve
        curveBuffer = ctx.device.makeBuffer(bytes: &curve, length: curve.count * 4, options: .storageModeShared)
        func tex(_ format: MTLPixelFormat, _ label: String) -> MTLTexture? {
            let d = MTLTextureDescriptor()
            d.textureType = .type3D
            d.pixelFormat = format
            d.width = clSizeX; d.height = clSizeY; d.depth = clSizeZ
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            let t = ctx.device.makeTexture(descriptor: d)
            t?.label = label
            return t
        }
        rgb = tex(.rgba16Float, "MetalMC colored light")
        aux = tex(.rg16Float, "MetalMC colored light (fire, vanilla's equivalent)")
        let dd = MTLTextureDescriptor()
        dd.textureType = .type3D
        dd.pixelFormat = .rgba16Float
        dd.width = 1; dd.height = 1; dd.depth = 1
        dd.usage = .shaderRead
        dummy3D = ctx.device.makeTexture(descriptor: dd)
        guard blocks != nil, light != nil, changed != nil, list != nil, args != nil, rgb != nil, aux != nil, dummy3D != nil else {
            log("coloredlight: couldn't allocate the volume")
            failed = true
            return false
        }
        log("coloredlight: volume \(clSizeX) x \(clSizeY) x \(clSizeZ), \((cells * 18 + clBricks * 8) / 1_000_000) MB, \(clIterations) passes a frame")
        return true
    }

    /// Forgets what the volume holds (a new world): every section is uploaded again.
    func invalidate() {
        org = SIMD3(Int.min, Int.min, Int.min)
        for i in 0..<clSlots { slotKey[i] = Int64.min; slotVersion[i] = 0 }
        checkedStamp = .max
        pendingSlots = true
    }

    /// The frame's volume work, before the relight: the volume follows the camera (world position, doubles), takes the
    /// sections that entered it or changed and the blocks that changed, and runs its flood-fill passes and resolve.
    func frame(camera: SIMD3<Double>) {
        guard clEnabled, ctx.pass == nil, ensure(), let uploadPipe, let editPipe, let listBeginPipe, let listPipe,
              let propagatePipe, let resolvePipe, let clearPipe, let blocks, let light, let changed, let rgb, let aux else { return }
        frames += 1
        // The volume's corner: whole sections, the camera near its middle (moved when the camera is 12 blocks off it).
        let cam = SIMD3<Int>(Int(camera.x.rounded(.down)), Int(camera.y.rounded(.down)), Int(camera.z.rounded(.down)))
        let wantY = min(max(floorDiv(cam.y - clBelow + 8, 16) * 16, ClStore.minSectionY * 16 - 16), (ClStore.maxSectionY + 1) * 16 + 16 - clSizeY)
        var newOrg = org
        if org.x == Int.min || abs(cam.x - (org.x + clSizeX / 2)) > 12 { newOrg.x = floorDiv(cam.x - clSizeX / 2 + 8, 16) * 16 }
        if org.z == Int.min || abs(cam.z - (org.z + clSizeZ / 2)) > 12 { newOrg.z = floorDiv(cam.z - clSizeZ / 2 + 8, 16) * 16 }
        if org.y == Int.min || abs(cam.y - (org.y + clBelow)) > 8 { newOrg.y = wantY }
        let moved = newOrg != org
        let oldOrg = org
        org = newOrg

        // Sections to upload: every slot whose section changed (it moved, or the store has a newer version), nearest first.
        struct Up { let slot: Int; let sx: Int; let sy: Int; let sz: Int; let d: Int }
        var ups: [Up] = []
        var codesOf: [Int: ClSection] = [:]
        var edits: [SIMD4<UInt32>] = []
        store.lock.lock()
        let storeStamp = store.stamp
        store.lock.unlock()
        if moved || pendingSlots || storeStamp != checkedStamp {
            store.lock.lock()
            let s0 = SIMD3(floorDiv(org.x, 16), floorDiv(org.y, 16), floorDiv(org.z, 16))
            let cs = SIMD3(floorDiv(cam.x, 16), floorDiv(cam.y, 16), floorDiv(cam.z, 16))
            for k in 0..<clSlotsZ {
                for j in 0..<clSlotsY {
                    for i in 0..<clSlotsX {
                        // The section of the volume in slot (i, j, k): the one in range whose coordinates are i, j, k modulo
                        // the slots.
                        let sx = s0.x + ((i - s0.x) % clSlotsX + clSlotsX) % clSlotsX
                        let sy = s0.y + ((j - s0.y) % clSlotsY + clSlotsY) % clSlotsY
                        let sz = s0.z + ((k - s0.z) % clSlotsZ + clSlotsZ) % clSlotsZ
                        let slot = (k * clSlotsY + j) * clSlotsX + i
                        let key = clSectionKey(sx, sy, sz)
                        let sec = store.section(sx, sy, sz)
                        if slotKey[slot] == key && slotVersion[slot] == sec.version { continue }
                        let d = (sx - cs.x) * (sx - cs.x) + (sy - cs.y) * (sy - cs.y) + (sz - cs.z) * (sz - cs.z)
                        ups.append(Up(slot: slot, sx: sx, sy: sy, sz: sz, d: d))
                        codesOf[slot] = sec
                    }
                }
            }
            let pendingEdits = store.edits
            store.edits.removeAll()
            store.lock.unlock()
            checkedStamp = storeStamp
            ups.sort { $0.d < $1.d }
            pendingSlots = ups.count > clMaxSlotsPerFrame
            if ups.count > clMaxSlotsPerFrame { ups.removeLast(ups.count - clMaxSlotsPerFrame) }
            // Block changes in sections the volume holds as they are now (others arrive with their section's upload).
            for (x, y, z, code) in pendingEdits {
                let sx = floorDiv(x, 16), sy = floorDiv(y, 16), sz = floorDiv(z, 16)
                guard x >= org.x, x < org.x + clSizeX, y >= org.y, y < org.y + clSizeY, z >= org.z, z < org.z + clSizeZ else { continue }
                let slot = ((((sz % clSlotsZ) + clSlotsZ) % clSlotsZ) * clSlotsY + (((sy % clSlotsY) + clSlotsY) % clSlotsY)) * clSlotsX
                    + (((sx % clSlotsX) + clSlotsX) % clSlotsX)
                guard slotKey[slot] == clSectionKey(sx, sy, sz) else { continue }
                edits.append(SIMD4(UInt32(((x % clSizeX) + clSizeX) % clSizeX), UInt32(((y % clSizeY) + clSizeY) % clSizeY),
                                   UInt32(((z % clSizeZ) + clSizeZ) % clSizeZ), UInt32(code)))
            }
        }

        let cb = debugCB ?? { ctx.endBlit(); return ctx.ensureCB() }()
        if !cleared {
            // Once: the textures and buffers start empty (dark, nothing open: the relight keeps vanilla's light there).
            if let blit = cb.makeBlitCommandEncoder() {
                blit.fill(buffer: blocks, range: 0..<blocks.length, value: 0)
                blit.fill(buffer: light, range: 0..<light.length, value: 0)
                blit.fill(buffer: changed, range: 0..<changed.length, value: 0)
                blit.endEncoding()
            }
            if let enc = cb.makeComputeCommandEncoder() {
                enc.setComputePipelineState(clearPipe)
                enc.setTexture(rgb, index: 0)
                enc.setTexture(aux, index: 1)
                enc.dispatchThreads(MTLSize(width: clSizeX, height: clSizeY, depth: clSizeZ), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
                enc.endEncoding()
            }
            cleared = true
        }
        guard let enc = cb.makeComputeCommandEncoder(descriptor: debugCB == nil ? profComputePass("colored light") : MTLComputePassDescriptor()) else { return }
        enc.label = "MetalMC colored light"
        encodeWork(enc, ups: ups.map { ($0.slot, $0.sx, $0.sy, $0.sz) }, codesOf: codesOf, edits: edits, moved: moved, oldOrg: oldOrg,
                   pipes: (uploadPipe, editPipe, listBeginPipe, listPipe, propagatePipe, resolvePipe))
        enc.endEncoding()

        // The relight's parameters for this frame.
        var f = ClFrameGPU()
        let rel = SIMD3<Double>(camera.x - Double(org.x), camera.y - Double(org.y), camera.z - Double(org.z))
        let orgMod = SIMD3<Int>(((org.x % clSizeX) + clSizeX) % clSizeX, ((org.y % clSizeY) + clSizeY) % clSizeY, ((org.z % clSizeZ) + clSizeZ) % clSizeZ)
        f.camTex = SIMD4(Float(rel.x + Double(orgMod.x)), Float(rel.y + Double(orgMod.y)), Float(rel.z + Double(orgMod.z)), 1)
        f.camVol = SIMD4(Float(rel.x), Float(rel.y), Float(rel.z), 15)
        let t = debugTime >= 0 ? debugTime : ProcessInfo.processInfo.systemUptime.truncatingRemainder(dividingBy: 3600)
        f.size = SIMD4(Float(clSizeX), Float(clSizeY), Float(clSizeZ), Float(t))
        f.tune = SIMD4(clGain, clFlicker, clSlack, 1)
        f.fire = SIMD4(clBucketColors[clFireBucket], 0)
        f.curve0 = SIMD4(clCurve[0], clCurve[1], clCurve[2], clCurve[3])
        f.curve1 = SIMD4(clCurve[4], clCurve[5], clCurve[6], clCurve[7])
        f.curve2 = SIMD4(clCurve[8], clCurve[9], clCurve[10], clCurve[11])
        f.curve3 = SIMD4(clCurve[12], clCurve[13], clCurve[14], clCurve[15])
        frameParams = f
        frameTime = DispatchTime.now().uptimeNanoseconds
        if frames % 1200 == 0 {
            log("coloredlight: \(stats.uploads) sections and \(stats.edits) block changes uploaded, flood fill in \(stats.framesWithWork) of the last 1200 frames (\(stats.bricks / max(stats.framesWithWork, 1)) bricks on average); \(store.sectionCount) sections stored")
            stats = (0, 0, 0, 0)
        }
    }

    private func encodeWork(_ enc: MTLComputeCommandEncoder, ups: [(Int, Int, Int, Int)], codesOf: [Int: ClSection], edits: [SIMD4<UInt32>],
                            moved: Bool, oldOrg: SIMD3<Int>,
                            pipes: (MTLComputePipelineState, MTLComputePipelineState, MTLComputePipelineState, MTLComputePipelineState,
                                    MTLComputePipelineState, MTLComputePipelineState)) {
        guard let blocks, let light, let changed, let list, let args, let rgb, let aux, let colorBuffer, let curveBuffer else { return }
        let first = stamp
        var p = ClParamsGPU()
        let orgMod = SIMD3<Int>(((org.x % clSizeX) + clSizeX) % clSizeX, ((org.y % clSizeY) + clSizeY) % clSizeY, ((org.z % clSizeZ) + clSizeZ) % clSizeZ)
        p.orgMod = SIMD4(Int32(orgMod.x), Int32(orgMod.y), Int32(orgMod.z), 0)
        var classTable = clClassTable
        enc.setBuffer(blocks, offset: 0, index: 2)
        enc.setBuffer(light, offset: 0, index: 3)
        enc.setBuffer(changed, offset: 0, index: 4)
        enc.setBytes(&classTable, length: classTable.count * 4, index: 6)
        enc.setBuffer(list, offset: 0, index: 7)
        enc.setBuffer(args, offset: 0, index: 8)
        enc.setBuffer(colorBuffer, offset: 0, index: 9)
        enc.setBuffer(curveBuffer, offset: 0, index: 10)
        // Uploads: the sections' codes into a staging buffer, then the kernel puts them in place. After the volume moved,
        // the bricks along its trailing edges are stamped too (their light came partly from cells now outside it).
        var slots: [ClSlotGPU] = []
        var staging: [UInt16] = []
        for (slot, sx, sy, sz) in ups {
            guard let sec = codesOf[slot] else { continue }
            let i = slot % clSlotsX, j = (slot / clSlotsX) % clSlotsY, k = slot / (clSlotsX * clSlotsY)
            var g = ClSlotGPU()
            g.tex = SIMD4(Int32(i * 16), Int32(j * 16), Int32(k * 16), 0)
            if let codes = sec.codes {
                g.info = SIMD4(0, UInt32(staging.count), 0, 0)
                staging.append(contentsOf: codes)
            } else {
                g.info = SIMD4(1, 0, UInt32(sec.uniform), 0)
            }
            slots.append(g)
            slotKey[slot] = clSectionKey(sx, sy, sz)
            slotVersion[slot] = sec.version
        }
        if moved && oldOrg.x != Int.min {
            // Slots on the faces of the volume that its move left as edges: the first and last slot along each axis.
            let s0 = SIMD3(floorDiv(org.x, 16), floorDiv(org.y, 16), floorDiv(org.z, 16))
            func ring(_ axis: Int, _ s: Int) {
                let a = (s % [clSlotsX, clSlotsY, clSlotsZ][axis] + [clSlotsX, clSlotsY, clSlotsZ][axis]) % [clSlotsX, clSlotsY, clSlotsZ][axis]
                for v in 0..<(axis == 1 ? clSlotsX * clSlotsZ : (axis == 0 ? clSlotsY * clSlotsZ : clSlotsX * clSlotsY)) {
                    var i = 0, j = 0, k = 0
                    switch axis {
                    case 0: i = a; j = v % clSlotsY; k = v / clSlotsY
                    case 1: j = a; i = v % clSlotsX; k = v / clSlotsX
                    default: k = a; i = v % clSlotsX; j = v / clSlotsX
                    }
                    var g = ClSlotGPU()
                    g.tex = SIMD4(Int32(i * 16), Int32(j * 16), Int32(k * 16), 0)
                    g.info = SIMD4(2, 0, 0, 0)
                    slots.append(g)
                }
            }
            if org.x != oldOrg.x { ring(0, s0.x); ring(0, s0.x + clSlotsX - 1) }
            if org.y != oldOrg.y { ring(1, s0.y); ring(1, s0.y + clSlotsY - 1) }
            if org.z != oldOrg.z { ring(2, s0.z); ring(2, s0.z + clSlotsZ - 1) }
        }
        p.stamp = SIMD4(first, 0, 0, 0)
        if !slots.isEmpty {
            if staging.isEmpty { staging = [0] }
            guard let stagingBuffer = ctx.device.makeBuffer(bytes: staging, length: staging.count * 2, options: .storageModeShared),
                  let slotBuffer = ctx.device.makeBuffer(bytes: slots, length: slots.count * MemoryLayout<ClSlotGPU>.stride, options: .storageModeShared)
            else { return }
            p.stamp.w = UInt32(slots.count)
            enc.setComputePipelineState(pipes.0)
            enc.setBuffer(stagingBuffer, offset: 0, index: 0)
            enc.setBuffer(slotBuffer, offset: 0, index: 1)
            enc.setBytes(&p, length: MemoryLayout<ClParamsGPU>.stride, index: 5)
            enc.dispatchThreads(MTLSize(width: 16, height: 16, depth: 16 * slots.count), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 8))
            stats.uploads += ups.count
        }
        if !edits.isEmpty {
            guard let editBuffer = ctx.device.makeBuffer(bytes: edits, length: edits.count * 16, options: .storageModeShared) else { return }
            p.stamp.w = UInt32(edits.count)
            enc.setComputePipelineState(pipes.1)
            enc.setBuffer(editBuffer, offset: 0, index: 0)
            enc.setBytes(&p, length: MemoryLayout<ClParamsGPU>.stride, index: 5)
            enc.dispatchThreads(MTLSize(width: edits.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(64, edits.count), height: 1, depth: 1))
            stats.edits += edits.count
        }
        // The last listing's count (a frame or two old: the stats only).
        let listed = Int(args.contents().load(as: UInt32.self))
        lastListed = listed
        if listed > 0 { stats.framesWithWork += 1; stats.bricks += listed }
        // The bricks to run over: changed since the previous frame's first pass (its passes, and this frame's uploads).
        p.stamp = SIMD4(first, lastFrameFirst &+ 1, first, 0)
        enc.setComputePipelineState(pipes.2)
        enc.dispatchThreads(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
        enc.setComputePipelineState(pipes.3)
        enc.setBytes(&p, length: MemoryLayout<ClParamsGPU>.stride, index: 5)
        enc.dispatchThreads(MTLSize(width: clBricks, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.setComputePipelineState(pipes.4)
        for it in 1...clIterations {
            p.stamp.x = first &+ UInt32(it)
            enc.setBytes(&p, length: MemoryLayout<ClParamsGPU>.stride, index: 5)
            enc.dispatchThreadgroups(indirectBuffer: args, indirectBufferOffset: 0, threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 8))
        }
        enc.setComputePipelineState(pipes.5)
        p.stamp.z = first
        enc.setBytes(&p, length: MemoryLayout<ClParamsGPU>.stride, index: 5)
        enc.setTexture(rgb, index: 0)
        enc.setTexture(aux, index: 1)
        enc.dispatchThreadgroups(indirectBuffer: args, indirectBufferOffset: 0, threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 8))
        lastFrameFirst = first
        stamp = first &+ UInt32(clIterations) &+ 1
    }

    /// This frame's volume for the relight (set by `frame` in the last 100 ms; offline, with the flicker's clock held, the
    /// last one whenever it was), or a disabled one (1 x 1 x 1 stand-ins: the relight then keeps vanilla's block light),
    /// also if the volume's kernels failed to build.
    private func current() -> (rgb: MTLTexture, aux: MTLTexture, frame: ClFrameGPU)? {
        if !ensure() || dummy3D == nil {
            let dd = MTLTextureDescriptor()
            dd.textureType = .type3D
            dd.pixelFormat = .rgba16Float
            dd.width = 1; dd.height = 1; dd.depth = 1
            dd.usage = .shaderRead
            if dummy3D == nil { dummy3D = ctx.device.makeTexture(descriptor: dd) }
        }
        guard let dummy3D else { return nil }
        var f = frameParams
        if (debugTime < 0 && DispatchTime.now().uptimeNanoseconds - frameTime > 100_000_000) || !cleared { f.camTex.w = 0 }
        f.camTex.w = shadingOn ? f.camTex.w : 0
        guard f.camTex.w > 0, let rgb, let aux else { f.camTex.w = 0; return (dummy3D, dummy3D, f) }
        return (rgb, aux, f)
    }

    /// Offline checks: the volume's min corner (world blocks), and a copy of its light and codes (the GPU's, by cell index
    /// as the kernels store them), after the work submitted so far.
    var debugOrigin: SIMD3<Int> { org }
    func debugReadback() -> (light: [UInt32], blocks: [UInt16])? {
        guard let light, let blocks, let lb = ctx.device.makeBuffer(length: light.length, options: .storageModeShared),
              let bb = ctx.device.makeBuffer(length: blocks.length, options: .storageModeShared),
              let cb = ctx.queue.makeCommandBuffer(), let blit = cb.makeBlitCommandEncoder() else { return nil }
        blit.copy(from: light, sourceOffset: 0, to: lb, destinationOffset: 0, size: light.length)
        blit.copy(from: blocks, sourceOffset: 0, to: bb, destinationOffset: 0, size: blocks.length)
        blit.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        let n = clSizeX * clSizeY * clSizeZ
        return (Array(UnsafeBufferPointer(start: lb.contents().assumingMemoryBound(to: UInt32.self), count: n)),
                Array(UnsafeBufferPointer(start: bb.contents().assumingMemoryBound(to: UInt16.self), count: n)))
    }

    /// Offline: the bricks the last listing ran over (exact once its command buffer is done).
    func debugListed() -> Int { args.map { Int($0.contents().load(as: UInt32.self)) } ?? 0 }

    /// The kernels' cell index of toroidal cell (x, y, z) (bricks of 8^3, brick-major).
    static func cellIndex(_ x: Int, _ y: Int, _ z: Int) -> Int {
        (((z >> 3) * clBricksY + (y >> 3)) * clBricksX + (x >> 3)) * 512 + ((z & 7) * 8 + (y & 7)) * 8 + (x & 7)
    }

    /// Lit hook (Lit.relight, its own pass): textures 8 and 9 and buffer 2 of lit_relight_fs.
    func bindRelight(_ enc: MTLRenderCommandEncoder) {
        guard var c = current() else { return }
        enc.setFragmentTexture(c.rgb, index: 8)
        enc.setFragmentTexture(c.aux, index: 9)
        enc.setFragmentBytes(&c.frame, length: MemoryLayout<ClFrameGPU>.stride, index: 2)
    }

    /// Lit hook (Lit.bindDeferred, the anti-aliasing's resolve): textures 15 and 16 and buffer 4 of taa_resolve.
    func bindDeferred(_ enc: MTLComputeCommandEncoder) {
        guard var c = current() else { return }
        enc.setTexture(c.rgb, index: 15)
        enc.setTexture(c.aux, index: 16)
        enc.setBytes(&c.frame, length: MemoryLayout<ClFrameGPU>.stride, index: 4)
    }
}

// MARK: - C entry points (the Java side: metalmc.light.ColoredLight)

/// 1 if colored block light is on (METALMC_EXP=lit,coloredlight).
@_cdecl("mmc_cl_enabled")
public func mmc_cl_enabled() -> Int32 { clEnabled ? 1 : 0 }

/// The color class of a block by its registry name ("minecraft:soul_torch"), for the codes the Java side sends.
@_cdecl("mmc_cl_classify")
public func mmc_cl_classify(_ name: UnsafePointer<CChar>) -> Int32 {
    Int32(clClassify(String(cString: name)).rawValue)
}

/// A new world (or dimension): forgets every section; chunks and block changes of other generations are ignored after.
@_cdecl("mmc_cl_reset")
public func mmc_cl_reset(_ generation: Int32) {
    ColoredLight.shared.store.reset(generation: generation)
}

/// One chunk's blocks: `count` sections, their y (section coordinates, -4..19) in `ys`, and 4096 codes each (laid out y,
/// z, x; ClCode: dampening | emission << 4 | class << 8) in `codes`. Replaces whatever the chunk had.
@_cdecl("mmc_cl_chunk")
public func mmc_cl_chunk(_ generation: Int32, _ cx: Int32, _ cz: Int32, _ count: Int32, _ ys: UnsafePointer<Int32>, _ codes: UnsafePointer<UInt16>) {
    let s = ColoredLight.shared.store
    s.lock.lock(); let g = s.generation; s.lock.unlock()
    guard generation == g else { return }
    var list: [(Int, [UInt16])] = []
    for i in 0..<Int(count) {
        list.append((Int(ys[i]), Array(UnsafeBufferPointer(start: codes + i * 4096, count: 4096))))
    }
    s.putChunk(cx: Int(cx), cz: Int(cz), list)
}

/// One block changed to `code`.
@_cdecl("mmc_cl_block")
public func mmc_cl_block(_ generation: Int32, _ x: Int32, _ y: Int32, _ z: Int32, _ code: Int32) {
    let s = ColoredLight.shared.store
    s.lock.lock(); let g = s.generation; s.lock.unlock()
    guard generation == g else { return }
    s.edit(x: Int(x), y: Int(y), z: Int(z), code: UInt16(truncatingIfNeeded: code))
}

/// The frame's volume work, before the relight (GameRendererLodMixin): the camera's world position.
@_cdecl("mmc_cl_frame")
public func mmc_cl_frame(_ x: Double, _ y: Double, _ z: Double) {
    guard clEnabled else { return }
    ColoredLight.shared.frame(camera: SIMD3(x, y, z))
}
