import Foundation
import Metal
import MetalMCCore
import simd

// Bounce light (global illumination) from a world-space irradiance cache (METALMC_EXP=gi, prototype; in the frame with
// lit mode, METALMC_EXP=lit,gi: docs/gi-design.md has the design and the measurements, docs/lighting-design.md "Lit mode
// with the GI cache" the wiring).
//
// Minecraft's world is axis-aligned block faces that rarely change, so indirect light is stored on the world instead
// of the screen: one cache cell per block face near the player (keyed by the air block in front of the face and the
// face's direction), and cells of 2^L x 2^L x 2^L blocks farther out, the level picked by distance so a cell covers
// about the same number of pixels everywhere. Cells live in a hash table of 16-slot buckets (a bucket's fingerprints
// are one 64-byte line, so a lookup is one memory access) and hold the irradiance from the sky and from bounced
// light, plus the share of the sun's disk the cell sees.
//
// Each frame, in one compute encoder:
// 1. request: a sixteenth of the pixels (a different one of each 4 x 4 block each frame) find or create the cell their
//    surface lies in, mark it seen and list it for an update (three quarters of the list are for visible cells). A new
//    cell starts from its parent (the next level's cell holding it).
// 2. schedule: a slice of the table is swept (sized so the cells it lists fill the last quarter of the list); cells
//    unseen for 10 s are evicted, and cells that aren't visible (seen a while ago, or only reached by bounce rays) are
//    listed every 8th time the sweep passes them.
// 3. update: the listed cells, up to a budget, trace a few rays each with the hardware ray tracer over the LOD's
//    acceleration structures: toward the sun's disk, and one cosine-weighted bounce that reads the cell it lands on,
//    so light bounces once more with every update. Samples follow a low-discrepancy sequence per cell and are averaged
//    over the cell's last 64 updates: stable in motion, since nothing depends on the screen.
// 4. resolve: at half resolution, the 2 x 2 nearest cells on the surface's plane (only cells that exist, so light
//    doesn't cross walls), blended toward the next level near level boundaries; the lighting pass upsamples it by
//    face and plane (giUpsample).
//
// Direct sunlight on screen stays the per-pixel shadow ray (RtShadows), and block light the flood fill vanilla and the
// LOD already carry; the cache holds what those can't: sky light that follows the real openings, and bounced light.
//
// In lit mode (METALMC_EXP=lit,gi) RtShadows runs the frame after building its instance structure (encodeFrame), and
// the relight takes the cache's light in place of its sky term (litRelightPixel, giUpsample). What a bounce ray needs
// from its hit (material, face, block light, sky cover) comes from the quad it hit, read from the LOD node's own quad
// buffer (GI_ZERO_COPY: each tile's structure has one geometry per quad range, and the hit's instance, geometry and
// primitive index find the quad through giTiles); the structures carry no per-triangle data. The cells hold light per
// unit of the frame's daylight (GiLight.scale), so the time of day, rain and lightning show at once.

/// METALMC_EXP=gi: the world-space irradiance cache (with lit, wired into the frame: litGi).
let lodGi = experiments.contains("gi")
/// METALMC_EXP=lit,gi: the cache runs in the frame and lit mode's relight takes its light (nothing changes without both).
let litGi = litEnabled && lodGi
private func giEnv(_ key: String) -> String? { ProcessInfo.processInfo.environment[key] }
/// METALMC_GICAP: log2 of the cache's slots (default 21: 2 M cells, 72 MB).
let giCapacityLog2 = min(24, max(14, Int(giEnv("METALMC_GICAP") ?? "") ?? 21))
/// METALMC_GIBUDGET: cells updated per frame at most (default 16384: about 0.5 ms of rays at 4 samples).
let giBudget = max(1024, Int(giEnv("METALMC_GIBUDGET") ?? "") ?? 16384)
/// METALMC_GISPP: samples per cell update, each a sun ray and a bounce ray, plus a probe at coarse levels (default 4:
/// in a room lit through a small window, 4 samples over 64 updates came 2.3x closer to a many-sample reference than 2
/// over 32, for about 1.6x the rays per cell).
let giSamplesPerUpdate = min(16, max(1, Int(giEnv("METALMC_GISPP") ?? "") ?? 4))
/// METALMC_GIRANGE: distance (blocks) where level-0 cells (one block face) end; each level after that reaches twice as
/// far. At 160 a block face is about 10 px on the panel (3456 x 2234, 70 degree field of view).
let giLevel0Range = Float(giEnv("METALMC_GIRANGE") ?? "") ?? 160
/// The most and fewest frames the schedule takes to sweep the whole table: it sweeps faster while the cells it lists fit
/// the budget, but not the whole table every frame (2 M slots cost 0.35-0.6 ms to sweep; an eighth, under 0.1).
let giSweepFramesMax = 64, giSweepFramesMin = 8
/// METALMC_GIHISTORY: updates a cell's average holds at most (new samples weigh 1/history once it has converged;
/// default 64, at most 255).
let giHistory = Float(min(255, max(1, Int(giEnv("METALMC_GIHISTORY") ?? "") ?? 64)))
/// METALMC_GISPATIAL: how much each coplanar neighbor cell weighs in a cell's update, relative to its own rays (spatial
/// reuse; default 0, off). It cuts the noise but biases enclosed rooms bright: blurring along a floor moves light from
/// cells that see the sky to cells the room reflects more, and multiple bounces amplify it (+27% in the test house at
/// 0.25, none outdoors).
let giSpatial = max(0, Float(giEnv("METALMC_GISPATIAL") ?? "") ?? 0)
/// Frames a cell may go unseen before it's evicted (10 s at 120 Hz).
let giEvictFrames: Float = 1200
/// Pixels per request sample along each axis.
let giRequestScale = 4
/// Slots per bucket; must match GI_BUCKET in the shader.
let giBucketSlots = 16
/// Bytes per slot: fingerprint 4, key and first surface point 16, last seen 4, light 8, count and flags 4.
let giBytesPerCell = 36

/// The way back from gi_resolve's half-resolution output to full resolution (giUpsample), shared by the cache's library
/// and lit mode's relight (Lit.swift, Taa.swift, with METALMC_EXP=lit,gi only).
let giUpsampleHeader = """

// gi_resolve's output per half-resolution sample, one RG32Uint texel: x the irradiance as RG11B10Float's bits
// (giPackIrradiance), y the code word: the surface's plane (its camera-relative coordinate along the face's axis, as a
// half) | face << 16, and GI_CODE_DATA when the irradiance is there, GI_CODE_EMPTY when the surface's cells have no
// samples yet; 0 for the sky or past the cached range. One texel holds both, so the upsample's common case is one read
// whose address depends on nothing but the pixel (two textures took the code, then the irradiance it picked).
#define GI_CODE_DATA (1u << 19)
#define GI_CODE_EMPTY (1u << 20)

// RG11B10Float's bits back to its value: its 5-bit exponents have the half's bias, so each channel is a half whose low
// mantissa bits are zero (exact).
static float3 giUnpackIrradiance(uint v) {
    return float3(float(as_type<half>(ushort((v & 0x7FFu) << 4))), float(as_type<half>(ushort(((v >> 11) & 0x7FFu) << 4))),
                  float(as_type<half>(ushort((v >> 22) << 5))));
}

// The texel of gi_resolve's output that giUpsample starts from for full-resolution pixel `gid` (its 2 x 2 block's own),
// for callers that read it early (its address depends on nothing else).
static uint2 giUpsampleTexel(texture2d<uint> gi, uint2 gid) {
    return gi.read(min(gid / 2u, uint2(gi.get_width() - 1, gi.get_height() - 1))).rg;
}

// gi_resolve's half-resolution irradiance at full-resolution pixel `gid`, whose surface is on `face` at camera-relative
// `rel` (`own`: giUpsampleTexel's): its own 2 x 2 block's sample when that is on the same face and plane (most pixels:
// that one read; 92.6% of the surface pixels in the region test), else the nearest of the 8 around it that is (an edge,
// or a face too small for a sample of its own, like the riser of a one-block step: planes up to a block and a half apart
// count, the next step's light). The samples are bilinear across cells 10 or more pixels wide, so nearest-sample steps of
// 2 pixels don't show; a bilinear upsample (4 taps, or a gather of the codes and a filtered read) cost 0.5-0.9 ms at the
// panel's resolution, since so many pixels lie near a face's edge in this world that the taps rarely all match. The 8
// code words come in four 2 x 2 gathers (the anti-aliasing resolve's load loop runs a row of pixels per SIMD group, so
// nearly every group has a pixel that needs them: 0.07 ms less than 8 reads offline), then one read of the chosen
// sample's light. w: 1 with data; 0 when nothing matched (the caller may take the cells itself, giIrradiance); -1 when
// the surface's cells have no samples yet (they'd find nothing: not worth the lookups, which just after a turn cost 1 ms).
static float4 giUpsample(texture2d<uint> gi, uint2 own, uint2 gid, uint face, float3 rel) {
    int2 hsize = int2(gi.get_width(), gi.get_height());
    int2 c0 = min(int2(gid / 2u), hsize - 1);
    float plane = rel[face >> 1];
    float tol = max(0.25, 0.002 * length(rel));   // depth precision and the half's fall with distance
    uint fw = face << 16;
    uint c = own.y;
    float cd = abs(float(as_type<half>(ushort(c & 0xFFFFu))) - plane);
    if ((c & ((7u << 16) | GI_CODE_DATA)) == (fw | GI_CODE_DATA) && cd <= tol) return float4(giUnpackIrradiance(own.x), 1.0);
    bool empty = (c & ((7u << 16) | GI_CODE_EMPTY)) == (fw | GI_CODE_EMPTY) && cd <= tol;
    // The 3 x 3 code words around c0, gathered at its four corners (clamped at the edges, as the reads were).
    constexpr sampler sp(coord::pixel, filter::nearest, address::clamp_to_edge);
    float2 cc = float2(c0);
    uint4 g00 = gi.gather(sp, cc, int2(0), component::y);                       // x (-1, 0), z (0, -1), w (-1, -1)
    uint4 g10 = gi.gather(sp, cc + float2(1.0, 0.0), int2(0), component::y);   // y (1, 0), z (1, -1)
    uint4 g01 = gi.gather(sp, cc + float2(0.0, 1.0), int2(0), component::y);   // x (-1, 1), y (0, 1)
    uint g11 = gi.gather(sp, cc + float2(1.0, 1.0), int2(0), component::y).y;  // (1, 1)
    // The 8 around, edge neighbors first, the nearest plane winning (a later one only if nearer by more than 0.01). All
    // 8 every time: stopping at an exact plane (the same choice) made the anti-aliasing resolve 0.065 ms slower offline
    // (measured; likely the loop no longer unrolls).
    const int2 around[8] = { int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1), int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1) };
    uint codes[8] = { g10.y, g00.x, g01.y, g00.z, g11, g01.x, g10.z, g00.w };
    int best = -1;
    float bestD = tol + 1.5;
    for (int k = 0; k < 8; k++) {
        uint n = codes[k];
        if ((n & (7u << 16)) != fw) continue;
        float nd = abs(float(as_type<half>(ushort(n & 0xFFFFu))) - plane);
        if ((n & GI_CODE_DATA) == 0u) { empty = empty || ((n & GI_CODE_EMPTY) != 0u && nd <= tol); continue; }
        if (nd < bestD - 0.01) { bestD = nd; best = k; }
    }
    if (best >= 0) return float4(giUnpackIrradiance(gi.read(uint2(clamp(c0 + around[best], int2(0), hsize - 1))).r), 1.0);
    return float4(0.0, 0.0, 0.0, empty ? -1.0 : 0.0);
}
"""

/// The cache's kernels. `zeroCopy`: a bounce hit finds its quad in the LOD node's buffer through giTiles (the game's
/// route, structures without per-triangle data); otherwise it reads the structure's primitive data (giTileGeometry,
/// +54% structure memory; offline comparisons only).
private func giShaderSource(zeroCopy: Bool) -> String {
    "#define GI_ZERO_COPY \(zeroCopy ? 1 : 0)\n" + skyShaderHeader + giUpsampleHeader + "\n" + giKernelSource
}

/// Debug: the upsample header and kernels, for an offline compile check, or as the start of a kernel variant to time
/// against them (mmc_debug_gi_reload, tools/gicache.py exp). Returns their length.
@_cdecl("mmc_debug_gi_shader_source")
public func mmc_debug_gi_shader_source(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let bytes = Array((giUpsampleHeader + "\n" + giKernelSource).utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}

private let giKernelSource = """
#include <metal_raytracing>
using namespace raytracing;

struct GiParams {
    float4x4 invViewProj;  // request, resolve, debug view: inverse of the projection * view rotation the depth was drawn with
    int4 origin;           // xyz: the acceleration structure's origin (world blocks); w: frame number
    int4 camBlock;         // xyz: the camera's block (floor of its position); w: buckets - 1
    float4 camFrac;        // xyz: camera position minus camBlock; w: distance where level-0 cells end (blocks)
    float4 sun;            // xyz: direction to the sun; w: 1 with the sun up, 0 without
    float4 rays;           // x: sun disk radius (radians), y: max sun ray length (blocks), z: updates a cell's average
                           // holds at most, w: the cloud layer's bottom relative to the camera (the request skips the 4
                           // blocks above it: vanilla's clouds write depth; a huge value without clouds)
    float4 blockLight;     // rgb: irradiance at block light 15 (torch color; 0 in lit mode, whose block light stays
                           // vanilla's); w: frames unseen before eviction
    float4 sizes;          // full width, full height, request width, request height
    float4 limits;         // x: farthest distance cached (blocks), y: ray offset from surfaces, z: debug view mode,
                           // w: max bounce ray length (blocks)
    float4 look;           // x: debug view exposure, y: emission seen by bounce rays, z: emission drawn by the debug view,
                           // w: weight of each coplanar neighbor in an update (spatial reuse)
    uint4 counts;          // x: update list capacity (the budget), y: first slot of this frame's sweep, z: slots swept,
                           // w: sweeps completed
    uint4 sample;          // x: pixels per request sample, y-z: this frame's pixel in that block, w: samples per update
    uint4 sched;           // x: chance (of 2^32) that a visible cell with samples is listed this frame, y-w: unused
};

// The cache's light, in its own units: light per unit of the frame's daylight, so that a change of daylight (the time of
// day, rain, a lightning flash) shows at once instead of after the cells' history (64 updates, up to seconds): the
// resolve multiplies what the cells hold by `scale`. In lit mode gi_light makes it each frame from the relight's sun and
// sky light (Lit.swift's env), so `scale` x the cells' light is in the relight's units; the offline test fills it with
// the prototype's light (scale 1).
struct GiLight {
    float4 sun;            // rgb: the sun's irradiance on a surface facing it
    float4 scale;          // rgb: what the cells' light is multiplied by where it's read (the frame's daylight)
    float4 skyView;        // rgb: Sky's sky view table (skyLuminance) to these units; w: 1 to take the sky from it
    float4 zenith;         // rgb: sky radiance straight up (without the table)
    float4 horizon;        // rgb: sky radiance at the horizon (without the table)
    float4 ground;         // rgb: radiance below the horizon, where rays leave past the LOD's edge
    float4 open[6];        // rgb: irradiance from the open sky on each face direction (a bounce hit on a cell without
                           // samples takes half of it)
};

#if GI_ZERO_COPY
// Per instance of the instance structure (in its order): the LOD node's quad buffer, by its GPU address (the encoder
// declares the buffers resident), and the first quad of each geometry of the tile's structure (its opaque faces, then
// its tile-edge skirts; giTileMesh). A geometry has two triangles per quad in the buffer's order.
struct GiTile {
    const device uint* quads;
    uint start[2];
};
// The hit's quad, as giTileGeometry packs a triangle's data: material | face << 8 | block light << 11 | water depth or
// sky cover << 15. One dependent read per hit instead of 4 bytes per triangle in the structure (+54% memory).
static uint giHitQuad(const device GiTile* tiles, uint instance, uint geometry, uint primitive) {
    const device GiTile& t = tiles[instance];
    uint q = t.start[min(geometry, 1u)] + primitive / 2u;
    uint w0 = t.quads[2u * q], w1 = t.quads[2u * q + 1u];
    return (w1 & 255u) | (((w0 >> 25) & 7u) << 8) | (((w1 >> 24) & 15u) << 11) | (((w0 >> 28) & 15u) << 15);
}
#define GI_HIT(h) giHitQuad(tiles, h.instance_id, h.geometry_id, h.primitive_id)
#define GI_TILES_ARG , const device GiTile* tiles [[buffer(17)]]
#else
#define GI_HIT(h) giPrim(h.primitive_data)
#define GI_TILES_ARG
#endif

#define GI_BUCKET 16u
#define GI_MAX_LEVEL 12u
// A cell requested by the screen within this many frames is visible: the request lists it for an update whenever it
// touches it (first touch in a frame), and its bounce rays may create the cells they reach. Others (seen a while ago, or
// only reached by bounce rays) are listed by the sweep, every 8th time it passes them, and create nothing, so the cache
// doesn't grow into the closure of everything bounce rays can reach.
#define GI_RECENT 32u
// meta: bits 0-7 updates in the average, bit 8 re-check the surface (an edit nearby), bit 9 no surface found (stat),
// bits 16-31 the cell's next sample index.
#define GI_VERIFY 256u
#define GI_NOSURF 512u
// The update list: the first 3/4 for visible cells (the request's), the rest for the sweep's, so cells off screen keep
// a share of the budget however much is visible.
// counters: 0 visible cells listed this frame (may overshoot their part of the list), 1 cells to update this frame,
// 2 cells created by the screen, 3 inserts that found their bucket full, 4 evictions, 5 updates that found no surface,
// 6-7 live and converged cells (gi_count), 8-10 the update's indirect dispatch, 11 rays traced, 12 bounce hits read from
// the cache, 13 bounce hits without data, 14 cells created by bounce rays, 15 samples from a cell's first surface point
// (the probe missed), 16 cells the sweep listed this frame (may overshoot its part), 17 the visible cells in the list,
// 18 visible cells seen this frame (first touches), 19 the same, kept for the CPU once the frame's request is done.

constant float3 kGiNormal[6] = { float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0), float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1) };
constant float kGiPi = 3.14159265;

static uint giMix(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
static float giUnit(uint x) { return float(x >> 8) * (1.0 / 16777216.0); }
// Point i of the R2 low-discrepancy sequence (Roberts), in 32-bit fixed point so it stays exact, shifted per cell.
static float2 giR2(uint i, uint2 shift) { return float2(giUnit(i * 3242174890u + shift.x), giUnit(i * 2447445414u + shift.y)); }
// floor(v / 2^s) for negative v too.
static int giFloorShift(int v, uint s) { return v >= 0 ? (v >> s) : ~((~v) >> s); }
// The face's (u, v) axes as the mesher's: X faces u = y, v = z; Y faces u = x, v = z; Z faces u = x, v = y.
static uint2 giTangents(uint face) { uint a = face >> 1; return a == 0u ? uint2(1, 2) : (a == 1u ? uint2(0, 2) : uint2(0, 1)); }
static uint giFaceClass(uint face) { return face == 2u ? 0u : (face == 3u ? 2u : 1u); }   // top, side, bottom

// The air block in front of a surface point given as an integer base plus a float offset (base + f): on the face's
// axis the surface lies on the nearest block boundary, and the air is on the normal's side of it.
static int3 giAirBlock(int3 base, float3 f, uint face) {
    uint a = face >> 1;
    int3 b = base + int3(floor(f));
    int plane = base[a] + int(rint(f[a]));
    b[a] = (face & 1u) == 0u ? plane : plane - 1;
    return b;
}
// A block's cell at a level: x and z in cells of 2^level blocks, y counted from the world bottom (y -64), so cells line
// up with the LOD's voxels.
static int3 giCellOf(int3 block, uint level) {
    return int3(giFloorShift(block.x, level), giFloorShift(max(block.y + 64, 0), level), giFloorShift(block.z, level));
}
// 64-bit key: x and z (24 bits each, modulo 2^24 cells), y (9 bits), face (3), level (4).
static uint2 giKey(int3 c, uint face, uint level) {
    uint x = uint(c.x) & 0xFFFFFFu, z = uint(c.z) & 0xFFFFFFu, y = uint(clamp(c.y, 0, 511));
    return uint2(x | (z << 24), (z >> 8) | (y << 16) | (face << 25) | (level << 28));
}
// A key's cell, with x and z unwrapped around `near` (any cell within 2^23 cells of it).
static int3 giUnkey(uint2 k, int3 near) {
    int x = int(k.x & 0xFFFFFFu), z = int((k.x >> 24) | ((k.y & 0xFFFFu) << 8)), y = int((k.y >> 16) & 511u);
    int dx = (x - near.x) & 0xFFFFFF, dz = (z - near.z) & 0xFFFFFF;
    if (dx >= 0x800000) dx -= 0x1000000;
    if (dz >= 0x800000) dz -= 0x1000000;
    return int3(near.x + dx, y, near.z + dz);
}
static uint giBucketOf(constant GiParams& p, uint2 k) { return (giMix(k.x ^ giMix(k.y + 0x9e3779b9u)) & uint(p.camBlock.w)) * GI_BUCKET; }
static uint giPrint(uint2 k) { return giMix(k.y ^ giMix(k.x + 0x85ebca6bu)) | 1u; }
// Level as a real number: 0 inside the level-0 range, then one more per doubling of distance.
static float giLevelF(constant GiParams& p, float d) { return clamp(log2(max(d, 1.0) / p.camFrac.w) + 1.0, 0.0, float(GI_MAX_LEVEL)); }

// Lookups scan the whole bucket: evictions leave holes, so an empty slot doesn't end the search. Four 16-byte loads,
// issued together and then the first match in slot order taken: with an early exit after each, a SIMD group waited on
// them one after another for any lane whose key wasn't there (a bilinear's corners past a surface's edge; the resolve
// took 0.02 ms longer).
static int giFind(const device uint* check, uint base, uint fp) {
    const device uint4* b = (const device uint4*)(check + base);
    uint4 b0 = b[0], b1 = b[1], b2 = b[2], b3 = b[3];
    uint4 i0 = select(uint4(16u), uint4(0u, 1u, 2u, 3u), b0 == uint4(fp));
    uint4 i1 = select(uint4(16u), uint4(4u, 5u, 6u, 7u), b1 == uint4(fp));
    uint4 i2 = select(uint4(16u), uint4(8u, 9u, 10u, 11u), b2 == uint4(fp));
    uint4 i3 = select(uint4(16u), uint4(12u, 13u, 14u, 15u), b3 == uint4(fp));
    uint4 m = min(min(i0, i1), min(i2, i3));
    uint i = min(min(m.x, m.y), min(m.z, m.w));
    return i < 16u ? int(base + i) : -1;
}
// The same scan in a kernel that also inserts: plain loads (a slot claimed in this dispatch may still read empty; the
// claim's exchanges then see it).
static int giFindA(device atomic_uint* check, uint base, uint fp) { return giFind((const device uint*)check, base, fp); }
// Finds the key's slot or claims the first empty one; `created` when this thread claimed it, -1 if the bucket is full.
// Every thread tries the empty slots in the same order and a failed exchange shows what took the slot, so two threads
// inserting the same key end up in the same slot.
static int giClaim(device atomic_uint* check, uint base, uint fp, thread bool& created) {
    created = false;
    int found = giFindA(check, base, fp);
    if (found >= 0) return found;
    for (uint i = 0; i < GI_BUCKET; i++) {
        uint expected = 0u;
        while (!atomic_compare_exchange_weak_explicit(&check[base + i], &expected, fp, memory_order_relaxed, memory_order_relaxed)) {
            if (expected != 0u) break;   // taken; a weak exchange may also fail spuriously and leave it 0: try again
        }
        if (expected == 0u) { created = true; return int(base + i); }
        if (expected == fp) return int(base + i);
    }
    return -1;
}
// A cell's value: rgb its irradiance from the sky and bounced light, a 1 + the share of the sun's disk it sees once it
// has samples. Zero (cleared memory, an evicted or new cell) reads as no samples, so a cell claimed in the same
// dispatch never passes for a black one.
static bool giSampled(half4 v) { return v.a >= 1.0h; }
// The corner of a cell in world blocks.
static int3 giCorner(int3 c, uint level) { int s = 1 << level; return int3(c.x * s, c.y * s - 64, c.z * s); }
// A surface point of a cell, where it was first seen (base + f, air block `air`): its position across the face in
// 1/256 of the cell (8 bits each) and the face's plane in blocks from the cell's corner (16 bits). Coarse cells hold
// faces on several planes and often over only part of the cell (the steps of a hillside), so when an update's probe
// finds no surface it falls back to this point.
static uint giRepPoint(int3 base, float3 f, int3 air, uint face, uint level) {
    uint2 t = giTangents(face);
    uint a = face >> 1;
    int s = 1 << level;
    int3 corner = giCorner(giCellOf(air, level), level);
    float3 local = float3(base - corner) + f;
    uint u = uint(clamp(local[t.x] / float(s) * 256.0, 0.0, 255.0)), v = uint(clamp(local[t.y] / float(s) * 256.0, 0.0, 255.0));
    uint plane = uint(clamp(air[a] - corner[a] + int(face & 1u), 0, 65535));
    return u | (v << 8) | (plane << 16);
}
// A new cell starts from its parent (the next level's cell holding it), counted as two updates so its own samples
// take over quickly: coming closer refines the light instead of starting it black.
static void giInit(constant GiParams& p, device atomic_uint* check, device uint4* keys, device half4* value, device uint* meta,
                   uint e, uint2 key, int3 air, uint face, uint level, uint rep) {
    keys[e] = uint4(key, rep, 0u);
    half4 v = half4(0.0h);
    uint m = 0u;
    if (level < GI_MAX_LEVEL) {
        uint2 pk = giKey(giCellOf(air, level + 1u), face, level + 1u);
        int pe = giFindA(check, giBucketOf(p, pk), giPrint(pk));
        if (pe >= 0) {
            half4 pv = value[pe];
            if (giSampled(pv)) { v = pv; m = 2u; }
        }
    }
    value[e] = v;
    meta[e] = m;
}
static uint giVisiblePart(constant GiParams& p) { return p.counts.x - p.counts.x / 4u; }
static void giEnqueueVisible(constant GiParams& p, device uint* list, device atomic_uint* counters, uint e) {
    uint i = atomic_fetch_add_explicit(&counters[0], 1u, memory_order_relaxed);
    if (i < giVisiblePart(p)) list[i] = e;
}

// The sky's radiance in direction d: the atmosphere's (Sky.swift's sky view table, as lit mode's relight integrates it)
// or the prototype's gradient. Below the horizon a ray has left the geometry (past the LOD's edge): the ground.
static float3 giSky(constant GiLight& L, constant SkyFrame& sf, texture2d<float> skyView, float3 d) {
    if (d.y < 0.0) return L.ground.rgb;
    if (L.skyView.w > 0.5) return skyLuminance(sf, d, skyView) * L.skyView.rgb;
    return mix(L.horizon.rgb, L.zenith.rgb, sqrt(d.y));
}
// Irradiance from vanilla's block light level: its lightmap curve, f / (4 - 3f).
static float3 giBlockLight(constant GiParams& p, uint level) {
    float f = float(level) / 15.0;
    return p.blockLight.rgb * (f / (4.0 - 3.0 * f));
}
static float3 giSunOn(constant GiParams& p, constant GiLight& L, uint face) {
    return L.sun.rgb * (p.sun.w * max(0.0, dot(kGiNormal[face], p.sun.xyz)));
}

struct GiSurface { float3 rel; uint face; };

static float3 giRelAt(constant GiParams& p, uint2 q, float z) {
    float2 uv = (float2(q) + 0.5) / p.sizes.xy;
    float4 h = p.invViewProj * float4(uv * 2.0 - 1.0, z, 1.0);
    return h.xyz / h.w;
}
// The camera-relative position and axis-aligned face of the surface at pixel q, as rt_shadow finds them: the normal
// from the neighbors on the same surface (closest depth) on each axis, snapped to the nearest axis.
static bool giSurfaceAt(constant GiParams& p, depth2d<float, access::read> depth, uint2 q, thread GiSurface& s) {
    uint2 full = uint2(p.sizes.xy);
    float d = depth.read(q);
    if (d <= 0.0) return false;   // sky: reverse-Z puts the far plane at 0
    float3 pos = giRelAt(p, q, d);
    uint2 rx = uint2(min(q.x + 1, full.x - 1), q.y), lx = uint2(q.x > 0 ? q.x - 1 : 0, q.y);
    uint2 dy = uint2(q.x, min(q.y + 1, full.y - 1)), uy = uint2(q.x, q.y > 0 ? q.y - 1 : 0);
    float drx = depth.read(rx), dlx = depth.read(lx), ddy = depth.read(dy), duy = depth.read(uy);
    // At the image's edge one neighbor is the pixel itself: take the other.
    bool useR = rx.x != q.x && (lx.x == q.x || abs(drx - d) < abs(dlx - d));
    bool useD = dy.y != q.y && (uy.y == q.y || abs(ddy - d) < abs(duy - d));
    float3 ex = useR ? giRelAt(p, rx, drx) - pos : pos - giRelAt(p, lx, dlx);
    float3 ey = useD ? giRelAt(p, dy, ddy) - pos : pos - giRelAt(p, uy, duy);
    float3 n = cross(ey, ex);
    if (dot(n, -pos) < 0.0) n = -n;
    float3 an = abs(n);
    uint axis = an.x > an.y && an.x > an.z ? 0u : (an.y > an.z ? 1u : 2u);
    s.rel = pos;
    s.face = axis * 2u + (n[axis] < 0.0 ? 1u : 0u);
    return true;
}

// Clears the per-frame counters.
kernel void gi_begin(device atomic_uint* counters [[buffer(8)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    atomic_store_explicit(&counters[0], 0u, memory_order_relaxed);
    atomic_store_explicit(&counters[1], 0u, memory_order_relaxed);
    atomic_store_explicit(&counters[16], 0u, memory_order_relaxed);
    atomic_store_explicit(&counters[18], 0u, memory_order_relaxed);
}

static uint giRequestCell(constant GiParams& p, device atomic_uint* check, device uint4* keys, device atomic_uint* stamp, device half4* value,
                          device uint* meta, device atomic_uint* counters, float3 f, int3 air, uint face, uint level, thread uint& listed) {
    listed = ~0u;
    uint2 k = giKey(giCellOf(air, level), face, level);
    bool created;
    int e = giClaim(check, giBucketOf(p, k), giPrint(k), created);
    if (e < 0) { atomic_fetch_add_explicit(&counters[3], 1u, memory_order_relaxed); return 0u; }
    if (created) {
        giInit(p, check, keys, value, meta, uint(e), k, air, face, level, giRepPoint(p.camBlock.xyz, f, air, face, level));
        atomic_fetch_add_explicit(&counters[2], 1u, memory_order_relaxed);
    }
    // Seen this frame; the first thread to see it may list it (in `listed`). When more cells are visible than their part
    // of the list holds, a random share (sched.x, from last frame's count) is listed, a different one each frame:
    // appending all of them would fill the list with the same cells every frame (the first in dispatch order). Cells
    // without samples (new ones) are always listed. Most touches come after the first (a cell takes about six request
    // samples a frame): a plain read sees that without an exchange (it may be stale, never ahead: the exchange decides).
    uint frame = uint(p.origin.w);
    if (((device uint*)stamp)[e] == frame) return 0u;
    if (atomic_exchange_explicit(&stamp[e], frame, memory_order_relaxed) == frame) return 0u;
    if (!giSampled(value[e]) || giMix(uint(e) ^ giMix(frame)) <= p.sched.x) listed = uint(e);
    return 1u;
}

static uint4 giResolveSample(constant GiParams& p, const device uint* check, const device half4* value, constant GiLight& L,
                             depth2d<float, access::read> depth, uint2 hs);\(clEnabled ? "\nstatic float3 giResolveBlock(constant GiParams& p, const device uint* check, const device half4* value, const device half4* bv, depth2d<float, access::read> depth, uint2 hs);" : "")

// Per sample x sample block of pixels (4 x 4), the request and the resolve, in one pass over the depth buffer:
// - the request: one pixel of the block (a different one each frame) finds or creates its surface's cell, marks it seen
//   and lists it for an update. Near a level boundary the next level's cell too, since the resolve blends the two. A
//   cell 4 x 4 pixels or larger is touched every frame, so while the budget allows, visible cells update every frame.
// - the resolve: the block's 2 x 2 half-resolution samples (giResolveSample), from the cells as last frame's update left
//   them. The two passes read the whole depth buffer each (31 MB at the panel's resolution: every one of its cache lines,
//   though the request needs one pixel in 16) and the same visible cells; done together the light no longer waits for
//   this frame's update, which can then run beside the frame's later passes. A cell's new light shows a frame later.
kernel void gi_request(constant GiParams& p [[buffer(1)]],
                       device atomic_uint* check [[buffer(2)]],
                       device uint4* keys [[buffer(3)]],
                       device atomic_uint* stamp [[buffer(4)]],
                       device half4* value [[buffer(5)]],
                       device uint* meta [[buffer(6)]],
                       device uint* list [[buffer(7)]],
                       device atomic_uint* counters [[buffer(8)]],
                       constant GiLight& L [[buffer(15)]],
                       depth2d<float, access::read> depth [[texture(0)]],
                       texture2d<uint, access::write> out [[texture(1)]],\(clEnabled ? "\n                       const device half4* clBounce [[buffer(19)]], texture2d<float, access::write> outB [[texture(5)]]," : "")
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(p.sizes.z) || gid.y >= uint(p.sizes.w)) return;
    uint2 full = uint2(p.sizes.xy);
    uint2 q = min(gid * p.sample.x + p.sample.yz, full - 1);
    GiSurface s;
    uint seen = 0u, l0 = ~0u, l1 = ~0u;
    // (Vanilla's clouds write depth: no cells for them.)
    if (giSurfaceAt(p, depth, q, s) && length(s.rel) <= p.limits.x && !(s.rel.y > p.rays.w - 0.1 && s.rel.y < p.rays.w + 4.1)) {
        float lf = giLevelF(p, length(s.rel));
        uint level = min(uint(lf), GI_MAX_LEVEL);
        float3 f = p.camFrac.xyz + s.rel;
        int3 air = giAirBlock(p.camBlock.xyz, f, s.face);
        seen = giRequestCell(p, check, keys, stamp, value, meta, counters, f, air, s.face, level, l0);
        if (lf - float(level) > 0.75 && level < GI_MAX_LEVEL) {
            seen += giRequestCell(p, check, keys, stamp, value, meta, counters, f, air, s.face, level + 1u, l1);
        }
    }
    // The visible cells seen this frame (one atomic per SIMD group), for next frame's share.
    uint n = simd_sum(seen);
    if (simd_is_first() && n > 0u) atomic_fetch_add_explicit(&counters[18], n, memory_order_relaxed);
    // The cells to list, one atomic per SIMD group too (a frame lists up to 12 K: one counter takes them all).
    uint nl = (l0 != ~0u ? 1u : 0u) + (l1 != ~0u ? 1u : 0u), total = simd_sum(nl);
    if (total > 0u) {
        uint base = 0u;
        if (simd_is_first()) base = atomic_fetch_add_explicit(&counters[0], total, memory_order_relaxed);
        uint i = simd_broadcast_first(base) + simd_prefix_exclusive_sum(nl), part = giVisiblePart(p);
        if (l0 != ~0u) {
            if (i < part) list[i] = l0;
            i++;
        }
        if (l1 != ~0u && i < part) list[i] = l1;
    }
    // The resolve's samples of this block.
    uint2 hsize = uint2(out.get_width(), out.get_height());
    for (uint j = 0; j < 4u; j++) {
        uint2 hs = gid * 2u + uint2(j & 1u, j >> 1);
        if (hs.x < hsize.x && hs.y < hsize.y) out.write(giResolveSample(p, (const device uint*)check, value, L, depth, hs), hs);\(clEnabled ? "\n        if (hs.x < outB.get_width() && hs.y < outB.get_height()) outB.write(float4(giResolveBlock(p, (const device uint*)check, value, clBounce, depth, hs), 1.0), hs);" : "")
    }
}

// A slice of the table: evict cells unseen for too long, and list cells that aren't visible (the request lists those)
// every 8th time the sweep passes them (by slot), or whenever they have no samples yet (cells bounce rays created).
kernel void gi_schedule(constant GiParams& p [[buffer(1)]],
                        device atomic_uint* check [[buffer(2)]],
                        device uint* stamp [[buffer(4)]],
                        device half4* value [[buffer(5)]],
                        device uint* meta [[buffer(6)]],
                        device uint* list [[buffer(7)]],
                        device atomic_uint* counters [[buffer(8)]],
                        uint tid [[thread_position_in_grid]]) {
    if (tid >= p.counts.z) return;
    uint slot = (p.counts.y + tid) & ((uint(p.camBlock.w) + 1u) * GI_BUCKET - 1u);
    bool listed = false;
    if (atomic_load_explicit(&check[slot], memory_order_relaxed) != 0u) {
        uint age = uint(p.origin.w) - stamp[slot];
        if (age > uint(p.blockLight.w)) {
            // Zeroed (no samples, giSampled) before the slot is freed, so a cell that claims it later never reads another
            // cell's light.
            value[slot] = half4(0.0h);
            meta[slot] = 0u;
            atomic_store_explicit(&check[slot], 0u, memory_order_relaxed);
            atomic_fetch_add_explicit(&counters[4], 1u, memory_order_relaxed);
        } else {
            listed = age > GI_RECENT && (((slot + p.counts.w) & 7u) == 0u || !giSampled(value[slot]));
        }
    }
    // One atomic per SIMD group for the list's slots, in the list's last quarter.
    uint n = simd_sum(listed ? 1u : 0u);
    uint first = 0u;
    if (simd_is_first() && n > 0u) first = atomic_fetch_add_explicit(&counters[16], n, memory_order_relaxed);
    uint i = simd_broadcast_first(first) + simd_prefix_exclusive_sum(listed ? 1u : 0u);
    if (listed && i < p.counts.x / 4u) list[giVisiblePart(p) + i] = slot;
}

// Threads per cell in the update: its samples are split among them (one each at the default 4), so each thread's chain
// of dependent rays is a quarter as long and four times as many rays are in flight (16 K cells are only a few hundred
// SIMD groups on the whole GPU; measured offline, the stage took 0.2 ms less, 15%). A power of two, at most 4, so a
// cell's lanes sit side by side in one SIMD group.
static uint giLanes(uint spp) { return spp >= 4u ? 4u : (spp >= 2u ? 2u : 1u); }

// The update's dispatch size from the two parts of the list (each capped at its share of the budget).
kernel void gi_args(constant GiParams& p [[buffer(1)]], device atomic_uint* counters [[buffer(8)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint nv = min(atomic_load_explicit(&counters[0], memory_order_relaxed), giVisiblePart(p));
    uint n = nv + min(atomic_load_explicit(&counters[16], memory_order_relaxed), p.counts.x / 4u);
    atomic_store_explicit(&counters[1], n, memory_order_relaxed);
    atomic_store_explicit(&counters[17], nv, memory_order_relaxed);
    atomic_store_explicit(&counters[19], atomic_load_explicit(&counters[18], memory_order_relaxed), memory_order_relaxed);
    atomic_store_explicit(&counters[8], max((n * giLanes(p.sample.w) + 63u) / 64u, 1u), memory_order_relaxed);
    atomic_store_explicit(&counters[9], 1u, memory_order_relaxed);
    atomic_store_explicit(&counters[10], 1u, memory_order_relaxed);
}

// Per triangle (primitive data): material | face << 8 | block light << 11 | water depth or sky cover << 15.
static uint giPrim(const device void* d) { return *(const device uint*)d; }

\(clEnabled ? giClBounceHeader : "")kernel void gi_update(instance_acceleration_structure accel [[buffer(0)]],
                      constant GiParams& p [[buffer(1)]],
                      device atomic_uint* check [[buffer(2)]],
                      device uint4* keys [[buffer(3)]],
                      device uint* stamp [[buffer(4)]],
                      device half4* value [[buffer(5)]],
                      device uint* meta [[buffer(6)]],
                      const device uint* list [[buffer(7)]],
                      device atomic_uint* counters [[buffer(8)]],
                      constant float4* mats [[buffer(9)]],
                      constant GiLight& L [[buffer(15)]],
                      constant SkyFrame& sf [[buffer(16)]],
                      texture2d<float> skyView [[texture(6)]]\(clEnabled ? ",\n                      device half4* clBounce [[buffer(19)]], texture3d<float> clRGB [[texture(7)]], texture3d<float> clAux [[texture(8)]], constant GiCl& gcl [[buffer(20)]]" : "")
                      GI_TILES_ARG,
                      uint tid [[thread_position_in_grid]],
                      ushort simdLane [[thread_index_in_simdgroup]]) {
    // giLanes threads per cell: lane k takes the cell's samples k, k + lanes, ...; the first lane sums them and writes.
    uint lanes = giLanes(p.sample.w), lane = tid & (lanes - 1u), ci = tid / lanes;
    if (ci >= atomic_load_explicit(&counters[1], memory_order_relaxed)) return;
    uint nv = atomic_load_explicit(&counters[17], memory_order_relaxed);
    uint e = list[ci < nv ? ci : giVisiblePart(p) + (ci - nv)];
    // What decides the cell's samples, as its first lane saw it, so its lanes agree (they sum their samples in the end).
    ushort first = simdLane - ushort(lane);
    uint live = simd_shuffle(atomic_load_explicit(&check[e], memory_order_relaxed), first);
    if (live == 0u) return;   // evicted since it was listed
    uint4 k = keys[e];
    uint face = (k.y >> 25) & 7u, level = k.y >> 28;
    if (face > 5u || level > GI_MAX_LEVEL) return;
    uint m = simd_shuffle(meta[e], first);
    uint frame = uint(p.origin.w);
    bool visible = frame - simd_shuffle(stamp[e], first) <= GI_RECENT;
    int s = 1 << level;
    int3 c = giUnkey(k.xy, int3(giFloorShift(p.origin.x, level), 0, giFloorShift(p.origin.z, level)));
    // The cell's corner relative to the structure's origin, in blocks.
    float3 corner = float3(giCorner(c, level) - p.origin.xyz);
    float3 n = kGiNormal[face];
    uint a = face >> 1;
    uint2 t = giTangents(face);
    float3 camRel = float3(p.camBlock.xyz - p.origin.xyz) + p.camFrac.xyz;
    bool verify = (m & GI_VERIFY) != 0u;
    uint seq = m >> 16;
    // Independent shifts of the sequence per cell for the point on the face, the bounce and the point on the sun.
    uint2 shiftA = uint2(giMix(k.x), giMix(k.y ^ 0x68bc21ebu)), shiftB = uint2(giMix(shiftA.x), giMix(shiftA.y));
    uint2 shiftC = uint2(giMix(shiftB.x), giMix(shiftB.y));
    float3 sunT1 = normalize(cross(p.sun.xyz, float3(0, 0, 1)));
    float3 sunT2 = cross(p.sun.xyz, sunT1);
    float offset = p.limits.y * (1.0 + 0.1 * float(s));
    // The LOD's faces are counterclockwise seen from outside (lodFaceCorners); for Metal's ray tracing that is the
    // clockwise winding (gi_test_rays checks it: counterclockwise reported every face hit from outside as a back face).
    intersector<triangle_data, instancing> bounce;
    bounce.set_triangle_front_facing_winding(winding::clockwise);
    intersector<triangle_data, instancing> probe;
    probe.set_triangle_front_facing_winding(winding::clockwise);
    probe.set_triangle_cull_mode(triangle_cull_mode::back);
    intersector<instancing> shadow;
    shadow.accept_any_intersection(true);
    shadow.assume_geometry_type(geometry_type::triangle);
    float3 sumE = 0.0;\(clEnabled ? "\n    float3 sumB = 0.0;" : "")
    float sumSun = 0.0;
    uint valid = 0, sunSamples = 0, rays = 0, cached = 0, uncached = 0, spawned = 0, repUsed = 0;
    for (uint i = lane; i < p.sample.w; i += lanes) {
        uint si = seq + i;
        float2 uv = giR2(si, shiftA);
        float3 o = corner;
        o[t.x] += uv.x * float(s);
        o[t.y] += uv.y * float(s);
        if (level == 0u && !verify) {
            // A block face lies on the side of its air block toward the solid.
            o[a] += (face & 1u) == 0u ? 0.0 : 1.0;
        } else {
            // A coarser cell holds faces on any of its planes (and after an edit the face may be gone): a probe from
            // the cell's far side back along the normal finds the surface, passing through solid (back faces culled).
            // It reaches exactly the cell's near side (where flat ground lies), and no further plane can be there.
            float3 st = o;
            st[a] += (face & 1u) == 0u ? float(s) : 0.0;
            ray pr(st, -n, 0.001, float(s) + 0.01);
            auto ph = probe.intersect(pr, accel, 0xFF);
            rays++;
            if (ph.type != intersection_type::none && ((GI_HIT(ph) >> 8) & 7u) == face) {
                o = st - n * ph.distance;
            } else if (verify) {
                continue;   // checking whether the surface is still there: no fallback
            } else {
                // The faces don't cover this part of the cell: the point where the cell was first seen, moved by up
                // to a sixteenth of a block across the face so samples don't all start at one point.
                o = corner;
                o[t.x] += (float(k.z & 255u) + 0.5) / 256.0 * float(s) + (uv.x - 0.5) * 0.125;
                o[t.y] += (float((k.z >> 8) & 255u) + 0.5) / 256.0 * float(s) + (uv.y - 0.5) * 0.125;
                o[a] += float(k.z >> 16);
                repUsed++;
            }
        }
        o += n * offset;
        valid++;
        // How much of the sun's disk the face sees (a different point of the disk per sample).
        if (p.sun.w > 0.0 && dot(n, p.sun.xyz) > 0.0) {
            float2 dk = giR2(si, shiftC);
            float r = p.rays.x * sqrt(dk.x), ang = 2.0 * kGiPi * dk.y;
            ray sr(o, normalize(p.sun.xyz + sunT1 * (r * cos(ang)) + sunT2 * (r * sin(ang))), 0.0, p.rays.y);
            rays++;
            sunSamples++;
            if (shadow.intersect(sr, accel, 0xFF).type == intersection_type::none) sumSun += 1.0;
        }
        // One cosine-weighted bounce (the irradiance estimate is pi times the radiance it finds): the sky where it
        // escapes, else the light leaving the surface it reaches, taken from that surface's own cell.
        float2 dd = giR2(si, shiftB);
        float rr = sqrt(dd.x), phi = 2.0 * kGiPi * dd.y;
        float3 dir = n * sqrt(max(0.0, 1.0 - dd.x));
        dir[t.x] += rr * cos(phi);
        dir[t.y] += rr * sin(phi);
        ray br(o, dir, 0.0, p.limits.w);
        rays++;
        auto bh = bounce.intersect(br, accel, 0xFF);
        if (bh.type == intersection_type::none) { sumE += kGiPi * giSky(L, sf, skyView, dir); continue; }
        if (!bh.triangle_front_facing) continue;   // the back of a surface: the ray is inside terrain, no light
        uint pd = GI_HIT(bh);
        uint hm = pd & 255u, hf = min((pd >> 8) & 7u, 5u), hbl = (pd >> 11) & 15u;
        float3 hp = o + dir * bh.distance;
        int3 hair = giAirBlock(p.origin.xyz, hp, hf);
        uint hl = min(uint(giLevelF(p, distance(hp, camRel))), GI_MAX_LEVEL);
        uint2 hk = giKey(giCellOf(hair, hl), hf, hl);
        uint hb = giBucketOf(p, hk), hfp = giPrint(hk);
        int he;
        if (visible) {
            // A visible cell's rays create the cells they reach (behind the camera, around corners), so light keeps
            // bouncing off them; those are kept while rays keep reaching them, as cells seen a while ago.
            bool created;
            he = giClaim(check, hb, hfp, created);
            if (created) {
                giInit(p, check, keys, value, meta, uint(he), hk, hair, hf, hl, giRepPoint(p.origin.xyz, hp, hair, hf, hl));
                stamp[he] = frame - GI_RECENT - 1u;
                spawned++;
            } else if (he < 0) {
                atomic_fetch_add_explicit(&counters[3], 1u, memory_order_relaxed);
            }
        } else {
            he = giFindA(check, hb, hfp);
        }
        float3 eh;
        float3 sunH = giSunOn(p, L, hf);
        half4 hv = he >= 0 ? value[he] : half4(0.0h);
        if (giSampled(hv)) {
            eh = float3(hv.rgb) + (float(hv.a) - 1.0) * sunH;
            if (frame - stamp[he] > uint(p.blockLight.w) / 2u) stamp[he] = frame - GI_RECENT - 1u;
            cached++;
        } else {
            // Nothing cached there yet: its direct sun from one more ray, and half the open sky.
            float vis = 0.0;
            if (sunH.x + sunH.y + sunH.z > 0.0) {
                ray hs(hp + kGiNormal[hf] * offset, p.sun.xyz, 0.0, p.rays.y);
                rays++;
                vis = shadow.intersect(hs, accel, 0xFF).type == intersection_type::none ? 1.0 : 0.0;
            }
            eh = vis * sunH + 0.5 * L.open[hf].rgb;
            uncached++;
        }
        eh += giBlockLight(p, hbl);
        sumE += mats[hm * 4u + giFaceClass(hf)].rgb * eh + mats[hm * 4u + 3u].rgb * p.look.y;\(clEnabled ? "\n        // Colored block light (ColoredLightBounce.swift): the volume's at the hit, and the light it bounced there.\n        sumB += mats[hm * 4u + giFaceClass(hf)].rgb * (giClBlock(gcl, clRGB, clAux, hp, hf) + (he >= 0 && giClOwn(clBounce[he], hk) ? float3(clBounce[he].rgb) : float3(0.0)));" : "")
    }
    // The cell's lanes sum what they found (every lane of the cell gets here: they return early or not together).
    for (uint d = 1u; d < lanes; d <<= 1) {
        ushort x = ushort(d);
        sumE += simd_shuffle_xor(sumE, x);\(clEnabled ? "\n        sumB += simd_shuffle_xor(sumB, x);" : "")
        sumSun += simd_shuffle_xor(sumSun, x);
        valid += simd_shuffle_xor(valid, x);
        sunSamples += simd_shuffle_xor(sunSamples, x);
        rays += simd_shuffle_xor(rays, x);
        cached += simd_shuffle_xor(cached, x);
        uncached += simd_shuffle_xor(uncached, x);
        spawned += simd_shuffle_xor(spawned, x);
        repUsed += simd_shuffle_xor(repUsed, x);
    }
    if (lane != 0u) return;
    atomic_fetch_add_explicit(&counters[11], rays, memory_order_relaxed);
    if (cached > 0u) atomic_fetch_add_explicit(&counters[12], cached, memory_order_relaxed);
    if (uncached > 0u) atomic_fetch_add_explicit(&counters[13], uncached, memory_order_relaxed);
    if (spawned > 0u) atomic_fetch_add_explicit(&counters[14], spawned, memory_order_relaxed);
    if (repUsed > 0u) atomic_fetch_add_explicit(&counters[15], repUsed, memory_order_relaxed);
    if (valid == 0u) {
        atomic_fetch_add_explicit(&counters[5], 1u, memory_order_relaxed);
        if (verify) {
            // An edit removed the face: evict the cell (zeroed first, as in gi_schedule).
            value[e] = half4(0.0h);
            meta[e] = 0u;
            atomic_store_explicit(&check[e], 0u, memory_order_relaxed);
            atomic_fetch_add_explicit(&counters[4], 1u, memory_order_relaxed);
        } else {
            meta[e] = m | GI_NOSURF;
        }
        return;
    }
    float3 est = sumE / float(valid);
    // Spatial reuse (off by default, METALMC_GISPATIAL): the cell's coplanar neighbors (same face, level and plane) each
    // average many samples already; blending them into this update's estimate, each weighing look.w of it, cuts the
    // noise of cells lit by small bright sources (a sunlit patch seen through a window). Walls stop it (beside a floor
    // cell at the foot of a wall there is no floor cell on the same plane), but it feeds the bounces, so it biases
    // enclosed rooms bright (giSpatial).
    if (p.look.w > 0.0) {
        float3 nsum = 0.0;
        float nw = 0.0;
        for (uint d = 0; d < 4u; d++) {
            int3 cc = c;
            cc[d < 2u ? t.x : t.y] += (d & 1u) == 0u ? 1 : -1;
            uint2 nk = giKey(cc, face, level);
            int ne = giFindA(check, giBucketOf(p, nk), giPrint(nk));
            if (ne < 0) continue;
            half4 nv = value[ne];
            if (!giSampled(nv)) continue;
            nsum += float3(nv.rgb);
            nw += 1.0;
        }
        est = (est + p.look.w * nsum) / (1.0 + p.look.w * nw);
    }
    // A running mean until the average holds `history` updates, then an exponential one.
    half4 old = value[e];
    uint cnt = giSampled(old) ? (m & 255u) : 0u, cap = uint(p.rays.z);
    float alpha = 1.0 / float(min(cnt, cap) + 1u);
    float3 rgb = mix(float3(old.rgb), est, alpha);
    // The sun's share only changes when it was tested (a face turned away from it, or night, keeps the last one, or 0
    // if it never had one).
    float oldVis = giSampled(old) ? float(old.a) - 1.0 : 0.0;
    float vis = sunSamples > 0u ? mix(oldVis, sumSun / float(sunSamples), alpha) : oldVis;
    value[e] = half4(half3(rgb), half(1.0 + vis));\(clEnabled ? "\n    // The block channel, averaged alike (a slot that held another cell starts over).\n    {\n        uint2 ek = keys[e].xy;\n        half4 ob = clBounce[e];\n        bool own = giClOwn(ob, ek);\n        float3 nb = mix(own ? float3(ob.rgb) : float3(0.0), sumB / float(valid), own ? alpha : 1.0);\n        clBounce[e] = half4(half3(nb), half(float(giPrint(ek) & 1023u)));\n    }" : "")
    meta[e] = min(cnt + 1u, cap) | (((seq + p.sample.w) & 0xFFFFu) << 16);
}

// Irradiance at a point (camBlock + f, air block `air`) from the 2 x 2 cells nearest it on its face's plane, bilinear;
// cells that don't exist or have no samples yet are left out and the rest renormalized. w: the weight found (0: none).
static float4 giInterp(constant GiParams& p, const device uint* check, const device half4* value, float3 f, int3 air, uint face, uint level) {
    uint2 t = giTangents(face);
    int s = 1 << level;
    int3 c = giCellOf(air, level);
    float3 local = float3(p.camBlock.xyz - giCorner(c, level)) + f;   // relative to the cell's corner
    float2 q = float2(local[t.x], local[t.y]) / float(s) - 0.5;
    float2 fl = floor(q), w = q - fl;
    int2 o = int2(fl);
    // The 4 corners' scans, then their values, with no branch between (their loads in flight together: 0.02 ms less in
    // the resolve than a loop that skipped ahead), then summed in the order the loop had.
    int e[4];
    for (int k = 0; k < 4; k++) {
        int3 cc = c;
        cc[t.x] += o.x + (k & 1);
        cc[t.y] += o.y + (k >> 1);
        uint2 key = giKey(cc, face, level);
        e[k] = giFind(check, giBucketOf(p, key), giPrint(key));
    }
    half4 v[4];
    for (int k = 0; k < 4; k++) v[k] = value[max(e[k], 0)];
    float3 sum = 0.0;
    float ws = 0.0;
    for (int k = 0; k < 4; k++) {
        float wt = ((k & 1) == 0 ? 1.0 - w.x : w.x) * ((k >> 1) == 0 ? 1.0 - w.y : w.y);
        if (wt <= 0.0 || e[k] < 0 || !giSampled(v[k])) continue;
        sum += wt * float3(v[k].rgb);
        ws += wt;
    }
    return float4(ws > 0.0 ? sum / ws : float3(0.0), ws);
}

// Irradiance at `f` on a face (air block `air`) at real level lf: bilinear (giInterp), blended toward the next level in
// the last quarter of the level's range so level boundaries don't show as seams.
static float4 giIrradiance(constant GiParams& p, const device uint* check, const device half4* value, float3 f, int3 air, uint face, float lf) {
    uint level = min(uint(lf), GI_MAX_LEVEL);
    float4 e = giInterp(p, check, value, f, air, face, level);
    float b = saturate((lf - float(level) - 0.75) * 4.0);
    if (b > 0.0 && level < GI_MAX_LEVEL) {
        float4 e1 = giInterp(p, check, value, f, air, face, level + 1u);
        if (e1.w > 0.0) e = e.w > 0.0 ? float4(mix(e.rgb, e1.rgb, b), max(e.w, e1.w)) : e1;
    }
    return e;
}

// A float as an unsigned small float of RG11B10Float (5 exponent bits, `mbits` mantissa bits), truncated as the
// texture unit's conversion does on the M3 (not rounded: gi_test_pack checks every case against it). Negative values
// and -0 give 0; x must be finite.
static uint giPackSmallFloat(float x, uint mbits) {
    uint b = as_type<uint>(x);
    if ((b >> 31) != 0u) return 0u;
    if (b < 0x38800000u) return uint(x * float(1u << (14u + mbits)));   // under 2^-14: the small float's subnormals
    return min((b - 0x38000000u) >> (23u - mbits), (31u << mbits) - 1u);   // the exponent rebiased from 127 to 15
}
// The irradiance as RG11B10Float's bits (giUnpackIrradiance reads them back).
static uint giPackIrradiance(float3 c) {
    return giPackSmallFloat(c.r, 6u) | (giPackSmallFloat(c.g, 6u) << 11) | (giPackSmallFloat(c.b, 5u) << 22);
}

// Indirect irradiance (sky and bounced light) at half resolution: one sample per 2 x 2 pixels (`hs`), at the block's
// nearest surface, with its code word (giUpsampleHeader) in one RG32Uint texel so the lighting pass can upsample it by
// face and plane without the depth buffer (giUpsample). At the panel's resolution a full-resolution resolve cost 1.4-1.6
// ms, mostly its lookups (4 per pixel) and its output's bandwidth; a cell spans 10 pixels or more, so a quarter of the
// lookups loses nothing but the edges, which the upsample keeps. The cells' light is per unit of daylight: times this
// frame's (GiLight.scale).
static uint4 giResolveSample(constant GiParams& p, const device uint* check, const device half4* value, constant GiLight& L,
                             depth2d<float, access::read> depth, uint2 hs) {
    uint2 full = uint2(p.sizes.xy), base = hs * 2u;
    // The block's nearest surface (reverse-Z: the largest depth), so foreground edges get their own sample.
    uint best = 0u;
    float bd = -1.0;
    for (uint i = 0; i < 4u; i++) {
        float d = depth.read(min(base + uint2(i & 1u, i >> 1), full - 1));
        if (d > bd) { bd = d; best = i; }
    }
    uint2 q = min(base + uint2(best & 1u, best >> 1), full - 1);
    GiSurface s;
    if (bd <= 0.0 || !giSurfaceAt(p, depth, q, s) || length(s.rel) > p.limits.x) return uint4(0u);
    float3 f = p.camFrac.xyz + s.rel;
    float4 e = giIrradiance(p, check, value, f, giAirBlock(p.camBlock.xyz, f, s.face), s.face, giLevelF(p, length(s.rel)));
    uint pc = uint(as_type<ushort>(half(s.rel[s.face >> 1]))) | (s.face << 16);
    return uint4(e.w > 0.0 ? giPackIrradiance(e.rgb * L.scale.rgb) : 0u, pc | (e.w > 0.0 ? GI_CODE_DATA : GI_CODE_EMPTY), 0u, 0u);
}

\(clEnabled ? giClResolveSource : "")// The resolve alone, on the cells as they stand (the debug views and tests; a frame's comes with its request, gi_request).
kernel void gi_resolve(constant GiParams& p [[buffer(1)]],
                       const device uint* check [[buffer(2)]],
                       const device half4* value [[buffer(5)]],
                       constant GiLight& L [[buffer(15)]],
                       depth2d<float, access::read> depth [[texture(0)]],
                       texture2d<uint, access::write> out [[texture(1)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    out.write(giResolveSample(p, check, value, L, depth, gid), gid);
}

static float3 giTonemap(float3 c) {
    c = saturate((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14));   // ACES fit (Narkowicz)
    return pow(c, float3(1.0 / 2.2));
}

// Debug view (limits.z): 0 lit (sun from the per-pixel shadow, sky and bounced light from the cache, block light),
// 1 the cache's irradiance alone (magenta: none), 2 cells (a color per cell, dimmed where it has no samples), 3 samples
// (red none, green converged, magenta: no cell), 4 without the cache (sun, vanilla's sky light as flat ambient, block
// light), 5 bounced and sky light only (albedo times the cache); lit mode's relight (Lit.swift, without its AO and
// lightmap: block light by vanilla's curve, sRGB-encoded and clipped like its SDR output) 6 with its sky term (the open
// sky's light on the face times the sky light level's curve) and 7 with the cache's light in its place where the cache
// has data; 9 what lit mode's relight takes from the cache, unencoded for comparisons (giUpsample on the face the
// primitive data gives, as the relight's G-buffer does, without lookups of its own: the irradiance where it matched,
// -1 where nothing did, -2 where the cells have no samples yet, -3 for the sky). `gi` is gi_resolve's output; `prim`
// holds each pixel's primitive data (material, face, block light, sky cover) and `sunVis` its sun visibility; in the
// game, 1 x 1 stand-ins give white and full sun.
kernel void gi_debug_view(constant GiParams& p [[buffer(1)]],
                          const device uint* check [[buffer(2)]],
                          const device half4* value [[buffer(5)]],
                          const device uint* meta [[buffer(6)]],
                          constant float4* mats [[buffer(9)]],
                          constant GiLight& L [[buffer(15)]],
                          constant SkyFrame& sf [[buffer(16)]],
                          texture2d<float> skyView [[texture(6)]],
                          depth2d<float, access::read> depth [[texture(0)]],
                          texture2d<uint> gi [[texture(1)]],
                          texture2d<uint, access::read> prim [[texture(2)]],
                          texture2d<half, access::read> sunVis [[texture(3)]],
                          texture2d<float, access::write> out [[texture(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(p.sizes.x) || gid.y >= uint(p.sizes.y)) return;
    uint mode = uint(p.limits.z);
    GiSurface s;
    if (!giSurfaceAt(p, depth, gid, s)) {
        float3 d = normalize(giRelAt(p, gid, 1.0));
        float3 sky = giSky(L, sf, skyView, d) * kGiPi * L.scale.rgb;
        out.write(mode == 9u ? float4(-3.0) : float4(mode >= 6u ? skyEncode(saturate(sky * p.look.x / kGiPi)) : giTonemap(sky * p.look.x), 1.0), gid);
        return;
    }
    bool dummy = prim.get_width() == 1u;
    uint pd = dummy ? 0u : prim.read(gid).r;
    if (mode == 9u) {
        float4 u = giUpsample(gi, giUpsampleTexel(gi, gid), gid, dummy ? s.face : min((pd >> 8) & 7u, 5u), s.rel);
        out.write(u.w > 0.0 ? float4(u.rgb, 1.0) : float4(u.w < 0.0 ? -2.0 : -1.0), gid);
        return;
    }
    uint mat = pd & 255u, bl = (pd >> 11) & 15u, cover = (pd >> 15) & 15u;
    float3 albedo = dummy ? float3(0.6) : mats[mat * 4u + giFaceClass(s.face)].rgb;
    float3 emit = dummy ? float3(0.0) : mats[mat * 4u + 3u].rgb * p.look.z;
    float vis = sunVis.get_width() == 1u ? 1.0 : float(sunVis.read(gid).r);
    float3 sunI = giSunOn(p, L, s.face) * L.scale.rgb * vis;
    float4 up = giUpsample(gi, giUpsampleTexel(gi, gid), gid, s.face, s.rel);
    if (up.w == 0.0 && length(s.rel) <= p.limits.x) {
        // No half-resolution sample on this pixel's face and plane (a face a pixel or two wide): its own lookups, which
        // only these few pixels pay for.
        float3 f = p.camFrac.xyz + s.rel;
        up = giIrradiance(p, check, value, f, giAirBlock(p.camBlock.xyz, f, s.face), s.face, giLevelF(p, length(s.rel)));
        up.rgb *= L.scale.rgb;
    }
    float3 ir = up.rgb;
    bool has = up.w > 0.0;
    float3 ambient = L.open[2].rgb * L.scale.rgb * (s.face == 2u ? 1.0 : (s.face == 3u ? 0.3 : 0.6));
    if (mode >= 6u) {
        // Lit mode's relight: sun, the sky term (or the cache's light), block light by vanilla's curve (no lightmap).
        float sky = float(15u - min(cover, 15u)), b = float(bl) / 15.0;
        float fall = skyDecode(float3((sky / 15.0) / (4.0 - 3.0 * (sky / 15.0)))).x;
        float3 skyTerm = mode == 7u && has ? ir : L.open[s.face].rgb * L.scale.rgb * fall;
        float3 lin = albedo * (sunI + skyTerm + float3(b / (4.0 - 3.0 * b) * (b / (4.0 - 3.0 * b))));
        out.write(float4(skyEncode(saturate(lin * p.look.x)), 1.0), gid);
        return;
    }
    float3 c;
    if (mode == 1u) {
        c = has ? ir : float3(1.0, 0.0, 1.0);
    } else if (mode == 2u || mode == 3u) {
        float lf = giLevelF(p, length(s.rel));
        uint level = min(uint(lf), GI_MAX_LEVEL);
        float3 f = p.camFrac.xyz + s.rel;
        uint2 k = giKey(giCellOf(giAirBlock(p.camBlock.xyz, f, s.face), level), s.face, level);
        int e = giFind(check, giBucketOf(p, k), giPrint(k));
        uint cnt = e >= 0 ? (meta[e] & 255u) : 0u;
        if (mode == 2u) {
            uint h = giMix(k.x ^ giMix(k.y));
            c = float3(float(h & 255u), float((h >> 8) & 255u), float((h >> 16) & 255u)) / 255.0 * (cnt > 0u ? 1.0 : 0.3);
        } else {
            float r = float(cnt) / p.rays.z;
            c = e < 0 ? float3(1.0, 0.0, 1.0) : mix(float3(1.0, 0.0, 0.0), float3(0.0, 1.0, 0.0), saturate(r));
        }
        out.write(float4(c, 1.0), gid);
        return;
    } else if (mode == 4u) {
        float f = float(15u - min(cover, 15u)) / 15.0;
        c = albedo * (sunI + ambient * (f / (4.0 - 3.0 * f)) + giBlockLight(p, bl)) + emit;
    } else if (mode == 5u) {
        c = albedo * ir;
    } else {
        float3 ind = has ? ir : ambient;
        c = albedo * (sunI + ind + giBlockLight(p, bl)) + emit;
    }
    out.write(float4(giTonemap(c * p.look.x), 1.0), gid);
}

// Cells an edit may have changed: level-0 cells within `radius` blocks of it restart their averages, and at coarser
// levels the 3 x 3 x 3 cells around it. Those next to it re-check their surface in this frame's update (listed here: once
// their face is gone nothing on screen lists them, and their neighbors would keep reading their stale light).
// edits: world block xyz, radius.
kernel void gi_invalidate(constant GiParams& p [[buffer(1)]],
                          const device uint* check [[buffer(2)]],
                          device uint* meta [[buffer(6)]],
                          device uint* list [[buffer(7)]],
                          device atomic_uint* counters [[buffer(8)]],
                          constant int4* edits [[buffer(10)]],
                          uint2 gid [[thread_position_in_grid]]) {
    int4 ed = edits[gid.y];
    int r = ed.w, side = 2 * r + 1;
    uint near = uint(side * side * side) * 6u;
    uint i = gid.x;
    int3 air;
    uint face, level;
    if (i < near) {
        face = i % 6u;
        uint v = i / 6u;
        air = ed.xyz + int3(int(v % uint(side)) - r, int((v / uint(side)) % uint(side)) - r, int(v / uint(side * side)) - r);
        level = 0u;
    } else {
        uint j = i - near;
        level = 1u + j / 162u;
        if (level > GI_MAX_LEVEL) return;
        uint v = j % 162u;
        face = v % 6u;
        v /= 6u;
        int sl = 1 << level;
        air = ed.xyz + int3(int(v % 3u) - 1, int((v / 3u) % 3u) - 1, int(v / 9u) - 1) * sl;
    }
    uint2 k = giKey(giCellOf(air, level), face, level);
    int e = giFind(check, giBucketOf(p, k), giPrint(k));
    if (e < 0) return;
    uint m = meta[e];
    bool touching = level == 0u && all(abs(air - ed.xyz) <= int3(1));
    uint keep = level == 0u ? 1u : 4u;
    meta[e] = (m & 0xFFFF0000u) | min(m & 255u, keep) | (touching ? GI_VERIFY : 0u);
    if (touching) giEnqueueVisible(p, list, counters, uint(e));
}

// Live and converged cells (counters 6 and 7), for statistics.
kernel void gi_count(constant GiParams& p [[buffer(1)]],
                     const device uint* check [[buffer(2)]],
                     const device uint* meta [[buffer(6)]],
                     device atomic_uint* counters [[buffer(8)]],
                     uint tid [[thread_position_in_grid]]) {
    bool live = tid < (uint(p.camBlock.w) + 1u) * GI_BUCKET && check[tid] != 0u;
    bool conv = live && float(meta[tid] & 255u) >= p.rays.z;
    uint nl = simd_sum(live ? 1u : 0u), nc = simd_sum(conv ? 1u : 0u);
    if (simd_is_first()) {
        if (nl > 0u) atomic_fetch_add_explicit(&counters[6], nl, memory_order_relaxed);
        if (nc > 0u) atomic_fetch_add_explicit(&counters[7], nc, memory_order_relaxed);
    }
}

// Offline test only (mmc_debug_gi_run): what the game's depth buffer and shadows would give, from primary rays: depth
// (reverse-Z, written to a buffer the test copies into a depth texture), primitive data, and a sharp sun shadow.
kernel void gi_test_primary(instance_acceleration_structure accel [[buffer(0)]],
                            constant GiParams& p [[buffer(1)]],
                            constant float4x4& viewProj [[buffer(11)]],
                            device float* depthOut [[buffer(12)]],
                            texture2d<uint, access::write> prim [[texture(2)]],
                            texture2d<half, access::write> sunVis [[texture(3)]]
                            GI_TILES_ARG,
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(p.sizes.x) || gid.y >= uint(p.sizes.y)) return;
    float3 dir = normalize(giRelAt(p, gid, 1.0));
    float3 cam = float3(p.camBlock.xyz - p.origin.xyz) + p.camFrac.xyz;
    ray r(cam, dir, 0.0, 1.0e5);
    intersector<triangle_data, instancing> isect;
    auto h = isect.intersect(r, accel, 0xFF);
    uint idx = gid.y * uint(p.sizes.x) + gid.x;
    if (h.type == intersection_type::none) {
        depthOut[idx] = 0.0;
        prim.write(uint4(0xFFFFFFFFu), gid);
        sunVis.write(half4(1.0h), gid);
        return;
    }
    float3 rel = dir * h.distance;
    float4 clip = viewProj * float4(rel, 1.0);
    depthOut[idx] = clip.z / clip.w;
    uint pd = GI_HIT(h);
    prim.write(uint4(pd), gid);
    uint face = min((pd >> 8) & 7u, 5u);
    float vis = 0.0;
    if (p.sun.w > 0.0 && dot(kGiNormal[face], p.sun.xyz) > 0.0) {
        intersector<instancing> sh;
        sh.accept_any_intersection(true);
        sh.assume_geometry_type(geometry_type::triangle);
        ray sr(cam + rel + kGiNormal[face] * 0.02, p.sun.xyz, 0.0, p.rays.y);
        vis = sh.intersect(sr, accel, 0xFF).type == intersection_type::none ? 1.0 : 0.0;
    }
    sunVis.write(half4(half(vis)), gid);
}

// Offline test only: what a deferred lighting pass would add per pixel to use the cache: giUpsample, and its own
// lookups where no half-resolution sample matches, with the pixel's face from the G-buffer (here the primitive data)
// and its position from one depth read; it writes only the luminance so the output's bandwidth doesn't hide it.
kernel void gi_test_upsample(constant GiParams& p [[buffer(1)]],
                             const device uint* check [[buffer(2)]],
                             const device half4* value [[buffer(5)]],
                             depth2d<float, access::read> depth [[texture(0)]],
                             texture2d<uint> gi [[texture(1)]],
                             texture2d<uint, access::read> prim [[texture(2)]],
                             texture2d<half, access::write> out [[texture(4)]],
                             device atomic_uint* counters [[buffer(8)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(p.sizes.x) || gid.y >= uint(p.sizes.y)) return;
    float d = depth.read(gid);
    uint pd = prim.read(gid).r;
    uint surface = 0u, looked = 0u, none = 0u;
    GiSurface s;
    s.rel = giRelAt(p, gid, d);
    s.face = min((pd >> 8) & 7u, 5u);
    float4 up = float4(0.0);
    if (p.limits.z == 99.0) {
        up = float4(s.rel, float(s.face));   // the baseline: the same reads and write without the cache
    } else if (d > 0.0 && length(s.rel) <= p.limits.x) {
        surface = 1u;
        up = giUpsample(gi, giUpsampleTexel(gi, gid), gid, s.face, s.rel);
        if (up.w < 0.0) {
            none = 1u;
        } else if (up.w == 0.0) {
            looked = 1u;
            if (p.limits.z != 97.0) {   // 97: without the lookups (the lighting pass's own ambient there instead)
                float3 f = p.camFrac.xyz + s.rel;
                up = giIrradiance(p, check, value, f, giAirBlock(p.camBlock.xyz, f, s.face), s.face, giLevelF(p, length(s.rel)));
                none = up.w > 0.0 ? 0u : 1u;
            }
        }
    }
    out.write(half4(half(dot(up.rgb, float3(0.2126, 0.7152, 0.0722)))), gid);
    // Test statistics (counters 20-22): pixels with a surface, that needed the cells' own lookups, that had no data.
    if (p.limits.z == 98.0) {
        uint a = simd_sum(surface), b = simd_sum(looked), c = simd_sum(none);
        if (simd_is_first()) {
            atomic_fetch_add_explicit(&counters[20], a, memory_order_relaxed);
            atomic_fetch_add_explicit(&counters[21], b, memory_order_relaxed);
            atomic_fetch_add_explicit(&counters[22], c, memory_order_relaxed);
        }
    }
}

// Offline self-test: rays (origin, max distance; direction) against the structure, with no culling and with the probe's
// culling. out per ray: hit, face, front facing, distance bits (no culling), then hit and face with back faces culled.
kernel void gi_test_rays(instance_acceleration_structure accel [[buffer(0)]],
                         const device float4* rays [[buffer(13)]],
                         device uint4* out [[buffer(14)]]
                         GI_TILES_ARG,
                         uint tid [[thread_position_in_grid]]) {
    float4 o = rays[2 * tid], d = rays[2 * tid + 1];
    ray r(o.xyz, d.xyz, 0.0, o.w);
    intersector<triangle_data, instancing> a;
    a.set_triangle_front_facing_winding(winding::clockwise);
    auto ha = a.intersect(r, accel, 0xFF);
    intersector<triangle_data, instancing> b;
    b.set_triangle_front_facing_winding(winding::clockwise);
    b.set_triangle_cull_mode(triangle_cull_mode::back);
    auto hb = b.intersect(r, accel, 0xFF);
    bool hitA = ha.type != intersection_type::none, hitB = hb.type != intersection_type::none;
    out[2 * tid] = uint4(hitA ? 1u : 0u, hitA ? (GI_HIT(ha) >> 8) & 7u : 9u, hitA && ha.triangle_front_facing ? 1u : 0u,
                         as_type<uint>(hitA ? ha.distance : -1.0));
    out[2 * tid + 1] = uint4(hitB ? 1u : 0u, hitB ? (GI_HIT(hb) >> 8) & 7u : 9u, 0u, as_type<uint>(hitB ? hb.distance : -1.0));
}

// Lit mode (METALMC_EXP=lit,gi): the frame's GiLight from the relight's sun and sky light. env: lit_env's output (with the
// atmosphere: env[0].w is its scale from the sky view table to these units) or litDaylightEnv's, in the relight's units
// per unit of albedo: the sun's irradiance, then the sky's on +X -X +Y -Y +Z -Z and on faces that aren't axis-aligned.
// opt.x: 1 to take the sky's radiance from the sky view table. The daylight the cells are measured in is the sun's
// irradiance plus the open sky's on a top face; the sky without the table is even (its irradiance on a top is pi times
// its radiance, as litDaylightEnv's sides are half sky and half ground).
kernel void gi_light(constant float4* env [[buffer(16)]], constant float4& opt [[buffer(18)]], device GiLight& L [[buffer(15)]],
                     uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    float3 d = max(env[0].rgb + env[3].rgb, float3(1e-4));
    L.sun = float4(env[0].rgb / d, 0.0);
    L.scale = float4(d, 0.0);
    L.skyView = opt.x > 0.5 ? float4(float3(env[0].w) / d, 1.0) : float4(0.0);
    L.zenith = float4(env[3].rgb / (kGiPi * d), 0.0);
    L.horizon = L.zenith;
    L.ground = float4(env[4].rgb / (kGiPi * d), 0.0);
    for (uint f = 0; f < 6u; f++) L.open[f] = float4(env[1u + f].rgb / d, 0.0);
}

// Offline test only (mmc_debug_gi_profile): what the game's ray-traced shadows do just before the cache's frame
// (RtShadows' rt_shadow: one ray toward the sun per 4 x 4 pixels, from the depth buffer's surface), so the cache's stages
// find the depth buffer and the structures as warm as the game leaves them.
kernel void gi_test_shadow(instance_acceleration_structure accel [[buffer(0)]],
                           constant GiParams& p [[buffer(1)]],
                           depth2d<float, access::read> depth [[texture(0)]],
                           texture2d<half, access::write> out [[texture(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(p.sizes.z) || gid.y >= uint(p.sizes.w)) return;
    uint2 q = min(gid * p.sample.x + p.sample.yz, uint2(p.sizes.xy) - 1u);
    GiSurface s;
    half v = 1.0h;
    if (giSurfaceAt(p, depth, q, s) && dot(kGiNormal[s.face], p.sun.xyz) > -0.05) {
        float3 camRel = float3(p.camBlock.xyz - p.origin.xyz) + p.camFrac.xyz;
        ray r(camRel + s.rel + kGiNormal[s.face] * (0.03 + length(s.rel) * 0.0008), p.sun.xyz, 0.0, p.rays.y);
        intersector<instancing> isect;
        isect.accept_any_intersection(true);
        isect.assume_geometry_type(geometry_type::triangle);
        v = isect.intersect(r, accel, 0xFF).type == intersection_type::none ? 1.0h : 0.0h;
    }
    out.write(half4(v), gid);
}

// Offline test only (mmc_debug_gi_profile): the anti-aliasing resolve's load loop with lit mode's relight in it
// (Taa.swift's taa_resolve, Lit.swift's litRelightPixel), reduced to what the cache changes there, to time its share as
// the game lays the work out: 32 x 32 pixels per threadgroup of 16 x 16 threads, the 34 x 34 loads (a 1-pixel border) in
// the same order (a SIMD group's loads are a row of pixels, as there), each reading the depth and the primitive data
// (the G-buffer's stand-in) and, as the relight does, the cache's texel early and giUpsample on the pixel's face; then
// each pixel's 3 x 3 neighborhood from threadgroup memory. limits.z == 99: without the cache's part (the baseline).
kernel void gi_test_taa(constant GiParams& p [[buffer(1)]],
                        depth2d<float, access::read> depth [[texture(0)]],
                        texture2d<uint> gi [[texture(1)]],
                        texture2d<uint, access::read> prim [[texture(2)]],
                        texture2d<half, access::write> out [[texture(4)]],
                        uint2 lid [[thread_position_in_threadgroup]],
                        uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup half4 tile[34 * 34];
    threadgroup float dtile[34 * 34];
    int2 size = int2(p.sizes.xy);
    int2 base = int2(tgid) * 32 - 1;
    bool withGi = p.limits.z != 99.0;
    for (uint i = lid.y * 16u + lid.x; i < 34u * 34u; i += 256u) {
        uint2 q = uint2(clamp(base + int2(i % 34u, i / 34u), int2(0), size - 1));
        float d = depth.read(q);
        uint pd = prim.read(q).r;
        float3 c = float3(float(pd & 255u), float((pd >> 15) & 15u), 255.0) / 255.0;
        if (d > 0.0) {
            uint2 own = withGi ? giUpsampleTexel(gi, q) : uint2(0u);
            float3 rel = giRelAt(p, q, d);
            float fog = saturate(length(rel) / 4096.0);
            c = c * (1.0 - fog) + fog * 0.5;
            if (withGi) {
                float4 g = giUpsample(gi, own, q, min((pd >> 8) & 7u, 5u), rel);
                if (g.w > 0.0) c *= g.rgb;
            }
        }
        tile[i] = half4(half3(c), 1.0h);
        dtile[i] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint quarter = 0; quarter < 4u; quarter++) {
        uint2 lp = lid + uint2((quarter & 1u) * 16u, (quarter >> 1) * 16u);
        uint2 gid = tgid * 32u + lp;
        if (int(gid.x) >= size.x || int(gid.y) >= size.y) continue;
        uint c0 = (lp.y + 1u) * 34u + lp.x + 1u;
        half3 lo = tile[c0].xyz, hi = lo;
        float nearest = dtile[c0];
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                uint k = uint(int(c0) + dy * 34 + dx);
                lo = min(lo, tile[k].xyz);
                hi = max(hi, tile[k].xyz);
                nearest = max(nearest, dtile[k]);
            }
        }
        out.write(half4(dot(lo + hi, half3(0.5h)) + half(nearest)), gid);
    }
}

// Offline self-test: values through the texture unit's RG11B10Float conversion (a write to `hw`, 256 texels a row) and
// through giPackIrradiance (into `out`), to compare bit for bit.
kernel void gi_test_pack(const device float* vals [[buffer(13)]], device uint* out [[buffer(14)]],
                         texture2d<float, access::write> hw [[texture(1)]], uint tid [[thread_position_in_grid]]) {
    float v = vals[tid];
    hw.write(float4(v, v, v, 1.0), uint2(tid & 255u, tid >> 8));
    out[tid] = giPackIrradiance(float3(v));
}

// Offline self-test: cell keys of blocks through giAirBlock, giCellOf, giKey and back through giUnkey.
// in: block xyz, face | level << 8; out: the cell, then the unkeyed cell (near: the cell itself plus 1000 on x and z).
kernel void gi_test_keys(const device int4* blocks [[buffer(13)]], device int4* out [[buffer(14)]], uint tid [[thread_position_in_grid]]) {
    int4 b = blocks[tid];
    uint face = uint(b.w) & 255u, level = uint(b.w) >> 8;
    int3 air = giAirBlock(b.xyz, float3(0.5, 0.5, 0.5) + kGiNormal[face] * 0.5, face);
    int3 c = giCellOf(air, level);
    uint2 k = giKey(c, face, level);
    out[2 * tid] = int4(c, int(k.y >> 25));
    out[2 * tid + 1] = int4(giUnkey(k, c + int3(1000, 0, -1000)), int(k.y >> 28));
}
"""

/// Mirrors GiParams in the shader (256 bytes).
struct GiParams {
    var invViewProj = matrix_identity_float4x4
    var origin = SIMD4<Int32>.zero
    var camBlock = SIMD4<Int32>.zero
    var camFrac = SIMD4<Float>.zero
    var sun = SIMD4<Float>.zero
    var rays = SIMD4<Float>.zero
    var blockLight = SIMD4<Float>.zero
    var sizes = SIMD4<Float>.zero
    var limits = SIMD4<Float>.zero
    var look = SIMD4<Float>.zero
    var counts = SIMD4<UInt32>.zero
    var sample = SIMD4<UInt32>.zero
    var sched = SIMD4<UInt32>.zero
}

/// The prototype's light, in linear units where the sun on a surface facing it is 3 (a look to tune against the SEUS
/// stills, and to replace with the sky model once there is one).
let giSunIrradiance = SIMD3<Float>(1.0, 0.93, 0.82) * 3.0
let giSkyZenith = SIMD3<Float>(0.10, 0.17, 0.33)
let giSkyHorizon = SIMD3<Float>(0.24, 0.30, 0.38)
let giTorch = SIMD3<Float>(1.6, 1.15, 0.7)
/// Emission of a light-15 block (glowstone, lava), pi times its radiance.
let giEmission: Float = 4.0

/// GiLight (the shader's: sun, scale, skyView, zenith, horizon, ground, open[6]) holding the prototype's light, for the
/// offline test without lit mode: the cells' units are the light's own (scale 1).
func giPrototypeLight(sunUp: Float) -> [SIMD4<Float>] {
    let mid = (giSkyHorizon + (giSkyZenith - giSkyHorizon) * 0.6) * .pi
    return [SIMD4(giSunIrradiance * sunUp, 0), SIMD4(1, 1, 1, 0), .zero, SIMD4(giSkyZenith, 0), SIMD4(giSkyHorizon, 0),
            SIMD4(giSkyHorizon * 0.3, 0)] + [SIMD4<Float>](repeating: SIMD4(mid, 0), count: 6)
}

/// Lit mode's light for the cache this frame (encodeFrame): the relight's sun and sky light (Lit.swift's env), either
/// the atmosphere's (lit_env from this frame's sky tables, the sky's radiance from its sky view table; litDaylightEnv's
/// values if the tables aren't there) or litDaylightEnv's values.
enum GiLightSource {
    case atmosphere(fallback: [SIMD4<Float>])
    case daylight([SIMD4<Float>])
}

/// Per material: top, side and bottom albedo (linear) and emission (pi times radiance), four float4 each.
func giMaterialTable(_ colors: [SIMD4<Float>]) -> [SIMD4<Float>] {
    var t = [SIMD4<Float>](repeating: .zero, count: 256 * 4)
    func lin(_ c: SIMD4<Float>) -> SIMD4<Float> { SIMD4(Foundation.pow(c.x, 2.2), Foundation.pow(c.y, 2.2), Foundation.pow(c.z, 2.2), 1) }
    for m in 0..<256 {
        for f in 0..<3 { t[m * 4 + f] = lin(colors[m * 3 + f]) }
        if lodEmission[m] != 0 && lodKinds[m] != MaterialKind.air.rawValue {
            let c = t[m * 4 + 1], mx = max(c.x, c.y, c.z, 1e-3), e = Float(lodEmission[m]) / 15 * giEmission
            t[m * 4 + 3] = SIMD4(c.x / mx * e, c.y / mx * e, c.z / mx * e, 0)
        }
    }
    return t
}

/// The cache: its table, pipelines and per-frame state. One per dimension would be the in-game shape (cells are keyed
/// by world position); there is one, and lit mode only runs in the overworld.
final class GiCache: @unchecked Sendable {
    /// Lit mode's cache (METALMC_EXP=lit,gi), made on first use with the LOD's colors as it draws them (nil without lit,gi
    /// or without ray tracing). Render thread.
    static let shared: GiCache? = litGi ? GiCache(capacityLog2: giCapacityLog2, colors: LodRenderer.shared.giColors()) : nil

    let slots: Int
    let listCapacity: Int   // the most cells one frame can update
    /// GI_ZERO_COPY: bounce hits read their quad from the LOD node's buffer (`tiles` must describe the instance structure).
    let zeroCopy: Bool
    let check: MTLBuffer, keys: MTLBuffer, stamp: MTLBuffer, value: MTLBuffer, meta: MTLBuffer, list: MTLBuffer
    let counters: MTLBuffer   // shared: the statistics are read on the CPU
    let mats: MTLBuffer
    /// GiLight: gi_light writes it each frame in lit mode, the offline test the prototype's light (giPrototypeLight).
    let light: MTLBuffer
    /// Lit mode with the atmosphere: lit_env's output for gi_light (8 float4 and its sky view scale).
    private let envBuffer: MTLBuffer
    private let dummySkyView: MTLTexture
    /// The sky for this frame's bounce rays (lit mode with the atmosphere: Sky's frame and sky view table).
    private var skyFrame = SkyFrameGPU()
    private var skyView: MTLTexture?
    /// Zero copy: the instance structure's GiTile per instance and the node buffers their addresses point into.
    var tiles: (table: MTLBuffer, buffers: [MTLBuffer])?
    private var pipes: [String: MTLComputePipelineState] = [:]
    var frame: Int32 = 1
    private var sweepPos = 0, sweepLen = 0, sweeps = 0
    private var edits: [SIMD4<Int32>] = []
    private let lock = NSLock()
    /// Cells' average and eviction settings, overridable for tests (the budget up to listCapacity).
    var budget = giBudget, samplesPerUpdate = giSamplesPerUpdate, level0Range = giLevel0Range, history = giHistory, evictFrames = giEvictFrames
    var spatial = giSpatial
    /// Farthest surface cached, and the rays' reach: the sun's as RtShadows', bounces shorter (past a few hundred
    /// blocks what a bounce ray finds is low on the horizon and weighs little, and long rays cost the most).
    var maxDistance: Float = 8192, sunRayLength: Float = 4000, bounceRayLength: Float = 512

    init?(capacityLog2: Int, colors: [SIMD4<Float>] = lodColorTable(), zeroCopy: Bool = true) {
        let dev = ctx.device
        guard dev.supportsRaytracing else { log("gi: no ray tracing on this device"); return nil }
        guard MemoryLayout<GiParams>.stride == 256 else { log("gi: GiParams is \(MemoryLayout<GiParams>.stride) bytes, the shader's 256"); return nil }
        slots = 1 << capacityLog2
        listCapacity = giBudget
        self.zeroCopy = zeroCopy
        func buf(_ bytes: Int, _ mode: MTLResourceOptions = .storageModePrivate) -> MTLBuffer? { dev.makeBuffer(length: max(bytes, 16), options: mode) }
        let table = giMaterialTable(colors)
        let sv = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        sv.usage = .shaderRead
        guard let c = buf(slots * 4), let k = buf(slots * 16), let s = buf(slots * 4), let v = buf(slots * 8), let m = buf(slots * 4),
              let l = buf(listCapacity * 4), let n = buf(64 * 4, .storageModeShared),
              let t = dev.makeBuffer(bytes: table, length: table.count * 16, options: .storageModeShared),
              let li = buf(12 * 16, .storageModeShared), let env = buf(9 * 16), let dsv = dev.makeTexture(descriptor: sv) else { return nil }
        check = c; keys = k; stamp = s; value = v; meta = m; list = l; counters = n; mats = t; light = li; envBuffer = env; dummySkyView = dsv
        if clEnabled { clBounce = buf(slots * 8) }   // colored block light bounced (ColoredLightBounce.swift)
        do {
            // Lab mode (ShaderLab.swift): after an edit, the kernels are swapped and the table kept (as debugReload does).
            let lib = try ShaderLab.library(zeroCopy ? "gi" : "gi_copied", giShaderSource(zeroCopy: zeroCopy), device: dev) { [weak self] new in
                guard let self else { return }
                var next: [String: MTLComputePipelineState] = [:]
                for name in self.pipes.keys {
                    guard let f = new.makeFunction(name: name), let p = try? dev.makeComputePipelineState(function: f) else {
                        log("gi: reload: \(name) failed, the old kernels stay"); return
                    }
                    next[name] = p
                }
                self.pipes = next
            }
            for name in ["gi_begin", "gi_request", "gi_schedule", "gi_args", "gi_update", "gi_resolve", "gi_debug_view", "gi_invalidate",
                         "gi_count", "gi_light", "gi_test_primary", "gi_test_rays", "gi_test_keys", "gi_test_upsample", "gi_test_pack",
                         "gi_test_shadow", "gi_test_taa"] {
                guard let f = lib.makeFunction(name: name) else { log("gi: no function \(name)"); return nil }
                pipes[name] = try dev.makeComputePipelineState(function: f)
            }
        } catch {
            log("gi: shaders failed: \(error)")
            return nil
        }
        memset(counters.contents(), 0, counters.length)
        setLight(giPrototypeLight(sunUp: 1))
        guard let cb = ctx.queue.makeCommandBuffer() else { return nil }
        clear(cb)
        cb.commit()
        cb.waitUntilCompleted()
        log("gi: cache of \(slots) cells, \((slots * giBytesPerCell) >> 20) MB, \(zeroCopy ? "hit quads from the LOD's buffers (zero copy)" : "per-triangle data")")
    }

    func pipe(_ name: String) -> MTLComputePipelineState { pipes[name]! }

    /// Colored block light bounced (ColoredLightBounce.swift): the cells' block channel, the request's output for it
    /// (half resolution, RG11B10Float; set by RtShadows.encodeGi), and the frame's origin for the update's hits.
    var clBounce: MTLBuffer?
    var clBlockOut: MTLTexture?
    var clOrigin = SIMD3<Double>.zero

    /// Empties the table.
    func clear(_ cb: MTLCommandBuffer) {
        guard let b = cb.makeBlitCommandEncoder() else { return }
        for x in [check, keys, stamp, value, meta] + (clBounce.map { [$0] } ?? []) { b.fill(buffer: x, range: 0..<x.length, value: 0) }
        b.endEncoding()
    }

    /// Sets the light from the CPU (GiLight's 12 float4; the offline test, with nothing in flight).
    func setLight(_ l: [SIMD4<Float>]) {
        l.withUnsafeBytes { light.contents().copyMemory(from: $0.baseAddress!, byteCount: min($0.count, light.length)) }
    }

    /// Queues an edited block (world coordinates): cells within `radius` blocks restart their averages next frame.
    func invalidate(x: Int, y: Int, z: Int, radius: Int = 6) {
        lock.lock(); defer { lock.unlock() }
        edits.append(SIMD4(Int32(truncatingIfNeeded: x), Int32(y), Int32(truncatingIfNeeded: z), Int32(radius)))
    }

    /// The frame's parameters. `cam`: camera position (world); `origin`: the acceleration structure's origin (whole
    /// blocks); `sunDir`: direction to the sun, `sunUp` 0-1; `width` x `height`: the depth buffer's size.
    func params(invViewProj: simd_float4x4, cam: SIMD3<Double>, origin: SIMD3<Double>, sunDir: SIMD3<Float>, sunUp: Float,
                width: Int, height: Int) -> GiParams {
        let camFloor = cam.rounded(.down)
        let sc = giRequestScale
        let pat = giRequestPattern[Int(UInt32(bitPattern: frame) % 16)]
        var p = GiParams()
        p.invViewProj = invViewProj
        p.origin = SIMD4(Int32(truncatingIfNeeded: Int(origin.x)), Int32(Int(origin.y)), Int32(truncatingIfNeeded: Int(origin.z)), frame)
        p.camBlock = SIMD4(Int32(truncatingIfNeeded: Int(camFloor.x)), Int32(Int(camFloor.y)), Int32(truncatingIfNeeded: Int(camFloor.z)),
                           Int32(slots / giBucketSlots - 1))
        p.camFrac = SIMD4(Float(cam.x - camFloor.x), Float(cam.y - camFloor.y), Float(cam.z - camFloor.z), level0Range)
        p.sun = SIMD4(simd_normalize(sunDir), sunUp)
        p.rays = SIMD4(0.0105, sunRayLength, history, 1e9)
        p.blockLight = SIMD4(giTorch, evictFrames)
        p.sizes = SIMD4(Float(width), Float(height), Float((width + sc - 1) / sc), Float((height + sc - 1) / sc))
        p.limits = SIMD4(maxDistance, 0.02, 0, bounceRayLength)
        p.look = SIMD4(1, 0, 1, spatial)
        sweepLen = sweepLength()
        p.counts = SIMD4(UInt32(min(budget, listCapacity)), UInt32(sweepPos), UInt32(sweepLen), UInt32(truncatingIfNeeded: sweeps))
        p.sample = SIMD4(UInt32(sc), pat.x, pat.y, UInt32(samplesPerUpdate))
        // The share of visible cells to list: their part of the list over how many were seen last frame (counter 19).
        let seen = Double(counters.contents().load(fromByteOffset: 19 * 4, as: UInt32.self))
        let part = Double(min(budget, listCapacity) - min(budget, listCapacity) / 4)
        p.sched = SIMD4(UInt32(min(1, part / max(seen, 1)) * 4294967295.0), 0, 0, 0)
        return p
    }

    /// Slots to sweep this frame: scaled from the last sweep by how far the cells it listed (counter 16, a frame or two
    /// old in the game) fell short of its quarter of the budget, so that part of the list fills.
    /// Between 1/giSweepFramesMax and 1/giSweepFramesMin of the table.
    private func sweepLength() -> Int {
        let listed = Int(counters.contents().load(fromByteOffset: 16 * 4, as: UInt32.self))
        var n = sweepLen == 0 ? slots / giSweepFramesMax : sweepLen
        n = Int(Double(n) * min(2, max(0.5, Double(min(budget, listCapacity) / 4) / Double(max(listed, 16)))))
        n = min(slots / giSweepFramesMin, max(slots / giSweepFramesMax, n))
        return (n + giBucketSlots - 1) / giBucketSlots * giBucketSlots
    }

    /// Advances to the next frame (the sweep's position, the frame number).
    func advance() {
        frame &+= 1
        if frame <= 0 { frame = 1 }
        sweepPos += sweepLen
        if sweepPos >= slots { sweepPos %= slots; sweeps += 1 }
    }

    private func bindTable(_ enc: MTLComputeCommandEncoder) {
        enc.setBuffer(check, offset: 0, index: 2)
        enc.setBuffer(keys, offset: 0, index: 3)
        enc.setBuffer(stamp, offset: 0, index: 4)
        enc.setBuffer(value, offset: 0, index: 5)
        enc.setBuffer(meta, offset: 0, index: 6)
        enc.setBuffer(list, offset: 0, index: 7)
        enc.setBuffer(counters, offset: 0, index: 8)
        enc.setBuffer(mats, offset: 0, index: 9)
        enc.setBuffer(light, offset: 0, index: 15)
        if let clBounce { enc.setBuffer(clBounce, offset: 0, index: 19) }   // colored block light bounced
    }

    /// What the kernels that trace rays need beside the table: the hit quads' buffers (zero copy, buffer 17) and the sky
    /// (buffer 16, texture 6).
    func bindRays(_ enc: MTLComputeCommandEncoder) {
        if zeroCopy, let tiles {
            enc.setBuffer(tiles.table, offset: 0, index: 17)
            enc.useResources(tiles.buffers, usage: .read)
        }
        var f = skyFrame
        enc.setBytes(&f, length: MemoryLayout<SkyFrameGPU>.stride, index: 16)
        enc.setTexture(skyView ?? dummySkyView, index: 6)
        if clEnabled { giClBind(enc, origin: clOrigin) }   // colored block light at the hits: textures 7, 8, buffer 20
    }

    private func dispatch1D(_ enc: MTLComputeCommandEncoder, _ name: String, _ n: Int) {
        enc.setComputePipelineState(pipe(name))
        enc.dispatchThreads(MTLSize(width: max(n, 1), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// This frame's sun and sky light as lit_env made it (Lit.encodeEnv) when encodeLight took it from the atmosphere,
    /// else nil: the relight reads it rather than run lit_env again on the same sky tables (RtShadows.takeLitGi).
    private(set) var envThisFrame: MTLBuffer?

    /// Lit mode: the frame's GiLight from the relight's light (gi_light), and the sky its bounce rays see.
    private func encodeLight(_ enc: MTLComputeCommandEncoder, _ source: GiLightSource) {
        var opt = SIMD4<Float>.zero
        var daylight: [SIMD4<Float>]
        envThisFrame = nil
        switch source {
        case .atmosphere(let fallback):
            daylight = fallback
            if let view = Sky.shared.skyView, Lit.shared.encodeEnv(enc, into: envBuffer) {
                enc.setBuffer(envBuffer, offset: 0, index: 16)
                skyFrame = Sky.shared.frame
                skyView = view
                opt.x = 1
                daylight = []
                envThisFrame = envBuffer
            }
        case .daylight(let env):
            daylight = env
        }
        if !daylight.isEmpty {
            daylight.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 16) }
            skyFrame = SkyFrameGPU()
            skyView = nil
        }
        enc.setBytes(&opt, length: 16, index: 18)
        enc.setBuffer(light, offset: 0, index: 15)
        dispatch1D(enc, "gi_light", 1)
    }

    /// Stage 0: clear the frame's counters and apply queued edits.
    func encodeBegin(_ enc: MTLComputeCommandEncoder, params p: inout GiParams) {
        bindTable(enc)
        enc.setBytes(&p, length: MemoryLayout<GiParams>.stride, index: 1)
        dispatch1D(enc, "gi_begin", 1)
        lock.lock()
        let pending = edits
        edits.removeAll()
        lock.unlock()
        // In chunks of 256 (4 KB of bytes set on the encoder, copied as it encodes: a buffer reused next frame could be
        // overwritten while the GPU still reads it).
        var start = 0
        while start < pending.count {
            let chunk = Array(pending[start..<min(start + 256, pending.count)])
            start += chunk.count
            chunk.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 10) }
            let r = Int(chunk.map { $0.w }.max() ?? 1), side = 2 * r + 1
            enc.setComputePipelineState(pipe("gi_invalidate"))
            enc.dispatchThreads(MTLSize(width: side * side * side * 6 + Int(12 * 162), height: chunk.count, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
    }

    /// Stage 1: request (and list) the cells the depth buffer shows, and resolve their light at half resolution into `out`
    /// (rg32Uint, half the depth buffer's size rounded up: the irradiance's bits and the code word, giUpsampleHeader;
    /// giUpsample is the way back to full resolution) from the cells as the last update left them.
    func encodeRequest(_ enc: MTLComputeCommandEncoder, depth: MTLTexture, out: MTLTexture, params p: inout GiParams) {
        bindTable(enc)
        enc.setBytes(&p, length: MemoryLayout<GiParams>.stride, index: 1)
        enc.setComputePipelineState(pipe("gi_request"))
        enc.setTexture(depth, index: 0)
        enc.setTexture(out, index: 1)
        if clEnabled { giClBindOut(enc, clBlockOut) }   // colored block light bounced: texture 5
        enc.dispatchThreads(MTLSize(width: Int(p.sizes.z), height: Int(p.sizes.w), depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }

    /// Stage 2: sweep this frame's slice of the table and size the update.
    func encodeSchedule(_ enc: MTLComputeCommandEncoder, params p: inout GiParams) {
        bindTable(enc)
        enc.setBytes(&p, length: MemoryLayout<GiParams>.stride, index: 1)
        dispatch1D(enc, "gi_schedule", Int(p.counts.z))
        dispatch1D(enc, "gi_args", 1)
    }

    /// Stage 3: trace the listed cells' rays. `accels`: the structures `accel` references.
    func encodeUpdate(_ enc: MTLComputeCommandEncoder, accel: MTLAccelerationStructure, accels: [MTLAccelerationStructure], params p: inout GiParams) {
        bindTable(enc)
        bindRays(enc)
        enc.setBytes(&p, length: MemoryLayout<GiParams>.stride, index: 1)
        enc.setAccelerationStructure(accel, bufferIndex: 0)
        enc.useResources(accels, usage: .read)
        enc.setComputePipelineState(pipe("gi_update"))
        enc.dispatchThreadgroups(indirectBuffer: counters, indirectBufferOffset: 32, threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// The resolve alone (the debug views and tests: a frame's comes with its request) into `out`, as encodeRequest's.
    func encodeResolve(_ enc: MTLComputeCommandEncoder, depth: MTLTexture, out: MTLTexture, params p: inout GiParams) {
        bindTable(enc)
        enc.setBytes(&p, length: MemoryLayout<GiParams>.stride, index: 1)
        enc.setComputePipelineState(pipe("gi_resolve"))
        enc.setTexture(depth, index: 0)
        enc.setTexture(out, index: 1)
        enc.dispatchThreads(MTLSize(width: (depth.width + 1) / 2, height: (depth.height + 1) / 2, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }

    /// Lit mode's frame (METALMC_EXP=lit,gi; RtShadows.trace, once its instance structure is built, after the level is
    /// drawn), in two encoders: the light from the relight's, then the request with the resolve into `out` (rg32Uint, half
    /// the depth buffer's size rounded up); then the schedule and the update. `out` is done in the first (it shows the
    /// cells as the last frame's update left them), so what reads it later in the frame (the relight) waits for that one
    /// only, and the update can run beside it. The instance structure must cover every direction (RtShadows' tiles are
    /// the LOD's selection, not frustum-culled) and, with zero copy, `tiles` must describe its instances.
    /// `cloudHeight`: the cloud layer's bottom (world y). Block light isn't in the cells: the relight's stays vanilla's
    /// flood fill, and it couldn't follow the daylight's scale. Each encoder comes from `pass` (its label: traced frames'
    /// timestamps, -PbenchTrace=1, and the offline profile's). `fence` (offline profile only): each encoder waits for it
    /// and updates it, so none overlaps another and each one's time is its own.
    func encodeFrame(_ cb: MTLCommandBuffer, depth: MTLTexture, out: MTLTexture, invViewProj: simd_float4x4, cam: SIMD3<Double>,
                     origin: SIMD3<Double>, accel: MTLAccelerationStructure, accels: [MTLAccelerationStructure], sunDir: SIMD3<Float>,
                     sunUp: Float, cloudHeight: Float, light: GiLightSource, fence: MTLFence? = nil,
                     pass: (String) -> MTLComputePassDescriptor = profComputePass) -> Bool {
        if zeroCopy && tiles == nil { return false }
        clOrigin = origin
        var p = params(invViewProj: invViewProj, cam: cam, origin: origin, sunDir: sunDir, sunUp: sunUp, width: depth.width, height: depth.height)
        p.blockLight = SIMD4(0, 0, 0, evictFrames)
        p.rays.w = cloudHeight - Float(cam.y)
        func begin(_ label: String) -> MTLComputeCommandEncoder? {
            let e = cb.makeComputeCommandEncoder(descriptor: pass(label))
            e?.label = "MetalMC " + label
            if let fence { e?.waitForFence(fence) }
            return e
        }
        func end(_ e: MTLComputeCommandEncoder) {
            if let fence { e.updateFence(fence) }
            e.endEncoding()
        }
        // (Every stage binds what it uses, so they don't need the same encoder.)
        guard let first = begin("GI cache: light, begin, request and resolve") else { return false }
        encodeLight(first, light)
        encodeBegin(first, params: &p)
        encodeRequest(first, depth: depth, out: out, params: &p)
        end(first)
        guard let second = begin("GI cache: schedule, update (rays)") else { return false }
        encodeSchedule(second, params: &p)
        encodeUpdate(second, accel: accel, accels: accels, params: &p)
        end(second)
        advance()
        return true
    }

    /// Offline test: recompiles the kernels from `body` (the upsample header and kernels, as mmc_debug_gi_shader_source
    /// gives them; nil: the built-in ones) and swaps the pipelines, keeping the table.
    func debugReload(_ body: String?) -> Bool {
        let src = "#define GI_ZERO_COPY \(zeroCopy ? 1 : 0)\n" + skyShaderHeader + (body ?? (giUpsampleHeader + "\n" + giKernelSource))
        do {
            let lib = try ctx.device.makeLibrary(source: src, options: nil)
            var next: [String: MTLComputePipelineState] = [:]
            for name in pipes.keys {
                guard let f = lib.makeFunction(name: name) else { log("gi: reload: no function \(name)"); return false }
                next[name] = try ctx.device.makeComputePipelineState(function: f)
            }
            pipes = next
            return true
        } catch {
            log("gi: reload failed: \(error)")
            return false
        }
    }

    /// Offline test: the CPU's part of the cache's state (frame number, sweep), to save and restore with the table.
    func debugState() -> [Int64] { [Int64(frame), Int64(sweepPos), Int64(sweepLen), Int64(sweeps)] }
    func debugRestore(_ s: [Int64]) {
        guard s.count >= 4 else { return }
        frame = Int32(truncatingIfNeeded: s[0]); sweepPos = Int(s[1]); sweepLen = Int(s[2]); sweeps = Int(s[3])
    }

    /// Offline test: lit mode's light from litDaylightEnv at `sunAngle` (vanilla's, radians) into `light`, as encodeFrame
    /// makes it without the atmosphere. Blocks until done.
    func debugDaylightLight(sunAngle: Float) {
        guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
        encodeLight(enc, .daylight(litDaylightEnv(sunAngle: sunAngle)))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
}

extension LodRenderer {
    /// The LOD's colors as it draws them (the game's biome tints, the resource pack's ratios), for the cache's albedos.
    func giColors() -> [SIMD4<Float>] {
        var c = lodColorTable(tints: tints)
        for i in 0..<min(c.count, packRatio.count) { c[i] = SIMD4(c[i].x * packRatio[i].x, c[i].y * packRatio[i].y, c[i].z * packRatio[i].z, c[i].w) }
        return c
    }
}

/// A 4 x 4 visiting order that spreads consecutive frames' request samples apart (a Bayer order, as RtShadows').
let giRequestPattern: [SIMD2<UInt32>] = [0, 10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 14, 6, 12].map { SIMD2(UInt32($0 % 4), UInt32($0 / 4)) }

// MARK: - Acceleration structures the bounce rays can read their hits from

/// A tile's opaque quads and tile-edge skirts (the quads RtShadows takes, in its order) as triangles in node-local blocks,
/// for a structure with one geometry per quad range (zero copy, giHitQuad: a hit's geometry and primitive index find its
/// quad in the node's buffer). `ranges`: per geometry, its first quad in the node's buffer and its quad count (the opaque
/// faces' buckets, then the skirts', empty ones left out). Two triangles per quad, in the buffer's order.
func giTileMesh(quads: UnsafePointer<UInt32>, start: [Int], tile t: Int, level: Int) -> (verts: [Float], idx: [UInt32], ranges: [(first: Int, count: Int)]) {
    let scale = Float(1 << level)
    var verts: [Float] = [], idx: [UInt32] = [], ranges: [(first: Int, count: Int)] = []
    for (k0, k1) in [(0, 6), (12, lodBucketsPerTile)] {
        let a = start[lodBucketIndex(t, k0, 0)], b = start[lodBucketIndex(t, k1 - 1, lodSubtilesPerTile)]
        if b <= a { continue }
        ranges.append((a, b - a))
        verts.reserveCapacity(verts.count + (b - a) * 12)
        idx.reserveCapacity(idx.count + (b - a) * 6)
        for i in a..<b {
            let w0 = quads[2 * i], w1 = quads[2 * i + 1]
            let face = Int((w0 >> 25) & 7)
            let base = UInt32(verts.count / 3)
            if face > 5 {
                // Not a block face (the mesher makes none): a degenerate quad keeps the triangles in step with the quads.
                verts += [Float](repeating: 0, count: 12)
            } else {
                let local = SIMD3(Float(w0 & 255), Float((w0 >> 16) & 511), Float((w0 >> 8) & 255))
                let qw = Float(((w1 >> 8) & 255) + 1), qh = Float(((w1 >> 16) & 255) + 1)
                let ext: SIMD3<Float> = face < 2 ? SIMD3(1, qw, qh) : (face < 4 ? SIMD3(qw, 1, qh) : SIMD3(qw, qh, 1))
                for (cx, cy, cz) in lodFaceCorners[face] {
                    let p = (local + SIMD3(Float(cx), Float(cy), Float(cz)) * ext) * scale
                    verts += [p.x, p.y, p.z]
                }
            }
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
        }
    }
    return (verts, idx, ranges)
}

/// The geometry descriptors of a tile's structure: one per quad range (giTileMesh's, consecutive in `ib`), or one for all
/// of `ib`'s `triangles` without ranges.
func giGeometries(vb: MTLBuffer, ib: MTLBuffer, triangles: Int, ranges: [(first: Int, count: Int)]?) -> [MTLAccelerationStructureTriangleGeometryDescriptor] {
    var out: [MTLAccelerationStructureTriangleGeometryDescriptor] = []
    var first = 0
    for count in ranges.map({ $0.map { 2 * $0.count } }) ?? [triangles] {
        let g = MTLAccelerationStructureTriangleGeometryDescriptor()
        g.vertexBuffer = vb
        g.vertexStride = 12
        g.vertexFormat = .float3
        g.indexBuffer = ib
        g.indexBufferOffset = first * 12
        g.indexType = .uint32
        g.triangleCount = count
        g.opaque = true
        out.append(g)
        first += count
    }
    return out
}

/// GiTile (giHitQuad) for an instance whose structure giTileMesh built: the node's quad buffer and its ranges' first quads.
func giTileEntry(_ buffer: MTLBuffer, _ ranges: [(first: Int, count: Int)]) -> SIMD4<UInt32> {
    let a = buffer.gpuAddress
    return SIMD4(UInt32(truncatingIfNeeded: a), UInt32(truncatingIfNeeded: a >> 32), UInt32(ranges.first?.first ?? 0),
                 UInt32(ranges.count > 1 ? ranges[1].first : 0))
}

/// A tile's opaque quads (and tile-edge skirts, as RtShadows takes them) as triangles in node-local blocks, with 4 bytes
/// per triangle for the cache's bounce rays: material | face << 8 | block light << 11 | water depth or sky cover << 15.
/// `quads`: the node's quad words; `start`: its (tile, bucket, sub-tile) prefix offsets (LodMeshNode.start).
func giTileGeometry(quads: UnsafePointer<UInt32>, start: [Int], tile t: Int, level: Int) -> (verts: [Float], idx: [UInt32], prims: [UInt32]) {
    let scale = Float(1 << level)
    var verts: [Float] = [], idx: [UInt32] = [], prims: [UInt32] = []
    for k in 0..<lodBucketsPerTile where k < 6 || k >= 12 {
        let a = start[lodBucketIndex(t, k, 0)], b = start[lodBucketIndex(t, k, lodSubtilesPerTile)]
        for i in a..<b {
            let w0 = quads[2 * i], w1 = quads[2 * i + 1]
            let face = Int((w0 >> 25) & 7)
            if face > 5 { continue }
            let local = SIMD3(Float(w0 & 255), Float((w0 >> 16) & 511), Float((w0 >> 8) & 255))
            let qw = Float(((w1 >> 8) & 255) + 1), qh = Float(((w1 >> 16) & 255) + 1)
            let ext: SIMD3<Float> = face < 2 ? SIMD3(1, qw, qh) : (face < 4 ? SIMD3(qw, 1, qh) : SIMD3(qw, qh, 1))
            let base = UInt32(verts.count / 3)
            for (cx, cy, cz) in lodFaceCorners[face] {
                let p = (local + SIMD3(Float(cx), Float(cy), Float(cz)) * ext) * scale
                verts += [p.x, p.y, p.z]
            }
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
            let pd = (w1 & 255) | UInt32(face) << 8 | ((w1 >> 24) & 15) << 11 | ((w0 >> 28) & 15) << 15
            prims += [pd, pd]
        }
    }
    return (verts, idx, prims)
}

/// Builds and compacts one primitive structure: with per-triangle data `prims` (one geometry), or one geometry per range
/// of `ranges` (giTileMesh's), or one geometry without either. Blocks until done.
func giBuildBlas(verts: [Float], idx: [UInt32], prims: [UInt32]?, ranges: [(first: Int, count: Int)]? = nil,
                 queue: MTLCommandQueue) -> MTLAccelerationStructure? {
    let dev = ctx.device
    guard !idx.isEmpty,
          let vb = dev.makeBuffer(bytes: verts, length: verts.count * 4, options: .storageModeShared),
          let ib = dev.makeBuffer(bytes: idx, length: idx.count * 4, options: .storageModeShared) else { return nil }
    let geometries = giGeometries(vb: vb, ib: ib, triangles: idx.count / 3, ranges: prims == nil ? ranges : nil)
    if let prims, let g = geometries.first {
        guard let pb = dev.makeBuffer(bytes: prims, length: prims.count * 4, options: .storageModeShared) else { return nil }
        g.primitiveDataBuffer = pb
        g.primitiveDataStride = 4
        g.primitiveDataElementSize = 4
    }
    let d = MTLPrimitiveAccelerationStructureDescriptor()
    d.geometryDescriptors = geometries
    let sizes = dev.accelerationStructureSizes(descriptor: d)
    guard let accel = dev.makeAccelerationStructure(size: sizes.accelerationStructureSize),
          let scratch = dev.makeBuffer(length: max(sizes.buildScratchBufferSize, 16), options: .storageModePrivate),
          let sizeBuf = dev.makeBuffer(length: 8, options: .storageModeShared),
          let cb = queue.makeCommandBuffer(), let enc = cb.makeAccelerationStructureCommandEncoder() else { return nil }
    enc.build(accelerationStructure: accel, descriptor: d, scratchBuffer: scratch, scratchBufferOffset: 0)
    enc.writeCompactedSize(accelerationStructure: accel, buffer: sizeBuf, offset: 0, sizeDataType: .ulong)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let compacted = Int(sizeBuf.contents().load(as: UInt64.self))
    guard compacted > 0, let small = dev.makeAccelerationStructure(size: compacted),
          let ccb = queue.makeCommandBuffer(), let cenc = ccb.makeAccelerationStructureCommandEncoder() else { return accel }
    cenc.copyAndCompact(sourceAccelerationStructure: accel, destinationAccelerationStructure: small)
    cenc.endEncoding()
    ccb.commit()
    ccb.waitUntilCompleted()
    return small
}

/// An unsigned small float of RG11B10 (5 exponent bits, `mantissaBits` mantissa bits) to a Float (test readback).
func giUnpackSmallFloat(_ v: UInt32, mantissaBits: Int) -> Float {
    let e = Int(v >> UInt32(mantissaBits)) & 31, m = Float(v & ((1 << UInt32(mantissaBits)) - 1)) / Float(1 << mantissaBits)
    if e == 0 { return m * Foundation.pow(2, -14) }
    if e == 31 { return .infinity }
    return (1 + m) * Foundation.pow(2, Float(e - 15))
}

// MARK: - Offline test (tools/gicache.py)

/// The offline test's scene: LOD nodes built from a synthetic grid or a save's region files, one structure per tile as
/// RtShadows builds them, an instance structure over them, and a cache.
private final class GiTestScene {
    var grid: LodGrid?                          // the synthetic scene's level-0 grid (editable)
    var nodes: [(level: Int, x0: Int, z0: Int, mesh: LodMesh)] = []
    var blas: [MTLAccelerationStructure] = []
    var tlas: MTLAccelerationStructure?
    var origin = SIMD3<Int>(0, 0, 0)
    var cache: GiCache?
    var triangles = 0, blasBytes = 0
    var buildMs = 0.0
    /// The structures' per-triangle route: zero copy (the game's: one geometry per quad range, the hit's quad read from
    /// the node's buffer), primitive data (4 bytes per triangle in the structure), or none (what RtShadows builds without
    /// the cache; no cache then).
    enum Route { case zeroCopy, primitiveData, none }
    var route = Route.zeroCopy
    var nodeBuffers: [MTLBuffer] = []   // zero copy: each node's quads
    var tiles: MTLBuffer?               // zero copy: GiTile per instance
    var litLight = false                // lit mode's light (mmc_debug_gi_lit_light) instead of the prototype's
    var lastResolve: [Float] = []   // the last frame's resolved irradiance (half resolution, rgba)
}
nonisolated(unsafe) private var giTest: GiTestScene?

/// The synthetic scene around (128, 128): grass ground at y 63 with stone under it; a planks house (x 98-117, z 98-117,
/// y 64-71, stone roof) with a white floor, one red inner wall, a 4 x 3 window facing east (+x, where the morning sun
/// is) and a door facing west; a closed stone room lit only by a glowstone block; a stone hill with a 3 x 3 tunnel
/// running west into it from its east face (the mouth at x 209) to a chamber 30 blocks in.
private func giSyntheticGrid() -> LodGrid {
    var g = LodGrid(level: 0)
    func set(_ x: Int, _ y: Int, _ z: Int, _ m: Mat) {
        guard x >= 0, z >= 0, x < lodNodeVoxels, z < lodNodeVoxels, y >= lodWorldMinY, y < lodWorldMinY + lodWorldHeight else { return }
        g.v[g.index(x, y - lodWorldMinY, z)] = m == .grass ? lodTinted(m.rawValue, 0) : m.rawValue
    }
    func box(_ x0: Int, _ x1: Int, _ y0: Int, _ y1: Int, _ z0: Int, _ z1: Int, _ m: Mat) {
        for y in y0...y1 { for z in z0...z1 { for x in x0...x1 { set(x, y, z, m) } } }
    }
    box(0, 255, -64, 59, 0, 255, .stone)
    box(0, 255, 60, 62, 0, 255, .dirt)
    box(0, 255, 63, 63, 0, 255, .grass)
    // The house: walls, roof, floor, then the openings.
    box(98, 117, 64, 71, 98, 117, .planks)
    box(99, 116, 64, 71, 99, 116, .air)
    box(98, 117, 72, 72, 98, 117, .stone)
    box(99, 116, 63, 63, 99, 116, .wool)
    box(99, 99, 64, 71, 99, 116, .redTerracotta)
    box(117, 117, 66, 68, 105, 108, .air)     // window, east
    box(98, 98, 64, 65, 107, 107, .air)       // door, west
    // The closed room with glowstone.
    box(60, 71, 64, 70, 60, 71, .stone)
    box(61, 70, 64, 69, 61, 70, .air)
    set(65, 69, 65, .glowstone)
    // The hill and its tunnel.
    for z in 150...215 {
        for x in 150...215 {
            let d = simd_length(SIMD2<Float>(Float(x - 182), Float(z - 182)))
            let top = 63 + Int(max(0, 28 - d * 0.9))
            if top > 63 { box(x, x, 64, top, z, z, .stone) }
        }
    }
    box(176, 215, 64, 66, 181, 183, .air)     // tunnel, from the east face inward
    box(168, 177, 64, 69, 176, 188, .air)     // chamber
    for y in 0..<g.height { for i in 0..<(lodNodeVoxels * lodNodeVoxels) where g.v[y * lodNodeVoxels * lodNodeVoxels + i] == Mat.glowstone.rawValue {
        g.emitters.append(Int32(y * lodNodeVoxels * lodNodeVoxels + i))
    } }
    return g
}

private func giBuildScene(_ s: GiTestScene) -> Bool {
    let dev = ctx.device
    guard let queue = dev.makeCommandQueue() else { return false }
    let t0 = Date()
    s.blas = []
    s.triangles = 0
    s.blasBytes = 0
    s.nodeBuffers = []
    var instances: [MTLAccelerationStructureInstanceDescriptor] = []
    var tiles: [SIMD4<UInt32>] = []
    for n in s.nodes {
        var starts = [0]
        for c in n.mesh.counts { starts.append(starts.last! + c) }
        // Zero copy: the node's quads in a buffer of their own, as the LOD's nodes have them (LodMeshNode.buffer).
        var nodeBuffer: MTLBuffer?
        if s.route == .zeroCopy {
            guard let b = dev.makeBuffer(bytes: n.mesh.quads, length: max(n.mesh.quads.count * 4, 16), options: .storageModeShared) else { return false }
            nodeBuffer = b
            s.nodeBuffers.append(b)
        }
        n.mesh.quads.withUnsafeBufferPointer { q in
            for t in 0..<(lodTilesPerSide * lodTilesPerSide) {
                let b: MTLAccelerationStructure?, triangles: Int
                if let nodeBuffer {
                    let (verts, idx, ranges) = giTileMesh(quads: q.baseAddress!, start: starts, tile: t, level: n.level)
                    b = giBuildBlas(verts: verts, idx: idx, prims: nil, ranges: ranges, queue: queue)
                    if b != nil { tiles.append(giTileEntry(nodeBuffer, ranges)) }
                    triangles = idx.count / 3
                } else {
                    let (verts, idx, prims) = giTileGeometry(quads: q.baseAddress!, start: starts, tile: t, level: n.level)
                    b = giBuildBlas(verts: verts, idx: idx, prims: s.route == .primitiveData ? prims : nil, queue: queue)
                    triangles = idx.count / 3
                }
                guard let b else { continue }
                var inst = MTLAccelerationStructureInstanceDescriptor()
                inst.transformationMatrix = MTLPackedFloat4x3(columns: (MTLPackedFloat3Make(1, 0, 0), MTLPackedFloat3Make(0, 1, 0), MTLPackedFloat3Make(0, 0, 1),
                                                                        MTLPackedFloat3Make(Float(n.x0 - s.origin.x), Float(lodWorldMinY - s.origin.y), Float(n.z0 - s.origin.z))))
                inst.options = .opaque
                inst.mask = 0xFF
                inst.accelerationStructureIndex = UInt32(s.blas.count)
                instances.append(inst)
                s.blas.append(b)
                s.triangles += triangles
                s.blasBytes += b.size
            }
        }
    }
    s.tiles = tiles.isEmpty ? nil : dev.makeBuffer(bytes: tiles, length: tiles.count * 16, options: .storageModeShared)
    guard !instances.isEmpty,
          let ib = dev.makeBuffer(bytes: instances, length: instances.count * MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride, options: .storageModeShared) else { return false }
    let d = MTLInstanceAccelerationStructureDescriptor()
    d.instancedAccelerationStructures = s.blas
    d.instanceCount = instances.count
    d.instanceDescriptorBuffer = ib
    let sizes = dev.accelerationStructureSizes(descriptor: d)
    guard let tlas = dev.makeAccelerationStructure(size: sizes.accelerationStructureSize),
          let scratch = dev.makeBuffer(length: max(sizes.buildScratchBufferSize, 16), options: .storageModePrivate),
          let cb = queue.makeCommandBuffer(), let enc = cb.makeAccelerationStructureCommandEncoder() else { return false }
    enc.build(accelerationStructure: tlas, descriptor: d, scratchBuffer: scratch, scratchBufferOffset: 0)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    s.tlas = tlas
    s.buildMs = Date().timeIntervalSince(t0) * 1000
    if let tiles = s.tiles { s.cache?.tiles = (tiles, s.nodeBuffers) }
    return true
}

/// Debug: builds the offline test's scene. kind 0: the synthetic scene (giSyntheticGrid) at the origin; kind 1: the save's
/// region file `path` (r.X.Z.mca) at level 0 (its four quarters) and the 8 regions around it at level 1, as the LOD
/// meshes them. flags bit 0: no per-triangle data at all (what RtShadows builds without the cache, to measure the
/// routes against; no cache); bit 1: per-triangle data in the structures (primitiveDataBuffer) instead of the game's
/// zero copy. out: triangles, structure bytes (compacted), build ms (meshing and structures), nodes. Returns 1 if it worked.
@_cdecl("mmc_debug_gi_scene")
public func mmc_debug_gi_scene(_ kind: Int32, _ path: UnsafePointer<CChar>?, _ flags: Int32, _ out: UnsafeMutablePointer<Double>) -> Int32 {
    let s = GiTestScene()
    s.route = flags & 1 != 0 ? .none : (flags & 2 != 0 ? .primitiveData : .zeroCopy)
    let t0 = Date()
    if kind == 0 {
        // No hidden-air fill: it would fill the closed room.
        let g = giSyntheticGrid()
        s.grid = g
        s.nodes = [(0, 0, 0, LodBuild.mesh(g, maxMerge: 64, skyCover: true))]
        s.origin = SIMD3(0, 0, 0)
    } else {
        guard let path else { return 0 }
        let file = URL(fileURLWithPath: String(cString: path))
        let parts = file.lastPathComponent.split(separator: ".")
        guard parts.count == 4, let rx = Int(parts[1]), let rz = Int(parts[2]) else { return 0 }
        let dir = file.deletingLastPathComponent()
        s.origin = SIMD3(rx * 512 + 256, 0, rz * 512 + 256)
        let lock = NSLock()
        var nodes: [(level: Int, x0: Int, z0: Int, mesh: LodMesh)] = []
        // Level 0: the center region's quarters; level 1: its neighbors (built in parallel, as the LOD's passes are).
        var jobs: [(Int, Int, Int)] = (0..<4).map { (0, $0 & 1, $0 >> 1) }
        for dz in -1...1 { for dx in -1...1 where dx != 0 || dz != 0 { jobs.append((1, rx + dx, rz + dz)) } }
        DispatchQueue.concurrentPerform(iterations: jobs.count) { j in
            let (level, a, b) = jobs[j]
            if level == 0 {
                guard var g = LodBuild.regionQuarterGrid(path: file.path, qx: a, qz: b) else { return }
                g.fillUnreachable(deepRadius: 16, deepDepth: 8)
                let m = LodBuild.mesh(g, maxMerge: 64, skyCover: lodSkyCover)
                lock.lock(); nodes.append((0, rx * 512 + a * 256, rz * 512 + b * 256, m)); lock.unlock()
            } else {
                guard var g = LodBuild.regionGrid(path: dir.appendingPathComponent("r.\(a).\(b).mca").path) else { return }
                g.fillUnreachable()
                let m = LodBuild.mesh(g, maxMerge: 64, skyCover: lodSkyCover)
                lock.lock(); nodes.append((1, a * 512, b * 512, m)); lock.unlock()
            }
        }
        s.nodes = nodes
    }
    guard giBuildScene(s) else { return 0 }
    out[0] = Double(s.triangles); out[1] = Double(s.blasBytes); out[2] = Date().timeIntervalSince(t0) * 1000; out[3] = Double(s.nodes.count)
    if s.route != .none {
        s.cache = GiCache(capacityLog2: giCapacityLog2, zeroCopy: s.route == .zeroCopy)
        guard let cache = s.cache else { return 0 }
        if let tiles = s.tiles { cache.tiles = (tiles, s.nodeBuffers) }
    }
    giTest = s
    return 1
}

/// Debug: sets a block of the synthetic scene (world coordinates, material id), re-meshes it, rebuilds its structures and
/// tells the cache. Returns 1 if it worked.
@_cdecl("mmc_debug_gi_set_block")
public func mmc_debug_gi_set_block(_ x: Int32, _ y: Int32, _ z: Int32, _ material: Int32) -> Int32 {
    guard let s = giTest, var g = s.grid, x >= 0, z >= 0, Int(x) < lodNodeVoxels, Int(z) < lodNodeVoxels,
          Int(y) >= lodWorldMinY, Int(y) < lodWorldMinY + lodWorldHeight else { return 0 }
    g.v[g.index(Int(x), Int(y) - lodWorldMinY, Int(z))] = UInt8(truncatingIfNeeded: material)
    s.grid = g
    s.nodes = [(0, 0, 0, LodBuild.mesh(g, maxMerge: 64, skyCover: true))]
    guard giBuildScene(s) else { return 0 }
    s.cache?.invalidate(x: Int(x), y: Int(y), z: Int(z))
    return 1
}

/// The offline test's view: a fixed camera (Minecraft's yaw and pitch in degrees: yaw 0 looks south (+z), 90 west (-x);
/// positive pitch looks down), a 70 degree vertical field of view and a reverse-Z infinite projection like the game's
/// (depth 1 at the near plane, 0 at infinity), and the sun at vanilla's angle `sunDeg` (0 noon, -60 mid-morning in the
/// east).
private struct GiTestView {
    let w: Int, h: Int
    let fwd: SIMD3<Float>
    let viewProj: simd_float4x4, invViewProj: simd_float4x4
    let sunA: Float, sunDir: SIMD3<Float>, sunUp: Float
    let cam: SIMD3<Double>, origin: SIMD3<Double>

    init(_ s: GiTestScene, _ camX: Double, _ camY: Double, _ camZ: Double, _ yawDeg: Float, _ pitchDeg: Float, _ width: Int32, _ height: Int32,
         _ sunDeg: Float) {
        w = Int(width); h = Int(height)
        // View space looks down -z.
        let yaw = yawDeg * .pi / 180, pitch = pitchDeg * .pi / 180
        fwd = simd_normalize(SIMD3<Float>(-sin(yaw) * cos(pitch), -sin(pitch), cos(yaw) * cos(pitch)))
        let right = simd_normalize(simd_cross(fwd, SIMD3<Float>(0, 1, 0)))
        let up = simd_cross(right, fwd)
        let view = simd_float4x4(rows: [SIMD4(right, 0), SIMD4(up, 0), SIMD4(-fwd, 0), SIMD4(0, 0, 0, 1)])
        let f = 1 / tan(Float(35) * .pi / 180), aspect = Float(w) / Float(h), near: Float = 0.05
        let proj = simd_float4x4(columns: (SIMD4(f / aspect, 0, 0, 0), SIMD4(0, f, 0, 0), SIMD4(0, 0, 0, -1), SIMD4(0, 0, near, 0)))
        viewProj = proj * view
        invViewProj = viewProj.inverse
        sunA = sunDeg * .pi / 180
        sunDir = SIMD3<Float>(-sin(sunA), cos(sunA), 0)
        sunUp = sunDir.y > 0.1 ? 1 : max(0, sunDir.y * 10)
        cam = SIMD3(camX, camY, camZ)
        origin = SIMD3(Double(s.origin.x), Double(s.origin.y), Double(s.origin.z))
    }

    func params(_ cache: GiCache) -> GiParams {
        cache.params(invViewProj: invViewProj, cam: cam, origin: origin, sunDir: sunDir, sunUp: sunUp, width: w, height: h)
    }
}

/// The offline test's targets at one size: what the game's depth buffer and shadow pass would give (depth; each pixel's
/// primitive data, standing in for lit mode's G-buffer; the sun's visibility), the resolve's output, the debug view's,
/// gi_test_upsample's and gi_test_taa's, and gi_test_shadow's (a quarter of the size each way).
private final class GiTestTargets {
    let w: Int, h: Int, hw: Int, hh: Int
    let depthBuf: MTLBuffer
    let depth: MTLTexture, prim: MTLTexture, sunVis: MTLTexture, gi: MTLTexture, viewF: MTLTexture, lum: MTLTexture, shadow: MTLTexture

    init?(_ w: Int, _ h: Int) {
        let dev = ctx.device
        self.w = w; self.h = h; hw = (w + 1) / 2; hh = (h + 1) / 2
        func tex(_ format: MTLPixelFormat, _ usage: MTLTextureUsage, half: Bool = false) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: half ? (w + 1) / 2 : w, height: half ? (h + 1) / 2 : h,
                                                             mipmapped: false)
            d.usage = usage
            d.storageMode = .private
            return dev.makeTexture(descriptor: d)
        }
        guard let db = dev.makeBuffer(length: w * h * 4, options: .storageModePrivate),
              let dt = tex(.depth32Float, [.shaderRead]), let pr = tex(.r32Uint, [.shaderRead, .shaderWrite]),
              let sv = tex(.r8Unorm, [.shaderRead, .shaderWrite]), let g = tex(.rg32Uint, [.shaderRead, .shaderWrite], half: true),
              let vf = tex(.rgba32Float, [.shaderWrite]), let lu = tex(.r16Float, [.shaderWrite]) else { return nil }
        let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: (w + 3) / 4, height: (h + 3) / 4, mipmapped: false)
        sd.usage = [.shaderWrite]
        sd.storageMode = .private
        guard let sh = dev.makeTexture(descriptor: sd) else { return nil }
        depthBuf = db; depth = dt; prim = pr; sunVis = sv; gi = g; viewF = vf; lum = lu; shadow = sh
    }
}

/// The test's G-buffer for view `v` (primary rays, gi_test_primary) into `t`. Blocks until done; returns its GPU ms.
@discardableResult
private func giTestPrimary(_ s: GiTestScene, _ cache: GiCache, _ v: GiTestView, _ t: GiTestTargets, _ p: inout GiParams) -> Double {
    guard let tlas = s.tlas, let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return -1 }
    var viewProj = v.viewProj
    enc.setComputePipelineState(cache.pipe("gi_test_primary"))
    enc.setAccelerationStructure(tlas, bufferIndex: 0)
    enc.useResources(s.blas, usage: .read)
    cache.bindRays(enc)
    enc.setBytes(&p, length: MemoryLayout<GiParams>.stride, index: 1)
    enc.setBytes(&viewProj, length: 64, index: 11)
    enc.setBuffer(t.depthBuf, offset: 0, index: 12)
    enc.setTexture(t.prim, index: 2)
    enc.setTexture(t.sunVis, index: 3)
    enc.dispatchThreads(MTLSize(width: t.w, height: t.h, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
    enc.endEncoding()
    guard let b = cb.makeBlitCommandEncoder() else { return -1 }
    b.copy(from: t.depthBuf, sourceOffset: 0, sourceBytesPerRow: t.w * 4, sourceBytesPerImage: t.w * t.h * 4,
           sourceSize: MTLSize(width: t.w, height: t.h, depth: 1), to: t.depth, destinationSlice: 0, destinationLevel: 0,
           destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
    b.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    return (cb.gpuEndTime - cb.gpuStartTime) * 1000
}

/// The test's light for view `v`: the prototype's, or lit mode's daylight curve at its sun angle (mmc_debug_gi_lit_light)
/// as gi_light makes it in the game.
private func giTestLight(_ s: GiTestScene, _ cache: GiCache, _ v: GiTestView) {
    if s.litLight { cache.debugDaylightLight(sunAngle: v.sunA) } else { cache.setLight(giPrototypeLight(sunUp: v.sunUp)) }
}

/// A lighting pass's use of the cache (gi_test_upsample) into `t.lum`, in `enc`. variant (limits.z): 0 as the pass
/// would, 99 its reads and write without the cache (the baseline), 97 without the cells' own lookups, 98 counting pixels.
private func giTestUpsample(_ enc: MTLComputeCommandEncoder, _ cache: GiCache, _ t: GiTestTargets, _ p: GiParams, variant: Float) {
    var q = p
    q.limits.z = variant
    enc.setComputePipelineState(cache.pipe("gi_test_upsample"))
    enc.setBytes(&q, length: MemoryLayout<GiParams>.stride, index: 1)
    enc.setBuffer(cache.check, offset: 0, index: 2)
    enc.setBuffer(cache.value, offset: 0, index: 5)
    enc.setBuffer(cache.counters, offset: 0, index: 8)
    enc.setTexture(t.depth, index: 0)
    enc.setTexture(t.gi, index: 1)
    enc.setTexture(t.prim, index: 2)
    enc.setTexture(t.lum, index: 4)
    enc.dispatchThreads(MTLSize(width: t.w, height: t.h, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
}

/// Debug view `mode` (gi_debug_view) of the resolve's output in `t` at `exposure`, into `rgba` (8-bit, top row first,
/// box-filtered by `scale` each way) and, if given, `floats` (the view's float RGBA, width x height x 4, top row first).
/// Returns the view's GPU ms.
@discardableResult
private func giTestDebugView(_ cache: GiCache, _ t: GiTestTargets, _ p: GiParams, mode: Int32, exposure: Float, scale: Int,
                             rgba: UnsafeMutablePointer<UInt8>?, floats: UnsafeMutablePointer<Float>?) -> Double {
    var q = p
    q.limits.z = Float(mode)
    q.look.x = exposure
    let w = t.w, h = t.h
    guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return -1 }
    enc.setComputePipelineState(cache.pipe("gi_debug_view"))
    enc.setBytes(&q, length: MemoryLayout<GiParams>.stride, index: 1)
    enc.setBuffer(cache.check, offset: 0, index: 2)
    enc.setBuffer(cache.value, offset: 0, index: 5)
    enc.setBuffer(cache.meta, offset: 0, index: 6)
    enc.setBuffer(cache.mats, offset: 0, index: 9)
    enc.setBuffer(cache.light, offset: 0, index: 15)
    cache.bindRays(enc)
    enc.setTexture(t.depth, index: 0)
    enc.setTexture(t.gi, index: 1)
    enc.setTexture(t.prim, index: 2)
    enc.setTexture(t.sunVis, index: 3)
    enc.setTexture(t.viewF, index: 4)
    enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
    guard let rcb = ctx.queue.makeCommandBuffer(), let rb = rcb.makeBlitCommandEncoder(),
          let fbuf = ctx.device.makeBuffer(length: w * h * 16, options: .storageModeShared) else { return -1 }
    rb.copy(from: t.viewF, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: w, height: h, depth: 1),
            to: fbuf, destinationOffset: 0, destinationBytesPerRow: w * 16, destinationBytesPerImage: w * h * 16)
    rb.endEncoding()
    rcb.commit()
    rcb.waitUntilCompleted()
    // Texture row 0 is the bottom of the image (the game's convention): flipped.
    let px = fbuf.contents().bindMemory(to: Float.self, capacity: w * h * 4)
    if let floats {
        for y in 0..<h { (floats + y * w * 4).update(from: px + (h - 1 - y) * w * 4, count: w * 4) }
    }
    if let rgba {
        let k = max(1, scale), ow = w / k, oh = h / k
        for y in 0..<oh {
            for x in 0..<ow {
                for ch in 0..<4 {
                    var acc: Float = 0
                    for dy in 0..<k { for dx in 0..<k { acc += max(0, min(1, px[((h - 1 - (y * k + dy)) * w + x * k + dx) * 4 + ch])) } }
                    rgba[(y * ow + x) * 4 + ch] = UInt8(acc / Float(k * k) * 255 + 0.5)
                }
            }
        }
    }
    return ms
}

/// Debug: runs the cache for `frames` frames from a fixed camera (Minecraft yaw and pitch in degrees, 70 degree
/// vertical field of view) at width x height, then draws debug view `mode` (gi_debug_view) at `exposure` into `rgba`
/// (width x height x 4 bytes, top row first). The sun is vanilla's at `sunDeg` (0 noon, -60 mid-morning in the east). The
/// cache persists between calls (mmc_debug_gi_reset clears it).
/// stats: 0-4 mean GPU ms per frame of the primary rays (test only), request (with the resolve, as a frame runs it),
/// schedule, update, the resolve alone (after the update, for the views) over the frames after the first quarter; 5-8
/// their maxima (request, schedule, update, resolve); 9 live cells; 10 converged cells;
/// 11 cells created by the screen (cumulative); 12 bucket-full inserts; 13 evictions; 14 surfaces not found; 15 rays traced in the
/// last frame; 16 cells updated in the last frame; 17 bounce hits from the cache, 18 without (cumulative);
/// 19 mean relative change of the resolved irradiance between the last two frames (flicker); 20 resolve samples with
/// cache data (share of all, sky included); 21 debug view ms; 22 cells created by bounce rays; 23 samples taken from a
/// cell's first surface point; 24 ms of gi_test_upsample (a lighting pass's use of the cache, with its reads); 25 the
/// same less its reads and write (what the cache adds to a deferred lighting pass); 26-28 pixels with a surface, that
/// needed the cells' own lookups (no matching half-resolution sample), that found no data (cells without samples);
/// 29 what the cache adds to a deferred lighting pass without those lookups.
@_cdecl("mmc_debug_gi_run")
public func mmc_debug_gi_run(_ camX: Double, _ camY: Double, _ camZ: Double, _ yawDeg: Float, _ pitchDeg: Float, _ width: Int32, _ height: Int32,
                             _ frames: Int32, _ sunDeg: Float, _ mode: Int32, _ exposure: Float, _ rgba: UnsafeMutablePointer<UInt8>,
                             _ stats: UnsafeMutablePointer<Double>) -> Int32 {
    guard let s = giTest, let cache = s.cache, let tlas = s.tlas else { return 0 }
    let queue = ctx.queue
    let v = GiTestView(s, camX, camY, camZ, yawDeg, pitchDeg, width, height, sunDeg)
    guard let t = GiTestTargets(v.w, v.h), let readback = ctx.device.makeBuffer(length: t.hw * t.hh * 8, options: .storageModeShared) else { return 0 }
    let hw = t.hw, hh = t.hh
    func resolve(_ enc: MTLComputeCommandEncoder, _ p: inout GiParams) { cache.encodeResolve(enc, depth: t.depth, out: t.gi, params: &p) }
    // METALMC_GIREPEAT=<n>: the request, update and resolve run n times in their command buffers (times are per run),
    // after a warm-up, so the GPU's clock has ramped up as it would in a busy frame; one run each leaves it idling
    // between tiny command buffers. Repeated requests find their cells listed already (cheaper than the first) and
    // repeated updates update the same cells again: timing only.
    let repeats = max(1, Int(giEnv("METALMC_GIREPEAT") ?? "") ?? 1)
    var times = [[Double]](repeating: [], count: 5)
    func timed(_ i: Int, _ runs: Int = 1, _ body: (MTLCommandBuffer) -> Void) {
        guard let cb = queue.makeCommandBuffer() else { return }
        for _ in 0..<runs { body(cb) }
        cb.commit()
        cb.waitUntilCompleted()
        times[i].append((cb.gpuEndTime - cb.gpuStartTime) * 1000 / Double(runs))
    }
    giTestLight(s, cache, v)
    var p = v.params(cache)
    // The G-buffer (what the game's depth buffer and shadow pass would provide), once: the camera doesn't move.
    times[0].append(giTestPrimary(s, cache, v, t, &p))
    if repeats > 1, let cb = queue.makeCommandBuffer() {
        for _ in 0..<60 { if let enc = cb.makeComputeCommandEncoder() { resolve(enc, &p); enc.endEncoding() } }
        cb.commit()
        cb.waitUntilCompleted()
    }
    var prevResolve: [Float] = []
    var lastRays = 0.0, lastUpdated = 0.0
    let n = max(1, Int(frames))
    for fr in 0..<n {
        p = v.params(cache)
        let rays0 = cache.counters.contents().load(fromByteOffset: 44, as: UInt32.self)
        if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
            cache.encodeBegin(enc, params: &p)
            enc.endEncoding()
            cb.commit()
        }
        timed(1, repeats) { cb in if let enc = cb.makeComputeCommandEncoder() { cache.encodeRequest(enc, depth: t.depth, out: t.gi, params: &p); enc.endEncoding() } }
        timed(2) { cb in if let enc = cb.makeComputeCommandEncoder() { cache.encodeSchedule(enc, params: &p); enc.endEncoding() } }
        timed(3, repeats) { cb in if let enc = cb.makeComputeCommandEncoder() { cache.encodeUpdate(enc, accel: tlas, accels: s.blas, params: &p); enc.endEncoding() } }
        timed(4, repeats) { cb in if let enc = cb.makeComputeCommandEncoder() { resolve(enc, &p); enc.endEncoding() } }
        let c = cache.counters.contents().bindMemory(to: UInt32.self, capacity: 16)
        lastRays = Double(c[11] &- rays0) / Double(repeats)
        lastUpdated = Double(c[1])
        cache.advance()
        if fr >= n - 2 {
            // The resolved irradiance of the last two frames, for the flicker measure.
            guard let cb = queue.makeCommandBuffer(), let b = cb.makeBlitCommandEncoder() else { return 0 }
            b.copy(from: t.gi, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: hw, height: hh, depth: 1),
                   to: readback, destinationOffset: 0, destinationBytesPerRow: hw * 8, destinationBytesPerImage: hw * hh * 8)
            b.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            // rgb and 1 (data) or 0 per sample.
            let words = readback.contents().bindMemory(to: UInt32.self, capacity: hw * hh * 2)
            var cur = [Float](repeating: 0, count: hw * hh * 4)
            for i in 0..<(hw * hh) where words[2 * i + 1] & (1 << 19) != 0 {
                let v = words[2 * i]
                cur[4 * i] = giUnpackSmallFloat(v & 0x7FF, mantissaBits: 6)
                cur[4 * i + 1] = giUnpackSmallFloat((v >> 11) & 0x7FF, mantissaBits: 6)
                cur[4 * i + 2] = giUnpackSmallFloat(v >> 22, mantissaBits: 5)
                cur[4 * i + 3] = 1
            }
            if fr == n - 1 { s.lastResolve = cur } else { prevResolve = cur }
        }
    }
    // The lighting pass's share (gi_test_upsample), timed like the stages, less the same kernel without the cache (the
    // reads and write a deferred pass does anyway).
    var ms = [0.0, 0.0, 0.0]
    for (i, variant) in [Float(0), 99, 97].enumerated() {
        timed(0, repeats) { cb in
            guard let enc = cb.makeComputeCommandEncoder() else { return }
            giTestUpsample(enc, cache, t, p, variant: variant)
            enc.endEncoding()
        }
        ms[i] = times[0].last ?? 0
    }
    stats[24] = ms[0]; stats[25] = ms[0] - ms[1]; stats[29] = ms[2] - ms[1]
    // Once more, counting which way the pixels went.
    let counts = cache.counters.contents().bindMemory(to: UInt32.self, capacity: 32)
    counts[20] = 0; counts[21] = 0; counts[22] = 0
    if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
        giTestUpsample(enc, cache, t, p, variant: 98)
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
    }
    stats[26] = Double(counts[20]); stats[27] = Double(counts[21]); stats[28] = Double(counts[22])
    // Statistics, the flicker measure, then the debug view.
    giTestCount(cache, &p)
    let c = cache.counters.contents().bindMemory(to: UInt32.self, capacity: 16)
    let skip = max(1, n / 4)
    func mean(_ a: [Double], _ from: Int) -> Double { let t = a.count > from ? Array(a[from...]) : a; return t.isEmpty ? 0 : t.reduce(0, +) / Double(t.count) }
    stats[0] = times[0].first ?? 0   // the primary rays (times[0] also holds the upsample's, stats[24])
    for i in 1...4 { stats[i] = mean(times[i], skip); stats[4 + i] = (times[i].count > skip ? times[i][skip...] : times[i][...]).max() ?? 0 }
    stats[9] = Double(c[6]); stats[10] = Double(c[7]); stats[11] = Double(c[2]); stats[12] = Double(c[3]); stats[13] = Double(c[4]); stats[14] = Double(c[5])
    stats[15] = lastRays; stats[16] = lastUpdated; stats[17] = Double(c[12]); stats[18] = Double(c[13])
    var diff = 0.0, sum = 0.0, withData = 0
    let cur = s.lastResolve
    if prevResolve.count == cur.count {
        for i in stride(from: 0, to: cur.count, by: 4) where cur[i + 3] > 0 && prevResolve[i + 3] > 0 {
            for ch in 0..<3 { diff += abs(Double(cur[i + ch]) - Double(prevResolve[i + ch])); sum += Double(cur[i + ch]) }
        }
    }
    for i in stride(from: 0, to: cur.count, by: 4) where cur[i + 3] > 0 { withData += 1 }
    stats[19] = sum > 0 ? diff / sum : 0
    // The resolve writes w = 0 for the sky and for no data alike, so this is over all (half-resolution) samples.
    stats[20] = Double(withData) / Double(hw * hh)
    stats[22] = Double(c[14]); stats[23] = Double(c[15])
    stats[21] = giTestDebugView(cache, t, p, mode: mode, exposure: exposure, scale: 1, rgba: rgba, floats: nil)
    return 1
}

/// Live and converged cells into counters 6 and 7 (gi_count). Blocks until done.
private func giTestCount(_ cache: GiCache, _ p: inout GiParams) {
    guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return }
    let c = cache.counters.contents().bindMemory(to: UInt32.self, capacity: 16)
    c[6] = 0; c[7] = 0
    enc.setBytes(&p, length: MemoryLayout<GiParams>.stride, index: 1)
    enc.setBuffer(cache.check, offset: 0, index: 2)
    enc.setBuffer(cache.meta, offset: 0, index: 6)
    enc.setBuffer(cache.counters, offset: 0, index: 8)
    enc.setComputePipelineState(cache.pipe("gi_count"))
    enc.dispatchThreads(MTLSize(width: cache.slots, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
}

/// Debug: GPU times of the cache's frame as the game runs it: encodeFrame with lit mode's light (its daylight curve, as
/// gi_light makes it without the sky), with the game profile's per-encoder timestamps, over `frames` frames. Each frame starts with a blit writing `flushMB` MB, standing in for the rest of a
/// frame's traffic (the cache's reads start cold, as in the game), and ends with the anti-aliasing resolve's load loop
/// as lit mode runs it (gi_test_taa: its share of the cache's light, the upsample). Frames are committed one at a time,
/// each once the last is done, so no two frames' encoders overlap (they would: nothing orders one frame's last
/// encoders before the next one's first). The camera and light are fixed: the first frames converge the cells as
/// mmc_debug_gi_run's do.
/// mode: bit 0 first the shadow rays the game traces just before the cache (gi_test_shadow: they warm the depth buffer
/// and the structures); bit 1 every encoder waits for the one before (a fence: each time is its own; without it,
/// independent encoders overlap as they do in a frame); bit 2 every other frame without the cache (and the load loop
/// without its share): the frame's time with the cache less without it is what the cache costs the frame. `fly`: blocks
/// the camera moves along its view each frame (the game's benchmark is a flight: cells keep coming into view and
/// changing level), its depth buffer drawn again each frame.
/// out: per encoder label, in the order they first appear (up to 12), the median GPU ms over the timed frames (after the
/// first quarter) that have it and the fastest (out[2i], out[2i + 1]); 24-25 the cache's encoders together, the first's
/// start to the last's end; 26-27 the frame with the cache, 28-29 without it (bit 2); 30 rays a frame (mean over the timed
/// frames with the cache); 31 cells updated in the last frame; 32 live cells. `labels` gets the labels, separated by '|'.
/// Returns the number of labels, 0 if it didn't run.
@_cdecl("mmc_debug_gi_profile")
public func mmc_debug_gi_profile(_ camX: Double, _ camY: Double, _ camZ: Double, _ yawDeg: Float, _ pitchDeg: Float, _ width: Int32, _ height: Int32,
                                 _ frames: Int32, _ sunDeg: Float, _ flushMB: Int32, _ mode: Int32, _ fly: Float, _ out: UnsafeMutablePointer<Double>,
                                 _ labels: UnsafeMutablePointer<CChar>, _ labelsLen: Int32) -> Int32 {
    guard let s = giTest, let cache = s.cache, let tlas = s.tlas else { return 0 }
    let dev = ctx.device, queue = ctx.queue
    var v = GiTestView(s, camX, camY, camZ, yawDeg, pitchDeg, width, height, sunDeg)
    guard let t = GiTestTargets(v.w, v.h) else { return 0 }
    // (params moves the sweep's length along: put it back, so only encodeFrame's own calls do.)
    let state = cache.debugState()
    var p = v.params(cache)
    cache.debugRestore(state)
    giTestPrimary(s, cache, v, t, &p)
    let shadowFirst = mode & 1 != 0, serial = mode & 2 != 0, ab = mode & 4 != 0
    guard dev.supportsCounterSampling(.atStageBoundary),
          let set = dev.counterSets?.first(where: { $0.name == MTLCommonCounterSet.timestamp.rawValue }) else { return 0 }
    let sd = MTLCounterSampleBufferDescriptor()
    sd.counterSet = set
    sd.storageMode = .shared
    sd.sampleCount = 32
    guard let samples = try? dev.makeCounterSampleBuffer(descriptor: sd) else { return 0 }
    let fence = serial ? dev.makeFence() : nil
    let flushBytes = Int(max(0, flushMB)) << 20
    let flush = flushBytes > 0 ? dev.makeBuffer(length: flushBytes, options: .storageModePrivate) : nil
    let daylight = litDaylightEnv(sunAngle: v.sunA)
    let n = max(4, Int(frames)), skip = n / 4
    var names: [String] = [], times: [String: [Double]] = [:]
    var cacheTotal: [Double] = [], frameWith: [Double] = [], frameWithout: [Double] = []
    var rays = 0.0, raysFrames = 0
    for f in 0..<n {
        let withGi = !ab || f % 2 == 0
        if fly != 0 && f > 0 {
            let c = SIMD3<Double>(camX, camY, camZ) + SIMD3<Double>(v.fwd) * Double(fly * Float(f))
            v = GiTestView(s, c.x, c.y, c.z, yawDeg, pitchDeg, width, height, sunDeg)
            let st = cache.debugState()
            p = v.params(cache)
            cache.debugRestore(st)
            giTestPrimary(s, cache, v, t, &p)
        }
        guard let cb = queue.makeCommandBuffer() else { return 0 }
        var encs: [String] = [], cacheEncs: [Int] = []
        func attach(_ a: MTLComputePassSampleBufferAttachmentDescriptor, _ label: String) {
            a.sampleBuffer = samples
            a.startOfEncoderSampleIndex = 2 * encs.count
            a.endOfEncoderSampleIndex = 2 * encs.count + 1
            encs.append(label)
        }
        func plain(_ label: String) -> MTLComputePassDescriptor {
            let d = MTLComputePassDescriptor()
            attach(d.sampleBufferAttachments[0]!, label)
            return d
        }
        func pass(_ label: String) -> MTLComputePassDescriptor {
            cacheEncs.append(encs.count)
            return plain(label)
        }
        if let flush {
            let bd = MTLBlitPassDescriptor()
            let a = bd.sampleBufferAttachments[0]!
            a.sampleBuffer = samples
            a.startOfEncoderSampleIndex = 2 * encs.count
            a.endOfEncoderSampleIndex = 2 * encs.count + 1
            encs.append("flush")
            guard let b = cb.makeBlitCommandEncoder(descriptor: bd) else { return 0 }
            if let fence { b.waitForFence(fence) }
            b.fill(buffer: flush, range: 0..<flushBytes, value: UInt8(truncatingIfNeeded: f))
            if let fence { b.updateFence(fence) }
            b.endEncoding()
        }
        if shadowFirst {
            guard let enc = cb.makeComputeCommandEncoder(descriptor: plain("shadow rays (stand-in)")) else { return 0 }
            if let fence { enc.waitForFence(fence) }
            var q = p
            let pat = giRequestPattern[f % 16]
            q.sample.y = pat.x; q.sample.z = pat.y
            enc.setComputePipelineState(cache.pipe("gi_test_shadow"))
            enc.setAccelerationStructure(tlas, bufferIndex: 0)
            enc.useResources(s.blas, usage: .read)
            enc.setBytes(&q, length: MemoryLayout<GiParams>.stride, index: 1)
            enc.setTexture(t.depth, index: 0)
            enc.setTexture(t.shadow, index: 3)
            enc.dispatchThreads(MTLSize(width: t.shadow.width, height: t.shadow.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            if let fence { enc.updateFence(fence) }
            enc.endEncoding()
        }
        if withGi {
            guard cache.encodeFrame(cb, depth: t.depth, out: t.gi, invViewProj: v.invViewProj, cam: v.cam, origin: v.origin, accel: tlas,
                                    accels: s.blas, sunDir: v.sunDir, sunUp: v.sunUp, cloudHeight: Float(v.cam.y) + 1e9, light: .daylight(daylight),
                                    fence: fence, pass: pass) else { return 0 }
        }
        guard let enc = cb.makeComputeCommandEncoder(descriptor: plain(withGi ? "anti-aliasing load (proxy)" : "anti-aliasing load (proxy) without the cache")) else { return 0 }
        if let fence { enc.waitForFence(fence) }
        var q = p
        q.limits.z = withGi ? 0 : 99
        enc.setComputePipelineState(cache.pipe("gi_test_taa"))
        enc.setBytes(&q, length: MemoryLayout<GiParams>.stride, index: 1)
        enc.setTexture(t.depth, index: 0)
        enc.setTexture(t.gi, index: 1)
        enc.setTexture(t.prim, index: 2)
        enc.setTexture(t.lum, index: 4)
        enc.dispatchThreadgroups(MTLSize(width: (t.w + 31) / 32, height: (t.h + 31) / 32, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        if let fence { enc.updateFence(fence) }
        enc.endEncoding()
        let rays0 = cache.counters.contents().load(fromByteOffset: 44, as: UInt32.self)
        cb.commit()
        cb.waitUntilCompleted()
        guard f >= skip, let data = try? samples.resolveCounterRange(0..<(2 * encs.count)) else { continue }
        let ts = data.withUnsafeBytes { Array($0.bindMemory(to: UInt64.self)) }
        func valid(_ x: UInt64) -> Bool { x != 0 && x != UInt64.max }
        var first = UInt64.max, last: UInt64 = 0
        for (i, label) in encs.enumerated() where valid(ts[2 * i]) && valid(ts[2 * i + 1]) && ts[2 * i + 1] >= ts[2 * i] {
            if !names.contains(label) { names.append(label) }
            times[label, default: []].append(Double(ts[2 * i + 1] - ts[2 * i]) / 1e6)
            first = min(first, ts[2 * i]); last = max(last, ts[2 * i + 1])
        }
        if last > first {
            if withGi { frameWith.append(Double(last - first) / 1e6) } else { frameWithout.append(Double(last - first) / 1e6) }
        }
        if let a = cacheEncs.first, let b = cacheEncs.last, valid(ts[2 * a]), valid(ts[2 * b + 1]), ts[2 * b + 1] > ts[2 * a] {
            cacheTotal.append(Double(ts[2 * b + 1] - ts[2 * a]) / 1e6)
        }
        if withGi {
            rays += Double(cache.counters.contents().load(fromByteOffset: 44, as: UInt32.self) &- rays0)
            raysFrames += 1
        }
    }
    func stat(_ v: [Double]) -> (Double, Double) {
        let s = v.sorted()
        return s.isEmpty ? (-1, -1) : (s[s.count / 2], s[0])
    }
    for i in 0..<33 { out[i] = -1 }
    for (i, name) in names.prefix(12).enumerated() { (out[2 * i], out[2 * i + 1]) = stat(times[name] ?? []) }
    (out[24], out[25]) = stat(cacheTotal)
    (out[26], out[27]) = stat(frameWith)
    (out[28], out[29]) = stat(frameWithout)
    out[30] = raysFrames > 0 ? rays / Double(raysFrames) : 0
    let c = cache.counters.contents().bindMemory(to: UInt32.self, capacity: 20)
    out[31] = Double(c[1])
    giTestCount(cache, &p)
    out[32] = Double(c[6])
    let text = Array(names.prefix(12).joined(separator: "|").utf8.prefix(Int(labelsLen) - 1))
    for (i, b) in text.enumerated() { labels[i] = CChar(bitPattern: b) }
    labels[text.count] = 0
    return Int32(min(names.count, 12))
}

/// Debug: debug view `mode` of the cache as it stands, without running a frame (the resolve on the current cells, then
/// gi_debug_view): the same table, camera and light give the same picture, so two builds can be compared to the bit on a
/// table one of them saved (mmc_debug_gi_save, _load). Into `rgba` (8-bit, top row first, box-filtered by `scale` each
/// way: width / scale x height / scale x 4 bytes) and, if `floats` isn't null, the view's float RGBA (width x height x
/// 4, top row first). Returns 1 if it worked.
@_cdecl("mmc_debug_gi_view")
public func mmc_debug_gi_view(_ camX: Double, _ camY: Double, _ camZ: Double, _ yawDeg: Float, _ pitchDeg: Float, _ width: Int32, _ height: Int32,
                              _ sunDeg: Float, _ mode: Int32, _ exposure: Float, _ scale: Int32, _ rgba: UnsafeMutablePointer<UInt8>,
                              _ floats: UnsafeMutablePointer<Float>?) -> Int32 {
    guard let s = giTest, let cache = s.cache else { return 0 }
    let v = GiTestView(s, camX, camY, camZ, yawDeg, pitchDeg, width, height, sunDeg)
    guard let t = GiTestTargets(v.w, v.h) else { return 0 }
    giTestLight(s, cache, v)
    // (params moves the sweep's length along: put it back, so a view doesn't change the frames after it.)
    let state = cache.debugState()
    var p = v.params(cache)
    cache.debugRestore(state)
    giTestPrimary(s, cache, v, t, &p)
    guard let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
    cache.encodeResolve(enc, depth: t.depth, out: t.gi, params: &p)
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    giTestDebugView(cache, t, p, mode: mode, exposure: exposure, scale: Int(scale), rgba: rgba, floats: floats)
    return 1
}

/// Debug: writes the test cache's table, counters and the CPU's state of it to `path` (for mmc_debug_gi_load, in this
/// build or another with the same table). Returns 1 if it worked.
@_cdecl("mmc_debug_gi_save")
public func mmc_debug_gi_save(_ path: UnsafePointer<CChar>) -> Int32 {
    guard let cache = giTest?.cache else { return 0 }
    let bufs = [cache.check, cache.keys, cache.stamp, cache.value, cache.meta]
    let total = bufs.reduce(0) { $0 + $1.length }
    guard let staging = ctx.device.makeBuffer(length: total, options: .storageModeShared), let cb = ctx.queue.makeCommandBuffer(),
          let b = cb.makeBlitCommandEncoder() else { return 0 }
    var off = 0
    for x in bufs { b.copy(from: x, sourceOffset: 0, to: staging, destinationOffset: off, size: x.length); off += x.length }
    b.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    var data = Data()
    let header: [Int64] = [Int64(cache.slots), Int64(total), Int64(cache.counters.length)] + cache.debugState()
    header.withUnsafeBytes { data.append(contentsOf: $0) }
    data.append(Data(bytes: cache.counters.contents(), count: cache.counters.length))
    data.append(Data(bytes: staging.contents(), count: total))
    do { try data.write(to: URL(fileURLWithPath: String(cString: path))) } catch { return 0 }
    return 1
}

/// Debug: reads a table mmc_debug_gi_save wrote into the test cache (the same number of slots). Returns 1 if it worked.
@_cdecl("mmc_debug_gi_load")
public func mmc_debug_gi_load(_ path: UnsafePointer<CChar>) -> Int32 {
    guard let cache = giTest?.cache, let data = try? Data(contentsOf: URL(fileURLWithPath: String(cString: path))), data.count >= 56 else { return 0 }
    let header = data.prefix(56).withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }
    let bufs = [cache.check, cache.keys, cache.stamp, cache.value, cache.meta]
    let total = bufs.reduce(0) { $0 + $1.length }
    guard header[0] == Int64(cache.slots), header[1] == Int64(total), header[2] == Int64(cache.counters.length),
          data.count == 56 + cache.counters.length + total,
          let staging = ctx.device.makeBuffer(length: total, options: .storageModeShared), let cb = ctx.queue.makeCommandBuffer(),
          let b = cb.makeBlitCommandEncoder() else { return 0 }
    data.withUnsafeBytes { raw in
        cache.counters.contents().copyMemory(from: raw.baseAddress! + 56, byteCount: cache.counters.length)
        staging.contents().copyMemory(from: raw.baseAddress! + 56 + cache.counters.length, byteCount: total)
    }
    var off = 0
    for x in bufs { b.copy(from: staging, sourceOffset: off, to: x, destinationOffset: 0, size: x.length); off += x.length }
    b.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    cache.debugRestore(Array(header[3...]))
    return 1
}

/// Debug: compares two float RGBA pictures of debug view 9 as mmc_debug_gi_view writes them (raw files, 4 floats a pixel:
/// the irradiance where lit mode's relight takes the cache's light, a negative code where it doesn't). out: 0 pixels,
/// 1 the same to the bit, 2 with light in both, 3 in one only, 4 in neither but with different codes, 5-6 the mean
/// luminance of a and b where both have light, 7 the mean absolute difference of their luminance there (relative to
/// a's mean), 8 the largest relative difference of a channel (of values over 1e-3), 9 pixels with a channel more than
/// 1/32 apart (relative; RG11B10's steps are 1/64 and 1/32). Returns 1 if it worked.
@_cdecl("mmc_debug_gi_diff")
public func mmc_debug_gi_diff(_ pathA: UnsafePointer<CChar>, _ pathB: UnsafePointer<CChar>, _ out: UnsafeMutablePointer<Double>) -> Int32 {
    guard let a = try? Data(contentsOf: URL(fileURLWithPath: String(cString: pathA))),
          let b = try? Data(contentsOf: URL(fileURLWithPath: String(cString: pathB))), a.count == b.count, a.count % 16 == 0 else { return 0 }
    let n = a.count / 16
    var same = 0, both = 0, one = 0, codes = 0, far = 0
    var sumA = 0.0, sumB = 0.0, sumD = 0.0, maxRel = 0.0
    a.withUnsafeBytes { ra in
        b.withUnsafeBytes { rb in
            let fa = ra.bindMemory(to: Float.self), fb = rb.bindMemory(to: Float.self)
            let wa = ra.bindMemory(to: UInt32.self), wb = rb.bindMemory(to: UInt32.self)
            for i in 0..<n {
                let o = 4 * i
                if wa[o] == wb[o] && wa[o + 1] == wb[o + 1] && wa[o + 2] == wb[o + 2] && wa[o + 3] == wb[o + 3] { same += 1 }
                let ha = fa[o] >= 0, hb = fb[o] >= 0
                if ha && hb {
                    both += 1
                    let la = 0.2126 * Double(fa[o]) + 0.7152 * Double(fa[o + 1]) + 0.0722 * Double(fa[o + 2])
                    let lb = 0.2126 * Double(fb[o]) + 0.7152 * Double(fb[o + 1]) + 0.0722 * Double(fb[o + 2])
                    sumA += la; sumB += lb; sumD += abs(la - lb)
                    var big = false
                    for ch in 0..<3 {
                        let x = Double(fa[o + ch]), y = Double(fb[o + ch]), m = max(x, y)
                        if m > 1e-3 {
                            let r = abs(x - y) / m
                            maxRel = max(maxRel, r)
                            if r > 1.0 / 32.0 { big = true }
                        }
                    }
                    if big { far += 1 }
                } else if ha != hb {
                    one += 1
                } else if fa[o] != fb[o] {
                    codes += 1
                }
            }
        }
    }
    out[0] = Double(n); out[1] = Double(same); out[2] = Double(both); out[3] = Double(one); out[4] = Double(codes)
    out[5] = both > 0 ? sumA / Double(both) : 0; out[6] = both > 0 ? sumB / Double(both) : 0
    out[7] = sumA > 0 ? sumD / sumA : 0; out[8] = maxRel; out[9] = Double(far)
    return 1
}

/// Debug: recompiles the test cache's kernels from the file at `path` (a variant of mmc_debug_gi_shader_source's text;
/// null: the built-in source), keeping its table: kernel variants timed against each other in one process (gicache.py
/// exp). Returns 1 if it worked.
@_cdecl("mmc_debug_gi_reload")
public func mmc_debug_gi_reload(_ path: UnsafePointer<CChar>?) -> Int32 {
    guard let cache = giTest?.cache else { return 0 }
    var body: String?
    if let path {
        guard let s = try? String(contentsOfFile: String(cString: path), encoding: .utf8) else { return 0 }
        body = s
    }
    return cache.debugReload(body) ? 1 : 0
}

/// Debug: empties the test cache and its statistics.
@_cdecl("mmc_debug_gi_reset")
public func mmc_debug_gi_reset() {
    guard let cache = giTest?.cache, let cb = ctx.queue.makeCommandBuffer() else { return }
    cache.clear(cb)
    cb.commit()
    cb.waitUntilCompleted()
    memset(cache.counters.contents(), 0, cache.counters.length)
}

/// Debug: 1 lights the test scene with lit mode's daylight curve (litDaylightEnv at each run's sun angle, through
/// gi_light, as in the game without the atmosphere; debug views 6 and 7 are lit mode's relight without and with the
/// cache), 0 with the prototype's light.
@_cdecl("mmc_debug_gi_lit_light")
public func mmc_debug_gi_lit_light(_ on: Int32) {
    giTest?.litLight = on != 0
}

/// Debug: the kernels' self-checks on the current scene. out: [key round trips checked, mismatches, rays checked, rays
/// whose results were wrong], then per test ray (up to 8) no-cull hit, face, front facing, distance, culled hit, face,
/// distance. The rays (synthetic scene): down onto the ground at (20, 70, 20), up from under the house's roof inside
/// it, from inside the stone under the ground upward (the ground's top from below: back face), and west along the
/// tunnel's axis from outside the hill.
@_cdecl("mmc_debug_gi_selftest")
public func mmc_debug_gi_selftest(_ out: UnsafeMutablePointer<Double>) -> Int32 {
    guard let s = giTest, let cache = s.cache, let tlas = s.tlas else { return 0 }
    let dev = ctx.device
    // Keys: blocks at both signs, both sides of 2^23, the world's bottom and top, every face and levels 0-12.
    var blocks: [SIMD4<Int32>] = []
    for x in [-30_000_000, -8_388_609, -513, -1, 0, 1, 255, 256, 8_388_607, 29_999_999] {
        for y in [-64, -1, 0, 63, 319] {
            for f in 0..<6 { for l in [0, 1, 3, 8, 12] { blocks.append(SIMD4(Int32(x), Int32(y), Int32(-x / 3), Int32(f | l << 8))) } }
        }
    }
    guard let bin = dev.makeBuffer(bytes: blocks, length: blocks.count * 16, options: .storageModeShared),
          let bout = dev.makeBuffer(length: blocks.count * 32, options: .storageModeShared),
          let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
    enc.setComputePipelineState(cache.pipe("gi_test_keys"))
    enc.setBuffer(bin, offset: 0, index: 13)
    enc.setBuffer(bout, offset: 0, index: 14)
    enc.dispatchThreads(MTLSize(width: blocks.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    // Rays in the scene's frame (relative to its origin).
    let o = SIMD3<Float>(Float(-s.origin.x), Float(-s.origin.y), Float(-s.origin.z))
    let rays: [(SIMD3<Float>, SIMD3<Float>)] = [(SIMD3(20.5, 70, 20.5), SIMD3(0, -1, 0)), (SIMD3(108.5, 66.5, 108.5), SIMD3(0, 1, 0)),
                                                 (SIMD3(20.5, 50.5, 20.5), SIMD3(0, 1, 0)), (SIMD3(230.5, 65.5, 182.5), SIMD3(-1, 0, 0))]
    var rdata: [SIMD4<Float>] = []
    for (a, d) in rays { rdata.append(SIMD4(a + o, 1000)); rdata.append(SIMD4(d, 0)) }
    guard let rin = dev.makeBuffer(bytes: rdata, length: rdata.count * 16, options: .storageModeShared),
          let rout = dev.makeBuffer(length: rays.count * 32, options: .storageModeShared) else { return 0 }
    enc.setComputePipelineState(cache.pipe("gi_test_rays"))
    enc.setAccelerationStructure(tlas, bufferIndex: 0)
    enc.useResources(s.blas, usage: .read)
    cache.bindRays(enc)
    enc.setBuffer(rin, offset: 0, index: 13)
    enc.setBuffer(rout, offset: 0, index: 14)
    enc.dispatchThreads(MTLSize(width: rays.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: rays.count, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    // Expected keys on the CPU.
    let r = bout.contents().bindMemory(to: SIMD4<Int32>.self, capacity: blocks.count * 2)
    var bad = 0
    let normals: [SIMD3<Int>] = [SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)]
    func fdiv(_ v: Int, _ l: Int) -> Int { Int((Double(v) / Double(1 << l)).rounded(.down)) }
    for (i, b) in blocks.enumerated() {
        let f = Int(b.w & 255), l = Int(b.w >> 8)
        let air = SIMD3(Int(b.x), Int(b.y), Int(b.z)) &+ normals[f]
        let want = SIMD3(fdiv(air.x, l), fdiv(max(air.y + 64, 0), l), fdiv(air.z, l))
        let got = r[2 * i], back = r[2 * i + 1]
        if Int(got.x) != want.x || Int(got.y) != want.y || Int(got.z) != want.z || Int(got.w) & 7 != f { bad += 1; continue }
        if back.x != got.x || back.y != got.y || back.z != got.z || Int(back.w) != l { bad += 1 }
    }
    out[0] = Double(blocks.count); out[1] = Double(bad)
    // Rays: expected (no-cull face, front facing, culled hit): ground from above: +Y, front, hit +Y; under the roof going
    // up: -Y, front, hit; from inside the stone upward: the ground's top face from below (+Y, back), and culled: the next
    // front face upward (the house? none at x 20: no hit); west along the tunnel: +X face at its far end (front).
    let q = rout.contents().bindMemory(to: SIMD4<UInt32>.self, capacity: rays.count * 2)
    let expect: [(UInt32, UInt32, UInt32, UInt32)] = [(2, 1, 1, 2), (3, 1, 1, 3), (2, 0, 0, 9), (0, 1, 1, 0)]
    var rayBad = 0
    for i in 0..<rays.count {
        let a = q[2 * i], b = q[2 * i + 1]
        let e = expect[i]
        if a.y != e.0 || a.z != e.1 || b.x != e.2 || b.y != e.3 { rayBad += 1 }
        let base = 4 + i * 7
        out[base] = Double(a.x); out[base + 1] = Double(a.y); out[base + 2] = Double(a.z); out[base + 3] = Double(Float(bitPattern: a.w))
        out[base + 4] = Double(b.x); out[base + 5] = Double(b.y); out[base + 6] = Double(Float(bitPattern: b.w))
    }
    out[2] = Double(rays.count); out[3] = Double(rayBad)
    // The resolve's packing against the texture unit's RG11B10Float conversion: every small float's value, the halfway
    // points between neighbors and a float's step either side of them (where rounding and truncating part), pseudo-random
    // values in between, and zero, -0, negatives and values past the largest.
    var vals: [Float] = [0, -0.0, -1, -1e-20, 1e-30, 1e-8, 65000, 64512, 64600, 65024, 65100, 70000]
    var rng: UInt32 = 12345
    for e in -24...15 {
        for m in 0..<64 {
            let a = Float(sign: .plus, exponent: e, significand: 1 + Float(m) / 64), b = Float(sign: .plus, exponent: e, significand: 1 + Float(m + 1) / 64)
            // (Blue keeps 5 mantissa bits: its halfway points are the odd m's a.)
            let mid = (a + b) / 2
            vals += [a, a.nextUp, a.nextDown, mid, mid.nextUp, mid.nextDown]
            for _ in 0..<4 {
                rng = rng &* 1664525 &+ 1013904223
                vals.append(a + (b - a) * Float(rng >> 8) / 16777216)
            }
        }
    }
    while vals.count % 256 != 0 { vals.append(0.5) }
    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg11b10Float, width: 256, height: vals.count / 256, mipmapped: false)
    d.usage = [.shaderWrite, .shaderRead]
    d.storageMode = .private
    guard let vb = dev.makeBuffer(bytes: vals, length: vals.count * 4, options: .storageModeShared),
          let mine = dev.makeBuffer(length: vals.count * 4, options: .storageModeShared),
          let theirs = dev.makeBuffer(length: vals.count * 4, options: .storageModeShared), let hw = dev.makeTexture(descriptor: d),
          let pcb = ctx.queue.makeCommandBuffer(), let penc = pcb.makeComputeCommandEncoder() else { return 0 }
    penc.setComputePipelineState(cache.pipe("gi_test_pack"))
    penc.setBuffer(vb, offset: 0, index: 13)
    penc.setBuffer(mine, offset: 0, index: 14)
    penc.setTexture(hw, index: 1)
    penc.dispatchThreads(MTLSize(width: vals.count, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    penc.endEncoding()
    guard let pb = pcb.makeBlitCommandEncoder() else { return 0 }
    pb.copy(from: hw, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: 256, height: vals.count / 256, depth: 1),
            to: theirs, destinationOffset: 0, destinationBytesPerRow: 256 * 4, destinationBytesPerImage: vals.count * 4)
    pb.endEncoding()
    pcb.commit()
    pcb.waitUntilCompleted()
    let mw = mine.contents().bindMemory(to: UInt32.self, capacity: vals.count), tw = theirs.contents().bindMemory(to: UInt32.self, capacity: vals.count)
    var packBad = 0
    for i in 0..<vals.count where mw[i] != tw[i] {
        if packBad == 0 { out[42] = Double(vals[i]); out[43] = Double(tw[i]); out[44] = Double(mw[i]) }
        packBad += 1
    }
    out[40] = Double(vals.count); out[41] = Double(packBad)
    return 1
}
