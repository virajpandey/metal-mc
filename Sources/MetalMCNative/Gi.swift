import Foundation
import Metal
import MetalMCCore
import simd

// Bounce light (global illumination) from a world-space irradiance cache (METALMC_EXP=gi, prototype: the kernels and
// an offline test, not yet wired into the frame; docs/gi-design.md has the design, the measurements and the hooks).
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

/// METALMC_EXP=gi: the world-space irradiance cache (prototype; not drawn yet).
let lodGi = experiments.contains("gi")
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

private let giShaderSource = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct GiParams {
    float4x4 invViewProj;  // request, resolve, debug view: inverse of the projection * view rotation the depth was drawn with
    int4 origin;           // xyz: the acceleration structure's origin (world blocks); w: frame number
    int4 camBlock;         // xyz: the camera's block (floor of its position); w: buckets - 1
    float4 camFrac;        // xyz: camera position minus camBlock; w: distance where level-0 cells end (blocks)
    float4 sun;            // xyz: direction to the sun; w: 1 with the sun up, 0 without
    float4 sunColor;       // rgb: sun irradiance on a surface facing it (linear); w: sun disk radius (radians)
    float4 skyZenith;      // rgb: sky radiance straight up; w: max sun ray length (blocks)
    float4 skyHorizon;     // rgb: sky radiance at the horizon; w: updates a cell's average holds at most
    float4 blockLight;     // rgb: irradiance at block light 15 (torch color); w: frames unseen before eviction
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

// Lookups scan the whole bucket: evictions leave holes, so an empty slot doesn't end the search. Four 16-byte loads.
static int giFind(const device uint* check, uint base, uint fp) {
    const device uint4* b = (const device uint4*)(check + base);
    for (uint i = 0; i < GI_BUCKET / 4u; i++) {
        bool4 m = b[i] == uint4(fp);
        if (any(m)) return int(base + i * 4u + (m.x ? 0u : (m.y ? 1u : (m.z ? 2u : 3u))));
    }
    return -1;
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

static float3 giSky(constant GiParams& p, float3 d) {
    // Below the horizon a ray has left the geometry (past the LOD's edge): dim ground.
    if (d.y < 0.0) return p.skyHorizon.rgb * 0.3;
    return mix(p.skyHorizon.rgb, p.skyZenith.rgb, sqrt(d.y));
}
// Irradiance from vanilla's block light level: its lightmap curve, f / (4 - 3f).
static float3 giBlockLight(constant GiParams& p, uint level) {
    float f = float(level) / 15.0;
    return p.blockLight.rgb * (f / (4.0 - 3.0 * f));
}
static float3 giSunOn(constant GiParams& p, uint face) {
    return p.sunColor.rgb * (p.sun.w * max(0.0, dot(kGiNormal[face], p.sun.xyz)));
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
                          device uint* meta, device uint* list, device atomic_uint* counters, float3 f, int3 air, uint face, uint level) {
    uint2 k = giKey(giCellOf(air, level), face, level);
    bool created;
    int e = giClaim(check, giBucketOf(p, k), giPrint(k), created);
    if (e < 0) { atomic_fetch_add_explicit(&counters[3], 1u, memory_order_relaxed); return 0u; }
    if (created) {
        giInit(p, check, keys, value, meta, uint(e), k, air, face, level, giRepPoint(p.camBlock.xyz, f, air, face, level));
        atomic_fetch_add_explicit(&counters[2], 1u, memory_order_relaxed);
    }
    // Seen this frame; the first thread to see it may list it. When more cells are visible than their part of the list
    // holds, a random share (sched.x, from last frame's count) is listed, a different one each frame: appending all of
    // them would fill the list with the same cells every frame (the first in dispatch order). Cells without samples
    // (new ones) are always listed.
    if (atomic_exchange_explicit(&stamp[e], uint(p.origin.w), memory_order_relaxed) == uint(p.origin.w)) return 0u;
    if (!giSampled(value[e]) || giMix(uint(e) ^ giMix(uint(p.origin.w))) <= p.sched.x) giEnqueueVisible(p, list, counters, uint(e));
    return 1u;
}

// One pixel of each sample x sample block (a different one each frame): find or create its surface's cell, mark it
// seen and list it for an update. Near a level boundary the next level's cell too, since the resolve blends the two.
// A cell 4 x 4 pixels or larger is touched every frame, so while the budget allows, visible cells update every frame.
kernel void gi_request(constant GiParams& p [[buffer(1)]],
                       device atomic_uint* check [[buffer(2)]],
                       device uint4* keys [[buffer(3)]],
                       device atomic_uint* stamp [[buffer(4)]],
                       device half4* value [[buffer(5)]],
                       device uint* meta [[buffer(6)]],
                       device uint* list [[buffer(7)]],
                       device atomic_uint* counters [[buffer(8)]],
                       depth2d<float, access::read> depth [[texture(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(p.sizes.z) || gid.y >= uint(p.sizes.w)) return;
    uint2 full = uint2(p.sizes.xy);
    uint2 q = min(gid * p.sample.x + p.sample.yz, full - 1);
    GiSurface s;
    uint seen = 0u;
    if (giSurfaceAt(p, depth, q, s) && length(s.rel) <= p.limits.x) {
        float lf = giLevelF(p, length(s.rel));
        uint level = min(uint(lf), GI_MAX_LEVEL);
        float3 f = p.camFrac.xyz + s.rel;
        int3 air = giAirBlock(p.camBlock.xyz, f, s.face);
        seen = giRequestCell(p, check, keys, stamp, value, meta, list, counters, f, air, s.face, level);
        if (lf - float(level) > 0.75 && level < GI_MAX_LEVEL) {
            seen += giRequestCell(p, check, keys, stamp, value, meta, list, counters, f, air, s.face, level + 1u);
        }
    }
    // The visible cells seen this frame (one atomic per SIMD group), for next frame's share.
    uint n = simd_sum(seen);
    if (simd_is_first() && n > 0u) atomic_fetch_add_explicit(&counters[18], n, memory_order_relaxed);
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

// The update's dispatch size from the two parts of the list (each capped at its share of the budget).
kernel void gi_args(constant GiParams& p [[buffer(1)]], device atomic_uint* counters [[buffer(8)]], uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint nv = min(atomic_load_explicit(&counters[0], memory_order_relaxed), giVisiblePart(p));
    uint n = nv + min(atomic_load_explicit(&counters[16], memory_order_relaxed), p.counts.x / 4u);
    atomic_store_explicit(&counters[1], n, memory_order_relaxed);
    atomic_store_explicit(&counters[17], nv, memory_order_relaxed);
    atomic_store_explicit(&counters[19], atomic_load_explicit(&counters[18], memory_order_relaxed), memory_order_relaxed);
    atomic_store_explicit(&counters[8], max((n + 63u) / 64u, 1u), memory_order_relaxed);
    atomic_store_explicit(&counters[9], 1u, memory_order_relaxed);
    atomic_store_explicit(&counters[10], 1u, memory_order_relaxed);
}

// Per triangle (primitive data): material | face << 8 | block light << 11 | water depth or sky cover << 15.
static uint giPrim(const device void* d) { return *(const device uint*)d; }

kernel void gi_update(instance_acceleration_structure accel [[buffer(0)]],
                      constant GiParams& p [[buffer(1)]],
                      device atomic_uint* check [[buffer(2)]],
                      device uint4* keys [[buffer(3)]],
                      device uint* stamp [[buffer(4)]],
                      device half4* value [[buffer(5)]],
                      device uint* meta [[buffer(6)]],
                      const device uint* list [[buffer(7)]],
                      device atomic_uint* counters [[buffer(8)]],
                      constant float4* mats [[buffer(9)]],
                      uint tid [[thread_position_in_grid]]) {
    if (tid >= atomic_load_explicit(&counters[1], memory_order_relaxed)) return;
    uint nv = atomic_load_explicit(&counters[17], memory_order_relaxed);
    uint e = list[tid < nv ? tid : giVisiblePart(p) + (tid - nv)];
    if (atomic_load_explicit(&check[e], memory_order_relaxed) == 0u) return;   // evicted since it was listed
    uint4 k = keys[e];
    uint face = (k.y >> 25) & 7u, level = k.y >> 28;
    if (face > 5u || level > GI_MAX_LEVEL) return;
    uint m = meta[e];
    uint frame = uint(p.origin.w);
    bool visible = frame - stamp[e] <= GI_RECENT;
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
    float3 sumE = 0.0;
    float sumSun = 0.0;
    uint valid = 0, sunSamples = 0, rays = 0, cached = 0, uncached = 0, spawned = 0, repUsed = 0;
    for (uint i = 0; i < p.sample.w; i++) {
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
            if (ph.type != intersection_type::none && ((giPrim(ph.primitive_data) >> 8) & 7u) == face) {
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
            float r = p.sunColor.w * sqrt(dk.x), ang = 2.0 * kGiPi * dk.y;
            ray sr(o, normalize(p.sun.xyz + sunT1 * (r * cos(ang)) + sunT2 * (r * sin(ang))), 0.0, p.skyZenith.w);
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
        if (bh.type == intersection_type::none) { sumE += kGiPi * giSky(p, dir); continue; }
        if (!bh.triangle_front_facing) continue;   // the back of a surface: the ray is inside terrain, no light
        uint pd = giPrim(bh.primitive_data);
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
        float3 sunH = giSunOn(p, hf);
        half4 hv = he >= 0 ? value[he] : half4(0.0h);
        if (giSampled(hv)) {
            eh = float3(hv.rgb) + (float(hv.a) - 1.0) * sunH;
            if (frame - stamp[he] > uint(p.blockLight.w) / 2u) stamp[he] = frame - GI_RECENT - 1u;
            cached++;
        } else {
            // Nothing cached there yet: its direct sun from one more ray, and half the open sky.
            float vis = 0.0;
            if (sunH.x + sunH.y + sunH.z > 0.0) {
                ray hs(hp + kGiNormal[hf] * offset, p.sun.xyz, 0.0, p.skyZenith.w);
                rays++;
                vis = shadow.intersect(hs, accel, 0xFF).type == intersection_type::none ? 1.0 : 0.0;
            }
            eh = vis * sunH + 0.5 * kGiPi * mix(p.skyHorizon.rgb, p.skyZenith.rgb, 0.6);
            uncached++;
        }
        eh += giBlockLight(p, hbl);
        sumE += mats[hm * 4u + giFaceClass(hf)].rgb * eh + mats[hm * 4u + 3u].rgb * p.look.y;
    }
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
    uint cnt = giSampled(old) ? (m & 255u) : 0u, cap = uint(p.skyHorizon.w);
    float alpha = 1.0 / float(min(cnt, cap) + 1u);
    float3 rgb = mix(float3(old.rgb), est, alpha);
    // The sun's share only changes when it was tested (a face turned away from it, or night, keeps the last one, or 0
    // if it never had one).
    float oldVis = giSampled(old) ? float(old.a) - 1.0 : 0.0;
    float vis = sunSamples > 0u ? mix(oldVis, sumSun / float(sunSamples), alpha) : oldVis;
    value[e] = half4(half3(rgb), half(1.0 + vis));
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
    float3 sum = 0.0;
    float ws = 0.0;
    for (int j = 0; j < 2; j++) {
        for (int i = 0; i < 2; i++) {
            float wt = (i == 0 ? 1.0 - w.x : w.x) * (j == 0 ? 1.0 - w.y : w.y);
            if (wt <= 0.0) continue;
            int3 cc = c;
            cc[t.x] += o.x + i;
            cc[t.y] += o.y + j;
            uint2 k = giKey(cc, face, level);
            int e = giFind(check, giBucketOf(p, k), giPrint(k));
            if (e < 0) continue;
            half4 v = value[e];
            if (!giSampled(v)) continue;
            sum += wt * float3(v.rgb);
            ws += wt;
        }
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

// gi_resolve's code word per half-resolution sample: the surface's plane (its camera-relative coordinate along the face's
// axis, as a half) | face << 16, and GI_CODE_DATA when the irradiance is there, GI_CODE_EMPTY when the surface's cells
// have no samples yet; 0 for the sky or past the cached range.
#define GI_CODE_DATA (1u << 19)
#define GI_CODE_EMPTY (1u << 20)

// Indirect irradiance (sky and bounced light) at half resolution: one sample per 2 x 2 pixels, at the block's nearest
// surface, into an RG11B10 texture, with a code word per sample (above) so the lighting pass can upsample it by face and
// plane without the depth buffer (giUpsample). At the panel's resolution a full-resolution resolve cost
// 1.4-1.6 ms, mostly its lookups (4 per pixel) and its output's bandwidth; a cell spans 10 pixels or more, so a quarter
// of the lookups loses nothing but the edges, which the upsample keeps.
kernel void gi_resolve(constant GiParams& p [[buffer(1)]],
                       const device uint* check [[buffer(2)]],
                       const device half4* value [[buffer(5)]],
                       depth2d<float, access::read> depth [[texture(0)]],
                       texture2d<float, access::write> out [[texture(1)]],
                       texture2d<uint, access::write> code [[texture(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    uint2 full = uint2(p.sizes.xy), base = gid * 2u;
    if (base.x >= full.x || base.y >= full.y) return;
    // The block's nearest surface (reverse-Z: the largest depth), so foreground edges get their own sample.
    uint best = 0u;
    float bd = -1.0;
    for (uint i = 0; i < 4u; i++) {
        float d = depth.read(min(base + uint2(i & 1u, i >> 1), full - 1));
        if (d > bd) { bd = d; best = i; }
    }
    uint2 q = min(base + uint2(best & 1u, best >> 1), full - 1);
    GiSurface s;
    if (bd <= 0.0 || !giSurfaceAt(p, depth, q, s) || length(s.rel) > p.limits.x) {
        out.write(float4(0.0), gid);
        code.write(uint4(0u), gid);
        return;
    }
    float3 f = p.camFrac.xyz + s.rel;
    float4 e = giIrradiance(p, check, value, f, giAirBlock(p.camBlock.xyz, f, s.face), s.face, giLevelF(p, length(s.rel)));
    uint pc = uint(as_type<ushort>(half(s.rel[s.face >> 1]))) | (s.face << 16);
    out.write(float4(e.w > 0.0 ? e.rgb : float3(0.0), 1.0), gid);
    code.write(uint4(pc | (e.w > 0.0 ? GI_CODE_DATA : GI_CODE_EMPTY)), gid);
}

// gi_resolve's half-resolution irradiance at full-resolution pixel `gid`, whose surface is on `face` at camera-relative
// `rel`: its own 2 x 2 block's sample when that is on the same face and plane (most pixels: two reads), else the nearest
// of the 8 around it that is (an edge, or a face too small for a sample of its own, like the riser of a one-block step:
// planes up to a block and a half apart count, the next step's light). The samples are bilinear across cells 10 or
// more pixels wide, so nearest-sample steps of 2 pixels don't show; a bilinear upsample (4 taps, or a gather of the codes
// and a filtered read) cost 0.5-0.9 ms at the panel's resolution, since so many pixels lie near a face's edge in this
// world that the taps rarely all match. w: 1 with data; 0 when nothing matched (the caller may take the cells itself,
// giIrradiance); -1 when the surface's cells have no samples yet (they'd find nothing: not worth the lookups, which just
// after a turn cost 1 ms).
static float4 giUpsample(texture2d<float> irr, texture2d<uint> code, uint2 gid, uint face, float3 rel) {
    int2 hsize = int2(irr.get_width(), irr.get_height());
    int2 c0 = min(int2(gid / 2u), hsize - 1);
    float plane = rel[face >> 1];
    float tol = max(0.25, 0.002 * length(rel));   // depth precision and the half's fall with distance
    uint c = code.read(uint2(c0)).r;
    uint cf = (c >> 16) & 7u;
    float cd = abs(float(as_type<half>(ushort(c & 0xFFFFu))) - plane);
    if (cf == face && (c & GI_CODE_DATA) != 0u && cd <= tol) return float4(irr.read(uint2(c0)).rgb, 1.0);
    bool empty = cf == face && (c & GI_CODE_EMPTY) != 0u && cd <= tol;
    // The 8 around, edge neighbors first, the nearest plane winning.
    const int2 around[8] = { int2(1, 0), int2(-1, 0), int2(0, 1), int2(0, -1), int2(1, 1), int2(-1, 1), int2(1, -1), int2(-1, -1) };
    int best = -1;
    float bestD = tol + 1.5;
    for (int k = 0; k < 8; k++) {
        uint2 hc = uint2(clamp(c0 + around[k], int2(0), hsize - 1));
        uint n = code.read(hc).r;
        if (((n >> 16) & 7u) != face) continue;
        float nd = abs(float(as_type<half>(ushort(n & 0xFFFFu))) - plane);
        if ((n & GI_CODE_DATA) == 0u) { empty = empty || ((n & GI_CODE_EMPTY) != 0u && nd <= tol); continue; }
        if (nd < bestD - 0.01) { bestD = nd; best = k; }
    }
    if (best >= 0) return float4(irr.read(uint2(clamp(c0 + around[best], int2(0), hsize - 1))).rgb, 1.0);
    return float4(0.0, 0.0, 0.0, empty ? -1.0 : 0.0);
}

static float3 giTonemap(float3 c) {
    c = saturate((c * (2.51 * c + 0.03)) / (c * (2.43 * c + 0.59) + 0.14));   // ACES fit (Narkowicz)
    return pow(c, float3(1.0 / 2.2));
}

// Debug view (limits.z): 0 lit (sun from the per-pixel shadow, sky and bounced light from the cache, block light),
// 1 the cache's irradiance alone (magenta: none), 2 cells (a color per cell, dimmed where it has no samples), 3 samples
// (red none, green converged, magenta: no cell), 4 without the cache (sun, vanilla's sky light as flat ambient, block
// light), 5 bounced and sky light only (albedo times the cache). `irr` and `code` are gi_resolve's output; `prim` holds
// each pixel's primitive data (material, face, block light, sky cover) and `sunVis` its sun visibility; in the game,
// 1 x 1 stand-ins give white and full sun.
kernel void gi_debug_view(constant GiParams& p [[buffer(1)]],
                          const device uint* check [[buffer(2)]],
                          const device half4* value [[buffer(5)]],
                          const device uint* meta [[buffer(6)]],
                          constant float4* mats [[buffer(9)]],
                          depth2d<float, access::read> depth [[texture(0)]],
                          texture2d<float> irr [[texture(1)]],
                          texture2d<uint, access::read> prim [[texture(2)]],
                          texture2d<half, access::read> sunVis [[texture(3)]],
                          texture2d<float, access::write> out [[texture(4)]],
                          texture2d<uint> code [[texture(5)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(p.sizes.x) || gid.y >= uint(p.sizes.y)) return;
    uint mode = uint(p.limits.z);
    GiSurface s;
    if (!giSurfaceAt(p, depth, gid, s)) {
        float3 d = normalize(giRelAt(p, gid, 1.0));
        out.write(float4(giTonemap(giSky(p, d) * kGiPi * p.look.x), 1.0), gid);
        return;
    }
    bool dummy = prim.get_width() == 1u;
    uint pd = dummy ? 0u : prim.read(gid).r;
    uint mat = pd & 255u, bl = (pd >> 11) & 15u, cover = (pd >> 15) & 15u;
    float3 albedo = dummy ? float3(0.6) : mats[mat * 4u + giFaceClass(s.face)].rgb;
    float3 emit = dummy ? float3(0.0) : mats[mat * 4u + 3u].rgb * p.look.z;
    float vis = sunVis.get_width() == 1u ? 1.0 : float(sunVis.read(gid).r);
    float3 sunI = giSunOn(p, s.face) * vis;
    float4 up = giUpsample(irr, code, gid, s.face, s.rel);
    if (up.w == 0.0 && length(s.rel) <= p.limits.x) {
        // No half-resolution sample on this pixel's face and plane (a face a pixel or two wide): its own lookups, which
        // only these few pixels pay for.
        float3 f = p.camFrac.xyz + s.rel;
        up = giIrradiance(p, check, value, f, giAirBlock(p.camBlock.xyz, f, s.face), s.face, giLevelF(p, length(s.rel)));
    }
    float3 ir = up.rgb;
    bool has = up.w > 0.0;
    float3 ambient = kGiPi * mix(p.skyHorizon.rgb, p.skyZenith.rgb, 0.6) * (s.face == 2u ? 1.0 : (s.face == 3u ? 0.3 : 0.6));
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
            float r = float(cnt) / p.skyHorizon.w;
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
    bool conv = live && float(meta[tid] & 255u) >= p.skyHorizon.w;
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
                            texture2d<half, access::write> sunVis [[texture(3)]],
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
    uint pd = giPrim(h.primitive_data);
    prim.write(uint4(pd), gid);
    uint face = min((pd >> 8) & 7u, 5u);
    float vis = 0.0;
    if (p.sun.w > 0.0 && dot(kGiNormal[face], p.sun.xyz) > 0.0) {
        intersector<instancing> sh;
        sh.accept_any_intersection(true);
        sh.assume_geometry_type(geometry_type::triangle);
        ray sr(cam + rel + kGiNormal[face] * 0.02, p.sun.xyz, 0.0, p.skyZenith.w);
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
                             texture2d<float> irr [[texture(1)]],
                             texture2d<uint, access::read> prim [[texture(2)]],
                             texture2d<half, access::write> out [[texture(4)]],
                             texture2d<uint> code [[texture(5)]],
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
        up = giUpsample(irr, code, gid, s.face, s.rel);
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
                         device uint4* out [[buffer(14)]],
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
    out[2 * tid] = uint4(hitA ? 1u : 0u, hitA ? (giPrim(ha.primitive_data) >> 8) & 7u : 9u, hitA && ha.triangle_front_facing ? 1u : 0u,
                         as_type<uint>(hitA ? ha.distance : -1.0));
    out[2 * tid + 1] = uint4(hitB ? 1u : 0u, hitB ? (giPrim(hb.primitive_data) >> 8) & 7u : 9u, 0u, as_type<uint>(hitB ? hb.distance : -1.0));
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

/// Mirrors GiParams in the shader (288 bytes).
struct GiParams {
    var invViewProj = matrix_identity_float4x4
    var origin = SIMD4<Int32>.zero
    var camBlock = SIMD4<Int32>.zero
    var camFrac = SIMD4<Float>.zero
    var sun = SIMD4<Float>.zero
    var sunColor = SIMD4<Float>.zero
    var skyZenith = SIMD4<Float>.zero
    var skyHorizon = SIMD4<Float>.zero
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
/// by world position); the prototype has one.
final class GiCache: @unchecked Sendable {
    let slots: Int
    let listCapacity: Int   // the most cells one frame can update
    let check: MTLBuffer, keys: MTLBuffer, stamp: MTLBuffer, value: MTLBuffer, meta: MTLBuffer, list: MTLBuffer
    let counters: MTLBuffer   // shared: the statistics are read on the CPU
    let mats: MTLBuffer
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

    init?(capacityLog2: Int, colors: [SIMD4<Float>] = lodColorTable()) {
        let dev = ctx.device
        guard dev.supportsRaytracing else { log("gi: no ray tracing on this device"); return nil }
        guard MemoryLayout<GiParams>.stride == 288 else { log("gi: GiParams is \(MemoryLayout<GiParams>.stride) bytes, the shader's 288"); return nil }
        slots = 1 << capacityLog2
        listCapacity = giBudget
        func buf(_ bytes: Int, _ mode: MTLResourceOptions = .storageModePrivate) -> MTLBuffer? { dev.makeBuffer(length: max(bytes, 16), options: mode) }
        let table = giMaterialTable(colors)
        guard let c = buf(slots * 4), let k = buf(slots * 16), let s = buf(slots * 4), let v = buf(slots * 8), let m = buf(slots * 4),
              let l = buf(listCapacity * 4), let n = buf(64 * 4, .storageModeShared),
              let t = dev.makeBuffer(bytes: table, length: table.count * 16, options: .storageModeShared) else { return nil }
        check = c; keys = k; stamp = s; value = v; meta = m; list = l; counters = n; mats = t
        do {
            let lib = try dev.makeLibrary(source: giShaderSource, options: nil)
            for name in ["gi_begin", "gi_request", "gi_schedule", "gi_args", "gi_update", "gi_resolve", "gi_debug_view", "gi_invalidate",
                         "gi_count", "gi_test_primary", "gi_test_rays", "gi_test_keys", "gi_test_upsample"] {
                guard let f = lib.makeFunction(name: name) else { log("gi: no function \(name)"); return nil }
                pipes[name] = try dev.makeComputePipelineState(function: f)
            }
        } catch {
            log("gi: shaders failed: \(error)")
            return nil
        }
        memset(counters.contents(), 0, counters.length)
        guard let cb = ctx.queue.makeCommandBuffer() else { return nil }
        clear(cb)
        cb.commit()
        cb.waitUntilCompleted()
        log("gi: cache of \(slots) cells, \((slots * giBytesPerCell) >> 20) MB")
    }

    func pipe(_ name: String) -> MTLComputePipelineState { pipes[name]! }

    /// Empties the table.
    func clear(_ cb: MTLCommandBuffer) {
        guard let b = cb.makeBlitCommandEncoder() else { return }
        for x in [check, keys, stamp, value, meta] { b.fill(buffer: x, range: 0..<x.length, value: 0) }
        b.endEncoding()
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
        p.sunColor = SIMD4(giSunIrradiance * sunUp, 0.0105)
        p.skyZenith = SIMD4(giSkyZenith, sunRayLength)
        p.skyHorizon = SIMD4(giSkyHorizon, history)
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
    }

    private func dispatch1D(_ enc: MTLComputeCommandEncoder, _ name: String, _ n: Int) {
        enc.setComputePipelineState(pipe(name))
        enc.dispatchThreads(MTLSize(width: max(n, 1), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
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

    /// Stage 1: request (and list) the cells the depth buffer shows.
    func encodeRequest(_ enc: MTLComputeCommandEncoder, depth: MTLTexture, params p: inout GiParams) {
        bindTable(enc)
        enc.setBytes(&p, length: MemoryLayout<GiParams>.stride, index: 1)
        enc.setComputePipelineState(pipe("gi_request"))
        enc.setTexture(depth, index: 0)
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
        enc.setBytes(&p, length: MemoryLayout<GiParams>.stride, index: 1)
        enc.setAccelerationStructure(accel, bufferIndex: 0)
        enc.useResources(accels, usage: .read)
        enc.setComputePipelineState(pipe("gi_update"))
        enc.dispatchThreadgroups(indirectBuffer: counters, indirectBufferOffset: 32, threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    }

    /// Stage 4: indirect irradiance at half resolution into `out` (rg11b10Float) and `code` (r32Uint), both half the depth
    /// buffer's size rounded up (gi_resolve has the formats, giUpsample the way back to full resolution).
    func encodeResolve(_ enc: MTLComputeCommandEncoder, depth: MTLTexture, out: MTLTexture, code: MTLTexture, params p: inout GiParams) {
        bindTable(enc)
        enc.setBytes(&p, length: MemoryLayout<GiParams>.stride, index: 1)
        enc.setComputePipelineState(pipe("gi_resolve"))
        enc.setTexture(depth, index: 0)
        enc.setTexture(out, index: 1)
        enc.setTexture(code, index: 5)
        enc.dispatchThreads(MTLSize(width: (depth.width + 1) / 2, height: (depth.height + 1) / 2, depth: 1),
                            threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    }

    /// The whole frame in one encoder, for the game (not wired yet; docs/gi-design.md, "Integration"): request, schedule,
    /// update and resolve, after the level is drawn and RtShadows has built its instance structure, into `out` and `code`.
    func encodeFrame(_ cb: MTLCommandBuffer, depth: MTLTexture, out: MTLTexture, code: MTLTexture, invViewProj: simd_float4x4, cam: SIMD3<Double>,
                     origin: SIMD3<Double>, accel: MTLAccelerationStructure, accels: [MTLAccelerationStructure], sunDir: SIMD3<Float>,
                     sunUp: Float) -> Bool {
        var p = params(invViewProj: invViewProj, cam: cam, origin: origin, sunDir: sunDir, sunUp: sunUp, width: depth.width, height: depth.height)
        guard let enc = cb.makeComputeCommandEncoder() else { return false }
        encodeBegin(enc, params: &p)
        encodeRequest(enc, depth: depth, params: &p)
        encodeSchedule(enc, params: &p)
        encodeUpdate(enc, accel: accel, accels: accels, params: &p)
        encodeResolve(enc, depth: depth, out: out, code: code, params: &p)
        enc.endEncoding()
        advance()
        return true
    }
}

/// A 4 x 4 visiting order that spreads consecutive frames' request samples apart (a Bayer order, as RtShadows').
let giRequestPattern: [SIMD2<UInt32>] = [0, 10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 14, 6, 12].map { SIMD2(UInt32($0 % 4), UInt32($0 / 4)) }

// MARK: - Acceleration structures with per-triangle data

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

/// Builds and compacts one primitive structure (with per-triangle data unless `prims` is nil). Blocks until done.
func giBuildBlas(verts: [Float], idx: [UInt32], prims: [UInt32]?, queue: MTLCommandQueue) -> MTLAccelerationStructure? {
    let dev = ctx.device
    guard !idx.isEmpty,
          let vb = dev.makeBuffer(bytes: verts, length: verts.count * 4, options: .storageModeShared),
          let ib = dev.makeBuffer(bytes: idx, length: idx.count * 4, options: .storageModeShared) else { return nil }
    let g = MTLAccelerationStructureTriangleGeometryDescriptor()
    g.vertexBuffer = vb
    g.vertexStride = 12
    g.vertexFormat = .float3
    g.indexBuffer = ib
    g.indexType = .uint32
    g.triangleCount = idx.count / 3
    g.opaque = true
    if let prims {
        guard let pb = dev.makeBuffer(bytes: prims, length: prims.count * 4, options: .storageModeShared) else { return nil }
        g.primitiveDataBuffer = pb
        g.primitiveDataStride = 4
        g.primitiveDataElementSize = 4
    }
    let d = MTLPrimitiveAccelerationStructureDescriptor()
    d.geometryDescriptors = [g]
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
    var withPrims = true
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
    var instances: [MTLAccelerationStructureInstanceDescriptor] = []
    for n in s.nodes {
        var starts = [0]
        for c in n.mesh.counts { starts.append(starts.last! + c) }
        n.mesh.quads.withUnsafeBufferPointer { q in
            for t in 0..<(lodTilesPerSide * lodTilesPerSide) {
                let (verts, idx, prims) = giTileGeometry(quads: q.baseAddress!, start: starts, tile: t, level: n.level)
                guard let b = giBuildBlas(verts: verts, idx: idx, prims: s.withPrims ? prims : nil, queue: queue) else { continue }
                var inst = MTLAccelerationStructureInstanceDescriptor()
                inst.transformationMatrix = MTLPackedFloat4x3(columns: (MTLPackedFloat3Make(1, 0, 0), MTLPackedFloat3Make(0, 1, 0), MTLPackedFloat3Make(0, 0, 1),
                                                                        MTLPackedFloat3Make(Float(n.x0 - s.origin.x), Float(lodWorldMinY - s.origin.y), Float(n.z0 - s.origin.z))))
                inst.options = .opaque
                inst.mask = 0xFF
                inst.accelerationStructureIndex = UInt32(s.blas.count)
                instances.append(inst)
                s.blas.append(b)
                s.triangles += idx.count / 3
                s.blasBytes += b.size
            }
        }
    }
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
    return true
}

/// Debug: builds the offline test's scene. kind 0: the synthetic scene (giSyntheticGrid) at the origin; kind 1: the save's
/// region file `path` (r.X.Z.mca) at level 0 (its four quarters) and the 8 regions around it at level 1, as the LOD
/// meshes them. flags bit 0: no per-triangle data (only to measure what it costs; the cache needs it).
/// out: triangles, structure bytes (compacted), build ms (meshing and structures), nodes. Returns 1 if it worked.
@_cdecl("mmc_debug_gi_scene")
public func mmc_debug_gi_scene(_ kind: Int32, _ path: UnsafePointer<CChar>?, _ flags: Int32, _ out: UnsafeMutablePointer<Double>) -> Int32 {
    let s = GiTestScene()
    s.withPrims = flags & 1 == 0
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
    if s.withPrims {
        s.cache = GiCache(capacityLog2: giCapacityLog2)
        if s.cache == nil { return 0 }
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

/// Debug: runs the cache for `frames` frames from a fixed camera (Minecraft yaw and pitch in degrees, 70 degree
/// vertical field of view) at width x height, then draws debug view `mode` (gi_debug_view) at `exposure` into `rgba`
/// (width x height x 4 bytes, top row first). The sun is vanilla's at `sunDeg` (0 noon, -60 mid-morning in the east). The
/// cache persists between calls (mmc_debug_gi_reset clears it).
/// stats: 0-4 mean GPU ms per frame of the primary rays (test only), request, schedule, update, resolve over the frames
/// after the first quarter; 5-8 their maxima (request, schedule, update, resolve); 9 live cells; 10 converged cells;
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
    let dev = ctx.device, queue = ctx.queue
    let w = Int(width), h = Int(height)
    // Camera: Minecraft's yaw 0 looks south (+z), 90 west (-x); positive pitch looks down. View space looks down -z.
    let yaw = yawDeg * .pi / 180, pitch = pitchDeg * .pi / 180
    let fwd = simd_normalize(SIMD3<Float>(-sin(yaw) * cos(pitch), -sin(pitch), cos(yaw) * cos(pitch)))
    let right = simd_normalize(simd_cross(fwd, SIMD3<Float>(0, 1, 0)))
    let up = simd_cross(right, fwd)
    let view = simd_float4x4(rows: [SIMD4(right, 0), SIMD4(up, 0), SIMD4(-fwd, 0), SIMD4(0, 0, 0, 1)])
    // Reverse-Z infinite projection like the game's (depth 1 at the near plane, 0 at infinity).
    let f = 1 / tan(Float(35) * .pi / 180), aspect = Float(w) / Float(h), near: Float = 0.05
    let proj = simd_float4x4(columns: (SIMD4(f / aspect, 0, 0, 0), SIMD4(0, f, 0, 0), SIMD4(0, 0, 0, -1), SIMD4(0, 0, near, 0)))
    var viewProj = proj * view
    let invViewProj = viewProj.inverse
    let sunA = sunDeg * .pi / 180
    let sunDir = SIMD3<Float>(-sin(sunA), cos(sunA), 0)
    let sunUp: Float = sunDir.y > 0.1 ? 1 : max(0, sunDir.y * 10)
    let cam = SIMD3<Double>(camX, camY, camZ), origin = SIMD3<Double>(Double(s.origin.x), Double(s.origin.y), Double(s.origin.z))
    let hw = (w + 1) / 2, hh = (h + 1) / 2
    func tex(_ format: MTLPixelFormat, _ usage: MTLTextureUsage, half: Bool = false) -> MTLTexture? {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: half ? hw : w, height: half ? hh : h, mipmapped: false)
        d.usage = usage
        d.storageMode = .private
        return dev.makeTexture(descriptor: d)
    }
    guard let depthBuf = dev.makeBuffer(length: w * h * 4, options: .storageModePrivate),
          let depth = tex(.depth32Float, [.shaderRead]), let prim = tex(.r32Uint, [.shaderRead, .shaderWrite]),
          let sunVis = tex(.r8Unorm, [.shaderRead, .shaderWrite]), let irr = tex(.rg11b10Float, [.shaderRead, .shaderWrite], half: true),
          let irrCode = tex(.r32Uint, [.shaderRead, .shaderWrite], half: true),
          let viewF = tex(.rgba32Float, [.shaderWrite]),
          let readback = dev.makeBuffer(length: hw * hh * 8, options: .storageModeShared) else { return 0 }
    func resolve(_ enc: MTLComputeCommandEncoder, _ p: inout GiParams) { cache.encodeResolve(enc, depth: depth, out: irr, code: irrCode, params: &p) }
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
    var p = cache.params(invViewProj: invViewProj, cam: cam, origin: origin, sunDir: sunDir, sunUp: sunUp, width: w, height: h)
    // The G-buffer (what the game's depth buffer and shadow pass would provide), once: the camera doesn't move.
    timed(0) { cb in
        guard let enc = cb.makeComputeCommandEncoder() else { return }
        enc.setComputePipelineState(cache.pipe("gi_test_primary"))
        enc.setAccelerationStructure(tlas, bufferIndex: 0)
        enc.useResources(s.blas, usage: .read)
        enc.setBytes(&p, length: MemoryLayout<GiParams>.stride, index: 1)
        enc.setBytes(&viewProj, length: 64, index: 11)
        enc.setBuffer(depthBuf, offset: 0, index: 12)
        enc.setTexture(prim, index: 2)
        enc.setTexture(sunVis, index: 3)
        enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
        guard let b = cb.makeBlitCommandEncoder() else { return }
        b.copy(from: depthBuf, sourceOffset: 0, sourceBytesPerRow: w * 4, sourceBytesPerImage: w * h * 4, sourceSize: MTLSize(width: w, height: h, depth: 1),
               to: depth, destinationSlice: 0, destinationLevel: 0, destinationOrigin: MTLOrigin(x: 0, y: 0, z: 0))
        b.endEncoding()
    }
    if repeats > 1, let cb = queue.makeCommandBuffer() {
        for _ in 0..<60 { if let enc = cb.makeComputeCommandEncoder() { resolve(enc, &p); enc.endEncoding() } }
        cb.commit()
        cb.waitUntilCompleted()
    }
    var prevResolve: [Float] = []
    var lastRays = 0.0, lastUpdated = 0.0
    let n = max(1, Int(frames))
    for fr in 0..<n {
        p = cache.params(invViewProj: invViewProj, cam: cam, origin: origin, sunDir: sunDir, sunUp: sunUp, width: w, height: h)
        let rays0 = cache.counters.contents().load(fromByteOffset: 44, as: UInt32.self)
        if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
            cache.encodeBegin(enc, params: &p)
            enc.endEncoding()
            cb.commit()
        }
        timed(1, repeats) { cb in if let enc = cb.makeComputeCommandEncoder() { cache.encodeRequest(enc, depth: depth, params: &p); enc.endEncoding() } }
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
            b.copy(from: irr, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: hw, height: hh, depth: 1),
                   to: readback, destinationOffset: 0, destinationBytesPerRow: hw * 4, destinationBytesPerImage: hw * hh * 4)
            b.copy(from: irrCode, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: hw, height: hh, depth: 1),
                   to: readback, destinationOffset: hw * hh * 4, destinationBytesPerRow: hw * 4, destinationBytesPerImage: hw * hh * 4)
            b.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            // rgb and 1 (data) or 0 per sample.
            let words = readback.contents().bindMemory(to: UInt32.self, capacity: hw * hh * 2)
            var cur = [Float](repeating: 0, count: hw * hh * 4)
            for i in 0..<(hw * hh) where words[hw * hh + i] & (1 << 19) != 0 {
                let v = words[i]
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
    if let lum = tex(.r16Float, [.shaderWrite]) {
        var ms = [0.0, 0.0, 0.0]
        for (i, variant) in [Float(0), 99, 97].enumerated() {
            var q = p
            q.limits.z = variant
            timed(0, repeats) { cb in
                guard let enc = cb.makeComputeCommandEncoder() else { return }
                enc.setComputePipelineState(cache.pipe("gi_test_upsample"))
                enc.setBytes(&q, length: MemoryLayout<GiParams>.stride, index: 1)
                enc.setBuffer(cache.check, offset: 0, index: 2)
                enc.setBuffer(cache.value, offset: 0, index: 5)
                enc.setBuffer(cache.counters, offset: 0, index: 8)
                enc.setTexture(depth, index: 0)
                enc.setTexture(irr, index: 1)
                enc.setTexture(prim, index: 2)
                enc.setTexture(lum, index: 4)
                enc.setTexture(irrCode, index: 5)
                enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
                enc.endEncoding()
            }
            ms[i] = times[0].last ?? 0
        }
        stats[24] = ms[0]; stats[25] = ms[0] - ms[1]; stats[29] = ms[2] - ms[1]
        // Once more, counting which way the pixels went.
        var q = p
        q.limits.z = 98
        let counts = cache.counters.contents().bindMemory(to: UInt32.self, capacity: 32)
        counts[20] = 0; counts[21] = 0; counts[22] = 0
        if let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() {
            enc.setComputePipelineState(cache.pipe("gi_test_upsample"))
            enc.setBytes(&q, length: MemoryLayout<GiParams>.stride, index: 1)
            enc.setBuffer(cache.check, offset: 0, index: 2)
            enc.setBuffer(cache.value, offset: 0, index: 5)
            enc.setBuffer(cache.counters, offset: 0, index: 8)
            enc.setTexture(depth, index: 0)
            enc.setTexture(irr, index: 1)
            enc.setTexture(prim, index: 2)
            enc.setTexture(lum, index: 4)
            enc.setTexture(irrCode, index: 5)
            enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
        }
        stats[26] = Double(counts[20]); stats[27] = Double(counts[21]); stats[28] = Double(counts[22])
    }
    // Statistics, the flicker measure, then the debug view.
    guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
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
    // The debug view.
    p.limits.z = Float(mode)
    p.look.x = exposure
    guard let vcb = queue.makeCommandBuffer(), let venc = vcb.makeComputeCommandEncoder() else { return 0 }
    venc.setComputePipelineState(cache.pipe("gi_debug_view"))
    venc.setBytes(&p, length: MemoryLayout<GiParams>.stride, index: 1)
    venc.setBuffer(cache.check, offset: 0, index: 2)
    venc.setBuffer(cache.value, offset: 0, index: 5)
    venc.setBuffer(cache.meta, offset: 0, index: 6)
    venc.setBuffer(cache.mats, offset: 0, index: 9)
    venc.setTexture(depth, index: 0)
    venc.setTexture(irr, index: 1)
    venc.setTexture(prim, index: 2)
    venc.setTexture(sunVis, index: 3)
    venc.setTexture(viewF, index: 4)
    venc.setTexture(irrCode, index: 5)
    venc.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
    venc.endEncoding()
    vcb.commit()
    vcb.waitUntilCompleted()
    stats[21] = (vcb.gpuEndTime - vcb.gpuStartTime) * 1000
    // Float view into bytes, flipping rows: texture row 0 is the bottom of the image (the game's convention).
    guard let rcb = queue.makeCommandBuffer(), let rb = rcb.makeBlitCommandEncoder(),
          let fbuf = dev.makeBuffer(length: w * h * 16, options: .storageModeShared) else { return 0 }
    rb.copy(from: viewF, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0), sourceSize: MTLSize(width: w, height: h, depth: 1),
            to: fbuf, destinationOffset: 0, destinationBytesPerRow: w * 16, destinationBytesPerImage: w * h * 16)
    rb.endEncoding()
    rcb.commit()
    rcb.waitUntilCompleted()
    let px = fbuf.contents().bindMemory(to: Float.self, capacity: w * h * 4)
    for y in 0..<h {
        let src = (h - 1 - y) * w * 4, dst = y * w * 4
        for i in 0..<(w * 4) { rgba[dst + i] = UInt8(max(0, min(255, px[src + i] * 255 + 0.5))) }
    }
    return 1
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
    return 1
}
