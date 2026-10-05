import Foundation
import Metal
import MetalMCCore
import simd

// Far-terrain LOD drawn inside Minecraft's main world pass, after solid terrain (Java hook:
// metalmc.lod.mixin.LevelRendererLodMixin). It shares the game's projection (with the far plane pushed
// out) and reverse-Z depth, so it depth-tests correctly against vanilla. It reproduces vanilla's fog
// exactly, so it fades into the same sky and fog colors.

let lodShaderSource = """
#include <metal_stdlib>
using namespace metal;

// Lit mode (METALMC_EXP=lit, Lit.swift): the opaque quads also write the terrain G-buffer.
#define LIT_MODE \(litEnabled ? 1 : 0)
\(litShaderHeader)

struct LodUniforms {
    float4x4 proj;
    float4x4 view;
    float4 fogColor;
    float envStart, envEnd, rdStart, rdEnd;
    float discardRadius;   // horizontal distance inside which vanilla chunks are drawn instead
    float sky;             // daylight factor 0..1
    float alpha;           // output alpha: 1 for opaque quads, vanilla's water texture alpha for water
    float lightmapOn;      // 1 if vanilla's lightmap is bound (texture 29), else the fallback daylight curve
    float4 camFrac;        // xyz: fractional part of the camera position; w: 1 if texture detail is on
    float4 camInSection;   // xyz: camera position within its chunk section (0-16); w: seam bitmap half-size H
    int4 seamInfo;         // x: camera section y - world bottom section; y: bitmap width W (2H + 1);
                           // z: sky light at open surfaces (15, or 0 in a dimension without sky light)
};
// Per base material: top and side sprite rectangles in the block atlas (u0, v0, u1, v1), and the mean luma
// of each texture (x = top, y = side).
struct LodSpriteGPU { float4 top; float4 side; float4 luma; };
// Per draw: section origin relative to the camera (xyz) and voxel size in blocks (w).
struct Xform { float4 offsetScale; };
struct VOut {
    float4 pos [[position]];
    uint color [[flat]];     // shaded material color (RGB8): constant over a quad, since ambient occlusion is per pixel
    float3 rel;
    float2 quv;              // position within the quad in voxels, along its (u, v) axes
    uint color2 [[flat]];    // full-resolution grass sides: the biome-tinted grass color, shaded like `color` (RGB8)
    uint matFace [[flat]];   // base material | face << 8 | (w - 1) << 11 | (h - 1) << 17 | grass side << 23 | level 0 << 24 | deep water << 25
    uint ao [[flat]];        // first rim ambient-occlusion value of the quad, or 0xFFFFFFFF for none
};

// Unit-cube corners per face, counter-clockwise seen from outside. Face order: +X -X +Y -Y +Z -Z.
constant float3 kCorners[6][4] = {
    { float3(1,0,0), float3(1,1,0), float3(1,1,1), float3(1,0,1) },
    { float3(0,0,1), float3(0,1,1), float3(0,1,0), float3(0,0,0) },
    { float3(0,1,0), float3(0,1,1), float3(1,1,1), float3(1,1,0) },
    { float3(0,0,0), float3(1,0,0), float3(1,0,1), float3(0,0,1) },
    { float3(1,0,1), float3(1,1,1), float3(0,1,1), float3(0,0,1) },
    { float3(0,0,0), float3(0,1,0), float3(1,1,0), float3(1,0,0) },
};
// Face normals, same order.
constant float3 kNormal[6] = { float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0), float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1) };
// Directional shading close to vanilla's face shading (top 1.0, bottom 0.5, X 0.6, Z 0.8).
constant float kShade[6] = { 0.6, 0.6, 1.0, 0.5, 0.8, 0.8 };

static float3 extentScale(uint face, float w, float h) {
    if (face < 2) return float3(1, w, h);
    if (face < 4) return float3(w, 1, h);
    return float3(w, h, 1);
}

// Quads are 8 bytes: word0 = x | z << 8 | y << 16 (9 bits) | face << 25 | depth << 28 (voxel coordinates within
// the node; depth: blocks of water above an underwater face, or 15 minus the sky light of covered air in front of
// the face; 0 for none), word1 = material | (w - 1) << 8 | (h - 1) << 16 | block light << 24 (greedy extents
// along the face's u and v axes; block light of the air in front, level 0). Ambient occlusion is per pixel from
// the quad's rim values (see lodAO).
// Brightness for 0, 1, 2 and 3 occluders: vanilla's smooth-lighting steps (1, 0.8, 0.6, 0.4) at 60% strength,
// because an LOD voxel's corner gradient spans 2+ blocks where vanilla's spans one (measured with the
// fidelity score: full strength left the LOD 3.5 levels too dark, none 3.5 too bright).
constant float kAO[4] = { 1.0, 0.88, 0.76, 0.64 };
// At full resolution a voxel is a block, so vanilla's steps apply as they are.
constant float kAO0[4] = { 1.0, 0.8, 0.6, 0.4 };

// Rim point (a, b) of a w x h quad to its index: counterclockwise in (u, v) from (0, 0), as LodBuild.mesh stores them.
static uint rimIndex(uint a, uint b, uint w, uint h) {
    if (b == 0) return a;
    if (a == w) return w + b;
    if (b == h) return 2 * w + h - a;
    return 2 * w + 2 * h - b;
}
static float rimAO(const device uint* bits, uint base, uint a, uint b, uint w, uint h, bool full) {
    if (a != 0 && b != 0 && a != w && b != h) return 1.0;   // inside the quad: nothing in front occludes
    uint i = base + rimIndex(a, b, w, h);
    uint level = (bits[i >> 4] >> ((i & 15) * 2)) & 3;
    return full ? kAO0[level] : kAO[level];
}
// Ambient occlusion like vanilla's smooth lighting, per pixel: bilinear between the corners of the voxel face
// under the pixel. Only faces touching the quad's rim can have occluded corners.
static float lodAO(VOut in, const device uint* bits) {
    if (in.ao == 0xFFFFFFFFu) return 1.0;
    uint w = ((in.matFace >> 11) & 63) + 1, h = ((in.matFace >> 17) & 63) + 1;
    uint i = uint(clamp(floor(in.quv.x), 0.0, float(w - 1))), j = uint(clamp(floor(in.quv.y), 0.0, float(h - 1)));
    if (i > 0 && j > 0 && i + 1 < w && j + 1 < h) return 1.0;
    float2 f = clamp(in.quv - float2(i, j), 0.0, 1.0);
    bool full = (in.matFace >> 24) & 1;
    float a00 = rimAO(bits, in.ao, i, j, w, h, full), a10 = rimAO(bits, in.ao, i + 1, j, w, h, full);
    float a01 = rimAO(bits, in.ao, i, j + 1, w, h, full), a11 = rimAO(bits, in.ao, i + 1, j + 1, w, h, full);
    return mix(mix(a00, a10, f.x), mix(a01, a11, f.x), f.y);
}
// Skylight under d blocks of water (it loses a level per block): vanilla's lightmap brightness for sky light
// 15 - d, relative to 15 (overworld ambient 0.04, default brightness setting).
constant float kWaterLight[16] = { 1.0, 0.908, 0.822, 0.747, 0.676, 0.609, 0.544, 0.482,
                                   0.423, 0.367, 0.315, 0.265, 0.218, 0.174, 0.133, 0.094 };
// Vanilla's water surface is 1/9 block below the top of the highest water block. A level-0 voxel is that block;
// coarser voxels end on the grid, and at sea level (y 62) the block's top is 1 block below a voxel boundary.
constant float kWaterSurfaceDrop0 = 1.0 / 9.0, kWaterSurfaceDrop = 10.0 / 9.0;
// Light like vanilla's terrain: its lightmap (16 x 16, block light along x, sky light along y) at no block light
// and full sky light, or 15 - depth under water (sky light loses a level per block of water). That follows the
// day, dusk and night, night vision and the darkness effect exactly as vanilla chunks do.
static float3 lodLight(constant LodUniforms& u, texture2d<float> lightmap, sampler s, float skyLevel, uint depth,
                       float blockLevel = 0.0) {
    if (u.lightmapOn > 0.5) return lightmap.sample(s, float2((blockLevel + 0.5) / 16.0, (skyLevel + 0.5) / 16.0), level(0)).rgb;
    return float3(blockLevel > 0.0 ? 1.0 : kWaterLight[depth] * mix(0.2, 1.0, u.sky));
}

vertex VOut lod_vs(uint vid [[vertex_id]], uint draw [[base_instance]],
                   const device uint2* quads [[buffer(18)]],
                   constant LodUniforms& u [[buffer(19)]],
                   const device Xform* xforms [[buffer(20)]],
                   constant float4* colors [[buffer(21)]],
                   const device uint* aoOffsets [[buffer(22)]],
                   texture2d<float> lightmap [[texture(29)]], sampler lightSampler [[sampler(14)]]) {
    uint2 q = quads[vid >> 2];
    uint corner = vid & 3;
    uint face = (q.x >> 25) & 7;
    float3 local = float3(q.x & 255, (q.x >> 16) & 511, (q.x >> 8) & 255);
    float w = float(((q.y >> 8) & 255) + 1), h = float(((q.y >> 16) & 255) + 1);
    float3 ext = extentScale(face, w, h);
    float4 xs = xforms[draw].offsetScale;
    uint m = q.y & 255;
    bool water = (m >= 128u && m < 160u) || m == MAT_WATER;
    float3 rel = xs.xyz + (local + kCorners[face][corner] * ext) * xs.w;
    if (water && kCorners[face][corner].y > 0.5) rel.y -= xs.w < 1.5 ? kWaterSurfaceDrop0 : kWaterSurfaceDrop;
    // Drop whole quads inside the range vanilla draws: decide by the quad's center so all four corners
    // agree, and collapse the quad to a point (no fragment discard, which would disable hidden-surface removal).
    float3 center = xs.xyz + (local + 0.5 * (kCorners[face][0] + kCorners[face][2]) * ext) * xs.w;
    VOut o;
    if (length(center.xz) < u.discardRadius || (u.seamInfo.w != 0 && ((vid >> 2) & 3) != 0)) {
        o.pos = float4(0.0, 0.0, 0.0, 1.0);
        o.color = 0;
        o.rel = float3(0.0);
        o.quv = float2(0.0);
        o.color2 = 0;
        o.matFace = 0;
        o.ao = 0xFFFFFFFFu;
        return o;
    }
    float4 clip = u.proj * (u.view * float4(rel, 1.0));
    clip.y = -clip.y;   // same vertical flip as every translated Minecraft shader (flip_vert_y)
    o.pos = clip;
    uint faceClass = face == 2 ? 0u : (face == 3 ? 2u : 1u);   // top, side, bottom
    // An underwater face's depth darkens it; on water, depth 15 marks deep water (drawn opaque, no floor meshed).
    uint depthField = (q.x >> 28) & 15;
    bool deep = water && depthField == 15;
    uint depth = water ? 0u : depthField;
    // Block light: lava and solid light sources glow at night like vanilla's; other faces get the light of the
    // air in front of them (torches, lanterns).
    float blockLevel = (m == MAT_LAVA || m == MAT_MAGMA || (m >= MAT_GLOW_FIRST && m <= MAT_GLOW_LAST)) ? 15.0 : float((q.y >> 24) & 15u);
    float3 k = kShade[face] * lodLight(u, lightmap, lightSampler, max(0.0, float(u.seamInfo.z) - float(depth)), depth, blockLevel);
    // A full-resolution grass side is one grass block: dirt with vanilla's tinted fringe on top (lodShade), not
    // the average of the two that coarser voxels use (the fringe would repeat on every block of a 2-block side).
    bool grassSide = xs.w < 1.5 && faceClass == 1u && u.camFrac.w > 0.5 && ((m >= 64u && m < 96u) || m == MAT_GRASS);
#if LIT_MODE
    // Lit mode: the colors go out unlit, and the alpha byte carries the depth field and block light (4 bits each), from
    // which the fragment shader works k out again (lodLitK): the G-buffer gets the albedo without it.
    (void)k;
    o.color = pack_float_to_unorm4x8(float4(grassSide ? colors[MAT_DIRT * 3 + 1].rgb : colors[m * 3 + faceClass].rgb,
                                            float(depth | (uint(blockLevel) << 4)) / 255.0));
    o.color2 = grassSide ? pack_float_to_unorm4x8(float4(colors[m * 3].rgb, 0.0))
             : (deep ? pack_float_to_unorm4x8(float4(float3(0.56, 0.53, 0.45) * lodLight(u, lightmap, lightSampler, 2.0, 13), 0.0)) : 0u);
#else
    // Varyings cost vertex output bandwidth on this tile-based GPU (a float3 color2 cost 9% of the frame at a
    // million quads), so the colors are flat and packed.
    o.color = pack_float_to_unorm4x8(float4((grassSide ? colors[MAT_DIRT * 3 + 1].rgb : colors[m * 3 + faceClass].rgb) * k, 0.0));
    // Flat: per-pixel AO is applied in the fragment shader and the rest of k is constant over the quad.
    // Deep water carries the light of its (unmeshed) floor, a dozen or more blocks down, for its blend.
    o.color2 = grassSide ? pack_float_to_unorm4x8(float4(colors[m * 3].rgb * k, 0.0))
             : (deep ? pack_float_to_unorm4x8(float4(float3(0.56, 0.53, 0.45) * lodLight(u, lightmap, lightSampler, 2.0, 13), 0.0)) : 0u);
#endif
    o.rel = rel;
    float3 c = kCorners[face][corner];
    o.quv = (face < 2 ? c.yz : (face < 4 ? c.xz : c.xy)) * float2(w, h);
    // Biome-tinted variants (64 + t grass, 96 + t leaves, 128 + t water) use their base material's texture.
    uint baseMat = (m >= 128u && m < 160u) ? MAT_WATER : ((m >= 96u && m < 128u) ? MAT_LEAVES : ((m >= 64u && m < 96u) ? MAT_GRASS : m));
    o.matFace = baseMat | (face << 8) | (((q.y >> 8) & 63) << 11) | (((q.y >> 16) & 63) << 17) | (grassSide ? 1u << 23 : 0u) | (xs.w < 1.5 ? 1u << 24 : 0u) | (deep ? 1u << 25 : 0u);
    o.ao = aoOffsets[vid >> 2];
    return o;
}

// Quad culling (METALMC_EXP=quadcull): quads that can't make a single fragment aren't issued to the vertex stage, whose
// cost on this GPU is per vertex invoked, culled or not (docs/lod-design.md, "What didn't pay"). A quad makes no fragment
// if no pixel center lies in its bounding box on screen (off screen, or thinner than the gap between pixel centers), or if
// both its triangles face away from the camera. The test is conservative: each projected corner may be off by `margin`
// pixels (the rasterizer snaps vertices to a sub-pixel grid, and lod_vs rounds a little differently from this), more for
// corners far outside the view, and quads with a corner near or behind the camera's plane are kept.
struct LodCullParams {
    float4x4 viewProj;   // proj * view, as lod_vs applies them (before its vertical flip, which changes nothing here)
    float2 viewport;     // the pass's size in pixels: its viewport is the whole target
    float margin;        // pixels; negative keeps every quad (measuring the culling pass and the indirect draws alone)
    float minW;          // a corner with clip w under this keeps the quad (the near plane is at 0.05)
    uint jobs;           // lod_cull's job count
    uint pad[3];
};
static float lodL1(float2 v) { return abs(v.x) + abs(v.y); }
static bool lodQuadCulled(uint2 q, float4 xs, constant LodCullParams& cp) {
    uint face = (q.x >> 25) & 7;
    if (face > 5 || cp.margin < 0.0) return false;
    float3 local = float3(q.x & 255, (q.x >> 16) & 511, (q.x >> 8) & 255);
    float w = float(((q.y >> 8) & 255) + 1), h = float(((q.y >> 16) & 255) + 1);
    float3 ext = extentScale(face, w, h) * xs.w;   // blocks along x, y and z (a voxel along the normal)
    uint m = q.y & 255;
    bool water = (m >= 128u && m < 160u) || m == MAT_WATER;
    float drop = water ? (xs.w < 1.5 ? kWaterSurfaceDrop0 : kWaterSurfaceDrop) : 0.0;
    // Corner c is base + kCorners[face][c] * ext, its y lowered by `drop` where kCorners' y is 1 (lod_vs's water tops),
    // so its clip position is base's plus the matrix's columns scaled by the extents, those kCorners has.
    float4 cb = cp.viewProj * float4(xs.xyz + local * xs.w, 1.0);
    float4 cx = cp.viewProj[0] * ext.x, cy = cp.viewProj[1] * (ext.y - drop), cz = cp.viewProj[2] * ext.z;
    float2 hv = 0.5 * cp.viewport;
    float2 s[4];
    for (uint c = 0; c < 4; c++) {
        float3 k = kCorners[face][c];
        float4 clip = cb + k.x * cx + k.y * cy + k.z * cz;
        if (!(clip.w > cp.minW)) return false;
        s[c] = clip.xy / clip.w * hv;   // pixels from the viewport's center
    }
    float2 lo = min(min(s[0], s[1]), min(s[2], s[3])), hi = max(max(s[0], s[1]), max(s[2], s[3]));
    float e = cp.margin + 1e-5 * max(max(abs(lo.x), abs(hi.x)), max(abs(lo.y), abs(hi.y)));
    // Pixel centers lie at k + 0.5 from the viewport's corner, k = 0 ..< size (with the vertical flip too: the size is
    // whole). The rasterizer only samples there.
    float2 k0 = max(ceil(lo - e + hv - 0.5), float2(0.0)), k1 = min(floor(hi + e + hv - 0.5), cp.viewport - 1.0);
    if (k0.x > k1.x || k0.y > k1.y) return true;
    // Facing away: front faces wind counterclockwise here (y up), so both triangles (corners 0 1 2 and 0 2 3) have
    // negative doubled area, by more than moving each corner up to e can change it (2e(|a|1 + |b|1) + 8e^2 for
    // cross(a, b)) or rounding the products can.
    float2 a = s[1] - s[0], b = s[2] - s[0], d = s[3] - s[0];
    float a1 = a.x * b.y - a.y * b.x, a2 = b.x * d.y - b.y * d.x;
    float m1 = 2.0 * e * (lodL1(a) + lodL1(b)) + 8.0 * e * e + 1e-6 * (abs(a.x * b.y) + abs(a.y * b.x));
    float m2 = 2.0 * e * (lodL1(b) + lodL1(d)) + 8.0 * e * e + 1e-6 * (abs(b.x * d.y) + abs(b.y * d.x));
    return a1 < -m1 && a2 < -m2;
}

// The culling pass: per job (a run of at most lodCullJobQuads consecutive quads of one draw), its surviving quads and
// their AO offsets, in their order, to [outBase, outBase + survivors) of the frame's lists, and the indexed indirect draw
// that draws them with lod_vs unchanged: the same quads and draw slot, so the same vertices. Order matters (water
// blends, and equal depths go to the later quad). Copies rather than indices, so lod_vs stays as it is: an indexed
// variant drew the same picture as itself but not quite lod_vs's (the compiler rounds the shared code differently), and
// what it saved in the culling pass it lost in the vertex stage (measured).
struct LodCullJob { const device uint2* quads; const device uint* aoOffsets; uint count; uint slot; uint outBase; uint pad; };
struct LodDrawArgs { uint indexCount; uint instanceCount; uint indexStart; int baseVertex; uint baseInstance; };
// One SIMD group per job, no threadgroup memory or barriers: each step takes 4 runs of 32 quads (a quad per lane in
// each, their loads in flight together) and writes each run's survivors after the last's. The pass runs at memory speed:
// it reads 8 bytes a quad and 4 and writes 12 per survivor.
kernel void lod_cull(const device LodCullJob* jobs [[buffer(0)]], constant LodCullParams& cp [[buffer(1)]],
                     const device Xform* xforms [[buffer(2)]], device uint2* outQuads [[buffer(3)]],
                     device uint* outAo [[buffer(4)]], device LodDrawArgs* args [[buffer(5)]],
                     uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                     uint groups [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint job = tg * groups + sg;
    if (job >= cp.jobs) return;   // the whole SIMD group
    LodCullJob j = jobs[job];
    float4 xs = xforms[j.slot].offsetScale;
    uint total = 0;
    for (uint base = 0; base < j.count; base += 128) {
        uint2 q[4];
        for (uint r = 0; r < 4; r++) {
            uint i = base + 32 * r + lane;
            q[r] = i < j.count ? j.quads[i] : uint2(0);
        }
        for (uint r = 0; r < 4; r++) {
            uint i = base + 32 * r + lane;
            bool keep = i < j.count && !lodQuadCulled(q[r], xs, cp);
            uint rank = simd_prefix_exclusive_sum(uint(keep));   // every lane active
            if (keep) {
                outQuads[j.outBase + total + rank] = q[r];
                outAo[j.outBase + total + rank] = j.aoOffsets[i];
            }
            total += simd_sum(uint(keep));
        }
    }
    if (lane == 0) {
        LodDrawArgs a;
        a.indexCount = 6 * total;
        a.instanceCount = 1;
        a.indexStart = 0;
        a.baseVertex = int(4 * j.outBase);
        a.baseInstance = j.slot;
        args[job] = a;
    }
}

// Mesh-shader path (METALMC_EXP=meshshader): one thread per quad, 64 quads per threadgroup. The per-quad work
// (decode, color, lighting, AO lookup) runs once instead of in each of the quad's four vertices, and quads
// facing away from the camera or entirely off one side of the view are culled before rasterization.
struct MeshDraw { uint first; uint count; uint slot; uint pad; };
struct LodPrim { bool culled [[primitive_culled]]; };
using LodMeshOut = metal::mesh<VOut, LodPrim, 4 * MESH_QUADS, 2 * MESH_QUADS, metal::topology::triangle>;

[[mesh]] void lod_mesh(LodMeshOut out,
                       uint tid [[thread_index_in_threadgroup]], uint gid [[threadgroup_position_in_grid]],
                       constant MeshDraw& md [[buffer(17)]],
                       const device uint2* quads [[buffer(18)]],
                       constant LodUniforms& u [[buffer(19)]],
                       const device Xform* xforms [[buffer(20)]],
                       constant float4* colors [[buffer(21)]],
                       const device uint* aoOffsets [[buffer(22)]],
                       texture2d<float> lightmap [[texture(29)]], sampler lightSampler [[sampler(14)]]) {
    uint base = gid * MESH_QUADS;
    uint n = min(uint(MESH_QUADS), md.count - base);
    if (tid == 0) out.set_primitive_count(n * 2);
    if (tid >= n) return;
    uint qi = md.first + base + tid;
    uint2 q = quads[qi];
    uint face = (q.x >> 25) & 7;
    float3 local = float3(q.x & 255, (q.x >> 16) & 511, (q.x >> 8) & 255);
    float w = float(((q.y >> 8) & 255) + 1), h = float(((q.y >> 16) & 255) + 1);
    float3 ext = extentScale(face, w, h);
    float4 xs = xforms[md.slot].offsetScale;
    uint m = q.y & 255;
    bool water = (m >= 128u && m < 160u) || m == MAT_WATER;
    float drop = xs.w < 1.5 ? kWaterSurfaceDrop0 : kWaterSurfaceDrop;
    uint faceClass = face == 2 ? 0u : (face == 3 ? 2u : 1u);
    uint depthField = (q.x >> 28) & 15;
    bool deep = water && depthField == 15;
    uint depth = water ? 0u : depthField;
    float blockLevel = (m == MAT_LAVA || m == MAT_MAGMA || (m >= MAT_GLOW_FIRST && m <= MAT_GLOW_LAST)) ? 15.0 : float((q.y >> 24) & 15u);
    float3 k = kShade[face] * lodLight(u, lightmap, lightSampler, max(0.0, float(u.seamInfo.z) - float(depth)), depth, blockLevel);
    bool grassSide = xs.w < 1.5 && faceClass == 1u && u.camFrac.w > 0.5 && ((m >= 64u && m < 96u) || m == MAT_GRASS);
#if LIT_MODE
    // Lit mode: unlit colors, the depth field and block light in the alpha byte (as lod_vs).
    (void)k;
    uint color = pack_float_to_unorm4x8(float4(grassSide ? colors[MAT_DIRT * 3 + 1].rgb : colors[m * 3 + faceClass].rgb,
                                               float(depth | (uint(blockLevel) << 4)) / 255.0));
    uint color2 = grassSide ? pack_float_to_unorm4x8(float4(colors[m * 3].rgb, 0.0))
                : (deep ? pack_float_to_unorm4x8(float4(float3(0.56, 0.53, 0.45) * lodLight(u, lightmap, lightSampler, 2.0, 13), 0.0)) : 0u);
#else
    uint color = pack_float_to_unorm4x8(float4((grassSide ? colors[MAT_DIRT * 3 + 1].rgb : colors[m * 3 + faceClass].rgb) * k, 0.0));
    uint color2 = grassSide ? pack_float_to_unorm4x8(float4(colors[m * 3].rgb * k, 0.0))
                : (deep ? pack_float_to_unorm4x8(float4(float3(0.56, 0.53, 0.45) * lodLight(u, lightmap, lightSampler, 2.0, 13), 0.0)) : 0u);
#endif
    uint baseMat = (m >= 128u && m < 160u) ? MAT_WATER : ((m >= 96u && m < 128u) ? MAT_LEAVES : ((m >= 64u && m < 96u) ? MAT_GRASS : m));
    uint matFace = baseMat | (face << 8) | (((q.y >> 8) & 63) << 11) | (((q.y >> 16) & 63) << 17) | (grassSide ? 1u << 23 : 0u)
                 | (xs.w < 1.5 ? 1u << 24 : 0u) | (deep ? 1u << 25 : 0u);
    uint ao = aoOffsets[qi];
    // Culled: facing away (the camera is at the origin of these camera-relative coordinates, so it's in front
    // if it's on the normal's side of the quad's plane) or all four corners beyond the same clip plane. Culled
    // quads write no vertices.
    float3 rels[4];
    float4 clips[4];
    uint outL = 0, outR = 0, outB = 0, outT = 0;
    for (uint c = 0; c < 4; c++) {
        float3 cc = kCorners[face][c];
        float3 rel = xs.xyz + (local + cc * ext) * xs.w;
        if (water && cc.y > 0.5) rel.y -= drop;
        float4 clip = u.proj * (u.view * float4(rel, 1.0));
        clip.y = -clip.y;
        outL += clip.x < -clip.w; outR += clip.x > clip.w; outB += clip.y < -clip.w; outT += clip.y > clip.w;
        rels[c] = rel;
        clips[c] = clip;
    }
    bool culled = dot(kNormal[face], rels[0]) >= 0.0 || outL == 4 || outR == 4 || outB == 4 || outT == 4 || (u.seamInfo.w != 0 && (qi & 3) != 0);
    LodPrim p;
    p.culled = culled;
    out.set_primitive(tid * 2, p);
    out.set_primitive(tid * 2 + 1, p);
    if (culled) return;
    for (uint c = 0; c < 4; c++) {
        float3 cc = kCorners[face][c];
        float3 rel = rels[c];
        float4 clip = clips[c];
        VOut o;
        o.pos = clip;
        o.color = color;
        o.color2 = color2;
        o.rel = rel;
        o.quv = (face < 2 ? cc.yz : (face < 4 ? cc.xz : cc.xy)) * float2(w, h);
        o.matFace = matFace;
        o.ao = ao;
        out.set_vertex(tid * 4 + c, o);
    }
    uint v = tid * 4, i = tid * 6;
    out.set_index(i, v); out.set_index(i + 1, v + 1); out.set_index(i + 2, v + 2);
    out.set_index(i + 3, v); out.set_index(i + 4, v + 2); out.set_index(i + 5, v + 3);
}

static float linearFog(float d, float s, float e) {
    if (d <= s) return 0.0;
    if (d >= e) return 1.0;
    return (d - s) / (e - s);
}

// Texture detail: the flat material color times the ratio of the texel's luma to the texture's mean luma,
// so colors (and biome tints) stay calibrated while blocks show their texture. The texture repeats once per
// block; mip selection uses the gradients of the continuous block coordinate, so it doesn't break at tile
// seams, and far away the smallest mips average back to the flat color. Transparent texels (leaves, ice)
// darken slightly instead of showing whatever color they store (through vanilla's cutout leaves you see
// shaded leaves further in); water's texels are all translucent and keep the flat color, since water blends.
// Full-resolution grass sides blend vanilla's tinted fringe (grass_block_side_overlay) over the dirt.
static float4 lodShade(VOut in, constant LodUniforms& u, constant LodSpriteGPU* sprites, const device uint* aoBits,
                       texture2d<float> atlas, sampler atlasSampler) {
    float ao = lodAO(in, aoBits);
    float3 color = unpack_unorm4x8_to_float(in.color).rgb * ao;
    if (u.camFrac.w > 0.5) {
        uint face = (in.matFace >> 8) & 7, mat = in.matFace & 255;
        float3 wp = in.rel + u.camFrac.xyz;
        float2 bc = face < 2 ? float2(wp.z, -wp.y) : (face < 4 ? wp.xz : float2(wp.x, -wp.y));
        bool top = face == 2 || face == 3;
        float4 rect = top ? sprites[mat].top : sprites[mat].side;
        float2 size = rect.zw - rect.xy;
        float4 t = atlas.sample(atlasSampler, rect.xy + fract(bc) * size, gradient2d(dfdx(bc) * size, dfdy(bc) * size));
        float luma = dot(t.rgb, float3(0.2126, 0.7152, 0.0722));
        float mean = top ? sprites[mat].luma.x : sprites[mat].luma.y;
        color *= clamp(mix(mat == MAT_WATER ? 1.0 : TRANSPARENT_SHADE, luma / max(mean, 0.02), t.a), 0.0, 2.0);
        if ((in.matFace >> 23) & 1) {
            float4 orect = sprites[GRASS_SIDE_SPRITE].top;
            float2 osize = orect.zw - orect.xy;
            float4 o = atlas.sample(atlasSampler, orect.xy + fract(bc) * osize, gradient2d(dfdx(bc) * osize, dfdy(bc) * osize));
            color = mix(color, unpack_unorm4x8_to_float(in.color2).rgb * ao * (o.r / GRASS_GRAY), o.a);
        }
    }
    float horiz = length(in.rel.xz);
    float spherical = length(in.rel);
    float cylindrical = max(horiz, abs(in.rel.y));
    float fog = max(linearFog(spherical, u.envStart, u.envEnd), linearFog(cylindrical, u.rdStart, u.rdEnd));
    // Deep water: opaque, with what the skipped floor would have added through the water (a sand-gravel floor
    // under 12+ blocks of water, at vanilla's lightmap brightness for sky light 0-3).
    float alpha = u.alpha;
    if ((in.matFace >> 25) & 1) {
        color = color * alpha + unpack_unorm4x8_to_float(in.color2).rgb * (1.0 - alpha);
        alpha = 1.0;
    }
    return float4(mix(color, u.fogColor.rgb, fog * u.fogColor.a), alpha);
}

#if LIT_MODE
// Lit mode: the color and the G-buffer. The water pipelines write it only with water on (METALMC_EXP=water): the flag for
// the relight's reflections (litPackWater; without it water keeps its forward color).
struct LodOut { float4 color [[color(0)]]; uint2 gbuf [[color(1)]]; };

// The light lod_vs multiplies in without lit mode (the face's shade and vanilla's lightmap), from the depth field and
// block light it packed into the color's alpha byte; also the sky and block light levels for the G-buffer.
static float3 lodLitK(VOut in, constant LodUniforms& u, texture2d<float> lightmap, sampler ls, thread float& sky, thread float& block) {
    uint levels = in.color >> 24;
    uint depth = levels & 15u;
    block = float(levels >> 4);
    sky = max(0.0, float(u.seamInfo.z) - float(depth));
    return kShade[(in.matFace >> 8) & 7] * lodLight(u, lightmap, ls, sky, depth, block);
}

// lodShade with unlit vertex colors: the same color (k applied here), and the albedo (material color, texture detail
// and the grass fringe, without light, shade or AO) for the G-buffer.
static LodOut lodShadeLit(VOut in, constant LodUniforms& u, constant LodSpriteGPU* sprites, const device uint* aoBits,
                          texture2d<float> atlas, sampler atlasSampler, texture2d<float> lightmap, sampler ls) {
    float ao = lodAO(in, aoBits);
    float sky, block;
    float3 k = lodLitK(in, u, lightmap, ls, sky, block);
    float3 albedo = unpack_unorm4x8_to_float(in.color).rgb;
    if (u.camFrac.w > 0.5) {
        uint face = (in.matFace >> 8) & 7, mat = in.matFace & 255;
        float3 wp = in.rel + u.camFrac.xyz;
        float2 bc = face < 2 ? float2(wp.z, -wp.y) : (face < 4 ? wp.xz : float2(wp.x, -wp.y));
        bool top = face == 2 || face == 3;
        float4 rect = top ? sprites[mat].top : sprites[mat].side;
        float2 size = rect.zw - rect.xy;
        float4 t = atlas.sample(atlasSampler, rect.xy + fract(bc) * size, gradient2d(dfdx(bc) * size, dfdy(bc) * size));
        float luma = dot(t.rgb, float3(0.2126, 0.7152, 0.0722));
        float mean = top ? sprites[mat].luma.x : sprites[mat].luma.y;
        albedo *= clamp(mix(mat == MAT_WATER ? 1.0 : TRANSPARENT_SHADE, luma / max(mean, 0.02), t.a), 0.0, 2.0);
        if ((in.matFace >> 23) & 1) {
            float4 orect = sprites[GRASS_SIDE_SPRITE].top;
            float2 osize = orect.zw - orect.xy;
            float4 o = atlas.sample(atlasSampler, orect.xy + fract(bc) * osize, gradient2d(dfdx(bc) * osize, dfdy(bc) * osize));
            albedo = mix(albedo, unpack_unorm4x8_to_float(in.color2).rgb * (o.r / GRASS_GRAY), o.a);
        }
    }
    float3 color = albedo * k * ao;
    float horiz = length(in.rel.xz);
    float spherical = length(in.rel);
    float cylindrical = max(horiz, abs(in.rel.y));
    float fog = max(linearFog(spherical, u.envStart, u.envEnd), linearFog(cylindrical, u.rdStart, u.rdEnd));
    float alpha = u.alpha;
    if ((in.matFace >> 25) & 1) {
        color = color * alpha + unpack_unorm4x8_to_float(in.color2).rgb * (1.0 - alpha);
        alpha = 1.0;
    }
    LodOut out;
    out.color = float4(mix(color, u.fogColor.rgb, fog * u.fogColor.a), alpha);
    out.gbuf = litPack(albedo, ao, (in.matFace >> 8) & 7, in.pos.z, sky, block);
#if LIT_WATER
    // Water (METALMC_EXP=water, Lit.swift): its surface flagged for the relight's reflections (the water pipelines write
    // the G-buffer with water on).
    if ((in.matFace & 255u) == MAT_WATER) out.gbuf = litPackWater((in.matFace >> 8) & 7u, in.pos.z, sky, block);
#endif
    return out;
}

// The fragment functions' return type, the lightmap they take in lit mode, and the shading they call.
#define LOD_FS_OUT LodOut
#define LOD_FS_LIGHTMAP , texture2d<float> lightmap [[texture(29)]], sampler lightSampler [[sampler(14)]]
#define LOD_SHADE(in, u, sprites, aoBits, atlas, atlasSampler) lodShadeLit(in, u, sprites, aoBits, atlas, atlasSampler, lightmap, lightSampler)
#else
#define LOD_FS_OUT float4
#define LOD_FS_LIGHTMAP
#define LOD_SHADE(in, u, sprites, aoBits, atlas, atlasSampler) lodShade(in, u, sprites, aoBits, atlas, atlasSampler)
#endif

fragment LOD_FS_OUT lod_fs(VOut in [[stage_in]], constant LodUniforms& u [[buffer(19)]],
                           constant LodSpriteGPU* sprites [[buffer(20)]], const device uint* aoBits [[buffer(22)]],
                           texture2d<float> atlas [[texture(30)]], sampler atlasSampler [[sampler(15)]] LOD_FS_LIGHTMAP) {
    return LOD_SHADE(in, u, sprites, aoBits, atlas, atlasSampler);
}

// Position-only vertex function, used by the quad-visibility measurement (METALMC_EXP=quadvis). A full "slim" path
// built on it (vertex outputs cut from 52 to 24 bytes, the fragment shader rebuilding color, light and AO from the quad)
// measured no faster (main pass 4.21 vs 4.22 ms at 1.8 M quads): the LOD's vertex-stage cost is the number of
// triangles, not the size of their outputs.
struct VOutSlim {
    float4 pos [[position]];
    uint quad [[flat]];
    uint slot [[flat]];
};

vertex VOutSlim lod_vs_slim(uint vid [[vertex_id]], uint draw [[base_instance]],
                            const device uint2* quads [[buffer(18)]],
                            constant LodUniforms& u [[buffer(19)]],
                            const device Xform* xforms [[buffer(20)]]) {
    uint2 q = quads[vid >> 2];
    uint face = (q.x >> 25) & 7;
    float3 local = float3(q.x & 255, (q.x >> 16) & 511, (q.x >> 8) & 255);
    float w = float(((q.y >> 8) & 255) + 1), h = float(((q.y >> 16) & 255) + 1);
    float4 xs = xforms[draw].offsetScale;
    float3 rel = xs.xyz + (local + kCorners[face][vid & 3] * extentScale(face, w, h)) * xs.w;
    VOutSlim o;
    float4 clip = u.proj * (u.view * float4(rel, 1.0));
    clip.y = -clip.y;
    o.pos = clip;
    o.quad = vid >> 2;
    o.slot = draw;
    return o;
}

// Debug (METALMC_EXP=quadvis): re-drawn with depth test "equal" after the LOD, marks every quad that owns a final
// pixel. buffer 27: one byte per submitted quad; buffer 28: this draw's first index into it.
[[early_fragment_tests]] fragment void lod_fs_vis(VOutSlim in [[stage_in]], device uchar* marks [[buffer(27)]],
                                                  constant uint& base [[buffer(28)]], constant uint& first [[buffer(24)]]) {
    marks[base + in.quad - first] = 1;
}

// The seam with vanilla: tiles that overlap vanilla's area use this variant, which drops the pixels of LOD
// voxels whose chunk section vanilla drew this frame (a bitmap of sections around the camera). It's exact
// at any camera height (vanilla picks sections by 3D distance), leaves no gap and never draws LOD over
// vanilla terrain. Only these tiles pay for the discard, which turns off hidden-surface removal for them.
fragment LOD_FS_OUT lod_fs_seam(VOut in [[stage_in]], constant LodUniforms& u [[buffer(19)]],
                                constant LodSpriteGPU* sprites [[buffer(20)]],
                                const device uint* vanilla [[buffer(21)]], const device uint* aoBits [[buffer(22)]],
                                texture2d<float> atlas [[texture(30)]], sampler atlasSampler [[sampler(15)]] LOD_FS_LIGHTMAP) {
    // A point just inside the voxel this face belongs to.
    float3 p = in.rel + u.camInSection.xyz - kNormal[(in.matFace >> 8) & 7] * 0.01;
    int3 sec = int3(floor(p / 16.0));
    int H = int(u.camInSection.w), W = u.seamInfo.y;
    int ix = sec.x + H, iz = sec.z + H, iy = u.seamInfo.x + sec.y;
    if (ix >= 0 && iz >= 0 && ix < W && iz < W && iy >= 0 && iy < 24) {
        uint bit = uint((iy * W + iz) * W + ix);
        if ((vanilla[bit >> 5] & (1u << (bit & 31))) != 0) discard_fragment();
    }
    return LOD_SHADE(in, u, sprites, aoBits, atlas, atlasSampler);
}

// Level transitions: a tile that appears fades in over a few frames through a screen-door pattern, and the tile
// it replaces keeps drawing the complementary pattern until then (buffer 24: progress, 1 if fading out, 1 if
// the seam bitmap is bound at 21). Every
// pixel shows one of the two. Only fading tiles pay for the discard (and the seam test, which they always run).
constant float kBayer4[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
fragment LOD_FS_OUT lod_fs_fade(VOut in [[stage_in]], constant LodUniforms& u [[buffer(19)]],
                                constant LodSpriteGPU* sprites [[buffer(20)]],
                                const device uint* vanilla [[buffer(21)]], const device uint* aoBits [[buffer(22)]],
                                constant float4& fade [[buffer(24)]],
                                texture2d<float> atlas [[texture(30)]], sampler atlasSampler [[sampler(15)]] LOD_FS_LIGHTMAP) {
    uint2 px = uint2(in.pos.xy) & 3u;
    float t = (kBayer4[px.y * 4 + px.x] + 0.5) / 16.0;
    if (fade.y > 0.5 ? t < fade.x : t >= fade.x) discard_fragment();
    if (fade.z > 0.5) {
        float3 p = in.rel + u.camInSection.xyz - kNormal[(in.matFace >> 8) & 7] * 0.01;
        int3 sec = int3(floor(p / 16.0));
        int H = int(u.camInSection.w), W = u.seamInfo.y;
        int ix = sec.x + H, iz = sec.z + H, iy = u.seamInfo.x + sec.y;
        if (ix >= 0 && iz >= 0 && ix < W && iz < W && iy >= 0 && iy < 24) {
            uint bit = uint((iy * W + iz) * W + ix);
            if ((vanilla[bit >> 5] & (1u << (bit & 31))) != 0) discard_fragment();
        }
    }
    return LOD_SHADE(in, u, sprites, aoBits, atlas, atlasSampler);
}

// Occlusion test. After the LOD is drawn, every candidate tile's bounding box is rasterized in the same
// pass with depth testing and no writes. At that point the depth buffer holds vanilla's solid terrain and
// the LOD. Fragments that survive mark the tile visible; the CPU reads the marks once the frame completes.
struct BoxOut {
    float4 pos [[position]];
    uint slot [[flat]];
};
// 12 triangles over the 8 box corners (bit 0 = x, bit 1 = y, bit 2 = z), counter-clockwise seen from
// outside like the LOD quads, so back-face culling keeps the faces toward the camera.
constant ushort kBoxIndex[36] = {
    0, 4, 6, 0, 6, 2,   1, 3, 7, 1, 7, 5,   0, 1, 5, 0, 5, 4,
    2, 6, 7, 2, 7, 3,   0, 2, 3, 0, 3, 1,   4, 5, 7, 4, 7, 6,
};
vertex BoxOut lod_box_vs(uint vid [[vertex_id]], uint iid [[instance_id]],
                         constant LodUniforms& u [[buffer(19)]],
                         const device float4* boxes [[buffer(22)]]) {
    float3 lo = boxes[2 * iid].xyz, hi = boxes[2 * iid + 1].xyz;
    uint c = kBoxIndex[vid];
    float3 p = float3((c & 1) ? hi.x : lo.x, (c & 2) ? hi.y : lo.y, (c & 4) ? hi.z : lo.z);
    float4 clip = u.proj * (u.view * float4(p, 1.0));
    clip.y = -clip.y;
    BoxOut o;
    o.pos = clip;
    o.slot = iid;
    return o;
}
[[early_fragment_tests]]
fragment void lod_box_fs(BoxOut in [[stage_in]], device uint* visible [[buffer(23)]]) {
    visible[in.slot] = 1;
}
"""

/// Uniform block layout shared with the shader (must match LodUniforms).
struct LodUniforms {
    var proj: simd_float4x4
    var view: simd_float4x4
    var fogColor: SIMD4<Float>
    var envStart: Float, envEnd: Float, rdStart: Float, rdEnd: Float
    var discardRadius: Float
    var sky: Float
    var alpha: Float = 1
    var lightmapOn: Float = 0
    var camFrac: SIMD4<Float> = .zero
    var camInSection: SIMD4<Float> = .zero
    var seamInfo: SIMD4<Int32> = .zero
}

/// Half-size of the vanilla-section bitmap around the camera section (render distance 32 plus margin).
let lodSeamHalf = 34
let lodSeamWidth = 2 * lodSeamHalf + 1
let lodSeamWords = (lodSeamWidth * lodSeamWidth * 24 + 31) / 32

/// METALMC_EXP=tilesel picks levels per tile instead of per node (tile distance < lodTileSplit x half the
/// node size hands a tile to the finer level). METALMC_TILESPLIT sets the factor.
let lodTileSelection = experiments.contains("tilesel")
let lodTileSplit = Double(ProcessInfo.processInfo.environment["METALMC_TILESPLIT"] ?? "") ?? 2.8

/// Frames a level transition takes to fade (METALMC_FADE, 0 turns fading off): tiles that appear dither in while
/// the ones they replace dither out, instead of popping. 24 frames is 0.2 s at 120 Hz.
/// METALMC_EXP=nozoomlod: levels by distance alone, even zoomed in.
let lodNoZoom = experiments.contains("nozoomlod")
/// A node splits into finer children when the camera is closer than this times the child's size (METALMC_SPLIT).
/// Higher values keep finer levels farther out: smaller voxels on screen, more quads. 3 keeps voxels past the
/// level-0 ring under about 4 px at the panel's resolution (6 px at 2) for 18% more quads, and a 120 Hz flight
/// still dropped 1-2 frames, as at 2.
let lodSplitFactor: Double = Double(ProcessInfo.processInfo.environment["METALMC_SPLIT"] ?? "") ?? 3.0
let lodFadeFrames = Int(ProcessInfo.processInfo.environment["METALMC_FADE"] ?? "") ?? 24

/// METALMC_EXP=meshshader draws the LOD with a mesh shader (one thread per quad) instead of indexed vertices.
let lodMeshShaders = experiments.contains("meshshader")
/// Horizon occlusion of LOD sub-tiles (lodHorizonTest), METALMC_EXP=horizon. Off: on real terrain it hid about a quarter
/// of the sub-tiles in view but only 11% of the quads (4.22 -> 3.99 ms main pass) for 3.5 ms of CPU per frame, because
/// most hidden LOD quads are hidden at a much finer grain (foreshortened steps and faces behind their neighbors).
let lodHorizon = experiments.contains("horizon")
/// Sub-tiles outside the view frustum aren't drawn (METALMC_EXP=nostfrustum draws whole tiles).
let lodSubtileFrustum = !experiments.contains("nostfrustum")
/// Debug (METALMC_EXP=cullfrac): drop 3 of every 4 LOD quads before rasterization, to measure what culling saves.
let lodCullFrac = experiments.contains("cullfrac")
/// METALMC_EXP=quadvis: every 240 frames, count the drawn opaque LOD quads that own at least one final pixel (per level).
let lodQuadVis = experiments.contains("quadvis")
/// Profiling (holes in the picture): METALMC_EXP=ffskip doesn't draw the far field, lodskip doesn't draw the LOD's quads,
/// so a traced flight's main pass minus the plain one is what they cost (tools/bench/passes.py).
let lodSkipFar = experiments.contains("ffskip")
let lodSkipQuads = experiments.contains("lodskip")
/// lodSkipQuads, or an offline A/B's (mmc_debug_lod_paths).
nonisolated(unsafe) var lodSkipQuadsOn = lodSkipQuads
/// METALMC_EXP=cullstats: every 600 frames, the CPU goes over the frame's drawn quads and counts, per level, those that
/// can't make a fragment: facing away from the camera (each quad, where the draw lists cull by tile), off screen, or
/// so thin on screen that no pixel center falls in their bounding box. What culling quads before the vertex stage
/// (whose cost is per vertex invoked) could save, measured before building it.
let lodCullStats = experiments.contains("cullstats")

/// lodCullStats' counts for the frame's draws (see there): per level, quads drawn, facing away, off screen, too thin.
func lodCountCullable(_ draws: [(node: LodMeshNode, slot: Int, first: Int, count: Int)], xforms: [SIMD4<Float>],
                      projView: simd_float4x4, width: Int, height: Int) -> [[Int]] {
    var out = [[Int]](repeating: [0, 0, 0, 0], count: 9)
    let corners: [[SIMD3<Float>]] = [
        [SIMD3(1, 0, 0), SIMD3(1, 1, 0), SIMD3(1, 1, 1), SIMD3(1, 0, 1)],
        [SIMD3(0, 0, 1), SIMD3(0, 1, 1), SIMD3(0, 1, 0), SIMD3(0, 0, 0)],
        [SIMD3(0, 1, 0), SIMD3(0, 1, 1), SIMD3(1, 1, 1), SIMD3(1, 1, 0)],
        [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 0, 1), SIMD3(0, 0, 1)],
        [SIMD3(1, 0, 1), SIMD3(1, 1, 1), SIMD3(0, 1, 1), SIMD3(0, 0, 1)],
        [SIMD3(0, 0, 0), SIMD3(0, 1, 0), SIMD3(1, 1, 0), SIMD3(1, 0, 0)],
    ]
    let normals: [SIMD3<Float>] = [SIMD3(1, 0, 0), SIMD3(-1, 0, 0), SIMD3(0, 1, 0), SIMD3(0, -1, 0), SIMD3(0, 0, 1), SIMD3(0, 0, -1)]
    let W = Float(width), H = Float(height)
    for d in draws {
        let level = min(8, d.node.level)
        let xs = xforms[d.slot]
        let q = d.node.buffer.contents().bindMemory(to: SIMD2<UInt32>.self, capacity: d.node.quadCount)
        for i in d.first..<(d.first + d.count) {
            out[level][0] += 1
            let w0 = q[i].x, w1 = q[i].y
            let face = Int((w0 >> 25) & 7)
            if face > 5 { continue }
            let local = SIMD3(Float(w0 & 255), Float((w0 >> 16) & 511), Float((w0 >> 8) & 255))
            let w = Float(((w1 >> 8) & 255) + 1), h = Float(((w1 >> 16) & 255) + 1)
            let ext = face < 2 ? SIMD3<Float>(1, w, h) : (face < 4 ? SIMD3<Float>(w, 1, h) : SIMD3<Float>(w, h, 1))
            let base = SIMD3(xs.x, xs.y, xs.z)
            let p0 = base + (local + corners[face][0] * ext) * xs.w
            // The camera is at the origin: a face can be seen only from in front of its plane.
            if simd_dot(normals[face], p0) >= 0 { out[level][1] += 1; continue }
            var lo = SIMD2<Float>(repeating: .infinity), hi = SIMD2<Float>(repeating: -.infinity)
            var behind = false
            for c in 0..<4 {
                let p = base + (local + corners[face][c] * ext) * xs.w
                let clip = projView * SIMD4(p, 1)
                if clip.w <= 1e-4 { behind = true; break }
                let s = SIMD2((clip.x / clip.w * 0.5 + 0.5) * W, (clip.y / clip.w * 0.5 + 0.5) * H)
                lo = simd_min(lo, s)
                hi = simd_max(hi, s)
            }
            if behind { continue }
            if hi.x < 0 || hi.y < 0 || lo.x > W || lo.y > H { out[level][2] += 1; continue }
            // A pixel center k + 0.5 inside [lo, hi] on both axes (a little slack: kept if in doubt).
            let e: Float = 1e-3
            let cx = (hi.x + e - 0.5).rounded(.down) >= (lo.x - e - 0.5).rounded(.up)
            let cy = (hi.y + e - 0.5).rounded(.down) >= (lo.y - e - 0.5).rounded(.up)
            if !(cx && cy) { out[level][3] += 1 }
        }
    }
    return out
}
nonisolated(unsafe) var lodVisPipe: MTLRenderPipelineState?
/// Quads per mesh threadgroup (METALMC_MESHQUADS, default 32).
let lodMeshQuads = Int(ProcessInfo.processInfo.environment["METALMC_MESHQUADS"] ?? "") ?? 32

/// METALMC_EXP=quadcull: quads that can't make a single fragment (lodQuadCulled in the shaders: no pixel center in their
/// bounding box on screen, or facing away) aren't issued to the vertex stage, whose cost is per vertex invoked. A compute
/// pass ahead of the frame writes each draw's surviving quads, in order, and the indirect draws that draw them
/// (LodQuadCull). The picture is the same: only quads that would make no fragment are left out. The vertex path only
/// (with meshshader the LOD draws as before).
let lodQuadCullEnv = experiments.contains("quadcull")
/// The paths in use: the switches above, or an offline A/B's in one process (mmc_debug_lod_paths).
nonisolated(unsafe) var lodQuadCullOn = lodQuadCullEnv
nonisolated(unsafe) var lodMeshOn = lodMeshShaders
/// Pixels each projected corner may be off by in the culling test (METALMC_CULLMARGIN): the rasterizer snaps vertices
/// to a sub-pixel grid, and lod_vs rounds a little differently from the test. On this GPU a covered pixel center lies
/// at most 0.0023 px outside its triangle (measured over 41 K random thin triangles: 8 fractional bits), so 1/16 is
/// about 27 times that, and it would still cover snapping to 4 bits.
let lodCullMargin = Float(ProcessInfo.processInfo.environment["METALMC_CULLMARGIN"] ?? "") ?? 0.0625
/// Quads per culling job (one SIMD group, one indirect draw, METALMC_CULLJOB): longer draws are split into consecutive
/// jobs, which draw in the same order.
let lodCullJobQuads = max(32, Int(ProcessInfo.processInfo.environment["METALMC_CULLJOB"] ?? "") ?? 4096)
/// Measuring (mmc_debug_lod_paths): the culling pass keeps every quad, so the frame pays for the pass and the indirect
/// draws without culling anything.
nonisolated(unsafe) var lodCullKeepAll = false

/// Must match LodCullParams in the shaders.
struct LodCullParams {
    var viewProj: simd_float4x4
    var viewport: SIMD2<Float>
    var margin: Float
    var minW: Float
    var jobs: UInt32 = 0
    var pad0: UInt32 = 0, pad1: UInt32 = 0, pad2: UInt32 = 0
}
/// Must match LodCullJob in the shaders: GPU addresses of the job's first quad and AO offset.
struct LodCullJob {
    var quads: UInt64
    var aoOffsets: UInt64
    var count: UInt32
    var slot: UInt32
    var outBase: UInt32
    var pad: UInt32 = 0
}

/// One frame's culling buffers: the job table and transforms (written by the CPU), the surviving quads and AO offsets
/// and the indirect draws (written by the culling pass, read by the frame's main pass).
final class LodCullFrame {
    var jobs: MTLBuffer?
    var xforms: MTLBuffer?
    var quads: MTLBuffer?
    var ao: MTLBuffer?
    var args: MTLBuffer?
    var pending = 0          // command buffers still to complete (the culling pass and the frame); LodQuadCull.lock
    var counted = true       // its survivors are in the stats
    var jobCount = 0
    var quadsIn = 0
    var gpuSeconds = 0.0
    var cb: MTLCommandBuffer?
    // The node buffers the jobs read through their GPU addresses, held until the frame completes: a node the LOD drops
    // meanwhile must not free them before the culling pass has run.
    var reads: [MTLResource] = []
    // GPU start and end of the culling pass's command buffer and of the frame's (measuring: the culling pass runs while
    // the frame's first passes start, and the frame's draws wait for it).
    var cullTimes = (start: 0.0, end: 0.0), frameTimes = (start: 0.0, end: 0.0)
}

/// The vertex path's quad culling (lodQuadCullOn): see lod_cull.
final class LodQuadCull: @unchecked Sendable {
    static let shared = LodQuadCull()
    let lock = NSLock()
    var frames: [LodCullFrame] = []
    var pipe: MTLComputePipelineState?
    var last: LodCullFrame?
    // Since the last log line: frames, quads submitted to the pass, quads it kept, its GPU time.
    var statFrames = 0, statIn = 0, statOut = 0, statGpu = 0.0

    /// The culling pipeline, nil until it's compiled (the frame draws directly until then). `compile`: compile it now
    /// from the LOD's library if it isn't. The LOD's background warm-up does that with the switch on; on the render thread
    /// only an offline A/B that turned culling on does (mmc_debug_lod_paths).
    func pipeline(compile: Bool) -> MTLComputePipelineState? {
        lock.lock(); let p = pipe; lock.unlock()
        if let p { return p }
        guard compile, let fn = LodRenderer.shared.library?.makeFunction(name: "lod_cull") else { return nil }
        var made: MTLComputePipelineState?
        do {
            made = try ctx.device.makeComputePipelineState(function: fn)
        } catch {
            log("LOD quad culling: pipeline failed: \(error)")
        }
        lock.lock(); defer { lock.unlock() }
        if pipe == nil { pipe = made }
        return pipe
    }

    /// Adds a finished frame's survivors to the stats (render thread, under the lock).
    private func count(_ f: LodCullFrame) {
        guard !f.counted, let args = f.args else { return }
        f.counted = true
        let a = args.contents().bindMemory(to: UInt32.self, capacity: 5 * f.jobCount)
        var kept = 0
        for j in 0..<f.jobCount { kept += Int(a[5 * j]) / 6 }
        statFrames += 1
        statIn += f.quadsIn
        statOut += kept
        statGpu += f.gpuSeconds
    }

    /// A frame's buffers that no command buffer still uses, big enough for `quads` quads and `jobs` jobs.
    private func acquire(quads: Int, jobs: Int, xforms: Int) -> LodCullFrame? {
        lock.lock()
        for f in frames where f.pending == 0 { count(f) }
        var free = frames.first { $0.pending == 0 }
        if free == nil && frames.count < 4 {
            free = LodCullFrame()
            frames.append(free!)
        }
        lock.unlock()
        guard let f = free else { return nil }
        func grow(_ b: MTLBuffer?, _ bytes: Int, _ options: MTLResourceOptions, _ label: String) -> MTLBuffer? {
            if let b, b.length >= bytes { return b }
            let made = ctx.device.makeBuffer(length: max(4096, bytes + bytes / 2), options: options)
            made?.label = label
            return made
        }
        f.jobs = grow(f.jobs, jobs * MemoryLayout<LodCullJob>.stride, .storageModeShared, "MetalMC LOD cull jobs")
        f.xforms = grow(f.xforms, xforms * 16, .storageModeShared, "MetalMC LOD cull transforms")
        f.quads = grow(f.quads, quads * 8, .storageModePrivate, "MetalMC LOD culled quads")
        f.ao = grow(f.ao, quads * 4, .storageModePrivate, "MetalMC LOD culled AO offsets")
        f.args = grow(f.args, jobs * 20, .storageModeShared, "MetalMC LOD culled draws")
        guard f.jobs != nil, f.xforms != nil, f.quads != nil, f.ao != nil, f.args != nil else { return nil }
        return f
    }

    /// Culls this frame's draws (node, first quad, count, transform slot), in the order they'll be drawn: encodes the
    /// culling pass in a command buffer of its own and commits it now, ahead of the frame's (committed when the frame
    /// is submitted), like the far field's fill. Returns the buffers and each draw's jobs (its indirect draws in the
    /// frame's args), or nil to draw directly this frame.
    func encode(_ draws: [(node: LodMeshNode, first: Int, count: Int, slot: Int)], xforms: [SIMD4<Float>], viewProj: simd_float4x4,
                width: Int, height: Int) -> (frame: LodCullFrame, jobs: [Range<Int>])? {
        guard let pipe = pipeline(compile: !lodQuadCullEnv), let frameCB = ctx.cb, width > 0, height > 0 else { return nil }
        var jobCount = 0, total = 0
        for d in draws {
            jobCount += (d.count + lodCullJobQuads - 1) / lodCullJobQuads
            total += d.count
        }
        guard jobCount > 0, let f = acquire(quads: total, jobs: jobCount, xforms: xforms.count),
              let jobsBuffer = f.jobs, let xformBuffer = f.xforms, let quads = f.quads, let ao = f.ao, let args = f.args,
              let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return nil }
        let jobs = jobsBuffer.contents().bindMemory(to: LodCullJob.self, capacity: jobCount)
        var ranges: [Range<Int>] = []
        ranges.reserveCapacity(draws.count)
        var j = 0, out = 0
        var seen = Set<ObjectIdentifier>()
        var read: [MTLResource] = []
        for d in draws {
            let j0 = j
            var k = 0
            while k < d.count {
                let n = min(lodCullJobQuads, d.count - k)
                jobs[j] = LodCullJob(quads: d.node.buffer.gpuAddress + UInt64(8 * (d.first + k)),
                                     aoOffsets: d.node.aoOffsets.gpuAddress + UInt64(4 * (d.first + k)),
                                     count: UInt32(n), slot: UInt32(d.slot), outBase: UInt32(out))
                j += 1
                k += n
                out += n
            }
            ranges.append(j0..<j)
            if seen.insert(ObjectIdentifier(d.node)).inserted {
                read.append(d.node.buffer)
                read.append(d.node.aoOffsets)
            }
        }
        xforms.withUnsafeBytes { xformBuffer.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count) }
        var params = LodCullParams(viewProj: viewProj, viewport: SIMD2(Float(width), Float(height)), margin: lodCullKeepAll ? -1 : lodCullMargin,
                                   minW: 0.05, jobs: UInt32(jobCount))
        cb.label = "MetalMC LOD quad culling"
        enc.label = "MetalMC LOD quad culling"
        enc.setComputePipelineState(pipe)
        enc.setBuffer(jobsBuffer, offset: 0, index: 0)
        enc.setBytes(&params, length: MemoryLayout<LodCullParams>.stride, index: 1)
        enc.setBuffer(xformBuffer, offset: 0, index: 2)
        enc.setBuffer(quads, offset: 0, index: 3)
        enc.setBuffer(ao, offset: 0, index: 4)
        enc.setBuffer(args, offset: 0, index: 5)
        // The jobs reach the nodes' buffers through their GPU addresses.
        enc.useResources(read, usage: .read)
        // A SIMD group per job, 8 to a threadgroup.
        let groups = max(1, min(8, pipe.maxTotalThreadsPerThreadgroup / pipe.threadExecutionWidth))
        enc.dispatchThreadgroups(MTLSize(width: (jobCount + groups - 1) / groups, height: 1, depth: 1),
                                 threadsPerThreadgroup: MTLSize(width: groups * pipe.threadExecutionWidth, height: 1, depth: 1))
        enc.endEncoding()
        lock.lock()
        f.pending = 2
        f.counted = false
        f.jobCount = jobCount
        f.quadsIn = total
        f.cb = cb
        f.reads = read
        last = f
        lock.unlock()
        cb.addCompletedHandler { [self] cb in
            lock.lock()
            f.gpuSeconds = cb.gpuEndTime - cb.gpuStartTime
            f.cullTimes = (cb.gpuStartTime, cb.gpuEndTime)
            f.pending -= 1
            if f.pending == 0 { f.reads = [] }
            lock.unlock()
            if cb.status == .error { log("LOD quad culling: GPU error \(cb.error.map { "\($0)" } ?? "")") }
        }
        frameCB.addCompletedHandler { [self] fcb in
            lock.lock()
            f.frameTimes = (fcb.gpuStartTime, fcb.gpuEndTime)
            f.pending -= 1
            if f.pending == 0 { f.reads = [] }
            lock.unlock()
        }
        cb.commit()
        return (f, ranges)
    }

    /// "kept of submitted (share), culling pass GPU ms per frame" since the last call, or nil with no finished frame.
    func takeStats() -> String? {
        lock.lock(); defer { lock.unlock() }
        for f in frames where f.pending == 0 { count(f) }
        guard statFrames > 0 else { return nil }
        let s = String(format: "%d of %d quads issued (%.1f%%), culling pass %.3f ms GPU, per frame over %d frames",
                       statOut / statFrames, statIn / statFrames, 100 * Double(statOut) / Double(max(statIn, 1)),
                       1000 * statGpu / Double(statFrames), statFrames)
        statFrames = 0; statIn = 0; statOut = 0; statGpu = 0
        return s
    }
}

/// METALMC_EXP=nolightmap lights the LOD with the old daylight curve instead of vanilla's lightmap (A/B).
let lodNoLightmap = experiments.contains("nolightmap")

/// Mean alpha of vanilla's water texture (water_still), the LOD water's opacity.
let lodWaterAlpha: Float = Float(ProcessInfo.processInfo.environment["METALMC_WATERALPHA"] ?? "") ?? 0.706
/// Brightness of transparent texels in texture detail (leaves, ice): through vanilla's cutout leaves you see shaded leaves
/// further in. METALMC_TRANSPARENTSHADE for calibration runs.
let lodTransparentShade: Float = Float(ProcessInfo.processInfo.environment["METALMC_TRANSPARENTSHADE"] ?? "") ?? 0.75

/// METALMC_EXP=lodflat turns off LOD texture detail (flat colors, for A/B comparisons).
let lodFlat = experiments.contains("lodflat")

/// One frame's occlusion test: the boxes drawn and the GPU-written visibility marks, one per slot.
final class LodVisSet {
    static let capacity = 8192
    let marks: MTLBuffer        // UInt32 per slot, written by lod_box_fs
    let boxes: MTLBuffer        // two float4 per slot (lo, hi), camera-relative
    var slots: [(LodMeshNode, Int)] = []
    var frame: UInt64 = 0
    var pending = false         // encoded, not yet harvested (render thread)
    var done = false            // command buffer completed (set on the completion thread, under visLock)

    init?() {
        guard let m = ctx.device.makeBuffer(length: Self.capacity * 4, options: [.storageModeShared]),
              let b = ctx.device.makeBuffer(length: Self.capacity * 32, options: [.storageModeShared]) else { return nil }
        m.label = "MetalMC LOD visibility"
        b.label = "MetalMC LOD boxes"
        marks = m
        boxes = b
    }
}

/// METALMC_EXP=noocc turns off LOD occlusion culling (for A/B comparisons).
let lodOcclusion = !experiments.contains("noocc")
/// METALMC_EXP=occnocull tests boxes but never skips tiles (measures the test's own cost).
let lodOccNoCull = experiments.contains("occnocull")

final class LodRenderer: @unchecked Sendable {
    static let shared = LodRenderer()

    let lock = NSLock()
    var world: LodWorld?                   // the dimension the player is in
    var worlds: [String: LodWorld] = [:]   // every dimension opened for this save or server; the others are paused
    var nextWorldId = 1
    var colorBuffer: MTLBuffer?
    var indexBuffer: MTLBuffer?        // shared pattern 4q + {0,1,2,0,2,3}
    var indexQuads = 0
    var pipelines: [String: MTLRenderPipelineState] = [:]
    var library: MTLLibrary?
    var lastFormats: (colors: [MTLPixelFormat], depth: MTLPixelFormat)?   // of the last pass the LOD drew into (debug)
    // Texture detail: Minecraft's block atlas and the sprite table (render thread).
    var atlas: MTLTexture?
    var spriteBuffer: MTLBuffer?
    // Vanilla's lightmap (render thread): LOD terrain is lit the way vanilla terrain is.
    var lightmap: MTLTexture?
    var lightSampler: MTLSamplerState?
    // The chunk sections vanilla drew this frame (SectionPos.asLong keys), set before each draw, and a ring
    // of bitmap buffers for the seam shader.
    var vanillaSections: [Int64] = []
    // Every section vanilla has compiled in its view area (drawn or not), and its render distance in chunks:
    // a tile whose sections are all compiled and in vanilla's range is left to vanilla.
    var compiledSections: [Int64] = []
    var vanillaDistance = 0
    var compiledWords = [UInt32](repeating: 0, count: lodSeamWords)
    var seamBuffers: [MTLBuffer] = []
    var atlasSampler: MTLSamplerState?
    var dummyTexture: MTLTexture?
    // Occlusion culling (render thread, except LodVisSet.done).
    let visLock = NSLock()
    var visSets: [LodVisSet] = []
    var horizonCulledTiles = 0      // tiles the horizon test hid entirely (debug counter)
    var frame: UInt64 = 0
    var latestResult: UInt64 = 0       // newest frame whose occlusion results have been read
    var lastCamera = SIMD3<Double>(repeating: .nan)
    var jumpFrame: UInt64 = 0          // last frame the camera jumped; results tested before it are stale
    var coveredTiles = 0               // tiles skipped because vanilla drew all their sections (logged every 1000 frames)
    // Level transitions (render thread): last frame's tiles per node, tiles fading in (and since when) and tiles
    // fading out (kept alive until their fade ends, even if their node was dropped).
    var lastSelection: [ObjectIdentifier: (LodMeshNode, UInt16)] = [:]
    var fadeIn: [ObjectIdentifier: (mask: UInt16, start: UInt64)] = [:]
    var fadeOut: [(node: LodMeshNode, mask: UInt16, start: UInt64)] = []
    var skipDebug = [0, 0, 0, 0, 0]    // seam tiles failing: y range, bitmap bounds, horizontal distance, not compiled; passing

    private let pipelineLock = NSLock()
    private let libraryLock = NSLock()      // held while the shader library is compiled and pipelines are made from it
    private var compiling = Set<String>()   // pass formats whose variants are being compiled in the background
    private var compiled = Set<String>()    // pass formats whose variants are all compiled
    private let compileQueue = DispatchQueue(label: "metalmc.lod.compile", qos: .userInitiated)

    /// The pipeline for one variant, or nil while it's being compiled. The first request for a pass's formats
    /// compiles every variant the LOD draws with them on a background queue (the shader library alone takes tens of
    /// milliseconds), so none of them compiles on the render thread: the LOD appears a few frames later instead of
    /// the frame it first draws (or first fades a tile) taking 10+ ms longer.
    func pipeline(colorFormats: [MTLPixelFormat], depth: MTLPixelFormat, box: Bool = false, seam: Bool = false, water: Bool = false,
                  mesh: Bool = false, fade: Bool = false) -> MTLRenderPipelineState? {
        let formats = colorFormats.map { String($0.rawValue) }.joined(separator: ",") + "/\(depth.rawValue)"
        let key = formats + (box ? "/box" : "") + (seam ? "/seam" : "") + (water ? "/water" : "") + (mesh ? "/mesh" : "") + (fade ? "/fade" : "")
        pipelineLock.lock()
        if let p = pipelines[key] { pipelineLock.unlock(); return p }
        let done = compiled.contains(formats), start = !done && !compiling.contains(formats)
        if start { compiling.insert(formats) }
        pipelineLock.unlock()
        if done {
            // A variant outside the warm-up set: compile it here.
            return makePipeline(key: key, colorFormats: colorFormats, depth: depth, box: box, seam: seam, water: water, mesh: mesh, fade: fade)
        }
        if start {
            compileQueue.async { [self] in
                let t0 = DispatchTime.now().uptimeNanoseconds
                for mesh in lodMeshShaders ? [false, true] : [false] {
                    for (box, seam, water, fade) in [(false, false, false, false), (false, true, false, false), (false, false, true, false),
                                                     (false, true, true, false), (false, false, false, true), (false, false, true, true),
                                                     (true, false, false, false)] where !(box && mesh) {
                        let k = formats + (box ? "/box" : "") + (seam ? "/seam" : "") + (water ? "/water" : "") + (mesh ? "/mesh" : "") + (fade ? "/fade" : "")
                        _ = makePipeline(key: k, colorFormats: colorFormats, depth: depth, box: box, seam: seam, water: water, mesh: mesh, fade: fade)
                    }
                }
                // Quad culling's pass (its pipeline doesn't depend on the pass's formats).
                if lodQuadCullEnv { _ = LodQuadCull.shared.pipeline(compile: true) }
                pipelineLock.lock(); compiling.remove(formats); compiled.insert(formats); pipelineLock.unlock()
                log("LOD: pipelines for \(formats) compiled in \((DispatchTime.now().uptimeNanoseconds - t0) / 1_000_000) ms")
            }
        }
        return nil
    }

    private func makePipeline(key: String, colorFormats: [MTLPixelFormat], depth: MTLPixelFormat, box: Bool, seam: Bool, water: Bool,
                              mesh: Bool, fade: Bool) -> MTLRenderPipelineState? {
        pipelineLock.lock()
        if let p = pipelines[key] { pipelineLock.unlock(); return p }
        pipelineLock.unlock()
        do {
            libraryLock.lock()
            defer { libraryLock.unlock() }
            if library == nil {
                let src = lodShaderSource
                    .replacingOccurrences(of: "MAT_WATER", with: "\(Mat.water.rawValue)u")
                    .replacingOccurrences(of: "MAT_LEAVES", with: "\(Mat.leaves.rawValue)u")
                    .replacingOccurrences(of: "MAT_GRASS", with: "\(Mat.grass.rawValue)u")
                    .replacingOccurrences(of: "MAT_DIRT", with: "\(Mat.dirt.rawValue)u")
                    .replacingOccurrences(of: "MAT_LAVA", with: "\(Mat.lava.rawValue)u")
                    .replacingOccurrences(of: "MAT_MAGMA", with: "\(Mat.magma.rawValue)u")
                    .replacingOccurrences(of: "MAT_GLOW_FIRST", with: "\(Mat.glowstone.rawValue)u")
                    .replacingOccurrences(of: "MAT_GLOW_LAST", with: "\(Mat.froglight.rawValue)u")
                    .replacingOccurrences(of: "MESH_QUADS", with: "\(lodMeshQuads)")
                    .replacingOccurrences(of: "GRASS_SIDE_SPRITE", with: "\(lodMaterialSprites.count - 1)u")
                    .replacingOccurrences(of: "GRASS_GRAY", with: "\(lodGrassGray)f")
                    .replacingOccurrences(of: "TRANSPARENT_SHADE", with: "\(lodTransparentShade)f")
                // Lab mode (ShaderLab.swift): after an edit, forget the library and every pipeline; the next frame compiles
                // them again in the background from the new library.
                library = try ShaderLab.library("lod", src) { [self] _ in
                    libraryLock.lock(); library = nil; libraryLock.unlock()
                    pipelineLock.lock(); pipelines = [:]; compiled = []; compiling = []; pipelineLock.unlock()
                    LodQuadCull.shared.lock.lock(); LodQuadCull.shared.pipe = nil; LodQuadCull.shared.lock.unlock()
                }
            }
            if mesh {
                let d = MTLMeshRenderPipelineDescriptor()
                d.label = "MetalMC LOD (mesh)"
                d.meshFunction = library!.makeFunction(name: "lod_mesh")
                d.fragmentFunction = library!.makeFunction(name: fade ? "lod_fs_fade" : (seam ? "lod_fs_seam" : "lod_fs"))
                d.maxTotalThreadsPerMeshThreadgroup = lodMeshQuads
                for (i, f) in colorFormats.enumerated() {
                    d.colorAttachments[i].pixelFormat = f
                    if i > 0 && !(litWritesGbuffer(i, f) && (!water || litWater)) { d.colorAttachments[i].writeMask = [] }
                }
                if water {
                    let a = d.colorAttachments[0]!
                    a.isBlendingEnabled = true
                    a.sourceRGBBlendFactor = .sourceAlpha
                    a.destinationRGBBlendFactor = .oneMinusSourceAlpha
                    a.sourceAlphaBlendFactor = .one
                    a.destinationAlphaBlendFactor = .oneMinusSourceAlpha
                }
                d.depthAttachmentPixelFormat = depth
                let (p, _) = try ctx.device.makeRenderPipelineState(descriptor: d, options: [])
                pipelineLock.lock(); pipelines[key] = p; pipelineLock.unlock()
                return p
            }
            let d = MTLRenderPipelineDescriptor()
            d.label = box ? "MetalMC LOD occlusion boxes" : "MetalMC LOD"
            d.vertexFunction = library!.makeFunction(name: box ? "lod_box_vs" : "lod_vs")
            d.fragmentFunction = library!.makeFunction(name: box ? "lod_box_fs" : (fade ? "lod_fs_fade" : (seam ? "lod_fs_seam" : "lod_fs")))
            for (i, f) in colorFormats.enumerated() {
                d.colorAttachments[i].pixelFormat = f
                // Only the main color target gets LOD color; extra targets (OIT) are left untouched, except lit mode's
                // G-buffer, which the opaque quads write (and with water on, the water). The occlusion boxes write no color.
                if (i > 0 && !(litWritesGbuffer(i, f) && (!water || litWater))) || box { d.colorAttachments[i].writeMask = [] }
            }
            if water {
                // Vanilla's translucent blending.
                let a = d.colorAttachments[0]!
                a.isBlendingEnabled = true
                a.sourceRGBBlendFactor = .sourceAlpha
                a.destinationRGBBlendFactor = .oneMinusSourceAlpha
                a.sourceAlphaBlendFactor = .one
                a.destinationAlphaBlendFactor = .oneMinusSourceAlpha
            }
            d.depthAttachmentPixelFormat = depth
            let p = try ctx.device.makeRenderPipelineState(descriptor: d)
            pipelineLock.lock(); pipelines[key] = p; pipelineLock.unlock()
            return p
        } catch {
            log("LOD pipeline failed: \(error)")
            return nil
        }
    }

    /// Reads the marks of every completed occlusion test (oldest first) into the nodes' tile state, then
    /// returns a free set for this frame (nil if all are still in flight).
    func harvestAndAcquire() -> LodVisSet? {
        visLock.lock()
        let ready = visSets.filter { $0.pending && $0.done }.sorted { $0.frame < $1.frame }
        visLock.unlock()
        for s in ready {
            let marks = s.marks.contents().bindMemory(to: UInt32.self, capacity: LodVisSet.capacity)
            for (i, (node, t)) in s.slots.enumerated() {
                node.tileTested[t] = s.frame
                if marks[i] != 0 { node.tileVisible[t] = s.frame }
            }
            latestResult = max(latestResult, s.frame)
            s.slots.removeAll(keepingCapacity: true)
            visLock.lock(); s.pending = false; s.done = false; visLock.unlock()
        }
        if let free = visSets.first(where: { !$0.pending }) { return free }
        if visSets.count < 4, let s = LodVisSet() { visSets.append(s); return s }
        return nil
    }

    func ensureIndexBuffer(quads: Int) {
        if indexQuads >= quads, indexBuffer != nil { return }
        var n = max(1024, indexQuads)
        while n < quads { n *= 2 }
        var idx = [UInt32](repeating: 0, count: n * 6)
        for q in 0..<n {
            let b = UInt32(4 * q)
            idx[6 * q] = b; idx[6 * q + 1] = b + 1; idx[6 * q + 2] = b + 2
            idx[6 * q + 3] = b; idx[6 * q + 4] = b + 2; idx[6 * q + 5] = b + 3
        }
        indexBuffer = ctx.device.makeBuffer(bytes: idx, length: idx.count * 4, options: [.storageModeShared])
        indexQuads = n
    }

    /// Sampler for the atlas (nearest texels, linear between mips, like Minecraft) and a 1 x 1 stand-in
    /// texture for when the atlas hasn't been set.
    func ensureTextureDefaults() {
        if atlasSampler == nil {
            let d = MTLSamplerDescriptor()
            d.minFilter = .nearest
            d.magFilter = .nearest
            d.mipFilter = .linear
            d.sAddressMode = .clampToEdge
            d.tAddressMode = .clampToEdge
            atlasSampler = ctx.device.makeSamplerState(descriptor: d)
        }
        if dummyTexture == nil {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
            d.storageMode = .shared
            let t = ctx.device.makeTexture(descriptor: d)
            var white: UInt32 = 0xFFFF_FFFF
            t?.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &white, bytesPerRow: 4)
            dummyTexture = t
        }
    }

    /// Material colors from the texture averages in LodColors.swift, indexed by material id.
    /// Per color-table entry scale for the loaded resource pack's textures (all 1 with vanilla's; mmc_lod_set_atlas).
    var packRatio = [SIMD3<Float>](repeating: SIMD3(1, 1, 1), count: 256 * 3)
    /// Biome tints as the game computes them (mmc_lod_set_tints); the built-in table until then.
    var tints = lodTints

    func ensureColors() {
        if colorBuffer != nil { return }
        var c = lodColorTable(tints: tints)
        for i in 0..<c.count { c[i] = SIMD4(c[i].x * packRatio[i].x, c[i].y * packRatio[i].y, c[i].z * packRatio[i].z, c[i].w) }
        // Calibration runs (METALMC_WATERGAIN, METALMC_GRASSGAIN, "r,g,b" or one value): scale the water and grass colors.
        func gain(_ name: String) -> SIMD3<Float>? {
            guard let v = ProcessInfo.processInfo.environment[name] else { return nil }
            let f = v.split(separator: ",").compactMap { Float($0) }
            return f.count == 3 ? SIMD3(f[0], f[1], f[2]) : (f.count == 1 ? SIMD3(repeating: f[0]) : nil)
        }
        func scale(_ mats: [Int], _ g: SIMD3<Float>) {
            for m in mats { for k in 0..<3 where (m * 3 + k) < c.count { let v = c[m * 3 + k]; c[m * 3 + k] = SIMD4(v.x * g.x, v.y * g.y, v.z * g.z, v.w) } }
        }
        if let g = gain("METALMC_WATERGAIN") { scale([Int(Mat.water.rawValue)] + (0..<32).map { Int(lodWaterBase) + $0 }, g) }
        if let g = gain("METALMC_GRASSGAIN") { scale([Int(Mat.grass.rawValue)] + (0..<32).map { Int(lodGrassBase) + $0 }, g) }
        colorBuffer = ctx.device.makeBuffer(bytes: c, length: c.count * 16, options: [.storageModeShared])
    }

    /// Quadtree selection. A node splits when the camera is closer than `splitFactor` x the child size
    /// (level 1 into level 0 within `level0Radius`): existing children are visited, and for each missing
    /// child the parent draws just the 2 x 2 tiles covering that quarter (missing children are either empty
    /// or past the finest level's range). Returns nodes with a 16-bit mask of the tiles to draw.
    static func select(_ meshes: [LodNodeKey: LodMeshNode], maxLevel: Int, camX: Double, camZ: Double, splitFactor: Double,
                       level0Radius: Double, zoom: Double = 1) -> [(LodMeshNode, UInt16)] {
        var out: [(LodMeshNode, UInt16)] = []
        // Tile masks for each quarter (tiles t = tz * 4 + tx; quarter (qx, qz) covers tx, tz in 2q ..< 2q + 2).
        func quarterMask(_ qx: Int, _ qz: Int) -> UInt16 {
            var m: UInt16 = 0
            for tz in (2 * qz)..<(2 * qz + 2) { for tx in (2 * qx)..<(2 * qx + 2) { m |= 1 << UInt16(tz * 4 + tx) } }
            return m
        }
        func visit(_ level: Int, _ nx: Int, _ nz: Int) {
            let node = meshes[LodNodeKey(level: level, x: nx, z: nz)]
            let size = Double(lodNodeVoxels << level)
            let x0 = Double(nx) * size, z0 = Double(nz) * size
            let dx = max(x0 - camX, 0, camX - (x0 + size)), dz = max(z0 - camZ, 0, camZ - (z0 + size))
            // Zoomed in (spyglass, low field of view), terrain looks as close as its distance over the zoom.
            let dist = (dx * dx + dz * dz).squareRoot() / zoom
            // Coarser nodes also split within the level-0 radius, so a large radius reaches level 0 through level 1.
            if level > 1 ? (dist < splitFactor * size / 2 || dist < level0Radius) : (level == 1 && dist < level0Radius) {
                var parentMask: UInt16 = 0
                for qz in 0...1 {
                    for qx in 0...1 {
                        let cx = 2 * nx + qx, cz = 2 * nz + qz
                        if meshes[LodNodeKey(level: level - 1, x: cx, z: cz)] != nil || hasDescendant(level - 1, cx, cz) {
                            visit(level - 1, cx, cz)
                        } else {
                            parentMask |= quarterMask(qx, qz)
                        }
                    }
                }
                if let node, parentMask != 0 { out.append((node, parentMask)) }
                return
            }
            if let node { out.append((node, 0xFFFF)) }
        }
        // A missing node may still have finer descendants (never happens today, but keep the recursion honest).
        func hasDescendant(_ level: Int, _ nx: Int, _ nz: Int) -> Bool { false }
        if lodTileSelection {
            // Per tile: a tile of level L hands its area to level L - 1 (the 2 x 2 child tiles covering it) when
            // it's within the finer level's reach, measured to the tile rather than the node.
            func visitTiles(_ level: Int, _ nx: Int, _ nz: Int, _ mask: UInt16) {
                let node = meshes[LodNodeKey(level: level, x: nx, z: nz)]
                let size = Double(lodNodeVoxels << level), tile = size / Double(lodTilesPerSide)
                let reach = level > 1 ? max(lodTileSplit * size / 2, level0Radius) : level0Radius
                var draw: UInt16 = 0
                var childMask = [UInt16](repeating: 0, count: 4)
                for t in 0..<16 where mask & (1 << UInt16(t)) != 0 {
                    let tx = t % 4, tz = t / 4
                    let x0 = Double(nx) * size + Double(tx) * tile, z0 = Double(nz) * size + Double(tz) * tile
                    let dx = max(x0 - camX, 0, camX - (x0 + tile)), dz = max(z0 - camZ, 0, camZ - (z0 + tile))
                    let q = (tz / 2) * 2 + tx / 2
                    if level >= 1 && (dx * dx + dz * dz).squareRoot() < reach
                        && meshes[LodNodeKey(level: level - 1, x: 2 * nx + tx / 2, z: 2 * nz + tz / 2)] != nil {
                        let cx = (tx % 2) * 2, cz = (tz % 2) * 2
                        for j in 0...1 { for i in 0...1 { childMask[q] |= 1 << UInt16((cz + j) * 4 + cx + i) } }
                    } else {
                        draw |= 1 << UInt16(t)
                    }
                }
                if let node, draw != 0 { out.append((node, draw)) }
                for q in 0..<4 where childMask[q] != 0 { visitTiles(level - 1, 2 * nx + q % 2, 2 * nz + q / 2, childMask[q]) }
            }
            for k in meshes.keys where k.level == maxLevel || meshes[LodNodeKey(level: k.level + 1, x: k.x >> 1, z: k.z >> 1)] == nil {
                visitTiles(k.level, k.x, k.z, 0xFFFF)
            }
            return out
        }
        // Roots: every node whose parent doesn't exist, not just the top level. Levels are built bottom-up,
        // so while the coarse levels are still building (18-25 s for 8 km), the finer ones already draw.
        for k in meshes.keys where k.level == maxLevel || meshes[LodNodeKey(level: k.level + 1, x: k.x >> 1, z: k.z >> 1)] == nil {
            visit(k.level, k.x, k.z)
        }
        return out
    }
}

@_cdecl("mmc_lod_open")
public func mmc_lod_open(_ worldDir: UnsafePointer<CChar>, _ far: Int32, _ centerX: Int32, _ centerZ: Int32) -> Int32 {
    let dir = String(cString: worldDir)
    guard let regionDir = Anvil.regionDirectory(URL(fileURLWithPath: dir)) else {
        log("LOD: no region directory under \(dir)")
        return 0
    }
    mmc_lod_close()
    return lodOpen(regionDir: regionDir, storeDir: nil, dimension: "minecraft:overworld", far: Int(far),
                   centerX: Int(centerX), centerZ: Int(centerZ)) > 0 ? 1 : 0
}

/// Opens the LOD with either source or both: `worldDir` is a single-player save ("" in multiplayer) and
/// `storeDir` is where live chunks are saved between sessions ("" to keep them in memory only).
@_cdecl("mmc_lod_open2")
public func mmc_lod_open2(_ worldDir: UnsafePointer<CChar>, _ storeDir: UnsafePointer<CChar>, _ far: Int32,
                          _ centerX: Int32, _ centerZ: Int32) -> Int32 {
    let dir = String(cString: worldDir), store = String(cString: storeDir)
    var regionDir: URL?
    if !dir.isEmpty {
        regionDir = Anvil.regionDirectory(URL(fileURLWithPath: dir))
        if regionDir == nil { log("LOD: no region directory under \(dir)") }
    }
    let storeURL = store.isEmpty ? nil : URL(fileURLWithPath: store)
    if regionDir == nil && storeURL == nil { return 0 }
    mmc_lod_close()
    return lodOpen(regionDir: regionDir, storeDir: storeURL, dimension: "minecraft:overworld", far: Int(far),
                   centerX: Int(centerX), centerZ: Int(centerZ)) > 0 ? 1 : 0
}

/// Opens the LOD of one dimension (e.g. minecraft:the_end) of a single-player save (`worldDir`, "" in
/// multiplayer) and/or a live-chunk store (`storeDir`, "" for none). A dimension opened before resumes where it
/// was, so coming back from the End is instant; the dimension the player left is paused, not dropped
/// (mmc_lod_close drops them all, for a different save or server). Returns the world's id, which the calls
/// that feed it (ingest, far terrain) pass back, or 0 if there's nothing to build from.
@_cdecl("mmc_lod_open3")
public func mmc_lod_open3(_ worldDir: UnsafePointer<CChar>, _ storeDir: UnsafePointer<CChar>, _ cacheDir: UnsafePointer<CChar>,
                          _ dimension: UnsafePointer<CChar>, _ far: Int32, _ centerX: Int32, _ centerZ: Int32) -> Int64 {
    let dir = String(cString: worldDir), store = String(cString: storeDir), dim = String(cString: dimension)
    let cache = String(cString: cacheDir)
    let regionDir = dir.isEmpty ? nil : Anvil.regionDirectory(URL(fileURLWithPath: dir), dimension: dim)
    let storeURL = store.isEmpty ? nil : URL(fileURLWithPath: store)
    if regionDir == nil && storeURL == nil { return 0 }
    let r = LodRenderer.shared
    r.lock.lock()
    if let w = r.worlds[dim], w.regionDir == regionDir, w.live.saveDir == storeURL {
        if r.world !== w { r.world?.setPaused(true) }
        r.world = w
        r.lastCamera = SIMD3(repeating: .nan)   // like a teleport: no fades from the other dimension
        r.lock.unlock()
        w.setCenter(x: Int(centerX), z: Int(centerZ), vanillaRadius: 0)
        w.setPaused(false)
        log("LOD: resuming \(dim) (world \(w.id))")
        return Int64(w.id)
    }
    r.lock.unlock()
    return Int64(lodOpen(regionDir: regionDir, storeDir: storeURL, cacheDir: cache.isEmpty ? nil : URL(fileURLWithPath: cache), dimension: dim,
                         far: Int(far), centerX: Int(centerX), centerZ: Int(centerZ)))
}

private func lodOpen(regionDir: URL?, storeDir: URL?, cacheDir: URL? = nil, dimension: String, far: Int, centerX: Int, centerZ: Int) -> Int {
    let r = LodRenderer.shared
    var maxLevel = 1
    while (lodNodeVoxels << maxLevel) < far && maxLevel < 8 { maxLevel += 1 }
    // Level-1 nodes exist far enough out for the level-2 nodes that split into them (within the split factor x 512
    // blocks; 1536 at the default 2) and for level 0's parents (METALMC_LOD0 can reach past 1.5 km).
    let fine = max(Int(768 * lodSplitFactor), lodLevel0Radius + 512)
    r.lock.lock()
    let id = r.nextWorldId
    r.nextWorldId += 1
    // 26.x's End has sky light (sky_light_color #ac60cd is what tints its end stone pink); the Nether has none.
    let w = LodWorld(id: id, dimension: dimension, floating: dimension == "minecraft:the_end",
                     hasSkyLight: dimension != "minecraft:the_nether", regionDir: regionDir, storeDir: storeDir, cacheDir: cacheDir,
                     maxLevel: maxLevel, fineRadius: fine, centerX: centerX, centerZ: centerZ)
    w.distance = far
    r.worlds[dimension]?.stop()
    r.world?.setPaused(true)
    r.worlds[dimension] = w
    r.world = w
    r.lastCamera = SIMD3(repeating: .nan)
    r.lock.unlock()
    log("LOD: streaming \(dimension) from \(regionDir?.path ?? "no region files") + live chunks\(storeDir.map { " (saved to \($0.path))" } ?? "") far=\(far) levels 1...\(maxLevel) around (\(centerX), \(centerZ)), world \(id)")
    w.start()
    return id
}

/// The world with this id (open, active or paused), for calls that feed a specific dimension.
func lodWorld(_ id: Int64) -> LodWorld? {
    let r = LodRenderer.shared
    r.lock.lock(); defer { r.lock.unlock() }
    if let w = r.world, w.id == Int(id) { return w }
    return r.worlds.values.first { $0.id == Int(id) }
}

/// Stops streaming and drops the LOD of every dimension (the player left the world).
@_cdecl("mmc_lod_close")
public func mmc_lod_close() {
    let r = LodRenderer.shared
    r.lock.lock()
    r.world?.stop()
    for w in r.worlds.values { w.stop() }
    r.world = nil
    r.worlds.removeAll()
    r.lock.unlock()
    LodSmartState.shared.reset()
}

/// Tells the LOD where the player is, so the finest level follows them.
@_cdecl("mmc_lod_center")
public func mmc_lod_center(_ x: Int32, _ z: Int32, _ vanillaRadius: Int32) {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    w?.setCenter(x: Int(x), z: Int(z), vanillaRadius: Int(vanillaRadius))
}

/// out: state (0 none, 1 building, 2 has nodes), nodes, quads, and 1 once the first full build has finished.
@_cdecl("mmc_lod_status")
public func mmc_lod_status(_ out: UnsafeMutablePointer<Int64>) {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { out[0] = 0; out[1] = 0; out[2] = 0; out[3] = 0; return }
    let snap = w.snapshot()
    out[0] = snap.meshes.isEmpty ? 1 : 2
    out[1] = Int64(snap.meshes.count)
    out[2] = Int64(snap.meshes.values.reduce(0) { $0 + $1.quadCount })
    out[3] = w.firstPassDone ? 1 : 0
}

/// Draws the LOD into the open render pass. p: proj[16], view[16] (column-major), fog color[4],
/// envStart, envEnd, rdStart, rdEnd, discardRadius, sky. cam: camera position (world, doubles).
/// Returns the number of draws. Leaves ctx.pipe nil so Java re-applies Minecraft's pipeline state.
@_cdecl("mmc_lod_draw")
public func mmc_lod_draw(_ p: UnsafePointer<Float>, _ cam: UnsafePointer<Double>) -> Int32 {
    let t0 = DispatchTime.now().uptimeNanoseconds
    defer {
        let dt = DispatchTime.now().uptimeNanoseconds - t0
        ctx.statLodNanos += dt
        ctx.hitchLodNanos += dt
    }
    let r = LodRenderer.shared
    // ExtPipe hook: an external pipeline's G-buffer pass has its own targets and GL's depth convention: no LOD there.
    guard let enc = ctx.pass, !ctx.scissorEmpty, !extPassActive else { return 0 }
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return 0 }
    let snap = w.snapshot()
    guard !snap.meshes.isEmpty else { return 0 }
    let useMesh = lodMeshOn
    r.lastFormats = (ctx.passColorFormats, ctx.passDepthFormat)
    guard let pipe = r.pipeline(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat, mesh: useMesh) else { return 0 }
    r.ensureColors()
    let cx = cam[0], cy = cam[1], cz = cam[2]
    // Zoom relative to vanilla's default 70-degree field of view (proj[1][1] = cot(35 degrees)), counted only when it's
    // clearly zoomed (the spyglass is about 10x; sprinting widens the view a little, which changes nothing).
    let fovZoom = Double(p[5]) / 1.4281
    let zoom = fovZoom > 1.25 && !lodNoZoom ? fovZoom : 1
    // METALMC_EXP=smartlod: per tile by projected error within a quad budget (LodSmart.swift).
    let chosen = lodSmart
        ? LodRenderer.selectSmartFrame(w, meshes: snap.meshes, generation: snap.generation, cam: SIMD3(cx, cy, cz),
                                       projView: lodMatrix(p, 0) * lodMatrix(p, 16), zoom: zoom)
        : LodRenderer.select(snap.meshes, maxLevel: w.maxLevel, camX: cx, camZ: cz, splitFactor: lodSplitFactor,
                             level0Radius: Double(lodLevel0Radius), zoom: zoom)
    guard !chosen.isEmpty else { return 0 }
    if let spec = lodDumpSpec, !lodDumpDone, cx >= spec[0] { lodDumpDone = true; lodDumpChosen(chosen, spec) }
    r.ensureIndexBuffer(quads: chosen.map { $0.0.quadCount }.max() ?? 1)
    guard let ib = r.indexBuffer else { return 0 }

    func mat(_ o: Int) -> simd_float4x4 {
        simd_float4x4(SIMD4(p[o], p[o + 1], p[o + 2], p[o + 3]), SIMD4(p[o + 4], p[o + 5], p[o + 6], p[o + 7]),
                      SIMD4(p[o + 8], p[o + 9], p[o + 10], p[o + 11]), SIMD4(p[o + 12], p[o + 13], p[o + 14], p[o + 15]))
    }
    var u = LodUniforms(proj: mat(0), view: mat(16), fogColor: SIMD4(p[32], p[33], p[34], p[35]),
                        envStart: p[36], envEnd: p[37], rdStart: p[38], rdEnd: p[39], discardRadius: p[40], sky: p[41])

    // Frustum side planes from proj * view (camera-relative). Points behind the camera fail them too.
    let m = u.proj * u.view
    let r0 = SIMD4(m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x)
    let r1 = SIMD4(m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y)
    let r3 = SIMD4(m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w)
    let planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1]
    func visible(_ lo: SIMD3<Float>, _ hi: SIMD3<Float>) -> Bool {
        for pl in planes {
            let v = SIMD3(pl.x >= 0 ? hi.x : lo.x, pl.y >= 0 ? hi.y : lo.y, pl.z >= 0 ? hi.z : lo.z)
            if pl.x * v.x + pl.y * v.y + pl.z * v.z + pl.w < 0 { return false }
        }
        return true
    }

    // Occlusion: tiles whose box was hidden in the newest completed test are skipped. Every candidate
    // tile's box is tested again this frame, so a tile that comes into view reappears within 2-3 frames.
    r.frame += 1
    let vis = lodOcclusion ? r.harvestAndAcquire() : nil
    let camera = SIMD3(cx, cy, cz)
    let jumped = simd_distance(camera, r.lastCamera)
    if !(jumped < 16) { r.jumpFrame = r.frame }   // teleports, respawns (and NaN on the first frame)
    r.lastCamera = camera
    let latest = r.latestResult >= r.jumpFrame ? r.latestResult : 0
    let boxOut = vis.map { $0.boxes.contents().bindMemory(to: SIMD4<Float>.self, capacity: 2 * LodVisSet.capacity) }
    let markOut = vis.map { $0.marks.contents().bindMemory(to: UInt32.self, capacity: LodVisSet.capacity) }

    // The seam bitmap: which chunk sections around the camera vanilla drew this frame.
    let H = lodSeamHalf, W = lodSeamWidth
    let csx = Int((cx / 16).rounded(.down)), csy = Int((cy / 16).rounded(.down)), csz = Int((cz / 16).rounded(.down))
    let bottomSection = lodWorldMinY >> 4
    u.camInSection = SIMD4(Float(cx - Double(csx * 16)), Float(cy - Double(csy * 16)), Float(cz - Double(csz * 16)), Float(H))
    u.seamInfo = SIMD4(Int32(csy - bottomSection), Int32(W), w.hasSkyLight ? 15 : 0, lodCullFrac ? 1 : 0)
    var seamBuffer: MTLBuffer?
    if !r.vanillaSections.isEmpty {
        while r.seamBuffers.count < 3, let b = ctx.device.makeBuffer(length: lodSeamWords * 4, options: [.storageModeShared]) {
            b.label = "MetalMC LOD seam"
            r.seamBuffers.append(b)
        }
        if r.seamBuffers.count == 3 {
            let b = r.seamBuffers[Int(r.frame % 3)]
            let words = b.contents().bindMemory(to: UInt32.self, capacity: lodSeamWords)
            for i in 0..<lodSeamWords { words[i] = 0 }
            for k in r.vanillaSections {
                // SectionPos.asLong: x in bits 42-63, z in bits 20-41, y in bits 0-19 (all signed).
                let ix = Int(k >> 42) - csx + H, iz = Int((k << 22) >> 42) - csz + H, iy = Int((k << 44) >> 44) - bottomSection
                guard ix >= 0, iz >= 0, ix < W, iz < W, iy >= 0, iy < 24 else { continue }
                let bit = (iy * W + iz) * W + ix
                words[bit >> 5] |= 1 << UInt32(bit & 31)
            }
            seamBuffer = b
        }
    }
    // Sections vanilla has compiled, as a bitmap like the seam's.
    let rd = r.vanillaDistance
    var anyCompiled = false
    if rd > 0 && !r.compiledSections.isEmpty {
        for i in 0..<lodSeamWords { r.compiledWords[i] = 0 }
        for k in r.compiledSections {
            let ix = Int(k >> 42) - csx + H, iz = Int((k << 22) >> 42) - csz + H, iy = Int((k << 44) >> 44) - bottomSection
            guard ix >= 0, iz >= 0, ix < W, iz < W, iy >= 0, iy < 24 else { continue }
            let bit = (iy * W + iz) * W + ix
            r.compiledWords[bit >> 5] |= 1 << UInt32(bit & 31)
        }
        anyCompiled = true
    }
    /// True if vanilla takes care of every chunk section tile `t` of `n` has quads in (edge skirts aside):
    /// compiled, within its horizontal view distance (ChunkTrackingView.isWithinDistance) and within its
    /// vertical range (render distance in sections above and below the camera). Vanilla draws those it can
    /// see, and nothing in the others can be seen, so the tile is skipped. Vanilla doesn't compile sections
    /// of only air, which is why the test is per section with LOD geometry rather than per box.
    func leftToVanilla(_ n: LodMeshNode, _ t: Int) -> Bool {
        guard anyCompiled, n.level <= 1 else { return false }
        let side = lodTileSections(n.level), words = lodTileSectionWords(n.level)
        let tileSX = (n.x0 >> 4) + (t % lodTilesPerSide) * side, tileSZ = (n.z0 >> 4) + (t / lodTilesPerSide) * side
        for wi in 0..<words {
            var bits = n.sectionMask[t * words + wi]
            while bits != 0 {
                let b = wi * 32 + bits.trailingZeroBitCount
                bits &= bits - 1
                let sy = bottomSection + b / (side * side), sz = tileSZ + (b / side) % side, sx = tileSX + b % side
                if abs(sy - csy) > rd { r.skipDebug[0] += 1; return false }
                let ix = sx - csx + H, iz = sz - csz + H
                if ix < 0 || iz < 0 || ix >= W || iz >= W { r.skipDebug[1] += 1; return false }
                let dx = max(0, abs(sx - csx) - 1), dz = max(0, abs(sz - csz) - 1)
                if dx * dx + dz * dz >= rd * rd { r.skipDebug[2] += 1; return false }
                let bit = ((sy - bottomSection) * W + iz) * W + ix
                if r.compiledWords[bit >> 5] & (1 << UInt32(bit & 31)) == 0 { r.skipDebug[3] += 1; return false }
            }
        }
        r.skipDebug[4] += 1
        return true
    }

    struct Draw {
        var node: LodMeshNode; var slot: Int; var first: Int; var count: Int; var seam: Bool; var water: Bool
        var fade: Float = 0; var fadeOut = false   // fade progress 0-1 for a tile in transition, 0 otherwise
    }
    var draws: [Draw] = []
    // Transitions: compare this frame's tiles with last frame's. Not on the first frame or right after a jump.
    var current: [ObjectIdentifier: (LodMeshNode, UInt16)] = [:]
    for (n, m) in chosen { current[ObjectIdentifier(n), default: (n, 0)].1 |= m }
    let fadeFrames = UInt64(lodFadeFrames)
    if lodFadeFrames > 0 && !r.lastSelection.isEmpty && r.frame > r.jumpFrame + 1 {
        for (id, (_, mask)) in current {
            let appeared = mask & ~(r.lastSelection[id]?.1 ?? 0)
            if appeared != 0 { r.fadeIn[id] = (appeared | (r.fadeIn[id]?.mask ?? 0), r.frame) }
        }
        for (id, (n, prev)) in r.lastSelection {
            let gone = prev & ~(current[id]?.1 ?? 0)
            if gone != 0 { r.fadeOut.append((n, gone, r.frame)) }
        }
    } else if r.frame <= r.jumpFrame + 1 {
        r.fadeIn.removeAll()
        r.fadeOut.removeAll()
    }
    r.lastSelection = current
    r.fadeIn = r.fadeIn.filter { r.frame < $0.value.start + fadeFrames }
    r.fadeOut.removeAll { r.frame >= $0.start + fadeFrames }
    var slots: [ObjectIdentifier: Int] = [:]
    var xforms = [SIMD4<Float>]()
    xforms.reserveCapacity(chosen.count)
    // Tiles overlapping vanilla's area (its render distance, as a square) take the seam variant; nothing is
    // skipped for being close any more, since vanilla picks sections by 3D distance.
    let vanillaHalf = u.discardRadius
    u.discardRadius = 0
    // Tiles of one node: culling, the occlusion test's boxes (not for fading-out tiles) and draw ranges.
    func addTiles(_ n: LodMeshNode, _ tileMask: UInt16, fadeOutStart: UInt64?) {
        let voxel = Float(1 << n.level)
        let nodeLo = SIMD3(Float(Double(n.x0) - cx), Float(Double(lodWorldMinY) - cy), Float(Double(n.z0) - cz))
        let nodeHi = nodeLo + SIMD3(Float(n.size), Float(lodWorldHeight), Float(n.size))
        if !visible(nodeLo, nodeHi) { return }
        let id = ObjectIdentifier(n)
        var slot = slots[id] ?? -1
        let tileSize = Float(lodTileVoxels) * voxel
        let fadeIn = fadeOutStart == nil ? r.fadeIn[id] : nil
        for t in 0..<(lodTilesPerSide * lodTilesPerSide) where tileMask & (1 << UInt16(t)) != 0 {
            var fade: Float = 0
            if let start = fadeOutStart {
                fade = Float(r.frame - start) / Float(fadeFrames)
            } else if let fi = fadeIn, fi.mask & (1 << UInt16(t)) != 0 {
                fade = max(0.001, Float(r.frame - fi.start) / Float(fadeFrames))
            }
            let yMin = n.tileY[2 * t], yMax = n.tileY[2 * t + 1]
            if yMin > yMax { continue }
            let tx = t % lodTilesPerSide, tz = t / lodTilesPerSide
            let lo = SIMD3(nodeLo.x + Float(tx) * tileSize, nodeLo.y + Float(yMin) * voxel, nodeLo.z + Float(tz) * tileSize)
            let hi = SIMD3(lo.x + tileSize, nodeLo.y + Float(yMax) * voxel, lo.z + tileSize)
            if !visible(lo, hi) { continue }
            let nearVanilla = lo.x < vanillaHalf && hi.x > -vanillaHalf && lo.z < vanillaHalf && hi.z > -vanillaHalf
            if nearVanilla && leftToVanilla(n, t) {
                r.coveredTiles += 1
                continue
            }
            // With nothing drawn by vanilla this frame (high above the terrain, or before its chunks load) there's
            // no bitmap and nothing to cut out: the plain variant.
            let seam = nearVanilla && seamBuffer != nil
            // Only box faces toward the camera are rasterized, so a camera inside a box would see nothing
            // of it: such tiles are left untested, which keeps them drawn.
            let inside = lo.x - voxel < 0 && hi.x + voxel > 0 && lo.y - voxel < 0 && hi.y + voxel > 0
                && lo.z - voxel < 0 && hi.z + voxel > 0
            if !inside, fadeOutStart == nil, let vis, let boxOut, let markOut, vis.slots.count < LodVisSet.capacity {
                // Box grown by one voxel so the tile's own surfaces never hide it.
                let i = vis.slots.count
                boxOut[2 * i] = SIMD4(lo - voxel, 0)
                boxOut[2 * i + 1] = SIMD4(hi + voxel, 0)
                markOut[i] = 0
                vis.slots.append((n, t))
            }
            if latest > 0 && !lodOccNoCull && n.tileTested[t] == latest && n.tileVisible[t] != latest { continue }
            // Face buckets that can face the camera (camera past the tile's nearest plane on that axis).
            let faceVisible = [0 > lo.x, 0 < hi.x, 0 > lo.y, 0 < hi.y, 0 > lo.z, 0 < hi.z]
            if slot < 0 {
                slot = xforms.count
                slots[id] = slot
                xforms.append(SIMD4(nodeLo.x, nodeLo.y, nodeLo.z, voxel))
            }
            // Sub-tiles the horizon test left visible (all of them for fading-out tiles, or with the test off).
            var stMask: UInt16 = fadeOutStart == nil ? (horizonVis[id]?[t] ?? 0xFFFF) : 0xFFFF
            if lodSubtileFrustum {
                // Sub-tiles whose quads lie entirely outside the view (tiles at the screen's edges).
                for st in 0..<lodSubtilesPerTile where stMask & (1 << UInt16(st)) != 0 {
                    let si = 8 * (t * lodSubtilesPerTile + st)
                    if n.subtiles[si] > n.subtiles[si + 1] { continue }
                    let slo = nodeLo + SIMD3(Float(n.subtiles[si]), Float(n.subtiles[si + 2]), Float(n.subtiles[si + 4])) * voxel
                    let shi = nodeLo + SIMD3(Float(n.subtiles[si + 1]), Float(n.subtiles[si + 3]), Float(n.subtiles[si + 5])) * voxel
                    if !visible(slo, shi) { stMask &= ~(UInt16(1) << UInt16(st)) }
                }
            }
            if stMask == 0 { r.horizonCulledTiles += 1; continue }
            var lastEnd = -1, lastWater = false
            func emit(_ first: Int, _ end: Int, _ water: Bool) {
                if first == end { return }
                if first == lastEnd && water == lastWater, var d = draws.popLast() {
                    d.count += end - first
                    draws.append(d)
                } else {
                    draws.append(Draw(node: n, slot: slot, first: first, count: end - first, seam: seam, water: water, fade: fade,
                                      fadeOut: fadeOutStart != nil))
                }
                lastEnd = end
                lastWater = water
            }
            // Runs of visible sub-tiles of bucket k (empty ones bridge runs). Contiguous runs merge, also across
            // buckets, so a fully visible tile's facing buckets are one draw as before.
            func bucket(_ k: Int, _ water: Bool) {
                var st = 0
                while st < lodSubtilesPerTile {
                    let b = lodBucketIndex(t, k, st)
                    if stMask & (1 << UInt16(st)) == 0 || n.start[b + 1] == n.start[b] { st += 1; continue }
                    var e = st + 1
                    while e < lodSubtilesPerTile {
                        let be = lodBucketIndex(t, k, e)
                        if stMask & (1 << UInt16(e)) == 0 && n.start[be + 1] != n.start[be] { break }
                        e += 1
                    }
                    emit(n.start[b], n.start[lodBucketIndex(t, k, e - 1) + 1], water)
                    st = e
                }
            }
            for f in 0..<6 where faceVisible[f] { bucket(f, false) }
            // Tile-edge skirts toward neighbor tiles of this node that it doesn't draw (a finer level does).
            for (e, (dx, dz, face)) in [(-1, 0, 1), (1, 0, 0), (0, -1, 5), (0, 1, 4)].enumerated() {
                let ntx = tx + dx, ntz = tz + dz
                if ntx < 0 || ntz < 0 || ntx >= lodTilesPerSide || ntz >= lodTilesPerSide || !faceVisible[face] { continue }
                if tileMask & (1 << UInt16(ntz * lodTilesPerSide + ntx)) != 0 { continue }
                bucket(12 + e, false)
            }
            for f in 0..<6 where faceVisible[f] { bucket(6 + f, true) }
        }
    }
    let horizonVis: [ObjectIdentifier: [UInt16]] = lodHorizon
        ? lodHorizonTest(chosen, cx: cx, cy: cy, cz: cz, fadingIn: { id, t in (r.fadeIn[id]?.mask ?? 0) & (1 << UInt16(t)) != 0 },
                         inView: visible)
        : [:]
    // Far field: its levels are ray-marched instead of drawn as quads, once its rings are filled.
    let farOn = lodFarFieldLevel > 0 && !w.floating
        && FarField.shared.prepare(meshes: snap.meshes, generation: snap.generation, chosen: chosen, maxLevel: w.maxLevel, cx: cx, cz: cz)
    var farNearest = Double.infinity   // horizontal distance to the nearest tile the far field draws
    if farOn {
        for (n, mask) in chosen where n.level >= lodFarFieldLevel {
            let tb = Double(lodTileVoxels << n.level)
            for t in 0..<16 where mask & (1 << UInt16(t)) != 0 {
                let x0 = Double(n.x0) + Double(t % 4) * tb, z0 = Double(n.z0) + Double(t / 4) * tb
                let dx = max(x0 - cx, 0, cx - (x0 + tb)), dz = max(z0 - cz, 0, cz - (z0 + tb))
                farNearest = min(farNearest, (dx * dx + dz * dz).squareRoot())
            }
        }
    }
    for (n, tileMask) in chosen where !(farOn && n.level >= lodFarFieldLevel) { addTiles(n, tileMask, fadeOutStart: nil) }
    if lodRtShadows { RtShadows.shared.chosen = chosen }
    for f in r.fadeOut where !(farOn && f.node.level >= lodFarFieldLevel) { addTiles(f.node, f.mask, fadeOutStart: f.start) }
    if r.frame % 1000 == 0 {
        var perLevel = [Int](repeating: 0, count: 9), quadsPerLevel = [Int](repeating: 0, count: 9)
        for c in chosen { perLevel[min(8, c.0.level)] += 1 }
        for d in draws { quadsPerLevel[min(8, d.node.level)] += d.count }
        log("LOD: frame \(r.frame): \(draws.count) draws, \(draws.reduce(0) { $0 + $1.count }) quads, chosen per level \(perLevel), K quads per level \(quadsPerLevel.map { $0 / 1000 }), \(chosen.filter { $0.0.level == 0 }.count) level-0 nodes, \(r.coveredTiles) tiles covered by vanilla in 1000 frames; vanilla drew \(r.vanillaSections.count) sections, compiled \(r.compiledSections.count), distance \(r.vanillaDistance), skip checks \(r.skipDebug)")
        r.skipDebug = [0, 0, 0, 0, 0]
        r.coveredTiles = 0
        if let s = LodQuadCull.shared.takeStats() { log("LOD quad culling: \(s)") }
    }
    if lodCullStats && r.frame % 600 == 300 {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let c = lodCountCullable(draws.filter { $0.fade == 0 }.map { ($0.node, $0.slot, $0.first, $0.count) }, xforms: xforms,
                                 projView: u.proj * u.view, width: ctx.passWidth, height: ctx.passHeight)
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        let rows = c.enumerated().filter { $0.element[0] > 0 }.map { "level \($0.offset): \($0.element[0]) quads, \($0.element[1]) facing away, \($0.element[2]) off screen, \($0.element[3]) too thin" }
        log("LOD cull stats (frame \(r.frame), \(String(format: "%.0f", ms)) ms): " + rows.joined(separator: "; "))
    }
    if lodCullStats && r.frame % 600 == 300 {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let c = lodCountCullable(draws.filter { $0.fade == 0 }.map { ($0.node, $0.slot, $0.first, $0.count) }, xforms: xforms,
                                 projView: u.proj * u.view, width: ctx.passWidth, height: ctx.passHeight)
        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        let rows = c.enumerated().filter { $0.element[0] > 0 }.map { "level \($0.offset): \($0.element[0]) quads, \($0.element[1]) facing away, \($0.element[2]) off screen, \($0.element[3]) too thin" }
        log("LOD cull stats (frame \(r.frame), \(String(format: "%.0f", ms)) ms): " + rows.joined(separator: "; "))
    }
    let testBoxes = vis.map { !$0.slots.isEmpty } ?? false
    guard !draws.isEmpty || testBoxes || farOn else { return 0 }

    enc.setRenderPipelineState(pipe)
    enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: true))
    enc.setCullMode(.back)
    enc.setTriangleFillMode(.fill)
    enc.setDepthBias(0, slopeScale: 0, clamp: 0)
    // Texture detail resources live at slots vanilla doesn't use (texture 30, sampler 15, buffer 20).
    r.ensureTextureDefaults()
    if !lodFlat, r.atlas != nil, r.spriteBuffer != nil {
        u.camFrac = SIMD4(Float(cx - cx.rounded(.down)), Float(cy - cy.rounded(.down)), Float(cz - cz.rounded(.down)), 1)
    }
    let seamPipe = seamBuffer == nil ? nil : r.pipeline(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat, seam: true, mesh: useMesh)
    let seamWaterPipe = seamBuffer == nil ? nil : r.pipeline(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat, seam: true, water: true, mesh: useMesh)
    let waterPipe = r.pipeline(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat, water: true, mesh: useMesh)
    let anyFade = draws.contains { $0.fade > 0 }
    let fadePipe = anyFade ? r.pipeline(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat, mesh: useMesh, fade: true) : nil
    let fadeWaterPipe = anyFade ? r.pipeline(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat, water: true, mesh: useMesh, fade: true) : nil
    if r.lightSampler == nil {
        let d = MTLSamplerDescriptor()
        d.minFilter = .nearest
        d.magFilter = .nearest
        d.sAddressMode = .clampToEdge
        d.tAddressMode = .clampToEdge
        r.lightSampler = ctx.device.makeSamplerState(descriptor: d)
    }
    u.lightmapOn = r.lightmap != nil && !lodNoLightmap ? 1 : 0
    enc.setVertexTexture(r.lightmap ?? r.dummyTexture, index: 29)
    enc.setVertexSamplerState(r.lightSampler, index: 14)
    if litEnabled {
        // Lit mode: the fragment shader lights the unlit vertex colors itself (lodLitK).
        enc.setFragmentTexture(r.lightmap ?? r.dummyTexture, index: 29)
        enc.setFragmentSamplerState(r.lightSampler, index: 14)
    }
    if useMesh {
        enc.setMeshTexture(r.lightmap ?? r.dummyTexture, index: 29)
        enc.setMeshSamplerState(r.lightSampler, index: 14)
        enc.setMeshBytes(&u, length: MemoryLayout<LodUniforms>.stride, index: 19)
        enc.setMeshBuffer(r.colorBuffer!, offset: 0, index: 21)
    }
    enc.setFragmentTexture(r.atlas ?? r.dummyTexture, index: 30)
    enc.setFragmentSamplerState(r.atlasSampler, index: 15)
    enc.setFragmentBuffer(r.spriteBuffer ?? r.colorBuffer, offset: 0, index: 20)
    enc.setVertexBytes(&u, length: MemoryLayout<LodUniforms>.stride, index: 19)
    enc.setFragmentBytes(&u, length: MemoryLayout<LodUniforms>.stride, index: 19)
    if xforms.count * 16 <= 4096 {
        enc.setVertexBytes(xforms, length: xforms.count * 16, index: 20)
        if useMesh { enc.setMeshBytes(xforms, length: xforms.count * 16, index: 20) }
    } else if let xb = ctx.device.makeBuffer(bytes: xforms, length: xforms.count * 16, options: [.storageModeShared]) {
        enc.setVertexBuffer(xb, offset: 0, index: 20)
        if useMesh { enc.setMeshBuffer(xb, offset: 0, index: 20) }
    }
    enc.setVertexBuffer(r.colorBuffer!, offset: 0, index: 21)
    var bound: ObjectIdentifier?
    ctx.statLodDraws += draws.count
    // Opaque quads first (plain tiles, then the seam tiles with the discarding variant), then water over them
    // with blending and no depth writes, so it never hides what's under it from later tests.
    // Fading tiles go last in each group, with the dithering variant.
    let ordered = draws.filter { !$0.water && !$0.seam && $0.fade == 0 } + draws.filter { !$0.water && $0.seam && $0.fade == 0 }
        + draws.filter { !$0.water && $0.fade > 0 }
        + draws.filter { $0.water && !$0.seam && $0.fade == 0 } + draws.filter { $0.water && $0.seam && $0.fade == 0 }
        + draws.filter { $0.water && $0.fade > 0 }
    // Quad culling (lodQuadCullOn): the culling pass for this frame's draws, in the order they're drawn (nil: drawn
    // directly, e.g. while its pipeline compiles).
    let culled = useMesh || lodSkipQuadsOn || !lodQuadCullOn ? nil
        : LodQuadCull.shared.encode(ordered.map { ($0.node, $0.first, $0.count, $0.slot) }, xforms: xforms, viewProj: u.proj * u.view,
                                    width: ctx.passWidth, height: ctx.passHeight)
    var culledBound = false             // the culling pass's lists are bound for the vertex stage
    var vertexNode: ObjectIdentifier?   // or the buffers of this node
    // The occlusion test's boxes go after the opaque LOD and before the water, which (with ray-traced shadows on)
    // writes depth: the test must not see water surfaces as occluders of the floors under them.
    var boxesDone = false
    func runBoxes() {
        boxesDone = true
        if let vis, testBoxes, let boxPipe = r.pipeline(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat, box: true),
           let cb = ctx.cb {
            enc.setRenderPipelineState(boxPipe)
            enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: false))
            enc.setCullMode(.back)   // front faces only: half the fragments of drawing every face
            enc.setVertexBuffer(vis.boxes, offset: 0, index: 22)
            enc.setFragmentBuffer(vis.marks, offset: 0, index: 23)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 36, instanceCount: vis.slots.count)
            vis.frame = r.frame
            vis.pending = true
            cb.addCompletedHandler { _ in
                r.visLock.lock(); vis.done = true; r.visLock.unlock()
            }
        } else {
            vis?.slots.removeAll(keepingCapacity: true)
        }
    }
    var farDone = !farOn
    func drawFar() {
        farDone = true
        if farNearest.isFinite, !lodSkipFar, let colors = r.colorBuffer {
            FarField.shared.draw(enc, u: u, colors: colors, lightmap: r.lightmap ?? r.dummyTexture!, lightSampler: r.lightSampler,
                                 cy: cy, nearest: farNearest)
        }
    }
    var state = (seam: false, water: false, fade: false)
    for (di, d) in ordered.enumerated() {
        if d.water && !boxesDone {
            if !farDone { drawFar() }
            runBoxes()
            bound = nil
            culledBound = false
            vertexNode = nil
            state = (seam: false, water: false, fade: false)
        }
        let fading = d.fade > 0
        if d.seam != state.seam || d.water != state.water || fading != state.fade {
            let p = fading ? (d.water ? fadeWaterPipe : fadePipe) : (d.water ? (d.seam ? seamWaterPipe : waterPipe) : (d.seam ? seamPipe : pipe))
            guard let p else { continue }   // a pipeline that failed to build: skip its draws, not everything after them
            if d.water && !state.water {
                // Water writes depth only for the ray-traced shadows (its surface, not the floor under it, receives them)
                // and in lit mode (the relight leaves the floor under it alone: its depth no longer matches).
                enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: lodRtShadows || litEnabled))
                u.alpha = lodOpaqueWater ? 1 : lodWaterAlpha
                enc.setFragmentBytes(&u, length: MemoryLayout<LodUniforms>.stride, index: 19)
            }
            enc.setRenderPipelineState(p)
            if d.seam || fading, let seamBuffer { enc.setFragmentBuffer(seamBuffer, offset: 0, index: 21) }
            state = (d.seam, d.water, fading)
        }
        if fading {
            var params = SIMD4<Float>(min(1, d.fade), d.fadeOut ? 1 : 0, seamBuffer != nil ? 1 : 0, 0)
            enc.setFragmentBytes(&params, length: 16, index: 24)
        }
        ctx.statLodQuads += d.count
        let id = ObjectIdentifier(d.node)
        if bound != id {
            if useMesh {
                enc.setMeshBuffer(d.node.buffer, offset: 0, index: 18)
                enc.setMeshBuffer(d.node.aoOffsets, offset: 0, index: 22)
            }
            enc.setFragmentBuffer(d.node.ao, offset: 0, index: 22)
            bound = id
        }
        if lodSkipQuadsOn {
            continue
        } else if useMesh {
            var md = SIMD4<UInt32>(UInt32(d.first), UInt32(d.count), UInt32(d.slot), 0)
            enc.setMeshBytes(&md, length: 16, index: 17)
            enc.drawMeshThreadgroups(MTLSize(width: (d.count + lodMeshQuads - 1) / lodMeshQuads, height: 1, depth: 1),
                                     threadsPerObjectThreadgroup: MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerMeshThreadgroup: MTLSize(width: lodMeshQuads, height: 1, depth: 1))
        } else if let culled, !culled.jobs[di].isEmpty, let quads = culled.frame.quads, let ao = culled.frame.ao, let args = culled.frame.args {
            // The draw's surviving quads, as consecutive indirect draws (one per job) from the frame's lists.
            if !culledBound {
                enc.setVertexBuffer(quads, offset: 0, index: 18)
                enc.setVertexBuffer(ao, offset: 0, index: 22)
                culledBound = true
                vertexNode = nil
            }
            for j in culled.jobs[di] {
                enc.drawIndexedPrimitives(type: .triangle, indexType: .uint32, indexBuffer: ib, indexBufferOffset: 0,
                                          indirectBuffer: args, indirectBufferOffset: 20 * j)
            }
        } else {
            if vertexNode != id {
                enc.setVertexBuffer(d.node.buffer, offset: 0, index: 18)
                enc.setVertexBuffer(d.node.aoOffsets, offset: 0, index: 22)
                vertexNode = id
                culledBound = false
            }
            enc.drawIndexedPrimitives(type: .triangle, indexCount: d.count * 6, indexType: .uint32, indexBuffer: ib,
                                      indexBufferOffset: 0, instanceCount: 1, baseVertex: 4 * d.first, baseInstance: d.slot)
        }
    }
    if lodQuadVis && r.frame % 240 == 0 && !useMesh {
        lodVisSlot = slots
        lodMeasureVisibility(enc, ordered.filter { !$0.water && $0.fade == 0 }.map { ($0.node, $0.first, $0.count) }, ib)
    }
    if !farDone { drawFar() }
    if !boxesDone { runBoxes() }
    // Minecraft's pipeline, depth, cull and bias state must be re-applied by the next setPipeline.
    ctx.pipe = nil
    ctx.boundPipeState = nil
    return Int32(draws.count)
}

/// Name of material `index`'s top (top != 0) or side texture under block/ in the atlas, written to `buf`.
/// Returns its length, 0 for none (air), or -1 past the last material.
@_cdecl("mmc_lod_sprite_name")
public func mmc_lod_sprite_name(_ index: Int32, _ top: Int32, _ buf: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    guard index >= 0, Int(index) < lodMaterialSprites.count else { return -1 }
    let s = lodMaterialSprites[Int(index)]
    let bytes = Array((top != 0 ? s.top : s.side).utf8.prefix(Int(len) - 1))
    for (i, b) in bytes.enumerated() { buf[i] = CChar(bitPattern: b) }
    buf[bytes.count] = 0
    return Int32(bytes.count)
}

/// Sets Minecraft's block atlas (a texture view handle) and the sprite rectangles for each material:
/// rects holds count x 8 floats (top u0 v0 u1 v1, side u0 v0 u1 v1). Render thread.
/// The block atlas and, per LOD material, its top and side textures' UV rects (8 floats each). `means`, if not null:
/// those textures' mean colors in the loaded resource pack (6 floats: top RGB, side RGB, opaque texels, untinted).
/// Their ratio to vanilla's means scales the LOD's colors and texture detail, so a pack's recolored blocks carry
/// over to far terrain.
@_cdecl("mmc_lod_set_atlas")
public func mmc_lod_set_atlas(_ view: Int64, _ rects: UnsafePointer<Float>, _ means: UnsafePointer<Float>?, _ count: Int32) {
    let r = LodRenderer.shared
    let n = Int(count)
    var table = [SIMD4<Float>](repeating: .zero, count: max(1, n) * 3)
    var ratio = [SIMD3<Float>](repeating: SIMD3(1, 1, 1), count: max(1, n) * 2)   // per row: top, side
    @inline(__always) func luma(_ c: SIMD3<Float>) -> Float { 0.2126 * c.x + 0.7152 * c.y + 0.0722 * c.z }
    for i in 0..<n {
        table[3 * i] = SIMD4(rects[8 * i], rects[8 * i + 1], rects[8 * i + 2], rects[8 * i + 3])
        table[3 * i + 1] = SIMD4(rects[8 * i + 4], rects[8 * i + 5], rects[8 * i + 6], rects[8 * i + 7])
        let s = i < lodMaterialSprites.count ? lodMaterialSprites[i]
            : LodSprite(top: "", side: "", topLuma: 1, sideLuma: 1, topMean: SIMD3(1, 1, 1), sideMean: SIMD3(1, 1, 1))
        var topLuma = s.topLuma, sideLuma = s.sideLuma
        if let means, !s.top.isEmpty {
            for (k, base) in [s.topMean, s.sideMean].enumerated() {
                let cur = SIMD3(means[6 * i + 3 * k], means[6 * i + 3 * k + 1], means[6 * i + 3 * k + 2])
                guard cur.x >= 0, luma(base) > 0.01 else { continue }
                // Per channel where the vanilla texture has some of it, else by brightness; within 0.2x-5x.
                let l = min(5, max(0.2, luma(cur) / luma(base)))
                var q = SIMD3<Float>(l, l, l)
                for ch in 0..<3 where base[ch] > 0.03 { q[ch] = min(5, max(0.2, cur[ch] / base[ch])) }
                ratio[2 * i + k] = q
            }
            topLuma *= min(5, max(0.2, luma(ratio[2 * i] * s.topMean) / max(0.001, luma(s.topMean))))
            sideLuma *= min(5, max(0.2, luma(ratio[2 * i + 1] * s.sideMean) / max(0.001, luma(s.sideMean))))
        }
        table[3 * i + 2] = SIMD4(topLuma, sideLuma, 0, 0)
    }
    // Color table scale: a material's top faces by its top texture, sides by its side texture, bottoms by the top;
    // biome-tinted grass, leaves and water by their base textures.
    var pack = [SIMD3<Float>](repeating: SIMD3(1, 1, 1), count: 256 * 3)
    for m in 0..<min(n, 256) where m < 64 || m >= 160 {   // 64-159 are the tinted ids, set below
        pack[3 * m] = ratio[2 * m]; pack[3 * m + 1] = ratio[2 * m + 1]; pack[3 * m + 2] = ratio[2 * m]
    }
    func row(_ m: Mat, side: Bool = false) -> SIMD3<Float> { Int(m.rawValue) < n ? ratio[2 * Int(m.rawValue) + (side ? 1 : 0)] : SIMD3(1, 1, 1) }
    for t in 0..<32 {
        let g = Int(lodGrassBase) + t, l = Int(lodLeavesBase) + t, w = Int(lodWaterBase) + t
        pack[3 * g] = row(.grass); pack[3 * g + 1] = row(.dirt); pack[3 * g + 2] = row(.dirt)
        for f in 0..<3 { pack[3 * l + f] = row(.leaves); pack[3 * w + f] = row(.water) }
    }
    r.packRatio = pack
    r.colorBuffer = nil   // rebuilt with the new scale on the next draw
    if means != nil {
        let all = ratio.flatMap { [$0.x, $0.y, $0.z] }
        let changed = (0..<n).filter { i in simd_reduce_max(simd_abs(ratio[2 * i] - 1)) > 0.03 || simd_reduce_max(simd_abs(ratio[2 * i + 1] - 1)) > 0.03 }
        log("LOD: resource pack colors: ratios \(String(format: "%.3f", all.min() ?? 1))...\(String(format: "%.3f", all.max() ?? 1)), \(changed.count) materials changed \(changed.prefix(8).map { lodMaterialSprites[$0].top })")
    }
    r.spriteBuffer = ctx.device.makeBuffer(bytes: table, length: table.count * 16, options: [.storageModeShared])
    r.atlas = view == 0 ? nil : (from(view) as TextureBox).texture
    if let a = r.atlas { log("LOD: texture detail from the block atlas \(a.width)x\(a.height), \(a.mipmapLevelCount) mips, \(n) materials") }
}

/// Tint class `index`'s representative biome (e.g. minecraft:forest) into `out` (UTF-8, NUL-terminated); returns its
/// length, or -1 past the last class.
@_cdecl("mmc_lod_tint_biome")
public func mmc_lod_tint_biome(_ index: Int32, _ out: UnsafeMutablePointer<CChar>, _ capacity: Int32) -> Int32 {
    guard index >= 0, Int(index) < lodTintBiomes.count else { return -1 }
    let bytes = Array(lodTintBiomes[Int(index)].utf8)
    guard bytes.count < Int(capacity) else { return -1 }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}

/// Grass, foliage and water colors (0xRRGGBB) of each tint class's biome, as the game computes them (resource packs'
/// color maps included). Render thread.
@_cdecl("mmc_lod_set_tints")
public func mmc_lod_set_tints(_ colors: UnsafePointer<UInt32>, _ count: Int32) {
    let r = LodRenderer.shared
    var t = lodTints
    var changed = 0
    for i in 0..<min(Int(count), t.count) where colors[3 * i] != UInt32.max {   // max: biome missing, keep the table's
        let n = LodTint(grass: colors[3 * i] & 0xFFFFFF, foliage: colors[3 * i + 1] & 0xFFFFFF, water: colors[3 * i + 2] & 0xFFFFFF)
        if n.grass != t[i].grass || n.foliage != t[i].foliage || n.water != t[i].water {
            changed += 1
            log(String(format: "LOD: tint %@: grass %06X -> %06X, foliage %06X -> %06X, water %06X -> %06X", lodTintBiomes[i],
                       t[i].grass, n.grass, t[i].foliage, n.foliage, t[i].water, n.water))
        }
        t[i] = n
    }
    r.tints = t
    r.colorBuffer = nil
    if changed > 0 { log("LOD: biome colors from the game: \(changed) of \(t.count) tint classes differ from the built-in table") }
}

/// Vanilla's lightmap (a texture view handle; 0 for none), for lighting LOD terrain like vanilla's. Render thread.
@_cdecl("mmc_lod_set_lightmap")
public func mmc_lod_set_lightmap(_ view: Int64) {
    LodRenderer.shared.lightmap = view == 0 ? nil : (from(view) as TextureBox).texture
}

/// The chunk sections vanilla drew this frame (SectionPos.asLong keys, compiled sections only), for the LOD's
/// seam: LOD pixels inside them are dropped. Call before mmc_lod_draw each frame.
@_cdecl("mmc_lod_set_vanilla")
public func mmc_lod_set_vanilla(_ keys: UnsafePointer<Int64>, _ count: Int32) {
    LodRenderer.shared.vanillaSections = Array(UnsafeBufferPointer(start: keys, count: Int(count)))
}

/// Offline A/B in one process (tools/litflow.swift): the mesh path on (1) or off (0) and quad culling on or off, in place
/// of METALMC_EXP's meshshader and quadcull; -1 leaves one as it is. flags: 1 doesn't draw the LOD's quads (lodskip),
/// 2 has the culling pass keep every quad. Returns 1.
@_cdecl("mmc_debug_lod_paths")
public func mmc_debug_lod_paths(_ mesh: Int32, _ cull: Int32, _ flags: Int32) -> Int32 {
    if mesh >= 0 { lodMeshOn = mesh != 0 }
    if cull >= 0 { lodQuadCullOn = cull != 0 }
    lodSkipQuadsOn = lodSkipQuads || flags & 1 != 0
    lodCullKeepAll = flags & 2 != 0
    return 1
}

/// Debug: makes every LOD pipeline for the formats of the last pass the LOD drew into (vertex and mesh paths; plain, seam,
/// water, fade and their mixes; the occlusion boxes) and the culling pass, waiting up to 20 s for the background warm-up
/// first. Returns how many failed (logged as "LOD pipeline failed"), or -1 if the LOD hasn't drawn yet.
@_cdecl("mmc_debug_lod_pipelines")
public func mmc_debug_lod_pipelines() -> Int32 {
    let r = LodRenderer.shared
    guard let (colors, depth) = r.lastFormats else { return -1 }
    let t0 = Date()
    while r.pipeline(colorFormats: colors, depth: depth) == nil && Date().timeIntervalSince(t0) < 20 { Thread.sleep(forTimeInterval: 0.05) }
    var failed = 0
    for mesh in [false, true] {
        for (box, seam, water, fade) in [(false, false, false, false), (false, true, false, false), (false, false, true, false),
                                         (false, true, true, false), (false, false, false, true), (false, false, true, true),
                                         (true, false, false, false)] where !(box && mesh) {
            if r.pipeline(colorFormats: colors, depth: depth, box: box, seam: seam, water: water, mesh: mesh, fade: fade) == nil {
                log("LOD pipeline check: failed: mesh \(mesh), box \(box), seam \(seam), water \(water), fade \(fade)")
                failed += 1
            }
        }
    }
    if LodQuadCull.shared.pipeline(compile: true) == nil { failed += 1 }
    return Int32(failed)
}

/// The vertex path's last culling pass, waited for: out = its GPU time (ms), quads submitted to it, quads it kept, jobs,
/// and once its frame completed (call after waiting for that), the frame's GPU span from the earlier of the two command
/// buffers' starts to the frame's end (ms), and the frame's own GPU time (ms). Returns 0 if there was none since the last
/// call.
@_cdecl("mmc_debug_lod_quadcull_last")
public func mmc_debug_lod_quadcull_last(_ out: UnsafeMutablePointer<Double>) -> Int32 {
    let q = LodQuadCull.shared
    q.lock.lock(); let f = q.last; q.last = nil; q.lock.unlock()
    guard let f, let cb = f.cb, let args = f.args else { return 0 }
    cb.waitUntilCompleted()
    // The frame's completion handler may run a moment after its waiter wakes.
    for _ in 0..<1000 {
        q.lock.lock(); let done = f.pending == 0; q.lock.unlock()
        if done { break }
        Thread.sleep(forTimeInterval: 0.0005)
    }
    let a = args.contents().bindMemory(to: UInt32.self, capacity: 5 * f.jobCount)
    var kept = 0
    for j in 0..<f.jobCount { kept += Int(a[5 * j]) / 6 }
    q.lock.lock(); let ct = f.cullTimes, ft = f.frameTimes; q.lock.unlock()
    out[0] = (cb.gpuEndTime - cb.gpuStartTime) * 1000
    out[1] = Double(f.quadsIn)
    out[2] = Double(kept)
    out[3] = Double(f.jobCount)
    out[4] = ft.end > 0 ? (ft.end - min(ct.start, ft.start)) * 1000 : -1
    out[5] = ft.end > 0 ? (ft.end - ft.start) * 1000 : -1
    return 1
}

/// Every section vanilla has compiled in its view area (SectionPos.asLong keys) and its render distance in
/// chunks. LOD tiles made only of such sections are skipped. Call before mmc_lod_draw each frame.
@_cdecl("mmc_lod_set_compiled")
public func mmc_lod_set_compiled(_ keys: UnsafePointer<Int64>, _ count: Int32, _ renderDistance: Int32) {
    let r = LodRenderer.shared
    r.compiledSections = Array(UnsafeBufferPointer(start: keys, count: Int(count)))
    r.vanillaDistance = Int(renderDistance)
}

/// Debug: the active world's nodes as (level, x, z, quads) quadruples, at most `max`. Returns the count.
@_cdecl("mmc_debug_lod_nodes")
public func mmc_debug_lod_nodes(_ out: UnsafeMutablePointer<Int64>, _ max: Int32) -> Int32 {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return 0 }
    var i = 0
    for (k, n) in w.snapshot().meshes.sorted(by: { ($0.key.level, $0.key.x, $0.key.z) < ($1.key.level, $1.key.x, $1.key.z) }) where i < Int(max) {
        out[4 * i] = Int64(k.level); out[4 * i + 1] = Int64(k.x); out[4 * i + 2] = Int64(k.z); out[4 * i + 3] = Int64(n.quadCount)
        i += 1
    }
    return Int32(i)
}

/// Debug: runs the quadtree selection for a camera at (x, z) on the active world. Writes (level, x, z, tile
/// mask, quads) per chosen node, at most `max`; returns the count.
@_cdecl("mmc_debug_lod_select")
public func mmc_debug_lod_select(_ camX: Double, _ camZ: Double, _ out: UnsafeMutablePointer<Int64>, _ max: Int32) -> Int32 {
    mmc_debug_lod_select2(camX, camZ, lodSplitFactor, out, max)
}

/// Debug: mmc_debug_lod_select with a given split factor. Quads count only the chosen tiles.
@_cdecl("mmc_debug_lod_select2")
public func mmc_debug_lod_select2(_ camX: Double, _ camZ: Double, _ splitFactor: Double, _ out: UnsafeMutablePointer<Int64>, _ max: Int32) -> Int32 {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return 0 }
    let snap = w.snapshot()
    let chosen = LodRenderer.select(snap.meshes, maxLevel: w.maxLevel, camX: camX, camZ: camZ, splitFactor: splitFactor,
                                    level0Radius: Double(lodLevel0Radius))
    var i = 0
    for (n, mask) in chosen where i < Int(max) {
        out[5 * i] = Int64(n.level); out[5 * i + 1] = Int64(n.x0 >> (8 + n.level)); out[5 * i + 2] = Int64(n.z0 >> (8 + n.level))
        out[5 * i + 3] = Int64(mask)
        var quads = 0
        for t in 0..<16 where mask & (1 << UInt16(t)) != 0 { quads += n.start[lodBucketIndex(t + 1, 0, 0)] - n.start[lodBucketIndex(t, 0, 0)] }
        out[5 * i + 4] = Int64(quads)
        i += 1
    }
    return Int32(i)
}

/// Debug: quads of the active world at voxel y 0-1 (the world bottom, where lodChunkMarker lives), as
/// (level, material, face, count) quadruples; returns the count.
@_cdecl("mmc_debug_lod_bottom_quads")
public func mmc_debug_lod_bottom_quads(_ out: UnsafeMutablePointer<Int64>, _ max: Int32) -> Int32 {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return 0 }
    var hist: [Int64: Int64] = [:]
    for (k, n) in w.snapshot().meshes {
        let q = n.buffer.contents().bindMemory(to: UInt32.self, capacity: 2 * n.quadCount)
        for i in 0..<n.quadCount {
            let w0 = q[2 * i], w1 = q[2 * i + 1]
            let y = Int((w0 >> 16) & 511)
            if y > 1 { continue }
            let key = Int64(k.level) << 32 | Int64(w1 & 255) << 16 | Int64((w0 >> 25) & 7)
            hist[key, default: 0] += 1
        }
    }
    var i = 0
    for (key, c) in hist.sorted(by: { $0.key < $1.key }) where i < Int(max) {
        out[4 * i] = key >> 32; out[4 * i + 1] = (key >> 16) & 255; out[4 * i + 2] = key & 7; out[4 * i + 3] = c
        i += 1
    }
    return Int32(i)
}

/// Debug: quads per level of the active world, and how many carry block light: out[2 * level] = quads,
/// out[2 * level + 1] = lit quads (levels 0-15).
@_cdecl("mmc_debug_lod_lit_quads")
public func mmc_debug_lod_lit_quads(_ out: UnsafeMutablePointer<Int64>) {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return }
    for (k, n) in w.snapshot().meshes where k.level < 16 {
        let q = n.buffer.contents().bindMemory(to: UInt32.self, capacity: 2 * n.quadCount)
        var lit = 0
        for i in 0..<n.quadCount where (q[2 * i + 1] >> 24) & 15 != 0 { lit += 1 }
        out[2 * k.level] += Int64(n.quadCount)
        out[2 * k.level + 1] += Int64(lit)
    }
}

/// Debug: the active world's quads that overlap the world-space rectangle [x0, x1) x [z0, z1), at every level,
/// in world blocks: out[8 * i ...] = level, x, y, z (the quad's min corner), face, extent along x, y, z. Returns
/// the count (at most cap).
@_cdecl("mmc_debug_lod_quads_in")
public func mmc_debug_lod_quads_in(_ x0: Int32, _ z0: Int32, _ x1: Int32, _ z1: Int32, _ out: UnsafeMutablePointer<Int64>, _ cap: Int32) -> Int32 {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return 0 }
    var i = 0
    for (k, n) in w.snapshot().meshes {
        let s = 1 << k.level, size = lodNodeVoxels << k.level
        let ox = k.x * size, oz = k.z * size
        if ox >= Int(x1) || oz >= Int(z1) || ox + size <= Int(x0) || oz + size <= Int(z0) { continue }
        let q = n.buffer.contents().bindMemory(to: UInt32.self, capacity: 2 * n.quadCount)
        for j in 0..<n.quadCount where i < Int(cap) {
            let w0 = q[2 * j], w1 = q[2 * j + 1]
            let lx = Int(w0 & 255), lz = Int((w0 >> 8) & 255), ly = Int((w0 >> 16) & 511), face = Int((w0 >> 25) & 7)
            let qw = Int((w1 >> 8) & 255) + 1, qh = Int((w1 >> 16) & 255) + 1
            // Extents per face: X faces u = y, v = z; Y faces u = x, v = z; Z faces u = x, v = y. The face lies on
            // the voxel's far side for + faces.
            var ex = 0, ey = 0, ez = 0, px = lx, py = ly, pz = lz
            switch face / 2 {
            case 0: ey = qw; ez = qh; if face == 0 { px += 1 }
            case 1: ex = qw; ez = qh; if face == 2 { py += 1 }
            default: ex = qw; ey = qh; if face == 4 { pz += 1 }
            }
            let wx = ox + px * s, wz = oz + pz * s, wy = py * s + lodWorldMinY
            if wx >= Int(x1) || wz >= Int(z1) || wx + max(ex, 0) * s < Int(x0) || wz + max(ez, 0) * s < Int(z0) { continue }
            let o = out + 8 * i
            o[0] = Int64(k.level); o[1] = Int64(wx); o[2] = Int64(wy); o[3] = Int64(wz)
            o[4] = Int64(face); o[5] = Int64(ex * s); o[6] = Int64(ey * s); o[7] = Int64(ez * s)
            i += 1
        }
    }
    return Int32(i)
}

/// Debug (METALMC_DUMPQUADS="triggerX,x0,z0,x1,z1,path"): once the camera reaches triggerX, writes the chosen
/// nodes and tile masks overlapping [x0, x1) x [z0, z1) ("N level x0 z0 mask") and their quads there ("Q level x y z
/// face ex ey ez tile", world blocks) to path.
let lodDumpSpec: [Double]? = {
    guard let v = ProcessInfo.processInfo.environment["METALMC_DUMPQUADS"] else { return nil }
    let parts = v.split(separator: ",")
    guard parts.count == 6 else { return nil }
    lodDumpPath = String(parts[5])
    return parts[0..<5].compactMap { Double($0) }
}()
nonisolated(unsafe) var lodDumpPath = ""
nonisolated(unsafe) var lodDumpDone = false

func lodDumpChosen(_ chosen: [(LodMeshNode, UInt16)], _ spec: [Double]) {
    let x0 = Int(spec[1]), z0 = Int(spec[2]), x1 = Int(spec[3]), z1 = Int(spec[4])
    var lines: [String] = []
    for (n, mask) in chosen {
        let s = 1 << n.level, size = lodNodeVoxels << n.level
        if n.x0 >= x1 || n.z0 >= z1 || n.x0 + size <= x0 || n.z0 + size <= z0 { continue }
        lines.append("N \(n.level) \(n.x0) \(n.z0) \(mask)")
        let q = n.buffer.contents().bindMemory(to: UInt32.self, capacity: 2 * n.quadCount)
        for j in 0..<n.quadCount {
            let w0 = q[2 * j], w1 = q[2 * j + 1]
            let lx = Int(w0 & 255), lz = Int((w0 >> 8) & 255), ly = Int((w0 >> 16) & 511), face = Int((w0 >> 25) & 7)
            let qw = Int((w1 >> 8) & 255) + 1, qh = Int((w1 >> 16) & 255) + 1
            var ex = 0, ey = 0, ez = 0, px = lx, py = ly, pz = lz
            switch face / 2 {
            case 0: ey = qw; ez = qh; if face == 0 { px += 1 }
            case 1: ex = qw; ez = qh; if face == 2 { py += 1 }
            default: ex = qw; ey = qh; if face == 4 { pz += 1 }
            }
            let wx = n.x0 + px * s, wz = n.z0 + pz * s, wy = py * s + lodWorldMinY
            if wx >= x1 || wz >= z1 || wx + ex * s < x0 || wz + ez * s < z0 { continue }
            let tile = (lz / lodTileVoxels) * lodTilesPerSide + lx / lodTileVoxels
            lines.append("Q \(n.level) \(wx) \(wy) \(wz) \(face) \(ex * s) \(ey * s) \(ez * s) \(tile) \(w1 & 255)")
        }
    }
    try? lines.joined(separator: "\n").write(toFile: lodDumpPath, atomically: true, encoding: .utf8)
}

/// Debug (lodQuadVis): re-draws `draws` with depth test "equal" and no color, marking quads that own a final pixel,
/// then logs visible / submitted per level once the GPU finishes.
func lodMeasureVisibility(_ enc: MTLRenderCommandEncoder, _ draws: [(LodMeshNode, Int, Int)], _ ib: MTLBuffer) {
    let r = LodRenderer.shared
    if lodVisPipe == nil {
        guard let lib = r.library, let vs = lib.makeFunction(name: "lod_vs_slim"), let fs = lib.makeFunction(name: "lod_fs_vis") else { return }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = vs
        d.fragmentFunction = fs
        for (i, f) in ctx.passColorFormats.enumerated() { d.colorAttachments[i].pixelFormat = f; d.colorAttachments[i].writeMask = [] }
        d.depthAttachmentPixelFormat = ctx.passDepthFormat
        lodVisPipe = try? ctx.device.makeRenderPipelineState(descriptor: d)
    }
    guard let pipe = lodVisPipe, let cb = ctx.cb else { return }
    let total = draws.reduce(0) { $0 + $1.2 }
    guard total > 0, let marks = ctx.device.makeBuffer(length: total, options: [.storageModeShared]) else { return }
    memset(marks.contents(), 0, total)
    enc.setRenderPipelineState(pipe)
    // Nearer-or-equal with a small bias toward the camera (reverse-Z: larger is nearer), so rounding differences between
    // this vertex function and the main one don't reject the front surface.
    enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: false))
    let bias = Float(ProcessInfo.processInfo.environment["METALMC_VISBIAS"] ?? "") ?? 2
    enc.setDepthBias(bias, slopeScale: bias, clamp: 0)
    enc.setFragmentBuffer(marks, offset: 0, index: 27)
    var base: UInt32 = 0
    var spans: [(level: Int, base: Int, count: Int)] = []
    var bound: ObjectIdentifier?
    for (node, first, count) in draws {
        let id = ObjectIdentifier(node)
        if bound != id { enc.setVertexBuffer(node.buffer, offset: 0, index: 18); bound = id }
        var b = base, f = UInt32(first)
        enc.setFragmentBytes(&b, length: 4, index: 28)
        enc.setFragmentBytes(&f, length: 4, index: 24)
        // baseInstance selects the node's transform slot, as in the main draw.
        enc.drawIndexedPrimitives(type: .triangle, indexCount: count * 6, indexType: .uint32, indexBuffer: ib,
                                  indexBufferOffset: 0, instanceCount: 1, baseVertex: 4 * first, baseInstance: lodVisSlot[id] ?? 0)
        spans.append((node.level, Int(base), count))
        base += UInt32(count)
    }
    enc.setDepthBias(0, slopeScale: 0, clamp: 0)
    cb.addCompletedHandler { _ in
        let p = marks.contents().bindMemory(to: UInt8.self, capacity: total)
        var sub = [Int](repeating: 0, count: 16), vis = [Int](repeating: 0, count: 16)
        for sp in spans where sp.level < 16 {
            sub[sp.level] += sp.count
            var v = 0
            for i in sp.base..<(sp.base + sp.count) where p[i] != 0 { v += 1 }
            vis[sp.level] += v
        }
        // Draws (a tile's facing buckets) with no visible quad at all: waste the tile-level culling could remove.
        var deadDraws = 0, deadQuads = 0
        for sp in spans {
            var any = false
            for i in sp.base..<(sp.base + sp.count) where p[i] != 0 { any = true; break }
            if !any { deadDraws += 1; deadQuads += sp.count }
        }
        let tv = vis.reduce(0, +), ts = sub.reduce(0, +)
        log(String(format: "quadvis: %d of %d draws have no visible quad (%d quads, %.1f%% of submitted)", deadDraws, spans.count, deadQuads, 100 * Double(deadQuads) / Double(max(ts, 1))))
        var line = String(format: "quadvis: %d of %d opaque LOD quads own a pixel (%.1f%%);", tv, ts, 100 * Double(tv) / Double(max(ts, 1)))
        for l in 0..<16 where sub[l] > 0 { line += String(format: " L%d %d/%d (%.0f%%)", l, vis[l], sub[l], 100 * Double(vis[l]) / Double(sub[l])) }
        log(line)
    }
}
nonisolated(unsafe) var lodVisSlot: [ObjectIdentifier: Int] = [:]

/// Horizon occlusion for LOD sub-tiles (16 x 16 voxels). Terrain is a height field seen from one point, so whether a
/// sub-tile is behind nearer terrain is a question of elevation angles: looking along an azimuth, a point is hidden if
/// some nearer column that's solid from the world's bottom rises above the line to it. Occluders are each sub-tile's
/// solid core (every column solid from the bottom up to it: never caves, overhangs, water, glass or houses), so the
/// test only ever hides what's really hidden, with no latency and no GPU readback. Sub-tiles are processed nearest
/// first; the horizon keeps, per azimuth bin, the highest core slope (height / distance) of the cores entirely nearer
/// than the sub-tile being tested, over the bins each core fully covers. A sub-tile is hidden if its highest point's
/// slope is below the horizon over every bin it touches. Returns, per node, a visible-sub-tile mask per tile.
func lodHorizonTest(_ chosen: [(LodMeshNode, UInt16)], cx: Double, cy: Double, cz: Double,
                    fadingIn: (ObjectIdentifier, Int) -> Bool,
                    inView: (SIMD3<Float>, SIMD3<Float>) -> Bool) -> [ObjectIdentifier: [UInt16]] {
    let t0 = DispatchTime.now().uptimeNanoseconds
    struct Occluder { var far: Float; var a0: Float; var a1: Float; var slope: Float }
    struct Target { var near: Float; var far: Float; var a0: Float; var a1: Float; var top: Float; var node: Int32; var tile: Int16; var st: Int16 }
    var occluders: [Occluder] = [], targets: [Target] = []
    occluders.reserveCapacity(chosen.count * 256)
    targets.reserveCapacity(chosen.count * 256)
    // A pseudo-angle in [0, 4), monotonic in the azimuth (the "diamond angle"): no trigonometry.
    @inline(__always) func pseudo(_ x: Float, _ z: Float) -> Float {
        if z >= 0 { return x >= 0 ? z / max(x + z, 1e-30) : 1 - x / (z - x) }
        return x < 0 ? 2 - z / (-x - z) : 3 + x / (x - z)
    }
    // Azimuth interval of a rectangle not containing the camera (a1 may exceed 4: it wraps through 0).
    @inline(__always) func angles(_ x0: Float, _ x1: Float, _ z0: Float, _ z1: Float) -> (Float, Float)? {
        if x0 <= 0 && x1 >= 0 && z0 <= 0 && z1 >= 0 { return nil }
        let p0 = pseudo(x0, z0), p1 = pseudo(x1, z0), p2 = pseudo(x0, z1), p3 = pseudo(x1, z1)
        let lo = min(min(p0, p1), min(p2, p3)), hi = max(max(p0, p1), max(p2, p3))
        return hi - lo > 2 ? (hi, lo + 4) : (lo, hi)
    }
    @inline(__always) func distances(_ x0: Float, _ x1: Float, _ z0: Float, _ z1: Float) -> (Float, Float) {
        let dx = max(max(x0, -x1), 0), dz = max(max(z0, -z1), 0)
        let fx = max(abs(x0), abs(x1)), fz = max(abs(z0), abs(z1))
        return ((dx * dx + dz * dz).squareRoot(), (fx * fx + fz * fz).squareRoot())
    }
    let side = lodTileVoxels / lodSubtileVoxels
    for (ni, (n, mask)) in chosen.enumerated() {
        let s = Float(1 << n.level)
        let ox = Float(Double(n.x0) - cx), oz = Float(Double(n.z0) - cz), oy = Float(Double(lodWorldMinY) - cy)
        let id = ObjectIdentifier(n)
        let tileSize = Float(lodTileVoxels) * s
        for t in 0..<(lodTilesPerSide * lodTilesPerSide) where mask & (1 << UInt16(t)) != 0 {
            let tx = t % lodTilesPerSide, tz = t / lodTilesPerSide
            // Only tiles in view: a tile entirely outside the frustum can't hide anything inside it either (rays
            // through the frustum never reach it).
            let lo = SIMD3(ox + Float(tx) * tileSize, oy, oz + Float(tz) * tileSize)
            if !inView(lo, lo + SIMD3(tileSize, Float(lodWorldHeight), tileSize)) { continue }
            let fading = fadingIn(id, t)
            for st in 0..<lodSubtilesPerTile {
                let si = 8 * (t * lodSubtilesPerTile + st)
                let core = n.subtiles[si + 6]
                if core >= 0 && !fading {
                    // A tile fading in is only partly drawn: not an occluder until it's opaque.
                    let x0 = ox + Float(tx * lodTileVoxels + (st % side) * lodSubtileVoxels) * s
                    let z0 = oz + Float(tz * lodTileVoxels + (st / side) * lodSubtileVoxels) * s
                    let x1 = x0 + Float(lodSubtileVoxels) * s, z1 = z0 + Float(lodSubtileVoxels) * s
                    if let (a0, a1) = angles(x0, x1, z0, z1) {
                        let (near, far) = distances(x0, x1, z0, z1)
                        if near >= 8 {
                            // The lowest slope anywhere on the core's top: at its farthest point if it's above the camera,
                            // its nearest if below.
                            let h = oy + Float(core + 1) * s
                            occluders.append(Occluder(far: far, a0: a0, a1: a1, slope: h >= 0 ? h / far : h / near))
                        }
                    }
                }
                if n.subtiles[si] <= n.subtiles[si + 1] {
                    let x0 = ox + Float(n.subtiles[si]) * s, x1 = ox + Float(n.subtiles[si + 1]) * s
                    let z0 = oz + Float(n.subtiles[si + 4]) * s, z1 = oz + Float(n.subtiles[si + 5]) * s
                    if let (a0, a1) = angles(x0, x1, z0, z1) {
                        let (near, far) = distances(x0, x1, z0, z1)
                        targets.append(Target(near: near, far: far, a0: a0, a1: a1, top: oy + Float(n.subtiles[si + 3]) * s,
                                              node: Int32(ni), tile: Int16(t), st: Int16(st)))
                    }
                }
            }
        }
    }
    occluders.sort { $0.far < $1.far }
    targets.sort { $0.near < $1.near }
    let bins = 4096, perUnit = Float(bins) / 4
    var horizon = [Float](repeating: -.infinity, count: bins)
    var vis = [[UInt16]](repeating: [UInt16](repeating: 0xFFFF, count: lodTilesPerSide * lodTilesPerSide), count: chosen.count)
    var oi = 0
    for tg in targets {
        // Every core entirely nearer than this sub-tile joins the horizon, over the bins it fully covers.
        while oi < occluders.count && occluders[oi].far <= tg.near {
            let o = occluders[oi]
            oi += 1
            var b = Int((o.a0 * perUnit).rounded(.up))
            let end = Int((o.a1 * perUnit).rounded(.down))
            while b < end {
                let k = b & (bins - 1)
                if horizon[k] < o.slope { horizon[k] = o.slope }
                b += 1
            }
        }
        // The highest slope of any point of the sub-tile's top: nearest point if above the camera, farthest if below.
        let slope = tg.top >= 0 ? tg.top / max(tg.near, 1e-3) : tg.top / tg.far
        let margin = 1e-4 + abs(slope) * 1e-4
        var b = Int((tg.a0 * perUnit).rounded(.down))
        let end = Int((tg.a1 * perUnit).rounded(.down))
        var hidden = true
        while b <= end {
            if horizon[b & (bins - 1)] <= slope + margin { hidden = false; break }
            b += 1
        }
        if hidden { vis[Int(tg.node)][Int(tg.tile)] &= ~(UInt16(1) << UInt16(tg.st)) }
    }
    var out: [ObjectIdentifier: [UInt16]] = [:]
    for (ni, (n, _)) in chosen.enumerated() { out[ObjectIdentifier(n)] = vis[ni] }
    lodHorizonCalls += 1
    if lodHorizonCalls % 240 == 0 {
        var hidden = 0
        for v in vis { for m in v { hidden += 16 - m.nonzeroBitCount } }
        log(String(format: "horizon: %d occluders, %d sub-tiles tested, %d hidden, %.2f ms", occluders.count, targets.count, hidden,
                   Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6))
    }
    return out
}
nonisolated(unsafe) var lodHorizonCalls = 0
