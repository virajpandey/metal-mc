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
    if (length(center.xz) < u.discardRadius) {
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
    float blockLevel = (m == MAT_LAVA || (m >= MAT_GLOW_FIRST && m <= MAT_GLOW_LAST)) ? 15.0 : float((q.y >> 24) & 15u);
    float3 k = kShade[face] * lodLight(u, lightmap, lightSampler, max(0.0, float(u.seamInfo.z) - float(depth)), depth, blockLevel);
    // A full-resolution grass side is one grass block: dirt with vanilla's tinted fringe on top (lodShade), not
    // the average of the two that coarser voxels use (the fringe would repeat on every block of a 2-block side).
    bool grassSide = xs.w < 1.5 && faceClass == 1u && u.camFrac.w > 0.5 && ((m >= 64u && m < 96u) || m == MAT_GRASS);
    // Varyings cost vertex output bandwidth on this tile-based GPU (a float3 color2 cost 9% of the frame at a
    // million quads), so the colors are flat and packed.
    o.color = pack_float_to_unorm4x8(float4((grassSide ? colors[MAT_DIRT * 3 + 1].rgb : colors[m * 3 + faceClass].rgb) * k, 0.0));
    // Flat: per-pixel AO is applied in the fragment shader and the rest of k is constant over the quad.
    // Deep water carries the light of its (unmeshed) floor, a dozen or more blocks down, for its blend.
    o.color2 = grassSide ? pack_float_to_unorm4x8(float4(colors[m * 3].rgb * k, 0.0))
             : (deep ? pack_float_to_unorm4x8(float4(float3(0.56, 0.53, 0.45) * lodLight(u, lightmap, lightSampler, 2.0, 13), 0.0)) : 0u);
    o.rel = rel;
    float3 c = kCorners[face][corner];
    o.quv = (face < 2 ? c.yz : (face < 4 ? c.xz : c.xy)) * float2(w, h);
    // Biome-tinted variants (64 + t grass, 96 + t leaves, 128 + t water) use their base material's texture.
    uint baseMat = m >= 128 ? MAT_WATER : (m >= 96 ? MAT_LEAVES : (m >= 64 ? MAT_GRASS : m));
    o.matFace = baseMat | (face << 8) | (((q.y >> 8) & 63) << 11) | (((q.y >> 16) & 63) << 17) | (grassSide ? 1u << 23 : 0u) | (xs.w < 1.5 ? 1u << 24 : 0u) | (deep ? 1u << 25 : 0u);
    o.ao = aoOffsets[vid >> 2];
    return o;
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
    float blockLevel = (m == MAT_LAVA || (m >= MAT_GLOW_FIRST && m <= MAT_GLOW_LAST)) ? 15.0 : float((q.y >> 24) & 15u);
    float3 k = kShade[face] * lodLight(u, lightmap, lightSampler, max(0.0, float(u.seamInfo.z) - float(depth)), depth, blockLevel);
    bool grassSide = xs.w < 1.5 && faceClass == 1u && u.camFrac.w > 0.5 && ((m >= 64u && m < 96u) || m == MAT_GRASS);
    uint color = pack_float_to_unorm4x8(float4((grassSide ? colors[MAT_DIRT * 3 + 1].rgb : colors[m * 3 + faceClass].rgb) * k, 0.0));
    uint color2 = grassSide ? pack_float_to_unorm4x8(float4(colors[m * 3].rgb * k, 0.0))
                : (deep ? pack_float_to_unorm4x8(float4(float3(0.56, 0.53, 0.45) * lodLight(u, lightmap, lightSampler, 2.0, 13), 0.0)) : 0u);
    uint baseMat = m >= 128 ? MAT_WATER : (m >= 96 ? MAT_LEAVES : (m >= 64 ? MAT_GRASS : m));
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
    bool culled = dot(kNormal[face], rels[0]) >= 0.0 || outL == 4 || outR == 4 || outB == 4 || outT == 4;
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
        color *= clamp(mix(mat == MAT_WATER ? 1.0 : 0.75, luma / max(mean, 0.02), t.a), 0.0, 2.0);
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

fragment float4 lod_fs(VOut in [[stage_in]], constant LodUniforms& u [[buffer(19)]],
                       constant LodSpriteGPU* sprites [[buffer(20)]], const device uint* aoBits [[buffer(22)]],
                       texture2d<float> atlas [[texture(30)]], sampler atlasSampler [[sampler(15)]]) {
    return lodShade(in, u, sprites, aoBits, atlas, atlasSampler);
}

// The seam with vanilla: tiles that overlap vanilla's area use this variant, which drops the pixels of LOD
// voxels whose chunk section vanilla drew this frame (a bitmap of sections around the camera). It's exact
// at any camera height (vanilla picks sections by 3D distance), leaves no gap and never draws LOD over
// vanilla terrain. Only these tiles pay for the discard, which turns off hidden-surface removal for them.
fragment float4 lod_fs_seam(VOut in [[stage_in]], constant LodUniforms& u [[buffer(19)]],
                            constant LodSpriteGPU* sprites [[buffer(20)]],
                            const device uint* vanilla [[buffer(21)]], const device uint* aoBits [[buffer(22)]],
                            texture2d<float> atlas [[texture(30)]], sampler atlasSampler [[sampler(15)]]) {
    // A point just inside the voxel this face belongs to.
    float3 p = in.rel + u.camInSection.xyz - kNormal[(in.matFace >> 8) & 7] * 0.01;
    int3 sec = int3(floor(p / 16.0));
    int H = int(u.camInSection.w), W = u.seamInfo.y;
    int ix = sec.x + H, iz = sec.z + H, iy = u.seamInfo.x + sec.y;
    if (ix >= 0 && iz >= 0 && ix < W && iz < W && iy >= 0 && iy < 24) {
        uint bit = uint((iy * W + iz) * W + ix);
        if ((vanilla[bit >> 5] & (1u << (bit & 31))) != 0) discard_fragment();
    }
    return lodShade(in, u, sprites, aoBits, atlas, atlasSampler);
}

// Level transitions: a tile that appears fades in over a few frames through a screen-door pattern, and the tile
// it replaces keeps drawing the complementary pattern until then (buffer 24: progress, 1 if fading out, 1 if
// the seam bitmap is bound at 21). Every
// pixel shows one of the two. Only fading tiles pay for the discard (and the seam test, which they always run).
constant float kBayer4[16] = { 0, 8, 2, 10, 12, 4, 14, 6, 3, 11, 1, 9, 15, 7, 13, 5 };
fragment float4 lod_fs_fade(VOut in [[stage_in]], constant LodUniforms& u [[buffer(19)]],
                            constant LodSpriteGPU* sprites [[buffer(20)]],
                            const device uint* vanilla [[buffer(21)]], const device uint* aoBits [[buffer(22)]],
                            constant float4& fade [[buffer(24)]],
                            texture2d<float> atlas [[texture(30)]], sampler atlasSampler [[sampler(15)]]) {
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
    return lodShade(in, u, sprites, aoBits, atlas, atlasSampler);
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
let lodFadeFrames = Int(ProcessInfo.processInfo.environment["METALMC_FADE"] ?? "") ?? 24

/// METALMC_EXP=meshshader draws the LOD with a mesh shader (one thread per quad) instead of indexed vertices.
let lodMeshShaders = experiments.contains("meshshader")
/// Quads per mesh threadgroup (METALMC_MESHQUADS, default 32).
let lodMeshQuads = Int(ProcessInfo.processInfo.environment["METALMC_MESHQUADS"] ?? "") ?? 32

/// METALMC_EXP=nolightmap lights the LOD with the old daylight curve instead of vanilla's lightmap (A/B).
let lodNoLightmap = experiments.contains("nolightmap")

/// Mean alpha of vanilla's water texture (water_still), the LOD water's opacity.
let lodWaterAlpha: Float = 0.706

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
                    .replacingOccurrences(of: "MAT_GLOW_FIRST", with: "\(Mat.glowstone.rawValue)u")
                    .replacingOccurrences(of: "MAT_GLOW_LAST", with: "\(Mat.froglight.rawValue)u")
                    .replacingOccurrences(of: "MESH_QUADS", with: "\(lodMeshQuads)")
                    .replacingOccurrences(of: "GRASS_SIDE_SPRITE", with: "\(lodMaterialSprites.count - 1)u")
                    .replacingOccurrences(of: "GRASS_GRAY", with: "\(lodGrassGray)f")
                library = try ctx.device.makeLibrary(source: src, options: nil)
            }
            if mesh {
                let d = MTLMeshRenderPipelineDescriptor()
                d.label = "MetalMC LOD (mesh)"
                d.meshFunction = library!.makeFunction(name: "lod_mesh")
                d.fragmentFunction = library!.makeFunction(name: fade ? "lod_fs_fade" : (seam ? "lod_fs_seam" : "lod_fs"))
                d.maxTotalThreadsPerMeshThreadgroup = lodMeshQuads
                for (i, f) in colorFormats.enumerated() {
                    d.colorAttachments[i].pixelFormat = f
                    if i > 0 { d.colorAttachments[i].writeMask = [] }
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
                // Only the main color target gets LOD color; extra targets (OIT) are left untouched.
                // The occlusion boxes write no color at all.
                if i > 0 || box { d.colorAttachments[i].writeMask = [] }
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
    func ensureColors() {
        if colorBuffer != nil { return }
        let c = lodColorTable()
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
    // Level-1 nodes exist far enough out for level 0's parents (METALMC_LOD0 can reach past 1.5 km).
    let fine = max(1536, lodLevel0Radius + 512)
    r.lock.lock()
    let id = r.nextWorldId
    r.nextWorldId += 1
    // 26.x's End has sky light (sky_light_color #ac60cd is what tints its end stone pink); the Nether has none.
    let w = LodWorld(id: id, dimension: dimension, floating: dimension == "minecraft:the_end",
                     hasSkyLight: dimension != "minecraft:the_nether", regionDir: regionDir, storeDir: storeDir, cacheDir: cacheDir,
                     maxLevel: maxLevel, fineRadius: fine, centerX: centerX, centerZ: centerZ)
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
    guard let enc = ctx.pass, !ctx.scissorEmpty else { return 0 }
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return 0 }
    let snap = w.snapshot()
    guard !snap.meshes.isEmpty else { return 0 }
    let useMesh = lodMeshShaders
    guard let pipe = r.pipeline(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat, mesh: useMesh) else { return 0 }
    r.ensureColors()
    let cx = cam[0], cy = cam[1], cz = cam[2]
    // Zoom relative to vanilla's default 70-degree field of view (proj[1][1] = cot(35 degrees)), counted only when it's
    // clearly zoomed (the spyglass is about 10x; sprinting widens the view a little, which changes nothing).
    let fovZoom = Double(p[5]) / 1.4281
    let chosen = LodRenderer.select(snap.meshes, maxLevel: w.maxLevel, camX: cx, camZ: cz, splitFactor: 2.0,
                                    level0Radius: Double(lodLevel0Radius), zoom: fovZoom > 1.25 && !lodNoZoom ? fovZoom : 1)
    guard !chosen.isEmpty else { return 0 }
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
    u.seamInfo = SIMD4(Int32(csy - bottomSection), Int32(W), w.hasSkyLight ? 15 : 0, 0)
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
            for water in [false, true] {
                let base = t * lodBucketsPerTile + (water ? 6 : 0)
                var f = 0
                while f < 6 {
                    if !faceVisible[f] || n.start[base + f + 1] == n.start[base + f] { f += 1; continue }
                    var e = f + 1
                    while e < 6 && (faceVisible[e] || n.start[base + e + 1] == n.start[base + e]) { e += 1 }
                    draws.append(Draw(node: n, slot: slot, first: n.start[base + f], count: n.start[base + e] - n.start[base + f],
                                      seam: seam, water: water, fade: fade, fadeOut: fadeOutStart != nil))
                    f = e
                }
            }
        }
    }
    for (n, tileMask) in chosen { addTiles(n, tileMask, fadeOutStart: nil) }
    for f in r.fadeOut { addTiles(f.node, f.mask, fadeOutStart: f.start) }
    if r.frame % 1000 == 0 {
        var perLevel = [Int](repeating: 0, count: 9), quadsPerLevel = [Int](repeating: 0, count: 9)
        for c in chosen { perLevel[min(8, c.0.level)] += 1 }
        for d in draws { quadsPerLevel[min(8, d.node.level)] += d.count }
        log("LOD: frame \(r.frame): \(draws.count) draws, \(draws.reduce(0) { $0 + $1.count }) quads, chosen per level \(perLevel), K quads per level \(quadsPerLevel.map { $0 / 1000 }), \(chosen.filter { $0.0.level == 0 }.count) level-0 nodes, \(r.coveredTiles) tiles covered by vanilla in 1000 frames; vanilla drew \(r.vanillaSections.count) sections, compiled \(r.compiledSections.count), distance \(r.vanillaDistance), skip checks \(r.skipDebug)")
        r.skipDebug = [0, 0, 0, 0, 0]
        r.coveredTiles = 0
    }
    let testBoxes = vis.map { !$0.slots.isEmpty } ?? false
    guard !draws.isEmpty || testBoxes else { return 0 }

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
    var state = (seam: false, water: false, fade: false)
    for d in ordered {
        let fading = d.fade > 0
        if d.seam != state.seam || d.water != state.water || fading != state.fade {
            let p = fading ? (d.water ? fadeWaterPipe : fadePipe) : (d.water ? (d.seam ? seamWaterPipe : waterPipe) : (d.seam ? seamPipe : pipe))
            guard let p else { continue }   // a pipeline that failed to build: skip its draws, not everything after them
            if d.water && !state.water {
                enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: false))
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
            } else {
                enc.setVertexBuffer(d.node.buffer, offset: 0, index: 18)
                enc.setVertexBuffer(d.node.aoOffsets, offset: 0, index: 22)
            }
            enc.setFragmentBuffer(d.node.ao, offset: 0, index: 22)
            bound = id
        }
        if useMesh {
            var md = SIMD4<UInt32>(UInt32(d.first), UInt32(d.count), UInt32(d.slot), 0)
            enc.setMeshBytes(&md, length: 16, index: 17)
            enc.drawMeshThreadgroups(MTLSize(width: (d.count + lodMeshQuads - 1) / lodMeshQuads, height: 1, depth: 1),
                                     threadsPerObjectThreadgroup: MTLSize(width: 1, height: 1, depth: 1),
                                     threadsPerMeshThreadgroup: MTLSize(width: lodMeshQuads, height: 1, depth: 1))
        } else {
            enc.drawIndexedPrimitives(type: .triangle, indexCount: d.count * 6, indexType: .uint32, indexBuffer: ib,
                                      indexBufferOffset: 0, instanceCount: 1, baseVertex: 4 * d.first, baseInstance: d.slot)
        }
    }
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
@_cdecl("mmc_lod_set_atlas")
public func mmc_lod_set_atlas(_ view: Int64, _ rects: UnsafePointer<Float>, _ count: Int32) {
    let r = LodRenderer.shared
    let n = Int(count)
    var table = [SIMD4<Float>](repeating: .zero, count: max(1, n) * 3)
    for i in 0..<n {
        table[3 * i] = SIMD4(rects[8 * i], rects[8 * i + 1], rects[8 * i + 2], rects[8 * i + 3])
        table[3 * i + 1] = SIMD4(rects[8 * i + 4], rects[8 * i + 5], rects[8 * i + 6], rects[8 * i + 7])
        let s = i < lodMaterialSprites.count ? lodMaterialSprites[i] : LodSprite(top: "", side: "", topLuma: 1, sideLuma: 1)
        table[3 * i + 2] = SIMD4(s.topLuma, s.sideLuma, 0, 0)
    }
    r.spriteBuffer = ctx.device.makeBuffer(bytes: table, length: table.count * 16, options: [.storageModeShared])
    r.atlas = view == 0 ? nil : (from(view) as TextureBox).texture
    if let a = r.atlas { log("LOD: texture detail from the block atlas \(a.width)x\(a.height), \(a.mipmapLevelCount) mips, \(n) materials") }
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
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return 0 }
    let snap = w.snapshot()
    let chosen = LodRenderer.select(snap.meshes, maxLevel: w.maxLevel, camX: camX, camZ: camZ, splitFactor: 2.0,
                                    level0Radius: Double(lodLevel0Radius))
    var i = 0
    for (n, mask) in chosen where i < Int(max) {
        out[5 * i] = Int64(n.level); out[5 * i + 1] = Int64(n.x0 >> (8 + n.level)); out[5 * i + 2] = Int64(n.z0 >> (8 + n.level))
        out[5 * i + 3] = Int64(mask); out[5 * i + 4] = Int64(n.quadCount)
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
