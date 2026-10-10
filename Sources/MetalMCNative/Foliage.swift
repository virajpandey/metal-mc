import Foundation
import Metal
import MetalMCCore
import simd

// Foliage: light through leaves and plants (METALMC_EXP=leaflight, with lit) and leaves and plants that sway in the wind
// (METALMC_EXP=wave). docs/lighting-design.md, "Light through leaves, and waving foliage".
//
// The G-buffer's foliage class (leaflight): lit terrain's texel keeps a material class in the two low bits of its
// albedo's blue byte (x bits 16-17): 0 anything else, 1 a leaf, 2 a plant (grass, ferns, flowers, crops, saplings).
// litPack rounds the blue byte to a multiple of 4 (at most 2/255 off) with leaflight on, so every writer leaves 0 there
// unless it sets a class (litPackFoliage): the LOD by its material (every leaf tint, cherry, the poplars 160-162), the far
// field's canopy by its material, the near chunks by the atlas sprite their quad's texture comes from (a class map from
// the sprites the Java side sends, metalmc.backend.MetalFoliage). Readers: litFoliageClass(g) in litShaderHeader, or
// (g.x >> 29) != 0 ? (g.x >> 16) & 3 : 0.
//
// Light through leaves (leaflight): the traced sun rays (RtShadows) go on through leaf blocks. The LOD meshes a canopy as
// one closed surface (faces between leaf blocks are culled), so a ray's closest hits come in pairs, into a canopy and out
// of it, and the distance between them is the blocks of leaves it crossed. The direct sun gets through
// exp(-foliageLeafSigma x blocks) (0.3 a block), stored as the visibility as before; the blocks crossed go in a second
// channel, from which the relight (litFoliageLight) works out the light that leaves pass on: scattered inside them, it
// falls off more slowly than the direct beam, green slowest, and comes out of every face, most out of the ones turned away
// from the sun, more toward the camera the nearer it looks into the sun. A leaf or plant under cover also takes the sky's
// light through the blocks of leaves its sky light level says are above it.
//
// Wave: leaves and plants sway in a world-space wind field (foliageWindHeader): gusts that run downwind across the land,
// and a quicker flutter. Leaves move by their vertices' world positions (corners shared by neighboring leaf blocks move
// together), about 0.04 blocks; a plant leans from its root, its top 0.12 blocks, the upper half of a double plant twice
// that. In the near chunks' vertex shader, and gently in the LOD's level 0 near the camera (fading out by 160 blocks). The
// traced shadows' structures don't move.

/// METALMC_EXP=leaflight (with lit): sunlight and sky light through leaves and plants.
let foliageLight = litEnabled && experiments.contains("leaflight")
/// METALMC_EXP=wave: leaves and plants sway in the wind (the near chunks).
let foliageWave = experiments.contains("wave")
/// METALMC_EXP=wave,wavelod: the LOD's full-resolution leaves sway too, near the camera. Off unless asked: past vanilla's
/// render distance the sway is under a pixel, and the test on every LOD vertex cost 0.12-0.16 ms (offline, forest and
/// mountain_view at 3456 x 2234). It matters only at a short render distance, where the LOD's level 0 comes close.
let foliageWaveLod = foliageWave && experiments.contains("wavelod")
/// The near chunks look up each quad's foliage class in the atlas sprites' class map (with either switch).
let foliageSprites = nearChunksEnabled && (foliageLight || foliageWave)

/// The direct sun through one block of leaves: 0.3 (METALMC_LEAFT, 0.01-1).
let foliageLeafTransmittance = min(max(Float(ProcessInfo.processInfo.environment["METALMC_LEAFT"] ?? "") ?? 0.3, 0.01), 1)
/// Per block of leaves: the direct beam's extinction (-ln of the above).
let foliageLeafSigma = -log(foliageLeafTransmittance)
/// The second channel of the traced texture: exp(-foliageLeafK x blocks of leaves crossed), so 8 bits hold up to about
/// 27 blocks.
let foliageLeafK: Float = 0.2

/// Sway (METALMC_EXP=wave): leaves' (blocks), plants' tops' (blocks), and the distance by which the LOD's leaves are still.
private let waveLeaf = max(0, Float(ProcessInfo.processInfo.environment["METALMC_WAVELEAF"] ?? "") ?? 0.04)
private let wavePlant = max(0, Float(ProcessInfo.processInfo.environment["METALMC_WAVEPLANT"] ?? "") ?? 0.12)
private let waveLodFade = max(1, Float(ProcessInfo.processInfo.environment["METALMC_WAVELODFADE"] ?? "") ?? 160)

/// The leaf materials of the LOD's quads (raw ids: the biome-tinted leaves 96-127 too) and of its G-buffer's base ids,
/// as a shader expression of a material `m` (the far field's canopy uses it too).
let foliageLodLeafTest = "(m == \(Mat.leaves.rawValue)u || m == \(Mat.cherryLeaves.rawValue)u || (m >= \(lodLeavesBase)u && m < \(lodLeavesBase + 32)u) || (m >= \(Mat.yellowPoplarLeaves.rawValue)u && m <= \(Mat.orangePoplarLeaves.rawValue)u))"

// MARK: - Shader text

/// The G-buffer's foliage class (spliced into litShaderHeader with leaflight only).
let foliagePackHeader = """
// Foliage (METALMC_EXP=leaflight, Foliage.swift): a class in the two low bits of the albedo's blue byte (x bits 16-17)
// of lit terrain (face code 1-7): 0 anything else, 1 a leaf, 2 a plant. litPack leaves 0 there (it rounds blue to a
// multiple of 4); the LOD and the near chunks set it with litPackFoliage.
#define LIT_FOLIAGE 1
#define LIT_LEAF 1u
#define LIT_PLANT 2u
static uint2 litPackFoliage(uint2 g, uint cls) {
    return (g.x >> 29) != 0u ? uint2((g.x & ~(3u << 16)) | ((cls & 3u) << 16), g.y) : g;
}
static uint litFoliageClass(uint2 g) { return (g.x >> 29) != 0u ? ((g.x >> 16) & 3u) : 0u; }

"""

/// The LOD's leaf test (spliced into the LOD's shader with leaflight or wave): m a quad's material (raw or base id).
let foliageLodHeader = """

// Foliage (Foliage.swift): the LOD's leaf materials (every leaf tint, cherry, the poplars).
static bool lodFoliageLeaf(uint m) { return \(foliageLodLeafTest); }

"""

/// The wind field (spliced into the near chunks' and the LOD's shaders with wave only).
let foliageWindHeader = """

// Wind (METALMC_EXP=wave, Foliage.swift). Positions are world blocks with x and z modulo 1024 (every wave here has a
// whole number of cycles across 1024 blocks, so the field has no seam where the camera's position wraps), times seconds
// modulo 600 (whole cycles too). Gusts: two long waves running downwind (toward +x, a little +z) at 5-6 blocks a second,
// so a gust's front crosses the land; flutter: quicker small motion whose phase changes every couple of blocks.
struct FoliageWind {
    float4 cam;   // xyz: the camera's world position, x and z modulo 1024 (blocks); w: the clock (s, modulo 600)
    float4 amp;   // x: leaves' sway (blocks), y: plants' (their tops), z: the LOD's leaves are still past this (blocks), w: 1 to sway
};
constant float kWindK = 6.28318531 / 1024.0;   // a cycle across 1024 blocks
constant float kWindW = 6.28318531 / 600.0;    // a cycle in 600 s
constant float2 kWindDir = float2(0.93828, 0.34568);   // (19, 7), normalized: the way the gusts run

// The gusts' strength at p (x and z) and time t: about 0.1 nearly calm to 1 a full gust.
static float foliageGust(float2 p, float t) {
    return 0.55 + 0.3 * sin(kWindK * dot(p, float2(19.0, 7.0)) - kWindW * 70.0 * t)
                + 0.15 * sin(kWindK * dot(p, float2(41.0, -9.0)) - kWindW * 130.0 * t);
}
// A leaf's sway at its vertex's world position p (about unit size): the gust leans it downwind, the flutter shakes it.
static float3 foliageLeafSway(float3 p, float t) {
    float g = foliageGust(p.xz, t);
    float ph = kWindK * (283.0 * p.x + 317.0 * p.z) + 1.37 * p.y;
    float ph2 = kWindK * (-229.0 * p.x + 263.0 * p.z) + 0.91 * p.y;
    float3 flutter = float3(sin(ph + kWindW * 1020.0 * t), 0.6 * sin(ph2 + kWindW * 1380.0 * t), sin(ph2 + kWindW * 1170.0 * t));
    return float3(kWindDir.x, 0.0, kWindDir.y) * (0.45 * g) + flutter * (0.25 + 0.45 * g);
}
// A plant's sway at its top (about unit size), from its root block b: it leans downwind with the gust and nods across
// it, the whole plant together.
static float3 foliagePlantSway(float3 b, float t) {
    float g = foliageGust(b.xz + 0.5, t);
    float ph = kWindK * (283.0 * b.x + 317.0 * b.z) + 1.37 * b.y;
    float along = g * (0.75 + 0.25 * sin(ph + kWindW * 1290.0 * t));
    float nod = sin(ph * 1.618 + kWindW * 750.0 * t) * (0.25 + 0.35 * g);
    float2 d = kWindDir * along + float2(-kWindDir.y, kWindDir.x) * nod;
    return float3(d.x, -0.2 * dot(d, d), d.y);
}

"""

/// lod_vs's sway (with wave only), before its clip position: `swayed` takes rel's place there.
let lodWaveSway = """
    // Wave (METALMC_EXP=wave, Foliage.swift): full-resolution leaves sway near the camera, fading out by wind.amp.z blocks
    // (vanilla's terrain or the near chunks are drawn there at the usual render distances). The texture coordinates and the
    // fog keep the leaf's own place (o.rel), so the texture moves with the leaf.
    float3 swayed = rel;
    if (xs.w < 1.5 && wind.amp.w > 0.0 && lodFoliageLeaf(m)) {
        float swayFade = 1.0 - smoothstep(0.5 * wind.amp.z, wind.amp.z, length(rel));
        if (swayFade > 0.0) swayed += foliageLeafSway(rel + wind.cam.xyz, wind.cam.w) * (wind.amp.x * swayFade);
    }

"""

/// The near chunks' foliage (spliced into their shader with leaflight or wave): the class of a quad's atlas sprite.
let foliageNearHeader = """

// Foliage (Foliage.swift): the class map of the block atlas's sprites (a texel per cell of the atlas, FoliageSprites):
// bits 0-1 the G-buffer's class (1 leaf, 2 plant), bits 2-4 how it sways (1 a leaf, 2 a plant or a double plant's lower
// half, 3 a double plant's upper half, 4 hanging), at a texture coordinate inside the sprite.
static uint nearFoliageClass(texture2d<uint, access::read> map, float2 uv) {
    uint2 size = uint2(map.get_width(), map.get_height());
    return map.read(min(uint2(max(uv, 0.0) * float2(size)), size - 1u)).r;
}

"""

/// near_vs's part (with leaflight or wave), before its clip position: the quad's foliage code, and with wave its sway.
let nearFoliageVertex: String = """
    // Foliage (Foliage.swift): the code of the quad's atlas sprite, looked up at the middle of its texture (vertex k and
    // the opposite corner, k ^ 2, span it).
    NearCorner opp = nearDecode(quads, vid >> 2, (vid & 3u) ^ 2u);
    uint fol = nearFoliageClass(foliageMap, 0.5 * (c.uv + opp.uv));
    o.foliage = fol;

""" + (foliageWave ? """
    // Wave (METALMC_EXP=wave): world positions with x and z modulo 1024 (the wind's period): the section's origin wrapped,
    // plus the vertex's place in it.
    uint sway = (fol >> 2) & 7u;
    if (sway != 0u && wind.amp.w > 0.0) {
        float3 sec0 = float3(float(sec.position.x & 1023), float(sec.position.y), float(sec.position.z & 1023));
        if (sway == 1u) {
            // A leaf: by its vertex's position, so the corners neighboring leaf blocks share move together.
            pos += foliageLeafSway(sec0 + c.pos, wind.cam.w) * wind.amp.x;
        } else {
            // A plant leans from its root: the vertices at the top of its texture (v grows downward) sway, those at the
            // bottom stay put (a double plant's upper half: its bottom as much as the lower half's top, its top twice
            // that; a hanging plant the other way up), all by its root block, so the whole plant moves together.
            float top = c.uv.y < opp.uv.y ? 1.0 : 0.0;
            float k = sway == 3u ? 1.0 + top : (sway == 4u ? 1.0 - top : top);
            float3 root = floor(sec0 + 0.5 * (c.pos + opp.pos)) - (sway == 3u ? float3(0.0, 1.0, 0.0) : float3(0.0));
            pos += foliagePlantSway(root, wind.cam.w) * (wind.amp.y * k);
        }
    }

""" : "")

/// The relight's part (spliced into Lit.swift's litRelightHeader with leaflight only).
let foliageRelightHeader = """

// Light through foliage (METALMC_EXP=leaflight, Foliage.swift). Leaves and plants are thin: sunlight that falls on them
// comes out of their other side too, in their own color, and a canopy glows from inside instead of going black. For a leaf
// or plant pixel (litFoliageClass), light added to the relight's (per unit of albedo, before the daylight scale and the
// exposure):
// - The sun through the leaves. The traced rays (RtShadows) go on through leaf blocks: the direct beam gets 0.3 a block
//   through (the relight's sun term, as before) and vis.g = exp(-RT_LEAF_K x the blocks of leaves crossed) toward the sun
//   (0: something opaque in the way). Light scattered inside leaves falls off more slowly than the direct beam, and the
//   leaf's own color more slowly than the rest (a green leaf absorbs little green, a pale oak's little of any): per
//   channel exp(-falloff x blocks), the falloff from LEAF_FALLOFF_MIN in the albedo's strongest channel to
//   LEAF_FALLOFF_MAX in a channel it lacks (by the linear albedo over its largest channel). It comes out of every face,
//   most out of the faces turned away from the sun (LEAF_FRONT to LEAF_BACK by the face's angle), and more toward the
//   camera the nearer it looks into the sun (forward scattering, a Henyey-Greenstein lobe, LEAF_FWD and LEAF_G). A plant
//   takes PLANT_TRANS and no lobe: a blade lets light through diffusely, and the grass around it, which the rays don't
//   see, shades it from a low sun.
// - The sky through the leaves: under cover, at least the open sky's light through the blocks of leaves the sky light
//   level says are above (vanilla's leaves dim sky light a level a block), with the same falloff (LEAF_SKY): with the GI
//   cache, whose rays stop at leaves, a canopy's inside would be black.
#define RT_LEAF_K \(foliageLeafK)
#ifndef LEAF_FRONT
#define LEAF_FRONT 0.1
#endif
#ifndef LEAF_BACK
#define LEAF_BACK 0.7
#endif
#ifndef PLANT_TRANS
#define PLANT_TRANS 0.12
#endif
#ifndef LEAF_FWD
#define LEAF_FWD 0.03
#endif
#ifndef LEAF_G
#define LEAF_G 0.6
#endif
#ifndef LEAF_FALLOFF_MIN
#define LEAF_FALLOFF_MIN 0.3
#endif
#ifndef LEAF_FALLOFF_MAX
#define LEAF_FALLOFF_MAX 1.0
#endif
#ifndef LEAF_SKY
#define LEAF_SKY 0.8
#endif
#ifndef LEAF_SHEEN
#define LEAF_SHEEN 0.5
#endif
#ifndef LEAF_ROUGH
#define LEAF_ROUGH 0.4
#endif

// The traced texel for pixel q: (direct visibility, exp(-RT_LEAF_K x blocks of leaves toward the sun)). Without rays this
// frame, the open sky's estimate from the sky light level (a block of cover a level).
static float2 litFoliageVis(texture2d<half, access::read> vis, uint2 q, constant LitFrame& f, float sky) {
    if (f.size.z > 0.0) {
        uint s = uint(f.size.z);
        return float2(vis.read(min(q / s, uint2(vis.get_width() - 1, vis.get_height() - 1))).rg);
    }
    float cover = max(15.0 - sky, 0.0);
    return float2(saturate((sky - 12.0) / 3.0), exp(-RT_LEAF_K * cover));
}

static float3 litFoliageLight(uint cls, uint fi, float3 n, float3 rel, uint2 q, texture2d<half, access::read> vis,
                              constant LitFrame& f, constant float4* env, float sky, float ao, float3 skyAmb, float3 albedo) {
    if (cls == 0u) return float3(0.0);
    float3 add = float3(0.0);
    // The falloff through leaves per channel, from the leaf's own color (the linear albedo, as its square, over its largest
    // channel), in log2 units per block.
    float3 alin = albedo * albedo;
    float3 tint = alin / max(max(alin.r, max(alin.g, alin.b)), 1e-4);
    float3 falloff2 = (LEAF_FALLOFF_MAX - (LEAF_FALLOFF_MAX - LEAF_FALLOFF_MIN) * tint) * 1.44269504;
    // The sun (not the moon: at night the rays went toward it, and its light through leaves is too faint to matter).
    float2 v = litFoliageVis(vis, q, f, sky);
    float3 L = f.sunDir.xyz;
    float3 toCam = -rel * rsqrt(max(dot(rel, rel), 1e-8));
    if (f.sunDir.w < 0.5 && v.y > 0.0) {
        // exp(-falloff x blocks), blocks = -ln(v.y) / RT_LEAF_K
        float blocks = log2(max(v.y, 1e-4)) * (-0.69314718 / RT_LEAF_K);
        float3 T = exp2(-falloff2 * blocks);
        float c = -dot(toCam, L);   // 1: looking into the sun through the leaf
        float g2 = LEAF_G * LEAF_G, x = max(1.0 + g2 - 2.0 * LEAF_G * c, 1e-4);
        float hg = (1.0 - g2) / (x * sqrt(x));   // Henyey-Greenstein x 4 pi: 1 for no lobe
        // A plant takes no lobe: a blade lets light through diffusely, and the grass around it, which the rays don't see,
        // shades it from a low sun.
        float facing = fi < 6u ? saturate(0.5 - 0.5 * dot(n, L)) : 0.5;   // 1: the face turned away from the sun
        float through = cls == LIT_LEAF ? mix(LEAF_FRONT, LEAF_BACK, facing) + LEAF_FWD * hg : PLANT_TRANS;
        add += env[0].rgb * T * through;
    }
    // A soft sheen (LEAF_SHEEN): a leaf's or a blade's waxy surface reflects the sun, most at grazing angles: a broad GGX
    // lobe (LEAF_ROUGH) with Schlick's Fresnel (F0 0.04) on the face (a plant: on the vertical), times the direct sun's
    // visibility, in the leaf's hue (E is per unit of albedo: the sheen over the albedo's luminance).
    if (LEAF_SHEEN > 0.0 && f.sunDir.w < 0.5 && v.x > 0.0) {
        float3 N = fi < 6u ? n : float3(0.0, 1.0, 0.0);
        float nl = dot(N, L), nv = dot(N, toCam);
        if (nl > 0.0 && nv > 0.0) {
            float3 H = normalize(L + toCam);
            float a2 = LEAF_ROUGH * LEAF_ROUGH, nh = saturate(dot(N, H)), dd = nh * nh * (a2 - 1.0) + 1.0;
            float D = a2 / (3.14159265 * dd * dd);
            float G = 0.5 / (nl * sqrt(nv * nv * (1.0 - a2) + a2) + nv * sqrt(nl * nl * (1.0 - a2) + a2));
            float m = 1.0 - saturate(dot(toCam, H)), m2 = m * m;
            float F = 0.04 + 0.96 * m2 * m2 * m;
            float lum = max(dot(alin, float3(0.2126, 0.7152, 0.0722)), 0.02);
            add += env[0].rgb * (LEAF_SHEEN * v.x * D * G * F * nl / lum);
        }
    }
    // The sky through the blocks of leaves above, where it gives more than the relight's sky term: under cover only (a
    // face out in the open sees the sky itself, which the sky term has; from a block of cover in, it sees leaves around).
    float cover = max(15.0 - sky, 0.0);
    float3 skyFol = env[3].rgb * exp2(-falloff2 * cover) * (LEAF_SKY * ao * saturate(cover * 0.5));
    add += max(skyFol - skyAmb, 0.0);
    return add;
}

"""

/// The relight's call (with leaflight only), after the light E is made: what leaves and plants pass on.
let foliageRelightCall = "        E += litFoliageLight(litFoliageClass(g), fi, n, rel, q, vis, f, env, sky, ao, skyAmb, albedo) * (dayScale * f.misc.w);\n"

/// The relight's debug views (with leaflight only): 13 the foliage class (leaf green, plant yellow, anything else gray),
/// 14 the traced texel (red: direct visibility, green: exp(-k x blocks of leaves toward the sun)).
let foliageDebugViews = """

        else if (view == 13u) c = litFoliageClass(g) == LIT_LEAF ? float3(0.1, 0.8, 0.15) : (litFoliageClass(g) == LIT_PLANT ? float3(0.9, 0.8, 0.1) : float3(0.25));
        else if (view == 14u) c = float3(litFoliageVis(vis, q, f, sky), 0.0);
"""

// MARK: - Ray-traced shadows' part

/// RtShadows' kernel with leaflight: the instance table to find what a ray hit, and the walk through leaves.
let foliageRtHeader = """

// Leaves let the sun through (METALMC_EXP=leaflight, Foliage.swift). A ray that meets a leaf goes on through it: the LOD
// meshes a canopy as one closed surface (faces between leaf blocks are culled), so the ray's closest hits come in pairs,
// into a canopy and out of it, and the distance between them is the blocks of leaves crossed. The direct sun gets
// through exp(-RT_LEAF_SIGMA x blocks) (0.3 a block: a layer or two let some through, a deep canopy none); the second
// channel keeps exp(-RT_LEAF_K x blocks) for the relight's light scattered through leaves. Something opaque anywhere along
// the ray: no light either way.
#define RT_LEAF_SIGMA \(foliageLeafSigma)
#define RT_LEAF_K \(foliageLeafK)
#define RT_LEAF_HITS 8
// Gi.swift's GiTile: per instance of the instance structure, its LOD node's quad buffer and the first quad of each of
// its structure's geometries (the opaque faces, then the tile-edge skirts), two triangles a quad.
struct RtTile { const device uint* quads; uint start[2]; };
constant float3 kRtNormal[6] = { float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0), float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1) };
static uint2 rtHitQuad(const device RtTile* tiles, uint instance, uint geometry, uint primitive) {
    const device RtTile& t = tiles[instance];
    uint q = t.start[min(geometry, 1u)] + primitive / 2u;
    return uint2(t.quads[2u * q], t.quads[2u * q + 1u]);
}
static bool rtLeaf(uint m) { return \(foliageLodLeafTest); }
// The ray's walk through leaves: (direct transmittance, blocks of leaves crossed), or (0, -1) if something opaque is in
// the way. Near the surface (RT_LEAF_NEAR blocks: the canopy it's in or under), the hits in order (closest first: a short
// ray's traversal is cheap); a hit into a canopy opens a stretch of leaves, one out of it closes it (the first hit out
// with none in: the ray started inside leaves). Past that, one any-hit query for the rest of the ray, as the plain
// shadow ray: nothing, the light gets through; something opaque, none; more leaves (another tree's crown, a forest
// ahead of a low sun), RT_LEAF_FAR blocks more of them.
// backLeaf: the ray leaves a leaf's face turned away from the sun, so it has to cross that leaf's own canopy; if the near
// range has no surface at all, the canopy isn't in the structures (the far field's, a tile still building): a crown's
// typical RT_LEAF_FAR blocks.
#define RT_LEAF_NEAR 24.0
#define RT_LEAF_FAR 3.0
static float2 rtThroughLeaves(instance_acceleration_structure accel, const device RtTile* tiles, ray r, bool backLeaf) {
    intersector<instancing> closest;
    closest.assume_geometry_type(geometry_type::triangle);
    float blocks = 0.0, enter = 0.0, last = r.min_distance, far = r.max_distance;
    bool inside = false;
    r.max_distance = min(far, RT_LEAF_NEAR);
    for (uint i = 0; i < RT_LEAF_HITS; i++) {
        auto h = closest.intersect(r, accel, 0xFF);
        if (h.type == intersection_type::none) {
            if (i == 0u && backLeaf) blocks = RT_LEAF_FAR;
            break;
        }
        uint2 w = rtHitQuad(tiles, h.instance_id, h.geometry_id, h.primitive_id);
        if (!rtLeaf(w.y & 255u)) return float2(0.0, -1.0);
        uint face = (w.x >> 25) & 7u;
        bool into = face < 6u && dot(kRtNormal[face], r.direction) < 0.0;
        float t = h.distance;
        if (into) {
            if (!inside) { inside = true; enter = t; }
        } else {
            blocks += t - (inside ? enter : last);
            inside = false;
        }
        last = t;
        if (blocks * RT_LEAF_SIGMA > 6.0) return float2(0.0, blocks);
        r.min_distance = t + 0.002;
    }
    // Still inside leaves at the near range's end (a canopy thicker than that along the ray): deep.
    if (inside) return float2(0.0, blocks + (r.max_distance - enter));
    if (r.max_distance < far) {
        r.min_distance = max(r.min_distance, r.max_distance - 0.01);
        r.max_distance = far;
        intersector<instancing> any;
        any.accept_any_intersection(true);
        any.assume_geometry_type(geometry_type::triangle);
        auto h = any.intersect(r, accel, 0xFF);
        if (h.type != intersection_type::none) {
            if (!rtLeaf(rtHitQuad(tiles, h.instance_id, h.geometry_id, h.primitive_id).y & 255u)) return float2(0.0, -1.0);
            blocks += RT_LEAF_FAR;
        }
    }
    return float2(exp(-RT_LEAF_SIGMA * blocks), blocks);
}

"""

/// RtShadows' kernel with leaflight: the class of the traced pixel (after its normal is found).
let foliageRtClassify = """
    // Leaves and plants (leaflight): the G-buffer's class at the pixel, where the terrain it describes is what shows. A
    // leaf's face turned away from the sun gets a ray too (through its own block); a plant (thin, every way up) takes the
    // vertical for the ray's offset and always gets one.
    uint cls = 0u;
    if (p.sample.w != 0u) {
        uint2 g = gbuf.read(fp).rg;
        uint dk = (((as_type<uint>(d) >> 4) & 0xFFFFu) - (g.y & 0xFFFFu)) & 0xFFFFu;
        if ((g.x >> 29) != 0u && (dk <= 1u || dk == 0xFFFFu)) cls = (g.x >> 16) & 3u;
        if (cls == 2u) n = float3(0.0, 1.0, 0.0);
    }

"""

/// RtShadows' kernel with leaflight: the ray (the walk through leaves, or the plain shadow ray in a frame without the
/// G-buffer and the instance table) and both channels written.
let foliageRtWrite = """
    // (direct transmittance, blocks of leaves toward the sun; -1: something opaque in the way). First the plain shadow ray
    // over the whole way, as without leaflight: nothing in the way (most sunlit terrain), the sun; something that isn't a
    // leaf, none; a leaf, the walk through leaves. A leaf's face turned away from the sun with nothing at all in the way:
    // its canopy isn't in the structures (the far field's, a tile still building), a crown's RT_LEAF_FAR blocks.
    float2 tl = float2(0.0, -1.0);
    intersector<instancing> isect;
    isect.accept_any_intersection(true);
    isect.assume_geometry_type(geometry_type::triangle);
    auto hit = isect.intersect(r, accel, 0xFF);
    if (hit.type == intersection_type::none) {
        tl = p.sample.w != 0u && cls == 1u && ndl < -0.05 ? float2(exp(-RT_LEAF_SIGMA * RT_LEAF_FAR), RT_LEAF_FAR) : float2(1.0, 0.0);
    } else if (p.sample.w != 0u && rtLeaf(rtHitQuad(tiles, hit.instance_id, hit.geometry_id, hit.primitive_id).y & 255u)) {
        tl = rtThroughLeaves(accel, tiles, r, false);
    }
    // Far away, where the haze takes over, toward the open sky's (1, 1), as shadeOf fades the visibility.
    float s = p.shade.x * (1.0 - smoothstep(p.shade.y, p.shade.z, length(pos)));
    float through = tl.y >= 0.0 ? exp(-RT_LEAF_K * tl.y) : 0.0;
    out.write(half4(shadeOf(p, pos, tl.x), half(1.0 - s * (1.0 - through)), 0.0h, 0.0h), gid);
"""

// MARK: - Native side

/// Mirrors FoliageWind in the shaders.
struct FoliageWindGPU {
    var cam = SIMD4<Float>.zero
    var amp = SIMD4<Float>.zero
}

final class Foliage: @unchecked Sendable {
    static let shared = Foliage()
    private let lock = NSLock()
    /// The block atlas's class map (R8Uint, a texel per cell) and the atlas texture it was made for.
    private var map: MTLTexture?
    private var mapAtlas: ObjectIdentifier?
    private var zeroMap: MTLTexture?
    private var warnedAtlas = false
    /// Offline: the clock held at this time (s; below 0 the clock's).
    var time: Double = -1
    /// The near chunks' and the LOD's binding slots (clear of what they bind themselves and of vanilla's).
    static let nearMapIndex = 18, nearWindIndex = 22, lodWindIndex = 26

    /// The wind's parameters this frame for a camera at `cam` (world blocks).
    func wind(cam: SIMD3<Double>) -> FoliageWindGPU {
        let t = time >= 0 ? time : ProcessInfo.processInfo.systemUptime
        func wrap(_ x: Double) -> Float { x.isFinite ? Float(x - (x / 1024).rounded(.down) * 1024) : 0 }
        return FoliageWindGPU(cam: SIMD4(wrap(cam.x), Float(cam.y.isFinite ? cam.y : 0), wrap(cam.z), Float(t.truncatingRemainder(dividingBy: 600))),
                              amp: SIMD4(waveLeaf, wavePlant, waveLodFade, foliageWave ? 1 : 0))
    }

    /// Sets the class map from the Java side's sprites: `rects` holds count x 5 ints (x, y, width, height in atlas pixels,
    /// then the sprite's code: class | sway << 2), for the atlas texture `atlas` (width x height pixels).
    func setSprites(atlas: MTLTexture?, width: Int, height: Int, rects: UnsafePointer<Int32>, count: Int) {
        guard width > 0, height > 0 else { return }
        // The cell: the largest power of two (up to 16) that every foliage sprite's position and size are multiples of, so
        // each cell lies inside one sprite or outside them all.
        var cell = 16
        for i in 0..<count {
            for k in 0..<4 {
                let v = Int(rects[5 * i + k])
                while cell > 1 && v % cell != 0 { cell /= 2 }
            }
        }
        let mw = max(1, width / cell), mh = max(1, height / cell)
        var cells = [UInt8](repeating: 0, count: mw * mh)
        var leaves = 0, plants = 0
        for i in 0..<count {
            let x0 = Int(rects[5 * i]) / cell, y0 = Int(rects[5 * i + 1]) / cell
            let x1 = min(mw, x0 + max(1, Int(rects[5 * i + 2]) / cell)), y1 = min(mh, y0 + max(1, Int(rects[5 * i + 3]) / cell))
            let code = UInt8(truncatingIfNeeded: rects[5 * i + 4])
            if code & 3 == 1 { leaves += 1 } else if code & 3 == 2 { plants += 1 }
            guard x0 >= 0, y0 >= 0, x0 < x1, y0 < y1 else { continue }
            for y in y0..<y1 { for x in x0..<x1 { cells[y * mw + x] = code } }
        }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Uint, width: mw, height: mh, mipmapped: false)
        d.usage = .shaderRead
        d.storageMode = .shared
        guard let t = ctx.device.makeTexture(descriptor: d) else { return }
        t.label = "MetalMC foliage sprite classes"
        cells.withUnsafeBytes { t.replace(region: MTLRegionMake2D(0, 0, mw, mh), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: mw) }
        lock.lock()
        map = t
        mapAtlas = atlas.map { ObjectIdentifier($0.parent ?? $0) }
        warnedAtlas = false
        lock.unlock()
        log("foliage: sprite classes for the \(width)x\(height) block atlas: \(leaves) leaf and \(plants) plant sprites, a \(mw)x\(mh) map (\(cell)-pixel cells)")
    }

    private func zero() -> MTLTexture? {
        if let zeroMap { return zeroMap }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Uint, width: 1, height: 1, mipmapped: false)
        d.usage = .shaderRead
        zeroMap = ctx.device.makeTexture(descriptor: d)
        var z: UInt8 = 0
        zeroMap?.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &z, bytesPerRow: 1)
        return zeroMap
    }

    /// The near chunks' draw (mmc_near_draw): the class map for the atlas it samples (none for another atlas) and the wind.
    func bindNear(_ enc: MTLRenderCommandEncoder, atlas: MTLTexture) {
        lock.lock()
        var m = map
        if let id = mapAtlas, id != ObjectIdentifier(atlas.parent ?? atlas) {
            if !warnedAtlas { log("foliage: the near chunks' atlas isn't the one the sprite classes came from; no classes until it's sent"); warnedAtlas = true }
            m = nil
        }
        lock.unlock()
        if let t = m ?? zero() { enc.setVertexTexture(t, index: Foliage.nearMapIndex) }
        var w = wind(cam: .zero)
        enc.setVertexBytes(&w, length: MemoryLayout<FoliageWindGPU>.stride, index: Foliage.nearWindIndex)
    }

    /// The LOD's draw (mmc_lod_draw, with wave): the wind for a camera at `cam`.
    func bindLod(_ enc: MTLRenderCommandEncoder, cam: SIMD3<Double>) {
        var w = wind(cam: cam)
        enc.setVertexBytes(&w, length: MemoryLayout<FoliageWindGPU>.stride, index: Foliage.lodWindIndex)
    }
}

/// Bits: 1 leaflight (with lit) on, 2 wave on, 4 wavelod on; the near chunks want the sprites' classes with 1 or 2.
@_cdecl("mmc_foliage_enabled")
public func mmc_foliage_enabled() -> Int32 { (foliageLight ? 1 : 0) | (foliageWave ? 2 : 0) | (foliageWaveLod ? 4 : 0) }

/// The block atlas's foliage sprites, from the Java side (FoliageSprites): `view` the atlas's texture view handle (0 for
/// none), its size in pixels, and `count` x 5 ints: x, y, width, height (pixels), code (class | sway << 2). Render thread.
@_cdecl("mmc_foliage_set_sprites")
public func mmc_foliage_set_sprites(_ view: Int64, _ width: Int32, _ height: Int32, _ rects: UnsafePointer<Int32>, _ count: Int32) {
    guard foliageSprites else { return }
    let atlas = view == 0 ? nil : (from(view) as TextureBox).texture
    Foliage.shared.setSprites(atlas: atlas, width: Int(width), height: Int(height), rects: rects, count: Int(count))
}

/// Offline: holds the wind's clock at `t` seconds (below 0: the clock's).
@_cdecl("mmc_debug_foliage_time")
public func mmc_debug_foliage_time(_ t: Double) {
    Foliage.shared.time = t
}

/// Offline A/B (leaflight): 0 traces leaves as opaque again, 1 (default) lets the sun through them. Returns 1 with
/// leaflight on.
@_cdecl("mmc_debug_foliage_rays")
public func mmc_debug_foliage_rays(_ on: Int32) -> Int32 {
    RtShadows.shared.leafRays = on != 0
    return foliageLight ? 1 : 0
}

/// Offline timing (leaflight): the ray-traced shadows' pass alone (the GI cache's frame left out), `runs` command buffers
/// each with the sun let through leaves and with leaves opaque, alternating, on this frame's depth and G-buffer (call
/// after a frame that drew the level). Arguments as mmc_shadows_apply's. out: the medians (through, opaque), then the
/// fastest (through, opaque), GPU ms.
@_cdecl("mmc_debug_foliage_rt_time")
public func mmc_debug_foliage_rt_time(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>, _ cam: UnsafePointer<Double>,
                                      _ sunAngle: Float, _ runs: Int32, _ out: UnsafeMutablePointer<Double>) {
    for i in 0..<4 { out[i] = -1 }
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    let rt = RtShadows.shared
    let savedGi = rt.giOn, savedLeaf = rt.leafRays
    rt.giOn = false
    defer { rt.giOn = savedGi; rt.leafRays = savedLeaf }
    var through: [Double] = [], opaque: [Double] = []
    for k in 0..<(2 * Int(runs)) {
        rt.leafRays = k % 2 == 0
        guard rt.apply(color: color, depth: depth, p: p, cam: SIMD3(cam[0], cam[1], cam[2]), sunAngle: sunAngle, strength: 0.42,
                       cloudHeight: 192, deferToTaa: false), let cb = ctx.cb else { break }
        ctx.cb = nil
        cb.commit()
        cb.waitUntilCompleted()
        let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        if k % 2 == 0 { through.append(ms) } else { opaque.append(ms) }
    }
    through.sort()
    opaque.sort()
    if !through.isEmpty { out[0] = through[through.count / 2]; out[2] = through[0] }
    if !opaque.isEmpty { out[1] = opaque[opaque.count / 2]; out[3] = opaque[0] }
}

/// The LOD's source as LodRenderer.makePipeline expands it (its material names replaced), for the offline compile check.
func lodShaderSourceForCheck() -> String {
    lodShaderSource
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
}

/// Debug: the shaders foliage changes (the LOD's, the near chunks', the ray-traced shadows'), for an offline compile
/// check: `which` 0 the LOD, 1 the near chunks, 2 the shadows. Returns the source's length (written if it fits).
@_cdecl("mmc_debug_foliage_shader_source")
public func mmc_debug_foliage_shader_source(_ which: Int32, _ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let src: String
    switch which {
    case 0: src = lodShaderSourceForCheck()
    case 1: src = nearShaderSource
    default: src = rtShadowSourceForCheck()
    }
    let bytes = Array(src.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}
