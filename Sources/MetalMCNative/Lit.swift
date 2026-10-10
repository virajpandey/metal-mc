import Foundation
import Metal
import simd

// Lit mode (METALMC_EXP=lit, prototype, off by default; docs/lighting-design.md, "Lit mode"): terrain relit after the
// level is drawn, step 1 of the lighting milestone. The terrain shaders we own (the LOD's quads, the far field's march,
// the near chunks) write a G-buffer beside their color in the level's main pass: what a lighting pass needs to light
// the pixel again. After the level, and after the ray-traced shadows (whose visibility it uses), a full-screen pass
// replaces the color of every pixel where that terrain is still what shows (its depth still matches) with
//
//     albedo x (sun x max(N.L, 0) x visibility + sky x sky light x AO + moon + block light x AO)
//
// in linear light, then the aerial perspective and the anti-aliasing carry on as before. The forward color stays what
// vanilla would draw: it's what shows wherever something nearer was drawn over the terrain (entities, water, clouds,
// particles), and what translucent layers blend with.
//
// Things drawn over terrain without writing depth (rain, block cracks, entity shadows, translucent layers with improved
// transparency) leave its depth as it was, so the pass also checks the color: if it isn't the forward color the stored
// albedo gives (vanilla's light, worked out again from the G-buffer), something was blended over it, and what's there is
// relit instead (its color divided by vanilla's light, times the new light), so the overlay stays.
//
// The G-buffer is one RG32Uint target: 8 bytes a pixel, the bandwidth of two RGBA8 targets (61.8 MB stored at the end of
// the main pass and read once by the relight at 3456 x 2234, against 123.5 MB each way for RGBA32Uint), in one attachment:
//   x: albedo r | g << 8 | b << 16 (sRGB-encoded 8 bits: the texture or material color with its tint, before light, face
//      shade and AO) | AO << 24 (5 bits) | face << 29 (0: not lit terrain, the clear value; 1-6: +X -X +Y -Y +Z -Z;
//      7: a face that isn't axis-aligned, like cross plants)
//   y: depth key (16 bits, the depth the terrain wrote folded in half) | sky light << 16 | block light << 24 (light levels
//      x 16, vanilla's light coordinates as the shader has them: smooth lighting interpolates them across a face)
// The exact depth would take 32 bits and the format to 16 bytes. A 16-bit key misses one depth change in 65,536 (a pixel
// of something drawn over terrain then shows the relit terrain for a frame).
// With leaflight (METALMC_EXP=leaflight, Foliage.swift) the two low bits of the albedo's blue byte (x bits 16-17) hold a
// foliage class on lit terrain (face code 1-7): 0 anything else, 1 a leaf (the LOD's leaf materials; the near chunks'
// leaf sprites), 2 a plant (grass, ferns, flowers, crops, saplings: the near chunks' sprites). litPack rounds blue to a
// multiple of 4 (at most 2/255 off) so every writer leaves 0 there; read it as litFoliageClass(g), or
// (g.x >> 29) != 0 ? (g.x >> 16) & 3 : 0. Without leaflight those bits are the albedo's, as before.
//
// Hooks: mmc_pass_begin adds the G-buffer to the pass the Java side marks (mmc_lit_level_pass, from LevelRenderer's main
// pass); PipelineBox gives vanilla's pipelines a variant with the extra target and no writes; the LOD, far field and
// near chunk shaders are compiled with LIT_MODE 1 and write it; RtShadows stores the raw visibility; mmc_lit_relight
// runs the pass. Without METALMC_EXP=lit none of it is compiled in or attached.
//
// With the GI cache (METALMC_EXP=lit,gi, Gi.swift): RtShadows runs the cache after its shadow rays and leaves its
// half-resolution light, and where the cache has data for a pixel (giUpsample) it replaces the sky term (the open sky's
// light on the face times the sky light level's curve): the sky as the real openings let it in, plus bounced light,
// times AO. Everything else is as above. Without gi the shaders are the same text as before (the GI parts are spliced in
// only with litGi).
//
// Water (METALMC_EXP=lit,water, with our sky): the writers that know water (the far field's march; the LOD's water
// pipeline once it writes litPackWater) mark its surface in the G-buffer, as a texel that isn't lit terrain (face code
// 0) with a flag bit and a depth key, and the relight reflects the sky and the sun in it (litWaterPixel, after
// litRelightPixel, before the aerial perspective): the water's color as drawn, mixed toward the sky in the reflected
// direction by Fresnel's reflectance, plus the sun's glint, over small procedural waves. Without water the shaders are
// the same text as before (its parts are spliced in only with litWater), and so is every pixel.

/// METALMC_EXP=lit: deferred relighting of terrain.
let litEnabled = experiments.contains("lit")

/// METALMC_EXP=water (with lit): water surfaces flagged in the G-buffer and the sky and the sun reflected in them, where
/// our sky gives the atmosphere's light (Sky.swift). Off by default until it's been seen in the game.
let litWater = litEnabled && experiments.contains("water")

/// The G-buffer's color attachment in the level's main pass (after vanilla's one color target) and its format.
let litGbufferIndex = 1
let litGbufferFormat = MTLPixelFormat.rg32Uint

/// True for the G-buffer's attachment in a pass's color formats: the terrain pipelines write it there (vanilla's don't).
@inline(__always) func litWritesGbuffer(_ index: Int, _ format: MTLPixelFormat) -> Bool {
    litEnabled && index == litGbufferIndex && format == litGbufferFormat
}

/// METALMC_LITEXPOSURE: scales the new light (1: a sunlit top at noon about as bright as vanilla's).
private let litExposure = max(0, Float(ProcessInfo.processInfo.environment["METALMC_LITEXPOSURE"] ?? "") ?? 1)

/// The G-buffer's packing, shared by the terrain shaders (their sources define LIT_MODE 0 or 1 and include this).
let litShaderHeader = """
#if LIT_MODE
// Lit mode's G-buffer (Lit.swift). The face is the LOD's numbering (0-5: +X -X +Y -Y +Z -Z), 7 for a face that isn't
// axis-aligned, LIT_NONE for a pixel that isn't lit terrain (water).
#define LIT_NONE 8u
// The depth key: bits 4-19 of the depth (the middle of its mantissa). The depth unit doesn't always store the fragment
// shader's position.z to the last bit (on the M3 about 1% of a frame's LOD pixels differ by a few steps of it), so
// keys match within one step of 16: the depth to within 16-32 of its last bit, about 4e-6 of the distance.
static uint litDepthKey(float z) {
    return (as_type<uint>(z) >> 4) & 0xFFFFu;
}
static bool litDepthMatches(float z, uint key) {
    uint d = (litDepthKey(z) - key) & 0xFFFFu;
    return d <= 1u || d == 0xFFFFu;
}
static uint2 litPack(float3 albedo, float ao, uint face, float depth, float sky, float block) {
    if (face == LIT_NONE) return uint2(0u);
    uint3 a = uint3(round(saturate(albedo) * 255.0));\(foliageLight ? "\n    a.b = min((a.b + 2u) & ~3u, 252u);   // leaflight: blue's two low bits hold the foliage class (litPackFoliage)" : "")
    uint code = face < 6u ? face + 1u : 7u;
    return uint2(a.r | (a.g << 8) | (a.b << 16) | (uint(round(saturate(ao) * 31.0)) << 24) | (code << 29),
                 litDepthKey(depth) | (uint(round(clamp(sky, 0.0, 15.0) * 16.0)) << 16) | (uint(round(clamp(block, 0.0, 15.0) * 16.0)) << 24));
}
\(foliageLight ? foliagePackHeader : "")\(litWater ? litWaterPackHeader : "")#endif
"""

/// With water (litWater only; spliced into litShaderHeader, so without it every shader is the same text as before): its
/// texel in the G-buffer. Terrain texels always have a face code (bits 29-31) of 1-7 and the clear value is 0, so a code
/// of 0 with bit 28 set is free: every reader that doesn't know water (the relight's terrain path, the offline checks)
/// sees "not lit terrain" there, as before. AO keeps its 5 bits.
private let litWaterPackHeader = """
// Water (METALMC_EXP=water): a water surface. x: the water layer's color as it was blended over what's under it
// (sRGB-encoded 8 bits a channel: the texture times the vertex color, vanilla's light in it) in bits 0-23, the face (0-5,
// the LOD's numbering) in bits 24-26, bit 28 set, face code 0; y: the depth key (so a boat or an entity drawn over the
// water isn't taken for it), the layer's alpha (0: not known, the writer didn't say) in bits 16-23, the sky light level
// (whole levels: it tells open water from water under cover) in 24-27, block light in 28-31. Vanilla's water writes its
// layer (Water.swift's patched fragment shader packs the same), so the relight can take it off again and see the floor.
#define LIT_WATER 1
#define LIT_WATER_FLAG (1u << 28)
static uint2 litPackWaterLayer(uint face, float depth, float sky, float block, float4 layer) {
    uint3 w = uint3(round(saturate(layer.rgb) * 255.0));
    return uint2(w.r | (w.g << 8) | (w.b << 16) | LIT_WATER_FLAG | (min(face, 7u) << 24),
                 litDepthKey(depth) | (uint(round(saturate(layer.a) * 255.0)) << 16)
                 | (uint(clamp(round(sky), 0.0, 15.0)) << 24) | (uint(clamp(round(block), 0.0, 15.0)) << 28));
}
static uint2 litPackWater(uint face, float depth, float sky, float block) { return litPackWaterLayer(face, depth, sky, block, float4(0.0)); }
static bool litIsWater(uint2 g) { return (g.x >> 28) == 1u; }

"""

/// With the GI cache (litGi only; empty otherwise, so the relight's text is unchanged without it): litRelightPixel's
/// extra arguments, its early read of the cache's texel, and the sky term taken from the cache.
private let litGiRelightArgsDoc = litGi ? """

// gi: the GI cache's half-resolution light and code words (gi_resolve's RG32Uint; a 1 x 1 stand-in when it didn't run).
// giStandIn: unused (where the light was before it shared a texel with its code; the anti-aliasing's resolve still binds
// one there).
""" : ""
private let litGiRelightArgs = litGi ? ", texture2d<float> giStandIn, texture2d<uint> gi" : ""
/// With colored block light (clEnabled only; ColoredLight.swift): litRelightPixel's extra arguments, the volume's two
/// textures and frame, and its block light in place of the lightmap's.
private let litClRelightArgs = clEnabled ? ", texture3d<float> clRGB, texture3d<float> clAux, constant ClFrame& clf, texture2d<half> clSh" : ""
private let litClBlockLight = clEnabled ? """

        // Colored block light (ColoredLight.swift): the light volume in place of the lightmap, checked against vanilla's level.
        blockLight = clBlockLight(blockLight, lm00, block, rel, fi, n, clf, clRGB, clAux, clSh, q, clDbg, view);
        clBL = blockLight;
""" : ""
private let litGiTexel = litGi ? """

    // The GI cache's texel for the pixel (giUpsample's common case needs no other), read now: its address depends on
    // nothing else, so its latency hides behind the work below.
    uint2 giOwn = giUpsampleTexel(gi, q);
""" : ""
private let litGiSkyTerm = litGi ? """

        // The GI cache (Gi.swift) where it has data for the pixel's face and plane: the sky's light as the real openings
        // let it in, plus bounced light, in place of the sky light level's curve. In env's units (the cache's light is
        // made from it), so the daylight curve's scale and the exposure apply as before. Faces that aren't axis-aligned
        // (plants) take the top face's beside them. Debug view 8: green where it applied, red where it didn't, blue for
        // light sources.
        if (gi.get_width() > 1u) {
            float4 cl = giUpsample(gi, giOwn, q, fi < 6u ? fi : 2u, rel);
            if (cl.w > 0.0) { skyAmb = cl.rgb * ao; giUsed = true; }
        }\(clEnabled ? "\n        // Colored block light bounced through the cache (ColoredLightBounce.swift), where vanilla's level isn't 0.\n        if (giStandIn.get_width() > 1u) blockLight += float3(giStandIn.read(giOwn).rgb) * saturate(block);" : "")
""" : ""

/// With the post chain (postEnabled only, Post.swift; empty otherwise, so lit mode's text is unchanged without it): light
/// sources brighter than vanilla's white, so they bloom and run white-hot through the tone curve.
private let litEmitterTerm = postEnabled ? """
    // Light sources (Post.swift, METALMC_EXP=post): their faces are flat-lit at the source's own level. Level 15 (glowstone,
    // lava, lanterns, fire, magma, sea lanterns) is theirs alone: smooth lighting averages four blocks, and next to a 15
    // they're 14 or less. LIT_EMIT times their light, in proportion to the texel's brightness: lava's glowing cracks more
    // than its crust, a lantern's flame more than its frame. Level 14 is a torch's, but also the face a lantern or a
    // glowstone stands on (15, 14, 14 and 13 averaged): there only near-white texels count (a flame's yellow-white core,
    // about 0.97; sand's brightest texels, about 0.88, speckled at 0.88-0.97).
#ifndef LIT_EMIT
#define LIT_EMIT 6.0
#endif
    if (block >= 13.97) {
        float emit = smoothstep(block >= 14.97 ? 0.35 : 0.93, block >= 14.97 ? 0.8 : 0.99, litLuma(albedo));
        if (emit > 0.0) E = max(E, float3(1.0)) * (1.0 + (LIT_EMIT - 1.0) * emit);
    }

""" : ""

/// The relight's per-pixel work (litRelightPixel), shared by its own pass and by the anti-aliasing's resolve (Taa.swift),
/// which applies it as it loads each pixel when anti-aliasing is on. Needs skyShaderHeader, then LIT_MODE 1 and
/// litShaderHeader (and with litGi, giUpsampleHeader), before it.
let litRelightHeader = """

struct LitFrame {
    float4x4 invViewProj;   // clip space to camera-relative world (the projection the level was drawn with, jittered)
    float4 size;            // xy: target size, z: pixels per traced visibility sample (0: none this frame), w: largest value the target holds
    float4 sunDir;          // xyz: toward the sun, w: 1 if this frame's visibility was traced toward the moon
    float4 moonDir;         // xyz: toward the moon, w: how much it's night (0 by day, 1 at night)
    float4 fogColor;        // vanilla's fog color, a: its strength (0: no fog)
    float4 fog;             // vanilla's fog: environmental start, end, render-distance start, end
    float4 misc;            // x: 1 if vanilla's lightmap is bound, y: 1 to scale the sun and sky by vanilla's daylight (no atmosphere), z: debug view, w: exposure
\(litWater ? "    float4 water;           // water: x the waves' time (s), y-z the camera's x and z modulo WATER_TILE, w 1 to reflect (0: offline A/B)\n    float4 water2;          // water: x radians per pixel at the screen's center, y how much of the sun's disk the sky draws (0 in rain)\n" + waterFrameFields : "")};

constant float3 kLitNormal[6] = { float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0), float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1) };
// Vanilla's face shade (the LOD's kShade): the forward color has it, so the overlay test works it out again.
constant float kLitShade[7] = { 0.6, 0.6, 1.0, 0.5, 0.8, 0.8, 1.0 };
// How much of the sky a face sees, for the moon's glow: all of it from a top, more than half from a side (the sky's glow
// is brightest low down), the ground's faint share from a bottom; faces that aren't axis-aligned (plants) in between.
constant float kLitHemi[7] = { 0.6, 0.6, 1.0, 0.2, 0.6, 0.6, 0.75 };
// The moon's light, bluer than vanilla's night (luminance 1).
constant float3 kLitMoonTint = float3(0.857, 1.006, 1.371);

static float litLuma(float3 c) { return dot(c, float3(0.2126, 0.7152, 0.0722)); }

// Vanilla's lightmap at fractional levels (texel centers at whole ones, linear between them).
static float3 litLightmap(texture2d<float> lm, float block, float sky) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return lm.sample(s, (float2(block, sky) + 0.5) / 16.0, level(0.0)).rgb;
}

// Vanilla's brightness for a light level (lightmap.fsh's get_brightness).
static float litBrightness(float level) {
    float x = saturate(level / 15.0);
    return x / (4.0 - 3.0 * x);
}

static float litFogValue(float d, float s, float e) {
    if (d <= s) return 0.0;
    if (d >= e) return 1.0;
    return (d - s) / (e - s);
}
\(clEnabled ? clRelightHeader + "\n" : "")\(foliageLight ? foliageRelightHeader : "")
// The relight of pixel q, whose color is dst (as the target holds it), depth d and G-buffer texel g: the color it gets.
// Pixels that aren't lit terrain keep theirs. env: the sun's light (0), the sky's on each face direction (1-6, the LOD's
// face order) and on faces that aren't axis-aligned (7), linear, per unit of albedo.\(litGiRelightArgsDoc)
static float3 litRelightPixel(float3 dst, uint2 q, float d, uint2 g, texture2d<half, access::read> vis, texture2d<float> lightmap,
                              constant LitFrame& f, constant float4* env\(litGiRelightArgs)\(litClRelightArgs)) {
    uint code = g.x >> 29;
    uint view = uint(f.misc.z);
    if (code == 0u) return view != 0u ? float3(0.0) : dst;
    // Something nearer was drawn over the terrain since (an entity, water, a cloud): its color stays.
    if (!litDepthMatches(d, g.y & 0xFFFFu)) return view != 0u ? float3(0.2, 0.0, 0.2) : dst;\(litGiTexel)
    float3 albedo = float3(float(g.x & 255u), float((g.x >> 8) & 255u), float((g.x >> 16) & 255u)) / 255.0;
    float ao = float((g.x >> 24) & 31u) / 31.0;
    float sky = float((g.y >> 16) & 255u) / 16.0, block = float(g.y >> 24) / 16.0;
    uint fi = code - 1u;   // 0-5, or 6 for a face that isn't axis-aligned
    float3 n = fi < 6u ? kLitNormal[fi] : float3(0.0, 1.0, 0.0);

    // Vanilla's fog, as the terrain shaders applied it (pushed out of reach while our sky is on).
    float2 ndc = (float2(q) + 0.5) / f.size.xy * 2.0 - 1.0;
    float4 h = f.invViewProj * float4(ndc, d, 1.0);
    float3 rel = h.xyz / h.w;
    float fog = 0.0;
    if (f.fogColor.a > 0.0) {
        fog = max(litFogValue(length(rel), f.fog.x, f.fog.y), litFogValue(max(length(rel.xz), abs(rel.y)), f.fog.z, f.fog.w)) * f.fogColor.a;
    }

    // Vanilla's light for the pixel (what the forward color has): the lightmap at its levels, the face's shade, AO.
    bool lm = f.misc.x > 0.5;
    float3 vlight = lm ? litLightmap(lightmap, block, sky) : float3(max(litBrightness(block), litBrightness(sky)));
    float3 V = vlight * (kLitShade[fi] * ao);
    // The color as the terrain left it, before the fog. If it isn't the forward color the albedo gives, something was
    // blended over the terrain without writing depth. One that darkened it (entity shadows, block cracks) is relit with
    // the surface: its reflectance against vanilla's light (the color over vanilla's light) takes the new light. One that
    // brightened it (rain, snow) keeps its own light, and the surface's change in light is added to it (overlayBright).
    float3 c0 = fog > 0.0 ? (dst - fog * f.fogColor.rgb) / max(1.0 - fog, 1e-3) : dst;
    float3 fwd = albedo * V;
    // (In thick fog the color left is mostly the fog's: no test.)
    bool plain = fog > 0.9 || all(abs(c0 - fwd) <= fwd * 0.06 + 2.5 / 255.0);
    bool overlayBright = !plain && litLuma(c0) > litLuma(fwd);
    float3 refl = plain || overlayBright ? albedo : saturate(c0 / max(V, float3(1.0 / 255.0)));

    // The new light, linear, per unit of albedo.
    float3 E;
    float vSunOut = 0.0;\(litGi ? "\n    bool giUsed = false;" : "")\(clEnabled ? "\n    float4 clDbg = float4(0.0);\n    float3 clBL = float3(0.0);" : "")
    if (block >= 15.0) {
        // Light sources (vanilla gives emissive faces full block light): full bright, like vanilla.
        E = float3(1.0);
    } else {
        float3 lm00 = lm ? skyDecode(litLightmap(lightmap, 0.0, 0.0)) : float3(0.0);
        // Vanilla's block light, with the dimension's ambient, night vision and the darkness effect, as its lightmap has it.
        float3 blockLight = lm ? skyDecode(litLightmap(lightmap, block, 0.0)) : float3(litBrightness(block) * litBrightness(block));\(litClBlockLight)
        // Vanilla's sky light at full sky (its day and night curve, rain and thunder darkening), for the moon and, without
        // our sky, the daylight curve.
        float3 vanillaSky = lm ? max(skyDecode(litLightmap(lightmap, 0.0, 15.0)) - lm00, 0.0) : float3(1.0);
        float dayScale = f.misc.y > 0.5 ? litLuma(vanillaSky) : 1.0;
        // How much of the open sky the sky light level stands for, in linear light (vanilla's curve, decoded).
        float skyFall = skyDecode(float3(litBrightness(sky))).x;
        // Visibility of the sun and moon: traced (toward one of them) where there are rays, else the open sky (sky light
        // 15; 12 or less, under trees or an overhang, counts as shade). Enclosed spaces never see either.
        float open = saturate((sky - 12.0) / 3.0);
        float traced = open;
        if (f.size.z > 0.0) {
            uint s = uint(f.size.z);
            traced = float(vis.read(min(q / s, uint2(vis.get_width() - 1, vis.get_height() - 1))).r) * saturate(sky / 2.0);
        }
        bool moonTraced = f.sunDir.w > 0.5;
        float vSun = moonTraced ? open : traced, vMoon = moonTraced ? traced : open;
        vSunOut = vSun;
        float ndlSun = fi < 6u ? max(dot(n, f.sunDir.xyz), 0.0) : 0.25 + 0.5 * saturate(f.sunDir.y);
        float ndlMoon = fi < 6u ? max(dot(n, f.moonDir.xyz), 0.0) : 0.25 + 0.5 * saturate(f.moonDir.y);
        float3 sun = env[0].rgb * ndlSun * vSun;
        float3 skyAmb = env[1u + fi].rgb * skyFall * ao;\(litGiSkyTerm)
        // The moon, calibrated to vanilla's night: a top face under a high moon gets vanilla's night sky light, 60% of it
        // from the moon itself, 40% from the glow of the night sky.
        float moonLum = litLuma(vanillaSky) * f.moonDir.w;
        float3 moon = moonLum * kLitMoonTint * (0.6 * ndlMoon / max(f.moonDir.y, 0.5) * vMoon + 0.4 * kLitHemi[fi] * skyFall * ao);
        E = ((sun + skyAmb) * dayScale + moon + blockLight * ao) * f.misc.w;
\(foliageLight ? foliageRelightCall : "")    }
\(litEmitterTerm)    float3 lin = skyDecode(refl) * E;
    float3 o = skyEncode(lin);
    if (overlayBright) o = c0 + (o - fwd);
    if (f.size.w <= 1.0) o = saturate(o);
    o = max(o, 0.0);
    if (fog > 0.0) o = mix(o, f.fogColor.rgb, fog);
    if (view != 0u) {
        // Debug views (METALMC_LITVIEW): 1 albedo, 2 face, 3 light levels (block red, sky green), 4 AO, 5 the new light alone
        // (on a mid-gray albedo), 6 sun visibility, 7 the overlay test (green: plain terrain, red: something blended over it).
        float3 c = float3(0.0);
        if (view == 1u) c = albedo;
        else if (view == 2u) c = fi < 6u ? n * 0.5 + 0.5 : float3(1.0, 1.0, 0.0);
        else if (view == 3u) c = float3(block / 15.0, sky / 15.0, 0.0);
        else if (view == 4u) c = float3(ao);
        else if (view == 5u) c = skyEncode(skyDecode(float3(0.5)) * E);
        else if (view == 6u) c = float3(vSunOut);
        else if (view == 7u) c = plain ? float3(0.0, 0.8, 0.0) : (overlayBright ? float3(0.9, 0.0, 0.9) : float3(0.9, 0.0, 0.0));\(litGi ? "\n        else if (view == 8u) c = giUsed ? float3(0.0, 0.8, 0.0) : (block >= 15.0 ? float3(0.0, 0.0, 0.9) : float3(0.9, 0.0, 0.0));" : "")\(clEnabled ? "\n        else if (view == 11u) c = skyEncode(skyDecode(float3(0.5)) * clBL * f.misc.w);\n        else if (view == 12u) c = clDbg.rgb;" : "")\(foliageLight ? foliageDebugViews : "")
        return saturate(c);
    }
    return o;
}\(litWater ? "\n" + waterShadeHeader : "")
"""

/// Water's look (litWater): METALMC_WATERWAVES scales the waves' slopes (0.6 since round 2, Water.swift: calmer; 0: a flat
/// surface), METALMC_WATERROUGH is the
/// surface's own roughness under them (GGX alpha: the ripples finer than the waves), which keeps a sun glint on the
/// flattest water.
let waterWaveScale = max(0, Double(ProcessInfo.processInfo.environment["METALMC_WATERWAVES"] ?? "") ?? 0.6)
let waterRough = min(max(Double(ProcessInfo.processInfo.environment["METALMC_WATERROUGH"] ?? "") ?? 0.05, 0.005), 1)
/// The waves' tile: waterTile blocks a side, in a texture of waterTexels a side (4 a block: the shortest wave, 1.4 blocks,
/// spans 5.6), repeated over the water. Every wave has a whole number of wavelengths across it along x and along z, so the
/// tile repeats without a seam, and the camera's position can be taken modulo it (float precision).
let waterTile: Double = 64
let waterTexels = 256
/// Its mip levels, down to 1 x 1, and how many of them (from the top) hold a wave: the rest, whose texels are at least
/// half the longest wave apart, have every wave faded and don't change with time (lit_water_waves).
let waterLevels = Int(log2(Double(waterTexels))) + 1
private let waterMovingLevels: Int = {
    let longest = waterWaveNumbers.map { $0.lambda }.max() ?? 1
    return (0..<waterLevels).first { waterTile / Double(waterTexels >> $0) >= longest / 2 } ?? waterLevels
}()
/// The sky map's size (a paraboloid map of the upper hemisphere): 256 a side puts a texel at about half a degree at the
/// horizon, where the sky changes fastest (the sky view table is finer there, but a reflection's waves blur it anyway).
let waterSkyTexels = 256
/// Small waves on water: wavelength (blocks, which are metres), direction (degrees from +x toward +z), slope amplitude
/// (radians, before METALMC_WATERWAVES) and phase. Wind waves on a lake: a few metres long, a degree or two steep each,
/// moving at deep water's speeds (angular frequency sqrt(g k)); together a slope of 0.053 RMS. Two to an octave, their
/// directions spread 70 degrees either side of the wind's: with one wave to an octave (a first try, 5 waves), the longest
/// ones, the last left where the shorter have faded, drew straight parallel stripes. Wavelengths and directions move to
/// the nearest whole wave numbers across the tile (10.1 blocks at 18 degrees, 8.2 at -40, 6.2 at 61, 4.8 at -13, 3.7 at
/// 35, 2.7 at -68, 1.9 at 9, 1.4 at 52).
let waterWaveSet: [(lambda: Double, angle: Double, steep: Double, phase: Double)] = [
    (10.0, 18, 0.032, 0.0), (8.3, -42, 0.032, 2.9), (6.1, 64, 0.028, 1.3), (4.9, -12, 0.028, 4.6),
    (3.6, 36, 0.024, 3.7), (2.7, -68, 0.024, 0.8), (1.9, 8, 0.02, 5.1), (1.4, 52, 0.02, 2.2)]
/// The waves' whole wave numbers across the tile (along x and z) and the wavelength (blocks) they come to.
let waterWaveNumbers: [(n: Double, m: Double, lambda: Double)] = waterWaveSet.map { w in
    let n = (waterTile / w.lambda * cos(w.angle * .pi / 180)).rounded(), m = (waterTile / w.lambda * sin(w.angle * .pi / 180)).rounded()
    return (n, m, waterTile / (n * n + m * m).squareRoot())
}

/// lit_relight_fs's last line, and with water the lines in its place: the reflections after the relight (this frame's sky
/// map and waves' tile are bound then, or stand-ins without our sky, when litWaterPixel leaves water alone).
private let litRelightFsReturn = "    return float4(litRelightPixel(dst.rgb, q, depth.read(q), g, vis, lightmap, f, env\(litGi ? ", giStandIn, gi" : "")\(clEnabled ? ", clRGB, clAux, clf, clSh" : "")), dst.a);"
private let litRelightFsWater = """
    float d = depth.read(q);
    float3 c = litRelightPixel(dst.rgb, q, d, g, vis, lightmap, f, env\(litGi ? ", giStandIn, gi" : "")\(clEnabled ? ", clRGB, clAux, clf, clSh" : ""));
    c = litWaterWet(c, q, d, g, f, waterSky);
    c = litWaterPixel(c, q, d, g, vis, f, env, waves, wavesDetail, waterSky, lightmap, waterTrace, waterTraceAS, waterColor, gbuf, depth\(litGi ? ", gi" : ""));
    return float4(litWaterFog(c, q, d, g, f, env, lightmap, vis, waves, wavesDetail), dst.a);
"""

/// With water: the kernels that make its waves' tile and sky map each frame (litWaterPixel samples them), appended to lit
/// mode's own library.
private let litWaterKernels = """

// Water's waves (METALMC_EXP=water): one mip level of their tile at time t (out: a view of that level), one thread a
// texel: the slopes of the waves the level can hold, each whole while its wavelength spans 4 of its texels and gone at 2,
// and the slopes' mean squared length, which adds the variance of the waves faded out. Every level is made this way, not
// averaged down from the top: a 2 x 2 box filter leaves waves near a level's limit in it, and the longest waves' crests
// then showed as rays converging on the horizon over mid-distance water.
kernel void lit_water_waves(texture2d<float, access::write> out [[texture(0)]], constant float& t [[buffer(0)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float spacing = WATER_TILE / float(out.get_width());
    float2 p = (float2(gid) + 0.5) * spacing;
    float2 s = 0.0;
    float faded = 0.0;
    for (uint i = 0; i < \(waterWaveSet.count)u; i++) {
        float4 w = kWaterWave[i], a = kWaterAmp[i];
        float fade = saturate(2.0 - spacing * w.w);
        s += (fade * cos(dot(w.xy, p) - w.z * t + a.z)) * a.xy;
        faded += a.w * (1.0 - fade * fade);
    }
    out.write(float4(s, dot(s, s) + faded, 0.0), gid);
}

// Water's sky map: the sky's light (skyLuminance, scene units) over the upper hemisphere, one thread a texel.
kernel void lit_water_sky(texture2d<float, access::write> out [[texture(0)]], constant SkyFrame& f [[buffer(0)]],
                          texture2d<float> skyView [[texture(1)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float2 uv = (float2(gid) + 0.5) / float2(out.get_width(), out.get_height());
    out.write(float4(skyLuminance(f, litWaterSkyDir(uv), skyView), 1.0), gid);
}
"""

private let litShaderSource = skyShaderHeader + """

#define LIT_MODE 1
\(litShaderHeader)\(litGi ? giUpsampleHeader : "")
\(litRelightHeader)

struct LitVOut { float4 pos [[position]]; };

vertex LitVOut lit_fullscreen_vs(uint vid [[vertex_id]]) {
    LitVOut o;
    float2 c = float2(float((vid << 1) & 2), float(vid & 2));
    o.pos = float4(c * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// The relight as a pass of its own (without anti-aliasing; with it, its resolve does this as it loads the color). Reads
// the color it replaces (programmable blending); pixels that aren't lit terrain cost a G-buffer read and keep their color.
fragment float4 lit_relight_fs(LitVOut in [[stage_in]], float4 dst [[color(0)]],
                               depth2d<float, access::read> depth [[texture(0)]],
                               texture2d<uint, access::read> gbuf [[texture(1)]],
                               texture2d<half, access::read> vis [[texture(2)]],
                               texture2d<float> lightmap [[texture(3)]],\(litGi ? "\n                               texture2d<float> giStandIn [[texture(4)]],\n                               texture2d<uint> gi [[texture(5)]]," : "")
                               constant LitFrame& f [[buffer(0)]],
                               constant float4* env [[buffer(1)]]\(litWater ? ",\n                               texture2d<float> waterSky [[texture(6)]],\n                               texture2d<float> waves [[texture(7)]],\n                               texture2d<float> waterTrace [[texture(11)]],\n                               texture2d<float> wavesDetail [[texture(13)]],\n                               texture2d<uint> waterTraceAS [[texture(14)]],\n                               texture2d<float, access::read> waterColor [[texture(12)]]" : "")\(clEnabled ? ",\n                               texture3d<float> clRGB [[texture(8)]],\n                               texture3d<float> clAux [[texture(9)]],\n                               constant ClFrame& clf [[buffer(2)]],\n                               texture2d<half> clSh [[texture(10)]]" : "")) {
    uint2 q = uint2(in.pos.xy);
    uint2 g = gbuf.read(q).rg;
    if ((g.x >> 29) == 0u && f.misc.z == 0.0\(litWater ? " && !litIsWater(g)" : "")) return dst;
\(litWater ? litRelightFsWater : litRelightFsReturn)
}

// The sun's light and the sky's on each face direction, from the atmosphere's tables (with our sky on), scaled by p.x so
// the sun at the zenith through the air at this altitude, plus about 12% for the sky, comes to 1 at noon's exposure:
// vanilla's sunlit top at noon. One threadgroup of 64 threads, 8 directions each over the upper hemisphere (uniform in
// solid angle, a spiral); below the horizon the ground, gray (the atmosphere's ground albedo) and lit by both.
kernel void lit_env(device float4* out [[buffer(0)]], constant SkyFrame& f [[buffer(1)]], constant float4& p [[buffer(2)]],
                    texture2d<float> trans [[texture(0)]], texture2d<float> skyView [[texture(1)]],
                    uint i [[thread_index_in_threadgroup]]) {
    threadgroup float4 acc[64][5];
    float3 s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0;
    for (uint k = 0; k < 8; k++) {
        uint j = i * 8u + k;
        float y = (float(j) + 0.5) / 512.0;
        float r = sqrt(max(0.0, 1.0 - y * y));
        float phi = float(j) * 2.39996323;
        float3 dir = float3(r * cos(phi), y, r * sin(phi));
        float3 L = skyLuminance(f, dir, skyView);
        s0 += L * dir.y;
        s1 += L * max(dir.x, 0.0);
        s2 += L * max(-dir.x, 0.0);
        s3 += L * max(dir.z, 0.0);
        s4 += L * max(-dir.z, 0.0);
    }
    acc[i][0] = float4(s0, 0.0); acc[i][1] = float4(s1, 0.0); acc[i][2] = float4(s2, 0.0);
    acc[i][3] = float4(s3, 0.0); acc[i][4] = float4(s4, 0.0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 32; s > 0; s >>= 1) {
        if (i < s) {
            for (uint k = 0; k < 5; k++) acc[i][k] += acc[i + s][k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i != 0) return;
    float w = 2.0 * SKY_PI / 512.0;
    float3 up = acc[0][0].rgb * w, px = acc[0][1].rgb * w, nx = acc[0][2].rgb * w, pz = acc[0][3].rgb * w, nz = acc[0][4].rgb * w;
    float r0 = f.view.y;
    float3 sun = skyTransmittanceToSpace(trans, f.atmo, r0, f.sun.y) * skySunVisible(f.atmo, r0, f.sun.y, f.sunHoriz.z) * f.sun.w;
    // The ground's light on a face turned to it (a bottom sees all of the ground, a side half).
    float3 ground = f.atmo.planet.w * (up + sun * max(f.sun.y, 0.0));
    float3 tz = skyTransmittanceToSpace(trans, f.atmo, r0, 1.0);
    float scale = 1.0 / max(p.x * dot(tz, float3(0.2126, 0.7152, 0.0722)), 1e-6);
    out[0] = float4(sun * scale, \(litGi || litWater ? "scale" : "0.0"));
    out[1] = float4((px + 0.5 * ground) * scale, 0.0);
    out[2] = float4((nx + 0.5 * ground) * scale, 0.0);
    out[3] = float4(up * scale, 0.0);
    out[4] = float4(ground * scale, 0.0);
    out[5] = float4((pz + 0.5 * ground) * scale, 0.0);
    out[6] = float4((nz + 0.5 * ground) * scale, 0.0);
    out[7] = float4((0.5 * up + 0.125 * (px + nx + pz + nz) + 0.25 * ground) * scale, 0.0);
}\(litWater ? litWaterKernels : "")
"""

/// Mirrors LitFrame in the shader. The last two are water's (litWater); without it the shader's LitFrame ends before them
/// and doesn't read them.
private struct LitFrameGPU {
    var invViewProj = matrix_identity_float4x4
    var size = SIMD4<Float>.zero
    var sunDir = SIMD4<Float>.zero
    var moonDir = SIMD4<Float>.zero
    var fogColor = SIMD4<Float>.zero
    var fog = SIMD4<Float>.zero
    var misc = SIMD4<Float>.zero
    var water = SIMD4<Float>.zero
    var water2 = SIMD4<Float>.zero
    /// Water's second round (Water.swift, waterFrameFields).
    var waterExtra = WaterFrameGPU()
}

private func litSmoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
    let t = min(max((x - e0) / (e1 - e0), 0), 1)
    return t * t * (3 - 2 * t)
}

/// Without our sky: the sun's and the sky's light from vanilla's sun angle alone, as shares of vanilla's daylight (the
/// relight scales them by its lightmap's sky light, which follows the day, rain and thunder). A top face under a high sun
/// gets 0.82 from the sun and 0.18 from the sky; the sun warms and fades as it nears the horizon, the sky lingers through
/// twilight, and both get the sky's eye adaptation (Sky.prepare: up to 3.5 stops as the sun goes from about 15 degrees
/// up to 9 under), so dusk isn't darker than with the atmosphere. Same layout as lit_env's output.
func litDaylightEnv(sunAngle: Float) -> [SIMD4<Float>] {
    let sun = SIMD3<Float>(-sin(sunAngle), cos(sunAngle), 0)
    let e = sun.y
    let adapt = exp2(3.5 * litSmoothstep(0.25, -0.15, e))
    let warm = 1 - litSmoothstep(0.0, 0.45, e)
    let sunColor = (SIMD3<Float>(1.0, 0.97, 0.92) * (1 - warm) + SIMD3<Float>(1.0, 0.62, 0.32) * warm) * 0.82 * litSmoothstep(-0.03, 0.1, e) * adapt
    let skyColor = SIMD3<Float>(0.62, 0.78, 1.0) * (0.18 / 0.7684) * litSmoothstep(-0.2, 0.1, e) * adapt
    let groundShare: Float = 0.3 * (0.18 + 0.82 * max(e, 0))
    let ground = SIMD3<Float>(repeating: groundShare) * litSmoothstep(-0.2, 0.1, e) * adapt
    let side = skyColor * 0.5 + ground * 0.5
    var out = [SIMD4<Float>](repeating: .zero, count: 8)
    out[0] = SIMD4(sunColor, 0)
    out[1] = SIMD4(side, 0); out[2] = SIMD4(side, 0)
    out[3] = SIMD4(skyColor, 0)
    out[4] = SIMD4(ground, 0)
    out[5] = SIMD4(side, 0); out[6] = SIMD4(side, 0)
    out[7] = SIMD4(skyColor * 0.75 + ground * 0.25, 0)
    return out
}

final class Lit: @unchecked Sendable {
    static let shared = Lit()

    /// Set by mmc_lit_level_pass: the next render pass is the level's main pass and gets the G-buffer.
    var pendingLevelPass = false
    private(set) var gbuffer: MTLTexture?
    /// The color target of the pass the G-buffer went to this frame (mmc_pass_begin), checked and cleared by the relight.
    private var frameTarget: (color: ObjectIdentifier, width: Int, height: Int)?
    private var library: MTLLibrary?
    private var relightPipes: [UInt: MTLRenderPipelineState] = [:]
    private var envPipe: MTLComputePipelineState?
    private var envBuffer: MTLBuffer?
    private var dummyLightmap: MTLTexture?
    private var dummyVis: MTLTexture?
    /// With the GI cache: the relight's two GI textures when the cache didn't run this frame (1 x 1 stand-ins; the first
    /// stands in always: litRelightPixel's giStandIn).
    private var dummyGi: (irr: MTLTexture, code: MTLTexture)?
    /// Offline timing with the GI cache: the cache's last light the relight took (the timing runs reuse it; the cache
    /// runs once a frame), and the textures to take instead of this frame's (debugTimeTaa).
    private var lastGi: (irr: MTLTexture, code: MTLTexture)?
    private var debugGi: (irr: MTLTexture, code: MTLTexture)?
    private var failed = false
    private var frames = 0, relit = 0, missed = 0, giFrames = 0
    /// Debug view (METALMC_LITVIEW, see lit_relight_fs; mmc_lit_set_view offline).
    var view: Float = Float(ProcessInfo.processInfo.environment["METALMC_LITVIEW"] ?? "") ?? 0
    /// Offline timing only: leave the G-buffer out of the main pass.
    var noAttach = false
    /// With the GI cache: whether the relight takes its light (offline A/B, mmc_debug_lit_gi; the cache runs either way).
    var useGi = true
    /// Whether the last relight had the atmosphere's light (the GI cache, which runs before it in the frame, takes the
    /// same kind of light: a frame late when the sky turns on or off).
    private(set) var lastAtmosphere = false
    /// With water: whether the relight reflects in it (offline A/B, mmc_debug_lit_water; the G-buffer flags it either
    /// way), and the waves' time in seconds (offline, for repeatable pictures; below 0 the clock's).
    var waterOn = true
    var waterTime: Double = -1
    /// With water: the waves' tile (RGBA16Float, mipmapped: slopes and their mean squared length; a view of each level for
    /// the kernel to write) and the sky map (RGBA16Float, the upper hemisphere), and the kernels that make them each frame.
    private var waves: MTLTexture?
    private var waveLevels: [MTLTexture] = []
    private var waveStillLevelsMade = false
    private var waterSky: MTLTexture?
    private var wavesPipe: MTLComputePipelineState?
    private var waterSkyPipe: MTLComputePipelineState?

    /// Adds the G-buffer to the level's main pass (mmc_pass_begin), cleared to "not lit terrain".
    func attach(_ d: MTLRenderPassDescriptor, color: MTLTexture) -> Bool {
        if noAttach { return false }
        if gbuffer == nil || gbuffer!.width != color.width || gbuffer!.height != color.height {
            let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: litGbufferFormat, width: color.width, height: color.height, mipmapped: false)
            td.usage = [.renderTarget, .shaderRead]
            td.storageMode = .private
            gbuffer = ctx.device.makeTexture(descriptor: td)
            gbuffer?.label = "MetalMC lit G-buffer"
            log("lit: G-buffer \(color.width)x\(color.height) RG32Uint, \(color.width * color.height * 8 / 1_000_000) MB")
        }
        guard let gbuffer else { return false }
        let a = d.colorAttachments[litGbufferIndex]!
        a.texture = gbuffer
        a.loadAction = .clear
        a.clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        a.storeAction = .store
        frameTarget = (ObjectIdentifier(color.parent ?? color), color.width, color.height)
        return true
    }

    /// The library, the sun and sky kernel and the stand-in textures.
    private func ensureLibrary() -> Bool {
        if failed { return false }
        if library != nil { return true }
        do {
            // Lab mode (ShaderLab.swift): after an edit, forget it all; the next use builds it again from the new library.
            let lib = try ShaderLab.library("lit", litShaderSource) { [self] _ in
                library = nil; envPipe = nil; wavesPipe = nil; waterSkyPipe = nil; relightPipes = [:]; waveStillLevelsMade = false; failed = false
            }
            envPipe = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "lit_env")!)
            if litWater {
                wavesPipe = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "lit_water_waves")!)
                waterSkyPipe = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "lit_water_sky")!)
                let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: waterTexels, height: waterTexels, mipmapped: true)
                td.usage = [.shaderRead, .shaderWrite]
                td.storageMode = .private
                waves = ctx.device.makeTexture(descriptor: td)
                waves?.label = "MetalMC water waves"
                waveLevels = (0..<waterLevels).compactMap {
                    waves?.makeTextureView(pixelFormat: .rgba16Float, textureType: .type2D, levels: $0..<($0 + 1), slices: 0..<1)
                }
                let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: waterSkyTexels, height: waterSkyTexels, mipmapped: false)
                sd.usage = [.shaderRead, .shaderWrite]
                sd.storageMode = .private
                waterSky = ctx.device.makeTexture(descriptor: sd)
                waterSky?.label = "MetalMC water sky map"
            }
            library = lib
        } catch {
            log("lit: shaders failed: \(error)")
            failed = true
            return false
        }
        envBuffer = ctx.device.makeBuffer(length: 8 * 16, options: .storageModePrivate)
        let d1 = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        d1.usage = .shaderRead
        dummyLightmap = ctx.device.makeTexture(descriptor: d1)
        let d2 = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: 1, height: 1, mipmapped: false)
        d2.usage = .shaderRead
        dummyVis = ctx.device.makeTexture(descriptor: d2)
        if litGi {
            let d3 = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg11b10Float, width: 1, height: 1, mipmapped: false)
            d3.usage = .shaderRead
            let d4 = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Uint, width: 1, height: 1, mipmapped: false)
            d4.usage = .shaderRead
            if let a = ctx.device.makeTexture(descriptor: d3), let b = ctx.device.makeTexture(descriptor: d4) { dummyGi = (a, b) }
        }
        return true
    }

    private func ensure(format: MTLPixelFormat) -> Bool {
        if !ensureLibrary() { return false }
        if relightPipes[format.rawValue] == nil, let lib = library {
            let d = MTLRenderPipelineDescriptor()
            d.label = "MetalMC lit relight"
            d.vertexFunction = lib.makeFunction(name: "lit_fullscreen_vs")
            d.fragmentFunction = lib.makeFunction(name: "lit_relight_fs")
            d.colorAttachments[0].pixelFormat = format
            do { relightPipes[format.rawValue] = try ctx.device.makeRenderPipelineState(descriptor: d) } catch {
                log("lit: relight pipeline failed: \(error)")
                failed = true
                return false
            }
        }
        return relightPipes[format.rawValue] != nil && envPipe != nil && envBuffer != nil
    }

    /// The GI cache's light (Gi.swift, encodeFrame with the atmosphere): this frame's sun and sky light from the sky's
    /// tables, as the relight will compute them (lit_env), into `buffer` (9 float4: the 8 of env, the sky view table's
    /// scale in the first one's w with litGi) in `enc`. False if the sky's tables or the kernel aren't there.
    func encodeEnv(_ enc: MTLComputeCommandEncoder, into buffer: MTLBuffer) -> Bool {
        let sky = Sky.shared
        guard ensureLibrary(), let envPipe, sky.ready, let trans = sky.transmittance, let skyView = sky.skyView else { return false }
        var frame = sky.frame
        var scale = SIMD4<Float>(1.12 * skyExposure, 0, 0, 0)
        enc.setComputePipelineState(envPipe)
        enc.setBuffer(buffer, offset: 0, index: 0)
        enc.setBytes(&frame, length: MemoryLayout<SkyFrameGPU>.stride, index: 1)
        enc.setBytes(&scale, length: 16, index: 2)
        enc.setTexture(trans, index: 0)
        enc.setTexture(skyView, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        return true
    }

    /// Left for the anti-aliasing's resolve this frame (see bindDeferred). `env`: the sun and sky light's buffer (with the
    /// atmosphere; `daylight` is empty then).
    private var deferred: (frame: LitFrameGPU, daylight: [SIMD4<Float>], env: MTLBuffer, vis: MTLTexture, lightmap: MTLTexture,
                           gi: (irr: MTLTexture, code: MTLTexture)?, width: Int, height: Int)?

    /// True if this frame's relight was left for the anti-aliasing's resolve (Taa.swift), at this size.
    func hasDeferred(width: Int, height: Int) -> Bool {
        guard let d = deferred else { return false }
        return d.width == width && d.height == height
    }

    /// Binds the deferred relight's inputs to the anti-aliasing's resolve (its taaLit variant: buffers 2 and 3, textures
    /// 8-10, with the GI cache 11 and 12, with water 13 and 14) and forgets it.
    func bindDeferred(_ enc: MTLComputeCommandEncoder) {
        guard var d = deferred, let gbuffer else { return }
        deferred = nil
        enc.setBytes(&d.frame, length: MemoryLayout<LitFrameGPU>.stride, index: 2)
        if d.daylight.isEmpty {
            enc.setBuffer(d.env, offset: 0, index: 3)
        } else {
            enc.setBytes(d.daylight, length: d.daylight.count * 16, index: 3)
        }
        enc.setTexture(gbuffer, index: 8)
        enc.setTexture(d.vis, index: 9)
        enc.setTexture(d.lightmap, index: 10)
        if let gi = d.gi {
            enc.setTexture(gi.irr, index: 11)
            enc.setTexture(gi.code, index: 12)
        }
        if litWater {
            enc.setTexture(waves ?? dummyLightmap, index: 13)
            enc.setTexture(waterSky ?? dummyLightmap, index: 14)
            enc.setTexture(WaterTrace.shared.texture ?? dummyLightmap, index: 18)   // water's rays (Water.swift)
            enc.setTexture(WaterTrace.shared.textureAS ?? dummyLightmap, index: 20)
            enc.setTexture(WaterRipples.shared.texture ?? dummyLightmap, index: 19)   // the ripples' tile (Water.swift)
        }
        if clEnabled { ColoredLight.shared.bindDeferred(enc) }   // colored block light: textures 15, 16, buffer 4
    }

    /// With the GI cache: the sun and sky light it made this frame from the atmosphere (lit_env on the same sky tables the
    /// relight would run it on: the relight takes it instead), set by giTextures.
    private var frameGiEnv: MTLBuffer?

    /// With the GI cache: this frame's light from it (RtShadows ran it), or the stand-ins.
    private func giTextures(width: Int, height: Int) -> (irr: MTLTexture, code: MTLTexture)? {
        frameGiEnv = nil
        guard litGi, let dummyGi else { return nil }
        if let debugGi { return debugGi }
        guard let g = RtShadows.shared.takeLitGi(width: width, height: height) else { return dummyGi }
        frameGiEnv = g.env
        lastGi = (dummyGi.irr, g.gi)
        guard useGi else { return dummyGi }
        giFrames += 1
        // Colored block light bounced through the cache (ColoredLightBounce.swift) in the slot giStandIn kept free.
        if clEnabled, let b = RtShadows.shared.giBlockOut, b.width == g.gi.width, b.height == g.gi.height { return (b, g.gi) }
        return (dummyGi.irr, g.gi)
    }

    /// Relights the terrain of the level just drawn into `color` (see the top of this file). Needs no pass open.
    /// p: the projection as drawn (jittered) and the view rotation (32 floats), then vanilla's sun angle (radians), the fog
    /// color (4), the fog's environmental start and end and render-distance start and end, and 1 if our sky drew this
    /// frame (its atmosphere gives the sun's and sky's light) or 0 (vanilla's daylight curve). 42 floats. With
    /// `deferToTaa` it's left for the anti-aliasing's resolve, which reads every pixel's color and depth anyway (offline at
    /// the panel's resolution: a pass of its own about 0.62 ms on an 8-bit target and 0.75-0.97 on a float one; in the
    /// resolve +0.46-0.64 and +0.13).
    func relight(color: MTLTexture, depth: MTLTexture, p: UnsafePointer<Float>, lightmap: MTLTexture?, deferToTaa: Bool = false) -> Bool {
        deferred = nil
        let target = frameTarget
        frameTarget = nil
        frames += 1
        guard let target, let gbuffer, target.color == ObjectIdentifier(color.parent ?? color), target.width == color.width,
              target.height == color.height else {
            missed += 1
            if missed == 1 || missed % 1200 == 0 { log("lit: no G-buffer in this frame's level pass (\(missed) frames so far)") }
            return false
        }
        guard ctx.pass == nil, ensure(format: color.pixelFormat), let pipe = relightPipes[color.pixelFormat.rawValue],
              let envPipe, let envBuffer else { return false }
        func mat(_ o: Int) -> simd_float4x4 {
            simd_float4x4(SIMD4(p[o], p[o + 1], p[o + 2], p[o + 3]), SIMD4(p[o + 4], p[o + 5], p[o + 6], p[o + 7]),
                          SIMD4(p[o + 8], p[o + 9], p[o + 10], p[o + 11]), SIMD4(p[o + 12], p[o + 13], p[o + 14], p[o + 15]))
        }
        let sunAngle = p[32]
        // Vanilla's sun: (-sin a, cos a, 0); its moon is opposite.
        let sun = SIMD3<Float>(-sin(sunAngle), cos(sunAngle), 0)
        var f = LitFrameGPU()
        f.invViewProj = (mat(0) * mat(16)).inverse
        f.size = SIMD4(Float(color.width), Float(color.height), 0, isFloatFrame(color.pixelFormat) ? 65504 : 1)
        f.sunDir = SIMD4(sun, 0)
        f.moonDir = SIMD4(-sun, litSmoothstep(0.1, -0.15, sun.y))
        f.fogColor = SIMD4(p[33], p[34], p[35], p[36])
        f.fog = SIMD4(p[37], p[38], p[39], p[40])
        let sky = Sky.shared
        let atmosphere = p[41] > 0.5 && sky.ready && sky.transmittance != nil && sky.skyView != nil
        f.misc = SIMD4(lightmap != nil ? 1 : 0, atmosphere ? 0 : 1, view, litExposure)
        if litWater {
            // The waves: their time (the clock's, wrapped where a float still holds their phases to a few thousandths of
            // a radian), and the camera's x and z modulo their tile, from the LOD's draw this frame (the LOD and the far
            // field it draws are what flags water), so they stay put in the world as the camera moves. And the angle a
            // pixel spans at the screen's center, which sets the footprint the waves are sampled at (proj[1][1] is the
            // cotangent of half the vertical field of view).
            let t = waterTime >= 0 ? waterTime : ProcessInfo.processInfo.systemUptime.truncatingRemainder(dividingBy: 3600)
            let cam = LodRenderer.shared.lastCamera
            func wrap(_ x: Double) -> Float { x.isFinite ? Float(x - (x / waterTile).rounded(.down) * waterTile) : 0 }
            f.water = SIMD4(Float(t), wrap(cam.x), wrap(cam.z), waterOn ? 1 : 0)
            f.water2 = SIMD4(2 / max(abs(p[5]) * Float(color.height), 1e-6), atmosphere ? sky.frame.sunHoriz.w : 0,
                             atmosphere ? 1 - sky.frame.aerial.z : 0, atmosphere ? sky.frame.aerial.z : 0)   // z: the moon's glint (none in rain), w: rain (Water.swift)
        }
        var visTexture = dummyVis!
        if let v = RtShadows.shared.takeLitVisibility(width: color.width, height: color.height) {
            visTexture = v.texture
            f.size.z = Float(v.scale)
            f.sunDir.w = v.moon ? 1 : 0
        }
        let gi = giTextures(width: color.width, height: color.height)
        lastAtmosphere = atmosphere
        ctx.endBlit()
        let cb = ctx.ensureCB()
        // Colored block light's shadows (ColoredLightShadows.swift): rays toward the lights, for the relight below.
        if clEnabled { ClShadows.shared.trace(cb: cb, depth: depth, invViewProj: f.invViewProj, width: color.width, height: color.height) }
        var daylight: [SIMD4<Float>] = []
        var env = envBuffer
        if atmosphere, let giEnv = frameGiEnv {
            // The GI cache ran lit_env on this frame's sky tables already (its light is made from the relight's).
            env = giEnv
        } else if atmosphere, let trans = sky.transmittance, let skyView = sky.skyView, let enc = cb.makeComputeCommandEncoder() {
            enc.label = "MetalMC lit sun and sky"
            var frame = sky.frame
            var scale = SIMD4<Float>(1.12 * skyExposure, 0, 0, 0)
            enc.setComputePipelineState(envPipe)
            enc.setBuffer(envBuffer, offset: 0, index: 0)
            enc.setBytes(&frame, length: MemoryLayout<SkyFrameGPU>.stride, index: 1)
            enc.setBytes(&scale, length: 16, index: 2)
            enc.setTexture(trans, index: 0)
            enc.setTexture(skyView, index: 1)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            enc.endEncoding()
        } else {
            daylight = litDaylightEnv(sunAngle: sunAngle)
        }
        // With water: this frame's waves summed into the levels of their tile that move (the rest, where every wave has
        // faded, made once), and the sky map (where the reflections run: with our sky). They write different textures, so
        // the dispatches run side by side (serial, the small levels' each added their latency: 0.02-0.03 ms, measured).
        if litWater, atmosphere, waterOn, let wavesPipe, let waterSkyPipe, waveLevels.count == waterLevels, let waterSky,
           let skyView = sky.skyView, let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) {
            enc.label = "MetalMC water waves and sky map"
            var t = f.water.x
            enc.setComputePipelineState(wavesPipe)
            enc.setBytes(&t, length: 4, index: 0)
            for (level, view) in waveLevels.enumerated() where level < waterMovingLevels || !waveStillLevelsMade {
                let n = waterTexels >> level, g = min(n, 16)
                enc.setTexture(view, index: 0)
                enc.dispatchThreads(MTLSize(width: n, height: n, depth: 1), threadsPerThreadgroup: MTLSize(width: g, height: g, depth: 1))
            }
            waveStillLevelsMade = true
            var frame = sky.frame
            enc.setComputePipelineState(waterSkyPipe)
            enc.setTexture(waterSky, index: 0)
            enc.setTexture(skyView, index: 1)
            enc.setBytes(&frame, length: MemoryLayout<SkyFrameGPU>.stride, index: 0)
            enc.dispatchThreads(MTLSize(width: waterSkyTexels, height: waterSkyTexels, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            enc.endEncoding()
        }
        if litWater {
            // Water's second round (Water.swift): its rays (terrain in the reflections, the water's depth), the camera under
            // water, and the projection the floor seen through the water is found with.
            f.waterExtra.viewProj = mat(0) * mat(16)
            f.waterExtra.water3 = SIMD4(0, waterCameraInWater ? 1 : 0, Float(frames % 4096), 0)
            if atmosphere, waterOn, let waves, let waterSky, let dummyLightmap {
                WaterRipples.shared.encode(cb: cb, time: f.water.x)
                f.waterExtra.water3.x = Float(WaterTrace.shared.trace(
                    cb: cb, depth: depth, gbuffer: gbuffer, invViewProj: f.invViewProj, sunDir: SIMD3(f.sunDir.x, f.sunDir.y, f.sunDir.z),
                    moonDir: f.moonDir, water: f.water, radiansPerPixel: f.water2.x, exposure: f.misc.w, env: env, waves: waves,
                    sky: waterSky, lightmap: lightmap, lightmapStandIn: dummyLightmap,
                    vis: (visTexture, f.size.z, f.sunDir.w > 0.5), giLight: gi?.code, giStandIn: dummyGi?.code ?? dummyLightmap))
            } else {
                _ = RtShadows.shared.takeWaterStructure()
            }
        }
        if deferToTaa {
            deferred = (f, daylight, env, visTexture, lightmap ?? dummyLightmap!, gi, color.width, color.height)
            countFrame(atmosphere: atmosphere, traced: f.size.z > 0)
            return true
        }
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = color
        d.colorAttachments[0].loadAction = .load
        d.colorAttachments[0].storeAction = .store
        profAttach(d, "lit relight \(color.width)x\(color.height)")
        guard let enc = cb.makeRenderCommandEncoder(descriptor: d) else { return false }
        enc.label = "MetalMC lit relight"
        enc.setRenderPipelineState(pipe)
        enc.setFragmentTexture(depth, index: 0)
        enc.setFragmentTexture(gbuffer, index: 1)
        enc.setFragmentTexture(visTexture, index: 2)
        enc.setFragmentTexture(lightmap ?? dummyLightmap, index: 3)
        if let gi {
            enc.setFragmentTexture(gi.irr, index: 4)
            enc.setFragmentTexture(gi.code, index: 5)
        }
        enc.setFragmentBytes(&f, length: MemoryLayout<LitFrameGPU>.stride, index: 0)
        if daylight.isEmpty {
            enc.setFragmentBuffer(env, offset: 0, index: 1)
        } else {
            enc.setFragmentBytes(daylight, length: daylight.count * 16, index: 1)
        }
        if litWater {
            // Water's reflections: this frame's sky map and waves' tile (made above with our sky; without it the shader
            // leaves water alone). The anti-aliasing's resolve gets them from bindDeferred.
            enc.setFragmentTexture(waterSky ?? dummyLightmap, index: 6)
            enc.setFragmentTexture(waves ?? dummyLightmap, index: 7)
            // Water's rays (Water.swift) and a stand-in for the frame (this pass draws into it: the floor isn't bent here).
            enc.setFragmentTexture(WaterTrace.shared.texture ?? dummyLightmap, index: 11)
            enc.setFragmentTexture(WaterTrace.shared.textureAS ?? dummyLightmap, index: 14)
            enc.setFragmentTexture(WaterTrace.shared.colorStandInTexture ?? dummyLightmap, index: 12)
            enc.setFragmentTexture(WaterRipples.shared.texture ?? dummyLightmap, index: 13)
        }
        if clEnabled { ColoredLight.shared.bindRelight(enc) }   // colored block light: textures 8, 9, buffer 2
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        ctx.statPasses += 1
        countFrame(atmosphere: atmosphere, traced: f.size.z > 0)
        return true
    }

    private func countFrame(atmosphere: Bool, traced: Bool) {
        relit += 1
        if frames % 1200 == 0 {
            log("lit: relit \(relit) of the last 1200 frames (\(atmosphere ? "atmosphere" : "vanilla daylight"), \(traced ? "traced visibility" : "no rays")\(litGi ? ", GI cache in \(giFrames)" : ""))")
            relit = 0
            giFrames = 0
        }
    }

    /// Offline checks: the G-buffer holds `color`'s level again (to relight a frame already submitted).
    func debugTarget(_ color: MTLTexture) {
        frameTarget = (ObjectIdentifier(color.parent ?? color), color.width, color.height)
    }

    /// Offline check: GPU times (ms) of the relight pass alone over `runs` command buffers, on this G-buffer: the median,
    /// and the fastest in `fastest`.
    func debugTime(color: MTLTexture, depth: MTLTexture, p: UnsafePointer<Float>, lightmap: MTLTexture?, runs: Int,
                   fastest: inout Double) -> Double {
        var times: [Double] = []
        for _ in 0..<runs {
            frameTarget = (ObjectIdentifier(color.parent ?? color), color.width, color.height)
            guard relight(color: color, depth: depth, p: p, lightmap: lightmap), let cb = ctx.cb else { return -1 }
            ctx.cb = nil
            cb.commit()
            cb.waitUntilCompleted()
            times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
        }
        times.sort()
        fastest = times.first ?? -1
        return times.isEmpty ? -1 : times[times.count / 2]
    }

    /// Offline check: GPU times (ms) over `runs` command buffers each of the anti-aliasing's resolve with the relight in its
    /// load and without it, alternating, on this G-buffer: out[0], out[1] the medians, out[2], out[3] the fastest (on a
    /// machine whose other work shares the GPU, the fastest is closer to the pass's own cost).
    func debugTimeTaa(color: MTLTexture, depth: MTLTexture, colorHandle: Int64, depthHandle: Int64, p: UnsafePointer<Float>,
                      lightmap: MTLTexture?, runs: Int, out: UnsafeMutablePointer<Double>) {
        var cam: [Double] = [0, 0, 0]
        var with: [Double] = [], without: [Double] = []
        for k in 0..<(2 * runs) {
            let relit = k % 2 == 0
            if relit {
                frameTarget = (ObjectIdentifier(color.parent ?? color), color.width, color.height)
                _ = relight(color: color, depth: depth, p: p, lightmap: lightmap, deferToTaa: true)
            }
            guard mmc_taa_apply(colorHandle, depthHandle, p, &cam, 0, 0, 0) == 1, let cb = ctx.cb else { out[0] = -1; out[1] = -1; return }
            ctx.cb = nil
            cb.commit()
            cb.waitUntilCompleted()
            let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            if relit { with.append(ms) } else { without.append(ms) }
        }
        with.sort()
        without.sort()
        out[0] = with.isEmpty ? -1 : with[with.count / 2]
        out[1] = without.isEmpty ? -1 : without[without.count / 2]
        out[2] = with.first ?? -1
        out[3] = without.first ?? -1
        // With the GI cache: the resolve with the relight taking the cache's light (its last output) and without it (the
        // sky term), alternating, `runs` each: what the cache's upsample costs in the anti-aliasing's resolve. Logged.
        guard litGi, let gi = lastGi, let dummyGi else { return }
        var on: [Double] = [], off: [Double] = []
        for k in 0..<(2 * runs) {
            debugGi = k % 2 == 0 ? gi : dummyGi
            frameTarget = (ObjectIdentifier(color.parent ?? color), color.width, color.height)
            _ = relight(color: color, depth: depth, p: p, lightmap: lightmap, deferToTaa: true)
            guard mmc_taa_apply(colorHandle, depthHandle, p, &cam, 0, 0, 0) == 1, let cb = ctx.cb else { break }
            ctx.cb = nil
            cb.commit()
            cb.waitUntilCompleted()
            let ms = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            if k % 2 == 0 { on.append(ms) } else { off.append(ms) }
        }
        debugGi = nil
        on.sort()
        off.sort()
        guard !on.isEmpty, !off.isEmpty else { return }
        log(String(format: "lit: anti-aliasing resolve with the relight at %dx%d: with the GI cache's light %.3f ms (fastest %.3f), without %.3f (fastest %.3f): +%.3f (fastest +%.3f), %d each",
                   color.width, color.height, on[on.count / 2], on[0], off[off.count / 2], off[0], on[on.count / 2] - off[off.count / 2], on[0] - off[0], on.count))
    }

    /// Offline check with water (litWater): GPU times (ms) over `runs` command buffers each, the reflections on and off
    /// alternating, on this G-buffer. out[0-3]: the anti-aliasing's resolve with the relight and the sky's aerial
    /// perspective in its load (its lit and sky variant, where the game runs water), on and off (medians), then on and off
    /// (the fastest); out[4-7]: the relight as a pass of its own, likewise. Needs our sky's tables (a frame with the sky
    /// before it) and `p` with our sky on (Lit.relight's 42 floats). Without water "on" and "off" are the same passes: the
    /// baseline that water's whole cost (its code compiled in, which takes registers, as well as its work) is measured from.
    func debugTimeWater(color: MTLTexture, depth: MTLTexture, colorHandle: Int64, depthHandle: Int64, p: UnsafePointer<Float>,
                        lightmap: MTLTexture?, runs: Int, out: UnsafeMutablePointer<Double>) -> Bool {
        let saved = waterOn
        defer { waterOn = saved }
        var cam: [Double] = [0, 0, 0]
        var t: [[Double]] = [[], [], [], []]
        for k in 0..<(4 * runs) {
            let taa = k < 2 * runs
            waterOn = k % 2 == 0
            frameTarget = (ObjectIdentifier(color.parent ?? color), color.width, color.height)
            guard relight(color: color, depth: depth, p: p, lightmap: lightmap, deferToTaa: taa) else { return false }
            if taa {
                guard mmc_sky_aerial(colorHandle, depthHandle, p, 1) == 1, mmc_taa_apply(colorHandle, depthHandle, p, &cam, 0, 0, 0) == 1 else { return false }
            }
            guard let cb = ctx.cb else { return false }
            ctx.cb = nil
            cb.commit()
            cb.waitUntilCompleted()
            t[(taa ? 0 : 2) + (waterOn ? 0 : 1)].append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
        }
        for i in 0..<4 {
            let s = t[i].sorted()
            out[(i / 2) * 4 + i % 2] = s.isEmpty ? -1 : s[s.count / 2]
            out[(i / 2) * 4 + 2 + i % 2] = s.first ?? -1
        }
        return true
    }
}

/// 1 if lit mode is on (METALMC_EXP=lit).
@_cdecl("mmc_lit_enabled")
public func mmc_lit_enabled() -> Int32 { litEnabled ? 1 : 0 }

/// The next render pass is the level's main pass (LevelRenderer's, where the terrain is drawn): it gets the G-buffer.
@_cdecl("mmc_lit_level_pass")
public func mmc_lit_level_pass() {
    if litEnabled { Lit.shared.pendingLevelPass = true }
}

/// Relights the level's terrain in `color` from the G-buffer its main pass wrote (see Lit.relight for `p`, 42 floats).
/// `lightmap`: vanilla's lightmap (a texture view handle, 0 for none). `taa` != 0: left for the anti-aliasing's resolve
/// (mmc_taa_apply must follow). Returns 1 if it ran (or was left for the anti-aliasing).
@_cdecl("mmc_lit_relight")
public func mmc_lit_relight(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>, _ lightmapHandle: Int64, _ taa: Int32) -> Int32 {
    guard litEnabled else { return 0 }
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    let lightmap = lightmapHandle == 0 ? nil : (from(lightmapHandle) as TextureBox).texture
    return Lit.shared.relight(color: color, depth: depth, p: p, lightmap: lightmap, deferToTaa: taa != 0) ? 1 : 0
}

/// Offline: the debug view (see lit_relight_fs; 0 the relit frame).
@_cdecl("mmc_lit_set_view")
public func mmc_lit_set_view(_ view: Int32) {
    Lit.shared.view = Float(view)
}

/// Offline timing: 1 leaves the G-buffer out of the main pass (the relight then has nothing to do).
@_cdecl("mmc_lit_set_no_attach")
public func mmc_lit_set_no_attach(_ on: Int32) {
    Lit.shared.noAttach = on != 0
}

/// Offline A/B with the GI cache (METALMC_EXP=lit,gi): 0 relights with the sky term as without the cache, which keeps
/// running; 1 (default) takes the cache's light. Returns 1 if the cache is wired (litGi).
@_cdecl("mmc_debug_lit_gi")
public func mmc_debug_lit_gi(_ on: Int32) -> Int32 {
    Lit.shared.useGi = on != 0
    return litGi ? 1 : 0
}

/// Offline: encodes a copy of the G-buffer (RG32Uint, 8 bytes a pixel, rows bottom-up like every target) into `buffer`
/// at `offset` into the current command buffer. Returns 0 if there's none.
@_cdecl("mmc_lit_copy_gbuffer")
public func mmc_lit_copy_gbuffer(_ bufferHandle: Int64, _ offset: Int64) -> Int32 {
    guard let g = Lit.shared.gbuffer else { return 0 }
    let b = (from(bufferHandle) as BufferBox).buffer
    guard b.length - Int(offset) >= g.width * g.height * 8 else { return 0 }
    ctx.blitEncoder().copy(from: g, sourceSlice: 0, sourceLevel: 0, sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                           sourceSize: MTLSize(width: g.width, height: g.height, depth: 1), to: b, destinationOffset: Int(offset),
                           destinationBytesPerRow: g.width * 8, destinationBytesPerImage: g.width * g.height * 8)
    return 1
}

/// Offline: the anti-aliasing's resolve with the relight in its load and without, `runs` times each, alternating; out:
/// the median GPU milliseconds of each. Call after a frame that drew the level with the G-buffer (it reuses it).
@_cdecl("mmc_debug_lit_time_taa")
public func mmc_debug_lit_time_taa(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>, _ lightmapHandle: Int64,
                                   _ runs: Int32, _ out: UnsafeMutablePointer<Double>) {
    guard litEnabled else { out[0] = -1; out[1] = -1; return }
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    let lightmap = lightmapHandle == 0 ? nil : (from(lightmapHandle) as TextureBox).texture
    Lit.shared.debugTimeTaa(color: color, depth: depth, colorHandle: colorHandle, depthHandle: depthHandle, p: p, lightmap: lightmap,
                            runs: Int(runs), out: out)
}

/// Offline: the relight pass alone, `runs` times in command buffers of its own; out: the median and the fastest GPU
/// milliseconds. Call after a frame that drew the level with the G-buffer (it reuses that G-buffer and depth).
@_cdecl("mmc_debug_lit_time")
public func mmc_debug_lit_time(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>, _ lightmapHandle: Int64,
                               _ runs: Int32, _ out: UnsafeMutablePointer<Double>) {
    guard litEnabled else { out[0] = -1; out[1] = -1; return }
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    let lightmap = lightmapHandle == 0 ? nil : (from(lightmapHandle) as TextureBox).texture
    var fastest = -1.0
    out[0] = Lit.shared.debugTime(color: color, depth: depth, p: p, lightmap: lightmap, runs: Int(runs), fastest: &fastest)
    out[1] = fastest
}

/// Offline A/B with water (METALMC_EXP=lit,water): `on` 0 relights without the reflections (the G-buffer flags water either
/// way), 1 (default) with them; `time` >= 0 holds the waves at that time (seconds, for repeatable pictures), below 0 they
/// follow the clock. Returns 1 if water is on (litWater).
@_cdecl("mmc_debug_lit_water")
public func mmc_debug_lit_water(_ on: Int32, _ time: Double) -> Int32 {
    Lit.shared.waterOn = on != 0
    Lit.shared.waterTime = time
    return litWater ? 1 : 0
}

/// Offline with water: the anti-aliasing's resolve (relight and aerial perspective in its load) and the relight's own pass
/// with the reflections on and off, `runs` times each, alternating (Lit.debugTimeWater; without water both halves are the
/// same passes, a baseline); out: 8 doubles. Call after a frame that drew the level with the G-buffer and our sky. Returns
/// 0 without lit mode.
@_cdecl("mmc_debug_lit_water_time")
public func mmc_debug_lit_water_time(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>, _ lightmapHandle: Int64,
                                     _ runs: Int32, _ out: UnsafeMutablePointer<Double>) -> Int32 {
    guard litEnabled else { return 0 }
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    let lightmap = lightmapHandle == 0 ? nil : (from(lightmapHandle) as TextureBox).texture
    return Lit.shared.debugTimeWater(color: color, depth: depth, colorHandle: colorHandle, depthHandle: depthHandle, p: p,
                                     lightmap: lightmap, runs: Int(runs), out: out) ? 1 : 0
}

/// Debug: lit mode's own shader source (the relight pass, the sun and sky kernel, with water its kernels), for an offline
/// compile check. Returns its length.
@_cdecl("mmc_debug_lit_shader_source")
public func mmc_debug_lit_shader_source(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let bytes = Array(litShaderSource.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}
