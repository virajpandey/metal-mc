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

/// METALMC_EXP=lit: deferred relighting of terrain.
let litEnabled = experiments.contains("lit")

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
    uint3 a = uint3(round(saturate(albedo) * 255.0));
    uint code = face < 6u ? face + 1u : 7u;
    return uint2(a.r | (a.g << 8) | (a.b << 16) | (uint(round(saturate(ao) * 31.0)) << 24) | (code << 29),
                 litDepthKey(depth) | (uint(round(clamp(sky, 0.0, 15.0) * 16.0)) << 16) | (uint(round(clamp(block, 0.0, 15.0) * 16.0)) << 24));
}
#endif
"""

/// With the GI cache (litGi only; empty otherwise, so the relight's text is unchanged without it): litRelightPixel's
/// extra arguments, and the sky term taken from the cache.
private let litGiRelightArgsDoc = litGi ? "\n// giIrr, giCode: the GI cache's half-resolution light and code words (gi_resolve; 1 x 1 stand-ins when it didn't run)." : ""
private let litGiRelightArgs = litGi ? ", texture2d<float> giIrr, texture2d<uint> giCode" : ""
private let litGiSkyTerm = litGi ? """

        // The GI cache (Gi.swift) where it has data for the pixel's face and plane: the sky's light as the real openings
        // let it in, plus bounced light, in place of the sky light level's curve. In env's units (the cache's light is
        // made from it), so the daylight curve's scale and the exposure apply as before. Faces that aren't axis-aligned
        // (plants) take the top face's beside them. Debug view 8: green where it applied, red where it didn't, blue for
        // light sources.
        if (giIrr.get_width() > 1u) {
            float4 gi = giUpsample(giIrr, giCode, q, fi < 6u ? fi : 2u, rel);
            if (gi.w > 0.0) { skyAmb = gi.rgb * ao; giUsed = true; }
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
};

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

// The relight of pixel q, whose color is dst (as the target holds it), depth d and G-buffer texel g: the color it gets.
// Pixels that aren't lit terrain keep theirs. env: the sun's light (0), the sky's on each face direction (1-6, the LOD's
// face order) and on faces that aren't axis-aligned (7), linear, per unit of albedo.\(litGiRelightArgsDoc)
static float3 litRelightPixel(float3 dst, uint2 q, float d, uint2 g, texture2d<half, access::read> vis, texture2d<float> lightmap,
                              constant LitFrame& f, constant float4* env\(litGiRelightArgs)) {
    uint code = g.x >> 29;
    uint view = uint(f.misc.z);
    if (code == 0u) return view != 0u ? float3(0.0) : dst;
    // Something nearer was drawn over the terrain since (an entity, water, a cloud): its color stays.
    if (!litDepthMatches(d, g.y & 0xFFFFu)) return view != 0u ? float3(0.2, 0.0, 0.2) : dst;
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
    float vSunOut = 0.0;\(litGi ? "\n    bool giUsed = false;" : "")
    if (block >= 15.0) {
        // Light sources (vanilla gives emissive faces full block light): full bright, like vanilla.
        E = float3(1.0);
    } else {
        float3 lm00 = lm ? skyDecode(litLightmap(lightmap, 0.0, 0.0)) : float3(0.0);
        // Vanilla's block light, with the dimension's ambient, night vision and the darkness effect, as its lightmap has it.
        float3 blockLight = lm ? skyDecode(litLightmap(lightmap, block, 0.0)) : float3(litBrightness(block) * litBrightness(block));
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
    }
    float3 lin = skyDecode(refl) * E;
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
        else if (view == 7u) c = plain ? float3(0.0, 0.8, 0.0) : (overlayBright ? float3(0.9, 0.0, 0.9) : float3(0.9, 0.0, 0.0));\(litGi ? "\n        else if (view == 8u) c = giUsed ? float3(0.0, 0.8, 0.0) : (block >= 15.0 ? float3(0.0, 0.0, 0.9) : float3(0.9, 0.0, 0.0));" : "")
        return saturate(c);
    }
    return o;
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
                               texture2d<float> lightmap [[texture(3)]],\(litGi ? "\n                               texture2d<float> giIrr [[texture(4)]],\n                               texture2d<uint> giCode [[texture(5)]]," : "")
                               constant LitFrame& f [[buffer(0)]],
                               constant float4* env [[buffer(1)]]) {
    uint2 q = uint2(in.pos.xy);
    uint2 g = gbuf.read(q).rg;
    if ((g.x >> 29) == 0u && f.misc.z == 0.0) return dst;
    return float4(litRelightPixel(dst.rgb, q, depth.read(q), g, vis, lightmap, f, env\(litGi ? ", giIrr, giCode" : "")), dst.a);
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
    out[0] = float4(sun * scale, \(litGi ? "scale" : "0.0"));
    out[1] = float4((px + 0.5 * ground) * scale, 0.0);
    out[2] = float4((nx + 0.5 * ground) * scale, 0.0);
    out[3] = float4(up * scale, 0.0);
    out[4] = float4(ground * scale, 0.0);
    out[5] = float4((pz + 0.5 * ground) * scale, 0.0);
    out[6] = float4((nz + 0.5 * ground) * scale, 0.0);
    out[7] = float4((0.5 * up + 0.125 * (px + nx + pz + nz) + 0.25 * ground) * scale, 0.0);
}
"""

/// Mirrors LitFrame in the shader.
private struct LitFrameGPU {
    var invViewProj = matrix_identity_float4x4
    var size = SIMD4<Float>.zero
    var sunDir = SIMD4<Float>.zero
    var moonDir = SIMD4<Float>.zero
    var fogColor = SIMD4<Float>.zero
    var fog = SIMD4<Float>.zero
    var misc = SIMD4<Float>.zero
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
    /// With the GI cache: 1 x 1 stand-ins for its light and code words when it didn't run this frame.
    private var dummyGi: (irr: MTLTexture, code: MTLTexture)?
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
            let lib = try ctx.device.makeLibrary(source: litShaderSource, options: nil)
            envPipe = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "lit_env")!)
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
            let d4 = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Uint, width: 1, height: 1, mipmapped: false)
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

    /// Left for the anti-aliasing's resolve this frame (see bindDeferred).
    private var deferred: (frame: LitFrameGPU, daylight: [SIMD4<Float>], vis: MTLTexture, lightmap: MTLTexture, gi: (irr: MTLTexture, code: MTLTexture)?,
                           width: Int, height: Int)?

    /// True if this frame's relight was left for the anti-aliasing's resolve (Taa.swift), at this size.
    func hasDeferred(width: Int, height: Int) -> Bool {
        guard let d = deferred else { return false }
        return d.width == width && d.height == height
    }

    /// Binds the deferred relight's inputs to the anti-aliasing's resolve (its taaLit variant: buffers 2 and 3, textures
    /// 8-10, and with the GI cache 11 and 12) and forgets it.
    func bindDeferred(_ enc: MTLComputeCommandEncoder) {
        guard var d = deferred, let gbuffer, let envBuffer else { return }
        deferred = nil
        enc.setBytes(&d.frame, length: MemoryLayout<LitFrameGPU>.stride, index: 2)
        if d.daylight.isEmpty {
            enc.setBuffer(envBuffer, offset: 0, index: 3)
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
    }

    /// With the GI cache: this frame's light from it (RtShadows ran it), or the stand-ins.
    private func giTextures(width: Int, height: Int) -> (irr: MTLTexture, code: MTLTexture)? {
        guard litGi, let dummyGi else { return nil }
        guard let g = RtShadows.shared.takeLitGi(width: width, height: height), useGi else { return dummyGi }
        giFrames += 1
        return (g.irr, g.code)
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
        var daylight: [SIMD4<Float>] = []
        if atmosphere, let trans = sky.transmittance, let skyView = sky.skyView, let enc = cb.makeComputeCommandEncoder() {
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
        if deferToTaa {
            deferred = (f, daylight, visTexture, lightmap ?? dummyLightmap!, gi, color.width, color.height)
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
            enc.setFragmentBuffer(envBuffer, offset: 0, index: 1)
        } else {
            enc.setFragmentBytes(daylight, length: daylight.count * 16, index: 1)
        }
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
