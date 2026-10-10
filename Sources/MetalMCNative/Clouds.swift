import Foundation
import Metal
import simd

// Volumetric clouds (METALMC_EXP=clouds, with METALMC_EXP=sky; docs/lighting-design.md, "Volumetric clouds"). A cumulus
// layer that follows the planet's curve in the sky's flat-world/round-planet mapping (Sky.swift), so it runs out to the
// horizon, lit by the sky's own tables, and shadowing every pixel the relight lights (the LOD, the far field, the near
// terrain) out to the horizon. Vanilla's clouds are off while these are on (metalmc.clouds.mixin.CloudRendererMixin).
//
// Shapes (after Schneider and Vos, "The real-time volumetric cloudscapes of Horizon Zero Dawn", 2015, and Schneider's
// "Nubis" talks): a 2D weather map (coverage and cloud type) that drifts with the wind, a 3D base shape (Perlin-Worley
// eroded by Worley octaves) that the coverage carves into clouds along a height profile per type, and a 3D detail noise
// that erodes their edges. All three are made once on the GPU (tileable, so the camera's position and the wind's drift
// are taken modulo their tiles and nothing drifts out of float precision).
//
// Per frame, after the level's main pass (GameRendererLodMixin, before the shadows):
// - the sky's light for the clouds (clouds_env: the open sky's mean radiance and the ground's, from the sky view table);
// - a deep shadow map in three cascades (clouds_shadow, 256 x 256 each, 16, 64 and 256 km across): each texel is the
//   light's ray through the layer, crossing its middle above the texel, and holds the optical depth from the layer's top
//   down to each quarter of it. It lights the clouds (a sample's optical depth toward the light: two samples of the
//   density up to the next stored quarter, then that quarter's stored value) and shadows the ground
//   (cloudTransmittanceToSun projects any point along the light onto it);
// - the ray march at half the resolution each way (clouds_march), each texel marched every 4th frame (a 2 x 2 Bayer
//   order; the others reproject the history): the view ray through the layer (up to two stretches, as the curve takes it
//   out and in again), jittered, longer steps through clear air, lit by the sun through Sky's transmittance table at the
//   cloud's altitude and local sun height (so clouds stay lit after the ground's sunset and go orange and pink first), the
//   deep shadow map's optical depth with Beer's law, the powder effect, a dual-lobe Henyey-Greenstein phase, Wrenninge's
//   multiple-scattering octaves and a diffuse-transmission floor, the sky's ambient by height in the layer; aerial
//   perspective from Sky's table at the cloud's depth; then into the history, reprojected by the cloud's own depth (the
//   sky's at infinity), rejected off screen, where the depth disagrees and where terrain covered it.
// The anti-aliasing's resolve composites them over each pixel as it loads it, after the sky's aerial perspective
// (cloudsComposite: a bilinear upsample that leaves out texels whose pixels were all terrain in front of the clouds, and
// keeps terrain that is in front of them), in linear light. Without anti-aliasing, a pass of its own does it
// (clouds_composite_fs). Cloud shadows: RtShadows' visibility is multiplied by the clouds' transmittance toward the light
// at the same pixels (shadeVisibility), which the relight (sun and moon terms) and the water's glint already read.

/// METALMC_EXP=clouds (with sky): volumetric clouds in place of vanilla's.
let cloudsEnabled = experiments.contains("clouds") && skyEnabled

private let cloudEnv = ProcessInfo.processInfo.environment
private func cloudSetting(_ name: String, _ fallback: Float) -> Float {
    guard let v = Float(cloudEnv[name] ?? ""), v.isFinite else { return fallback }
    return v
}
/// METALMC_CLOUDCOVER: clear, scattered (the default), overcast, or a number in coverage units (the weather map's
/// coverage is this plus its own variation, about +-0.05: 0.3 a few puffs, 0.4 scattered cumulus, 0.5 broken, 0.55 and
/// up a closed deck). Rain takes it to overcast with the game's rain level, thunder to a darker, towering deck. 0.43:
/// at 0.4 the second game session's noon_overview had only a few small clouds left (the weather map low there).
let cloudCoverDefault: Float = {
    switch cloudEnv["METALMC_CLOUDCOVER"] ?? "scattered" {
    case "clear": return 0.3
    case "scattered": return 0.43
    case "overcast": return 1.6
    case let s: return Float(s) ?? 0.43
    }
}()
/// METALMC_CLOUDBASE, METALMC_CLOUDTOP: the layer, in metres (blocks) above sea level. 650-1700 m: low fair-weather
/// cumulus (humid air's bases are 0.5-1.5 km). Offline, a 1.2-2.6 km layer (the textbook's) made the clouds small and far
/// from a world whose hills are 100-200 m: lens-shaped slivers low in the sky; at 650 m they fill the sky as SEUS's do, and
/// every flight stays under them (the bench flies at y 150, the build limit is 320; vanilla's slab at 192 was a ceiling).
private let cloudBase = max(cloudSetting("METALMC_CLOUDBASE", 650), 100)
private let cloudTop = max(cloudSetting("METALMC_CLOUDTOP", 1700), cloudBase + 100)
/// METALMC_CLOUDWIND: the wind's speed (m/s) the weather and the shapes drift with.
private let cloudWindSpeed = cloudSetting("METALMC_CLOUDWIND", 8)
/// METALMC_CLOUDTIME: a fixed wind time (s) for repeatable pictures; the clock's otherwise.
private let cloudTimeFixed: Double? = Double(cloudEnv["METALMC_CLOUDTIME"] ?? "")
/// METALMC_CLOUDVIEW: debug views (1 the clouds alone over black, 2 their transmittance, 3 the deep shadow map's cascades
/// over the screen, 4 cloud shadows only: the relight's visibility from the clouds alone).
private let cloudViewDefault = Int(cloudEnv["METALMC_CLOUDVIEW"] ?? "") ?? 0
/// METALMC_CLOUDMOON: the moon's illuminance on the clouds at night (scene units).
private let cloudMoonIllum = max(0, cloudSetting("METALMC_CLOUDMOON", 0.12))

/// The noise tiles (m): the weather map's, which the base (4096) and detail (512) tiles divide, so the camera's position
/// and the wind's drift are wrapped modulo it and every lookup stays seamless.
private let cloudWeatherTile: Double = 131_072
/// Each texel of the cloud buffer is marched every cloudUpdate frames (the shader's CLOUD_UPDATE: 4 or 8).
private let cloudUpdate = 8
/// The deep shadow map: texels a side per cascade, and the cascades' half widths (m).
private let cloudShadowTexels = 256
private let cloudCascadeHalf: [Double] = [8_192, 32_768, 131_072]

/// Structures and functions other passes share: the anti-aliasing's resolve (the composite), lit mode's relight through
/// RtShadows' visibility, and the god rays (cloudTransmittanceToSun). Needs skyShaderHeader before it.
let cloudsShaderHeader = cloudsEnabled ? """

// ---- Volumetric clouds (Clouds.swift, METALMC_EXP=clouds): what other passes share ----
struct CloudFrame {
    float4x4 invViewProj;   // the cloud buffer's clip space to camera-relative world (the level's projection, unjittered)
    float4x4 prevViewProj;  // last frame's projection * view rotation (unjittered), for positions relative to last frame's camera
    float4 camDelta;        // xyz: this frame's camera minus last frame's (blocks), w: 1 if the history holds a frame
    float4 size;            // xy: the frame's size, zw: the cloud buffer's (half each way)
    float4 layer;           // x: the layer's base, y: its top (m above sea level), z: the camera's altitude (m above sea level), w: 1 / (2 R) (1/m)
    float4 sun;             // xyz: toward the sun, w: its illuminance (scene units, Sky's)
    float4 moon;            // xyz: toward the moon, w: its illuminance (scene units)
    float4 light;           // the clouds' light (the sun, or the moon once the sun is under the layer's horizon): xyz toward it, w: its illuminance
    float4 noise;           // xy: the weather map's offset (m: the camera's world x, z plus the wind's drift, wrapped), zw: the 3D noise's
    float4 shape;           // x: coverage, y: density scale, z: rain (0-1), w: thunder (0-1)
    float4 cascade[3];      // the deep shadow map's cascades (side by side): xy center (camera-relative x, z on the layer's middle), z half width (m), w 1 if made
    float4 misc;            // x: frame (the jitter's and the update order's), y: debug view, z: a new sample's weight in the history, w: 1 if the clouds are on this frame
};

// How much of the light a cloud's optical depth tau lets through to the ground: the direct beam (Beer), and part of
// what the cloud scatters, which comes out below as diffuse light (a thick cumulus still passes about a third: Bohren's
// diffuse transmission, 1 / (1 + 0.75 (1 - g) tau) with g 0.85, scaled down).
#ifndef CLOUD_SHADOW_DIFFUSE
#define CLOUD_SHADOW_DIFFUSE 0.45
#endif

// A point's altitude above sea level (m) in the flat-world, round-planet mapping (Sky.swift): the camera's altitude, the
// point's height over it, and the planet's curve below it (|xz|^2 / 2R, which is 7 km at 300 km).
static float cloudsAltitude(float3 rel, constant CloudFrame& cf) {
    return cf.layer.z + rel.y + dot(rel.xz, rel.xz) * cf.layer.w;
}

// The two roots of a t^2 + b t + c = 0 (a > 0) in order, numerically stable. False if there are none.
static bool cloudsRoots(float a, float b, float c, thread float& r0, thread float& r1) {
    float disc = b * b - 4.0 * a * c;
    if (disc < 0.0) return false;
    float q = -0.5 * (b + (b >= 0.0 ? 1.0 : -1.0) * sqrt(disc));
    if (abs(q) < 1e-20) q = 1e-20;
    float x0 = q / a, x1 = c / q;
    r0 = min(x0, x1);
    r1 = max(x0, x1);
    return true;
}

// Where the light's ray through a camera-relative point at altitude alt crosses the layer's middle (camera-relative x, z):
// along the light, alt(s) = alt + s (L.y + 2 k rel.xz . L.xz) + s^2 k |L.xz|^2; the root nearest the point (ahead of it
// from below the middle, behind it from above).
static float2 cloudsToMiddle(float3 rel, float alt, constant CloudFrame& cf) {
    float3 L = cf.light.xyz;
    float k = cf.layer.w;
    float r0, r1;
    float s = 0.0;
    if (cloudsRoots(max(k * dot(L.xz, L.xz), 1e-14), L.y + 2.0 * k * dot(rel.xz, L.xz), alt - 0.5 * (cf.layer.x + cf.layer.y), r0, r1)) s = r1;
    return rel.xz + s * L.xz;
}

// The deep shadow map where the light's ray crosses the layer's middle at x: the optical depth from the layer's top down
// to 3/4, 1/2, 1/4 of it and its base. The finest cascade that covers x, blended into the next over its outer fifth;
// none (0) past the last.
static float4 cloudsDeepShadow(float2 x, constant CloudFrame& cf, texture2d<float> shadow) {
    constexpr sampler sm(filter::linear, address::clamp_to_edge);
    float n = float(shadow.get_height());
    float4 acc = 0.0;
    float left = 1.0;
    for (uint i = 0; i < 3u && left > 0.0; i++) {
        float4 c = cf.cascade[i];
        float2 u = (x - c.xy) / (2.0 * c.z) + 0.5;
        float edge = max(abs(u.x - 0.5), abs(u.y - 0.5));
        if (c.w <= 0.0 || edge >= 0.5 - 1.0 / n) continue;
        float w = left * (1.0 - smoothstep(0.3, 0.5 - 1.0 / n, edge));
        acc += shadow.sample(sm, float2((u.x + float(i)) / 3.0, u.y), level(0.0)) * w;
        left -= w;
    }
    return acc;
}

// The optical depth from the layer's top down to height h in it (0 its base, 1 its top), between the deep shadow map's four.
static float cloudsDeepAt(float4 d, float h) {
    h = saturate(h);
    if (h >= 0.75) return d.x * (1.0 - h) * 4.0;
    if (h >= 0.5) return mix(d.y, d.x, (h - 0.5) * 4.0);
    if (h >= 0.25) return mix(d.z, d.y, (h - 0.25) * 4.0);
    return mix(d.w, d.z, h * 4.0);
}

static float cloudsGroundTransmittance(float tau) {
    float direct = exp(-tau);
    return direct + (1.0 - direct) * CLOUD_SHADOW_DIFFUSE / (1.0 + 0.15 * tau);
}

// The clouds' transmittance toward the light (the sun by day, the moon at night: cf.light) for a camera-relative position
// (blocks): the deep shadow map where the light's ray through it crosses the layer, down to the point's height. 1 above
// the layer, past the cascades (about 130 km out along the light) and when the clouds are off. For the relight (through
// RtShadows' visibility) and the god rays: bind the map and this frame's CloudFrame (Clouds.shadowBinding).
static float cloudTransmittanceToSun(float3 rel, constant CloudFrame& cf, texture2d<float> shadow) {
    if (cf.misc.w <= 0.0 || cf.cascade[0].w <= 0.0) return 1.0;
    float alt = cloudsAltitude(rel, cf);
    if (alt >= cf.layer.y) return 1.0;
    float4 d = cloudsDeepShadow(cloudsToMiddle(rel, alt, cf), cf, shadow);
    return cloudsGroundTransmittance(cloudsDeepAt(d, (alt - cf.layer.x) / (cf.layer.y - cf.layer.x)));
}

// The clouds over a pixel of the level, its color c as the target holds it (sRGB-encoded scene-linear light with the
// post chain; display light without it), q its position, d its depth (reverse-Z, 0 the sky). The cloud buffer is half the
// resolution each way: its four texels around the pixel, bilinear, leaving out for a sky pixel the texels whose pixels were
// all terrain nearer than the clouds (their value is "no cloud", which would fringe the clouds along every silhouette).
// Terrain keeps its color where it is nearer than the clouds' front (the transmittance-weighted depth), and is covered
// where it's behind (the camera above the layer). Linear light: c * T + the clouds' light (aerial perspective included).
static float3 cloudsComposite(float3 c, uint2 q, float d, constant SkyFrame& sky, constant CloudFrame& cf,
                              texture2d<float> clouds, texture2d<float> cdepth) {
    if (cf.misc.w <= 0.0) return c;
    constexpr sampler sl(filter::linear, address::clamp_to_edge), sn(filter::nearest, address::clamp_to_edge);
    float2 uv = (float2(q) + 0.5) / cf.size.xy;
    bool skyPx = d <= 0.0;
    uint view = uint(cf.misc.y);
    float D = 0.0;
    if (!skyPx) {
        // Terrain whose segment from the camera stays under the layer's base: nothing in front, no fetch. (The altitude
        // along a straight segment is convex in the planet mapping, so the segment's highest point is one of its ends.)
        // That's nearly all terrain from under the layer: 0.06 ms of the 0.34 the composite took in the resolve at noon.
        float4 h = sky.invViewProj * float4(uv * 2.0 - 1.0, d, 1.0);
        float3 rel = h.xyz / h.w;
        if (max(cf.layer.z, cloudsAltitude(rel, cf)) < cf.layer.x) return view == 1u || view == 2u ? float3(view == 2u ? 1.0 : 0.0) : c;
        D = length(rel);
    }
    // The four texels' depths in one gather (its order: (0, 1), (1, 1), (1, 0), (0, 0) from the footprint's corner).
    float4 dd = cdepth.gather(sn, uv) * 1000.0;   // stored in km
    // The common case's value (the hardware's bilinear), fetched alongside the gather instead of after it: in the
    // resolve's load the two latencies one after the other were most of the composite's cost.
    float4 acc = clouds.sample(sl, uv, level(0.0));
    // Terrain where none of the four has a cloud (no cloud, or terrain in front of it): as it was.
    if (!skyPx && all(dd <= 0.0)) return view == 1u || view == 2u ? float3(view == 2u ? 1.0 : 0.0) : c;
    float2 fr = fract(uv * cf.size.zw - 0.5);
    float4 bw = float4((1.0 - fr.x) * fr.y, fr.x * fr.y, fr.x * (1.0 - fr.y), (1.0 - fr.x) * (1.0 - fr.y));
    if (skyPx && any(dd < 0.0)) {
        // A sky pixel beside texels whose pixels were all terrain nearer than the clouds: the others only.
        float2 hp = uv * cf.size.zw - 0.5;
        int2 i0 = int2(floor(hp)), hmax = int2(cf.size.zw) - 1;
        const int2 offs[4] = { int2(0, 1), int2(1, 1), int2(1, 0), int2(0, 0) };
        acc = 0.0;
        float wsum = 0.0;
        for (int k = 0; k < 4; k++) {
            if (dd[k] < 0.0) continue;
            float w = bw[k] + 1e-4;
            acc += clouds.read(uint2(clamp(i0 + offs[k], int2(0), hmax))) * w;
            wsum += w;
        }
        if (wsum <= 0.0) return view == 1u || view == 2u ? float3(view == 2u ? 1.0 : 0.0) : c;
        acc /= wsum;
    }
    float f = 1.0;
    if (!skyPx) {
        // Terrain: covered only where the clouds are in front of it (their depth: the texels with a cloud, bilinear).
        float4 has = float4(dd > 0.0) * bw;
        float front = dot(has, float4(1.0)) > 1e-6 ? dot(has, max(dd, 0.0)) / dot(has, float4(1.0)) : 0.0;
        f = front > 0.0 ? saturate((D - 0.9 * front) / (0.2 * front)) : 0.0;
    }
    // Debug views (METALMC_CLOUDVIEW): 1 the clouds alone, 2 their transmittance.
    if (view == 1u) return skyEncode(acc.rgb * f);
    if (view == 2u) return float3(mix(1.0, acc.a, f));
    float3 o = skyDecode(c) * mix(1.0, acc.a, f) + acc.rgb * f;
    // Without the post chain the frame holds display light (the sky's shoulder, Sky.swift): the sum goes through it again
    // (below its knee it's the identity, so only the brightest of what was there is compressed twice).
    if (sky.tone.y < 60000.0) o = skyToneMap(o, sky.tone);
    return skyEncode(o);
}
// The clouds in a direction of the upper hemisphere from the camera, for reflections (the water's, Lit.swift): rgb what they
// add over the sky behind them (aerial perspective included), a their transmittance. map: Clouds.reflectionBinding()'s
// texture, a paraboloid map with the water's sky map's mapping (the horizon the circle of radius 1/2 around the middle).
// In scene-linear light: reflected = sky(R) * a + rgb.
static float4 cloudsReflected(float3 dir, texture2d<float> map) {
    constexpr sampler sm(filter::linear, address::clamp_to_edge);
    float3 d = normalize(float3(dir.x, max(dir.y, 0.0), dir.z));
    return map.sample(sm, d.xz / (1.0 + d.y) * 0.5 + 0.5, level(0.0));
}
// ---- end of the clouds' shared part ----
""" : ""

/// The clouds' own library: the noise, the sky's light for them, the deep shadow map, the march, the cloud shadows on
/// the relight's visibility, the composite without anti-aliasing, and the offline view. Every constant of the look is a
/// #define at the top (lab mode: edit clouds.metal and save).
private let cloudsShaderSource = skyShaderHeader + cloudsShaderHeader + """

// ---- Volumetric clouds (Clouds.swift). The look's constants: edit them here in lab mode. ----

// The noise tiles (m; each divides the weather tile, Clouds.swift's cloudWeatherTile).
#define CLOUD_WEATHER_TILE \(Int(cloudWeatherTile)).0
#define CLOUD_BASE_TILE 4096.0
#define CLOUD_DETAIL_TILE 512.0

// Coverage: added to the frame's (METALMC_CLOUDCOVER, rain), and how far the weather map moves it either way. Rain's cloud type.
// The base shape's values bunch up, so the sky goes from a few puffs (0.3) through scattered cumulus (0.4) to a closed deck
// (0.55) in a narrow band: a spread of 1.8 (+-0.3 across the map) made it either a deck or empty overhead most places.
#define CLOUD_COVER_BIAS 0.0
#define CLOUD_COVER_SPREAD 0.3
#define CLOUD_RAIN_TYPE 0.55
// The height profile: how fast density rises off the flat base (1 / the share of the layer), stratus tops (share of the layer).
#define CLOUD_BASE_SHARP 14.0
#define CLOUD_STRATUS_TOP 0.3
// Extinction (1/m) at density 1, the detail noise's erosion, the single-scattering albedo.
#define CLOUD_SIGMA 0.07
#define CLOUD_EROSION 0.5
#define CLOUD_ALBEDO 0.98

// The march: steps per stretch of the layer (fewest, most), the shortest step (m), how much longer steps grow through
// clear air (at most), the step under which the detail noise is used, how far away it's used (m), where a ray counts as
// opaque, how far rays go (m).
#define CLOUD_STEPS_MIN 12
#define CLOUD_STEPS_MAX 48
#define CLOUD_STEP_MIN 50.0
#define CLOUD_SKIP_MAX 3.0
#define CLOUD_FINE_STEP 160.0
#define CLOUD_FINE_DIST 24000.0
#define CLOUD_T_MIN 0.01
#define CLOUD_MAX_DIST 300000.0

// Lighting: the phase's two lobes (forward g, back g, the back lobe's share), the multiple-scattering octaves (Wrenninge:
// each octave's light, optical depth and anisotropy scaled by A, B, C) and the diffusion regime the octaves miss (in a
// thick cloud most of the light has scattered many times: an isotropic source of DIFFUSE / 4 pi of the sunlight, falling
// with the optical depth toward the sun like diffuse transmission, 1 / (1 + FALL tau), and absorption, exp(-ABSORB tau);
// without it sunlit cumulus came out about five times too dark, gray instead of white), the powder effect (sunlit wisps darker when the light is behind the camera,
// none looking toward it: its strength and the local depth it takes), the sky's ambient (gain, its share at the base, the
// ground's bounce).
#define CLOUD_G_FORWARD 0.78
#define CLOUD_G_BACK -0.25
#define CLOUD_G_MIX 0.3
#define CLOUD_MS_A 0.55
#define CLOUD_MS_B 0.4
#define CLOUD_MS_C 0.5
#define CLOUD_MS_DIFFUSE 3.0
#define CLOUD_MS_FALL 0.12
#define CLOUD_MS_ABSORB 0.02
#define CLOUD_POWDER 0.6
#define CLOUD_POWDER_DEPTH 4.0
#define CLOUD_AMBIENT 1.0
#define CLOUD_AMB_BASE 0.45
#define CLOUD_AMB_GROUND 0.6
// Cirrus: a thin sheet of ice cloud high above the cumulus (altitude, m), how much of the sky it covers where the weather
// allows (0 none), its optical depth at full density looking straight up, the size of its streaks along and across the wind
// (m; powers of two over the integer frame (2, 1), (-1, 2), so they tile with the camera's wrap), its phase's forward g.
#define CIRRUS_ALT 8000.0
#define CIRRUS_COVER 0.55
#define CIRRUS_TAU 0.35
#define CIRRUS_ALONG 65536.0
#define CIRRUS_ACROSS 8192.0
#define CIRRUS_G 0.6
// The light's own stretch from each sample up to the deep shadow map's next stored height: samples, the longest (m).
#define CLOUD_LIGHT_SAMPLES 2
#define CLOUD_LIGHT_STRETCH 600.0
// The history: each texel is marched every CLOUD_UPDATE frames (4 or 8; the others carry their history over, reprojected;
// set in Clouds.swift, cloudUpdate, whose dispatch follows it: not to edit here), how far a cloud's depth may differ
// (share) before the history is rejected.
#define CLOUD_UPDATE \(cloudUpdate)
#define CLOUD_DEPTH_REJECT 0.35
// The detail noise erodes only density under this (the interior takes its mean erosion: one 3D sample less a step).
#define CLOUD_DETAIL_MAXD 0.6
// The deep shadow map: the fewest steps through the layer (a multiple of 4), the step length a low sun's longer rays keep
// to (m), the most steps per quarter in the near cascade (the others take 3/4 of it), how far along the light it looks (m).
#define CLOUD_SHADOW_STEPS 16
#define CLOUD_SHADOW_STEP 120.0
#define CLOUD_SHADOW_QUARTER_MAX 8
#define CLOUD_SHADOW_MAXLEN 24000.0

constexpr sampler kCloudRepeat(filter::linear, mip_filter::linear, address::repeat);
constexpr sampler kCloudClamp(filter::linear, address::clamp_to_edge);

static float cloudRemap(float x, float a, float b, float c, float d) { return c + (x - a) / (b - a) * (d - c); }

// ------------------------------------------------------------------------------------------------------------ noise

static uint cloudHash(uint3 p, uint seed) {
    uint h = seed ^ (p.x * 0x8da6b343u) ^ (p.y * 0xd8163841u) ^ (p.z * 0xcb1ab31fu);
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}
static float3 cloudHash3(uint3 p, uint seed) {
    return float3(float(cloudHash(p, seed) >> 8), float(cloudHash(p, seed + 0x9e3779b9u) >> 8), float(cloudHash(p, seed + 0x3c6ef372u) >> 8)) / 16777216.0;
}
static uint3 cloudWrap(int3 c, int period) { return uint3((c % period + period) % period); }

// Gradient (Perlin) noise tiling every `period` cells, about -0.9..0.9.
static float cloudPerlin(float3 p, int period, uint seed) {
    float3 i = floor(p), f = p - i;
    float3 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    int3 c = int3(i);
    float n[8];
    for (int k = 0; k < 8; k++) {
        int3 o = int3(k & 1, (k >> 1) & 1, k >> 2);
        float3 g = normalize(cloudHash3(cloudWrap(c + o, period), seed) * 2.0 - 1.0 + 1e-4);
        n[k] = dot(g, f - float3(o));
    }
    return mix(mix(mix(n[0], n[1], u.x), mix(n[2], n[3], u.x), u.y), mix(mix(n[4], n[5], u.x), mix(n[6], n[7], u.x), u.y), u.z);
}

// Worley (cellular) noise tiling every `period` cells: 1 at a cell's feature point, falling to 0 a cell's width away.
static float cloudWorley(float3 p, int period, uint seed) {
    float3 i = floor(p), f = p - i;
    int3 c = int3(i);
    float d2 = 9.0;
    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                int3 o = int3(x, y, z);
                float3 v = float3(o) + cloudHash3(cloudWrap(c + o, period), seed) - f;
                d2 = min(d2, dot(v, v));
            }
        }
    }
    return 1.0 - saturate(sqrt(d2));
}

// The base shape (CLOUD_BASE_TILE a side, one channel): a Perlin FBM from 4 cells a tile, dilated by Worley cells
// (Perlin-Worley: billowy, connected shapes), eroded by a Worley FBM from 4 to 32 cells (Schneider's base cloud).
kernel void clouds_noise_base(texture3d<float, access::write> out [[texture(0)]], uint3 gid [[thread_position_in_grid]]) {
    uint n = out.get_width();
    if (gid.x >= n || gid.y >= n || gid.z >= n) return;
    float3 p = (float3(gid) + 0.5) / float(n);
    float pf = 0.0, amp = 1.0, norm = 0.0;
    for (int o = 0; o < 4; o++) {
        int per = 4 << o;
        pf += amp * cloudPerlin(p * float(per), per, 11u + uint(o));
        norm += amp;
        amp *= 0.5;
    }
    pf = saturate(pf / norm * 1.1 + 0.5);
    // Worley octaves up to 32 cells a tile (4 texels a cell): at 64, 2 texels a cell, the feature points snap to the texel
    // grid and flat cloud tops showed a regular waffle lattice.
    float w[4];
    for (int o = 0; o < 4; o++) {
        int per = 4 << o;
        w[o] = cloudWorley(p * float(per), per, 101u + uint(o));
    }
    float w1 = w[0] * 0.625 + w[1] * 0.25 + w[2] * 0.125;
    float w2 = w[1] * 0.625 + w[2] * 0.25 + w[3] * 0.125;
    float w3 = w[2] * 0.75 + w[3] * 0.25;
    float pw = saturate(cloudRemap(pf, 0.0, 1.0, w1, 1.0));
    float fbm = w1 * 0.625 + w2 * 0.25 + w3 * 0.125;
    out.write(float4(saturate(cloudRemap(pw, fbm - 1.0, 1.0, 0.0, 1.0))), gid);
}

// The detail noise (CLOUD_DETAIL_TILE a side, one channel): a Worley FBM from 2 to 8 cells.
kernel void clouds_noise_detail(texture3d<float, access::write> out [[texture(0)]], uint3 gid [[thread_position_in_grid]]) {
    uint n = out.get_width();
    if (gid.x >= n || gid.y >= n || gid.z >= n) return;
    float3 p = (float3(gid) + 0.5) / float(n);
    // Up to 8 cells a tile (4 texels a cell), as the base's.
    float w[3];
    for (int o = 0; o < 3; o++) {
        int per = 2 << o;
        w[o] = cloudWorley(p * float(per), per, 201u + uint(o));
    }
    float a = w[0] * 0.625 + w[1] * 0.25 + w[2] * 0.125, b = w[1] * 0.75 + w[2] * 0.25, c = w[2];
    out.write(float4(a * 0.625 + b * 0.25 + c * 0.125), gid);
}

// The weather map (CLOUD_WEATHER_TILE a side): r coverage (a Perlin FBM from 16 km down to 1 km, clumped by 8 km Worley
// cells), g the cloud type (0 stratus, 1 cumulus; a slow FBM).
kernel void clouds_noise_weather(texture2d<float, access::write> out [[texture(0)]], uint2 gid [[thread_position_in_grid]]) {
    uint n = out.get_width();
    if (gid.x >= n || gid.y >= n) return;
    float2 p = (float2(gid) + 0.5) / float(n);
    float cov = 0.0, amp = 1.0, norm = 0.0;
    for (int o = 0; o < 5; o++) {
        int per = 8 << o;
        cov += amp * cloudPerlin(float3(p * float(per), 0.5), per, 301u + uint(o));
        norm += amp;
        amp *= 0.55;
    }
    cov = cov / norm * 1.1 + 0.5;
    float clump = cloudWorley(float3(p * 16.0, 0.5), 16, 311u);
    cov = saturate(cov * 0.8 + (clump - 0.5) * 0.45);
    float ty = 0.0;
    amp = 1.0;
    norm = 0.0;
    for (int o = 0; o < 3; o++) {
        int per = 4 << o;
        ty += amp * cloudPerlin(float3(p * float(per), 2.5), per, 321u + uint(o));
        norm += amp;
        amp *= 0.5;
    }
    ty = saturate(ty / norm * 1.2 + 0.5);
    out.write(float4(cov, ty, 0.0, 1.0), gid);
}

// ---------------------------------------------------------------------------------------------------------- density

// The cloud type's density over the height in the layer (0 its base, 1 its top): a quick rise off the flat base, then a
// rounded falloff toward a top that's low for stratus (type 0) and the layer's top for cumulus (type 1).
static float cloudsProfile(float h, float type) {
    float top = mix(CLOUD_STRATUS_TOP, 1.0, type);
    return saturate(h * CLOUD_BASE_SHARP) * (1.0 - smoothstep(top * 0.35, top * 0.92, h));
}

// Coverage and type where the weather map has them for a camera-relative x, z (lod: its mip, for long steps).
static float2 cloudsWeather(float2 xz, constant CloudFrame& cf, texture2d<float> weather, float lod = 0.0) {
    float4 w = weather.sample(kCloudRepeat, (xz + cf.noise.xy) * (1.0 / CLOUD_WEATHER_TILE), level(lod));
    float cov = saturate(cf.shape.x + CLOUD_COVER_BIAS + (w.r - 0.5) * CLOUD_COVER_SPREAD);
    float type = mix(saturate(0.62 + (w.g - 0.5) * 1.3), CLOUD_RAIN_TYPE, cf.shape.z);
    // Thunder: cumulonimbus, the deck towering to the layer's top.
    type = mix(type, 1.0, cf.shape.w);
    return float2(cov, type);
}

// Density (0-1, times the frame's scale) at a camera-relative point at altitude alt, with the weather there (coverage,
// type). fine: with the detail noise's erosion; lod: the base shape's mip (for long steps).
static float cloudsDensity(float3 rel, float alt, float2 wt, constant CloudFrame& cf, texture3d<float> base,
                           texture3d<float> detail, bool fine, float lod) {
    float h = (alt - cf.layer.x) / (cf.layer.y - cf.layer.x);
    if (h <= 0.0 || h >= 1.0 || wt.x <= 0.0) return 0.0;
    float prof = cloudsProfile(h, wt.y);
    if (prof <= 0.0) return 0.0;
    float3 np = float3(rel.x + cf.noise.z, alt, rel.z + cf.noise.w);
    float shape = base.sample(kCloudRepeat, np * (1.0 / CLOUD_BASE_TILE), level(lod)).r;
    float d = saturate(cloudRemap(shape * prof, 1.0 - wt.x, 1.0, 0.0, 1.0)) * wt.x;
    if (d > 0.0) {
        // Two lookups, the second turned 45 degrees and a factor sqrt 2 finer (an integer matrix: it still tiles with the
        // camera's wrap): one lattice of Worley cells is regular enough to show on flat cloud tops, two together aren't.
        // Without the detail (far away, long steps) its mean erosion still applies: dropping it made the far clouds
        // denser, a flat deck beyond a line where the detail stopped.
        float dfbm = 0.5;
        if (fine && d < CLOUD_DETAIL_MAXD) {
            float3 dp = np * (1.0 / CLOUD_DETAIL_TILE);
            dfbm = 0.5 * (detail.sample(kCloudRepeat, dp, level(0.0)).r
                        + detail.sample(kCloudRepeat, float3(dp.x - dp.z, dp.y * 1.41421356, dp.x + dp.z), level(0.0)).r);
        }
        // Wispy toward the base, billowy toward the top.
        float m = mix(dfbm, 1.0 - dfbm, saturate(h * 5.0));
        d = saturate(cloudRemap(d, m * CLOUD_EROSION, 1.0, 0.0, 1.0));
    }
    return d * cf.shape.y;
}

// Where the ray t dir (camera-relative, m) is inside the layer, t >= 0, before it reaches the ground (sea level) and
// CLOUD_MAX_DIST: up to two stretches (from inside or above the layer a ray can go down through it and, past the planet's
// curve, back up through it far away). The altitude along it: alt(t) = h + t dir.y + t^2 k, k = (1 - dir.y^2) / 2R, so
// the layer is [roots of alt = top] minus (roots of alt = base).
static int cloudsSegments(float3 dir, constant CloudFrame& cf, thread float2& s0, thread float2& s1) {
    float h = cf.layer.z;
    float a = max((1.0 - dir.y * dir.y) * cf.layer.w, 1e-14), b = dir.y;
    float tEnd = CLOUD_MAX_DIST;
    float g0, g1;
    if (h > 0.0 && cloudsRoots(a, b, h, g0, g1) && g1 > 0.0) tEnd = min(tEnd, g0 > 0.0 ? g0 : g1);
    float u0, u1;
    if (!cloudsRoots(a, b, h - cf.layer.y, u0, u1)) return 0;
    float v0, v1;
    bool dips = cloudsRoots(a, b, h - cf.layer.x, v0, v1);
    float2 x0 = float2(u0, dips ? min(v0, u1) : u1);
    float2 x1 = dips ? float2(max(v1, u0), u1) : float2(1.0, 0.0);
    x0 = float2(max(x0.x, 0.0), min(x0.y, tEnd));
    x1 = float2(max(x1.x, 0.0), min(x1.y, tEnd));
    int n = 0;
    if (x0.y > x0.x) { s0 = x0; n = 1; }
    if (x1.y > x1.x) { if (n == 0) s0 = x1; else s1 = x1; n++; }
    return n;
}

// The light reaching a point of the layer (cf.light: the sun, or the moon at night): its illuminance through Sky's
// transmittance table at the point's altitude and local light height (the vertical tilts by x / R across the planet),
// and the terminator (a cloud 1.5 km up sees the sun 1.2 degrees after the ground's sunset).
static float3 cloudsLightAt(float3 rel, float alt, constant CloudFrame& cf, constant SkyFrame& sky, texture2d<float> trans) {
    float3 L = cf.light.xyz;
    float r = sky.atmo.planet.x + max(alt, 1.0) * 0.001;
    float mu = clamp(L.y + dot(rel.xz, L.xz) * 2.0 * cf.layer.w, -1.0, 1.0);
    return skyTransmittanceToSpace(trans, sky.atmo, r, mu) * skySunVisible(sky.atmo, r, mu, sky.sunHoriz.z) * cf.light.w;
}

// The optical depth toward the light from a point of the layer (h its height in it): the light's own stretch from the point
// up to the deep shadow map's next stored height (a quarter of the layer) from CLOUD_LIGHT_SAMPLES samples of the density,
// then that height's stored optical depth, exact, from the map. Interpolating the map's four heights instead put the
// deck's own optical depth onto its top wherever a top sat near a stored height, and the map's 64 m texels showed as a
// waffle lattice on flat tops seen from above. A low sun's long stretches stop at CLOUD_LIGHT_STRETCH and take the map's
// interpolation from there.
static float cloudsLightDepth(float3 p, float h, constant CloudFrame& cf, texture2d<float> weather, texture3d<float> base,
                              texture3d<float> detail, texture2d<float> shadow, float jit) {
    float3 L = cf.light.xyz;
    float thick = cf.layer.y - cf.layer.x;
    float hb = min(floor(h * 4.0) + 1.0, 4.0) * 0.25;    // the next stored height above
    float ly = max(L.y + dot(p.xz, L.xz) * 2.0 * cf.layer.w, 0.02);
    float len = min((hb - h) * thick / ly, CLOUD_LIGHT_STRETCH);
    float tau = 0.0;
    for (int i = 0; i < CLOUD_LIGHT_SAMPLES; i++) {
        float3 q = p + L * (len * (float(i) + 0.25 + 0.5 * jit) / float(CLOUD_LIGHT_SAMPLES));
        float aq = cloudsAltitude(q, cf);
        tau += cloudsDensity(q, aq, cloudsWeather(q.xz, cf, weather), cf, base, detail, false, 0.0);
    }
    tau *= len / float(CLOUD_LIGHT_SAMPLES) * CLOUD_SIGMA;
    float3 e = p + L * len;
    float ae = cloudsAltitude(e, cf);
    if (ae >= cf.layer.y) return tau;
    float4 d = cloudsDeepShadow(cloudsToMiddle(e, ae, cf), cf, shadow);
    float he = (ae - cf.layer.x) / thick;
    // At the stored height (within a little): its value; short of it (a low sun's stretch cut), the map's interpolation.
    if (hb >= 1.0) return tau;
    float stored = hb > 0.7 ? d.x : (hb > 0.45 ? d.y : d.z);
    return tau + (abs(he - hb) < 0.02 ? stored : cloudsDeepAt(d, he));
}

static float cloudsHG(float g, float c) {
    float g2 = g * g;
    return (1.0 - g2) / (4.0 * SKY_PI * pow(max(1.0 + g2 - 2.0 * g * c, 1e-4), 1.5));
}

// The march's jitter: a blue-noise mask (void and cluster, 64 x 64, Clouds.swift) offset by the golden ratio every time
// the texel is marched (blue across texels, low-discrepancy in time, so the history converges evenly). Interleaved
// gradient noise across the 2 x 2 blocks drew bands at grazing angles; a hash across texels, white noise, converged
// to blotches.
static float cloudsJitter(uint2 p, float k, texture2d<float> bn) {
    return fract(bn.read(p % 64u).r + 0.6180339887 * fmod(k, 4096.0));
}

// A view ray through the layer: its light (scene units, before the air in front), transmittance and the
// transmittance-weighted depth of where its light came from (m; 0 if it met no cloud).
struct CloudsRay { float3 L; float T; float depth; };

static CloudsRay cloudsMarch(float3 dir, int segs, float2 s0, float2 s1, float jit, constant CloudFrame& cf,
                             constant SkyFrame& sky, device const float4* env, texture2d<float> weather,
                             texture3d<float> base, texture3d<float> detail, texture2d<float> trans, texture2d<float> shadow) {
    CloudsRay r;
    r.L = 0.0;
    r.T = 1.0;
    r.depth = 0.0;
    float c = dot(dir, cf.light.xyz);
    float ph[3];
    float gs = 1.0;
    for (int o = 0; o < 3; o++) {
        ph[o] = mix(cloudsHG(CLOUD_G_FORWARD * gs, c), cloudsHG(CLOUD_G_BACK * gs, c), CLOUD_G_MIX);
        gs *= CLOUD_MS_C;
    }
    float powderK = CLOUD_POWDER * saturate(0.5 - 0.5 * c);
    float3 ambSky = env[0].rgb * (0.5 * CLOUD_AMBIENT), ambGround = env[1].rgb * (0.5 * CLOUD_AMBIENT * CLOUD_AMB_GROUND);
    float thick = cf.layer.y - cf.layer.x;
    float dsum = 0.0, wsum = 0.0;
    bool lit = cf.light.w > 0.0 && cf.cascade[0].w > 0.0;
    for (int sg = 0; sg < segs; sg++) {
        float2 seg = sg == 0 ? s0 : s1;
        float len = seg.y - seg.x;
        int n = clamp(int(ceil(len / CLOUD_STEP_MIN)), CLOUD_STEPS_MIN, CLOUD_STEPS_MAX);
        float dt0 = len / float(n);
        float lod = max(log2(dt0 / 64.0), 0.0), wlod = max(log2(dt0 / 128.0), 0.0);
        bool fine = dt0 < CLOUD_FINE_STEP;
        float3 sunL = float3(-1.0);
        float t = seg.x + jit * dt0, dt = dt0, skipped = 0.0;
        for (int i = 0; i < n + 16 && t < seg.y; i++) {
            float3 p = dir * t;
            float alt = cloudsAltitude(p, cf);
            float2 wt = cloudsWeather(p.xz, cf, weather, wlod);
            float dens = wt.x > 0.0 ? cloudsDensity(p, alt, wt, cf, base, detail, fine && t < CLOUD_FINE_DIST, lod) : 0.0;
            if (dens <= 1e-4) {
                // Clear air: longer steps, until the next cloud.
                t += dt;
                skipped = dt;
                dt = min(dt * 1.5, dt0 * CLOUD_SKIP_MAX);
                continue;
            }
            dt = dt0;
            if (skipped > dt0 * 1.01) {
                // A cloud after a long step: back to one short step past the last clear sample, so its front is sampled
                // every short step from there. The jitter spans a short step, not a long one: long steps into the fronts at
                // the same distances in every texel drew rings around a camera inside or over the layer.
                t -= skipped - dt0;
                skipped = 0.0;
                continue;
            }
            skipped = 0.0;
            float sigma = dens * CLOUD_SIGMA;
            float hfr = saturate((alt - cf.layer.x) / thick);
            float3 S = ambSky * mix(CLOUD_AMB_BASE, 1.0, hfr) + ambGround * (1.0 - hfr);
            if (lit) {
                if (sunL.x < 0.0) sunL = cloudsLightAt(p, alt, cf, sky, trans);
                float tauL = cloudsLightDepth(p, hfr, cf, weather, base, detail, shadow, jit);
                float ms = CLOUD_MS_DIFFUSE / (4.0 * SKY_PI) * exp(-CLOUD_MS_ABSORB * tauL) / (1.0 + CLOUD_MS_FALL * tauL), ka = 1.0, kb = 1.0;
                for (int o = 0; o < 3; o++) {
                    ms += ka * ph[o] * exp(-kb * tauL);
                    ka *= CLOUD_MS_A;
                    kb *= CLOUD_MS_B;
                }
                float powder = mix(1.0, 1.0 - exp(-dens * CLOUD_POWDER_DEPTH), powderK);
                S += sunL * (ms * powder);
            }
            S *= sigma * CLOUD_ALBEDO;
            float st = exp(-sigma * dt0);
            r.L += r.T * (S - S * st) / sigma;
            float dT = r.T * (1.0 - st);
            dsum += t * dT;
            wsum += dT;
            r.T *= st;
            if (r.T < CLOUD_T_MIN) break;
            t += dt0;
        }
        if (r.T < CLOUD_T_MIN) break;
    }
    r.depth = wsum > 1e-6 ? dsum / wsum : 0.0;
    return r;
}

// The air between the camera and the cloud (Sky's aerial perspective at its depth): its light dimmed and the haze in front
// of it added where it covers what's behind. rgb what the clouds add to a pixel behind them, a their transmittance.
static float4 cloudsThroughAir(CloudsRay r, float3 dir, constant SkyFrame& sky, texture3d<float> apScatter, texture3d<float> apTrans) {
    if (r.depth <= 0.0) return float4(0.0, 0.0, 0.0, r.T);
    SkyAerial ap = skyAerialPerspective(dir * r.depth, sky, apScatter, apTrans);
    float3 front = skyOvercast(sky, ap.inscatter) + skyNight(sky, dir) * (1.0 - ap.transmittance);
    return float4(r.L * ap.transmittance + (1.0 - r.T) * front, r.T);
}

// Where a view ray meets the cirrus sheet (m; 0 if it doesn't: under the planet's horizon, or past CLOUD_MAX_DIST).
static float cloudsCirrusHit(float3 dir, constant CloudFrame& cf) {
    if (CIRRUS_COVER <= 0.0 || cf.layer.z >= CIRRUS_ALT) return 0.0;
    float a = max((1.0 - dir.y * dir.y) * cf.layer.w, 1e-14);
    float r0, r1;
    if (!cloudsRoots(a, dir.y, cf.layer.z - CIRRUS_ALT, r0, r1)) return 0.0;
    float t = r1;
    float g0, g1;
    if (cf.layer.z > 0.0 && cloudsRoots(a, dir.y, cf.layer.z, g0, g1) && g1 > 0.0 && (g0 > 0.0 ? g0 : g1) < t) return 0.0;
    return t < 1.3 * CLOUD_MAX_DIST ? t : 0.0;
}

// The cirrus at a ray's crossing tc: fibrous streaks (the detail noise stretched along the wind, in the integer frame
// (2, 1), (-1, 2)), in patches where the weather map (another part of it) allows, a thin sheet lit by a single scattering
// with a forward lobe (ice) and the sky's ambient, through the air in front. rgb what it adds, a its transmittance.
static float4 cloudsCirrus(float3 dir, float tc, constant CloudFrame& cf, constant SkyFrame& sky, device const float4* env,
                           texture2d<float> weather, texture3d<float> detail, texture2d<float> trans,
                           texture3d<float> apScatter, texture3d<float> apTrans) {
    float3 p = dir * tc;
    float2 xz = p.xz + cf.noise.xy;
    float patch = weather.sample(kCloudRepeat, xz * (1.0 / CLOUD_WEATHER_TILE) + 0.5, level(1.0)).r;
    float cov = saturate((patch - 0.5) * 3.0 + CIRRUS_COVER) * (1.0 - cf.shape.z);
    if (cov <= 0.0) return float4(0.0, 0.0, 0.0, 1.0);
    // The streaks bent and spaced unevenly by a slow field (another part of the weather map): straight, they repeated across
    // every tile of the noise, a comb (a barcode with 2 km tiles).
    float warp = weather.sample(kCloudRepeat, xz * (1.0 / CLOUD_WEATHER_TILE) + 0.25, level(2.0)).r;
    float2 uv = float2((2.0 * xz.x + xz.y) / CIRRUS_ALONG, (2.0 * xz.y - xz.x) / CIRRUS_ACROSS + (warp - 0.5) * 6.0);
    // The mip for the texel's footprint on the sheet (a half-resolution texel is about 1.1 mrad; long toward the horizon):
    // level 0 at a distance sparkled into dots.
    float foot = tc * 0.0011 / max(abs(dir.y), 0.05);
    float lod = clamp(log2(foot * (32.0 * 2.236 / CIRRUS_ACROSS)), 0.0, 5.0);
    float n = detail.sample(kCloudRepeat, float3(uv, 0.37), level(lod + 1.0)).r * 0.6
            + detail.sample(kCloudRepeat, float3(uv * float2(2.0, 2.0), 0.61), level(lod + 1.0)).r * 0.4;
    float d = smoothstep(1.0 - 0.6 * cov, 1.25 - 0.6 * cov, n) * cov;
    if (d <= 0.0) return float4(0.0, 0.0, 0.0, 1.0);
    // The sheet's slant: its local vertical tilts by x / R across the planet.
    float mu = max(abs(dir.y + dot(p.xz, dir.xz) * 2.0 * cf.layer.w), 0.03);
    float tau = d * CIRRUS_TAU / mu;
    float T = exp(-tau);
    float c = dot(dir, cf.light.xyz);
    float ph = mix(cloudsHG(CIRRUS_G, c), cloudsHG(-0.2, c), 0.25);
    float3 E = 0.0;
    if (cf.light.w > 0.0) {
        float r = sky.atmo.planet.x + CIRRUS_ALT * 0.001;
        float lmu = clamp(cf.light.y + dot(p.xz, cf.light.xz) * 2.0 * cf.layer.w, -1.0, 1.0);
        E = skyTransmittanceToSpace(trans, sky.atmo, r, lmu) * skySunVisible(sky.atmo, r, lmu, sky.sunHoriz.z) * cf.light.w * ph;
    }
    float3 L = (E + env[0].rgb * 0.5) * (1.0 - T);
    SkyAerial ap = skyAerialPerspective(p, sky, apScatter, apTrans);
    float3 front = skyOvercast(sky, ap.inscatter) + skyNight(sky, dir) * (1.0 - ap.transmittance);
    return float4(L * ap.transmittance + (1.0 - T) * front, T);
}

// ------------------------------------------------------------------------------------------------------- per frame

// The sky's light for the clouds: out[0] the open sky's mean radiance over the upper hemisphere (512 directions through
// the sky view table), out[1] the ground's (the atmosphere's ground albedo, lit by the sun and that sky), both scene units.
kernel void clouds_env(device float4* out [[buffer(0)]], constant SkyFrame& f [[buffer(1)]],
                       texture2d<float> trans [[texture(0)]], texture2d<float> skyView [[texture(1)]],
                       uint i [[thread_index_in_threadgroup]]) {
    threadgroup float4 acc[64][2];
    float3 s0 = 0.0, s1 = 0.0;
    for (uint k = 0; k < 8; k++) {
        uint j = i * 8u + k;
        float y = (float(j) + 0.5) / 512.0;
        float rr = sqrt(max(0.0, 1.0 - y * y));
        float phi = float(j) * 2.39996323;
        float3 dir = float3(rr * cos(phi), y, rr * sin(phi));
        float3 L = skyLuminance(f, dir, skyView);
        s0 += L;
        s1 += L * y;
    }
    acc[i][0] = float4(s0, 0.0);
    acc[i][1] = float4(s1, 0.0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 32; s > 0; s >>= 1) {
        if (i < s) {
            acc[i][0] += acc[i + s][0];
            acc[i][1] += acc[i + s][1];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (i != 0) return;
    float3 mean = acc[0][0].rgb / 512.0;
    float3 irr = acc[0][1].rgb * (2.0 * SKY_PI / 512.0);
    float r0 = f.view.y;
    float3 sun = skyTransmittanceToSpace(trans, f.atmo, r0, f.sun.y) * skySunVisible(f.atmo, r0, f.sun.y, f.sunHoriz.z) * f.sun.w;
    out[0] = float4(mean, 0.0);
    out[1] = float4(f.atmo.planet.w * (irr + sun * max(f.sun.y, 0.0)) / SKY_PI, 0.0);
}

// The deep shadow map: three cascades side by side, each texel the light's ray through the layer, crossing its middle
// above the texel's x, z (camera-relative): the optical depth from the layer's top down to 3/4, 1/2, 1/4 of the way and
// all of it (the altitude falls about evenly along the ray).
kernel void clouds_shadow(texture2d<float, access::write> out [[texture(0)]],
                          texture3d<float> base [[texture(1)]], texture3d<float> detail [[texture(2)]],
                          texture2d<float> weather [[texture(3)]], constant CloudFrame& cf [[buffer(0)]],
                          constant uint4& sel [[buffer(1)]], uint2 tid [[thread_position_in_grid]]) {
    // The cascades made this frame: sel.w of them, sel.x, y, z (the near one every frame, the others in turn).
    uint n = out.get_height();
    uint slot = tid.x / n;
    if (slot >= sel.w || tid.y >= n) return;
    uint ci = slot == 0u ? sel.x : (slot == 1u ? sel.y : sel.z);
    uint2 gid = uint2(ci * n + tid.x - slot * n, tid.y);
    float4 c = cf.cascade[ci];
    float2 u = (float2(gid.x - ci * n, gid.y) + 0.5) / float(n);
    float2 x = c.xy + (u * 2.0 - 1.0) * c.z;
    float3 L = cf.light.xyz;
    if (cf.light.w <= 0.0 || L.y <= 0.0) { out.write(float4(0.0), gid); return; }
    float k = cf.layer.w, mid = 0.5 * (cf.layer.x + cf.layer.y);
    float3 q = float3(x.x, mid - cf.layer.z - dot(x, x) * k, x.y);
    // Along the light through q: alt(s) = mid + s (L.y + 2 k x . L.xz) + s^2 k |L.xz|^2. The top ahead, the base behind.
    float a = max(k * dot(L.xz, L.xz), 1e-14), b = L.y + 2.0 * k * dot(x, L.xz);
    float r0, r1, sTop = CLOUD_SHADOW_MAXLEN, sBase = -CLOUD_SHADOW_MAXLEN;
    if (cloudsRoots(a, b, mid - cf.layer.y, r0, r1)) sTop = min(r1, CLOUD_SHADOW_MAXLEN);
    if (cloudsRoots(a, b, mid - cf.layer.x, r0, r1)) sBase = max(r1 < 0.0 ? r1 : (r0 < 0.0 ? r0 : -CLOUD_SHADOW_MAXLEN), -CLOUD_SHADOW_MAXLEN);
    // Steps of about CLOUD_SHADOW_STEP (a low sun's long rays take more, up to 4 x CLOUD_SHADOW_QUARTER_MAX), four quarters.
    int perQuarter = clamp(int(ceil((sTop - sBase) / (4.0 * CLOUD_SHADOW_STEP))), CLOUD_SHADOW_STEPS / 4,
                           ci == 0u ? CLOUD_SHADOW_QUARTER_MAX : CLOUD_SHADOW_QUARTER_MAX * 3 / 4);
    int steps = perQuarter * 4;
    float ds = (sTop - sBase) / float(steps);
    // The near and middle cascades at the same mip, so their optical depths agree where one blends into the other.
    float lod = ci < 2u ? 0.0 : 1.5;
    float tau = 0.0;
    float4 rec = 0.0;
    for (int i = 0; i < steps; i++) {
        float3 p = q + L * (sTop - (float(i) + 0.5) * ds);
        float alt = cloudsAltitude(p, cf);
        float2 wt = cloudsWeather(p.xz, cf, weather, lod);
        tau += (wt.x > 0.0 ? cloudsDensity(p, alt, wt, cf, base, detail, false, lod) : 0.0) * ds;
        if ((i + 1) % perQuarter == 0) {
            int quarter = (i + 1) / perQuarter;
            if (quarter == 1) rec.x = tau; else if (quarter == 2) rec.y = tau; else if (quarter == 3) rec.z = tau; else rec.w = tau;
        }
    }
    out.write(rec * CLOUD_SIGMA, gid);
}

// Which texels are marched in which frame: one of each block (2 x 2 for an update every 4th frame, 4 x 2 every 8th) in
// turn, in an order that spreads consecutive frames' texels apart. The march runs one thread per block, so every thread
// marches (texels spread through the screen would leave most of each SIMD group waiting on its few marched ones).
constant uint2 kCloudOrder4[4] = { uint2(0, 0), uint2(1, 1), uint2(1, 0), uint2(0, 1) };
constant uint2 kCloudOrder8[8] = { uint2(0, 0), uint2(2, 1), uint2(1, 0), uint2(3, 1), uint2(2, 0), uint2(0, 1), uint2(3, 0), uint2(1, 1) };
static uint2 cloudsBlock() { return CLOUD_UPDATE > 4 ? uint2(4, 2) : uint2(2, 2); }
static uint2 cloudsDueOffset(uint frame) {
    return CLOUD_UPDATE > 4 ? kCloudOrder8[frame % 8u] : kCloudOrder4[frame % 4u];
}

// A texel's view ray, where it runs through the layer, and whether its 2 x 2 pixels are all terrain nearer than where
// the ray meets the clouds (then none of them shows any).
struct CloudsTexel { float3 dir; float2 s0, s1; int segs; float tc; bool blocked; };
static CloudsTexel cloudsTexel(uint2 gid, constant CloudFrame& cf, depth2d<float> depth, bool terrain = true) {
    CloudsTexel x;
    float2 uv = (float2(gid) + 0.5) / cf.size.zw;
    float4 hp = cf.invViewProj * float4(uv * 2.0 - 1.0, 1.0, 1.0);
    x.dir = normalize(hp.xyz / hp.w);
    x.s0 = 0.0;
    x.s1 = 0.0;
    x.segs = cloudsSegments(x.dir, cf, x.s0, x.s1);
    x.tc = cloudsCirrusHit(x.dir, cf);
    x.blocked = false;
    if ((x.segs == 0 && x.tc <= 0.0) || !terrain) return x;
    constexpr sampler gs(filter::nearest, address::clamp_to_edge);
    float2 fuv = (float2(gid * 2u) + 1.0) / cf.size.xy;
    float4 g4 = depth.gather(gs, fuv);
    float dfar = min(min(g4.x, g4.y), min(g4.z, g4.w));
    if (dfar > 0.0) {
        float4 hq = cf.invViewProj * float4(fuv * 2.0 - 1.0, dfar, 1.0);
        x.blocked = length(hq.xyz / hq.w) < (x.segs > 0 ? x.s0.x : x.tc);
    }
    return x;
}

// Last frame's clouds for a texel: the cloud where its light came from (at refDepth along dir; with none, the sky at
// infinity), bilinear over last frame's texels that weren't terrain and whose depth agrees. False if there's none to take
// (off screen last frame, uncovered by terrain, no history).
static bool cloudsHistory(uint2 gid, float3 dir, float refDepth, constant CloudFrame& cf, texture2d<float> histColor,
                          texture2d<float> histDepth, thread float4& hist, thread float& histDepthV) {
    if (cf.camDelta.w <= 0.5) return false;
    float4 pc = refDepth > 0.0 ? cf.prevViewProj * float4(dir * refDepth + cf.camDelta.xyz, 1.0) : cf.prevViewProj * float4(dir, 0.0);
    if (pc.w <= 0.0) return false;
    float2 puv = pc.xy / pc.w * 0.5 + 0.5;
    if (!(all(puv > 0.0) && all(puv < 1.0))) return false;
    // Still (under a hundredth of a texel): the texel itself. The reprojection lands within float error of its center, now
    // and then just past it, and bilinear taps there took a neighbor's history: still views grew a regular lattice.
    float2 motion = puv * cf.size.zw - (float2(gid) + 0.5);
    if (dot(motion, motion) < 1e-4) {
        float dd = histDepth.read(gid).r * 1000.0;
        if (dd < 0.0) return false;
        hist = histColor.read(gid);
        histDepthV = max(dd, 0.0);
        return true;
    }
    // The four texels' depths in one gather (order (0, 1), (1, 1), (1, 0), (0, 0)), their bilinear weights, less where they
    // were terrain or their cloud's depth disagrees.
    constexpr sampler sl(filter::linear, address::clamp_to_edge), sn(filter::nearest, address::clamp_to_edge);
    float4 hd4 = histDepth.gather(sn, puv) * 1000.0;   // stored in km
    float2 fr = fract(puv * cf.size.zw - 0.5);
    float4 bw = float4((1.0 - fr.x) * fr.y, fr.x * fr.y, fr.x * (1.0 - fr.y), (1.0 - fr.x) * (1.0 - fr.y));
    float4 w = bw * float4(hd4 >= 0.0);
    if (refDepth > 0.0) w *= 1.0 - 0.95 * float4(hd4 > 0.0) * float4(abs(hd4 - refDepth) > CLOUD_DEPTH_REJECT * refDepth);
    float hw = dot(w, float4(1.0));
    if (hw <= 0.25) return false;
    if (all(w == bw)) {
        // Nothing left out: the hardware's bilinear (still, at the texel's center: the texel itself).
        hist = histColor.sample(sl, puv, level(0.0));
    } else {
        int2 p0 = int2(floor(puv * cf.size.zw - 0.5)), pmax = int2(cf.size.zw) - 1;
        const int2 offs[4] = { int2(0, 1), int2(1, 1), int2(1, 0), int2(0, 0) };
        hist = 0.0;
        for (int k = 0; k < 4; k++) {
            if (w[k] > 0.0) hist += histColor.read(uint2(clamp(p0 + offs[k], int2(0), pmax))) * w[k];
        }
        hist /= hw;
    }
    float4 cw = w * float4(hd4 > 0.0);
    float cs = dot(cw, float4(1.0));
    histDepthV = cs > 1e-6 ? dot(cw, hd4) / cs : 0.0;
    return true;
}

// A texel marched: its clouds through the air in front, and its depth (0 for none).
static float4 cloudsMarchTexel(CloudsTexel x, float jit, constant CloudFrame& cf, constant SkyFrame& sky, device const float4* env,
                               texture2d<float> weather, texture3d<float> base, texture3d<float> detail, texture2d<float> trans,
                               texture2d<float> shadow, texture3d<float> apScatter, texture3d<float> apTrans, thread float& depthOut) {
    CloudsRay r;
    r.L = 0.0;
    r.T = 1.0;
    r.depth = 0.0;
    if (x.segs > 0) r = cloudsMarch(x.dir, x.segs, x.s0, x.s1, jit, cf, sky, env, weather, base, detail, trans, shadow);
    depthOut = 1.0 - r.T > 0.002 ? r.depth : 0.0;
    float4 c = cloudsThroughAir(r, x.dir, sky, apScatter, apTrans);
    // The cirrus behind: what it adds comes through the cumulus in front.
    if (x.tc > 0.0 && r.T > CLOUD_T_MIN) {
        float4 ci = cloudsCirrus(x.dir, x.tc, cf, sky, env, weather, detail, trans, apScatter, apTrans);
        c = float4(c.rgb + c.a * ci.rgb, c.a * ci.a);
        if (depthOut <= 0.0 && ci.a < 0.998) depthOut = x.tc;
    }
    return c;
}

// The march at half the resolution each way: this frame's texel of each block (one thread a block), blended into its
// history. The new history is what the composite reads: rgb what the clouds add to whatever is behind them (their light
// through the air in front), a their transmittance; the depth texture the transmittance-weighted depth (m), 0 where
// there's no cloud, -1 where the texel's 2 x 2 pixels are all terrain nearer than the clouds (stored in km: half floats).
kernel void clouds_march(texture2d<float, access::write> outColor [[texture(0)]],
                         texture2d<float, access::write> outDepth [[texture(1)]],
                         texture2d<float> histColor [[texture(2)]],
                         texture2d<float> histDepth [[texture(3)]],
                         depth2d<float> depth [[texture(4)]],
                         texture3d<float> base [[texture(5)]],
                         texture3d<float> detail [[texture(6)]],
                         texture2d<float> weather [[texture(7)]],
                         texture2d<float> trans [[texture(8)]],
                         texture3d<float> apScatter [[texture(9)]],
                         texture3d<float> apTrans [[texture(10)]],
                         texture2d<float> shadow [[texture(11)]],
                         texture2d<float> bn [[texture(12)]],
                         constant CloudFrame& cf [[buffer(0)]],
                         constant SkyFrame& sky [[buffer(1)]],
                         device const float4* env [[buffer(2)]],
                         uint2 tid [[thread_position_in_grid]]) {
    uint frame = uint(cf.misc.x);
    uint2 gid = tid * cloudsBlock() + cloudsDueOffset(frame);
    if (gid.x >= uint(cf.size.z) || gid.y >= uint(cf.size.w)) return;
    CloudsTexel x = cloudsTexel(gid, cf, depth);
    if ((x.segs == 0 && x.tc <= 0.0) || x.blocked) {
        outColor.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        outDepth.write(float4(x.blocked ? -1.0 : 0.0), gid);
        return;
    }
    float curDepth;
    float4 cur = cloudsMarchTexel(x, cloudsJitter(gid, float(frame / uint(CLOUD_UPDATE)), bn), cf, sky, env, weather, base, detail, trans,
                                  shadow, apScatter, apTrans, curDepth);
    float4 hist = 0.0;
    float histDepthV = 0.0;
    float4 res = cur;
    float resDepth = curDepth;
    if (cloudsHistory(gid, x.dir, curDepth, cf, histColor, histDepth, hist, histDepthV)) {
        float a = cf.misc.z;
        res = mix(hist, cur, a);
        float wc = (1.0 - cur.a) * a, wh = (1.0 - hist.a) * (1.0 - a);
        resDepth = wc + wh > 1e-6 ? (curDepth * wc + histDepthV * wh) / (wc + wh) : 0.0;
    }
    outColor.write(res, gid);
    outDepth.write(float4(resDepth * 0.001), gid);
}

// Every other texel: its history carried over, reprojected (at last frame's depth here). Where there's none to take, it's
// marched now (the first frame, the screen's edges turning in, texels terrain uncovered). Offline at the panel's resolution
// this pass took about half the march stage's 1 ms with the terrain test (the full-resolution depth, 31 MB, read again)
// and R32Float depths: it's bandwidth (history read and written for 7 of every 8 texels).
kernel void clouds_carry(texture2d<float, access::write> outColor [[texture(0)]],
                         texture2d<float, access::write> outDepth [[texture(1)]],
                         texture2d<float> histColor [[texture(2)]],
                         texture2d<float> histDepth [[texture(3)]],
                         depth2d<float> depth [[texture(4)]],
                         texture3d<float> base [[texture(5)]],
                         texture3d<float> detail [[texture(6)]],
                         texture2d<float> weather [[texture(7)]],
                         texture2d<float> trans [[texture(8)]],
                         texture3d<float> apScatter [[texture(9)]],
                         texture3d<float> apTrans [[texture(10)]],
                         texture2d<float> shadow [[texture(11)]],
                         texture2d<float> bn [[texture(12)]],
                         constant CloudFrame& cf [[buffer(0)]],
                         constant SkyFrame& sky [[buffer(1)]],
                         device const float4* env [[buffer(2)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(cf.size.z) || gid.y >= uint(cf.size.w)) return;
    uint frame = uint(cf.misc.x);
    if (all(gid % cloudsBlock() == cloudsDueOffset(frame))) return;   // clouds_march writes it
    // Without the terrain test (it reads the whole depth buffer): a texel terrain has newly covered keeps its cloud until
    // its turn (sky pixels beside it want that), one it has uncovered finds no history (its texels were terrain) and is
    // marched now.
    CloudsTexel x = cloudsTexel(gid, cf, depth, false);
    if ((x.segs == 0 && x.tc <= 0.0) || x.blocked) {
        outColor.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        outDepth.write(float4(x.blocked ? -1.0 : 0.0), gid);
        return;
    }
    float4 hist = 0.0;
    float histDepthV = 0.0;
    if (cloudsHistory(gid, x.dir, max(histDepth.read(gid).r, 0.0) * 1000.0, cf, histColor, histDepth, hist, histDepthV)) {
        outColor.write(hist, gid);
        outDepth.write(float4(histDepthV * 0.001), gid);
        return;
    }
    float curDepth;
    float4 cur = cloudsMarchTexel(x, cloudsJitter(gid, float(frame), bn), cf, sky, env, weather, base, detail, trans, shadow, apScatter,
                                  apTrans, curDepth);
    outColor.write(cur, gid);
    outDepth.write(float4(curDepth * 0.001), gid);
}

// With the camera still (nothing moved since last frame): this frame's texel of each block blended into the history in
// place. Every other texel keeps what it holds, so there's no carry pass (a copy of the whole buffer: about half the
// march stage's cost offline) and no swap.
kernel void clouds_march_still(texture2d<float, access::read_write> hist [[texture(0)]],
                               texture2d<float, access::read_write> histDepth [[texture(1)]],
                               depth2d<float> depth [[texture(4)]],
                               texture3d<float> base [[texture(5)]],
                               texture3d<float> detail [[texture(6)]],
                               texture2d<float> weather [[texture(7)]],
                               texture2d<float> trans [[texture(8)]],
                               texture3d<float> apScatter [[texture(9)]],
                               texture3d<float> apTrans [[texture(10)]],
                               texture2d<float> shadow [[texture(11)]],
                               texture2d<float> bn [[texture(12)]],
                               constant CloudFrame& cf [[buffer(0)]],
                               constant SkyFrame& sky [[buffer(1)]],
                               device const float4* env [[buffer(2)]],
                               uint2 tid [[thread_position_in_grid]]) {
    uint frame = uint(cf.misc.x);
    uint2 gid = tid * cloudsBlock() + cloudsDueOffset(frame);
    if (gid.x >= uint(cf.size.z) || gid.y >= uint(cf.size.w)) return;
    CloudsTexel x = cloudsTexel(gid, cf, depth);
    if ((x.segs == 0 && x.tc <= 0.0) || x.blocked) {
        hist.write(float4(0.0, 0.0, 0.0, 1.0), gid);
        histDepth.write(float4(x.blocked ? -1.0 : 0.0), gid);
        return;
    }
    float curDepth;
    float4 cur = cloudsMarchTexel(x, cloudsJitter(gid, float(frame / uint(CLOUD_UPDATE)), bn), cf, sky, env, weather, base, detail, trans,
                                  shadow, apScatter, apTrans, curDepth);
    float4 res = cur;
    float resDepth = curDepth;
    float hd = histDepth.read(gid).r * 1000.0;
    if (cf.camDelta.w > 0.5 && hd >= 0.0) {
        float4 h = hist.read(gid);
        float a = cf.misc.z;
        res = mix(h, cur, a);
        float wc = (1.0 - cur.a) * a, wh = (1.0 - h.a) * (1.0 - a);
        resDepth = wc + wh > 1e-6 ? (curDepth * wc + hd * wh) / (wc + wh) : 0.0;
    }
    hist.write(res, gid);
    histDepth.write(float4(resDepth * 0.001), gid);
}

// The clouds into the water's sky map (Lit.swift's lit_water_sky: the same paraboloid mapping as the reflection map, scene
// units), after it's made each frame: every reflection that escapes to the sky (Water.swift's rays that miss, open tops)
// then shows them, with no change to the water's shading.
kernel void clouds_water_sky(texture2d<float, access::read_write> sky [[texture(0)]], texture2d<float> map [[texture(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= sky.get_width() || gid.y >= sky.get_height()) return;
    constexpr sampler sm(filter::linear, address::clamp_to_edge);
    float4 c = map.sample(sm, (float2(gid) + 0.5) / float2(sky.get_width(), sky.get_height()), level(0.0));
    float4 s = sky.read(gid);
    sky.write(float4(s.rgb * c.a + c.rgb, s.a), gid);
}

// The clouds over the upper hemisphere from the camera, for reflections (cloudsReflected): each texel a direction of the
// paraboloid map, marched with a new jitter every frame and blended into what it held (directions are fixed in the world,
// and the clouds are far: no reprojection).
kernel void clouds_reflection(texture2d<float, access::read_write> map [[texture(0)]],
                              texture3d<float> base [[texture(5)]],
                              texture3d<float> detail [[texture(6)]],
                              texture2d<float> weather [[texture(7)]],
                              texture2d<float> trans [[texture(8)]],
                              texture3d<float> apScatter [[texture(9)]],
                              texture3d<float> apTrans [[texture(10)]],
                              texture2d<float> shadow [[texture(11)]],
                              texture2d<float> bn [[texture(12)]],
                              constant CloudFrame& cf [[buffer(0)]],
                              constant SkyFrame& sky [[buffer(1)]],
                              device const float4* env [[buffer(2)]],
                              constant float4& blend [[buffer(3)]],
                              uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= map.get_width() || gid.y >= map.get_height()) return;
    float2 pp = (float2(gid) + 0.5) / float2(map.get_width(), map.get_height()) * 2.0 - 1.0;
    float r2 = dot(pp, pp);
    if (r2 > 1.0) { pp *= rsqrt(r2); r2 = 1.0; }
    CloudsTexel x;
    x.dir = normalize(float3(2.0 * pp.x, 1.0 - r2, 2.0 * pp.y) / (1.0 + r2));
    x.s0 = 0.0;
    x.s1 = 0.0;
    x.segs = cloudsSegments(x.dir, cf, x.s0, x.s1);
    x.tc = cloudsCirrusHit(x.dir, cf);
    x.blocked = false;
    float4 cur = float4(0.0, 0.0, 0.0, 1.0);
    if (x.segs > 0 || x.tc > 0.0) {
        float dd;
        cur = cloudsMarchTexel(x, cloudsJitter(gid, cf.misc.x, bn), cf, sky, env, weather, base, detail, trans, shadow, apScatter, apTrans, dd);
    }
    map.write(blend.x >= 1.0 ? cur : mix(map.read(gid), cur, blend.x), gid);
}

// Cloud shadows on RtShadows' visibility (lit mode): for each traced texel, at the pixel it traced this frame (the same
// rotating pixel of its block), the visibility times the clouds' transmittance toward the light there.
kernel void clouds_vis(texture2d<half, access::read> vis [[texture(0)]], texture2d<half, access::write> out [[texture(1)]],
                       depth2d<float, access::read> depth [[texture(2)]], texture2d<float> shadow [[texture(3)]],
                       constant CloudFrame& cf [[buffer(0)]], constant float4x4& inv [[buffer(1)]],
                       constant uint4& sp [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    half v = vis.read(gid).r;
    uint2 full = uint2(depth.get_width(), depth.get_height());
    uint2 fp = min(gid * sp.x + sp.yz, full - 1u);
    float d = depth.read(fp);
    if (d <= 0.0 || sp.w == 0u || (v <= 0.0h && uint(cf.misc.y) != 4u)) { out.write(half4(v), gid); return; }
    float2 uv = (float2(fp) + 0.5) / float2(full);
    float4 h = inv * float4(uv * 2.0 - 1.0, d, 1.0);
    float T = cloudTransmittanceToSun(h.xyz / h.w, cf, shadow);
    // Debug view 4: the clouds' shadows alone.
    if (uint(cf.misc.y) == 4u) { out.write(half4(half(T)), gid); return; }
    out.write(half4(v * half(T)), gid);
}

// Without anti-aliasing: the clouds over the frame as a pass of its own, after the sky's aerial perspective (programmable
// blending reads the color it replaces).
struct CloudsVOut { float4 pos [[position]]; };
vertex CloudsVOut clouds_fullscreen_vs(uint vid [[vertex_id]]) {
    CloudsVOut o;
    float2 c = float2(float((vid << 1) & 2), float(vid & 2));
    o.pos = float4(c * 2.0 - 1.0, 0.0, 1.0);
    return o;
}
fragment float4 clouds_composite_fs(CloudsVOut in [[stage_in]], float4 dst [[color(0)]],
                                    depth2d<float, access::read> depth [[texture(0)]],
                                    texture2d<float> clouds [[texture(1)]], texture2d<float> cdepth [[texture(2)]],
                                    constant SkyFrame& sky [[buffer(0)]], constant CloudFrame& cf [[buffer(1)]]) {
    uint2 q = uint2(in.pos.xy);
    return float4(cloudsComposite(dst.rgb, q, depth.read(q), sky, cf, clouds, cdepth), dst.a);
}

// ----------------------------------------------------------------------------------------------------- offline view

// Offline (mmc_debug_clouds_render): the sky and a flat ground at sea level (grass, lit by the sun through the clouds'
// shadows and by the sky, through the air), the clouds composited as the anti-aliasing does, tone mapped with AgX
// (Sobotka's, the published fit, as the post chain uses) into 8 bits.
static float3 cloudsAgx(float3 x) {
    const float3x3 inset = float3x3(float3(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
                                    float3(0.0784335999999992, 0.878468636469772, 0.0784336),
                                    float3(0.0792237451477643, 0.0791661274605434, 0.879142973793104));
    const float3x3 outset = float3x3(float3(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
                                     float3(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
                                     float3(-0.0990297440797205, -0.0989611768448433, 1.15107367264116));
    const float minEv = -12.47393, maxEv = 4.026069;
    float3 v = inset * max(x, 1e-10);
    v = (clamp(log2(max(v, 1e-10)), minEv, maxEv) - minEv) / (maxEv - minEv);
    float3 v2 = v * v, v4 = v2 * v2;
    v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v - 0.00232;
    float l = dot(v, float3(0.2126, 0.7152, 0.0722));
    v = l + 1.3 * (v - l);
    v = outset * v;
    return saturate(pow(max(v, 0.0), float3(2.2)));
}

kernel void clouds_debug_view(texture2d<float, access::write> out [[texture(0)]],
                              texture2d<float> clouds [[texture(1)]], texture2d<float> cdepth [[texture(2)]],
                              texture2d<float> skyView [[texture(3)]], texture2d<float> trans [[texture(4)]],
                              texture3d<float> apScatter [[texture(5)]], texture3d<float> apTrans [[texture(6)]],
                              texture2d<float> shadow [[texture(7)]],
                              constant SkyFrame& sky [[buffer(0)]], constant CloudFrame& cf [[buffer(1)]],
                              constant float4& opt [[buffer(2)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float2 uv = (float2(gid) + 0.5) / float2(out.get_width(), out.get_height());
    float4 h = sky.invViewProj * float4(uv * 2.0 - 1.0, 1.0, 1.0);
    float3 dir = normalize(h.xyz / h.w);
    float3 L;
    float d = 0.0;
    // opt.x: the camera's height above the ground (sea level, m); opt.y: 1 to draw the ground; opt.z: exposure.
    if (opt.y > 0.5 && dir.y < -1e-4) {
        float t = opt.x / -dir.y;
        float3 rel = dir * t;
        float Tc = cloudTransmittanceToSun(rel, cf, shadow);
        float r0 = sky.view.y;
        float3 sun = skyTransmittanceToSpace(trans, sky.atmo, r0, sky.sun.y) * skySunVisible(sky.atmo, r0, sky.sun.y, sky.sunHoriz.z);
        float3 tz = skyTransmittanceToSpace(trans, sky.atmo, r0, 1.0);
        // The relight's scale: the sun at the zenith plus about 12% for the sky is a white top's 1 (Lit.swift, lit_env).
        float3 E = sun * max(sky.sun.y, 0.0) * Tc / (1.12 * dot(tz, float3(0.2126, 0.7152, 0.0722))) + float3(0.62, 0.78, 1.0) * 0.12;
        L = skyApplyAerial(skyDecode(float3(0.35, 0.55, 0.22)) * E, rel, sky, apScatter, apTrans);
        // The ground's depth as the level's would hold it: the infinite reverse-Z projection's near (0.05) over the
        // distance along the view's axis (the near plane's middle, through the inverse, is the axis).
        float4 a = sky.invViewProj * float4(0.0, 0.0, 1.0, 1.0);
        d = 0.05 / max(dot(rel, normalize(a.xyz / a.w)), 1e-3);
    } else {
        L = skyPixel(sky, dir, skyView, trans, 0.001);
    }
    float3 c = cloudsComposite(skyEncode(L), uint2(gid), d, sky, cf, clouds, cdepth);
    float3 lin = skyDecode(c);
    uint view = uint(cf.misc.y);
    // Debug view 3: the deep shadow map's ground transmittance over the screen (its three cascades side by side).
    if (view == 3u) lin = float3(cloudsGroundTransmittance(shadow.sample(kCloudClamp, uv, level(0.0)).w));
    out.write(float4(view >= 1u ? saturate(lin) : cloudsAgx(lin * opt.z), 1.0), gid);
}
"""

/// Mirrors CloudFrame in the shaders.
struct CloudFrameGPU {
    var invViewProj = matrix_identity_float4x4
    var prevViewProj = matrix_identity_float4x4
    var camDelta = SIMD4<Float>.zero
    var size = SIMD4<Float>.zero
    var layer = SIMD4<Float>.zero
    var sun = SIMD4<Float>.zero
    var moon = SIMD4<Float>.zero
    var light = SIMD4<Float>.zero
    var noise = SIMD4<Float>.zero
    var shape = SIMD4<Float>.zero
    var cascade0 = SIMD4<Float>.zero
    var cascade1 = SIMD4<Float>.zero
    var cascade2 = SIMD4<Float>.zero
    var misc = SIMD4<Float>.zero
}

/// The planet's radius the sky uses (Sky.swift's atmosphere, km), in metres.
private let cloudPlanetRadius: Double = 6_360_000

final class Clouds: @unchecked Sendable {
    static let shared = Clouds()

    private var library: MTLLibrary?
    private var failed = false
    private var pipes: [String: MTLComputePipelineState] = [:]
    private var compositePipes: [UInt: MTLRenderPipelineState] = [:]
    private var baseNoise: MTLTexture?
    private var detailNoise: MTLTexture?
    private var weather: MTLTexture?
    private var noiseMade = false
    private var shadow: MTLTexture?
    private var env: MTLBuffer?
    private var history: [MTLTexture] = []      // two RGBA16Float, ping-ponged
    private var historyDepth: [MTLTexture] = []
    private var current = 0
    private var visOut: MTLTexture?
    /// The clouds over the upper hemisphere for reflections (cloudsReflected; RGBA16Float, 128 x 128), and whether it holds
    /// a frame yet (the first one is written whole, not blended).
    private var reflection: MTLTexture?
    private var reflectionFrames = 0
    private var dummyColor: MTLTexture?
    private var dummyDepth: MTLTexture?
    private var dummyShadow: MTLTexture?
    /// The march's jitter: a 64 x 64 blue-noise mask (void and cluster), R16Unorm ranks.
    private var blueNoise: MTLTexture?
    /// This frame's parameters (frame()), and whether they and the buffer are valid for this frame's composite (taken by
    /// the anti-aliasing's binding or the composite pass, so a frame that skips frame() can't composite stale clouds).
    private(set) var cf = CloudFrameGPU()
    private var frameReady: (width: Int, height: Int)?
    /// The shadow map was made in the last frame() (for shadowBinding, which the god rays may call after the composite), and
    /// the clouds were (for frameTexture, which the post chain calls after it).
    private var shadowMade = false
    private var madeThisFrame = false
    private var prevViewProj = matrix_identity_float4x4
    private var prevCam = SIMD3<Double>(repeating: .nan)
    /// The deep shadow map's cascades as they were made (their centers), and the light they were made for.
    private var cascadeMade: [SIMD4<Float>] = []
    private var cascadeLight = SIMD4<Float>(repeating: .nan)
    private var prevSize = (0, 0)
    private var lastFrameTime: Double = 0
    private var frames = 0
    private var logFrames = 0
    /// Debug view (METALMC_CLOUDVIEW; mmc_debug_clouds_render offline).
    var view: Float = Float(cloudViewDefault)
    /// Offline: the coverage in place of METALMC_CLOUDCOVER's (nan: the setting's), the wind's time in place of the clock's.
    var coverOverride: Float = .nan
    private var timeOverride: Double?

    /// The library, the pipelines, the noise textures and the small buffers.
    func ensure() -> Bool {
        if failed { return false }
        if library != nil { return true }
        do {
            // Lab mode (ShaderLab.swift): after an edit, forget the pipelines and remake the noise (its kernels may have changed).
            let lib = try ShaderLab.library("clouds", cloudsShaderSource) { [self] _ in
                library = nil; pipes = [:]; compositePipes = [:]; noiseMade = false; failed = false
            }
            for name in ["clouds_noise_base", "clouds_noise_detail", "clouds_noise_weather", "clouds_env", "clouds_shadow",
                         "clouds_march", "clouds_carry", "clouds_march_still", "clouds_vis", "clouds_reflection", "clouds_water_sky",
                         "clouds_debug_view"] {
                guard let f = lib.makeFunction(name: name) else { throw NSError(domain: "clouds", code: 1, userInfo: [NSLocalizedDescriptionKey: "no \(name)"]) }
                pipes[name] = try ctx.device.makeComputePipelineState(function: f)
            }
            library = lib
        } catch {
            log("clouds: shaders failed: \(error)")
            failed = true
            return false
        }
        let dev = ctx.device
        if baseNoise == nil {
            func tex3(_ n: Int, _ f: MTLPixelFormat) -> MTLTexture? {
                let d = MTLTextureDescriptor()
                d.textureType = .type3D
                d.pixelFormat = f
                d.width = n; d.height = n; d.depth = n
                d.mipmapLevelCount = Int(log2(Double(n))) + 1
                d.usage = [.shaderRead, .shaderWrite]
                d.storageMode = .private
                return dev.makeTexture(descriptor: d)
            }
            baseNoise = tex3(128, .r16Unorm)
            detailNoise = tex3(32, .r8Unorm)
            let wd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg8Unorm, width: 1024, height: 1024, mipmapped: true)
            wd.usage = [.shaderRead, .shaderWrite]
            wd.storageMode = .private
            weather = dev.makeTexture(descriptor: wd)
            let sd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 3 * cloudShadowTexels, height: cloudShadowTexels, mipmapped: false)
            sd.usage = [.shaderRead, .shaderWrite]
            sd.storageMode = .private
            shadow = dev.makeTexture(descriptor: sd)
            shadow?.label = "MetalMC cloud deep shadow map"
            env = dev.makeBuffer(length: 4 * 16, options: .storageModePrivate)
            // Stand-ins for the anti-aliasing's resolve when the clouds didn't run this frame (it binds them anyway).
            func one(_ f: MTLPixelFormat, _ bytes: [UInt8]) -> MTLTexture? {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: 1, height: 1, mipmapped: false)
                d.usage = .shaderRead
                d.storageMode = .shared
                let t = dev.makeTexture(descriptor: d)
                bytes.withUnsafeBytes { t?.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: bytes.count) }
                return t
            }
            // RGBA16Float (0, 0, 0, 1), R16Float 0, RGBA16Float 0 (no optical depth).
            dummyColor = one(.rgba16Float, [0, 0, 0, 0, 0, 0, 0x00, 0x3c])
            dummyDepth = one(.r16Float, [0, 0])
            dummyShadow = one(.rgba16Float, [0, 0, 0, 0, 0, 0, 0, 0])
            let ranks = cloudBlueNoise(64)
            let bd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r16Unorm, width: 64, height: 64, mipmapped: false)
            bd.usage = .shaderRead
            bd.storageMode = .shared
            blueNoise = dev.makeTexture(descriptor: bd)
            let scaled = ranks.map { UInt16((UInt32($0) * 65535 + 2047) / 4095) }
            scaled.withUnsafeBytes { blueNoise?.replace(region: MTLRegionMake2D(0, 0, 64, 64), mipmapLevel: 0, withBytes: $0.baseAddress!, bytesPerRow: 128) }
        }
        if baseNoise == nil || detailNoise == nil || weather == nil || shadow == nil || env == nil || dummyColor == nil || blueNoise == nil {
            log("clouds: textures failed")
            failed = true
            return false
        }
        return true
    }

    /// The noise, once (and after a lab-mode edit): the three textures and their mipmaps.
    private func encodeNoise(_ cb: MTLCommandBuffer) {
        guard let base = baseNoise, let detail = detailNoise, let weather, let pb = pipes["clouds_noise_base"],
              let pd = pipes["clouds_noise_detail"], let pw = pipes["clouds_noise_weather"],
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "MetalMC clouds: noise"
        enc.setComputePipelineState(pb)
        enc.setTexture(base, index: 0)
        enc.dispatchThreads(MTLSize(width: 128, height: 128, depth: 128), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
        enc.setComputePipelineState(pd)
        enc.setTexture(detail, index: 0)
        enc.dispatchThreads(MTLSize(width: 32, height: 32, depth: 32), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 4))
        enc.setComputePipelineState(pw)
        enc.setTexture(weather, index: 0)
        enc.dispatchThreads(MTLSize(width: 1024, height: 1024, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        if let blit = cb.makeBlitCommandEncoder() {
            blit.generateMipmaps(for: base)
            blit.generateMipmaps(for: detail)
            blit.generateMipmaps(for: weather)
            blit.endEncoding()
        }
        noiseMade = true
        log("clouds: noise made (base 128^3, detail 32^3, weather 1024^2), layer \(Int(cloudBase))-\(Int(cloudTop)) m above sea level, coverage \(cloudCoverDefault)")
    }

    private func ensureTargets(_ w: Int, _ h: Int) -> Bool {
        if history.count == 2, history[0].width == w, history[0].height == h { return true }
        func tex(_ f: MTLPixelFormat, _ label: String) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            let t = ctx.device.makeTexture(descriptor: d)
            t?.label = label
            return t
        }
        guard let a = tex(.rgba16Float, "MetalMC clouds"), let b = tex(.rgba16Float, "MetalMC clouds"),
              let c = tex(.r16Float, "MetalMC clouds depth"), let d = tex(.r16Float, "MetalMC clouds depth") else { return false }
        history = [a, b]
        historyDepth = [c, d]
        current = 0
        prevCam = SIMD3(repeating: .nan)
        log("clouds: buffer \(w)x\(h) (half the frame each way)")
        return true
    }

    /// The deep shadow map's cascades: centered where the camera's own column meets the layer's middle along the light,
    /// snapped to whole texels in the world so the map doesn't shimmer as the camera moves.
    private func cascades(cam: SIMD3<Double>, light: SIMD3<Float>, height: Double) -> [SIMD4<Float>] {
        let mid = 0.5 * Double(cloudBase + cloudTop)
        let ly = max(Double(light.y), 0.05)
        let s = max(mid - height, 0) / ly
        let cx = Double(light.x) * s, cz = Double(light.z) * s
        return cloudCascadeHalf.map { half in
            let texel = 2 * half / Double(cloudShadowTexels)
            let wx = ((cam.x + cx) / texel).rounded() * texel - cam.x, wz = ((cam.z + cz) / texel).rounded() * texel - cam.z
            return SIMD4(Float(wx), Float(wz), Float(half), 1)
        }
    }

    /// Before the shadows (GameRendererLodMixin): this frame's clouds, after the level's main pass (the march reads its depth
    /// to skip texels whose pixels are all terrain in front of the clouds). proj: the level's projection unjittered (the
    /// anti-aliasing's), viewRot: its view rotation. Returns false if the clouds aren't on this frame.
    func frame(color: MTLTexture, depth: MTLTexture, proj: simd_float4x4, view viewRot: simd_float4x4, cam: SIMD3<Double>,
               sunAngle: Float, rainBrightness: Float, thunder: Float, seaLevel: Double) -> Bool {
        frameReady = nil
        shadowMade = false
        madeThisFrame = false
        let sky = Sky.shared
        guard cloudsEnabled, ctx.pass == nil, sky.ready, ensure(), let env, let shadow, let base = baseNoise, let detail = detailNoise,
              let weather, let trans = sky.transmittance, let apScatter = sky.apScatter, let apTrans = sky.apTrans,
              let skyView = sky.skyView, let pEnv = pipes["clouds_env"], let pShadow = pipes["clouds_shadow"],
              let pMarch = pipes["clouds_march"], let pCarry = pipes["clouds_carry"], let pStill = pipes["clouds_march_still"],
              let pRefl = pipes["clouds_reflection"] else { return false }
        let w = color.width, h = color.height, hw = (w + 1) / 2, hh = (h + 1) / 2
        guard ensureTargets(hw, hh) else { return false }
        frames += 1
        let now = ProcessInfo.processInfo.systemUptime
        let time = timeOverride ?? cloudTimeFixed ?? now
        let height = cam.y - seaLevel
        let viewProj = proj * viewRot
        var f = CloudFrameGPU()
        f.invViewProj = viewProj.inverse
        f.prevViewProj = prevViewProj
        let delta = cam - prevCam
        let valid = prevSize == (w, h) && now - lastFrameTime < 0.25 && simd_length(delta) < 256
        f.camDelta = SIMD4(Float(valid ? delta.x : 0), Float(valid ? delta.y : 0), Float(valid ? delta.z : 0), valid ? 1 : 0)
        f.size = SIMD4(Float(w), Float(h), Float(hw), Float(hh))
        f.layer = SIMD4(cloudBase, cloudTop, Float(height), Float(1 / (2 * cloudPlanetRadius)))
        // Vanilla's sun: (-sin a, cos a, 0); the moon opposite. The clouds' light (their shading and shadows) is the sun while
        // it still lights the layer (a little under the ground's horizon), else the moon.
        let sun = SIMD3<Float>(-sin(sunAngle), cos(sunAngle), 0)
        let night = smoothstepF(0.05, -0.2, sun.y)
        f.sun = SIMD4(sun, sky.frame.sun.w)
        f.moon = SIMD4(-sun, cloudMoonIllum * night)
        f.light = sun.y > -0.06 ? f.sun : f.moon
        // The wind (from the west-southwest), and the shapes evolving: the 3D noise drifts a little across it.
        let windDir = SIMD2<Double>(0.94, 0.34)
        let drift = windDir * Double(cloudWindSpeed) * time
        let evolve = SIMD2<Double>(-windDir.y, windDir.x) * Double(cloudWindSpeed) * 0.15 * time
        func wrap(_ x: Double) -> Float { Float(x - (x / cloudWeatherTile).rounded(.down) * cloudWeatherTile) }
        f.noise = SIMD4(wrap(cam.x - drift.x), wrap(cam.z - drift.y), wrap(cam.x - drift.x - evolve.x), wrap(cam.z - drift.y - evolve.y))
        let rain = min(max(1 - rainBrightness, 0), 1)
        let cover = coverOverride.isNaN ? cloudCoverDefault : coverOverride
        // Rain takes the coverage to a closed deck and thickens it; thunder (which comes with rain) more, and darker.
        let storm = min(max(thunder, 0), 1)
        f.shape = SIMD4(cover + (1.6 - cover) * rain + 0.3 * storm, 1 + 0.4 * rain + 0.8 * storm, rain, storm)
        let cs = cascades(cam: cam, light: SIMD3(f.light.x, f.light.y, f.light.z), height: height)
        let lit = f.light.w > 0 && f.light.y > 0
        // One cascade a frame: the near one every other frame, the middle and far ones every fourth (the clouds drift a few
        // centimetres a frame, the near cascade's texels are 64 m); all three when the light changed (moon and sun) or
        // there's nothing made yet. Each keeps the center it was made with. (The near one every frame plus one of the others
        // took 0.16-0.36 ms in game.)
        let lightKey = SIMD4(f.light.x, f.light.y, f.light.z, f.light == f.sun ? 1 : 0)
        var renderCascades: [UInt32] = [0]
        let lightMoved = lightKey.w != cascadeLight.w || !(simd_dot(SIMD3(lightKey.x, lightKey.y, lightKey.z),
                                                                      SIMD3(cascadeLight.x, cascadeLight.y, cascadeLight.z)) > 0.9999)
        if !lit || cascadeMade.count != 3 || lightMoved {
            renderCascades = [0, 1, 2]
            cascadeMade = cs
            cascadeLight = lightKey
        } else {
            let c = frames % 2 == 0 ? 0 : (frames % 4 == 1 ? 1 : 2)
            renderCascades = [UInt32(c)]
            cascadeMade[c] = cs[c]
        }
        f.cascade0 = lit ? cascadeMade[0] : .zero
        f.cascade1 = lit ? cascadeMade[1] : .zero
        f.cascade2 = lit ? cascadeMade[2] : .zero
        f.misc = SIMD4(Float(frames % 4096), view, 0.3, 1)
        var skyFrame = sky.frame
        var cb = ctx.ensureCB()
        ctx.endBlit()
        if !noiseMade { encodeNoise(cb) }
        guard let enc = cb.makeComputeCommandEncoder(descriptor: profComputePass("clouds: sky light, shadow map")) else { return false }
        enc.label = "MetalMC clouds: sky light, shadow map"
        enc.setComputePipelineState(pEnv)
        enc.setBuffer(env, offset: 0, index: 0)
        enc.setBytes(&skyFrame, length: MemoryLayout<SkyFrameGPU>.stride, index: 1)
        enc.setTexture(trans, index: 0)
        enc.setTexture(skyView, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        if lit {
            enc.setComputePipelineState(pShadow)
            enc.setTexture(shadow, index: 0)
            enc.setTexture(base, index: 1)
            enc.setTexture(detail, index: 2)
            enc.setTexture(weather, index: 3)
            enc.setBytes(&f, length: MemoryLayout<CloudFrameGPU>.stride, index: 0)
            var sel = SIMD4<UInt32>(renderCascades[0], renderCascades.count > 1 ? renderCascades[1] : 0,
                                    renderCascades.count > 2 ? renderCascades[2] : 0, UInt32(renderCascades.count))
            enc.setBytes(&sel, length: 16, index: 1)
            enc.dispatchThreads(MTLSize(width: renderCascades.count * cloudShadowTexels, height: cloudShadowTexels, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        }
        enc.endEncoding()
        if debugSplit {
            // Offline timing: the sky light and shadow map on a command buffer of their own.
            ctx.cb = nil
            cb.commit()
            cb.waitUntilCompleted()
            debugStageMs[0] = (cb.gpuEndTime - cb.gpuStartTime) * 1000
            cb = ctx.ensureCB()
        }
        // Nothing moved since last frame (the camera, its view, the buffer): the march updates the history in place.
        let still = valid && viewProj == prevViewProj && delta == .zero
        let next = still ? current : 1 - current
        f.misc.z = cloudHistoryWeight
        guard let menc = cb.makeComputeCommandEncoder(descriptor: profComputePass("clouds: march")) else { return false }
        menc.label = "MetalMC clouds: march"
        menc.setComputePipelineState(still ? pStill : pMarch)
        menc.setTexture(history[next], index: 0)
        menc.setTexture(historyDepth[next], index: 1)
        if !still {
            menc.setTexture(history[current], index: 2)
            menc.setTexture(historyDepth[current], index: 3)
        }
        menc.setTexture(depth, index: 4)
        menc.setTexture(base, index: 5)
        menc.setTexture(detail, index: 6)
        menc.setTexture(weather, index: 7)
        menc.setTexture(trans, index: 8)
        menc.setTexture(apScatter, index: 9)
        menc.setTexture(apTrans, index: 10)
        menc.setTexture(shadow, index: 11)
        menc.setTexture(blueNoise, index: 12)
        menc.setBytes(&f, length: MemoryLayout<CloudFrameGPU>.stride, index: 0)
        menc.setBytes(&skyFrame, length: MemoryLayout<SkyFrameGPU>.stride, index: 1)
        menc.setBuffer(env, offset: 0, index: 2)
        // This frame's texel of each block (CLOUD_UPDATE: 2 x 2 or 4 x 2 blocks, one thread each), then every other texel's
        // history carried over: they write different texels.
        let bx = cloudUpdate > 4 ? 4 : 2
        menc.dispatchThreads(MTLSize(width: (hw + bx - 1) / bx, height: (hh + 1) / 2, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        if !still {
            menc.setComputePipelineState(pCarry)
            menc.dispatchThreads(MTLSize(width: hw, height: hh, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        }
        // The reflection map (the water's): every texel each frame, blended (a fifth new; the first frame whole).
        if reflection == nil {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 128, height: 128, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            reflection = ctx.device.makeTexture(descriptor: d)
            reflection?.label = "MetalMC clouds for reflections"
            reflectionFrames = 0
        }
        if let reflection {
            var blend = SIMD4<Float>(reflectionFrames == 0 ? 1 : 0.2, 0, 0, 0)
            menc.setComputePipelineState(pRefl)
            menc.setTexture(reflection, index: 0)
            menc.setBytes(&blend, length: 16, index: 3)
            menc.dispatchThreads(MTLSize(width: 128, height: 128, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
            reflectionFrames += 1
        }
        menc.endEncoding()
        current = next
        cf = f
        frameReady = (w, h)
        shadowMade = lit
        madeThisFrame = true
        prevViewProj = viewProj
        prevCam = cam
        prevSize = (w, h)
        lastFrameTime = now
        logFrames += 1
        if logFrames % 1200 == 0 {
            log(String(format: "clouds: %d frames; coverage %.2f, rain %.2f, camera %.0f m above sea level, light %@ %.2f", logFrames,
                       f.shape.x, rain, height, f.light == f.sun ? "sun" : "moon", f.light.w))
        }
        return true
    }

    /// A new sample's weight in the history (each texel is marched every 4th frame).
    var cloudHistoryWeight: Float = 0.3
    /// Offline timing: the sky light and shadow map committed apart from the march (debugRender's times).
    private var debugSplit = false
    private var debugStageMs: [Double] = [0, 0]

    /// For the anti-aliasing's resolve (Taa.swift, its sky variant): buffer 5 and textures 21 and 22 (18-20 are water's in
    /// the lit variant), this frame's clouds or stand-ins that turn the composite off.
    func bindTaa(_ enc: MTLComputeCommandEncoder, width: Int, height: Int) {
        var f = cf
        defer { frameReady = nil }
        if let r = frameReady, r.width == width, r.height == height, history.count == 2 {
            enc.setTexture(history[current], index: 21)
            enc.setTexture(historyDepth[current], index: 22)
        } else {
            f.misc.w = 0
            _ = ensure()
            enc.setTexture(dummyColor, index: 21)
            enc.setTexture(dummyDepth, index: 22)
        }
        enc.setBytes(&f, length: MemoryLayout<CloudFrameGPU>.stride, index: 5)
    }

    /// The clouds over the upper hemisphere, for passes that call cloudsReflected (the water's reflections): the map, or a
    /// 1 x 1 stand-in (no clouds: transmittance 1) when there's none yet.
    func reflectionBinding() -> MTLTexture? {
        guard cloudsEnabled, ensure() else { return nil }
        return shadowMade || reflectionFrames > 0 ? (reflection ?? dummyColor) : dummyColor
    }

    /// The water's sky map (Lit.swift, made this frame in `cb`): the clouds' reflection map blended in (sky x a + rgb), so the
    /// water's reflections that escape to the sky show the clouds. Nothing when the clouds didn't run this frame.
    func intoWaterSky(cb: MTLCommandBuffer, sky: MTLTexture) {
        guard cloudsEnabled, madeThisFrame, reflectionFrames > 0, let reflection, let p = pipes["clouds_water_sky"],
              let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "MetalMC clouds into the water's sky map"
        enc.setComputePipelineState(p)
        enc.setTexture(sky, index: 0)
        enc.setTexture(reflection, index: 1)
        enc.dispatchThreads(MTLSize(width: sky.width, height: sky.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
    }

    /// The clouds as the composite took them this frame (RGBA16Float at half the resolution each way: rgb their light, a their
    /// transmittance), for the post chain's light shafts (Post.swift): a 1 x 1 stand-in (no clouds) when they didn't run.
    func frameTexture() -> MTLTexture? {
        guard cloudsEnabled, ensure() else { return nil }
        return madeThisFrame && history.count == 2 ? history[current] : dummyColor
    }

    /// The deep shadow map and this frame's CloudFrame, for passes that call cloudTransmittanceToSun (the god rays): the
    /// texture, or a 1 x 1 stand-in (with misc.w 0) when the clouds didn't run this frame.
    func shadowBinding() -> (shadow: MTLTexture, frame: CloudFrameGPU)? {
        guard cloudsEnabled, ensure(), let dummyShadow else { return nil }
        if shadowMade, let shadow { return (shadow, cf) }
        var f = cf
        f.misc.w = 0
        return (dummyShadow, f)
    }

    /// Lit mode's cloud shadows (RtShadows.trace, after its rays): RtShadows' visibility `vis` (one texel per scale x scale
    /// pixels, traced at `sample`'s pixel of each block this frame) times the clouds' transmittance toward the light there,
    /// into a texture of the same size, which the relight then takes instead. towardMoon: the rays went toward the moon. Nil
    /// if the clouds aren't on this frame (the relight takes `vis` as it is).
    func shadeVisibility(cb: MTLCommandBuffer, depth: MTLTexture, vis: MTLTexture, invViewProj: simd_float4x4, towardMoon: Bool,
                         scale: Int, sample: SIMD2<UInt32>, width: Int, height: Int) -> MTLTexture? {
        guard let r = frameReady, r.width == width, r.height == height, let shadow, let p = pipes["clouds_vis"] else { return nil }
        // The shadow map was made toward the clouds' light; the rays toward the moon only match it at night.
        let mapMoon = cf.light != cf.sun
        let matches = mapMoon == towardMoon && shadowMade
        if visOut == nil || visOut!.width != vis.width || visOut!.height != vis.height {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: vis.width, height: vis.height, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            visOut = ctx.device.makeTexture(descriptor: d)
            visOut?.label = "MetalMC visibility with cloud shadows"
        }
        guard let out = visOut, let enc = cb.makeComputeCommandEncoder(descriptor: profComputePass("clouds: shadows on the visibility")) else { return nil }
        var f = cf
        var inv = invViewProj
        var sp = SIMD4<UInt32>(UInt32(scale), sample.x, sample.y, matches ? 1 : 0)
        enc.label = "MetalMC cloud shadows"
        enc.setComputePipelineState(p)
        enc.setTexture(vis, index: 0)
        enc.setTexture(out, index: 1)
        enc.setTexture(depth, index: 2)
        enc.setTexture(shadow, index: 3)
        enc.setBytes(&f, length: MemoryLayout<CloudFrameGPU>.stride, index: 0)
        enc.setBytes(&inv, length: 64, index: 1)
        enc.setBytes(&sp, length: 16, index: 2)
        enc.dispatchThreads(MTLSize(width: vis.width, height: vis.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        return out
    }

    /// Without anti-aliasing (the sky's aerial perspective ran as a pass of its own): the clouds over the frame, a pass of
    /// its own.
    func composite(color: MTLTexture, depth: MTLTexture, invViewProj: simd_float4x4) -> Bool {
        guard let r = frameReady, r.width == color.width, r.height == color.height, ctx.pass == nil, let lib = library,
              history.count == 2 else { return false }
        frameReady = nil
        if compositePipes[color.pixelFormat.rawValue] == nil {
            let d = MTLRenderPipelineDescriptor()
            d.label = "MetalMC clouds composite"
            d.vertexFunction = lib.makeFunction(name: "clouds_fullscreen_vs")
            d.fragmentFunction = lib.makeFunction(name: "clouds_composite_fs")
            d.colorAttachments[0].pixelFormat = color.pixelFormat
            do { compositePipes[color.pixelFormat.rawValue] = try ctx.device.makeRenderPipelineState(descriptor: d) } catch {
                log("clouds: composite pipeline failed: \(error)")
                return false
            }
        }
        guard let pipe = compositePipes[color.pixelFormat.rawValue] else { return false }
        var sky = Sky.shared.frame
        sky.invViewProj = invViewProj
        sky.view.z = Float(color.width)
        sky.view.w = Float(color.height)
        // The tone curve the frame went through (as Sky.levelFrame sets it): with post none, else the SDR shoulder or HDR's.
        let isFloat = isFloatFrame(color.pixelFormat)
        let hr: Float = isFloat ? max(Hdr.shared.headroom, 1) : 1
        sky.tone.x = hr
        sky.tone.y = skyKnee(hr)
        if postEnabled && isFloat { sky.tone.x = 65504; sky.tone.y = 65504 }
        var f = cf
        ctx.endBlit()
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = color
        d.colorAttachments[0].loadAction = .load
        d.colorAttachments[0].storeAction = .store
        profAttach(d, "clouds composite \(color.width)x\(color.height)")
        guard let enc = ctx.ensureCB().makeRenderCommandEncoder(descriptor: d) else { return false }
        enc.label = "MetalMC clouds composite"
        enc.setRenderPipelineState(pipe)
        enc.setFragmentTexture(depth, index: 0)
        enc.setFragmentTexture(history[current], index: 1)
        enc.setFragmentTexture(historyDepth[current], index: 2)
        enc.setFragmentBytes(&sky, length: MemoryLayout<SkyFrameGPU>.stride, index: 0)
        enc.setFragmentBytes(&f, length: MemoryLayout<CloudFrameGPU>.stride, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        ctx.statPasses += 1
        return true
    }

    /// Offline (mmc_debug_clouds_render): the clouds as the game makes them, `frames` frames one after another (the history
    /// accumulating, the camera still), then the debug view (the sky, optionally a flat ground at sea level with the clouds'
    /// shadows, the clouds composited, AgX) into `out` (w x h RGBA8, top row first). Each frame on its own command buffer.
    /// times[0]: the GPU milliseconds of the clouds' frame (the last eight frames' mean), [1] the view's, [2] the sky light
    /// and shadow map's, [3] the march's.
    func debugRender(sunAngle: Float, rainBrightness: Float, cam: SIMD3<Double>, yaw: Float, pitch: Float, fovY: Float,
                     time: Double, frames n: Int, ground: Bool, exposure: Float, width: Int, height: Int,
                     out: UnsafeMutablePointer<UInt8>, times: UnsafeMutablePointer<Double>) -> Bool {
        let seaLevel = 63.0
        // The sky's tables for this sun and camera.
        let cb0 = ctx.queue.makeCommandBuffer()!
        Sky.shared.prepare(cb: cb0, sunAngle: sunAngle, rainBrightness: rainBrightness, height: Float(cam.y - seaLevel),
                           fadeStart: 0, fadeEnd: 0, headroom: 1)
        cb0.commit()
        cb0.waitUntilCompleted()
        guard Sky.shared.ready, ensure(), let pView = pipes["clouds_debug_view"], let shadow else { return false }
        // The camera: Minecraft's yaw and pitch, an infinite reverse-Z projection like the game's (near 0.05).
        let fwd = SIMD3<Float>(-sin(yaw) * cos(pitch), -sin(pitch), cos(yaw) * cos(pitch))
        let right = simd_normalize(simd_cross(fwd, SIMD3(0, 1, 0))), up = simd_cross(right, fwd)
        let viewM = simd_float4x4(rows: [SIMD4(right, 0), SIMD4(up, 0), SIMD4(-fwd, 0), SIMD4(0, 0, 0, 1)])
        let fl = 1 / tan(fovY / 2), aspect = Float(width) / Float(height)
        let proj = simd_float4x4(SIMD4(fl / aspect, 0, 0, 0), SIMD4(0, fl, 0, 0), SIMD4(0, 0, 0, -1), SIMD4(0, 0, 0.05, 0))
        func tex(_ f: MTLPixelFormat, _ u: MTLTextureUsage, _ s: MTLStorageMode = .private) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: width, height: height, mipmapped: false)
            d.usage = u
            d.storageMode = s
            return ctx.device.makeTexture(descriptor: d)!
        }
        let color = tex(.rgba8Unorm, [.renderTarget, .shaderRead])
        let depth = tex(.depth32Float, [.renderTarget, .shaderRead])
        // The depth: all sky (cleared to 0), so every texel is marched (the debug view draws its ground itself).
        let cbd = ctx.queue.makeCommandBuffer()!
        let pd = MTLRenderPassDescriptor()
        pd.depthAttachment.texture = depth
        pd.depthAttachment.loadAction = .clear
        pd.depthAttachment.clearDepth = 0
        pd.depthAttachment.storeAction = .store
        cbd.makeRenderCommandEncoder(descriptor: pd)!.endEncoding()
        cbd.commit()
        prevCam = SIMD3(repeating: .nan)
        timeOverride = time
        defer { timeOverride = nil }
        var last = 0.0, lastShadow = 0.0
        // The frames accumulate; the last eight (each texel marched once in them) are timed, the sky light and shadow map
        // apart from the march.
        for k in 0..<max(n, 1) {
            lastFrameTime = ProcessInfo.processInfo.systemUptime
            debugSplit = k >= n - 8
            defer { debugSplit = false }
            guard frame(color: color, depth: depth, proj: proj, view: viewM, cam: cam, sunAngle: sunAngle, rainBrightness: rainBrightness,
                        thunder: 0, seaLevel: seaLevel), let cb = ctx.cb else { return false }
            ctx.cb = nil
            cb.commit()
            cb.waitUntilCompleted()
            if k >= n - 8 {
                last += (cb.gpuEndTime - cb.gpuStartTime) * 1000 / Double(min(n, 8))
                lastShadow += debugStageMs[0] / Double(min(n, 8))
            }
        }
        times[0] = last + lastShadow
        times[2] = lastShadow
        times[3] = last
        let outTex = tex(.rgba8Unorm, [.shaderWrite], .shared)
        var sky = Sky.shared.frame
        sky.invViewProj = (proj * viewM).inverse
        sky.view.z = Float(width)
        sky.view.w = Float(height)
        sky.tone.x = 65504
        sky.tone.y = 65504
        var f = cf
        var opt = SIMD4<Float>(Float(cam.y - seaLevel), ground ? 1 : 0, exposure, 0)
        let cb = ctx.queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pView)
        enc.setTexture(outTex, index: 0)
        enc.setTexture(history[current], index: 1)
        enc.setTexture(historyDepth[current], index: 2)
        enc.setTexture(Sky.shared.skyView, index: 3)
        enc.setTexture(Sky.shared.transmittance, index: 4)
        enc.setTexture(Sky.shared.apScatter, index: 5)
        enc.setTexture(Sky.shared.apTrans, index: 6)
        enc.setTexture(shadow, index: 7)
        enc.setBytes(&sky, length: MemoryLayout<SkyFrameGPU>.stride, index: 0)
        enc.setBytes(&f, length: MemoryLayout<CloudFrameGPU>.stride, index: 1)
        enc.setBytes(&opt, length: 16, index: 2)
        enc.dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        times[1] = (cb.gpuEndTime - cb.gpuStartTime) * 1000
        // Row 0 of the texture is the bottom of clip space: flip to top-first.
        let row = width * 4
        var tmp = [UInt8](repeating: 0, count: row * height)
        outTex.getBytes(&tmp, bytesPerRow: row, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        for y in 0..<height {
            tmp.withUnsafeBufferPointer { src in
                (out + (height - 1 - y) * row).update(from: src.baseAddress! + y * row, count: row)
            }
        }
        frameReady = nil
        return true
    }
}

/// A blue-noise dither mask (Ulichney's void and cluster, 1993) on an n x n torus: every pixel's rank, 0 ..< n^2. A
/// Gaussian energy (sigma 1.5) marks clusters and voids; an initial tenth of the pixels is relaxed (the tightest cluster's
/// point moved to the largest void until it stays), ranked downward as it's taken out tightest first, then the voids are
/// filled largest first, ranked upward. 64 x 64 takes about 20 ms (once); its 3 x 3 averages vary a fifth as much as white
/// noise's.
func cloudBlueNoise(_ n: Int, sigma: Double = 1.5) -> [UInt16] {
    let count = n * n
    let r = Int((sigma * 4).rounded(.up))
    let kw = 2 * r + 1
    var kernel = [Double](repeating: 0, count: kw * kw)
    for dy in -r...r { for dx in -r...r { kernel[(dy + r) * kw + dx + r] = exp(-Double(dx * dx + dy * dy) / (2 * sigma * sigma)) } }
    var energy = [Double](repeating: 0, count: count)
    var on = [Bool](repeating: false, count: count)
    func splat(_ i: Int, _ sign: Double) {
        let x = i % n, y = i / n
        for dy in -r...r {
            let yy = (y + dy + n) % n
            for dx in -r...r { energy[yy * n + (x + dx + n) % n] += sign * kernel[(dy + r) * kw + dx + r] }
        }
    }
    var s: UInt64 = 0x51ed270b
    func rnd() -> UInt64 { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s }
    let initial = count / 10
    var placed = 0
    while placed < initial {
        let i = Int(rnd() % UInt64(count))
        if !on[i] { on[i] = true; splat(i, 1); placed += 1 }
    }
    func tightest() -> Int { var b = 0, e = -Double.infinity; for i in 0..<count where on[i] && energy[i] > e { e = energy[i]; b = i }; return b }
    func largestVoid() -> Int { var b = 0, e = Double.infinity; for i in 0..<count where !on[i] && energy[i] < e { e = energy[i]; b = i }; return b }
    while true {
        let c = tightest()
        on[c] = false; splat(c, -1)
        let v = largestVoid()
        on[v] = true; splat(v, 1)
        if v == c { break }
    }
    var rank = [UInt16](repeating: 0, count: count)
    let keepOn = on, keepEnergy = energy
    var k = initial
    while k > 0 {
        let c = tightest()
        on[c] = false; splat(c, -1)
        k -= 1
        rank[c] = UInt16(k)
    }
    on = keepOn
    energy = keepEnergy
    k = initial
    while k < count {
        let v = largestVoid()
        on[v] = true; splat(v, 1)
        rank[v] = UInt16(k)
        k += 1
    }
    return rank
}

private func smoothstepF(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
    let t = min(max((x - e0) / (e1 - e0), 0), 1)
    return t * t * (3 - 2 * t)
}

private func cloudMatrix(_ p: UnsafePointer<Float>, _ o: Int) -> simd_float4x4 {
    simd_float4x4(SIMD4(p[o], p[o + 1], p[o + 2], p[o + 3]), SIMD4(p[o + 4], p[o + 5], p[o + 6], p[o + 7]),
                  SIMD4(p[o + 8], p[o + 9], p[o + 10], p[o + 11]), SIMD4(p[o + 12], p[o + 13], p[o + 14], p[o + 15]))
}

/// 1 if the clouds are on (METALMC_EXP=clouds with sky) and their shaders compile; the Java side then turns vanilla's off.
@_cdecl("mmc_clouds_enabled")
public func mmc_clouds_enabled() -> Int32 {
    guard cloudsEnabled else { return 0 }
    return Clouds.shared.ensure() ? 1 : 0
}

/// This frame's clouds (GameRendererLodMixin, after the level, before the shadows). color, depth: the level's TextureBox
/// handles. p: the projection unjittered (the anti-aliasing's; as drawn without it) [16], the view rotation [16], vanilla's
/// sun angle, rain brightness (1 clear), thunder level (0-1), sea level (y). cam: the camera's world position. Returns 1 if
/// the clouds are on this frame.
@_cdecl("mmc_clouds_frame")
public func mmc_clouds_frame(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>, _ cam: UnsafePointer<Double>) -> Int32 {
    guard cloudsEnabled else { return 0 }
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    return Clouds.shared.frame(color: color, depth: depth, proj: cloudMatrix(p, 0), view: cloudMatrix(p, 16),
                               cam: SIMD3(cam[0], cam[1], cam[2]), sunAngle: p[32], rainBrightness: p[33], thunder: p[34],
                               seaLevel: Double(p[35])) ? 1 : 0
}

/// Without anti-aliasing: the clouds over the level in `color` (after the sky's aerial perspective pass). p: the projection
/// as drawn and the view rotation (32 floats). Returns 1 if drawn.
@_cdecl("mmc_clouds_composite")
public func mmc_clouds_composite(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>) -> Int32 {
    guard cloudsEnabled else { return 0 }
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    return Clouds.shared.composite(color: color, depth: depth, invViewProj: (cloudMatrix(p, 0) * cloudMatrix(p, 16)).inverse) ? 1 : 0
}

/// Offline: see Clouds.debugRender. p: sun angle, rain brightness, camera x, y, z, yaw, pitch (degrees, Minecraft's),
/// vertical field of view (degrees), wind time (s), frames, ground (0/1), exposure, coverage (nan: the setting's), debug view.
/// out: w * h * 4 bytes (top row first); times: 4 doubles (the clouds' frame, the view, the sky light and shadow map, the
/// march; GPU ms, the last eight frames' mean).
@_cdecl("mmc_debug_clouds_render")
public func mmc_debug_clouds_render(_ p: UnsafePointer<Double>, _ w: Int32, _ h: Int32, _ out: UnsafeMutablePointer<UInt8>,
                                    _ times: UnsafeMutablePointer<Double>) -> Int32 {
    guard cloudsEnabled else { return 0 }
    let c = Clouds.shared
    c.coverOverride = Float(p[12])
    c.view = Float(p[13])
    defer { c.coverOverride = .nan; c.view = Float(cloudViewDefault) }
    let deg = Float.pi / 180
    return c.debugRender(sunAngle: Float(p[0]), rainBrightness: Float(p[1]), cam: SIMD3(p[2], p[3], p[4]), yaw: Float(p[5]) * deg,
                         pitch: Float(p[6]) * deg, fovY: Float(p[7]) * deg, time: p[8], frames: Int(p[9]), ground: p[10] > 0.5,
                         exposure: Float(p[11]), width: Int(w), height: Int(h), out: out, times: times) ? 1 : 0
}

/// Debug: the clouds' shader source (for an offline compile check). Returns its length.
@_cdecl("mmc_debug_clouds_shader_source")
public func mmc_debug_clouds_shader_source(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let bytes = Array(cloudsShaderSource.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}
