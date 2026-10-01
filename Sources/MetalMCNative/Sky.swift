import Foundation
import Metal
import simd

// Sky and atmosphere (METALMC_EXP=sky, lighting milestone 1, off by default). A physically based sky after Hillaire, "A
// Scalable and Production Ready Sky and Atmosphere Rendering Technique" (EGSR 2020): Rayleigh and Mie scattering and
// ozone absorption over an Earth-sized planet, precomputed on the GPU into four small tables:
// - transmittance (256 x 64): how much sunlight gets from a point to space, by altitude and sun angle;
// - multiple scattering (32 x 32): light scattered more than once, as an even glow per altitude and sun angle
//   (Hillaire's approximation: the second order's integral, summed as a geometric series over all orders);
// - sky view (192 x 108): the sky's color in every direction from the camera, rows dense near the horizon;
// - aerial perspective (32 x 64 x 32): along every view direction out to 400 km, the light the air scatters toward
//   the camera and how much of what's behind gets through: distance haze in the sun's color, for terrain.
// The world is flat and the planet is round: a camera-relative position p (blocks, which are metres) is the point
// (0, R + altitude, 0) + p / 1000 km from the planet's center, so terrain at the horizon fades into exactly the sky
// behind it (both are the same rays through the same air).
//
// Per frame (GameRenderer.renderLevel, SkyGameRendererMixin): the sky view and aerial perspective tables are rebuilt
// when the sun, the camera's altitude or the rain changed, the other two when the rain did. Before vanilla's sky pass
// the sky is worked out at a quarter of the resolution on each axis, and in the pass it's filtered up in place of
// vanilla's sky disc, sunrise fan and sun, with the sun's disk worked out per pixel (the moon and stars stay
// vanilla's). After the level, every pixel that isn't sky gets aerial perspective from the depth buffer (vanilla's
// chunks, the LOD, the far field, clouds and entities alike), the render distance's edge fades into the sky's own
// color (vanilla's fog is turned off while the sky is on), and the tone curve (Hdr.swift) makes it display light:
// in the anti-aliasing's resolve as it loads each pixel, or in a pass of its own without anti-aliasing.
//
// Units: scene-linear light where vanilla's white is 1 (vanilla's colors, decoded from sRGB). The sun's illuminance
// in those units is METALMC_SKYEXPOSURE; the tables hold light per unit of it times it.
//
// Other passes can apply the same aerial perspective themselves with `skyShaderHeader` (see docs/lighting-design.md
// for the LOD and far-field call sites): bind the frame's SkyFrame and the two aerial perspective textures and call
// skyApplyAerial on the decoded color.

/// METALMC_EXP=sky: the physically based sky and aerial perspective.
let skyEnabled = experiments.contains("sky")
/// METALMC_SKYEXPOSURE: the sun's illuminance in scene units (vanilla's white is 1). The default puts the noon zenith
/// near vanilla's sky blue with a bright, hazy horizon. (Lit mode, Lit.swift, scales its light by it.)
let skyExposure = max(0, Float(ProcessInfo.processInfo.environment["METALMC_SKYEXPOSURE"] ?? "") ?? 12)
/// Debug (METALMC_EXP=skyhazedebug): the level's pixels as their distance, haze and fade (skyLevelColor), the sky magenta.
let skyDebugHaze = experiments.contains("skyhazedebug")
/// METALMC_SKYHAZE: multiplies distances for aerial perspective (2: twice as hazy; 0.5: clearer). Default 1, physical.
private let skyHaze = max(0, Float(ProcessInfo.processInfo.environment["METALMC_SKYHAZE"] ?? "") ?? 1)
/// METALMC_SUNSIZE: the sun disk's angular radius in degrees. The default 0.6 matches the ray-traced shadows' penumbra
/// (RtShadows); the real sun's is 0.27, vanilla's square about 17.
private let skySunRadius = min(max(Float(ProcessInfo.processInfo.environment["METALMC_SUNSIZE"] ?? "") ?? 0.6, 0.05), 20) * .pi / 180

/// Structures and functions shared by every pass that uses the atmosphere. Other passes prepend this to their source.
let skyShaderHeader = """
#include <metal_stdlib>
using namespace metal;

#define SKY_TRANS_W 256
#define SKY_TRANS_H 64
#define SKY_MS_N 32
#define SKY_VIEW_W 192
#define SKY_VIEW_H 108
#define SKY_AP_W 32
#define SKY_AP_H 64
#define SKY_AP_D 32

constant float SKY_PI = 3.14159265358979;

// Distances in the atmosphere are kilometres; world positions (blocks) are metres.
struct SkyAtmosphere {
    float4 rayleigh;   // rgb: Rayleigh scattering at the ground (1/km), w: its scale height (km)
    float4 mie;        // x: Mie scattering at the ground (1/km), y: Mie extinction (1/km), z: scale height (km), w: asymmetry g
    float4 ozone;      // rgb: ozone absorption at its peak (1/km), w: the peak's altitude (km)
    float4 planet;     // x: ground radius (km), y: radius of the atmosphere's top (km), z: ozone layer half width (km), w: ground albedo
};

struct SkyFrame {
    float4x4 invViewProj;  // clip space to camera-relative world space (the projection and view rotation the level is drawn with)
    SkyAtmosphere atmo;
    float4 sun;            // xyz: direction to the sun (world, y up), w: the sun's illuminance (scene units)
    float4 sunHoriz;       // xy: the sun's horizontal direction (world x, z), normalized; z: disk angular radius (rad); w: disk brightness (0 in rain)
    float4 view;           // x: camera altitude above the ground (km), y: its distance from the planet's center (km), zw: target size (pixels)
    float4 tone;           // x: display headroom (1: SDR), y: the tone curve's knee, z: dither amplitude, w: the night sky's brightness
    float4 aerial;         // x: the aerial perspective volume's range (km), y: distance scale (haze), z: rain (0-1), w: frame (mod 64)
    float4 fade;           // x, y: render-distance fade start and end (blocks, cylindrical like vanilla's fog), z: 1 = on, w: unused
    float4 horizon;        // x: the planet's horizon's zenith angle, y: its depression below the horizontal, z: angle per pixel at the screen's center (rad), w: unused
};

struct SkyAerial {
    float3 inscatter;      // light the air adds between the camera and the point (scene units)
    float3 transmittance;  // how much of the point's own light reaches the camera
};

// Unit range [0, 1] to texture coordinates whose ends are the first and last texel centers.
static float skySubUV(float x, float n) { return (x * (n - 1.0) + 0.5) / n; }

// Distance along the ray o + d t (planet space, km, d normalized) to the sphere of radius R around the planet's
// center: the nearest crossing ahead, or -1. c is computed as (r - R)(r + R), which keeps its precision near the ground.
static float skyRaySphere(float3 o, float3 d, float R) {
    float r = length(o);
    float b = dot(o, d), c = (r - R) * (r + R);
    float disc = b * b - c;
    if (disc < 0.0) return -1.0;
    float s = sqrt(disc);
    float t0 = -b - s, t1 = -b + s;
    return t0 >= 0.0 ? t0 : (t1 >= 0.0 ? t1 : -1.0);
}

struct SkyMedium {
    float3 scattering;
    float3 extinction;
    float3 rayleigh;   // Rayleigh scattering
    float mie;         // Mie scattering
};

// The air at altitude h (km). Below the ground (terrain under sea level) it's the ground's.
static SkyMedium skyMediumAt(constant SkyAtmosphere& a, float h) {
    h = max(h, 0.0);
    float dr = exp(-h / a.rayleigh.w), dm = exp(-h / a.mie.z);
    float dz = max(0.0, 1.0 - abs(h - a.ozone.w) / a.planet.z);
    SkyMedium m;
    m.rayleigh = a.rayleigh.rgb * dr;
    m.mie = a.mie.x * dm;
    m.scattering = m.rayleigh + m.mie;
    m.extinction = m.rayleigh + a.mie.y * dm + a.ozone.rgb * dz;
    return m;
}

static float skyRayleighPhase(float c) { return 3.0 / (16.0 * SKY_PI) * (1.0 + c * c); }

// Cornette-Shanks: Henyey-Greenstein with a Rayleigh-like back lobe, as Hillaire uses.
static float skyMiePhase(float g, float c) {
    float k = 3.0 / (8.0 * SKY_PI) * (1.0 - g * g) / (2.0 + g * g);
    return k * (1.0 + c * c) / pow(max(1.0 + g * g - 2.0 * g * c, 1e-4), 1.5);
}

// The transmittance table's coordinates (Bruneton 2017): the distance to the top of the atmosphere, relative to its
// range at that altitude, and the altitude as the distance to the horizon. For rays that don't hit the ground.
static float2 skyTransmittanceUV(constant SkyAtmosphere& a, float r, float mu) {
    float Rg = a.planet.x, Rt = a.planet.y;
    float H = sqrt(Rt * Rt - Rg * Rg);
    float rho = sqrt(max(0.0, (r - Rg) * (r + Rg)));
    float d = max(0.0, -r * mu + sqrt(max(0.0, (Rt - r) * (Rt + r) + r * r * mu * mu)));
    float dmin = Rt - r, dmax = rho + H;
    float xmu = (d - dmin) / max(dmax - dmin, 1e-6);
    return float2(skySubUV(saturate(xmu), SKY_TRANS_W), skySubUV(saturate(rho / H), SKY_TRANS_H));
}

// Transmittance from radius r (km) to space along a ray whose zenith cosine is mu.
static float3 skyTransmittanceToSpace(texture2d<float> t, constant SkyAtmosphere& a, float r, float mu) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return t.sample(s, skyTransmittanceUV(a, r, mu)).rgb;
}

// How much of the sun's disk is above the planet's horizon from radius r (the terminator, soft over the disk).
static float skySunVisible(constant SkyAtmosphere& a, float r, float mus, float sunRadius) {
    float sinH = min(a.planet.x / r, 1.0), cosH = -sqrt(max(0.0, 1.0 - sinH * sinH));
    float w = sinH * max(sunRadius, 0.004);
    return smoothstep(-w, w, mus - cosH);
}

static float3 skyMultiScattering(texture2d<float> ms, constant SkyAtmosphere& a, float h, float mus) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = float2(skySubUV(saturate(mus * 0.5 + 0.5), SKY_MS_N), skySubUV(saturate(h / (a.planet.y - a.planet.x)), SKY_MS_N));
    return ms.sample(s, uv).rgb;
}

// Light scattered toward the viewer at planet-space point p (km), per unit of the sun's illuminance, and the air
// there. The phase functions are the ray's: the angle to the sun is the same all along a straight ray.
static float3 skySource(constant SkyFrame& f, float3 p, float3 sunDir, float phaseR, float phaseM,
                        texture2d<float> trans, texture2d<float> ms, thread SkyMedium& m) {
    float len = length(p);
    float r = max(len, f.atmo.planet.x + 1e-3);
    float h = r - f.atmo.planet.x;
    m = skyMediumAt(f.atmo, h);
    float mus = dot(p / len, sunDir);
    float3 sun = skyTransmittanceToSpace(trans, f.atmo, r, mus) * skySunVisible(f.atmo, r, mus, f.sunHoriz.z);
    return sun * (m.rayleigh * phaseR + m.mie * phaseM) + skyMultiScattering(ms, f.atmo, h, mus) * m.scattering;
}

// One segment of length dt with source S and extinction e: adds the light it scatters (seen through the transmittance
// T so far) to L, and its transmittance to T. Exact when S and e are constant over the segment (Hillaire).
static void skySegment(float3 S, float3 e, float dt, thread float3& L, thread float3& T) {
    float3 st = exp(-e * dt);
    L += T * (S - S * st) / max(e, float3(1e-7));
    T *= st;
}

// Light scattered toward o along o + dir t, t in [0, tmax] (per unit sun illuminance), in n segments growing
// quadratically (short near the camera, where the air is densest); T gets the transmittance over the ray.
static float3 skyIntegrate(constant SkyFrame& f, float3 o, float3 dir, float3 sun, float tmax, int n,
                           texture2d<float> trans, texture2d<float> ms, thread float3& T) {
    float c = dot(dir, sun);
    float pr = skyRayleighPhase(c), pm = skyMiePhase(f.atmo.mie.w, c);
    float3 L = 0.0;
    T = 1.0;
    float t0 = 0.0;
    for (int i = 0; i < n; i++) {
        float x = float(i + 1) / float(n);
        float t1 = tmax * x * x;
        float dt = t1 - t0;
        SkyMedium m;
        float3 S = skySource(f, o + dir * (t0 + 0.3 * dt), sun, pr, pm, trans, ms, m);
        skySegment(S, m.extinction, dt, L, T);
        t0 = t1;
    }
    return L;
}

// Angle around the vertical between a direction and the sun, as the tables' column coordinate (dense near the sun).
static float skyAzimuthU(constant SkyFrame& f, float3 dir) {
    float2 hd = dir.xz;
    float hl = length(hd);
    float cosPhi = hl > 1e-5 ? dot(hd / hl, f.sunHoriz.xy) : 1.0;
    return sqrt(saturate(0.5 - 0.5 * cosPhi));
}

// The sky view table (after Hillaire's): rows from the zenith down to the planet's horizon, squeezed toward the
// horizon, where the sky changes fastest; columns the angle around the vertical from the sun (0) to opposite it.
// Below the planet's horizon (about 0.3 degrees down from 87 m up) the sky is the horizon's, like vanilla's fog color
// there: in game, terrain covers those directions, and where it ends short of the horizon (a small render distance,
// the world's edge) the physical sky, air down to a black planet, would show as a dark band. f.horizon.x: the
// horizon's zenith angle (per frame).
static float2 skyViewUV(constant SkyFrame& f, float3 dir) {
    float vz = acos(clamp(dir.y, -1.0, 1.0));
    float v = 1.0 - sqrt(saturate(1.0 - vz / f.horizon.x));
    return float2(skySubUV(skyAzimuthU(f, dir), SKY_VIEW_W), skySubUV(v, SKY_VIEW_H));
}

// The aerial perspective volume: columns like the sky view's; rows the sine of the elevation above the camera's
// horizontal plane, squeezed toward it (terrain a few km away and beyond is all within a degree or two of it); slices
// the distance, quadratic out to f.aerial.x km (slice k of SKY_AP_D at (k / SKY_AP_D)^2 of it).
static float2 skyAerialUV(constant SkyFrame& f, float3 dir) {
    float y = clamp(dir.y, -1.0, 1.0);
    float v = 0.5 + 0.5 * sign(y) * sqrt(abs(y));
    return float2(skySubUV(skyAzimuthU(f, dir), SKY_AP_W), skySubUV(v, SKY_AP_H));
}

// Aerial perspective for a camera-relative position (blocks). Interpolated linearly in distance between slices (not
// in the slice coordinate: the sample's depth coordinate puts the hardware's interpolation weight at the distance's
// fraction of the way between them), from nothing at the camera to the first slice.
static SkyAerial skyAerialPerspective(float3 rel, constant SkyFrame& f, texture3d<float> apScatter, texture3d<float> apTrans) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    SkyAerial r;
    float dist = length(rel);
    float d = dist * 0.001 * f.aerial.y;
    if (!(d > 1e-6)) {
        r.inscatter = 0.0;
        r.transmittance = 1.0;
        return r;
    }
    float2 uv = skyAerialUV(f, rel / dist);
    float range = f.aerial.x;
    float w = sqrt(saturate(d / range)) * float(SKY_AP_D);
    float k0 = min(floor(w), float(SKY_AP_D - 1)), k1 = k0 + 1.0;
    float t0 = range * (k0 / SKY_AP_D) * (k0 / SKY_AP_D), t1 = range * (k1 / SKY_AP_D) * (k1 / SKY_AP_D);
    float a = saturate((d - t0) / (t1 - t0));
    // Slice k is at depth coordinate (k - 0.5) / SKY_AP_D. In front of the first one, blend from nothing to it.
    float z = (k0 > 0.0 ? k0 - 0.5 + a : 0.5) / SKY_AP_D;
    float blend = k0 > 0.0 ? 1.0 : a;
    r.inscatter = apScatter.sample(s, float3(uv, z)).rgb * blend;
    r.transmittance = mix(float3(1.0), apTrans.sample(s, float3(uv, z)).rgb, blend);
    return r;
}

// In rain the sky goes toward an even gray: under the clouds the light comes from everywhere, not from the sun.
static float3 skyOvercast(constant SkyFrame& f, float3 L) {
    return mix(L, float3(dot(L, float3(0.2126, 0.7152, 0.0722))), 0.75 * f.aerial.z);
}

// The night sky's faint glow (moonlight and airglow, standing in for the moon's own scattering), brighter low down.
static float3 skyNight(constant SkyFrame& f, float3 dir) {
    float n = f.tone.w;
    if (n <= 0.0) return 0.0;
    float low = 1.0 - saturate(dir.y);
    return n * float3(0.55, 0.7, 1.0) * (0.5 + 1.0 * low * low);
}

// The sky's light in a direction, without the sun's disk (scene units).
static float3 skyLuminance(constant SkyFrame& f, float3 dir, texture2d<float> skyView) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return skyOvercast(f, skyView.sample(s, skyViewUV(f, dir)).rgb) + skyNight(f, dir);
}

// Aerial perspective on a surface's light (linear, scene units) at camera-relative rel: the sunlit haze in front of it
// (overcast like the sky in rain), and at night the night sky's glow in place of the haze. haze gets how much of the
// surface's light the air took (0 up close).
static float3 skyApplyAerial(float3 lin, float3 rel, constant SkyFrame& f, texture3d<float> apScatter, texture3d<float> apTrans,
                             thread float& haze) {
    SkyAerial a = skyAerialPerspective(rel, f, apScatter, apTrans);
    float3 dir = rel / max(length(rel), 1e-6);
    haze = 1.0 - min(a.transmittance.r, min(a.transmittance.g, a.transmittance.b));
    return lin * a.transmittance + skyOvercast(f, a.inscatter) + skyNight(f, dir) * (1.0 - a.transmittance);
}

static float3 skyApplyAerial(float3 lin, float3 rel, constant SkyFrame& f, texture3d<float> apScatter, texture3d<float> apTrans) {
    float haze;
    return skyApplyAerial(lin, rel, f, apScatter, apTrans, haze);
}

// Vanilla's render-distance fog, measured like vanilla's shaders do (cylindrical), as a fade into the sky behind.
static float skyFadeAmount(float3 rel, constant SkyFrame& f) {
    if (f.fade.z <= 0.0) return 0.0;
    float d = max(length(rel.xz), abs(rel.y));
    return saturate((d - f.fade.x) / max(f.fade.y - f.fade.x, 1e-3));
}

// sRGB encoding (what vanilla's colors are stored in), extended past 1 for HDR, and back.
static float3 skyEncode(float3 c) {
    float3 a = abs(c);
    return sign(c) * select(1.055 * pow(a, 1.0 / 2.4) - 0.055, 12.92 * a, a <= 0.0031308);
}

static float3 skyDecode(float3 c) {
    float3 a = abs(c);
    return sign(c) * select(pow((a + 0.055) / 1.055, 2.4), a / 12.92, a <= 0.04045);
}

// The tone curve (see Hdr.swift): scene-linear light (vanilla's white is 1) to display-linear light (1 is the display's
// SDR white, tone.x its current headroom, 1 on an SDR display). The identity below the knee tone.y (1 with headroom to
// spare, 0.9 without), so vanilla's colors and the GUI come out unchanged; above it an exponential shoulder rolls
// highlights (the sun, the sky around it) into the headroom with a matching slope. The shoulder acts on the largest
// channel, which keeps the hue (a sunset glow stays orange), and moves toward the same curve per channel the further
// past the headroom the light is, which whitens it: the sun's disk, thousands of times brighter, comes out white and
// brighter than the glow around it.
static float skyShoulder(float x, float k, float H) {
    if (x <= k) return x;
    float span = max(H - k, 1e-4);
    return k + span * (1.0 - exp(-(x - k) / span));
}

static float3 skyToneMap(float3 x, float4 tone) {
    x = max(x, 0.0);
    float m = max(x.r, max(x.g, x.b));
    float k = tone.y, H = tone.x;
    if (m <= k) return x;
    float3 hue = x * (skyShoulder(m, k, H) / m);
    float3 chan = float3(skyShoulder(x.r, k, H), skyShoulder(x.g, k, H), skyShoulder(x.b, k, H));
    return mix(hue, chan, saturate((m - k) / (m + 8.0 * H)));
}

// Dither for 8-bit targets (tone.z: its amplitude, 0 for RGBA16Float), interleaved gradient noise moving every frame
// (aerial.w: the frame, mod 64) so the anti-aliasing averages it away: smooth sky gradients don't band. tone.z < 0:
// HDR's packed RG11B10Float, whose steps are relative (6-bit mantissas, 5 in blue): a dither relative to the value.
static float3 skyDither(float3 enc, uint2 q, constant SkyFrame& f) {
    if (f.tone.z == 0.0) return enc;
    float2 p = float2(q) + 5.588238 * f.aerial.w;
    float n = fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715))));
    if (f.tone.z < 0.0) return enc * (1.0 + (n - 0.5) * float3(1.0 / 64.0, 1.0 / 64.0, 1.0 / 32.0));
    return enc + (n - 0.5) * f.tone.z;
}

// Scene-linear light to what goes in the level's color target: tone mapped, sRGB-encoded like vanilla's own colors.
static float3 skyOutput(float3 x, constant SkyFrame& f) { return skyEncode(skyToneMap(x, f.tone)); }

// The level's color c (as the color target holds it) at pixel q with depth d, seen through the air: aerial
// perspective, the render distance's edge faded into the sky behind, the tone curve. Sky pixels (reverse-Z puts the far
// plane at 0) are left alone: the sky pass drew them through the air already. haze gets how much the air shows (0 up
// close). Used by the aerial perspective pass and, with anti-aliasing on, by its resolve as it loads each pixel.
static float3 skyLevelColor(float3 c, uint2 q, float d, constant SkyFrame& f, texture3d<float> apScatter,
                            texture3d<float> apTrans, texture2d<float> skyView, thread float& haze) {
    haze = 0.0;
    if (d <= 0.0) return \(skyDebugHaze ? "float3(1.0, 0.0, 1.0)" : "c");
    float2 ndc = (float2(q) + 0.5) / f.view.zw * 2.0 - 1.0;
    float4 h = f.invViewProj * float4(ndc, d, 1.0);
    float3 rel = h.xyz / h.w;
    float3 lin = skyApplyAerial(skyDecode(c), rel, f, apScatter, apTrans, haze);
    float fade = skyFadeAmount(rel, f);
    if (fade > 0.0) lin = mix(lin, skyLuminance(f, rel / length(rel), skyView), fade);
    haze = max(haze, fade);
    // Debug (METALMC_EXP=skyhazedebug): red the distance (400 km full), green the haze, blue the fade; sky magenta.
    if (\(skyDebugHaze ? "true" : "false")) return float3(saturate(length(rel) / 400000.0), haze, fade);
    return skyOutput(lin, f);
}

// A sky pixel: the sky, plus the sun's disk seen through the atmosphere (limb darkened after Hestroffer and Magnan
// 1998, exponents at about 680, 550 and 440 nm, normalized so the disk's mean is its illuminance), anti-aliased over
// pixelAngle radians.
static float3 skyPixel(constant SkyFrame& f, float3 dir, texture2d<float> skyView, texture2d<float> trans, float pixelAngle) {
    float3 L = skyLuminance(f, dir, skyView);
    float R = f.sunHoriz.z;
    // The chord to the sun's direction is 2 sin(angle / 2): a cheap test first, the angle itself only near the disk.
    float chord = length(dir - f.sun.xyz);
    if (chord < 2.0 * (R + pixelAngle) && f.sunHoriz.w > 0.0) {
        float ang = 2.0 * asin(min(0.5 * chord, 1.0));   // precise for small angles, unlike acos
        float edge = saturate((R - ang) / max(pixelAngle, 1e-6) + 0.5);
        float r = f.view.y;
        float3 T = skyRaySphere(float3(0.0, r, 0.0), dir, f.atmo.planet.x) > 0.0 ? float3(0.0)
                 : skyTransmittanceToSpace(trans, f.atmo, r, dir.y);
        float mu = sqrt(saturate(1.0 - (ang / R) * (ang / R)));
        float3 alpha = float3(0.406, 0.508, 0.641);
        float3 limb = pow(max(mu, 1e-3), alpha) * (2.0 + alpha) * 0.5;
        L += T * limb * (f.sun.w * f.sunHoriz.w / (SKY_PI * R * R)) * edge;
    }
    return L;
}
"""

private let skyShaderSource = skyShaderHeader + """

kernel void sky_transmittance_lut(texture2d<float, access::write> out [[texture(0)]],
                                  constant SkyFrame& f [[buffer(0)]],
                                  uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= SKY_TRANS_W || gid.y >= SKY_TRANS_H) return;
    float Rg = f.atmo.planet.x, Rt = f.atmo.planet.y;
    float H = sqrt(Rt * Rt - Rg * Rg);
    float xmu = float(gid.x) / (SKY_TRANS_W - 1), xr = float(gid.y) / (SKY_TRANS_H - 1);
    float rho = H * xr, r = sqrt(rho * rho + Rg * Rg);
    float dmin = Rt - r, dmax = rho + H, d = dmin + xmu * (dmax - dmin);
    float mu = d <= 0.0 ? 1.0 : clamp((H * H - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
    float3 o = float3(0.0, r, 0.0), dir = float3(sqrt(max(0.0, 1.0 - mu * mu)), mu, 0.0);
    // Optical depth to the top of the atmosphere, 40 midpoints.
    float3 depth = 0.0;
    for (int i = 0; i < 40; i++) {
        float3 p = o + dir * (d * (float(i) + 0.5) / 40.0);
        depth += skyMediumAt(f.atmo, length(p) - Rg).extinction;
    }
    out.write(float4(exp(-depth * d / 40.0), 1.0), gid);
}

// One 64-thread threadgroup per texel, one direction per thread (a uniform 8 x 8 grid over the sphere), summed in
// threadgroup memory: the light that arrives after one scattering with an even phase (with the ground's bounce), and
// the fraction f_ms of light scattered again; all orders sum to L / (1 - f_ms).
kernel void sky_multiscatter_lut(texture2d<float, access::write> out [[texture(0)]],
                                 texture2d<float> trans [[texture(1)]],
                                 constant SkyFrame& f [[buffer(0)]],
                                 uint2 tg [[threadgroup_position_in_grid]],
                                 uint li [[thread_index_in_threadgroup]]) {
    threadgroup float4 sumL[64];
    threadgroup float4 sumF[64];
    float Rg = f.atmo.planet.x, Rt = f.atmo.planet.y;
    float mus = float(tg.x) / (SKY_MS_N - 1) * 2.0 - 1.0;
    float r = Rg + max(float(tg.y) / (SKY_MS_N - 1) * (Rt - Rg), 0.005);
    float3 o = float3(0.0, r, 0.0), sun = float3(sqrt(max(0.0, 1.0 - mus * mus)), mus, 0.0);
    float cz = 1.0 - 2.0 * (float(li / 8) + 0.5) / 8.0, az = 2.0 * SKY_PI * (float(li % 8) + 0.5) / 8.0;
    float sz = sqrt(max(0.0, 1.0 - cz * cz));
    float3 dir = float3(sz * cos(az), cz, sz * sin(az));
    float tg0 = skyRaySphere(o, dir, Rg), tt = skyRaySphere(o, dir, Rt);
    bool ground = tg0 > 0.0;
    float tmax = ground ? tg0 : max(tt, 0.0);
    float phase = 1.0 / (4.0 * SKY_PI);
    float3 L = 0.0, T = 1.0, F = 0.0;
    float dt = tmax / 20.0;
    for (int i = 0; i < 20; i++) {
        float3 p = o + dir * ((float(i) + 0.5) * dt);
        float len = length(p), pr = max(len, Rg + 1e-3);
        SkyMedium m = skyMediumAt(f.atmo, pr - Rg);
        float pmus = dot(p / len, sun);
        float3 sunT = skyTransmittanceToSpace(trans, f.atmo, pr, pmus) * skySunVisible(f.atmo, pr, pmus, f.sunHoriz.z);
        float3 st = exp(-m.extinction * dt);
        float3 e = max(m.extinction, float3(1e-7));
        float3 S = sunT * m.scattering * phase;
        L += T * (S - S * st) / e;
        F += T * (m.scattering - m.scattering * st) / e;
        T *= st;
    }
    if (ground) {
        float3 n = normalize(o + dir * tg0);
        float gm = dot(n, sun);
        L += T * skyTransmittanceToSpace(trans, f.atmo, Rg, gm) * saturate(gm) * f.atmo.planet.w / SKY_PI;
    }
    sumL[li] = float4(L, 0.0);
    sumF[li] = float4(F, 0.0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = 32; s > 0; s >>= 1) {
        if (li < s) {
            sumL[li] += sumL[li + s];
            sumF[li] += sumF[li + s];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (li == 0) {
        float3 L2 = sumL[0].rgb / 64.0, fms = sumF[0].rgb / 64.0;
        out.write(float4(L2 / max(1.0 - fms, float3(1e-3)), 1.0), tg);
    }
}

kernel void sky_view_lut(texture2d<float, access::write> out [[texture(0)]],
                         texture2d<float> trans [[texture(1)]],
                         texture2d<float> ms [[texture(2)]],
                         constant SkyFrame& f [[buffer(0)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= SKY_VIEW_W || gid.y >= SKY_VIEW_H) return;
    float u = float(gid.x) / (SKY_VIEW_W - 1), v = float(gid.y) / (SKY_VIEW_H - 1);
    float r = f.view.y, Rg = f.atmo.planet.x, Rt = f.atmo.planet.y;
    // Rows from the zenith to just above the planet's horizon (see skyViewUV).
    float c = 1.0 - v;
    float vz = f.horizon.x * (1.0 - c * c) * (1.0 - 1e-4);
    float cosPhi = 1.0 - 2.0 * u * u, sinPhi = sqrt(max(0.0, 1.0 - cosPhi * cosPhi));
    float3 dir = float3(sin(vz) * cosPhi, cos(vz), sin(vz) * sinPhi);
    float mus = f.sun.y;
    float3 sun = float3(sqrt(max(0.0, 1.0 - mus * mus)), mus, 0.0);
    float3 o = float3(0.0, r, 0.0);
    float tg0 = skyRaySphere(o, dir, Rg), tt = skyRaySphere(o, dir, Rt);
    float tmax = tg0 > 0.0 ? tg0 : max(tt, 0.0);
    float3 T;
    float3 L = skyIntegrate(f, o, dir, sun, tmax, 40, trans, ms, T);
    out.write(float4(L * f.sun.w, 1.0), gid);
}

// One thread per column of the volume, marching once out to the range and writing each slice as it passes it (four
// segments per slice). Rays that go below the ground carry on through air at the ground's density: terrain below sea
// level (valleys, the sea floor) is still at the end of them.
kernel void sky_aerial_lut(texture3d<float, access::write> scatterOut [[texture(0)]],
                           texture3d<float, access::write> transOut [[texture(1)]],
                           texture2d<float> trans [[texture(2)]],
                           texture2d<float> ms [[texture(3)]],
                           constant SkyFrame& f [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= SKY_AP_W || gid.y >= SKY_AP_H) return;
    float u = float(gid.x) / (SKY_AP_W - 1), v = float(gid.y) / (SKY_AP_H - 1);
    float cosPhi = 1.0 - 2.0 * u * u, sinPhi = sqrt(max(0.0, 1.0 - cosPhi * cosPhi));
    float e = 2.0 * v - 1.0;
    float y = sign(e) * e * e, horiz = sqrt(max(0.0, 1.0 - y * y));
    float3 dir = float3(horiz * cosPhi, y, horiz * sinPhi);
    float mus = f.sun.y;
    float3 sun = float3(sqrt(max(0.0, 1.0 - mus * mus)), mus, 0.0);
    float3 o = float3(0.0, f.view.y, 0.0);
    float c = dot(dir, sun);
    float pr = skyRayleighPhase(c), pm = skyMiePhase(f.atmo.mie.w, c);
    float3 L = 0.0, T = 1.0;
    float t0 = 0.0;
    for (int k = 1; k <= SKY_AP_D; k++) {
        float x = float(k) / SKY_AP_D;
        float t1 = f.aerial.x * x * x;
        float dt = (t1 - t0) / 4.0;
        for (int j = 0; j < 4; j++) {
            SkyMedium m;
            float3 S = skySource(f, o + dir * (t0 + (float(j) + 0.5) * dt), sun, pr, pm, trans, ms, m);
            skySegment(S, m.extinction, dt, L, T);
        }
        scatterOut.write(float4(L * f.sun.w, 1.0), uint3(gid, uint(k - 1)));
        transOut.write(float4(T, 1.0), uint3(gid, uint(k - 1)));
        t0 = t1;
    }
}

struct SkyVOut { float4 pos [[position]]; };

vertex SkyVOut sky_fullscreen_vs(uint vid [[vertex_id]]) {
    SkyVOut o;
    float2 c = float2(float((vid << 1) & 2), float(vid & 2));
    o.pos = float4(c * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// The view direction through a point of the target, uv in [0, 1] (reverse-Z: the near plane is at 1). Texture row 0 is
// the bottom of clip space (vanilla's shaders are translated with a vertical flip), so normalized device y is 2 v - 1.
static float3 skyViewDir(constant SkyFrame& f, float2 uv) {
    float4 h = f.invViewProj * float4(uv * 2.0 - 1.0, 1.0, 1.0);
    return normalize(h.xyz / h.w);
}

// The sky, drawn in vanilla's sky pass in place of its sky disc, every pixel worked out (when the quarter-resolution
// sky below isn't ready).
fragment float4 sky_fs(SkyVOut in [[stage_in]],
                       constant SkyFrame& f [[buffer(19)]],
                       texture2d<float> skyView [[texture(25)]],
                       texture2d<float> trans [[texture(26)]]) {
    float3 dir = skyViewDir(f, in.pos.xy / f.view.zw);
    return float4(skyDither(skyOutput(skyPixel(f, dir, skyView, trans, f.horizon.z), f), uint2(in.pos.xy), f), 1.0);
}

// The sky at a quarter of the resolution on each axis, as it goes in the target (tone mapped, encoded), without the
// sun's disk: the sky has no edges (below the planet's horizon it's the horizon's color), so filtering it up to full
// size looks the same, for a sixteenth of the work.
kernel void sky_lowres(texture2d<float, access::write> out [[texture(0)]],
                       texture2d<float> skyView [[texture(1)]],
                       constant SkyFrame& f [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float2 uv = (float2(gid) + 0.5) / float2(out.get_width(), out.get_height());
    out.write(float4(skyOutput(skyLuminance(f, skyViewDir(f, uv), skyView), f), 1.0), gid);
}

// The sky from the quarter-resolution one, filtered up; near the sun, worked out per pixel with its disk.
fragment float4 sky_upsample_fs(SkyVOut in [[stage_in]],
                                constant SkyFrame& f [[buffer(19)]],
                                texture2d<float> skyView [[texture(25)]],
                                texture2d<float> trans [[texture(26)]],
                                texture2d<float> lowres [[texture(27)]]) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = in.pos.xy / f.view.zw;
    float3 dir = skyViewDir(f, uv);
    float3 o;
    if (length(dir - f.sun.xyz) < 2.0 * (f.sunHoriz.z + f.horizon.z) && f.sunHoriz.w > 0.0) {
        o = skyOutput(skyPixel(f, dir, skyView, trans, f.horizon.z), f);
    } else {
        o = lowres.sample(s, uv).rgb;
    }
    return float4(skyDither(o, uint2(in.pos.xy), f), 1.0);
}

// After the level, without anti-aliasing (with it, its resolve does this as it loads the color): the level through the
// air (skyLevelColor), after the ray-traced shadows' shade if they were left for this pass (so the haze in front of a
// shadow isn't darkened). Reads the color it replaces (programmable blending).
fragment float4 sky_aerial_fs(SkyVOut in [[stage_in]], float4 dst [[color(0)]],
                              depth2d<float, access::read> depth [[texture(0)]],
                              texture2d<half, access::read> lit [[texture(1)]],
                              texture3d<float> apScatter [[texture(2)]],
                              texture3d<float> apTrans [[texture(3)]],
                              texture2d<float> skyView [[texture(4)]],
                              constant SkyFrame& f [[buffer(0)]],
                              constant float4& shadow [[buffer(1)]]) {
    uint2 q = uint2(in.pos.xy);
    float d = depth.read(q);
    if (d <= 0.0) return dst;
    // The shade multiplies the stored color, as the shadows' own pass and the anti-aliasing apply it.
    float3 c = dst.rgb;
    if (shadow.x > 0.0) c *= float(lit.read(min(q / uint(shadow.y), uint2(lit.get_width() - 1, lit.get_height() - 1))).r);
    float haze;
    float3 o = skyLevelColor(c, q, d, f, apScatter, apTrans, skyView, haze);
    // Dither only where the air shows (smooth haze gradients), not over the near terrain's own texture.
    return float4(mix(o, skyDither(o, q, f), saturate(8.0 * haze)), dst.a);
}

// Offline check (mmc_debug_sky_render): an equirectangular view from the camera (x: azimuth from the sun, y: elevation,
// over the window view.xy, view.zw in radians), over a flat grass plain at sea level with rock ridges 2, 8, 30, 100 and
// 250 km away, lit like vanilla (their colors dim at night), with the aerial perspective and tone curve the game uses.
// out gets what the color target would hold; lin the scene-linear light before the tone curve.
kernel void sky_debug_panorama(texture2d<float, access::write> out [[texture(0)]],
                               texture2d<float, access::write> lin [[texture(1)]],
                               texture2d<float> skyView [[texture(2)]],
                               texture2d<float> trans [[texture(3)]],
                               texture3d<float> apScatter [[texture(4)]],
                               texture3d<float> apTrans [[texture(5)]],
                               constant SkyFrame& f [[buffer(0)]],
                               constant float4& cam [[buffer(1)]],
                               constant float4& view [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
    uint w = out.get_width(), hgt = out.get_height();
    if (gid.x >= w || gid.y >= hgt) return;
    // view: x, y the azimuth at the left and right edges, z, w the elevation at the top and bottom.
    float az = mix(view.x, view.y, (float(gid.x) + 0.5) / float(w));
    float el = mix(view.z, view.w, (float(gid.y) + 0.5) / float(hgt));
    float2 sh = f.sunHoriz.xy;
    float2 side = float2(-sh.y, sh.x);
    float2 hd = sh * cos(az) + side * sin(az);
    float3 dir = float3(hd.x * cos(el), sin(el), hd.y * cos(el));
    float pixelAngle = abs(view.z - view.w) / float(hgt);
    // cam.x: camera y (blocks), cam.y: sea level (blocks), cam.z: vanilla's daylight brightness for terrain (0-1).
    float height = cam.x - cam.y;
    float tHit = 1e30;
    float3 albedo = 0.0;
    if (dir.y < 0.0) {
        tHit = height / -dir.y;
        albedo = float3(0.35, 0.55, 0.22);   // vanilla's plains grass, sRGB-encoded
    }
    float ridges[5] = {2000.0, 8000.0, 30000.0, 100000.0, 250000.0};
    float hl = length(dir.xz);
    for (int i = 0; i < 5; i++) {
        // Each ridge covers 50 degrees of azimuth, starting at -150 + 60 i degrees from the sun, and rises 200-700 blocks.
        float a0 = (-150.0 + 60.0 * float(i)) * SKY_PI / 180.0;
        if (az < a0 || az > a0 + 50.0 * SKY_PI / 180.0 || hl < 1e-4) continue;
        float t = ridges[i] / hl;
        float y = dir.y * t;   // relative to the camera
        float top = 200.0 + 500.0 * abs(sin((az - a0) * 9.0)) * (0.6 + 0.4 * sin((az - a0) * 23.0));
        if (y > -height && y < top - height && t < tHit) {
            tHit = t;
            albedo = float3(0.45, 0.45, 0.47);   // stone
        }
    }
    float3 L;
    if (tHit < 1e29) {
        float3 rel = dir * tHit;
        float3 c = skyDecode(albedo) * cam.z;
        L = skyApplyAerial(c, rel, f, apScatter, apTrans);
        float fade = skyFadeAmount(rel, f);
        if (fade > 0.0) L = mix(L, skyLuminance(f, dir, skyView), fade);
    } else {
        L = skyPixel(f, dir, skyView, trans, pixelAngle);
    }
    lin.write(float4(L, 1.0), gid);
    out.write(float4(skyOutput(L, f), 1.0), gid);
}
"""

/// Mirrors SkyAtmosphere in the shaders.
struct SkyAtmosphereGPU {
    var rayleigh: SIMD4<Float>
    var mie: SIMD4<Float>
    var ozone: SIMD4<Float>
    var planet: SIMD4<Float>
}

/// Mirrors SkyFrame in the shaders.
struct SkyFrameGPU {
    var invViewProj = matrix_identity_float4x4
    var atmo = SkyAtmosphereGPU(rayleigh: .zero, mie: .zero, ozone: .zero, planet: .zero)
    var sun = SIMD4<Float>.zero
    var sunHoriz = SIMD4<Float>.zero
    var view = SIMD4<Float>.zero
    var tone = SIMD4<Float>(1, 0.9, 0, 0)
    var aerial = SIMD4<Float>.zero
    var fade = SIMD4<Float>.zero
    var horizon = SIMD4<Float>.zero
}

/// The aerial perspective volume's range (km): past it, terrain gets its last slice.
private let skyAerialRange: Float = 400
/// Mie (haze) density multiplier at full rain: visibility drops from about 200 km to about 10.
private let skyRainHaze: Float = 40
/// The night sky's brightness at the zenith (scene-linear; about vanilla's dark blue night sky once encoded).
private let skyNightLevel: Float = 0.004

/// The tone curve's knee for a display headroom (see skyToneMap).
func skyKnee(_ headroom: Float) -> Float { min(1, 0.9 + 0.4 * (max(headroom, 1) - 1)) }

final class Sky: @unchecked Sendable {
    static let shared = Sky()

    private var library: MTLLibrary?
    private var transPipe: MTLComputePipelineState?
    private var msPipe: MTLComputePipelineState?
    private var viewPipe: MTLComputePipelineState?
    private var aerialPipe: MTLComputePipelineState?
    private var debugPipe: MTLComputePipelineState?
    private var lowresPipe: MTLComputePipelineState?
    private var drawPipes: [String: MTLRenderPipelineState] = [:]   // by the pass's color and depth formats, and the variant
    /// The quarter-resolution sky, and the frame, view and target it was made for.
    private var lowres: MTLTexture?
    private var lowresView: (frame: Int, invViewProj: simd_float4x4, width: Int, height: Int, format: MTLPixelFormat)?
    private var aerialDrawPipes: [UInt: MTLRenderPipelineState] = [:]   // by the color target's format
    private(set) var transmittance: MTLTexture?
    private(set) var multiScatter: MTLTexture?
    private(set) var skyView: MTLTexture?
    private(set) var apScatter: MTLTexture?
    private(set) var apTrans: MTLTexture?
    private var failed = false
    /// This frame's parameters (mmc_sky_prepare); the matrix and target size are filled in per pass.
    var frame = SkyFrameGPU()
    /// The atmosphere the transmittance and multiple-scattering tables were built for, and what the sky view and
    /// aerial perspective were built for (sun, altitude, rain, exposure), to rebuild only on change.
    private var lutRain: Float = -1
    private var viewKey = SIMD4<Float>(repeating: .nan)
    /// Set when the tables for this frame are ready.
    private(set) var ready = false
    private var frames = 0
    private var rebuilds = 0

    func ensure() -> Bool {
        if failed { return false }
        if library != nil { return true }
        do {
            let lib = try ctx.device.makeLibrary(source: skyShaderSource, options: nil)
            func pipe(_ name: String) throws -> MTLComputePipelineState {
                try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: name)!)
            }
            transPipe = try pipe("sky_transmittance_lut")
            msPipe = try pipe("sky_multiscatter_lut")
            viewPipe = try pipe("sky_view_lut")
            aerialPipe = try pipe("sky_aerial_lut")
            debugPipe = try pipe("sky_debug_panorama")
            lowresPipe = try pipe("sky_lowres")
            func tex2(_ w: Int, _ h: Int) -> MTLTexture? {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: w, height: h, mipmapped: false)
                d.usage = [.shaderRead, .shaderWrite]
                d.storageMode = .private
                return ctx.device.makeTexture(descriptor: d)
            }
            func tex3() -> MTLTexture? {
                let d = MTLTextureDescriptor()
                d.textureType = .type3D
                d.pixelFormat = .rgba16Float
                d.width = 32
                d.height = 64
                d.depth = 32
                d.usage = [.shaderRead, .shaderWrite]
                d.storageMode = .private
                return ctx.device.makeTexture(descriptor: d)
            }
            transmittance = tex2(256, 64)
            multiScatter = tex2(32, 32)
            skyView = tex2(192, 108)
            apScatter = tex3()
            apTrans = tex3()
            library = lib
        } catch {
            log("sky: shaders failed: \(error)")
            failed = true
            return false
        }
        if transmittance == nil || multiScatter == nil || skyView == nil || apScatter == nil || apTrans == nil {
            log("sky: tables failed")
            failed = true
            return false
        }
        return true
    }

    /// Sets this frame's parameters from vanilla's sun angle (radians), rain brightness (1 clear, 0 full rain), the
    /// camera's height above sea level (blocks), and the render-distance fade (end <= start: none), and encodes the
    /// table rebuilds that changed into `cb`.
    func prepare(cb: MTLCommandBuffer, sunAngle: Float, rainBrightness: Float, height: Float,
                 fadeStart: Float, fadeEnd: Float, headroom: Float) {
        deferredAerial = nil
        guard ensure(), let transPipe, let msPipe, let viewPipe, let aerialPipe,
              let transmittance, let multiScatter, let skyView, let apScatter, let apTrans else { ready = false; return }
        frames += 1
        let rain = min(max(1 - rainBrightness, 0), 1)
        // Rain is quantized so its fade in and out rebuilds the two atmosphere tables 32 times, not every frame.
        let rainQ = (rain * 32).rounded() / 32
        let mieScale = 1 + skyRainHaze * rainQ
        var f = frame
        // Earth's air (Hillaire's values, per km). In rain, much more haze with a flatter phase function: cloud and
        // rain scatter light every way, so the sky evens out instead of glowing around the sun.
        f.atmo = SkyAtmosphereGPU(rayleigh: SIMD4(5.802e-3, 13.558e-3, 33.1e-3, 8),
                                  mie: SIMD4(3.996e-3 * mieScale, 4.44e-3 * mieScale, 1.2, 0.8 - 0.5 * rainQ),
                                  ozone: SIMD4(0.650e-3, 1.881e-3, 0.085e-3, 25),
                                  planet: SIMD4(6360, 6460, 15, 0.3))
        // Vanilla's sun: rotated -90 degrees about y, then by the sun angle about x, from straight up.
        let sun = SIMD3<Float>(-sin(sunAngle), cos(sunAngle), 0)
        let hl = simd_length(SIMD2(sun.x, sun.z))
        let horiz = hl > 1e-4 ? SIMD2(sun.x, sun.z) / hl : SIMD2<Float>(1, 0)
        // Altitude above sea level, kept inside the atmosphere (underground, and past 95 km, the sky is the nearest one's).
        let altitude = min(max(height.isFinite ? height : 1, 1), 95_000) / 1000
        // Eye adaptation from the sun's height (no histogram): up to 3.5 stops as the sun goes from about 15 degrees up to
        // 9 under the horizon, so dusk and dawn skies keep their color instead of going black (vanilla's lightmap dims the
        // terrain over the same span). The clouds of full rain take 80% of the sun's light.
        let adapt = exp2(3.5 * smoothstep(0.25, -0.15, sun.y))
        f.sun = SIMD4(sun, skyExposure * adapt * (1 - 0.8 * rain))
        f.sunHoriz = SIMD4(horiz.x, horiz.y, skySunRadius, 1 - rain)
        f.view.x = altitude
        f.view.y = f.atmo.planet.x + altitude
        // The planet's horizon from the camera: its zenith angle and how far below the horizontal it is (Hillaire's beta
        // is the angle at the camera between the planet's center and the horizon).
        let rg = f.atmo.planet.x
        let beta = acos(min(max(sqrt(max(0, (f.view.y - rg) * (f.view.y + rg))) / f.view.y, -1), 1))
        f.horizon = SIMD4(.pi - beta, .pi / 2 - beta, 0, 0)
        // The night sky's glow fades in as the sun goes under the horizon (full at 12 degrees under), before the
        // twilight it takes over from has gone.
        f.tone = SIMD4(max(headroom, 1), skyKnee(headroom), 0, skyNightLevel * smoothstep(0.1, -0.2, sun.y))
        f.aerial = SIMD4(skyAerialRange, skyHaze, rain, Float(frames % 64))
        f.fade = fadeEnd > fadeStart ? SIMD4(fadeStart, fadeEnd, 1, 0) : .zero
        frame = f

        let atmoChanged = rainQ != lutRain
        let key = SIMD4(sunAngle, altitude, rain, f.sun.w)
        // The sky view and aerial perspective: when the sun moved 0.01 degrees (about every 4 frames of a day at 120 Hz),
        // the camera's altitude changed 0.5%, or the rain or exposure changed.
        let viewChanged = atmoChanged || !(abs(key.x - viewKey.x) < 1.745e-4) || !(abs(key.y - viewKey.y) <= 0.005 * max(key.y, 0.01))
            || key.z != viewKey.z || key.w != viewKey.w
        if !atmoChanged && !viewChanged { ready = true; return }
        ctx.endBlit()
        guard let enc = cb.makeComputeCommandEncoder() else { ready = false; return }
        enc.label = "MetalMC sky tables"
        enc.setBytes(&f, length: MemoryLayout<SkyFrameGPU>.stride, index: 0)
        if atmoChanged {
            enc.setComputePipelineState(transPipe)
            enc.setTexture(transmittance, index: 0)
            enc.dispatchThreads(MTLSize(width: 256, height: 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
            enc.setComputePipelineState(msPipe)
            enc.setTexture(multiScatter, index: 0)
            enc.setTexture(transmittance, index: 1)
            enc.dispatchThreadgroups(MTLSize(width: 32, height: 32, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
            lutRain = rainQ
        }
        enc.setComputePipelineState(viewPipe)
        enc.setTexture(skyView, index: 0)
        enc.setTexture(transmittance, index: 1)
        enc.setTexture(multiScatter, index: 2)
        enc.dispatchThreads(MTLSize(width: 192, height: 108, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        enc.setComputePipelineState(aerialPipe)
        enc.setTexture(apScatter, index: 0)
        enc.setTexture(apTrans, index: 1)
        enc.setTexture(transmittance, index: 2)
        enc.setTexture(multiScatter, index: 3)
        enc.dispatchThreads(MTLSize(width: 32, height: 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
        viewKey = key
        rebuilds += 1
        ready = true
        if frames % 1200 == 0 {
            log("sky: \(rebuilds) table rebuilds in the last 1200 frames, sun \(String(format: "%.1f", sunAngle * 180 / .pi)) degrees, altitude \(String(format: "%.3f", altitude)) km, rain \(rainQ)")
            rebuilds = 0
        }
    }

    /// The quarter-resolution sky (sky_lowres) for this frame's view, before vanilla's sky pass opens. `color`: the
    /// target the sky will be drawn into (its size and format). Returns false if it wasn't made (the sky pass then works
    /// every pixel out).
    func prepareView(color: MTLTexture, invViewProj: simd_float4x4, headroom: Float) -> Bool {
        guard ready, ctx.pass == nil, let lowresPipe, let skyView else { return false }
        let lw = (color.width + 3) / 4, lh = (color.height + 3) / 4
        if lowres == nil || lowres!.width != lw || lowres!.height != lh {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: lw, height: lh, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            lowres = ctx.device.makeTexture(descriptor: d)
        }
        guard let lowres else { return false }
        var f = levelFrame(color: color, invViewProj: invViewProj, headroom: headroom)
        ctx.endBlit()
        guard let enc = ctx.ensureCB().makeComputeCommandEncoder() else { return false }
        enc.label = "MetalMC sky, quarter resolution"
        enc.setComputePipelineState(lowresPipe)
        enc.setTexture(lowres, index: 0)
        enc.setTexture(skyView, index: 1)
        enc.setBytes(&f, length: MemoryLayout<SkyFrameGPU>.stride, index: 0)
        enc.dispatchThreads(MTLSize(width: lw, height: lh, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        lowresView = (frames, invViewProj, color.width, color.height, color.pixelFormat)
        return true
    }

    /// Draws the sky into the open render pass (vanilla's sky pass). `invViewProj`: camera-relative clip to world.
    func draw(invViewProj: simd_float4x4, headroom: Float) -> Bool {
        guard ready, let enc = ctx.pass, let lib = library, let skyView, let transmittance else { return false }
        let colors = ctx.passColorFormats, depth = ctx.passDepthFormat
        // The quarter-resolution sky if it was made this frame, for this view and this target.
        var upsample: MTLTexture?
        if let v = lowresView, v.frame == frames, v.invViewProj == invViewProj, v.width == ctx.passWidth, v.height == ctx.passHeight,
           v.format == colors.first { upsample = lowres }
        let key = colors.map { String($0.rawValue) }.joined(separator: ",") + "/\(depth.rawValue)" + (upsample != nil ? "/up" : "")
        if drawPipes[key] == nil {
            let d = MTLRenderPipelineDescriptor()
            d.label = "MetalMC sky"
            d.vertexFunction = lib.makeFunction(name: "sky_fullscreen_vs")
            d.fragmentFunction = lib.makeFunction(name: upsample != nil ? "sky_upsample_fs" : "sky_fs")
            for (i, f) in colors.enumerated() {
                d.colorAttachments[i].pixelFormat = f
                if i > 0 { d.colorAttachments[i].writeMask = [] }
            }
            d.depthAttachmentPixelFormat = depth
            if depth == .depth32Float_stencil8 { d.stencilAttachmentPixelFormat = depth }
            do { drawPipes[key] = try ctx.device.makeRenderPipelineState(descriptor: d) } catch {
                log("sky: draw pipeline failed: \(error)")
                failed = true
                return false
            }
        }
        guard let pipe = drawPipes[key], let target = colors.first, target != .invalid else { return false }
        var f = frame
        f.invViewProj = invViewProj
        f.view.z = Float(ctx.passWidth)
        f.view.w = Float(ctx.passHeight)
        f.horizon.z = skyPixelAngle(invViewProj, width: f.view.z)
        // An SDR target holds at most 1, in 8 bits (dithered); a float one (METALMC_EXP=hdr) up to the display's headroom.
        let isFloat = isFloatFrame(target)
        let h: Float = isFloat ? max(headroom, 1) : 1
        f.tone.x = h
        f.tone.y = skyKnee(h)
        f.tone.z = isFloat ? (target == .rg11b10Float ? -1 : 0) : 1.0 / 255
        enc.setRenderPipelineState(pipe)
        enc.setDepthStencilState(ctx.depthState(compare: .always, write: false))
        enc.setCullMode(.none)
        enc.setTriangleFillMode(.fill)
        enc.setDepthBias(0, slopeScale: 0, clamp: 0)
        enc.setFragmentBytes(&f, length: MemoryLayout<SkyFrameGPU>.stride, index: 19)
        enc.setFragmentTexture(skyView, index: 25)
        enc.setFragmentTexture(transmittance, index: 26)
        if let upsample { enc.setFragmentTexture(upsample, index: 27) }
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        ctx.statDraws += 1
        // Minecraft's pipeline, depth, cull and bias state must be re-applied by its next setPipeline, and nothing it
        // binds is at these indices, but forget them anyway so a later bind there isn't skipped.
        ctx.pipe = nil
        ctx.boundPipeState = nil
        ctx.boundBuffers[1][19] = nil
        ctx.boundTextures[1][25] = nil
        ctx.boundTextures[1][26] = nil
        ctx.boundTextures[1][27] = nil
        return true
    }

    /// Left for the anti-aliasing's resolve this frame: the parameters for the level's size (see takeDeferredAerial).
    private var deferredAerial: (frame: SkyFrameGPU, width: Int, height: Int)?

    /// Aerial perspective, the render-distance fade and the tone curve on the level just drawn into `color`, from
    /// `depth` (skyLevelColor). With `deferToTaa` it's left for the anti-aliasing's resolve, which reads every pixel's
    /// color and depth anyway (at the panel's resolution that adds about 0.28 ms to it, where a pass of its own costs
    /// 0.38) and applies the ray-traced shadows' shade first. Otherwise it's its own render pass (needs no pass open),
    /// taking any shadows that were left for the anti-aliasing so they darken the surface before the haze is added.
    func aerial(color: MTLTexture, depth: MTLTexture, invViewProj: simd_float4x4, headroom: Float, deferToTaa: Bool) -> Bool {
        guard ready, ctx.pass == nil, let lib = library, let skyView, let apScatter, let apTrans else { return false }
        if deferToTaa {
            deferredAerial = (levelFrame(color: color, invViewProj: invViewProj, headroom: headroom), color.width, color.height)
            return true
        }
        if aerialDrawPipes[color.pixelFormat.rawValue] == nil {
            let d = MTLRenderPipelineDescriptor()
            d.label = "MetalMC aerial perspective"
            d.vertexFunction = lib.makeFunction(name: "sky_fullscreen_vs")
            d.fragmentFunction = lib.makeFunction(name: "sky_aerial_fs")
            d.colorAttachments[0].pixelFormat = color.pixelFormat
            do { aerialDrawPipes[color.pixelFormat.rawValue] = try ctx.device.makeRenderPipelineState(descriptor: d) } catch {
                log("sky: aerial perspective pipeline failed: \(error)")
                failed = true
                return false
            }
        }
        guard let pipe = aerialDrawPipes[color.pixelFormat.rawValue] else { return false }
        var f = levelFrame(color: color, invViewProj: invViewProj, headroom: headroom)
        let shadows = RtShadows.shared.takeDeferred(width: color.width, height: color.height)
        var shadow = shadows?.params ?? .zero
        ctx.endBlit()
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = color
        d.colorAttachments[0].loadAction = .load
        d.colorAttachments[0].storeAction = .store
        profAttach(d, "sky aerial perspective \(color.width)x\(color.height)")
        guard let enc = ctx.ensureCB().makeRenderCommandEncoder(descriptor: d) else { return false }
        enc.label = "MetalMC aerial perspective"
        enc.setRenderPipelineState(pipe)
        enc.setFragmentTexture(depth, index: 0)
        enc.setFragmentTexture(shadows?.lit ?? RtShadows.shared.dummyLit(), index: 1)
        enc.setFragmentTexture(apScatter, index: 2)
        enc.setFragmentTexture(apTrans, index: 3)
        enc.setFragmentTexture(skyView, index: 4)
        enc.setFragmentBytes(&f, length: MemoryLayout<SkyFrameGPU>.stride, index: 0)
        enc.setFragmentBytes(&shadow, length: 16, index: 1)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        ctx.statPasses += 1
        return true
    }

    /// This frame's parameters for the level in `color`: its matrix and size, and the tone curve for its format (an SDR
    /// target holds at most 1, in 8 bits, dithered; a float one, METALMC_EXP=hdr, up to the display's headroom).
    private func levelFrame(color: MTLTexture, invViewProj: simd_float4x4, headroom: Float) -> SkyFrameGPU {
        var f = frame
        f.invViewProj = invViewProj
        f.view.z = Float(color.width)
        f.view.w = Float(color.height)
        let isFloat = isFloatFrame(color.pixelFormat)
        let h: Float = isFloat ? max(headroom, 1) : 1
        f.tone.x = h
        f.tone.y = skyKnee(h)
        f.tone.z = isFloat ? (color.pixelFormat == .rg11b10Float ? -1 : 0) : 1.0 / 255
        return f
    }

    /// For the anti-aliasing's resolve (Taa.swift): the level-through-the-air work left for it this frame, if any.
    func takeDeferredAerial(width: Int, height: Int) -> (frame: SkyFrameGPU, apScatter: MTLTexture, apTrans: MTLTexture, skyView: MTLTexture)? {
        defer { deferredAerial = nil }
        guard let d = deferredAerial, d.width == width, d.height == height, let apScatter, let apTrans, let skyView else { return nil }
        return (d.frame, apScatter, apTrans, skyView)
    }

    /// Offline check: renders the panorama kernel (see sky_debug_panorama) over `window` (azimuth left, right, elevation
    /// top, bottom, radians) into `out` (w x h RGBA8, what an SDR color target would hold; values above 1 with a
    /// headroom above 1 are clipped) and `lin` (w x h x 4 floats, scene-linear), on its own command buffers, and writes
    /// GPU milliseconds per table into `times`: transmittance, multiple scattering, sky view, aerial perspective, then the view.
    func debugRender(sunAngle: Float, rainBrightness: Float, camY: Double, headroom: Float, window: SIMD4<Float>, width: Int, height: Int,
                     out: UnsafeMutablePointer<UInt8>, lin: UnsafeMutablePointer<Float>, times: UnsafeMutablePointer<Double>) -> Bool {
        guard ensure(), let debugPipe, let skyView, let transmittance, let apScatter, let apTrans else { return false }
        func timed(_ body: (MTLCommandBuffer) -> Void) -> Double {
            let cb = ctx.queue.makeCommandBuffer()!
            body(cb)
            cb.commit()
            cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000
        }
        // Build everything from scratch, one table per command buffer so each is timed on its own.
        lutRain = -1
        viewKey = SIMD4(repeating: .nan)
        _ = timed { cb in prepare(cb: cb, sunAngle: sunAngle, rainBrightness: rainBrightness, height: Float(camY) - 63,
                                  fadeStart: 0, fadeEnd: 0, headroom: headroom) }
        var f = frame
        func one(_ pipe: MTLComputePipelineState?, _ setup: (MTLComputeCommandEncoder) -> Void) -> Double {
            timed { cb in
                let enc = cb.makeComputeCommandEncoder()!
                enc.setComputePipelineState(pipe!)
                enc.setBytes(&f, length: MemoryLayout<SkyFrameGPU>.stride, index: 0)
                setup(enc)
                enc.endEncoding()
            }
        }
        times[0] = one(transPipe) { e in
            e.setTexture(transmittance, index: 0)
            e.dispatchThreads(MTLSize(width: 256, height: 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        }
        times[1] = one(msPipe) { e in
            e.setTexture(multiScatter, index: 0)
            e.setTexture(transmittance, index: 1)
            e.dispatchThreadgroups(MTLSize(width: 32, height: 32, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        }
        times[2] = one(viewPipe) { e in
            e.setTexture(skyView, index: 0)
            e.setTexture(transmittance, index: 1)
            e.setTexture(multiScatter, index: 2)
            e.dispatchThreads(MTLSize(width: 192, height: 108, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        }
        times[3] = one(aerialPipe) { e in
            e.setTexture(apScatter, index: 0)
            e.setTexture(apTrans, index: 1)
            e.setTexture(transmittance, index: 2)
            e.setTexture(multiScatter, index: 3)
            e.dispatchThreads(MTLSize(width: 32, height: 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        }
        let od = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        od.usage = [.shaderWrite]
        od.storageMode = .shared
        let ld = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: width, height: height, mipmapped: false)
        ld.usage = [.shaderWrite]
        ld.storageMode = .shared
        guard let outTex = ctx.device.makeTexture(descriptor: od), let linTex = ctx.device.makeTexture(descriptor: ld) else { return false }
        let h: Float = max(headroom, 1)
        f.tone.x = h
        f.tone.y = skyKnee(h)
        // Terrain's brightness as vanilla's lightmap would give it: full by day, about a fifth at night.
        let sunY = cos(sunAngle)
        var cam = SIMD4<Float>(Float(camY), 63, 0.2 + 0.8 * smoothstep(-0.2, 0.15, sunY), 0)
        var view = window
        times[4] = one(debugPipe) { e in
            e.setTexture(outTex, index: 0)
            e.setTexture(linTex, index: 1)
            e.setTexture(skyView, index: 2)
            e.setTexture(transmittance, index: 3)
            e.setTexture(apScatter, index: 4)
            e.setTexture(apTrans, index: 5)
            e.setBytes(&cam, length: 16, index: 1)
            e.setBytes(&view, length: 16, index: 2)
            e.dispatchThreads(MTLSize(width: width, height: height, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        }
        outTex.getBytes(out, bytesPerRow: width * 4, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        linTex.getBytes(lin, bytesPerRow: width * 16, from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        return true
    }

    /// Offline timing of the per-frame passes at a given size, encoded as the game encodes them: the sky pass and the
    /// aerial perspective over a synthetic depth buffer (half sky, half terrain from 10 blocks to the horizon). Returns
    /// GPU milliseconds (median of `runs`): [0] the sky view + aerial perspective rebuild, [1] the sky pass working out
    /// every pixel, [2] the aerial perspective as its own pass, [3] the sky pass from the quarter-resolution sky (its
    /// compute included), [4] the anti-aliasing alone, [5] the anti-aliasing with the aerial perspective in its load.
    func debugTimePasses(width: Int, height: Int, runs: Int, out: UnsafeMutablePointer<Double>) -> Bool {
        let cb0 = ctx.queue.makeCommandBuffer()!
        prepare(cb: cb0, sunAngle: 0.96, rainBrightness: 1, height: 87, fadeStart: 3000, fadeEnd: 3500, headroom: 1)
        cb0.commit()
        cb0.waitUntilCompleted()
        guard ready, let viewPipe, let aerialPipe, let skyView, let transmittance, let multiScatter,
              let apScatter, let apTrans else { return false }
        func tex(_ format: MTLPixelFormat, _ usage: MTLTextureUsage) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: width, height: height, mipmapped: false)
            d.usage = usage
            d.storageMode = .private
            return ctx.device.makeTexture(descriptor: d)!
        }
        let color = tex(.rgba8Unorm, [.renderTarget, .shaderRead])
        let depth = tex(.depth32Float, [.renderTarget, .shaderRead])
        let colorHandle = makeHandle(TextureBox(color)), depthHandle = makeHandle(TextureBox(depth))
        defer { mmc_handle_release(colorHandle); mmc_handle_release(depthHandle) }
        // Synthetic depth: reverse-Z values from 1 (near) down to 0 (sky) across the rows.
        let fill = try! ctx.device.makeLibrary(source: """
            #include <metal_stdlib>
            using namespace metal;
            struct V { float4 pos [[position]]; };
            vertex V fill_vs(uint vid [[vertex_id]]) { V o; float2 c = float2((vid << 1) & 2, vid & 2); o.pos = float4(c * 2.0 - 1.0, 0.0, 1.0); return o; }
            struct O { float4 c [[color(0)]]; float d [[depth(any)]]; };
            fragment O fill_fs(V in [[stage_in]], constant float2& size [[buffer(0)]]) {
                O o; float v = in.pos.y / size.y; o.c = float4(0.3, 0.5, 0.2, 1.0); o.d = v < 0.5 ? 0.0 : 0.05 / (1.0 + 400.0 * (1.0 - v)); return o;
            }
            """, options: nil)
        let fd = MTLRenderPipelineDescriptor()
        fd.vertexFunction = fill.makeFunction(name: "fill_vs")
        fd.fragmentFunction = fill.makeFunction(name: "fill_fs")
        fd.colorAttachments[0].pixelFormat = .rgba8Unorm
        fd.depthAttachmentPixelFormat = .depth32Float
        let fillPipe = try! ctx.device.makeRenderPipelineState(descriptor: fd)
        let proj = simd_float4x4(SIMD4(1.0 / 1.2, 0, 0, 0), SIMD4(0, 1.428, 0, 0), SIMD4(0, 0, 0, -1), SIMD4(0, 0, 0.05, 0))
        var f = frame
        f.invViewProj = proj.inverse
        func timeSky(quarter: Bool) -> Double {
            // The same frame all along: forget the last round's quarter-resolution sky so this one works out every pixel.
            if quarter { _ = prepareView(color: color, invViewProj: f.invViewProj, headroom: 1) } else { lowresView = nil }
            var h = colorHandle
            var clear: [Float] = [0, 0, 0, 0]
            guard mmc_pass_begin(&h, 1, 0, &clear, depthHandle, 0, 0, 0, 0, Int32(width), Int32(height)) == 1 else { return -1 }
            _ = draw(invViewProj: f.invViewProj, headroom: 1)
            mmc_pass_end()
            guard let cb = ctx.cb else { return -1 }
            ctx.cb = nil
            cb.commit()
            cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000
        }
        // The anti-aliasing's inputs: the same projection unjittered, a still camera.
        var taaMatrices = [Float](repeating: 0, count: 32)
        for c in 0..<4 {
            for r in 0..<4 {
                taaMatrices[c * 4 + r] = proj[c][r]
                taaMatrices[16 + c * 4 + r] = c == r ? 1 : 0
            }
        }
        var taaCam: [Double] = [0, 150, 0]
        func timeTaa(sky: Bool) -> Double {
            if sky { _ = aerial(color: color, depth: depth, invViewProj: f.invViewProj, headroom: 1, deferToTaa: true) }
            guard mmc_taa_apply(colorHandle, depthHandle, &taaMatrices, &taaCam, 0, 0, 0) == 1, let cb = ctx.cb else { return -1 }
            ctx.cb = nil
            cb.commit()
            cb.waitUntilCompleted()
            return (cb.gpuEndTime - cb.gpuStartTime) * 1000
        }
        var t0: [Double] = [], t1: [Double] = [], t2: [Double] = [], t3: [Double] = [], t4: [Double] = [], t5: [Double] = []
        // The first 10 rounds only warm the GPU's clocks up; the rest are timed.
        for round in 0..<(runs + 10) {
            if round == 10 { t0 = []; t1 = []; t2 = []; t3 = []; t4 = []; t5 = [] }
            let cb = ctx.queue.makeCommandBuffer()!
            let e = cb.makeComputeCommandEncoder()!
            e.setBytes(&f, length: MemoryLayout<SkyFrameGPU>.stride, index: 0)
            e.setComputePipelineState(viewPipe)
            e.setTexture(skyView, index: 0)
            e.setTexture(transmittance, index: 1)
            e.setTexture(multiScatter, index: 2)
            e.dispatchThreads(MTLSize(width: 192, height: 108, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
            e.setComputePipelineState(aerialPipe)
            e.setTexture(apScatter, index: 0)
            e.setTexture(apTrans, index: 1)
            e.setTexture(transmittance, index: 2)
            e.setTexture(multiScatter, index: 3)
            e.dispatchThreads(MTLSize(width: 32, height: 64, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
            e.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            t0.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)

            let cb2 = ctx.queue.makeCommandBuffer()!
            let pd = MTLRenderPassDescriptor()
            pd.colorAttachments[0].texture = color
            pd.colorAttachments[0].loadAction = .clear
            pd.colorAttachments[0].storeAction = .store
            pd.depthAttachment.texture = depth
            pd.depthAttachment.loadAction = .clear
            pd.depthAttachment.clearDepth = 0
            pd.depthAttachment.storeAction = .store
            let r = cb2.makeRenderCommandEncoder(descriptor: pd)!
            r.setRenderPipelineState(fillPipe)
            r.setDepthStencilState(ctx.depthState(compare: .always, write: true))
            var size = SIMD2<Float>(Float(width), Float(height))
            r.setFragmentBytes(&size, length: 8, index: 0)
            r.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            r.endEncoding()
            cb2.commit()

            t1.append(timeSky(quarter: false))

            // The aerial perspective pass as the game encodes it (ctx's command buffer, its own render pass).
            guard aerial(color: color, depth: depth, invViewProj: f.invViewProj, headroom: 1, deferToTaa: false), let cb4 = ctx.cb else { return false }
            ctx.cb = nil
            cb4.commit()
            cb4.waitUntilCompleted()
            t2.append((cb4.gpuEndTime - cb4.gpuStartTime) * 1000)

            t3.append(timeSky(quarter: true))
            t4.append(timeTaa(sky: false))
            t5.append(timeTaa(sky: true))
        }
        func median(_ a: [Double]) -> Double { a.sorted()[a.count / 2] }
        out[0] = median(t0)
        out[1] = median(t1)
        out[2] = median(t2)
        out[3] = median(t3)
        out[4] = median(t4)
        out[5] = median(t5)
        return true
    }
}

private func smoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
    let t = min(max((x - e0) / (e1 - e0), 0), 1)
    return t * t * (3 - 2 * t)
}

/// The angle one pixel spans at the screen's center, for the sun disk's anti-aliased edge.
private func skyPixelAngle(_ inv: simd_float4x4, width: Float) -> Float {
    let a = inv * SIMD4<Float>(0, 0, 1, 1), b = inv * SIMD4<Float>(2 / max(width, 1), 0, 1, 1)
    let da = simd_normalize(SIMD3(a.x, a.y, a.z) / a.w), db = simd_normalize(SIMD3(b.x, b.y, b.z) / b.w)
    return simd_length(db - da)
}

private func skyMatrix(_ p: UnsafePointer<Float>, _ o: Int) -> simd_float4x4 {
    simd_float4x4(SIMD4(p[o], p[o + 1], p[o + 2], p[o + 3]), SIMD4(p[o + 4], p[o + 5], p[o + 6], p[o + 7]),
                  SIMD4(p[o + 8], p[o + 9], p[o + 10], p[o + 11]), SIMD4(p[o + 12], p[o + 13], p[o + 14], p[o + 15]))
}

/// 1 if the sky experiment is on (METALMC_EXP=sky).
@_cdecl("mmc_sky_enabled")
public func mmc_sky_enabled() -> Int32 { skyEnabled ? 1 : 0 }

/// Start of a frame with the sky on (GameRenderer.renderLevel, before any pass). p: vanilla's sun angle (radians), rain
/// brightness (1 clear, 0 full rain), the camera's height above sea level (blocks), render-distance fade start and end
/// (blocks; end <= start: none). Rebuilds the tables that changed. Returns 1 if the sky is ready to draw this frame.
@_cdecl("mmc_sky_prepare")
public func mmc_sky_prepare(_ p: UnsafePointer<Float>) -> Int32 {
    guard skyEnabled, ctx.pass == nil else { return 0 }
    Sky.shared.prepare(cb: ctx.ensureCB(), sunAngle: p[0], rainBrightness: p[1], height: p[2],
                       fadeStart: p[3], fadeEnd: p[4], headroom: Hdr.shared.headroom)
    return Sky.shared.ready ? 1 : 0
}

/// Before vanilla's sky pass opens (SkyRenderer.render): the sky at a quarter of the resolution for this frame's view.
/// color: the TextureBox handle of the target the sky goes into; p: projection[16] as the level is drawn, view
/// rotation[16]. Returns 1 if made (the sky pass then filters it up instead of working out every pixel).
@_cdecl("mmc_sky_view")
public func mmc_sky_view(_ colorHandle: Int64, _ p: UnsafePointer<Float>) -> Int32 {
    guard skyEnabled else { return 0 }
    return Sky.shared.prepareView(color: (from(colorHandle) as TextureBox).texture, invViewProj: (skyMatrix(p, 0) * skyMatrix(p, 16)).inverse,
                                  headroom: Hdr.shared.headroom) ? 1 : 0
}

/// Draws the sky into the open render pass (vanilla's sky pass). p: projection[16] (as the level is drawn), view
/// rotation[16], column-major. Leaves ctx.pipe nil so Java re-applies Minecraft's pipeline state. Returns 1 if drawn.
@_cdecl("mmc_sky_draw")
public func mmc_sky_draw(_ p: UnsafePointer<Float>) -> Int32 {
    guard skyEnabled else { return 0 }
    return Sky.shared.draw(invViewProj: (skyMatrix(p, 0) * skyMatrix(p, 16)).inverse, headroom: Hdr.shared.headroom) ? 1 : 0
}

/// Aerial perspective, the render-distance fade and the tone curve on the level drawn into `color` (TextureBox handles;
/// after the ray-traced shadows, before the anti-aliasing). p: projection[16] as drawn, view rotation[16]. taa != 0: the
/// anti-aliasing runs next this frame and applies it as it loads the color.
@_cdecl("mmc_sky_aerial")
public func mmc_sky_aerial(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>, _ taa: Int32) -> Int32 {
    guard skyEnabled else { return 0 }
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    return Sky.shared.aerial(color: color, depth: depth, invViewProj: (skyMatrix(p, 0) * skyMatrix(p, 16)).inverse,
                             headroom: Hdr.shared.headroom, deferToTaa: taa != 0) ? 1 : 0
}

/// Offline check (tools, no game): see Sky.debugRender. window: 4 floats (azimuth left, right, elevation top, bottom,
/// radians); out: w * h * 4 bytes, lin: w * h * 4 floats, times: 5 doubles.
@_cdecl("mmc_debug_sky_render")
public func mmc_debug_sky_render(_ sunAngle: Float, _ rainBrightness: Float, _ camY: Double, _ headroom: Float, _ window: UnsafePointer<Float>,
                                 _ w: Int32, _ h: Int32, _ out: UnsafeMutablePointer<UInt8>, _ lin: UnsafeMutablePointer<Float>,
                                 _ times: UnsafeMutablePointer<Double>) -> Int32 {
    Sky.shared.debugRender(sunAngle: sunAngle, rainBrightness: rainBrightness, camY: camY, headroom: headroom,
                           window: SIMD4(window[0], window[1], window[2], window[3]),
                           width: Int(w), height: Int(h), out: out, lin: lin, times: times) ? 1 : 0
}

/// Offline timing (tools, no game): see Sky.debugTimePasses. out: 3 doubles.
@_cdecl("mmc_debug_sky_time")
public func mmc_debug_sky_time(_ w: Int32, _ h: Int32, _ runs: Int32, _ out: UnsafeMutablePointer<Double>) -> Int32 {
    Sky.shared.debugTimePasses(width: Int(w), height: Int(h), runs: Int(max(runs, 1)), out: out) ? 1 : 0
}
