import Foundation
import Metal
import simd

// Volumetric light from the traced shadows (METALMC_EXP=godrays, with lit, rtshadows, sky and post; off by default;
// docs/lighting-design.md, "Volumetric light"). A thin height fog lit by the sun, with real shadows in the air: beams under
// the canopy, through cave mouths and between hills, with the sun anywhere (behind the camera too), and the haze that
// thickens in the morning, at dusk and in rain.
//
// The fog. Its density is a layer over the land: the same inside it, half that at its top (sea level), falling off by e
// every few dozen blocks above (fewer in the morning, when the haze lies low), and gone below a floor (no fog in deep
// caves). Each edge is a logistic in height, whose integral along a straight ray has a closed form, so the optical depth
// to any point is exact and cheap, and so is its inverse. Light scattered once toward the eye along a ray,
//
//     L = albedo x integral of sigma(t) T(t) [phase x sun x V(t) + sky x A(t)] dt,
//
// and the integral of sigma T is (1 - T(end)) whatever the density's shape, so with
//
//     r = integral of sigma T V dt / (1 - T(end))   (the share of the fog's scattering the sun reaches)
//
// it's L = albedo x (1 - T) x (phase x sun x r + sky x A). Only r needs rays.
//
// Per frame, after the sun's shadow rays (RtShadows.trace calls encode, with its instance structure, depth and projection):
// - godrays_setup (one threadgroup): this frame's fog and light into a small buffer (GodraysShared): the sun's light
//   through the atmosphere's transmittance table (or the moon's at night), the sky's light the fog scatters (the sky view
//   table, and the ground's bounce), the density for the time of day and the rain. Both kernels and the anti-aliasing
//   read it, so the look's constants live in one place (godrays.metal, editable in lab mode).
// - godrays_table: the transmittance T from the camera (per channel) over 128 directions' heights x 64 distances, from
//   the closed form, for the resolve.
// - godrays_trace, one thread per 4 x 4 pixels (the ray through the block's pixel (2, 2)): GODRAYS_STEPS points along the
//   view ray, importance-sampled by sigma T (the closed-form inverse: every point carries the same weight), a different
//   stratified set each frame; at each, one ray toward a point on the sun's disk into RtShadows' structure (the LOD's
//   geometry out to its edge, the near terrain included) and, if it gets out, through the clouds (METALMC_EXP=clouds:
//   cloudTransmittanceToSun, their deep shadow map; beams under a broken deck come and go with its gaps). The mean is r;
//   with the end's sky light (lit mode's G-buffer: how much of the sky the ray's end sees) for the sky's share, the light
//   per unit of (1 - T): albedo x (phase x sun x r + sky x A). Then the history, reprojected from the ray's end and the
//   camera's motion and checked by distance, blended in (10% new when still): a quarter-resolution RGBA16Float pair.
// - The anti-aliasing's resolve (Taa.swift, its sky variant) applies it as it loads each pixel, after the aerial
//   perspective and the clouds (godraysApply): T per pixel from the table, the light per unit of (1 - T) from the four
//   texels around the pixel whose distance is like its own. A ratio upsamples cleanly across depth edges, where the light
//   itself wouldn't (a leaf 5 blocks off and the sky behind it differ a hundredfold in scattered light, not per unit of
//   1 - T), and the resolve's share is a table read, four texel reads and half-precision arithmetic.
//
// The fog is a medium of its own beside the atmosphere's (whose haze, in the aerial perspective, stays as it is and is
// a thousand times thinner near the ground): no light is counted twice. Its far end is the same for terrain and sky
// (GODRAYS_REACH, about where the planet's curve lifts a level ray out of the layer), so far terrain and the sky behind
// it fade into the same fog: no seam. It replaces the post chain's screen-space light shafts (Post.swift; their "lit air
// in front of terrain" is this, done in 3D), unless METALMC_GODRAYS_SHAFTS=1 keeps them for an A/B.
//
// Needs the post chain (the frame in scene-linear light) and the anti-aliasing (its resolve applies it): with either off it
// stays off.

/// METALMC_EXP=godrays: volumetric light and height fog (needs sky and post; meant for lit,rtshadows).
let godraysEnabled = experiments.contains("godrays") && skyEnabled && postEnabled
/// With godrays, the post chain's screen-space light shafts are off (this replaces them); METALMC_GODRAYS_SHAFTS=1 keeps them.
let godraysReplacesShafts = godraysEnabled && ProcessInfo.processInfo.environment["METALMC_GODRAYS_SHAFTS"] != "1"

private let godraysEnv = ProcessInfo.processInfo.environment
/// METALMC_GODRAYSVIEW: debug views (1 the fog's light alone, 2 the traced share r, 3 the transmittance, 4 the sky's share).
private let godraysViewDefault = Int32(godraysEnv["METALMC_GODRAYSVIEW"] ?? "") ?? 0
/// METALMC_GODRAYS_DENSITY: scales the fog's density (1: the defaults below).
private let godraysDensityScale = max(0, Float(godraysEnv["METALMC_GODRAYS_DENSITY"] ?? "") ?? 1)

private func godraysFloat(_ x: Float) -> String { String(format: "%.6g", x) }

/// The fog's functions, shared by its kernels (godrays.metal) and the anti-aliasing's resolve (taa.metal), which applies the
/// fog as it loads each pixel. Needs skyShaderHeader before it. In lab mode a file of its own (godrays_header.metal).
let godraysHeader = """

// ---- Volumetric light (Godrays.swift, METALMC_EXP=godrays): the fog, shared by its kernels and the anti-aliasing ----

// This frame's fog and its light, made on the GPU by godrays_setup from the atmosphere's tables and godrays.metal's
// constants. Heights are world y, distances blocks.
struct GodraysShared {
    float4 light;     // rgb: the sun's (or the moon's) light on the fog through the air (scene units), w: 1 on, 0 off
    float4 ambient;   // rgb: the sky's and the ground's light the fog scatters toward the eye (per unit of scattering), w: its share left in shade
    float4 lightDir;  // xyz: toward the sun (or the moon), w: 1 for the moon
    float4 shape;     // x: extinction inside the layer (per block, green), y: the layer's top (half the inside's density there), z: the fall-off above it, w: its floor
    float4 shape2;    // x: the floor's fall-off, y: the camera's y, z: the farthest distance counted, w: the fog's albedo
    float4 tint;      // rgb: extinction per channel over green's, w: the debug view
    float4 phase;     // x: the forward lobe's g, y: the broad lobe's g, z: the forward lobe's share, w: 1 / log2(1 + the farthest distance / 4) (the table's rows)
    float4 cam;       // x: softplus((top - camera y) / fall-off), y: the same for the floor, z-w: the render distance's fade (start, end; equal: none)
};

// The distance the fog counts to a surface at camera-relative rel (len = its length): its own, out to the farthest counted
// (shape2.z), and as far as the sky's where the aerial perspective fades it into the sky (vanilla's render distance, as
// skyFadeAmount measures it: cylindrical), so faded terrain and the sky behind it get the same fog. G: a pointer to the
// fog's frame, as the kernels read it (device) or the resolve (constant).
template <typename G>
static float godraysDistance(G g, float3 rel, float len) {
    float reach = g->shape2.z;
    float dist = min(len, reach);
    if (g->cam.w > g->cam.z) {
        float c = max(length(rel.xz), abs(rel.y));
        dist = mix(dist, reach, saturate((c - g->cam.z) / (g->cam.w - g->cam.z)));
    }
    return dist;
}

static float godraysSoftplus(float x) { return max(x, 0.0) + log(1.0 + exp(-abs(x))); }

// The density is shape.x * (1 / (1 + exp((y - top) / falloff)) - 1 / (1 + exp((y - floor) / floorFalloff))): the inside's
// below the top, falling off by e every `falloff` blocks above it, gone below the floor. Each term integrates in closed form
// along a straight ray (the integral of 1 / (1 + e^x) is -softplus(-x)). One term over [0, t] along y(s) = y0 + dy s; sp0
// is softplus((edge - y0) / h), the camera's (per frame). Near-level stretches take the midpoint (no cancellation).
static float godraysEdge(float y0, float dy, float t, float edge, float h, float sp0) {
    float rise = dy * t;
    if (abs(rise) < 0.02 * h) return t / (1.0 + exp((y0 + 0.5 * rise - edge) / h));
    return h / dy * (sp0 - godraysSoftplus((edge - y0 - rise) / h));
}

// Optical depth (green) from the camera along a direction whose y is dy, over t blocks.
static float godraysTau(device const GodraysShared* g, float dy, float t) {
    float a = godraysEdge(g->shape2.y, dy, t, g->shape.y, g->shape.z, g->cam.x);
    float b = godraysEdge(g->shape2.y, dy, t, g->shape.w, g->shape2.x, g->cam.y);
    return g->shape.x * max(a - b, 0.0);
}

// Henyey-Greenstein, and the fog's phase function: a forward lobe (the glow around the sun) and a broad one leaning back
// (what lets beams show with the sun behind the camera).
static float godraysHG(float g, float c) {
    float k = max(1.0 + g * g - 2.0 * g * c, 1e-4);
    return (1.0 - g * g) / (12.5663706 * k * sqrt(k));
}
static float godraysPhase(device const GodraysShared* g, float c) {
    return mix(godraysHG(g->phase.y, c), godraysHG(g->phase.x, c), g->phase.z);
}

// The transmittance table (godrays_table, every frame): T per channel from the camera along a direction whose y is dy,
// over a distance, at 128 x 64 texels. Columns: sign(dy) sqrt(|dy|) (dense near the horizontal, where it changes fastest
// with height); rows: log2(1 + distance / 4) over the farthest distance's (15% farther a row). One bilinear read stands in
// for the closed form's six transcendentals per pixel in the resolve.
#define GODRAYS_TABLE_W 128
#define GODRAYS_TABLE_H 64
template <typename G>
static float2 godraysTableUV(G g, float dy, float dist) {
    float u = 0.5 + 0.5 * sign(dy) * sqrt(abs(dy));
    float v = log2(1.0 + dist * 0.25) * g->phase.w;
    return float2((u * (GODRAYS_TABLE_W - 1) + 0.5) / GODRAYS_TABLE_W, (saturate(v) * (GODRAYS_TABLE_H - 1) + 0.5) / GODRAYS_TABLE_H);
}

// The traced texel for full-resolution pixel q: the light the fog scatters toward the eye per unit of (1 - T) (rgb: the
// sun's through the phase function times the share of the way it reaches, and the sky's). Quarter-resolution texel i holds
// the ray through pixel 4 i + 2, its distance in w; of the four around q, those whose distance is like q's count, bilinear
// among them, so a leaf against the sky takes the leaf's and the sky beside it the sky's. If none is like it (a twig
// thinner than a texel), all four. Most pixels have all four at about their own distance: one bilinear read then.
static half3 godraysTexel(texture2d<float> fog, uint2 q, float dist) {
    int2 size = int2(fog.get_width(), fog.get_height());
    float2 c = (float2(q) - 2.0) * 0.25;
    float inv = 16.0 / max(dist, 1.0);
    constexpr sampler sl(filter::linear, address::clamp_to_edge);
    float2 uv = (c + 0.5) / float2(size);
    if (all(abs(fog.gather(sl, uv, int2(0), component::w) - dist) * inv < 0.25)) return half3(fog.sample(sl, uv, level(0.0)).rgb);
    int2 i0 = int2(floor(c));
    float2 fr = c - float2(i0);
    float3 acc = 0.0, accAll = 0.0;
    float wsum = 0.0, wall = 0.0;
    for (int k = 0; k < 4; k++) {
        int2 o = int2(k & 1, k >> 1);
        float4 v = fog.read(uint2(clamp(i0 + o, int2(0), size - 1)));
        float wb = (o.x == 1 ? fr.x : 1.0 - fr.x) * (o.y == 1 ? fr.y : 1.0 - fr.y);
        float e = (v.w - dist) * inv;
        float wd = wb / (1.0 + e * e);
        acc += v.rgb * wd;
        wsum += wd;
        accAll += v.rgb * wb;
        wall += wb;
    }
    return half3(wsum > 0.02 * wall ? acc / wsum : accAll / max(wall, 1e-6));
}

// sRGB's encoding and back (skyEncode and skyDecode) in half precision, for the composite: the frame holds 6-bit mantissas
// (RG11B10Float) or 10 (RGBA16Float), and half's transcendentals are cheaper. The sun's disk (about 35,000 decoded) stays
// under half's largest.
static half3 godraysDecode(half3 c) {
    half3 a = min(abs(c), half3(90.0h));
    return sign(c) * select(pow((a + 0.055h) * (1.0h / 1.055h), half3(2.4h)), a * (1.0h / 12.92h), a <= 0.04045h);
}
static half3 godraysEncode(half3 c) {
    half3 a = abs(c);
    return sign(c) * select(1.055h * pow(a, half3(1.0h / 2.4h)) - 0.055h, 12.92h * a, a <= 0.0031308h);
}

// The frame's light at pixel q (as the frame holds it: sRGB-encoded scene-linear light, the post chain's) at depth d (0:
// the sky) through the fog: what the fog lets through (the table), plus what it scatters toward the eye on the way (the
// traced texels' light times 1 - T). Called by the anti-aliasing's resolve as it loads each pixel, after the aerial
// perspective (skyLevelColor) and the clouds: the fog is in front of both. The fog's frame in the constant address space
// (the same for every pixel).
static float3 godraysApply(float3 c, uint2 q, float d, constant SkyFrame& f, texture2d<float> fog, texture2d<float> table,
                           constant GodraysShared* g) {
    if (g->light.w < 0.5) return c;
    float2 ndc = (float2(q) + 0.5) / f.view.zw * 2.0 - 1.0;
    float4 h = f.invViewProj * float4(ndc, d > 0.0 ? d : 1.0, 1.0);
    float3 rel = h.xyz / h.w;
    float len = length(rel);
    float dy = rel.y / max(len, 1e-6);
    float dist = d > 0.0 ? godraysDistance(g, rel, len) : g->shape2.z;
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    half3 T = half3(table.sample(s, godraysTableUV(g, dy, dist), level(0.0)).rgb);
    uint view = uint(g->tint.w);
    if (view == 0u && all(T > 0.9995h)) return c;
    half3 S = godraysTexel(fog, q, dist);
    if (view == 0u) return float3(godraysEncode(godraysDecode(half3(c)) * T + S * (1.0h - T)));
    // Debug views (GODRAYS_VIEW): 1 the fog's light alone, 3 the transmittance; 2 and 4 the texels as they are (the trace
    // writes the traced share or the sky's into them).
    return skyEncode(float3(view == 1u ? S * (1.0h - T) : (view == 3u ? T : S)));
}
"""

// The clouds (METALMC_EXP=clouds, Clouds.swift): their deep shadow map and frame at the trace's buffer 3 and texture 4, and
// how much of the light gets through them to a point of the fog (cloudTransmittanceToSun, when their light is the fog's).
private let godraysCloudArgs = cloudsEnabled ? "\n                          constant CloudFrame& cf [[buffer(3)]], texture2d<float> cloudShadow [[texture(4)]]," : ""
private let godraysCloudCall = cloudsEnabled ? "clouds ? cloudTransmittanceToSun(x, cf, cloudShadow) : 1.0" : "1.0"
private let godraysCloudCheck = cloudsEnabled ? "\n        bool clouds = dot(cf.light.xyz, ld) > 0.99;   // the clouds' shadow map is toward the same light" : ""

private let godraysSource = skyShaderHeader + cloudsShaderHeader + """

#include <metal_raytracing>
using namespace raytracing;
""" + godraysHeader + """

// ---- The look's constants (lab mode: edit them here; godrays_setup turns them into each frame's fog) ----

// Density: extinction per block at the layer's top at noon in clear weather (visibility about 3.9 / this), and how many
// times that with the sun low in the east (the morning's mist: dense and low, where the beams show), low in the west
// (evening: a little, so a sunset's far shore stays a silhouette), at night, and in rain. The morning and evening's fade as
// the sun climbs from GODRAYS_LOW_FROM to GODRAYS_LOW_UP (sines of its height): the mist lifts by mid-morning.
#define GODRAYS_DENSITY \(godraysFloat(6e-5 * godraysDensityScale))
#define GODRAYS_MORNING 60.0
#define GODRAYS_EVENING 1.5
#define GODRAYS_NIGHT 6.0
#define GODRAYS_RAIN 25.0
#define GODRAYS_LOW_FROM 0.3
#define GODRAYS_LOW_UP 1.0
// Shape: the layer's top (world y: half the inside's density there), the fall-off above it (blocks per e) by day, with
// the morning's mist (low: it fills the lowlands and leaves the hills), in rain (it fills the air), and the floor below
// which there's none (caves).
#define GODRAYS_TOP 68.0
#define GODRAYS_FALLOFF 40.0
#define GODRAYS_FALLOFF_MORNING 10.0
#define GODRAYS_FALLOFF_RAIN 120.0
#define GODRAYS_FLOOR 30.0
#define GODRAYS_FLOOR_FALLOFF 4.0
// The farthest distance counted (blocks), for terrain and the sky alike: about where the planet's curve lifts a level ray
// out of the layer.
#define GODRAYS_REACH 32000.0
// Scattering: albedo, extinction per channel over green's (small particles take more blue: the far haze bluish, what's
// behind it warmer), the phase function's lobes (forward g, broad g, the forward one's share).
#define GODRAYS_ALBEDO 0.9
#define GODRAYS_TINT float3(0.65, 1.0, 1.5)
#define GODRAYS_G_FORWARD 0.7
#define GODRAYS_G_BROAD (-0.2)
#define GODRAYS_FORWARD 0.5
// The sky's light in the fog: its strength, the share left where the sun doesn't reach (shade under trees), the ground's
// albedo for the light it sends up, the moon's light at full night (scene units).
#define GODRAYS_AMBIENT 1.0
#define GODRAYS_SHADE_AMBIENT 0.4
#define GODRAYS_GROUND 0.1
#define GODRAYS_MOON 0.3
// Rays per traced texel per frame, the history's blend (the share of the new frame when still), the rays' length, and the
// least sunlight a ray's fog can scatter toward the eye (scene units) that is worth rays: below it none (r 1), an error
// of at most that much.
#define GODRAYS_STEPS 2
#define GODRAYS_BLEND 0.1
#define GODRAYS_RAY 4000.0
#define GODRAYS_MIN_LIGHT 0.0003
// Debug view (METALMC_GODRAYSVIEW): 1 the fog's light alone, 2 the traced share (the sun's), 3 the transmittance, 4 the
// sky's share.
#define GODRAYS_VIEW \(godraysViewDefault)

// This frame's fog (one threadgroup of 64): the sky's light over the upper hemisphere (256 directions, uniform in solid
// angle) for the light the fog scatters from it, the sun's through the air, the density for the time of day and the rain.
// p: x the camera's world y, y a debug view to use instead of GODRAYS_VIEW (-1: none).
kernel void godrays_setup(device GodraysShared* out [[buffer(0)]], constant SkyFrame& f [[buffer(1)]], constant float4& p [[buffer(2)]],
                          texture2d<float> trans [[texture(0)]], texture2d<float> skyView [[texture(1)]],
                          uint i [[thread_index_in_threadgroup]]) {
    threadgroup float4 acc[64][2];
    float3 s0 = 0.0, s1 = 0.0;
    for (uint k = 0; k < 4; k++) {
        uint j = i * 4u + k;
        float y = (float(j) + 0.5) / 256.0;
        float r = sqrt(max(0.0, 1.0 - y * y));
        float phi = float(j) * 2.39996323;
        float3 L = skyLuminance(f, float3(r * cos(phi), y, r * sin(phi)), skyView);
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
    float3 skyMean = acc[0][0].rgb / 256.0;                 // mean radiance of the upper hemisphere
    float3 skyDown = acc[0][1].rgb / 256.0 * 2.0 * SKY_PI;  // what it puts on a level surface
    float3 sun = f.sun.xyz;
    float r0 = f.view.y;
    float3 sunLight = skyTransmittanceToSpace(trans, f.atmo, r0, sun.y) * skySunVisible(f.atmo, r0, sun.y, f.sunHoriz.z) * f.sun.w;
    float night = smoothstep(0.05, -0.25, sun.y);
    // Toward the moon while the sun is down (as lit mode's shadow rays).
    bool moon = sun.y < -0.02;
    float3 light = moon ? GODRAYS_MOON * night * float3(0.55, 0.7, 1.0) : sunLight;
    float3 ld = moon ? -sun : sun;
    // The ground's light (its albedo over pi, times what falls on it) from below, the sky's from above: isotropic, so
    // each half of the sphere sends half its mean radiance.
    float3 ground = GODRAYS_GROUND / SKY_PI * (light * max(ld.y, 0.0) + skyDown);
    float3 ambient = GODRAYS_AMBIENT * 0.5 * (skyMean + ground);
    // Density: the morning's haze with the sun low in the east (vanilla's sun rises at +x), the evening's in the west, the
    // night's mist, the rain's.
    float low = 1.0 - smoothstep(GODRAYS_LOW_FROM, GODRAYS_LOW_UP, sun.y);
    float east = smoothstep(-0.15, 0.15, sun.x);
    float day = 1.0 - night;
    float rain = f.aerial.z;
    float m = 1.0 + day * low * mix(GODRAYS_EVENING - 1.0, GODRAYS_MORNING - 1.0, east) + night * (GODRAYS_NIGHT - 1.0);
    m *= 1.0 + (GODRAYS_RAIN - 1.0) * rain;
    float falloff = mix(mix(GODRAYS_FALLOFF, GODRAYS_FALLOFF_MORNING, day * low * east), GODRAYS_FALLOFF_RAIN, rain);
    float camY = p.x;
    out->light = float4(light, 1.0);
    out->ambient = float4(ambient, GODRAYS_SHADE_AMBIENT);
    out->lightDir = float4(ld, moon ? 1.0 : 0.0);
    out->shape = float4(2.0 * GODRAYS_DENSITY * m, GODRAYS_TOP, falloff, GODRAYS_FLOOR);
    out->shape2 = float4(GODRAYS_FLOOR_FALLOFF, camY, GODRAYS_REACH, GODRAYS_ALBEDO);
    out->tint = float4(GODRAYS_TINT, p.y >= 0.0 ? p.y : float(GODRAYS_VIEW));
    out->phase = float4(GODRAYS_G_FORWARD, GODRAYS_G_BROAD, GODRAYS_FORWARD, 1.0 / log2(1.0 + GODRAYS_REACH * 0.25));
    out->cam = float4(godraysSoftplus((GODRAYS_TOP - camY) / falloff), godraysSoftplus((GODRAYS_FLOOR - camY) / GODRAYS_FLOOR_FALLOFF),
                      f.fade.z > 0.0 ? f.fade.x : 0.0, f.fade.z > 0.0 ? f.fade.y : 0.0);
}

// This frame's transmittance table (after godrays_setup; godraysTableUV's layout): the closed form at each texel's
// direction and distance, per channel.
kernel void godrays_table(device const GodraysShared* g [[buffer(0)]], texture2d<float, access::write> out [[texture(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(GODRAYS_TABLE_W) || gid.y >= uint(GODRAYS_TABLE_H)) return;
    float s = float(gid.x) / float(GODRAYS_TABLE_W - 1) * 2.0 - 1.0;   // sign(dy) sqrt(|dy|)
    float dy = s * abs(s);
    float v = float(gid.y) / float(GODRAYS_TABLE_H - 1);
    float dist = (exp2(v / g->phase.w) - 1.0) * 4.0;
    out.write(float4(exp(-godraysTau(g, dy, dist) * g->tint.rgb), 1.0), gid);
}

struct GodraysParams {
    float4x4 invViewProj;    // this frame's (jittered) projection * view rotation, inverted: as the level was drawn
    float4x4 prevViewProj;   // last frame's, for the history
    float4 sizes;            // full width, full height, traced width, traced height
    float4 camOffset;        // xyz: the camera relative to the instance structure's origin
    float4 camDelta;         // xyz: this frame's camera minus last frame's, w: 1 to start the history over
    float4 camWorld;         // xyz: the camera's world position (float: for the clouds' hook), w: unused
    uint4 frame;             // x: frame number, y: 1 if lit mode's G-buffer is bound
};

// Where along the ray (blocks) the top term's integral reaches `a` (in blocks of the inside's density): its inverse, for
// the importance sampling. (The floor's term matters only deep down, where no light reaches.)
static float godraysInvert(device const GodraysShared* g, float dy, float a) {
    float top = g->shape.y, h = g->shape.z, y0 = g->shape2.y;
    if (abs(dy) < 1e-4) return a * (1.0 + exp((y0 - top) / h));
    float k = g->cam.x - a * dy / h;   // softplus((top - y) / h) at the point
    if (k <= 1e-7) return 1e30;        // past the whole layer (an upward ray)
    float om = k < 1e-3 ? k * (1.0 - 0.5 * k) : 1.0 - exp(-k);
    float y1 = top - h * (k + log(om));   // softplus's inverse: log(e^k - 1)
    return max((y1 - y0) / dy, 0.0);
}

static float godraysRand(uint2 q, uint f, uint k) {
    uint h = (q.x * 0x8da6b343u) ^ (q.y * 0xd8163841u) ^ ((f * 8u + k) * 0xcb1ab31fu);
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return float(h >> 8) * (1.0 / 16777216.0);
}

// One thread per 4 x 4 pixels: the ray through the block's pixel (2, 2) to its depth (the sky: GODRAYS_REACH), the share
// of the fog's scattering along it that the light reaches (GODRAYS_STEPS rays toward the sun's disk from points spread by
// sigma T, each through the clouds too), how much of the sky its end sees, the light the fog scatters toward the eye per
// unit of (1 - T) from both, and the history blended in. out: rgb that light, w the distance.
kernel void godrays_trace(instance_acceleration_structure accel [[buffer(0)]],
                          constant GodraysParams& p [[buffer(1)]],
                          device const GodraysShared* g [[buffer(2)]],\(godraysCloudArgs)
                          depth2d<float, access::read> depth [[texture(0)]],
                          texture2d<float, access::read> history [[texture(1)]],
                          texture2d<float, access::write> out [[texture(2)]],
                          texture2d<uint, access::read> gbuf [[texture(3)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(p.sizes.z) || gid.y >= uint(p.sizes.w)) return;
    uint2 full = uint2(p.sizes.xy);
    uint2 fp = min(gid * 4u + 2u, full - 1u);
    float d = depth.read(fp);
    float2 ndc = (float2(fp) + 0.5) / p.sizes.xy * 2.0 - 1.0;
    float4 h = p.invViewProj * float4(ndc, d > 0.0 ? d : 1.0, 1.0);
    float3 rel = h.xyz / h.w;
    float len = length(rel);
    float3 dir = rel / max(len, 1e-6);
    float reach = g->shape2.z;
    float dist = d > 0.0 ? godraysDistance(g, rel, len) : reach;
    // How much of the sky the ray's end sees: its sky light (lit mode's G-buffer, terrain or water whose depth key matches),
    // squared; 1 for the sky and anything else.
    float amb = 1.0;
    if (d > 0.0 && p.frame.y != 0u) {
        uint2 gb = gbuf.read(fp).rg;
        uint dk = (((as_type<uint>(d) >> 4) & 0xFFFFu) - (gb.y & 0xFFFFu)) & 0xFFFFu;
        if ((gb.x >> 28) != 0u && (dk <= 1u || dk == 0xFFFFu)) {
            float s = float((gb.y >> 16) & 255u) / 240.0;
            amb = s * s;
        }
    }
    // The light along the ray: points spread by the fog's scattering (each the same weight), stratified, a new set each
    // frame (interleaved gradient noise in space, the golden ratio in time). open: the share of them the terrain leaves in
    // the light (the sky's light is less in its shade: under trees, in a cave mouth); vis: that through the clouds too.
    float open = 1.0, vis = 1.0;
    float span = 1.0 - exp(-g->shape.x * godraysEdge(g->shape2.y, dir.y, dist, g->shape.y, g->shape.z, g->cam.x));
    float3 ld = g->lightDir.xyz;
    float ph = godraysPhase(g, dot(dir, ld));
    float most = span * g->shape2.w * ph * max(g->light.r, max(g->light.g, g->light.b));
    if (most > GODRAYS_MIN_LIGHT) {
        float3 t1 = normalize(cross(ld, float3(0.0, 0.0, 1.0))), t2 = cross(ld, t1);
        float xi = fract(52.9829189 * fract(dot(float2(gid), float2(0.06711056, 0.00583715))) + float(p.frame.x % 4096u) * 0.618034);
        float lit = 0.0, through = 0.0;\(godraysCloudCheck)
        intersector<instancing> isect;
        isect.accept_any_intersection(true);
        isect.assume_geometry_type(geometry_type::triangle);
        for (uint i = 0; i < GODRAYS_STEPS; i++) {
            float u = (float(i) + xi) / float(GODRAYS_STEPS);
            float a = -log(max(1.0 - u * span, 1e-30)) / g->shape.x;
            float t = min(godraysInvert(g, dir.y, a), dist * 0.999 - 0.05);
            float3 x = dir * max(t, 0.0);
            // A point on the sun's disk (radius about 0.6 degrees, as the shadows'): soft edges once accumulated.
            float rr = 0.0105 * sqrt(godraysRand(gid, p.frame.x, 2u * i)), ang = 6.2831853 * godraysRand(gid, p.frame.x, 2u * i + 1u);
            ray r;
            r.origin = x + p.camOffset.xyz;
            r.direction = normalize(ld + t1 * (rr * cos(ang)) + t2 * (rr * sin(ang)));
            r.min_distance = 0.0;
            r.max_distance = GODRAYS_RAY;
            auto hit = isect.intersect(r, accel, 0xFF);
            if (hit.type == intersection_type::none) {
                lit += 1.0;
                through += \(godraysCloudCall);
            }
        }
        open = lit / float(GODRAYS_STEPS);
        vis = through / float(GODRAYS_STEPS);
    }
    float3 S = g->shape2.w * (g->light.rgb * (ph * vis) + g->ambient.rgb * (amb * mix(g->ambient.w, 1.0, open)));
    // Debug views 2 and 4 (the composite shows the texels as they are): the traced share, the sky's.
    uint view = uint(g->tint.w);
    if (view == 2u) S = float3(vis);
    else if (view == 4u) S = float3(amb);
    float4 res = float4(S, dist);
    // The history: where this ray's end was last frame (the sky: only the camera's rotation moves it), its four texels whose
    // distance matches what this one's would have been, bilinear. (Every texel is written on the frame that starts it over.)
    if (p.camDelta.w < 0.5) {
        float3 prev = d > 0.0 ? rel + p.camDelta.xyz : dir;
        float4 pc = p.prevViewProj * float4(prev, d > 0.0 ? 1.0 : 0.0);
        if (pc.w > 1e-6) {
            float2 puv = pc.xy / pc.w * 0.5 + 0.5;
            if (all(puv > 0.0) && all(puv < 1.0)) {
                float expect = d > 0.0 ? godraysDistance(g, prev, length(prev)) : reach;
                float2 c = (puv * p.sizes.xy - 2.5) * 0.25;
                int2 i0 = int2(floor(c));
                float2 fr = c - float2(i0);
                int2 hs = int2(history.get_width(), history.get_height());
                float3 acc = 0.0;
                float wsum = 0.0;
                for (int k = 0; k < 4; k++) {
                    int2 o = int2(k & 1, k >> 1);
                    float4 hv = history.read(uint2(clamp(i0 + o, int2(0), hs - 1)));
                    float wb = (o.x == 1 ? fr.x : 1.0 - fr.x) * (o.y == 1 ? fr.y : 1.0 - fr.y);
                    float w = abs(hv.w - expect) < 0.1 * max(expect, 1.0) ? wb : 0.0;
                    acc += hv.rgb * w;
                    wsum += w;
                }
                if (wsum > 0.05) {
                    float motion = length((puv - (float2(fp) + 0.5) / p.sizes.xy) * p.sizes.zw);   // in traced texels
                    res.rgb = mix(acc / wsum, res.rgb, mix(GODRAYS_BLEND, 0.35, saturate(motion / 4.0)));
                }
            }
        }
    }
    out.write(res, gid);
}
"""

/// Mirrors GodraysParams in the shader.
private struct GodraysParamsGPU {
    var invViewProj = matrix_identity_float4x4
    var prevViewProj = matrix_identity_float4x4
    var sizes = SIMD4<Float>.zero
    var camOffset = SIMD4<Float>.zero
    var camDelta = SIMD4<Float>.zero
    var camWorld = SIMD4<Float>.zero
    var frame = SIMD4<UInt32>.zero
}

/// GodraysShared's size (8 float4).
private let godraysSharedSize = 8 * 16

final class Godrays: @unchecked Sendable {
    static let shared = Godrays()

    private var library: MTLLibrary?
    private var setupPipe: MTLComputePipelineState?
    private var tablePipe: MTLComputePipelineState?
    private var tracePipe: MTLComputePipelineState?
    private var failed = false
    /// The fog's light per unit of (1 - T) and the distance per quarter-resolution texel, ping-ponged (read one, write the other).
    private var history: [MTLTexture] = []
    private var current = 0
    /// The transmittance table (godrays_table, every frame), for the resolve.
    private var table: MTLTexture?
    /// The clouds' stand-ins (METALMC_EXP=clouds, when they have no shadow map to bind).
    private var dummyShadow: MTLTexture?
    /// GodraysShared, written by godrays_setup each frame (shared storage: the log reads it a frame or two late).
    private var sharedBuf: MTLBuffer?
    /// A GodraysShared with light.w 0: what the anti-aliasing binds when there's no fog this frame (it leaves the frame alone).
    private var offBuf: MTLBuffer?
    private var dummyTex: MTLTexture?
    private var dummyGbuf: MTLTexture?
    private var prevViewProj = matrix_identity_float4x4
    private var prevCam = SIMD3<Double>(repeating: .nan)
    private var lastNanos: UInt64 = 0
    private var frame: UInt32 = 0
    private var traced = 0, applied = 0
    /// This frame's traced texels, for the anti-aliasing's resolve (taken once).
    private var pending: (texture: MTLTexture, width: Int, height: Int)?
    /// Offline A/B (mmc_debug_godrays): off traces nothing and the resolve leaves the frame alone.
    var on = true
    /// Offline timing (mmc_debug_godrays_mode 1): traced, but the resolve leaves the frame alone (the composite's cost apart).
    var apply = true
    /// Debug view to use instead of GODRAYS_VIEW (-1: none; mmc_debug_godrays).
    var view: Int32 = -1

    /// The next frame starts the history over (offline: a switch shouldn't blend in light from another time of day).
    func restart() { lastNanos = 0 }

    private func ensure() -> Bool {
        if failed { return false }
        if library != nil { return true }
        do {
            // Lab mode (ShaderLab.swift): after an edit of godrays.metal (or godrays_header.metal), the pipelines are built
            // again from the new library on the next frame (the history stays).
            let lib = try ShaderLab.library("godrays", godraysSource) { [self] _ in
                library = nil; setupPipe = nil; tablePipe = nil; tracePipe = nil; failed = false
            }
            setupPipe = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "godrays_setup")!)
            tablePipe = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "godrays_table")!)
            tracePipe = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "godrays_trace")!)
            library = lib
        } catch {
            log("godrays: shaders failed: \(error)")
            failed = true
            return false
        }
        if sharedBuf == nil { sharedBuf = ctx.device.makeBuffer(length: godraysSharedSize, options: .storageModeShared) }
        if table == nil {
            // GODRAYS_TABLE_W x GODRAYS_TABLE_H in the shader.
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 128, height: 64, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            table = ctx.device.makeTexture(descriptor: d)
            table?.label = "MetalMC volumetric light: transmittance"
        }
        return sharedBuf != nil && table != nil
    }

    private func standIns() -> (tex: MTLTexture, buf: MTLBuffer)? {
        if dummyTex == nil {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
            d.usage = .shaderRead
            dummyTex = ctx.device.makeTexture(descriptor: d)
            offBuf = ctx.device.makeBuffer(length: godraysSharedSize, options: .storageModeShared)
            if let b = offBuf { memset(b.contents(), 0, b.length) }
        }
        guard let t = dummyTex, let b = offBuf else { return nil }
        return (t, b)
    }

    /// RtShadows.trace, after the sun's shadow rays (only when the anti-aliasing runs this frame: its resolve applies the
    /// fog): this frame's fog and its traced texels, on the shadows' instance structure, depth and projection (as drawn,
    /// jittered). Leaves them for the resolve (bindTaa).
    func encode(cb: MTLCommandBuffer, depth: MTLTexture, invViewProj: simd_float4x4, cam: SIMD3<Double>, camOffset: SIMD3<Float>,
                tlas: MTLAccelerationStructure, accels: [MTLAccelerationStructure], width: Int, height: Int) {
        pending = nil
        let sky = Sky.shared
        guard on, ctx.device.supportsRaytracing, sky.ready, let trans = sky.transmittance, let skyView = sky.skyView,
              ensure(), let setupPipe, let tablePipe, let tracePipe, let sharedBuf, let table else { return }
        let qw = (width + 3) / 4, qh = (height + 3) / 4
        var reset = false
        if history.count != 2 || history[0].width != qw || history[0].height != qh {
            history = (0..<2).compactMap { k in
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: qw, height: qh, mipmapped: false)
                d.usage = [.shaderRead, .shaderWrite]
                d.storageMode = .private
                let t = ctx.device.makeTexture(descriptor: d)
                t?.label = "MetalMC volumetric light \(k)"
                return t
            }
            guard history.count == 2 else { history = []; return }
            reset = true
            log("godrays: volumetric light traced at \(qw)x\(qh) (one ray per 4 x 4 pixels, its history blended), the screen-space light shafts \(godraysReplacesShafts ? "off" : "kept")")
        }
        // A new history after a jump (16 blocks in a frame: a teleport, a lab scene) or a gap (menus, a frame without it).
        let now = DispatchTime.now().uptimeNanoseconds
        let delta = cam - prevCam
        if !(simd_length(delta) < 16) || now &- lastNanos > 100_000_000 { reset = true }
        lastNanos = now
        if dummyGbuf == nil {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Uint, width: 1, height: 1, mipmapped: false)
            d.usage = .shaderRead
            dummyGbuf = ctx.device.makeTexture(descriptor: d)
        }
        let gbuf = litEnabled ? Lit.shared.gbuffer.flatMap { $0.width == width && $0.height == height ? $0 : nil } : nil
        guard let gtex = gbuf ?? dummyGbuf else { return }
        // The clouds' deep shadow map and frame (made this frame, before the shadows), or a frame that turns them off.
        var cloudFrame = CloudFrameGPU()
        var cloudShadow: MTLTexture?
        if cloudsEnabled {
            if let b = Clouds.shared.shadowBinding() {
                cloudFrame = b.frame
                cloudShadow = b.shadow
            } else {
                if dummyShadow == nil {
                    let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
                    d.usage = .shaderRead
                    dummyShadow = ctx.device.makeTexture(descriptor: d)
                }
                guard let dummyShadow else { return }
                cloudShadow = dummyShadow
            }
        }
        frame &+= 1
        var p = GodraysParamsGPU()
        p.invViewProj = invViewProj
        p.prevViewProj = prevViewProj
        p.sizes = SIMD4(Float(width), Float(height), Float(qw), Float(qh))
        p.camOffset = SIMD4(camOffset, 0)
        p.camDelta = reset ? SIMD4(0, 0, 0, 1) : SIMD4(Float(delta.x), Float(delta.y), Float(delta.z), 0)
        p.camWorld = SIMD4(Float(cam.x), Float(cam.y), Float(cam.z), 0)
        p.frame = SIMD4(frame, gbuf != nil ? 1 : 0, 0, 0)
        var frameSky = sky.frame
        var setup = SIMD4<Float>(Float(cam.y), Float(view), 0, 0)
        ctx.endBlit()
        guard let enc = cb.makeComputeCommandEncoder(descriptor: profComputePass("volumetric light (godrays)")) else { return }
        enc.label = "MetalMC volumetric light"
        enc.setComputePipelineState(setupPipe)
        enc.setBuffer(sharedBuf, offset: 0, index: 0)
        enc.setBytes(&frameSky, length: MemoryLayout<SkyFrameGPU>.stride, index: 1)
        enc.setBytes(&setup, length: 16, index: 2)
        enc.setTexture(trans, index: 0)
        enc.setTexture(skyView, index: 1)
        enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        enc.setComputePipelineState(tablePipe)
        enc.setBuffer(sharedBuf, offset: 0, index: 0)
        enc.setTexture(table, index: 0)
        enc.dispatchThreads(MTLSize(width: table.width, height: table.height, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 8, depth: 1))
        enc.setComputePipelineState(tracePipe)
        enc.setAccelerationStructure(tlas, bufferIndex: 0)
        enc.useResources(accels, usage: .read)
        enc.setBytes(&p, length: MemoryLayout<GodraysParamsGPU>.stride, index: 1)
        enc.setBuffer(sharedBuf, offset: 0, index: 2)
        if let cloudShadow {
            enc.setBytes(&cloudFrame, length: MemoryLayout<CloudFrameGPU>.stride, index: 3)
            enc.setTexture(cloudShadow, index: 4)
        }
        enc.setTexture(depth, index: 0)
        enc.setTexture(history[current], index: 1)
        enc.setTexture(history[1 - current], index: 2)
        enc.setTexture(gtex, index: 3)
        enc.dispatchThreads(MTLSize(width: qw, height: qh, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
        pending = (history[1 - current], width, height)
        current = 1 - current
        prevViewProj = invViewProj.inverse
        prevCam = cam
        traced += 1
        if traced % 1200 == 0 {
            // GodraysShared from a frame or two ago (the GPU writes it).
            let v = sharedBuf.contents().bindMemory(to: SIMD4<Float>.self, capacity: 8)
            log(String(format: "godrays: %d frames traced, %d applied; density %.2e per block inside the layer (top y %.0f, fall-off %.0f), light %.2f %.2f %.2f%@, sky's %.3f %.3f %.3f",
                       traced, applied, v[3].x, v[3].y, v[3].z, v[0].x, v[0].y, v[0].z, v[2].w > 0.5 ? " (the moon)" : "", v[1].x, v[1].y, v[1].z))
        }
    }

    /// The anti-aliasing's resolve (its sky variant, Taa.swift): this frame's traced texels at texture 24, the transmittance
    /// table at 25 and the fog at buffer 7, or stand-ins that leave the frame alone (no fog this frame). Taken once.
    func bindTaa(_ enc: MTLComputeCommandEncoder, width: Int, height: Int) {
        defer { pending = nil }
        if apply, let p = pending, p.width == width, p.height == height, let sharedBuf, let table {
            enc.setTexture(p.texture, index: 24)
            enc.setTexture(table, index: 25)
            enc.setBuffer(sharedBuf, offset: 0, index: 7)
            applied += 1
        } else if let s = standIns() {
            enc.setTexture(s.tex, index: 24)
            enc.setTexture(s.tex, index: 25)
            enc.setBuffer(s.buf, offset: 0, index: 7)
        }
    }
}

/// Offline (tools/litflow.swift): the fog on (1) or off (0) from the next frame, and a debug view (-1: GODRAYS_VIEW's).
/// Returns 1 with METALMC_EXP=godrays (and sky and post) on.
@_cdecl("mmc_debug_godrays")
public func mmc_debug_godrays(_ on: Int32, _ view: Int32) -> Int32 {
    guard godraysEnabled else { return 0 }
    Godrays.shared.on = on != 0
    Godrays.shared.apply = true
    Godrays.shared.view = view
    Godrays.shared.restart()
    return 1
}

/// Offline timing (tools/litflow.swift): 0 the fog off, 1 traced but not applied (the resolve leaves the frame alone), 2 on;
/// unlike mmc_debug_godrays its history isn't started over, so frames can alternate as in a steady state. Returns 1 with
/// godrays on.
@_cdecl("mmc_debug_godrays_mode")
public func mmc_debug_godrays_mode(_ mode: Int32) -> Int32 {
    guard godraysEnabled else { return 0 }
    Godrays.shared.on = mode >= 1
    Godrays.shared.apply = mode >= 2
    return 1
}

/// Debug: the fog's shader source (godrays.metal as built in), for an offline compile check. Returns its length.
@_cdecl("mmc_debug_godrays_shader_source")
public func mmc_debug_godrays_shader_source(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let bytes = Array(godraysSource.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}
