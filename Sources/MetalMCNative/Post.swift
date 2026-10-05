import Foundation
import Metal
import simd

// Post-processing (METALMC_EXP=post, off by default; docs/lighting-design.md, "Post-processing"): what makes the lit frame
// look filmed, after the anti-aliasing and before the hand and the HUD, in scene-linear light:
// - Bloom: the frame's light spread by a chain of downsampled levels (Jimenez's 13-tap filter down, a tent filter back
//   up, seven levels: from a few pixels to about a sixth of the screen), mixed in as a share of the light (energy
//   conserving: what the bloom adds around a bright thing it takes from the thing itself). Only very bright light shows
//   it: the sun, the sky around it, glints on water; light sources (lit mode's G-buffer: block light 14 and 15) weigh more.
// - Eye adaptation: a histogram of the frame's log luminance over the square root of lit terrain's albedo (halfway to
//   metering light rather than color; a quarter of the resolution, a little below the screen's middle weighted 4 times
//   its edges), its mean between two percentiles against the reference scene (noon outdoors); little of a darker scene's
//   difference is made up within 2.5 stops (daylight's swings), most of it past that (night, caves), within limits;
//   followed over time, faster into the light (the image overexposes for a moment and settles) than into the dark (it
//   takes a couple of seconds to open up). In stops on top of the sky's own adaptation to the sun's height (Sky.swift),
//   which stays: the units of docs/lighting-design.md are unchanged.
// - Light shafts: the share of sky in each quarter-resolution texel, blurred radially toward the sun's position on the
//   screen (two passes, the mean of 144 taps along the line), times the sky's own light in the texel's direction: the
//   air in front of terrain and clouds lit where the sun gets through, the sky darkened where something between it and
//   the sun shadows the air. Only with the sun up and near the screen.
// - The tone curve: AgX by default (METALMC_TONEMAP: aces for Hill's fit of ACES, gt for Uchimura's, legacy for the
//   sky's shoulder, the look before post), into SDR; with HDR output (METALMC_EXP=hdr) the same curve below a knee and
//   the highlights carried on into the display's headroom, the hue kept. It's the only tone curve the frame gets: with
//   post the sky pass and the aerial perspective step leave theirs out (Sky.swift), so nothing is mapped twice.
//
// The frame has to hold scene-linear light until here, so with post the main target is float (floatMainTarget in
// Hdr.swift: RG11B10Float, as with HDR) even on an SDR display, whose layer stays 8-bit; the anti-aliasing resolves
// scene-linear light into its float history. Post reads that history (without anti-aliasing, the frame), and its last
// step writes display light into the frame in place of the anti-aliasing's copy, with the same contrast-adaptive
// sharpening and dither, folded into the hand's pass the way that copy is (Backend.swift, pass folding).
//
// Every constant of the look is a #define at the top of the shader, so in lab mode (docs/lab-mode.md) post.metal can be
// edited while the game runs. The switches below (environment, read once) set their defaults.

/// METALMC_EXP=post: bloom, eye adaptation, light shafts and a filmic tone curve over the frame (needs our sky,
/// METALMC_EXP=sky, whose frames are scene-linear; meant for lit mode).
let postEnabled = experiments.contains("post")

private let postEnv = ProcessInfo.processInfo.environment
private func postSetting(_ name: String, _ fallback: Float) -> Float {
    guard let v = Float(postEnv[name] ?? ""), v.isFinite else { return fallback }
    return v
}
/// METALMC_TONEMAP: agx (default), aces (Hill's fit of the RRT and ODT), gt (Uchimura's), legacy (the sky's shoulder: the
/// look before post, for A/B).
let postToneCurves = ["legacy", "agx", "aces", "gt"]
private let postToneDefault = postToneCurves.firstIndex(of: postEnv["METALMC_TONEMAP"] ?? "agx") ?? 1
/// METALMC_BLOOM: the bloom's share of the light (0.05: 5% of it spread over the chain's levels).
private let postBloomDefault = max(0, postSetting("METALMC_BLOOM", 0.05))
/// METALMC_SHAFTS: the light shafts' strength (0: none).
private let postShaftsDefault = max(0, postSetting("METALMC_SHAFTS", 0.22))
/// METALMC_POSTEV: exposure compensation in stops, on top of the eye adaptation.
private let postEvDefault = postSetting("METALMC_POSTEV", 0)
/// METALMC_ADAPT=0: no eye adaptation (the exposure stays at the reference scene's).
private let postAdaptDefault = postEnv["METALMC_ADAPT"] != "0"
/// METALMC_POSTVIEW: debug views (1 the bloom alone, 2 the light shafts alone, 3 the exposure meter: false color by stop
/// against the reference, 4 no tone curve).
private let postViewDefault = Int(postEnv["METALMC_POSTVIEW"] ?? "") ?? 0
/// Bloom levels below the frame: the deepest is 1/128 of the resolution (27 x 17 texels at 3456 x 2234).
private let postBloomLevels = 7

private func postFloat(_ x: Float) -> String { String(format: "%.6g", x) }

private let postShaderSource = skyShaderHeader + (litEnabled ? "\n#define LIT_MODE 1\n" + litShaderHeader : "") + """

// ---- Post-processing (Post.swift). The look's constants: edit them here in lab mode. ----

// Bloom: its share of the light (mixed in as mix(scene, bloom, POST_BLOOM): energy conserving), the levels' weights
// (each level's share is POST_BLOOM_SHAPE times the one above it: 1 even, more spreads it wider), how much more light
// sources weigh (lit terrain with block light POST_LIGHT_LEVEL / 16 or more: torches 14, lava, glowstone and lanterns
// 15), and the firefly limit: 2 x 2 boxes brighter than POST_FIREFLY times the exposure's mid-gray are weighted down
// in the first downsample (a softened Karis average).
#define POST_BLOOM \(postFloat(postBloomDefault))
#define POST_BLOOM_SHAPE 1.0
#define POST_BLOOM_LIGHTS 2.0
#define POST_LIGHT_LEVEL 223u
#define POST_FIREFLY 256.0

// Eye adaptation: the histogram's range (log2), the percentiles its mean is taken between, the darkest albedo lit
// terrain's light is worked out with and how far the meter goes from luminance (0) toward light (1, luminance over
// albedo), the reference (the metered log2 value that gets 0 stops: noon outdoors), how much of a darker scene's
// difference the exposure makes up within the knee and past it, and of a brighter one's, its limits in stops, and how
// fast it follows (seconds: up, into the dark; down, into the light).
#define POST_BINS 128
#define POST_LOG_MIN (-14.0)
#define POST_LOG_MAX 10.0
#define POST_METER_LOW 0.30
#define POST_METER_HIGH 0.85
#define POST_METER_ALBEDO 0.04
#define POST_METER_LIGHT 0.5
#define POST_ADAPT_REF (\(litEnabled ? "-1.9" : "-2.1"))
#define POST_ADAPT_KNEE 2.5
#define POST_ADAPT_DARK_NEAR 0.2
#define POST_ADAPT_DARK 0.7
#define POST_ADAPT_BRIGHT 0.3
#define POST_EV_MIN (-2.0)
#define POST_EV_MAX 2.0
#define POST_TAU_UP 2.2
#define POST_TAU_DOWN 0.45
#define POST_EV \(postFloat(postEvDefault))

// Light shafts: strength over terrain and clouds, how much the sky darkens in their shadows, the distance (blocks) over
// which the air in front of a surface builds up to the shafts' full haze, taps per pass, the decay per tap along the
// line (of the 144 the two passes take).
#define POST_SHAFTS \(postFloat(postShaftsDefault))
#define POST_SHAFT_SHADOW 0.3
#define POST_SHAFT_DEPTH 192.0
#define POST_SHAFT_TAPS 12
#define POST_SHAFT_DECAY 0.994

// The tone curves' exposure (each curve's own mid-tones: these keep noon's terrain about as bright as before post) and
// AgX's look (ASC CDL power and saturation on its sigmoid's output: its "punchy" look's saturation without its
// contrast, which crushed dark scenes).
#define POST_AGX_EXPOSURE 1.0
#define POST_AGX_POWER 1.0
#define POST_AGX_SAT 1.3
#define POST_ACES_EXPOSURE 1.6
#define POST_GT_EXPOSURE 1.0
// HDR output: above this luminance (exposed) the highlights are carried on toward the headroom, over this span.
#define POST_EDR_KNEE 0.8
#define POST_EDR_SPAN 4.0

// Debug view (METALMC_POSTVIEW): 1 the bloom alone, 2 the light shafts alone, 3 the exposure meter, 4 no tone curve.
#define POST_VIEW \(postViewDefault)

struct PostFrame {
    float4x4 invViewProj;   // clip space to camera-relative world (the projection the frame holds, unjittered with anti-aliasing)
    float4 size;            // xy: the frame's size, zw: 1 / size
    float4 sun;             // xyz: toward the sun (world), w: the light shafts' share this frame (0: none)
    float4 sunUV;           // xy: the sun's position in texture coordinates (off the screen too), zw: unused
    float4 tone;            // x: display headroom (1: SDR), y: the legacy curve's knee, z: unused, w: the tone curve (0-3)
    float4 adapt;           // x: seconds since the last frame (0.1 at most), y: 1 to snap to the target, z: 1 for eye adaptation, w: unused
    float4 effects;         // x: bloom on, y: light shafts on, z: light sources in the G-buffer (lit mode), w: unused
};

constant float3 kPostLuma = float3(0.2126, 0.7152, 0.0722);
static float postLuma(float3 c) { return dot(c, kPostLuma); }

// ---------------------------------------------------------------------------------------------------------------- bloom

// Full resolution to half, Jimenez's 13 taps ("Next Generation Post Processing in Call of Duty: Advanced Warfare",
// SIGGRAPH 2014) as five overlapping 2 x 2 boxes of bilinear taps: the inner box half the weight, the four corner boxes
// an eighth each. The threadgroup's 36 x 36 pixels are read once, in linear light (the frame holds it sRGB-encoded,
// which a bilinear tap would average wrongly), light sources weighted up. A box brighter than the firefly limit is
// weighted down by how far past it it is: a glint's few pixels can't make the bloom flicker, a large bright area (the
// sun's disk, a lava lake) keeps its weight against boxes like it. Alpha carries what the light meter divides by (lit
// mode): lit terrain's albedo luminance, 1 for everything else, averaged down the chain like the light.
#define PB_T 16
#define PB_A (2 * PB_T + 4)
#define PB_PX(PX, PY) float4(tile[(o.y + (PY)) * PB_A + o.x + (PX)])
#define PB_BOX(BX, BY) ((PB_PX(BX, BY) + PB_PX((BX) + 1, BY) + PB_PX(BX, (BY) + 1) + PB_PX((BX) + 1, (BY) + 1)) * 0.25)

static float postBoxWeight(float3 c, float limit) {
    return 1.0 / (1.0 + max(postLuma(c) - limit, 0.0) / limit);
}

kernel void post_bloom_first(texture2d<float, access::read> src [[texture(0)]],
                             texture2d<float, access::write> dst [[texture(1)]],
                             texture2d<uint, access::read> gbuf [[texture(2)]],
                             depth2d<float, access::read> depth [[texture(3)]],
                             constant PostFrame& f [[buffer(0)]],
                             device const float4* expo [[buffer(1)]],
                             uint2 lid [[thread_position_in_threadgroup]],
                             uint2 tgid [[threadgroup_position_in_grid]]) {
    threadgroup half4 tile[PB_A * PB_A];
    int2 size = int2(src.get_width(), src.get_height());
    int2 base = int2(tgid) * (2 * PB_T) - 2;   // even: the tile's 2 x 2 blocks are the frame's
    bool lights = f.effects.z > 0.5;
#if LIT_MODE
    // Lit terrain's albedo luminance (for the meter; 1: not lit terrain) and the light source weight, once per 2 x 2 block
    // from its top-left pixel's G-buffer and depth: a quarter of the reads (12 bytes a pixel were most of this pass's
    // cost), for blurs and a meter that are far wider than a block.
    threadgroup half2 blocks[(PB_A / 2) * (PB_A / 2)];
    if (lights) {
        for (uint i = lid.y * PB_T + lid.x; i < (PB_A / 2) * (PB_A / 2); i += PB_T * PB_T) {
            uint2 q = uint2(clamp(base + 2 * int2(i % (PB_A / 2), i / (PB_A / 2)), int2(0), size - 1));
            uint2 g = gbuf.read(q).rg;
            half2 v = half2(1.0h);
            if ((g.x >> 29) != 0u && litDepthMatches(depth.read(q), g.y & 0xFFFFu)) {
                float3 al = skyDecode(float3(float(g.x & 255u), float((g.x >> 8) & 255u), float((g.x >> 16) & 255u)) / 255.0);
                v.x = half(max(postLuma(al), POST_METER_ALBEDO));
                if ((g.y >> 24) >= POST_LIGHT_LEVEL) v.y = half(POST_BLOOM_LIGHTS);
            }
            blocks[i] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
#endif
    for (uint i = lid.y * PB_T + lid.x; i < PB_A * PB_A; i += PB_T * PB_T) {
        uint2 q = uint2(clamp(base + int2(i % PB_A, i / PB_A), int2(0), size - 1));
        float3 c = max(skyDecode(src.read(q).rgb), 0.0);
        float albedo = 1.0;
#if LIT_MODE
        if (lights) {
            half2 v = blocks[((i / PB_A) / 2) * (PB_A / 2) + (i % PB_A) / 2];
            albedo = float(v.x);
            c *= float(v.y);
        }
#endif
        tile[i] = half4(half3(min(c, float3(60000.0))), half(albedo));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint2 gid = tgid * PB_T + lid;
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    // This texel's footprint: tile pixels 2 lid .. 2 lid + 5 on each axis. PB_BOX(x, y): the bilinear tap at offset
    // (x - 2, y - 2) from its center, a 2 x 2 box.
    uint2 o = lid * 2u;
    float4 a = PB_BOX(0, 0), b = PB_BOX(2, 0), c = PB_BOX(4, 0);
    float4 d = PB_BOX(1, 1), e = PB_BOX(3, 1);
    float4 l0 = PB_BOX(0, 2), g = PB_BOX(2, 2), h = PB_BOX(4, 2);
    float4 i0 = PB_BOX(1, 3), j = PB_BOX(3, 3);
    float4 k = PB_BOX(0, 4), l = PB_BOX(2, 4), m = PB_BOX(4, 4);
    float4 inner = (d + e + i0 + j) * 0.25;
    float4 c0 = (a + b + l0 + g) * 0.25, c1 = (b + c + g + h) * 0.25, c2 = (l0 + g + k + l) * 0.25, c3 = (g + h + l + m) * 0.25;
    float limit = POST_FIREFLY * 0.18 * exp2(-expo[0].x);
    float wi = 0.5 * postBoxWeight(inner.rgb, limit);
    float w0 = 0.125 * postBoxWeight(c0.rgb, limit), w1 = 0.125 * postBoxWeight(c1.rgb, limit);
    float w2 = 0.125 * postBoxWeight(c2.rgb, limit), w3 = 0.125 * postBoxWeight(c3.rgb, limit);
    dst.write((inner * wi + c0 * w0 + c1 * w1 + c2 * w2 + c3 * w3) / (wi + w0 + w1 + w2 + w3), gid);
}

// Down the chain: the same 13 taps from the level above (linear light already), plain weights; alpha (the light meter's
// albedo) along with it.
#define PB_S(SX, SY) src.sample(s, uv + t * float2(SX, SY), level(0.0))
kernel void post_bloom_down(texture2d<float> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 t = 1.0 / float2(src.get_width(), src.get_height());
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    float4 r = (PB_S(-1, -1) + PB_S(1, -1) + PB_S(-1, 1) + PB_S(1, 1)) * 0.125
             + (PB_S(-2, -2) + PB_S(2, -2) + PB_S(-2, 2) + PB_S(2, 2)) * 0.03125
             + (PB_S(0, -2) + PB_S(-2, 0) + PB_S(2, 0) + PB_S(0, 2)) * 0.0625
             + PB_S(0, 0) * 0.125;
    dst.write(r, gid);
}

// A level's share of the bloom: POST_BLOOM_SHAPE times the share of the level above it, the shares summing to 1.
static float postLevelShare(float i, float n) {
    float s = POST_BLOOM_SHAPE;
    return abs(s - 1.0) < 1e-3 ? 1.0 / n : pow(s, i - 1.0) * (1.0 - s) / (1.0 - pow(s, n));
}

// Up the chain: this level's own light (its share) plus the level below filtered up with a 3 x 3 tent (9 bilinear taps
// a texel of the level below apart). w: x this level's number (1: half resolution), y the number of levels, z 1 if the
// level below is the deepest (then its own share applies to it).
#define PB_L(LX, LY) low.sample(s, uv + t * float2(LX, LY), level(0.0)).rgb
kernel void post_bloom_up(texture2d<float> low [[texture(0)]], texture2d<float, access::read> own [[texture(1)]],
                          texture2d<float, access::write> dst [[texture(2)]], constant float4& w [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 t = 1.0 / float2(low.get_width(), low.get_height());
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    float3 tent = (PB_L(-1, -1) + PB_L(1, -1) + PB_L(-1, 1) + PB_L(1, 1)) * 0.0625
                + (PB_L(0, -1) + PB_L(-1, 0) + PB_L(1, 0) + PB_L(0, 1)) * 0.125 + PB_L(0, 0) * 0.25;
    float below = w.z > 0.5 ? postLevelShare(w.y, w.y) : 1.0;
    dst.write(float4(own.read(gid).rgb * postLevelShare(w.x, w.y) + tent * below, 1.0), gid);
}

// ------------------------------------------------------------------------------------------------------ eye adaptation

// Bin 0: darker than 2^POST_LOG_MIN; bins 1 .. POST_BINS - 1 evenly over log2 luminance up to 2^POST_LOG_MAX.
static float postBinLog(float bin) {
    return POST_LOG_MIN + (bin - 0.5) / float(POST_BINS - 1) * (POST_LOG_MAX - POST_LOG_MIN);
}

// The histogram of a bloom level (quarter resolution, linear light), each texel's weight 1 at the screen's edges up to 4
// a little below its middle (center-weighted metering that leans away from the sky). In lit mode it meters halfway
// between luminance and light: the texel's luminance over its mean albedo luminance (alpha, from the G-buffer in the
// bloom's first pass; 1 for sky, water and the rest) to the power POST_METER_LIGHT. Luminance alone (0) let a dark
// forest canopy read two stops darker than the plains under the same sun; light alone (1, an incident-light meter)
// opened sunsets up by a stop because their land is lit dimly while the sky is bright. Threadgroups of 16 x 16 count
// into threadgroup memory first.
kernel void post_histogram(texture2d<float, access::read> src [[texture(0)]], device atomic_uint* hist [[buffer(0)]],
                           uint2 gid [[thread_position_in_grid]], uint li [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint local[POST_BINS];
    if (li < POST_BINS) atomic_store_explicit(&local[li], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint w = src.get_width(), h = src.get_height();
    if (gid.x < w && gid.y < h) {
        float4 v = src.read(gid);
        float l = postLuma(v.rgb) / pow(clamp(v.a, POST_METER_ALBEDO, 1.0), POST_METER_LIGHT);
        float x = (log2(max(l, 1e-30)) - POST_LOG_MIN) / (POST_LOG_MAX - POST_LOG_MIN);
        uint bin = x < 0.0 ? 0u : uint(clamp(1.0 + x * float(POST_BINS - 1), 1.0, float(POST_BINS - 1)));
        float2 p = (float2(gid) + 0.5) / float2(w, h) * 2.0 - 1.0;
        uint weight = 1u + uint(round(3.0 * saturate(1.0 - length((p - float2(0.0, -0.3)) * float2(0.9, 1.0)))));
        atomic_fetch_add_explicit(&local[bin], weight, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (li < POST_BINS) {
        uint v = atomic_load_explicit(&local[li], memory_order_relaxed);
        if (v > 0u) atomic_fetch_add_explicit(&hist[li], v, memory_order_relaxed);
    }
}

// One threadgroup of POST_BINS threads: the histogram's mean log2 luminance between the two percentiles (the darkest
// and the brightest pixels don't count: a black corner, the sun), the exposure it calls for, and the exposure followed
// toward it. Clears the histogram for the next frame. expo[0]: x the exposure in stops, y its target, z the metered log2
// luminance, w 1 once it holds a value; expo[1]: the histogram's total weight and the two percentiles' (for the log).
kernel void post_exposure(device atomic_uint* hist [[buffer(0)]], device float4* expo [[buffer(1)]],
                          constant PostFrame& f [[buffer(2)]], uint li [[thread_index_in_threadgroup]]) {
    threadgroup float h[POST_BINS];
    h[li] = float(atomic_load_explicit(&hist[li], memory_order_relaxed));
    atomic_store_explicit(&hist[li], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (li != 0u) return;
    float total = 0.0;
    for (uint b = 0; b < POST_BINS; b++) total += h[b];
    float lo = total * POST_METER_LOW, hi = total * POST_METER_HIGH;
    float acc = 0.0, sum = 0.0, wsum = 0.0;
    for (uint b = 0; b < POST_BINS; b++) {
        float w = clamp(acc + h[b], lo, hi) - clamp(acc, lo, hi);   // the part of this bin between the percentiles
        acc += h[b];
        sum += w * postBinLog(float(b));
        wsum += w;
    }
    float metered = wsum > 0.0 ? sum / wsum : POST_ADAPT_REF;
    float d = POST_ADAPT_REF - metered;   // > 0: darker than the reference
    // Darker: little within POST_ADAPT_KNEE stops of the reference (daylight's own swings, which the sky's adaptation to
    // the sun's height already covers: dusk, shade, a forest), more past it (night, caves, interiors).
    float response = d > 0.0 ? POST_ADAPT_DARK_NEAR * min(d, POST_ADAPT_KNEE) + POST_ADAPT_DARK * max(d - POST_ADAPT_KNEE, 0.0)
                             : POST_ADAPT_BRIGHT * d;
    float target = f.adapt.z > 0.5 ? clamp(response, POST_EV_MIN, POST_EV_MAX) : 0.0;
    float4 e = expo[0];
    float ev = target;
    if (f.adapt.y < 0.5 && e.w > 0.5 && isfinite(e.x)) {
        float tau = target > e.x ? POST_TAU_UP : POST_TAU_DOWN;
        ev = e.x + (target - e.x) * (1.0 - exp(-f.adapt.x / tau));
    }
    expo[0] = float4(ev, target, metered, 1.0);
    expo[1] = float4(total, lo, hi, 0.0);
}

// --------------------------------------------------------------------------------------------------------- light shafts

// Each quarter-resolution texel's 4 x 4 pixels: r the share that is sky (reverse-Z: depth 0; vanilla's clouds write
// depth, so they shadow the air like terrain does), g how much air there is in front of the rest (1 - exp(-distance /
// POST_SHAFT_DEPTH) at their mean depth: the haze the shafts light builds up with distance, so a tree nearby gets
// little of it and a ridge a few hundred blocks off nearly all).
kernel void post_shaft_mask(depth2d<float> depth [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]],
                            constant PostFrame& f [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(filter::nearest, address::clamp_to_edge);
    float2 inv = 1.0 / float2(depth.get_width(), depth.get_height());
    float2 c = float2(gid) * 4.0 + 2.0;
    float4 a = depth.gather(s, (c + float2(-1.0, -1.0)) * inv), b = depth.gather(s, (c + float2(1.0, -1.0)) * inv);
    float4 d = depth.gather(s, (c + float2(-1.0, 1.0)) * inv), e = depth.gather(s, (c + float2(1.0, 1.0)) * inv);
    float4 sky = float4(a <= 0.0) + float4(b <= 0.0) + float4(d <= 0.0) + float4(e <= 0.0);
    float n = dot(sky, float4(1.0));
    float solid = 16.0 - n;
    float air = 1.0;
    if (solid > 0.5) {
        float mean = dot(a + b + d + e, float4(1.0)) / solid;   // the sky's depths are 0
        float4 h = f.invViewProj * float4(c * inv * 2.0 - 1.0, mean, 1.0);
        air = 1.0 - exp(-length(h.xyz / h.w) / POST_SHAFT_DEPTH);
    }
    dst.write(float4(n / 16.0, air, 0.0, 0.0), gid);
}

// One pass of the radial blur toward the sun (after Mitchell, "Volumetric Light Scattering as a Post-Process", GPU Gems
// 3): the weighted mean of POST_SHAFT_TAPS taps along the line from the texel toward the sun's position. The first pass
// (pass.x 0) spans the whole way, each tap's weight POST_SHAFT_DECAY^TAPS times the one before; the second spans one of
// its taps, decaying by POST_SHAFT_DECAY: together the mean of TAPS^2 taps along the whole line.
kernel void post_shaft_blur(texture2d<float> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]],
                            constant PostFrame& f [[buffer(0)]], constant float4& pass [[buffer(1)]],
                            uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    bool first = pass.x < 0.5;
    float2 step = (f.sunUV.xy - uv) * ((first ? 1.0 : 1.0 / float(POST_SHAFT_TAPS)) / float(POST_SHAFT_TAPS));
    float decay = first ? pow(POST_SHAFT_DECAY, float(POST_SHAFT_TAPS)) : POST_SHAFT_DECAY;
    float sum = 0.0, wsum = 0.0, w = 1.0;
    for (int i = 0; i < POST_SHAFT_TAPS; i++) {
        sum += src.sample(s, uv + step * float(i), level(0.0)).r * w;
        wsum += w;
        w *= decay;
    }
    dst.write(float4(sum / wsum), gid);
}

// The light the shafts add (or take), at quarter resolution: the sky's own light in the texel's direction (its glow
// around the sun; under the horizon, the horizon's) times how much of the way toward the sun is open sky. Over terrain and
// clouds the air in front is lit where the sun gets through; over the sky, the sky darkens where something between it
// and the sun shadows the air.
kernel void post_shaft_color(texture2d<float, access::read> blurred [[texture(0)]], texture2d<float, access::read> mask [[texture(1)]],
                             texture2d<float, access::write> dst [[texture(2)]], texture2d<float> skyView [[texture(3)]],
                             constant PostFrame& f [[buffer(0)]], constant SkyFrame& sky [[buffer(1)]],
                             uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    float2 uv = (float2(gid) + 0.5) / float2(dst.get_width(), dst.get_height());
    float4 h = f.invViewProj * float4(uv * 2.0 - 1.0, 1.0, 1.0);
    float3 dir = normalize(h.xyz / h.w);
    float open = blurred.read(gid).r;
    float2 m = mask.read(gid).rg;   // the sky's share, the air in front of the rest
    float share = POST_SHAFTS * open * m.y * (1.0 - m.x) - POST_SHAFT_SHADOW * (1.0 - open) * m.x;
    dst.write(float4(skyLuminance(sky, dir, skyView) * (share * f.sun.w), 1.0), gid);
}

// ---------------------------------------------------------------------------------------------------------- tone curves

// AgX (Troy Sobotka's, as in Blender 4; its sigmoid's polynomial fit and the matrices for linear Rec.709 from Benjamin
// Wrensch's minimal version): each channel of the inset primaries in a log2 encoding through a sigmoid, then back out.
// Bright, saturated light runs to white without the hue skews a curve on the original primaries gives.
static float3 postAgx(float3 x) {
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
    v = pow(max(v, 0.0), float3(POST_AGX_POWER));
    float l = postLuma(v);
    v = l + POST_AGX_SAT * (v - l);
    return pow(max(outset * v, 0.0), float3(2.2));
}

// ACES: Stephen Hill's fit of the RRT and the sRGB ODT, with its matrices in and out of the fit's primaries.
static float3 postAces(float3 x) {
    const float3x3 inM = float3x3(float3(0.59719, 0.07600, 0.02840), float3(0.35458, 0.90834, 0.13383), float3(0.04823, 0.01566, 0.83777));
    const float3x3 outM = float3x3(float3(1.60475, -0.10208, -0.00327), float3(-0.53108, 1.10813, -0.07276), float3(-0.07367, -0.00605, 1.07602));
    float3 v = inM * x;
    float3 a = v * (v + 0.0245786) - 0.000090537;
    float3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return saturate(outM * (a / b));
}

// Uchimura's curve (Gran Turismo Sport): a toe, a straight section of slope 1, an exponential shoulder to 1.
static float3 postGt(float3 x) {
    const float P = 1.0, a = 1.0, m = 0.22, l = 0.4, c = 1.33;
    float l0 = (P - m) * l / a, S0 = m + l0, S1 = m + a * l0;
    float CP = -(a * P / (P - S1)) / P;
    float3 w0 = 1.0 - smoothstep(float3(0.0), float3(m), x), w2 = step(float3(m + l0), x), w1 = 1.0 - w0 - w2;
    float3 T = m * pow(max(x, 0.0) / m, float3(c)), S = P - (P - S1) * exp(CP * (x - S0)), L = m + a * (x - m);
    return T * w0 + L * w1 + S * w2;
}

// Scene-linear light (exposed) to display-linear light: 1 is the display's SDR white, tone.x its headroom. Legacy is
// the sky's own curve (identity to the knee, a shoulder into the headroom: the look before post). The others are SDR
// curves; with headroom, the light above POST_EDR_KNEE is carried on toward it (zero slope where it starts, the curve's
// hue kept), so highlights go past SDR white instead of stopping there.
static float3 postToneMap(float3 x, float4 tone) {
    int curve = int(tone.w);
    float H = tone.x;
    x = max(x, 0.0);
    if (curve == 0) return skyToneMap(x, float4(H, tone.y, 0.0, 0.0));
    float3 d = curve == 2 ? postAces(x * POST_ACES_EXPOSURE) : (curve == 3 ? postGt(x * POST_GT_EXPOSURE) : postAgx(x * POST_AGX_EXPOSURE));
    if (H <= 1.0) return min(d, 1.0);
    float t = max(postLuma(x) - POST_EDR_KNEE, 0.0) / POST_EDR_SPAN;
    return min(d * (1.0 + (H - 1.0) * (1.0 - exp(-t * t))), H);
}

// Offline check (mmc_debug_post_curve): scene-linear inputs (exposed) through a tone curve at a headroom.
kernel void post_debug_curve(device const float4* in [[buffer(0)]], device float4* out [[buffer(1)]],
                             constant float4& tone [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    out[i] = float4(postToneMap(in[i].rgb, tone), 0.0);
}

// ------------------------------------------------------------------------------------------------------------ composite

// Every pixel: the frame's light, the bloom mixed in, the light shafts added, the exposure, the tone curve, sRGB-encoded
// display light into `out` (the copy puts it in the frame).
kernel void post_composite(texture2d<float, access::read> scene [[texture(0)]], texture2d<float> bloom [[texture(1)]],
                           texture2d<float> shafts [[texture(2)]], texture2d<float, access::write> out [[texture(3)]],
                           constant PostFrame& f [[buffer(0)]], device const float4* expo [[buffer(1)]],
                           uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    float2 uv = (float2(gid) + 0.5) * f.size.zw;
    float3 c = max(skyDecode(scene.read(gid).rgb), 0.0);
    float3 b = f.effects.x > 0.5 ? bloom.sample(s, uv, level(0.0)).rgb : c;
    float3 sh = f.effects.y > 0.5 ? shafts.sample(s, uv, level(0.0)).rgb : float3(0.0);
    c = max(mix(c, b, POST_BLOOM) + sh, 0.0);
    float ev = expo[0].x + POST_EV;
#if POST_VIEW == 1
    c = b;
#elif POST_VIEW == 2
    c = abs(sh) * 4.0;
#elif POST_VIEW == 3
    // Stops against the reference: blue -3, cyan -1.5, green 0, yellow +1.5, red +3 (of the exposed light), gray outside.
    float stops = log2(max(postLuma(c * exp2(ev)), 1e-6)) - POST_ADAPT_REF;
    float3 heat = stops < -3.0 || stops > 3.0 ? float3(0.2) : (stops < 0.0 ? mix(float3(0, 0, 1), float3(0, 1, 0), (stops + 3.0) / 3.0)
                                                                         : mix(float3(0, 1, 0), float3(1, 0, 0), stops / 3.0));
    out.write(float4(skyEncode(heat * 0.7), 1.0), gid);
    return;
#endif
#if POST_VIEW == 4
    float3 d = min(c * exp2(ev), f.tone.x);
#else
    float3 d = postToneMap(c * exp2(ev), f.tone);
#endif
    out.write(float4(skyEncode(d), 1.0), gid);
}

// -------------------------------------------------------------------------------------------------------- into the frame

struct PostVOut { float4 pos [[position]]; };

vertex PostVOut post_copy_vs(uint vid [[vertex_id]]) {
    PostVOut o;
    float2 p = float2(float((vid << 1) & 2), float(vid & 2));
    o.pos = float4(p * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// The composite's display light into the frame, as the anti-aliasing's copy writes its history: contrast-adaptive
// sharpening (after AMD's CAS; on the values over the display's largest, so it works the same in HDR) and the dither
// (relative on the packed RG11B10Float frame, whose 6-bit mantissas band smooth gradients). sc: x sharpening, y the
// largest value the frame takes (1, or the encoded headroom), z dither (0 none, < 0 relative, > 0 absolute), w frame.
fragment float4 post_copy_fs(PostVOut in [[stage_in]], texture2d<float, access::read> h [[texture(0)]],
                             constant float4& sc [[buffer(0)]]) {
    int2 size = int2(h.get_width(), h.get_height());
    int2 g = int2(in.pos.xy);
    float3 c = h.read(uint2(g)).rgb;
    if (sc.x > 0.0) {
        float k = 1.0 / sc.y;
        float3 n = h.read(uint2(clamp(g + int2(0, -1), int2(0), size - 1))).rgb * k;
        float3 s = h.read(uint2(clamp(g + int2(0, 1), int2(0), size - 1))).rgb * k;
        float3 w = h.read(uint2(clamp(g + int2(-1, 0), int2(0), size - 1))).rgb * k;
        float3 e = h.read(uint2(clamp(g + int2(1, 0), int2(0), size - 1))).rgb * k;
        float3 cc = c * k;
        float3 mn = min(cc, min(min(n, s), min(w, e))), mx = max(cc, max(max(n, s), max(w, e)));
        float3 amp = sqrt(saturate(min(mn, 2.0 - mx) / max(mx, 1e-4)));
        float3 wt = -amp * (0.2 * sc.x);
        c = (cc + (n + s + w + e) * wt) / (1.0 + 4.0 * wt) / k;
    }
    c = clamp(c, 0.0, sc.y);
    if (sc.z != 0.0) {
        float2 p = float2(g) + 5.588238 * sc.w;
        float nz = fract(52.9829189 * fract(dot(p, float2(0.06711056, 0.00583715)))) - 0.5;
        c = sc.z < 0.0 ? c * (1.0 + nz * float3(1.0 / 64.0, 1.0 / 64.0, 1.0 / 32.0)) : c + nz * sc.z;
    }
    return float4(c, 1.0);
}
"""

/// Mirrors PostFrame in the shader.
private struct PostFrameGPU {
    var invViewProj = matrix_identity_float4x4
    var size = SIMD4<Float>.zero
    var sun = SIMD4<Float>.zero
    var sunUV = SIMD4<Float>.zero
    var tone = SIMD4<Float>.zero
    var adapt = SIMD4<Float>.zero
    var effects = SIMD4<Float>.zero
}

private func postSmoothstep(_ e0: Float, _ e1: Float, _ x: Float) -> Float {
    let t = min(max((x - e0) / (e1 - e0), 0), 1)
    return t * t * (3 - 2 * t)
}

final class Post: @unchecked Sendable {
    static let shared = Post()

    private var library: MTLLibrary?
    private var failed = false
    private var firstPipe: MTLComputePipelineState?
    private var downPipe: MTLComputePipelineState?
    private var upPipe: MTLComputePipelineState?
    private var histPipe: MTLComputePipelineState?
    private var expoPipe: MTLComputePipelineState?
    private var maskPipe: MTLComputePipelineState?
    private var blurPipe: MTLComputePipelineState?
    private var shaftPipe: MTLComputePipelineState?
    private var compositePipe: MTLComputePipelineState?
    private var copyPipes: [String: MTLRenderPipelineState] = [:]   // by the pass's color, depth and stencil formats

    /// The bloom's levels: down[0] is half the frame's resolution, each next half the one before; up[i] holds levels
    /// i+1 and below filtered back up to down[i]'s size (up[0] is the bloom).
    private var down: [MTLTexture] = []
    private var up: [MTLTexture] = []
    private var shaftMask: MTLTexture?
    private var shaftTemp: MTLTexture?
    private var shaftBlur: MTLTexture?
    private var shaftColor: MTLTexture?
    /// The composite's display light, sRGB-encoded (the copy sharpens it into the frame).
    private(set) var output: MTLTexture?
    private var histogram: MTLBuffer?
    /// The exposure's state (see post_exposure), shared so the log can read it a frame or two late.
    private(set) var exposure: MTLBuffer?
    private var dummyGbuf: MTLTexture?
    private var sizeKey = ""

    private var frames = 0
    private var lastNanos: UInt64 = 0
    private var lastCam = SIMD3<Double>(repeating: .nan)
    private var snapNext = true

    /// Offline switches (mmc_debug_post): the effects (bit 0 bloom, 1 light shafts, 2 eye adaptation) and the tone curve.
    var effectMask = 7
    var toneCurve = postToneDefault
    /// Offline: seconds per frame for the adaptation (0: the clock's).
    var fixedStep: Double = 0

    func snap() { snapNext = true }

    private func ensureLibrary() -> Bool {
        if failed { return false }
        if library != nil { return true }
        do {
            // Lab mode (ShaderLab.swift): after an edit, forget the library and its pipelines; the next frame builds them
            // again (the bloom's levels and the exposure's state stay).
            let lib = try ShaderLab.library("post", postShaderSource) { [self] _ in
                library = nil; copyPipes = [:]; failed = false
            }
            func pipe(_ name: String) throws -> MTLComputePipelineState {
                guard let f = lib.makeFunction(name: name) else { throw NSError(domain: "metalmc", code: 1, userInfo: [NSLocalizedDescriptionKey: "no \(name)"]) }
                return try ctx.device.makeComputePipelineState(function: f)
            }
            firstPipe = try pipe("post_bloom_first")
            downPipe = try pipe("post_bloom_down")
            upPipe = try pipe("post_bloom_up")
            histPipe = try pipe("post_histogram")
            expoPipe = try pipe("post_exposure")
            maskPipe = try pipe("post_shaft_mask")
            blurPipe = try pipe("post_shaft_blur")
            shaftPipe = try pipe("post_shaft_color")
            compositePipe = try pipe("post_composite")
            library = lib
        } catch {
            log("post: shaders failed: \(error)")
            failed = true
            return false
        }
        if histogram == nil {
            histogram = ctx.device.makeBuffer(length: 4 * 128, options: .storageModeShared)
            if let h = histogram { memset(h.contents(), 0, h.length) }
            exposure = ctx.device.makeBuffer(length: 2 * 16, options: .storageModeShared)
            if let e = exposure { memset(e.contents(), 0, e.length) }
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Uint, width: 1, height: 1, mipmapped: false)
            d.usage = .shaderRead
            dummyGbuf = ctx.device.makeTexture(descriptor: d)
        }
        return histogram != nil && exposure != nil && dummyGbuf != nil
    }

    /// The bloom's levels, the light shafts' textures and the output for a frame of this size.
    private func ensureTargets(width: Int, height: Int) -> Bool {
        let key = "\(width)x\(height)"
        if key == sizeKey, output != nil { return true }
        func tex(_ format: MTLPixelFormat, _ w: Int, _ h: Int, _ label: String) -> MTLTexture? {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: max(w, 1), height: max(h, 1), mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            let t = ctx.device.makeTexture(descriptor: d)
            t?.label = label
            return t
        }
        var w = width, h = height
        var downs: [MTLTexture] = [], ups: [MTLTexture] = []
        for level in 1...postBloomLevels {
            w = (w + 1) / 2
            h = (h + 1) / 2
            guard let d = tex(.rgba16Float, w, h, "MetalMC post bloom down \(level)") else { return false }
            downs.append(d)
            if level < postBloomLevels {
                guard let u = tex(.rgba16Float, w, h, "MetalMC post bloom up \(level)") else { return false }
                ups.append(u)
            }
        }
        let qw = (width + 3) / 4, qh = (height + 3) / 4
        // The composite's display light: on an SDR display it's within [0, 1], which RGB10A2 holds finer than the 8-bit
        // present (half RGBA16Float's bandwidth for the composite's write and the copy's reads); past 1 with HDR output.
        guard let m = tex(.rg16Float, qw, qh, "MetalMC post shaft mask"), let t = tex(.r16Float, qw, qh, "MetalMC post shaft blur 1"),
              let b = tex(.r16Float, qw, qh, "MetalMC post shaft blur 2"), let c = tex(.rgba16Float, qw, qh, "MetalMC post shafts"),
              let o = tex(hdrOutput ? .rgba16Float : .rgb10a2Unorm, width, height, "MetalMC post output") else { return false }
        down = downs; up = ups
        shaftMask = m; shaftTemp = t; shaftBlur = b; shaftColor = c
        output = o
        sizeKey = key
        log("post: \(key), bloom \(postBloomLevels) levels down to \(w)x\(h), light shafts at \(qw)x\(qh), tone curve \(postToneCurves[toneCurve])")
        return true
    }

    /// The copy into the frame, for a pass with these formats (pass folding puts it in the hand's pass, with its depth).
    private func copyPipe(color: MTLPixelFormat, depth: MTLPixelFormat, stencil: MTLPixelFormat) -> MTLRenderPipelineState? {
        let k = "\(color.rawValue)/\(depth.rawValue)/\(stencil.rawValue)"
        if let p = copyPipes[k] { return p }
        guard let lib = library else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.label = "MetalMC post copy"
        d.vertexFunction = lib.makeFunction(name: "post_copy_vs")
        d.fragmentFunction = lib.makeFunction(name: "post_copy_fs")
        d.colorAttachments[0].pixelFormat = color
        d.depthAttachmentPixelFormat = depth
        d.stencilAttachmentPixelFormat = stencil
        guard let p = try? ctx.device.makeRenderPipelineState(descriptor: d) else { return nil }
        copyPipes[k] = p
        return p
    }

    /// The stages (bit 0 bloom, 1 exposure, 2 light shafts, 3 composite), for the offline timing to run one at a time.
    struct Stages: OptionSet {
        let rawValue: Int
        static let bloom = Stages(rawValue: 1), exposure = Stages(rawValue: 2), shafts = Stages(rawValue: 4), composite = Stages(rawValue: 8)
        static let all: Stages = [.bloom, .exposure, .shafts, .composite]
    }

    /// The frame's parameters: p is the projection the frame holds (unjittered with anti-aliasing) and the view rotation
    /// (32 floats), then vanilla's sun angle.
    private func frameParams(color: MTLTexture, p: UnsafePointer<Float>, cam: SIMD3<Double>) -> PostFrameGPU {
        func mat(_ o: Int) -> simd_float4x4 {
            simd_float4x4(SIMD4(p[o], p[o + 1], p[o + 2], p[o + 3]), SIMD4(p[o + 4], p[o + 5], p[o + 6], p[o + 7]),
                          SIMD4(p[o + 8], p[o + 9], p[o + 10], p[o + 11]), SIMD4(p[o + 12], p[o + 13], p[o + 14], p[o + 15]))
        }
        let viewProj = mat(0) * mat(16)
        var f = PostFrameGPU()
        f.invViewProj = viewProj.inverse
        f.size = SIMD4(Float(color.width), Float(color.height), 1 / Float(color.width), 1 / Float(color.height))
        // The sun: vanilla's direction, its place on the screen (a direction: w 0), and how much the light shafts show:
        // above the horizon, near the screen (fading out from its edge to as far again past it), in front of the camera,
        // less in rain.
        let sunAngle = p[32]
        let sun = SIMD3<Float>(-sin(sunAngle), cos(sunAngle), 0)
        let clip = viewProj * SIMD4(sun, 0)
        var shafts: Float = 0
        if clip.w > 1e-4 {
            let ndc = SIMD2(clip.x, clip.y) / clip.w
            f.sunUV = SIMD4(ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5, 0, 0)
            let off = max(abs(ndc.x), abs(ndc.y))
            shafts = (1 - postSmoothstep(1, 2, off)) * postSmoothstep(-0.03, 0.03, sun.y) * (1 - Sky.shared.frame.aerial.z)
        }
        f.sun = SIMD4(sun, shafts)
        let h = isFloatFrame(color.pixelFormat) && hdrOutput ? max(Hdr.shared.headroom, 1) : 1
        f.tone = SIMD4(h, skyKnee(h), 0, Float(toneCurve))
        // Time since the last frame, for the adaptation; a jump of the camera (a teleport, a respawn) or the first frame
        // snaps it to the scene.
        let now = DispatchTime.now().uptimeNanoseconds
        let dt = fixedStep > 0 ? fixedStep : (lastNanos == 0 ? 0 : Double(now - lastNanos) / 1e9)
        lastNanos = now
        let jump = !(simd_length(cam - lastCam) < 16)
        lastCam = cam
        let snap = snapNext || jump
        snapNext = false
        f.adapt = SIMD4(Float(min(max(dt, 0), 0.1)), snap ? 1 : 0, effectMask & 4 != 0 && postAdaptDefault ? 1 : 0, 0)
        let lights = litEnabled && Lit.shared.gbuffer.map { $0.width == color.width && $0.height == color.height } == true
        f.effects = SIMD4(effectMask & 1 != 0 && postBloomDefault > 0 ? 1 : 0,
                          effectMask & 2 != 0 && postShaftsDefault > 0 && shafts > 0 && Sky.shared.skyView != nil ? 1 : 0,
                          lights ? 1 : 0, 0)
        return f
    }

    /// Encodes the post chain over `input` (the anti-aliasing's new history, or the frame itself) into `output`, in
    /// `cb`. `depth`: the frame's depth (the light shafts' mask, the light sources' depth test).
    private func encode(cb: MTLCommandBuffer, input: MTLTexture, depth: MTLTexture, f frame: PostFrameGPU, stages: Stages) -> Bool {
        guard let firstPipe, let downPipe, let upPipe, let histPipe, let expoPipe, let maskPipe, let blurPipe, let shaftPipe,
              let compositePipe, let histogram, let exposure, let output, let shaftMask, let shaftTemp, let shaftBlur, let shaftColor,
              down.count == postBloomLevels, up.count == postBloomLevels - 1 else { return false }
        var f = frame
        func grid(_ t: MTLTexture) -> MTLSize { MTLSize(width: t.width, height: t.height, depth: 1) }
        let tg = MTLSize(width: 16, height: 16, depth: 1)
        // Bloom (and the histogram's source, its second level; the offline timing's exposure stage alone reuses the last).
        if stages.contains(.bloom) {
            guard let enc = cb.makeComputeCommandEncoder(descriptor: profComputePass("post bloom (13-tap down x\(postBloomLevels), tent up)")) else { return false }
            enc.label = "MetalMC post bloom"
            enc.setComputePipelineState(firstPipe)
            enc.setTexture(input, index: 0)
            enc.setTexture(down[0], index: 1)
            enc.setTexture(f.effects.z > 0.5 ? Lit.shared.gbuffer : dummyGbuf, index: 2)
            enc.setTexture(depth, index: 3)
            enc.setBytes(&f, length: MemoryLayout<PostFrameGPU>.stride, index: 0)
            enc.setBuffer(exposure, offset: 0, index: 1)
            enc.dispatchThreadgroups(MTLSize(width: (down[0].width + 15) / 16, height: (down[0].height + 15) / 16, depth: 1), threadsPerThreadgroup: tg)
            enc.setComputePipelineState(downPipe)
            for i in 1..<postBloomLevels {
                enc.setTexture(down[i - 1], index: 0)
                enc.setTexture(down[i], index: 1)
                enc.dispatchThreads(grid(down[i]), threadsPerThreadgroup: tg)
            }
            if stages.contains(.bloom) && f.effects.x > 0.5 {
                enc.setComputePipelineState(upPipe)
                let n = Float(postBloomLevels)
                for i in stride(from: postBloomLevels - 2, through: 0, by: -1) {
                    let deepest = i == postBloomLevels - 2
                    var w = SIMD4<Float>(Float(i + 1), n, deepest ? 1 : 0, 0)
                    enc.setTexture(deepest ? down[i + 1] : up[i + 1], index: 0)
                    enc.setTexture(down[i], index: 1)
                    enc.setTexture(up[i], index: 2)
                    enc.setBytes(&w, length: 16, index: 0)
                    enc.dispatchThreads(grid(up[i]), threadsPerThreadgroup: tg)
                }
            }
            enc.endEncoding()
        }
        // Eye adaptation: the histogram of the bloom's second level, then the exposure.
        if stages.contains(.exposure) {
            guard let enc = cb.makeComputeCommandEncoder(descriptor: profComputePass("post exposure (histogram, adaptation)")) else { return false }
            enc.label = "MetalMC post exposure"
            enc.setComputePipelineState(histPipe)
            enc.setTexture(down[1], index: 0)
            enc.setBuffer(histogram, offset: 0, index: 0)
            enc.dispatchThreads(grid(down[1]), threadsPerThreadgroup: tg)
            enc.setComputePipelineState(expoPipe)
            enc.setBuffer(histogram, offset: 0, index: 0)
            enc.setBuffer(exposure, offset: 0, index: 1)
            enc.setBytes(&f, length: MemoryLayout<PostFrameGPU>.stride, index: 2)
            enc.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
            enc.endEncoding()
        }
        // Light shafts: the sky mask, two passes of the radial blur, the light.
        if stages.contains(.shafts) && f.effects.y > 0.5, let skyView = Sky.shared.skyView {
            guard let enc = cb.makeComputeCommandEncoder(descriptor: profComputePass("post light shafts")) else { return false }
            enc.label = "MetalMC post light shafts"
            enc.setComputePipelineState(maskPipe)
            enc.setTexture(depth, index: 0)
            enc.setTexture(shaftMask, index: 1)
            enc.setBytes(&f, length: MemoryLayout<PostFrameGPU>.stride, index: 0)
            enc.dispatchThreads(grid(shaftMask), threadsPerThreadgroup: tg)
            enc.setComputePipelineState(blurPipe)
            var p1 = SIMD4<Float>(0, 0, 0, 0), p2 = SIMD4<Float>(1, 0, 0, 0)
            enc.setTexture(shaftMask, index: 0)
            enc.setTexture(shaftTemp, index: 1)
            enc.setBytes(&p1, length: 16, index: 1)
            enc.dispatchThreads(grid(shaftTemp), threadsPerThreadgroup: tg)
            enc.setTexture(shaftTemp, index: 0)
            enc.setTexture(shaftBlur, index: 1)
            enc.setBytes(&p2, length: 16, index: 1)
            enc.dispatchThreads(grid(shaftBlur), threadsPerThreadgroup: tg)
            var sky = Sky.shared.frame
            enc.setComputePipelineState(shaftPipe)
            enc.setTexture(shaftBlur, index: 0)
            enc.setTexture(shaftMask, index: 1)
            enc.setTexture(shaftColor, index: 2)
            enc.setTexture(skyView, index: 3)
            enc.setBytes(&f, length: MemoryLayout<PostFrameGPU>.stride, index: 0)
            enc.setBytes(&sky, length: MemoryLayout<SkyFrameGPU>.stride, index: 1)
            enc.dispatchThreads(grid(shaftColor), threadsPerThreadgroup: tg)
            enc.endEncoding()
        }
        if stages.contains(.composite) {
            guard let enc = cb.makeComputeCommandEncoder(descriptor: profComputePass("post composite (bloom, shafts, exposure, tone curve)")) else { return false }
            enc.label = "MetalMC post composite"
            enc.setComputePipelineState(compositePipe)
            enc.setTexture(input, index: 0)
            enc.setTexture(up[0], index: 1)
            enc.setTexture(shaftColor, index: 2)
            enc.setTexture(output, index: 3)
            enc.setBytes(&f, length: MemoryLayout<PostFrameGPU>.stride, index: 0)
            enc.setBuffer(exposure, offset: 0, index: 1)
            enc.dispatchThreads(grid(output), threadsPerThreadgroup: tg)
            enc.endEncoding()
        }
        return true
    }

    /// Runs the post chain on the frame just resolved into `color` (see the top of this file) and leaves its copy into
    /// the frame waiting in place of the anti-aliasing's (pass folding). Needs no pass open. p: the projection the frame
    /// holds (unjittered with anti-aliasing) and the view rotation, then vanilla's sun angle (33 floats).
    func apply(color: MTLTexture, depth: MTLTexture, p: UnsafePointer<Float>, cam: SIMD3<Double>, taa: Bool) -> Bool {
        guard ctx.pass == nil, Sky.shared.ready, ensureLibrary(), ensureTargets(width: color.width, height: color.height) else { return false }
        // The anti-aliasing's new history (its copy into the frame waits, and is replaced by ours), or the frame itself.
        let t = Taa.shared
        let taaCopy = ctx.pendingCopy.flatMap { sameImage($0.texture, color) ? $0 : nil }
        var input = color
        if taa, taaCopy != nil, t.history.count == 2, t.history[t.current].width == color.width, t.history[t.current].height == color.height {
            input = t.history[t.current]
        }
        let f = frameParams(color: color, p: p, cam: cam)
        // Take the waiting copy first: making the command buffer would run it.
        if taaCopy != nil { ctx.pendingCopy = nil }
        ctx.endBlit()
        let cb = ctx.ensureCBNoFlush()
        guard encode(cb: cb, input: input, depth: depth, f: f, stages: .all), let output else {
            ctx.pendingCopy = taaCopy
            return false
        }
        let isFloat = isFloatFrame(color.pixelFormat)
        let headroom = f.tone.x
        let sharpen: Float = input === color ? 0 : taaSharpenAmount
        let dither: Float = color.pixelFormat == .rgba16Float ? 0 : (color.pixelFormat == .rg11b10Float ? -1 : 1.0 / 255)
        let sc = SIMD4<Float>(sharpen, isFloat ? (headroom > 1 ? skyEncodeScalar(headroom) : 1) : 1, dither, Float(frames % 64))
        let colorFormat = color.pixelFormat
        ctx.pendingCopy = PendingCopy(texture: color) { [self] enc, depthFormat, stencilFormat in
            guard let p = copyPipe(color: colorFormat, depth: depthFormat, stencil: stencilFormat) else {
                log("post: no copy pipeline for depth \(depthFormat.rawValue), stencil \(stencilFormat.rawValue)")
                return
            }
            enc.setRenderPipelineState(p)
            enc.setFragmentTexture(output, index: 0)
            var s = sc
            enc.setFragmentBytes(&s, length: 16, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
        if !passFolding { ctx.runPending(cb) }
        frames += 1
        if frames % 1200 == 1, let e = exposure {
            let v = e.contents().bindMemory(to: SIMD4<Float>.self, capacity: 2)
            log(String(format: "post: %d frames; exposure %+.2f stops (target %+.2f, metered log2 light %.2f), from %@, tone curve %@%@",
                       frames, v[0].x, v[0].y, v[0].z, input === color ? "the frame" : "the anti-aliasing's history",
                       postToneCurves[toneCurve], headroom > 1 ? String(format: ", headroom %.2f", headroom) : ""))
        }
        return true
    }

    /// Offline check: tone curve `curve` (0-3) at `headroom` for n scene-linear RGB inputs (4 floats each) into out.
    func debugCurve(_ input: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ n: Int, headroom: Float, curve: Int) -> Bool {
        guard ensureLibrary(), let lib = library, let f = lib.makeFunction(name: "post_debug_curve"),
              let pipe = try? ctx.device.makeComputePipelineState(function: f),
              let ib = ctx.device.makeBuffer(bytes: input, length: n * 16, options: .storageModeShared),
              let ob = ctx.device.makeBuffer(length: n * 16, options: .storageModeShared),
              let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        var tone = SIMD4<Float>(max(headroom, 1), skyKnee(headroom), 0, Float(curve))
        enc.setComputePipelineState(pipe)
        enc.setBuffer(ib, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        enc.setBytes(&tone, length: 16, index: 2)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(n, 64), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        memcpy(out, ob.contents(), n * 16)
        return true
    }

    /// Offline: the exposure's state (8 floats: see post_exposure).
    func exposureState() -> [Float] {
        guard let e = exposure else { return [] }
        let v = e.contents().bindMemory(to: Float.self, capacity: 8)
        return (0..<8).map { v[$0] }
    }

    /// Offline timing: each stage in a command buffer of its own over the last frame's input (the anti-aliasing's
    /// history if there is one, else `color`), `runs` times, alternating; medians and the fastest into out: bloom,
    /// exposure, light shafts, composite, the copy into the frame (10 doubles: median, fastest for each).
    func debugTime(color: MTLTexture, depth: MTLTexture, p: UnsafePointer<Float>, runs: Int, out: UnsafeMutablePointer<Double>) -> Bool {
        guard ensureLibrary(), ensureTargets(width: color.width, height: color.height) else { return false }
        let t = Taa.shared
        let input = t.history.count == 2 && t.history[t.current].width == color.width ? t.history[t.current] : color
        var f = frameParams(color: color, p: p, cam: lastCam)
        f.effects.x = 1
        f.effects.y = Sky.shared.skyView != nil ? 1 : 0
        if f.sun.w <= 0 { f.sun.w = 1 }
        var times = [[Double]](repeating: [], count: 5)
        let stages: [Stages] = [.bloom, .exposure, .shafts, .composite]
        for _ in 0..<(runs + 3) {
            for (k, s) in stages.enumerated() {
                guard let cb = ctx.queue.makeCommandBuffer() else { return false }
                guard encode(cb: cb, input: input, depth: depth, f: f, stages: s) else { return false }
                cb.commit()
                cb.waitUntilCompleted()
                times[k].append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
            }
            guard let output, let cb = ctx.queue.makeCommandBuffer(),
                  let pipe = copyPipe(color: color.pixelFormat, depth: .invalid, stencil: .invalid) else { return false }
            let d = MTLRenderPassDescriptor()
            d.colorAttachments[0].texture = color
            d.colorAttachments[0].loadAction = .dontCare
            d.colorAttachments[0].storeAction = .store
            guard let enc = cb.makeRenderCommandEncoder(descriptor: d) else { return false }
            enc.setRenderPipelineState(pipe)
            enc.setFragmentTexture(output, index: 0)
            var sc = SIMD4<Float>(taaSharpenAmount, 1, color.pixelFormat == .rg11b10Float ? -1 : 0, 0)
            enc.setFragmentBytes(&sc, length: 16, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
            cb.commit()
            cb.waitUntilCompleted()
            times[4].append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
        }
        for k in 0..<5 {
            let s = times[k].dropFirst(3).sorted()
            out[2 * k] = s.isEmpty ? -1 : s[s.count / 2]
            out[2 * k + 1] = s.first ?? -1
        }
        return true
    }
}

/// sRGB encoding of one value (skyEncode in the shaders), for the headroom's encoded value.
private func skyEncodeScalar(_ x: Float) -> Float { x <= 0.0031308 ? 12.92 * x : 1.055 * pow(x, 1 / 2.4) - 0.055 }

/// The anti-aliasing's sharpening (METALMC_TAASHARPEN, Taa.swift), for the copy that takes the place of its own.
private let taaSharpenAmount: Float = Float(ProcessInfo.processInfo.environment["METALMC_TAASHARPEN"] ?? "") ?? 0.5

/// 1 with METALMC_EXP=post.
@_cdecl("mmc_post_enabled")
public func mmc_post_enabled() -> Int32 { postEnabled ? 1 : 0 }

/// After the anti-aliasing (or, without it, after the sky's aerial perspective): the post chain on the frame in `color`
/// (TextureBox handles; see Post.apply). p: the projection the frame holds (unjittered with anti-aliasing) and the view
/// rotation (column-major), then vanilla's sun angle: 33 floats. cam: the camera's position (3 doubles). taa: 1 if the
/// anti-aliasing ran this frame. Returns 1 if it ran.
@_cdecl("mmc_post_apply")
public func mmc_post_apply(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>, _ cam: UnsafePointer<Double>, _ taa: Int32) -> Int32 {
    guard postEnabled else { return 0 }
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    return Post.shared.apply(color: color, depth: depth, p: p, cam: SIMD3(cam[0], cam[1], cam[2]), taa: taa != 0) ? 1 : 0
}

/// Before the level goes through the air (GameRendererLodMixin, ahead of the sky's aerial perspective): the bottom of
/// vanilla's cloud slab, camera-relative (blocks), so skyLevelColor brightens the clouds by day (Sky.swift's
/// SKY_CLOUD_GAIN, times how much it's day: the sun above the horizon, less in rain). Without it they keep vanilla's
/// color, which the filmic curve takes to a light gray.
@_cdecl("mmc_post_clouds")
public func mmc_post_clouds(_ bottomRel: Float) {
    guard postEnabled, Sky.shared.ready else { return }
    let f = Sky.shared.frame
    let day = postSmoothstep(-0.05, 0.25, f.sun.y) * (1 - f.aerial.z)
    Sky.shared.frame.horizon.w = bottomRel.isFinite ? bottomRel : 0
    Sky.shared.frame.fade.w = bottomRel.isFinite ? day : 0
}

/// Offline (tools/litflow.swift): effects (bit 0 bloom, 1 light shafts, 2 eye adaptation; -1 leaves them), the tone curve
/// (0 legacy, 1 AgX, 2 ACES, 3 GT; -1 leaves it), seconds per frame for the adaptation (0: the clock's; -1 leaves it),
/// snap != 0: the next frame's exposure jumps to its target. Returns 1 with post on.
@_cdecl("mmc_debug_post")
public func mmc_debug_post(_ effects: Int32, _ curve: Int32, _ step: Double, _ snap: Int32) -> Int32 {
    guard postEnabled else { return 0 }
    let s = Post.shared
    if effects >= 0 { s.effectMask = Int(effects) }
    if curve >= 0 && Int(curve) < postToneCurves.count { s.toneCurve = Int(curve) }
    if step >= 0 { s.fixedStep = step }
    if snap != 0 { s.snap() }
    return 1
}

/// Offline: the exposure's state into out (8 floats: stops, target, metered log2 luminance, valid, the histogram's
/// total weight, its two percentiles, 0). Returns 1 if there is one.
@_cdecl("mmc_debug_post_exposure")
public func mmc_debug_post_exposure(_ out: UnsafeMutablePointer<Float>) -> Int32 {
    let v = Post.shared.exposureState()
    guard v.count == 8 else { return 0 }
    for i in 0..<8 { out[i] = v[i] }
    return 1
}

/// Offline check (tools, no game): tone curve `curve` (0 legacy, 1 AgX, 2 ACES, 3 GT) at `headroom` for n inputs of 4
/// floats (rgb, unused), exposed scene-linear light, into out (display-linear).
@_cdecl("mmc_debug_post_curve")
public func mmc_debug_post_curve(_ input: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ n: Int32, _ headroom: Float, _ curve: Int32) -> Int32 {
    guard postEnabled else { return 0 }
    return Post.shared.debugCurve(input, out, Int(n), headroom: headroom, curve: Int(curve)) ? 1 : 0
}

/// Offline timing (see Post.debugTime): out gets 10 doubles.
@_cdecl("mmc_debug_post_time")
public func mmc_debug_post_time(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>, _ runs: Int32,
                                _ out: UnsafeMutablePointer<Double>) -> Int32 {
    guard postEnabled else { return 0 }
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    return Post.shared.debugTime(color: color, depth: depth, p: p, runs: Int(max(runs, 1)), out: out) ? 1 : 0
}

/// Debug: the post chain's shader source, for an offline compile check. Returns its length.
@_cdecl("mmc_debug_post_shader_source")
public func mmc_debug_post_shader_source(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let bytes = Array(postShaderSource.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}
