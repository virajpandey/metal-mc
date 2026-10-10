import Foundation
import Metal
import simd

// Water at SEUS's level (METALMC_EXP=lit,water; docs/lighting-design.md, "Water that reflects" and "Water, round 2").
//
// Round 1 (Lit.swift) flagged the LOD's and the far field's water in the G-buffer and reflected the sky and the sun in it.
// This file adds what round 2 needs beside Lit.swift's shading:
//
// 1. Vanilla's water (the near chunks' range) in the G-buffer. Vanilla draws it with its translucent terrain pipelines
//    (translucent_terrain, translucent_terrain_multidraw: MSL that SPIRV-Cross made from vanilla's terrain.vsh/fsh on the
//    Java side). In the level's main pass those pipelines get a variant whose shaders are patched here: the vertex
//    shader passes the vertex's section-relative position and its light coordinates on, and the fragment shader writes
//    the G-buffer's water texel where its texel comes from the water sprites (the atlas rectangles of block/water_still
//    and block/water_flow, which the Java side sends when the atlas changes: the variant is compiled again then, off the
//    render thread). The texel holds the layer as it was blended (its color and alpha: the relight takes it off again to
//    see the floor), the face (top or not, from the position's derivatives), the depth key and the light levels. Glass,
//    ice and the rest of the translucent layer write "not lit terrain" (0): the relight already left whatever they cover
//    alone (their depth no longer matched the terrain's key), so nothing changes for them.
// 2. Terrain in the reflections and the floor's depth: one ray per 2 x 2 water pixels (water_trace) along the reflected
//    direction and one along the refracted one, through RtShadows' instance structure. A reflection hit is shaded as the
//    GI cache shades its bounce hits (the quad it hit read from the LOD node's buffer: material, face, block light), with
//    the cache's light at the hit (its irradiance and the share of the sun's disk its cell sees), or the sky where the
//    ray escapes. The refracted ray gives the water's depth along it (Beer-Lambert, the floor's shift).
// 3. Underwater: the camera's fluid (mmc_water_camera, from Java each frame).

/// The translucent terrain pipelines whose water goes into the G-buffer (vanilla's names, 26.3).
func waterIsTranslucentTerrain(_ name: String) -> Bool {
    name == "minecraft:pipeline/translucent_terrain" || name == "minecraft:pipeline/translucent_terrain_multidraw"
}

/// A translucent terrain pipeline's MSL as the Java side translated it.
struct WaterPipelineSource {
    let vs: String, vsEntry: String, fs: String, fsEntry: String
}

// MARK: - Vanilla's water in the G-buffer

/// The G-buffer's water texel, written by the patched fragment shader (the same packing as litPackWaterLayer in Lit.swift:
/// x the layer's color, sRGB-encoded 8 bits a channel, face in bits 24-26, bit 28 the water flag; y the depth key
/// (litDepthKey: bits 4-19 of the depth), the layer's alpha in bits 16-23, the sky light level in 24-27 and block light in
/// 28-31). The rectangles come in as macros (MMC_WATER_STILL, MMC_WATER_FLOW: u0, v0, u1, v1).
private let waterFragmentHelper = """

// MetalMC water (Water.swift): vanilla's water texels into lit mode's G-buffer (color attachment 1).
#ifndef MMC_WATER_STILL
#define MMC_WATER_STILL float4(2.0)
#define MMC_WATER_FLOW float4(2.0)
#endif
static inline bool mmcWaterIn(float2 uv, float4 r) { return all(uv >= r.xy) && all(uv < r.zw); }
static inline uint2 mmcWaterGbuffer(float4 c, float2 uv, float2 light, float z) {
    bool still = mmcWaterIn(uv, MMC_WATER_STILL);
    if (!still && !mmcWaterIn(uv, MMC_WATER_FLOW)) return uint2(0u);
    // The face: still water is only ever a top; flowing water is a top or a side (7: the relight finds which from the
    // depth buffer). Not from the position's derivatives: on the M3 they came out wrong along every block's edges and over
    // much of far water (measured in game: half the far water read as sides).
    uint face = still ? 2u : 7u;
    uint3 w = uint3(round(saturate(c.rgb) * 255.0));
    uint sky = uint(clamp(round(light.y), 0.0, 15.0)), block = uint(clamp(round(light.x), 0.0, 15.0));
    return uint2(w.r | (w.g << 8) | (w.b << 16) | (face << 24) | (1u << 28),
                 ((as_type<uint>(z) >> 4) & 0xFFFFu) | (uint(round(saturate(c.a) * 255.0)) << 16) | (sky << 24) | (block << 28));
}

"""

/// The body of a struct named `name` in SPIRV-Cross's output: the range between its opening brace's line and its closing
/// `};`.
private func waterStructRange(_ src: String, _ name: String) -> Range<String.Index>? {
    guard let start = src.range(of: "struct \(name)\n{\n") ?? src.range(of: "struct \(name)\n{\r\n") else { return nil }
    guard let end = src.range(of: "};", range: start.upperBound..<src.endIndex) else { return nil }
    return start.upperBound..<end.lowerBound
}

/// The member of a struct body declared with `attr` (e.g. "[[color(0)]]"): its name.
private func waterMember(_ body: Substring, _ attr: String) -> String? {
    for line in body.split(separator: "\n") where line.contains(attr) {
        let decl = line.split(separator: "[")[0].trimmingCharacters(in: .whitespaces)
        if let name = decl.split(separator: " ").last { return String(name) }
    }
    return nil
}

/// The member named `name` of a struct body, if it's there.
private func waterHasMember(_ body: Substring, _ name: String) -> Bool {
    body.split(separator: "\n").contains { line in
        let decl = line.split(separator: "[")[0].trimmingCharacters(in: .whitespaces)
        return decl.split(separator: " ").last.map(String.init) == name
    }
}

/// Inserts `text` before the last `return out;` of the source (SPIRV-Cross's entry point comes last and returns once).
private func waterBeforeLastReturn(_ src: String, _ text: String) -> String? {
    guard let r = src.range(of: "    return out;\n", options: .backwards) else { return nil }
    var s = src
    s.insert(contentsOf: text, at: r.lowerBound)
    return s
}

/// The vertex shader with one more output: its light coordinates, in levels. nil if the source isn't shaped as expected
/// (the pipeline then keeps its unpatched variant).
func waterPatchVertex(_ src: String) -> String? {
    guard let outR = waterStructRange(src, "main0_out"), let inR = waterStructRange(src, "main0_in") else { return nil }
    guard waterHasMember(src[inR], "UV2") else { return nil }
    var s = src
    s.insert(contentsOf: "    float2 mmcWaterLight [[user(mmcwaterlight)]];\n", at: outR.upperBound)
    return waterBeforeLastReturn(s, "    out.mmcWaterLight = float2(in.UV2) / 16.0;\n")
}

/// The fragment shader with the G-buffer's water texel as a second output (waterFragmentHelper).
func waterPatchFragment(_ src: String) -> String? {
    guard let outR0 = waterStructRange(src, "main0_out") else { return nil }
    let outBody = src[outR0]
    guard let colorName = waterMember(outBody, "[[color(0)]]"), !outBody.contains("[[color(1)]]") else { return nil }
    var s = src
    s.insert(contentsOf: "    uint2 mmcWaterGbuf [[color(1)]];\n", at: outR0.upperBound)
    guard let inR = waterStructRange(s, "main0_in"), waterHasMember(s[inR], "texCoord0") else { return nil }
    var coord = waterMember(s[inR], "[[position]]")
    var add = "    float2 mmcWaterLight [[user(mmcwaterlight)]];\n"
    if coord == nil {
        coord = "mmcWaterCoord"
        add += "    float4 mmcWaterCoord [[position]];\n"
    }
    s.insert(contentsOf: add, at: inR.upperBound)
    // The helper after the includes and `using namespace metal;`.
    guard let using = s.range(of: "using namespace metal;\n") else { return nil }
    s.insert(contentsOf: waterFragmentHelper, at: using.upperBound)
    return waterBeforeLastReturn(s, "    out.mmcWaterGbuf = mmcWaterGbuffer(out.\(colorName), in.texCoord0, in.mmcWaterLight, in.\(coord!).z);\n")
}

/// Keeps the translucent terrain pipelines and the water sprites' rectangles; compiles the patched shaders off the render
/// thread when either arrives, and hands each pipeline its functions (PipelineBox.installWater).
final class WaterNear: @unchecked Sendable {
    static let shared = WaterNear()
    private let lock = NSLock()
    private var boxes: [PipelineBox] = []
    private var rects: [Float] = []
    private var generation = 0
    private let queue = DispatchQueue(label: "metalmc.water.pipelines", qos: .userInitiated)
    private(set) var compiled = 0, failed = 0

    /// mmc_pipeline_create, for a translucent terrain pipeline (with lit mode and water).
    func register(_ box: PipelineBox) {
        lock.lock()
        boxes.append(box)
        let r = rects, g = generation
        lock.unlock()
        if !r.isEmpty { queue.async { self.compile(box, r, g) } }
    }

    /// The water sprites' rectangles (still u0 v0 u1 v1, then flow), when the block atlas changes.
    func setSprites(_ r: [Float]) {
        lock.lock()
        if r == rects { lock.unlock(); return }
        rects = r
        generation += 1
        let all = boxes, g = generation
        lock.unlock()
        log(String(format: "water: vanilla's water sprites at (%.5f %.5f)-(%.5f %.5f) and (%.5f %.5f)-(%.5f %.5f); %d translucent terrain pipelines to patch",
                   r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7], all.count))
        for b in all { queue.async { self.compile(b, r, g) } }
    }

    private func compile(_ box: PipelineBox, _ r: [Float], _ g: Int) {
        guard let src = box.waterSource else { return }
        guard let vs = waterPatchVertex(src.vs), let fs = waterPatchFragment(src.fs) else {
            lock.lock(); failed += 1; lock.unlock()
            log("water: \(box.name)'s shaders aren't shaped as expected; its water stays out of the G-buffer")
            return
        }
        let opts = MTLCompileOptions()
        opts.languageVersion = .version3_0
        opts.preserveInvariance = true
        // (No spaces: the compiler takes each macro as a command-line definition.)
        func f4(_ i: Int) -> String { String(format: "float4(%.9g,%.9g,%.9g,%.9g)", r[i], r[i + 1], r[i + 2], r[i + 3]) }
        opts.preprocessorMacros = ["MMC_WATER_STILL": f4(0) as NSString, "MMC_WATER_FLOW": f4(4) as NSString]
        do {
            let vlib = try ctx.device.makeLibrary(source: vs, options: opts)
            let flib = try ctx.device.makeLibrary(source: fs, options: opts)
            guard let vf = vlib.makeFunction(name: src.vsEntry), let ff = flib.makeFunction(name: src.fsEntry) else {
                throw NSError(domain: "metalmc", code: 3, userInfo: [NSLocalizedDescriptionKey: "entry points not found"])
            }
            lock.lock()
            let current = g == generation
            lock.unlock()
            guard current else { return }   // the atlas changed again since: that compile installs
            box.installWater(vs: vf, fs: ff)
            lock.lock(); compiled += 1; lock.unlock()
            log("water: \(box.name) writes vanilla's water into the G-buffer")
        } catch {
            lock.lock(); failed += 1; lock.unlock()
            log("water: \(box.name)'s patched shaders failed: \(error)")
        }
    }
}

/// The water sprites' atlas rectangles (8 floats: block/water_still u0 v0 u1 v1, block/water_flow u0 v0 u1 v1), from the
/// Java side when the block atlas changes. Vanilla's translucent terrain then writes its water into the G-buffer.
@_cdecl("mmc_water_sprites")
public func mmc_water_sprites(_ rects: UnsafePointer<Float>) {
    guard litWater else { return }
    WaterNear.shared.setSprites((0..<8).map { rects[$0] })
}

/// Debug: the patched vertex (which 0) or fragment (1) shader for `src`, into `out`; returns its length (0 if the source
/// isn't shaped as expected, the needed length if `len` is too small).
@_cdecl("mmc_debug_water_patch")
public func mmc_debug_water_patch(_ which: Int32, _ src: UnsafePointer<CChar>, _ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let s = String(cString: src)
    guard let p = which == 0 ? waterPatchVertex(s) : waterPatchFragment(s) else { return 0 }
    let bytes = Array(p.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count + 1) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}

// MARK: - The camera under water

/// Set each frame by the Java side (mmc_water_camera): whether the camera is in water (vanilla's fog type).
nonisolated(unsafe) var waterCameraInWater = false

@_cdecl("mmc_water_camera")
public func mmc_water_camera(_ inWater: Int32) {
    waterCameraInWater = inWater != 0
}

// MARK: - The look's settings

private func waterEnv(_ key: String) -> String? { ProcessInfo.processInfo.environment[key] }
private func waterEnvFloat(_ key: String, _ def: Double, _ lo: Double, _ hi: Double) -> Double {
    min(max(Double(waterEnv(key) ?? "") ?? def, lo), hi)
}
/// METALMC_WATERABSORB: water's absorption per block, r,g,b (Beer-Lambert; red goes first, so the floor turns teal, then
/// blue, then dark with depth).
let waterAbsorb: SIMD3<Double> = {
    let p = (waterEnv("METALMC_WATERABSORB") ?? "").split(separator: ",").compactMap { Double($0) }
    return p.count == 3 ? SIMD3(p[0], p[1], p[2]) : SIMD3(0.35, 0.10, 0.07)
}()
/// METALMC_WATERSCATTER: the water body's own light, as a share of the light falling on it (its color is the water's:
/// the biome's tint).
let waterScatter = waterEnvFloat("METALMC_WATERSCATTER", 0.035, 0, 1)
/// METALMC_WATERCAUSTICS: the caustics' strength on sunlit floors under water (0: none).
let waterCaustics = waterEnvFloat("METALMC_WATERCAUSTICS", 1, 0, 8)
/// METALMC_WATERREFRACT: how far the floor seen through the water is bent (0: not at all; 1: Snell's law).
let waterRefract = waterEnvFloat("METALMC_WATERREFRACT", 1, 0, 4)
/// METALMC_WATERRT: 0 turns the traced reflections off (the sky's map alone, as in round 1). METALMC_WATERRTSCALE: pixels
/// per traced texel along each axis (2, or 4 for a quarter of the rays).
let waterRtEnabled = litWater && waterEnv("METALMC_WATERRT") != "0"
let waterRtScale = Int(waterEnv("METALMC_WATERRTSCALE") ?? "") == 4 ? 4 : 2
/// METALMC_EXP=wet (with lit and water): in rain, lit terrain open to the sky gets wet: darker, with a film of water that
/// reflects the sky (litWaterWet).
let waterWet = litWater && experiments.contains("wet")

/// The wind's direction (degrees from +x toward +z; METALMC_WIND): the waves run with it (round 1's longest wave already
/// does, at 18 degrees). One constant for anything else the wind moves (the clouds).
let worldWindDegrees = waterEnvFloat("METALMC_WIND", 18, -360, 360)

/// The ripples' tile (item 3, close-up detail): waterDetailTile blocks a side in waterDetailTexels (32 a block), the waves
/// shorter than round 1's (1.6 down to 0.3 blocks), steeper (ripples), running with the wind and spread 80 degrees either
/// side of it; whole wave numbers across the tile so it repeats without a seam. Mipmapped like round 1's tile (each level
/// the waves it can hold and the variance of the rest), so the ripples fade into the surface's roughness with distance
/// instead of aliasing: up close they break the reflections up, far away they widen the sun's path.
/// METALMC_WATERRIPPLES scales the ripples' slopes (0: none).
let waterRippleScale = waterEnvFloat("METALMC_WATERRIPPLES", 0.5, 0, 4)
let waterDetailTile: Double = 8
let waterDetailTexels = 256
let waterDetailLevels = Int(log2(Double(waterDetailTexels))) + 1
let waterDetailSet: [(lambda: Double, angle: Double, steep: Double, phase: Double)] = [
    (1.6, 12, 0.05, 0.7), (1.25, -38, 0.05, 3.1), (0.95, 55, 0.045, 5.0), (0.75, -64, 0.045, 1.9),
    (0.58, 28, 0.04, 4.2), (0.45, -80, 0.04, 0.4), (0.36, 72, 0.035, 2.6), (0.29, -18, 0.035, 5.6)]
let waterDetailNumbers: [(n: Double, m: Double, lambda: Double)] = waterDetailSet.map { w in
    let a = (w.angle + worldWindDegrees) * .pi / 180
    let n = (waterDetailTile / w.lambda * cos(a)).rounded(), m = (waterDetailTile / w.lambda * sin(a)).rounded()
    return (n, m, waterDetailTile / (n * n + m * m).squareRoot())
}

/// Mirrors the LitFrame fields waterFrameFields adds (after water2).
struct WaterFrameGPU {
    var water3 = SIMD4<Float>.zero
    var viewProj = matrix_identity_float4x4
}

/// LitFrame's fields for water's second round (spliced after water2, with litWater only).
let waterFrameFields = """
    float4 water3;          // water: x pixels per traced texel along each axis (0: none this frame), y 1 with the camera in water, z the frame (for the waves' jitter)
    float4x4 viewProj;      // water: the projection the level was drawn with (jittered) times the view rotation: camera-relative to clip

"""

/// What the shading in the relight and the rays' kernel share: the look's settings, the waves, the sky map's coordinates,
/// Fresnel, the water texel's layer and the water's light.
let waterCommonHeader: String = {
    var waves: [String] = [], amps: [String] = []
    for (w, (n, m, _)) in zip(waterWaveSet, waterWaveNumbers) {
        let kx = 2 * Double.pi * n / waterTile, kz = 2 * Double.pi * m / waterTile
        let k = (kx * kx + kz * kz).squareRoot()
        waves.append(String(format: "float4(%.8f, %.8f, %.6f, %.6f)", kx, kz, (9.81 * k).squareRoot(), 4 * k / (2 * Double.pi)))
        let s = w.steep * waterWaveScale
        amps.append(String(format: "float4(%.8f, %.8f, %.4f, %.8f)", s * kx / k, s * kz / k, w.phase, 0.5 * s * s))
    }
    return """
// Water (Water.swift): the look's settings (lab mode: edit and save; the rays' kernel has its own copy in water_trace.metal).
#ifndef WATER_ABSORB
#define WATER_ABSORB float3(\(String(format: "%.4f, %.4f, %.4f", waterAbsorb.x, waterAbsorb.y, waterAbsorb.z)))   // per block (Beer-Lambert)
#endif
#ifndef WATER_SCATTER
#define WATER_SCATTER \(String(format: "%.4f", waterScatter))   // the water body's own light, a share of the light on it
#endif
#ifndef WATER_CAUSTICS
#define WATER_CAUSTICS \(String(format: "%.3f", waterCaustics))
#endif
#ifndef WATER_REFRACT
#define WATER_REFRACT \(String(format: "%.3f", waterRefract))
#endif
// The floor under water whose writer gave no layer to take off (the LOD's deep water, the far field's): sand, linear.
#ifndef WATER_FLOOR
#define WATER_FLOOR float3(0.45, 0.40, 0.30)
#endif
// How far the water's own color goes toward a clear sea's teal from the biome's (vanilla's blue is saturated).
#ifndef WATER_TEAL
#define WATER_TEAL 0.5
#endif
// Water's index of refraction.
#define WATER_ETA 1.333
#define WATER_TILE \(Int(waterTile)).0
#define WATER_TEXELS_PER_BLOCK \(String(format: "%.1f", Double(waterTexels) / waterTile))
#define WATER_DETAIL_TILE \(Int(waterDetailTile)).0
#define WATER_DETAIL_TEXELS_PER_BLOCK \(String(format: "%.1f", Double(waterDetailTexels) / waterDetailTile))
#define WATER_PI 3.14159265
constant float kWaterF0 = 0.02;
constant float kWaterRough2 = \(String(format: "%.6f", waterRough * waterRough));   // the surface's own roughness (GGX alpha squared) under the waves
// Per wave: wave vector (radians per block along x and z), angular frequency (radians per second), 4 over its wavelength
// (per block); its slope (radians) along x and z at a crest, phase, and its slopes' variance (half the crest's squared).
constant float4 kWaterWave[\(waves.count)] = { \(waves.joined(separator: ", ")) };
constant float4 kWaterAmp[\(waves.count)] = { \(amps.joined(separator: ", ")) };

static float litWaterFresnel(float c) {
    float m = 1.0 - saturate(c), m2 = m * m;
    return kWaterF0 + (1.0 - kWaterF0) * m2 * m2 * m;
}

// The sky map's coordinates for a direction of the upper hemisphere (a paraboloid map: the horizon is the circle of
// radius 1/2 around the middle, the zenith the middle), and back.
static float2 litWaterSkyUV(float3 dir) { return dir.xz / (1.0 + dir.y) * 0.5 + 0.5; }
static float3 litWaterSkyDir(float2 uv) {
    float2 p = uv * 2.0 - 1.0;
    float r2 = dot(p, p);
    if (r2 > 1.0) { p *= rsqrt(r2); r2 = 1.0; }   // past the horizon's circle (the corners, which bilinear taps reach): the horizon
    return float3(2.0 * p.x, 1.0 - r2, 2.0 * p.y) / (1.0 + r2);
}

// Camera-relative position of pixel q at depth z.
static float3 litWaterRelAt(float4x4 invViewProj, float2 size, uint2 q, float z) {
    float2 ndc = (float2(q) + 0.5) / size * 2.0 - 1.0;
    float4 h = invViewProj * float4(ndc, z, 1.0);
    return h.xyz / h.w;
}

// Whether the surface at pixel q (depth d) faces up, from the depth buffer as rt_shadow finds a surface's normal (on each
// axis the neighbor on the same surface: the nearer depth). For water the G-buffer leaves open (face 7: flowing water,
// a top or a side).
static bool litWaterIsTop(depth2d<float, access::read> depth, uint2 q, float d, float4x4 invViewProj, float2 size) {
    uint2 full = uint2(size);
    uint2 rx = uint2(min(q.x + 1u, full.x - 1u), q.y), lx = uint2(q.x > 0u ? q.x - 1u : 0u, q.y);
    uint2 dy = uint2(q.x, min(q.y + 1u, full.y - 1u)), uy = uint2(q.x, q.y > 0u ? q.y - 1u : 0u);
    float drx = depth.read(rx), dlx = depth.read(lx), ddy = depth.read(dy), duy = depth.read(uy);
    float3 pos = litWaterRelAt(invViewProj, size, q, d);
    float3 ex = abs(drx - d) < abs(dlx - d) ? litWaterRelAt(invViewProj, size, rx, drx) - pos : pos - litWaterRelAt(invViewProj, size, lx, dlx);
    float3 ey = abs(ddy - d) < abs(duy - d) ? litWaterRelAt(invViewProj, size, dy, ddy) - pos : pos - litWaterRelAt(invViewProj, size, uy, duy);
    float3 n = abs(cross(ey, ex));
    return n.y >= max(n.x, n.z);
}

// The waves at a point of the surface (world x and z modulo the tile: camera-relative plus the camera's offset) seen
// over a footprint of `foot` blocks, the long waves' tile and the ripples' together: the mean slope there gives the
// normal, the slopes' variance the roughness (LEAN mapping: the waves finer than the footprint, which would alias, widen
// the lobe instead).
static float3 litWaterNormal(texture2d<float> waves, texture2d<float> detail, float2 xz, float foot, thread float& rough2) {
    constexpr sampler ws(filter::linear, mip_filter::linear, address::repeat);
    float4 w = waves.sample(ws, xz * (1.0 / WATER_TILE), level(log2(max(foot * WATER_TEXELS_PER_BLOCK, 1.0))));
    float4 v = detail.sample(ws, xz * (1.0 / WATER_DETAIL_TILE), level(log2(max(foot * WATER_DETAIL_TEXELS_PER_BLOCK, 1.0))));
    rough2 = kWaterRough2 + max(w.z - dot(w.xy, w.xy), 0.0) + max(v.z - dot(v.xy, v.xy), 0.0);
    float2 s = w.xy + v.xy;
    return normalize(float3(-s.x, 1.0, -s.y));
}

// The sun's glint (GGX with Smith's height-correlated visibility, Fresnel at the half vector) for light from L of
// illuminance E (on a surface facing it), the surface's normal n and roughness (GGX alpha squared) rough2, seen from V.
static float3 litWaterGlint(float3 n, float3 V, float3 L, float rough2, float3 E) {
    float nl = dot(n, L), nv = max(dot(n, V), 1e-3);
    if (nl <= 0.0) return 0.0;
    float3 H = normalize(L + V);
    float nh = saturate(dot(n, H));
    float dd = nh * nh * (rough2 - 1.0) + 1.0;
    float D = rough2 / (WATER_PI * dd * dd);
    float G = 0.5 / (nl * sqrt(nv * nv * (1.0 - rough2) + rough2) + nv * sqrt(nl * nl * (1.0 - rough2) + rough2));
    return E * (D * G * litWaterFresnel(saturate(dot(V, H))) * nl);
}

// The water texel's layer: its color (sRGB-encoded) and alpha (0: not known).
static float4 litWaterLayer(uint2 g) {
    return float4(float(g.x & 255u), float((g.x >> 8) & 255u), float((g.x >> 16) & 255u), float((g.y >> 16) & 255u)) / 255.0;
}

// The water's own color (without its brightness) from its layer: the biome's water color, whatever the light and the
// texture's streaks (the water textures are gray, so the ratio of the channels is the tint's), taken partway toward a
// clear sea's teal. Luminance 1.
static float3 litWaterTint(float3 layer) {
    float3 lin = skyDecode(layer);
    float l = dot(lin, float3(0.2126, 0.7152, 0.0722));
    float3 biome = l > 1e-4 ? min(lin / l, float3(4.0)) : float3(0.26, 0.92, 3.9);
    return mix(biome, float3(0.30, 1.0, 1.35) / 0.893, WATER_TEAL);
}

// Vanilla's lightmap at fractional levels (texel centers at whole ones), and vanilla's brightness for a light level.
static float3 litWaterLightmap(texture2d<float> lm, float block, float sky) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return lm.sample(s, (float2(block, sky) + 0.5) / 16.0, level(0.0)).rgb;
}
static float litWaterBrightness(float level) {
    float x = saturate(level / 15.0);
    return x / (4.0 - 3.0 * x);
}
// Vanilla's light (sRGB-encoded) at whole levels: its lightmap (lm), or its brightness curve without one.
static float3 litWaterVanilla(texture2d<float> lightmap, bool lm, float block, float sky) {
    return lm ? litWaterLightmap(lightmap, block, sky) : float3(max(litWaterBrightness(block), litWaterBrightness(sky)));
}

// Our light on a horizontal water surface at sky light `sky` and block light `block` (whole levels), in the relight's units
// with its exposure `expo`: litRelightPixel's for a top face without AO. sun: the sun's illuminance on a surface facing it
// (sunIrr: lit_env's env[0]) times its visibility vSun; amb: the sky's (giAmb's where the GI cache has the pixel, w > 0;
// else the open sky's on a top, skyTop, by the sky light level's curve), the moon's glow and block light; moon: the
// moon's own light on a surface facing it (60% of vanilla's night sky light).
struct LitWaterLight { float3 sun; float3 amb; float3 moon; };
static LitWaterLight litWaterLight(texture2d<float> lightmap, bool lm, float sky, float block, float vSun, float3 sunIrr, float3 skyTop,
                                   float4 moonDir, float expo, float4 giAmb) {
    float3 lm00 = lm ? skyDecode(litWaterLightmap(lightmap, 0.0, 0.0)) : float3(0.0);
    float3 blockLight = lm ? skyDecode(litWaterLightmap(lightmap, block, 0.0)) : float3(litWaterBrightness(block) * litWaterBrightness(block));
    float3 vanillaSky = lm ? max(skyDecode(litWaterLightmap(lightmap, 0.0, 15.0)) - lm00, 0.0) : float3(1.0);
    float skyFall = skyDecode(float3(litWaterBrightness(sky))).x;
    float3 skyAmb = giAmb.w > 0.0 ? giAmb.rgb : skyTop * skyFall;
    float moonLum = dot(vanillaSky, float3(0.2126, 0.7152, 0.0722)) * moonDir.w;
    float3 tint = float3(0.857, 1.006, 1.371);
    LitWaterLight o;
    o.sun = sunIrr * (vSun * expo);
    o.amb = (skyAmb + moonLum * tint * (0.6 * saturate(moonDir.y) / max(moonDir.y, 0.5) * saturate((sky - 12.0) / 3.0) + 0.4 * skyFall) + blockLight) * expo;
    o.moon = moonLum * tint * (0.6 / max(moonDir.y, 0.5) * expo);
    return o;
}

// Caustics: how much the waves above focus the sun on a floor point (camera-relative pf, D blocks under the surface;
// tileOff: the camera's x and z modulo the tiles). The sun's refracted ray from pf back up (sunUp) crosses the surface at
// ps; the slopes' divergence there (the surface's curvature, from the slope tiles by central differences) bends the rays
// together under a crest and apart under a trough: to first order the light on the floor is 1 / (1 + D (1 - 1/1.33) div s).
// Both tiles count (the ripples make the fine net, the long waves its slow swell); each tile's level follows the depth
// (deeper, the shorter waves' focus blurs out), and the pattern fades out with depth.
static float litWaterDiv(texture2d<float> t, float2 ps, float tile, float texelsPerBlock, float lod) {
    constexpr sampler ws(filter::linear, mip_filter::linear, address::repeat);
    float e = exp2(lod) / (tile * texelsPerBlock);   // a texel of that level, in the tile's coordinates
    float2 uv = ps * (1.0 / tile);
    float sx1 = t.sample(ws, uv + float2(e, 0.0), level(lod)).x, sx0 = t.sample(ws, uv - float2(e, 0.0), level(lod)).x;
    float sz1 = t.sample(ws, uv + float2(0.0, e), level(lod)).y, sz0 = t.sample(ws, uv - float2(0.0, e), level(lod)).y;
    return (sx1 - sx0 + sz1 - sz0) / (2.0 * e * tile);   // per block
}
static float litWaterCaustic(texture2d<float> waves, texture2d<float> detail, float2 tileOff, float3 pf, float D, float3 sunUp) {
    if (WATER_CAUSTICS <= 0.0 || D <= 0.05) return 1.0;
    float2 ps = pf.xz + sunUp.xz * (D / max(sunUp.y, 0.2)) + tileOff;
    float div = litWaterDiv(waves, ps, WATER_TILE, WATER_TEXELS_PER_BLOCK, clamp(log2(1.0 + 0.35 * D), 0.0, 4.0))
              + litWaterDiv(detail, ps, WATER_DETAIL_TILE, WATER_DETAIL_TEXELS_PER_BLOCK, clamp(log2(1.0 + 2.0 * D), 0.0, 6.0));
    float k = 1.0 + WATER_CAUSTICS * min(D, 6.0) * (1.0 - 1.0 / WATER_ETA) * div;
    float c = 1.0 / max(k, 0.2);
    return mix(c * c, 1.0, saturate(D / 24.0));
}

// The packing of water_trace's second output: RG11B10Float's bits (non-negative; truncated, as the GI cache packs its
// light), and back.
static uint litWaterPack11(float x, uint mbits) {
    uint b = as_type<uint>(x);
    if ((b >> 31) != 0u || !(x > 0.0)) return 0u;
    if (b < 0x38800000u) return uint(x * float(1u << (14u + mbits)));
    return min((b - 0x38000000u) >> (23u - mbits), (31u << mbits) - 1u);
}
static uint litWaterPack(float3 c) { return litWaterPack11(c.r, 6u) | (litWaterPack11(c.g, 6u) << 11) | (litWaterPack11(c.b, 5u) << 22); }
static float3 litWaterUnpack(uint v) {
    return float3(float(as_type<half>(ushort((v & 0x7FFu) << 4))), float(as_type<half>(ushort(((v >> 11) & 0x7FFu) << 4))),
                  float(as_type<half>(ushort((v >> 22) << 5))));
}

"""
}()

// MARK: - The shading (in the relight: the anti-aliasing's resolve, or lit mode's own pass)

/// litWaterPixel's GI cache argument (with litGi): its half-resolution light, for the water's sky light where the rays'
/// kernel didn't run.
let waterGiParam = litGi ? ", texture2d<uint> gi" : ""

/// The water's shading (litWaterPixel, litWaterWet, litWaterFog), appended to litRelightHeader with litWater only.
let waterShadeHeader: String = waterCommonHeader + """
// Water (METALMC_EXP=water; Lit.swift round 1, Water.swift round 2), per water pixel in the relight, in scene-linear light
// (the sky's units, which the relit terrain is in too), before the aerial perspective:
//   body x (1 - F) + reflection x F + the sun's and the moon's glints
// F: Schlick's Fresnel reflectance of water (F0 0.02): a few percent looking down, most of the light at grazing angles.
// The rays' kernel (water_trace, Water.swift: per 2 x 2 water pixels) did the heavy part: the reflection (terrain lit as
// the GI cache lights its bounce hits, or the sky), the water's depth along the refracted ray, and the body's two terms,
// A and S: body = floor x A + S, where floor is what vanilla drew under the water's layer (the layer comes off again: the
// G-buffer holds the color and alpha that were blended over it) and A takes it to our light seen through the water
// (vanilla's light at the floor off, ours on: the sun through the water above, focused by the waves into caustics, and
// the sky's; then Beer-Lambert along the refracted ray), and S is the light the water scatters toward the camera (its
// tint). Per pixel here: the waves at the pixel (the normal, so F and the glints keep the full resolution), and the floor
// where the refracted ray meets it (the pixel whose own ray reaches that point shows it through the water too).
// Without the rays' kernel (METALMC_WATERRT=0, no GI cache, no ray tracing): the sky's map in the reflected direction,
// and the water as it was drawn, relit (its color over vanilla's light, times ours). Water seen from the side (falls) or
// from below: relit, no reflections.
#define WATER_WET \(waterWet ? 1 : 0)

// The light at a water pixel, for the paths without the rays' kernel: litWaterLight with the GI cache's light where it has
// the pixel.
static LitWaterLight litWaterLightAt(uint2 q, float3 rel, float sky, float block, float vSun, constant LitFrame& f,
                                     constant float4* env, texture2d<float> lightmap\(waterGiParam)) {
    float4 giAmb = 0.0;\(litGi ? """

    if (gi.get_width() > 1u) giAmb = giUpsample(gi, giUpsampleTexel(gi, q), q, 2u, rel);
""" : "")
    return litWaterLight(lightmap, f.misc.x > 0.5, sky, block, vSun, env[0].rgb, env[3].rgb, f.moonDir, f.misc.w, giAmb);
}

// Something drawn in vanilla's light `v` (sRGB-encoded) relit by `e` (linear): its reflectance against vanilla's light,
// times ours, as the relight does with what's blended over terrain. Linear.
static float3 litWaterRelit(float3 c, float3 v, float3 e) {
    return skyDecode(saturate(c / max(v, float3(1.0 / 255.0)))) * e;
}

// Water pixel q (color c as the relight left it, depth d, G-buffer texel g): its color with the reflections, the floor
// and the water's own light. trace, traceAS: water_trace's outputs (the reflection and the depth along the refracted ray,
// a < 0 where it has none; A and S packed); color, gbuf: the frame and the G-buffer, to look up the floor where the
// refracted ray meets it (color is a 1 x 1 stand-in in lit mode's own pass, which draws into the frame: there the floor
// isn't bent); depth: for flowing water, whether it's a top. Anything that isn't water comes back as it came.
static float3 litWaterPixel(float3 c, uint2 q, float d, uint2 g, texture2d<half, access::read> vis, constant LitFrame& f,
                            constant float4* env, texture2d<float> waves, texture2d<float> detail, texture2d<float> sky, texture2d<float> lightmap,
                            texture2d<float> trace, texture2d<uint> traceAS, texture2d<float, access::read> color, texture2d<uint, access::read> gbuf,
                            depth2d<float, access::read> depth\(waterGiParam)) {
    // Not water; reflections off (offline A/B); or no atmosphere (the daylight curve), so no sky to reflect.
    if (!litIsWater(g) || f.water.w <= 0.0 || f.misc.y > 0.5) return c;
    uint view = uint(f.misc.z);
    bool shows = litDepthMatches(d, g.y & 0xFFFFu);
    // Debug view 9: water that reflects (cyan; blue: it has its layer, so its floor shows), water with something nearer
    // drawn over it (magenta).
    if (view == 9u) return shows ? (((g.y >> 16) & 255u) > 5u ? float3(0.0, 0.3, 0.9) : float3(0.0, 0.8, 0.9)) : float3(0.9, 0.0, 0.9);
    // Something nearer was drawn over the water since (a boat, an entity, a cloud): its color stays.
    if (!shows) return c;
    float3 rel = litWaterRelAt(f.invViewProj, f.size.xy, q, d);
    float dist = length(rel);
    float3 V = -rel / max(dist, 1e-4);   // toward the camera
    float skyLevel = float((g.y >> 24) & 15u), blockLevel = float(g.y >> 28);
    // The sun's (or at night the moon's) visibility at the surface as the relight's sun term has it: traced where the
    // rays went, else the open sky.
    float vis0 = saturate((skyLevel - 12.0) / 3.0), vTraced = vis0;
    if (f.size.z > 0.0) {
        uint s = uint(f.size.z);
        vTraced = float(vis.read(min(q / s, uint2(vis.get_width() - 1, vis.get_height() - 1))).r) * saturate(skyLevel / 2.0);
    }
    float vSun = f.sunDir.w < 0.5 ? vTraced : vis0, vMoon = f.sunDir.w > 0.5 ? vTraced : vis0;
    uint face = (g.x >> 24) & 7u;
    bool top = face == 2u || (face == 7u && litWaterIsTop(depth, q, d, f.invViewProj, f.size.xy));
    float4 tr = float4(0.0, 0.0, 0.0, -1.0);
    uint2 tas = uint2(0u);
    if (top && V.y > 0.0 && f.water3.x > 0.0) {
        uint2 t = min(q / uint(f.water3.x), uint2(trace.get_width() - 1, trace.get_height() - 1));
        tr = trace.read(t);
        tas = traceAS.read(t).rg;
    }
    float3 o;
    if (!top || V.y <= 0.0 || tr.w < 0.0) {
        // A water column's side (falls, the world's edges), the surface seen from under it, or no rays here: the water as
        // it was drawn, relit (vanilla's water doesn't glow at night any more); tops get the sky's reflection in the
        // reflected direction where they're open to it, and the glints.
        LitWaterLight L = litWaterLightAt(q, rel, skyLevel, blockLevel, vSun, f, env, lightmap\(litGi ? ", gi" : ""));
        float3 surfE = L.sun * saturate(f.sunDir.y) + L.amb;
        o = litWaterRelit(c, litWaterVanilla(lightmap, f.misc.x > 0.5, blockLevel, skyLevel), surfE);
        if (top && V.y > 0.0) {
            float rough2;
            float3 n = litWaterNormal(waves, detail, rel.xz + f.water.yz, dist * f.water2.x / max(V.y, 0.02), rough2);
            float F = litWaterFresnel(max(dot(n, V), 1e-3));
            float open = skyLevel >= 15.0 ? 1.0 : skyDecode(float3(litWaterBrightness(skyLevel))).x;
            float3 R = reflect(-V, n);
            R = normalize(float3(R.x, max(R.y, 0.0), R.z));
            constexpr sampler ss(filter::linear, address::clamp_to_edge);
            o = o * (1.0 - F * open) + sky.sample(ss, litWaterSkyUV(R), level(0.0)).rgb * (F * open);
            if (f.water2.y > 0.0 && env[0].w > 0.0) o += litWaterGlint(n, V, f.sunDir.xyz, rough2, env[0].rgb * (f.water2.y / env[0].w * vSun));
            if (f.water2.z > 0.0 && f.moonDir.w > 0.0) o += litWaterGlint(n, V, f.moonDir.xyz, rough2, L.moon * (f.water2.z * vMoon));
        }
    } else {
        // The waves at the pixel's footprint on the surface (blocks, along the view, where it's longest).
        float rough2;
        float3 n = litWaterNormal(waves, detail, rel.xz + f.water.yz, dist * f.water2.x / max(V.y, 0.02), rough2);
        float F = litWaterFresnel(max(dot(n, V), 1e-3));
        float3 A = litWaterUnpack(tas.x), S = litWaterUnpack(tas.y);
        float3 body = S;
        if (any(A > 0.0)) {
            // The floor where the refracted ray meets it (tr.w blocks along it), on the screen. The waves bend it; the flat
            // surface's own bend (the floor raised toward the camera) is left as vanilla drew it.
            float4 layer = litWaterLayer(g);
            uint2 q2 = q;
            if (color.get_width() > 1u && WATER_REFRACT > 0.0) {
                float3 T = refract(-V, n, 1.0 / WATER_ETA), T0 = refract(-V, float3(0.0, 1.0, 0.0), 1.0 / WATER_ETA);
                float4 cp = f.viewProj * float4(rel - V * tr.w + (T - T0) * (tr.w * WATER_REFRACT), 1.0);
                if (cp.w > 1e-4) {
                    int2 qq = int2(floor((cp.xy / cp.w * 0.5 + 0.5) * f.size.xy));
                    if (all(qq >= int2(0)) && all(qq < int2(f.size.xy))) {
                        uint2 g2 = gbuf.read(uint2(qq)).rg;
                        uint f2 = (g2.x >> 24) & 7u;
                        if (litIsWater(g2) && ((g2.y >> 16) & 255u) > 5u && (f2 == 2u || f2 == 7u)) q2 = uint2(qq);
                    }
                }
            }
            bool moved = any(q2 != q);
            float3 c2 = moved ? color.read(q2).rgb : c;
            float4 l2 = moved ? litWaterLayer(gbuf.read(q2).rg) : layer;
            // The layer off: the floor as vanilla drew it.
            body += skyDecode(max((c2 - l2.a * l2.rgb) / max(1.0 - l2.a, 0.02), 0.0)) * A;
        }
        o = body * (1.0 - F) + tr.rgb * F;
        // The glints: the sun's (lit_env leaves the scale from its units to the relight's in env[0].w; f.water2.y: as much
        // of the sun's disk as the sky draws, none in rain) and at night the moon's (f.water2.z: none in rain).
        if (f.water2.y > 0.0 && env[0].w > 0.0) o += litWaterGlint(n, V, f.sunDir.xyz, rough2, env[0].rgb * (f.water2.y / env[0].w * vSun));
        if (f.water2.z > 0.0 && f.moonDir.w > 0.0 && dot(n, f.moonDir.xyz) > 0.0) {
            LitWaterLight L = litWaterLight(lightmap, f.misc.x > 0.5, skyLevel, blockLevel, 0.0, env[0].rgb, env[3].rgb, f.moonDir, f.misc.w, float4(0.0));
            o += litWaterGlint(n, V, f.moonDir.xyz, rough2, L.moon * (f.water2.z * vMoon));
        }
        // Debug views: 10 the reflection's share alone, 11 the body's.
        if (view == 10u) return saturate(skyEncode(tr.rgb * F));
        if (view == 11u) return saturate(skyEncode(body * (1.0 - F)));
    }
    if (view != 0u) return c;
    o = skyEncode(o);
    if (f.size.w <= 1.0) o = saturate(o);
    return max(o, 0.0);
}

// Rain on lit terrain (METALMC_EXP=wet): surfaces open to the sky (sky light 14 and up; tops most, sides a little,
// bottoms not at all) get wet as it rains (f.water2.w, the rain's strength): water in their pores darkens them (to about
// two thirds), and a film of it reflects the sky (water's Fresnel reflectance, the sky map in the mirrored direction),
// stronger in the hollows where puddles would lie (a world-space noise over the tops). c: the relit color, sRGB-encoded.
static float litWaterHash(int2 p) {
    uint h = uint(p.x) * 0x8da6b343u ^ uint(p.y) * 0xd8163841u;
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15;
    return float(h & 0xFFFFu) / 65535.0;
}
static float litWaterPuddle(float2 p) {
    float2 i = floor(p), u = p - i;
    u = u * u * (3.0 - 2.0 * u);
    int2 b = int2(i);
    return mix(mix(litWaterHash(b), litWaterHash(b + int2(1, 0)), u.x), mix(litWaterHash(b + int2(0, 1)), litWaterHash(b + int2(1, 1)), u.x), u.y);
}
static float3 litWaterWet(float3 c, uint2 q, float d, uint2 g, constant LitFrame& f, texture2d<float> sky) {
#if WATER_WET
    float rain = f.water2.w;
    uint code = g.x >> 29;
    if (rain <= 0.0 || code == 0u || code == 7u || f.misc.y > 0.5 || !litDepthMatches(d, g.y & 0xFFFFu)) return c;
    uint fi = code - 1u;
    float skyL = float((g.y >> 16) & 255u) / 16.0;
    float w = rain * saturate(skyL - 13.0) * (fi == 2u ? 1.0 : (fi == 3u ? 0.0 : 0.3));
    if (w <= 0.0) return c;
    float3 rel = litWaterRelAt(f.invViewProj, f.size.xy, q, d);
    float3 V = -normalize(rel);
    float3 n = kLitNormal[fi];
    float film = w;
    if (fi == 2u) {
        // Puddles on tops: where a noise of about 3 blocks is low, the film is a mirror; elsewhere a thin sheen.
        float2 wp = rel.xz + f.water.yz;
        float puddle = smoothstep(0.42, 0.30, litWaterPuddle(wp * 0.35) * 0.65 + litWaterPuddle(wp * 1.3) * 0.35);
        film = w * mix(0.35, 1.0, puddle);
    }
    float3 R = reflect(-V, n);
    R = normalize(float3(R.x, max(R.y, 0.0), R.z));
    constexpr sampler ss(filter::linear, address::clamp_to_edge);
    float F = litWaterFresnel(saturate(dot(n, V)));
    float3 lin = skyDecode(max(c, 0.0)) * (1.0 - 0.35 * w) * (1.0 - F * film) + sky.sample(ss, litWaterSkyUV(R), level(0.0)).rgb * (F * film);
    float3 o = skyEncode(lin);
    return f.size.w <= 1.0 ? saturate(o) : max(o, 0.0);
#else
    return c;
#endif
}

// The camera under water: everything it sees is seen through the water between (Beer-Lambert over the distance, up to
// 96 blocks; the sky through 32), with the light the water scatters toward it, which is the light under the surface
// tinted by the water. c: the pixel's color as the relight left it, sRGB-encoded.
static float3 litWaterFog(float3 c, uint2 q, float d, constant LitFrame& f, constant float4* env, texture2d<float> lightmap) {
    if (f.water3.y < 0.5 || f.water.w <= 0.0) return c;
    float dist = 32.0;
    if (d > 0.0) {
        float2 ndc = (float2(q) + 0.5) / f.size.xy * 2.0 - 1.0;
        float4 h = f.invViewProj * float4(ndc, d, 1.0);
        dist = min(length(h.xyz / h.w), 96.0);
    }
    float3 T = exp(-WATER_ABSORB * dist);
    // The light a few blocks under the surface: the sun's and the sky's (without our sky, the daylight curve's, scaled
    // by vanilla's daylight as the relight scales it).
    float dayScale = 1.0;
    if (f.misc.y > 0.5 && f.misc.x > 0.5) {
        float3 lm00 = skyDecode(litLightmap(lightmap, 0.0, 0.0));
        dayScale = litLuma(max(skyDecode(litLightmap(lightmap, 0.0, 15.0)) - lm00, 0.0));
    }
    float3 E = (env[0].rgb * saturate(f.sunDir.y) + env[3].rgb) * (dayScale * f.misc.w) * exp(-WATER_ABSORB * 4.0);
    float3 lin = skyDecode(max(c, 0.0)) * T + float3(0.26, 0.92, 3.9) * (WATER_SCATTER * E) * (1.0 - T);
    float3 o = skyEncode(lin);
    return f.size.w <= 1.0 ? saturate(o) : max(o, 0.0);
}
"""

// MARK: - The ripples' tile (water_ripples)

private let waterRipplesSource: String = {
    var waves: [String] = [], amps: [String] = []
    for (w, (n, m, _)) in zip(waterDetailSet, waterDetailNumbers) {
        let kx = 2 * Double.pi * n / waterDetailTile, kz = 2 * Double.pi * m / waterDetailTile
        let k = (kx * kx + kz * kz).squareRoot()
        waves.append(String(format: "float4(%.8f, %.8f, %.6f, %.6f)", kx, kz, (9.81 * k + 0.074 / 1000 * k * k * k).squareRoot(), 4 * k / (2 * Double.pi)))
        let s = w.steep * waterRippleScale
        amps.append(String(format: "float4(%.8f, %.8f, %.4f, %.8f)", s * kx / k, s * kz / k, w.phase, 0.5 * s * s))
    }
    return """
#include <metal_stdlib>
using namespace metal;

// The ripples (Water.swift): one mip level of their tile at time t, one thread a texel, as lit_water_waves makes round
// 1's (Lit.swift): the slopes of the ripples the level can hold, each whole while its wavelength spans 4 of its texels and
// gone at 2, and the slopes' mean squared length, which adds the variance of the ripples faded out.
constant float4 kRippleWave[\(waves.count)] = { \(waves.joined(separator: ", ")) };
constant float4 kRippleAmp[\(waves.count)] = { \(amps.joined(separator: ", ")) };

// The ripples' slopes, scaled (lab mode: edit and save).
#ifndef RIPPLE_GAIN
#define RIPPLE_GAIN 1.0
#endif

kernel void water_ripples(texture2d<float, access::write> out [[texture(0)]], constant float& t [[buffer(0)]],
                          uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= out.get_width() || gid.y >= out.get_height()) return;
    float spacing = \(Int(waterDetailTile)).0 / float(out.get_width());
    float2 p = (float2(gid) + 0.5) * spacing;
    float2 s = 0.0;
    float faded = 0.0;
    for (uint i = 0; i < \(waves.count)u; i++) {
        float4 w = kRippleWave[i], a = kRippleAmp[i];
        float fade = saturate(2.0 - spacing * w.w);
        s += (fade * cos(dot(w.xy, p) - w.z * t + a.z)) * a.xy;
        faded += a.w * (1.0 - fade * fade);
    }
    s *= RIPPLE_GAIN;
    out.write(float4(s, dot(s, s) + faded * (RIPPLE_GAIN * RIPPLE_GAIN), 0.0), gid);
}
"""
}()

/// The ripples' tile, made each frame with round 1's waves (Lit.relight).
final class WaterRipples: @unchecked Sendable {
    static let shared = WaterRipples()
    private var pipe: MTLComputePipelineState?
    private var failed = false
    private var tile: MTLTexture?
    private var levels: [MTLTexture] = []
    private var stillMade = false
    /// The tile, once it has been made (nil before: the relight binds a stand-in).
    private(set) var texture: MTLTexture?

    /// Levels whose texels are at least half the longest ripple apart hold none (every ripple faded): made once.
    private let moving: Int = {
        let longest = waterDetailNumbers.map { $0.lambda }.max() ?? 1
        return (0..<waterDetailLevels).first { waterDetailTile / Double(waterDetailTexels >> $0) >= longest / 2 } ?? waterDetailLevels
    }()

    func encode(cb: MTLCommandBuffer, time: Float) {
        if failed { return }
        if pipe == nil {
            do {
                let lib = try ShaderLab.library("water_ripples", waterRipplesSource) { [self] _ in pipe = nil; stillMade = false; failed = false }
                pipe = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "water_ripples")!)
                let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: waterDetailTexels, height: waterDetailTexels, mipmapped: true)
                td.usage = [.shaderRead, .shaderWrite]
                td.storageMode = .private
                tile = ctx.device.makeTexture(descriptor: td)
                tile?.label = "MetalMC water ripples"
                levels = (0..<waterDetailLevels).compactMap {
                    tile?.makeTextureView(pixelFormat: .rgba16Float, textureType: .type2D, levels: $0..<($0 + 1), slices: 0..<1)
                }
            } catch {
                log("water: the ripples' kernel failed: \(error)")
                failed = true
                return
            }
        }
        guard let pipe, levels.count == waterDetailLevels, let enc = cb.makeComputeCommandEncoder(dispatchType: .concurrent) else { return }
        enc.label = "MetalMC water ripples"
        var t = time
        enc.setComputePipelineState(pipe)
        enc.setBytes(&t, length: 4, index: 0)
        for (level, view) in levels.enumerated() where level < moving || !stillMade {
            let n = waterDetailTexels >> level, g = min(n, 16)
            enc.setTexture(view, index: 0)
            enc.dispatchThreads(MTLSize(width: n, height: n, depth: 1), threadsPerThreadgroup: MTLSize(width: g, height: g, depth: 1))
        }
        enc.endEncoding()
        stillMade = true
        texture = tile
    }
}

// MARK: - Terrain in the reflections, and the water's depth (water_trace)

/// RtShadows' instance structure this frame, for water's rays: its tiles' structures, the camera, the structure's origin
/// and (with the GI cache) its GiTile table and the node buffers it points into.
struct WaterStructure {
    let tlas: MTLAccelerationStructure
    let accels: [MTLAccelerationStructure]
    let cam: SIMD3<Double>
    let origin: SIMD3<Double>
    let gi: (table: MTLBuffer, buffers: [MTLBuffer])?
}

private let waterTraceKernel = """

// Water's rays (Water.swift): per 2 x 2 water pixels (a different one each frame, or the first that is water: shores), one
// ray along the reflected direction and one along the refracted one, through RtShadows' instance structure (the LOD's
// blocks, near terrain included; water itself isn't in it). A reflection hit is lit as the GI cache lights its bounce hits:
// the quad it hit, read from the LOD node's buffer (GI_HIT: material, face, block light, sky cover), its albedo times the
// sun (the share of the sun's disk its cell sees) plus the cache's irradiance there plus block light, in the relight's
// units; then the air along the reflected path (the aerial perspective's for that distance in that direction). Rays that
// escape take the sky's map. Out: rgb the reflected light, a the depth along the refracted ray (-1: no water pixel here).
struct WaterTraceParams {
    float4x4 invViewProj;   // clip space to camera-relative world (as the level was drawn: jittered)
    float4 size;            // full width, height; traced width, height
    float4 camOffset;       // xyz: the camera relative to the instance structure's origin
    float4 sunDir;          // xyz: toward the sun, w: 1 with the GI cache's light
    float4 moonDir;         // xyz: toward the moon, w: how much it's night
    float4 water;           // x: the waves' time, y-z: the camera's x and z modulo the tile, w: radians per pixel
    float4 limits;          // x: reflection rays' length, y: refraction rays' length, z: exposure, w: 1 with the lightmap
    float4 vis;             // x: pixels per traced shadow sample (0: none), y: 1 if traced toward the moon
    uint4 sample;           // x: pixels per traced texel along each axis, y-z: this frame's pixel of the block, w: the frame
};

static float3 wtRel(constant WaterTraceParams& p, uint2 q, float z) {
    float2 ndc = (float2(q) + 0.5) / p.size.xy * 2.0 - 1.0;
    float4 h = p.invViewProj * float4(ndc, z, 1.0);
    return h.xyz / h.w;
}
static float3 wtLightmap(texture2d<float> lm, float block, float sky) {
    constexpr sampler s(filter::linear, address::clamp_to_edge);
    return lm.sample(s, (float2(block, sky) + 0.5) / 16.0, level(0.0)).rgb;
}
static float wtBrightness(float level) {
    float x = saturate(level / 15.0);
    return x / (4.0 - 3.0 * x);
}
static uint wtHash(uint2 q, uint f) {
    uint h = (q.x * 0x8da6b343u) ^ (q.y * 0xd8163841u) ^ (f * 0xcb1ab31fu);
    h ^= h >> 16; h *= 0x7feb352du; h ^= h >> 15; h *= 0x846ca68bu; h ^= h >> 16;
    return h;
}

// The GI cache's light on a face at f (relative to the camera's block), as gi_resolve finds it: bilinear over the 2 x 2
// nearest cells on the face's plane at the level for its distance (cells without samples left out). w: the share of the
// sun's disk those cells see; -1 if none has samples.
static float4 wtCellLight(constant GiParams& p, const device uint* check, const device half4* value, float3 f, uint face) {
    uint level = min(uint(giLevelF(p, length(f - p.camFrac.xyz))), GI_MAX_LEVEL);
    int3 air = giAirBlock(p.camBlock.xyz, f, face);
    uint2 t = giTangents(face);
    int s = 1 << level;
    int3 c = giCellOf(air, level);
    float3 local = float3(p.camBlock.xyz - giCorner(c, level)) + f;
    float2 qq = float2(local[t.x], local[t.y]) / float(s) - 0.5;
    float2 fl = floor(qq), w = qq - fl;
    int2 o = int2(fl);
    int e[4];
    for (int k = 0; k < 4; k++) {
        int3 cc = c;
        cc[t.x] += o.x + (k & 1);
        cc[t.y] += o.y + (k >> 1);
        uint2 key = giKey(cc, face, level);
        e[k] = giFind(check, giBucketOf(p, key), giPrint(key));
    }
    float3 sum = 0.0;
    float sun = 0.0, ws = 0.0;
    for (int k = 0; k < 4; k++) {
        float wt = ((k & 1) == 0 ? 1.0 - w.x : w.x) * ((k >> 1) == 0 ? 1.0 - w.y : w.y);
        if (wt <= 0.0 || e[k] < 0) continue;
        half4 v = value[e[k]];
        if (!giSampled(v)) continue;
        sum += wt * float3(v.rgb);
        sun += wt * (float(v.a) - 1.0);
        ws += wt;
    }
    return ws > 0.0 ? float4(sum / ws, sun / ws) : float4(0.0, 0.0, 0.0, -1.0);
}

kernel void water_trace(instance_acceleration_structure accel [[buffer(0)]],
                        constant WaterTraceParams& p [[buffer(1)]],
                        constant GiParams& gp [[buffer(2)]],
                        const device uint* check [[buffer(3)]],
                        const device half4* value [[buffer(4)]],
                        constant float4* mats [[buffer(5)]],
                        constant GiLight& L [[buffer(6)]],
                        constant float4* env [[buffer(7)]],
                        const device GiTile* tiles [[buffer(8)]],
                        constant SkyFrame& sf [[buffer(9)]],
                        depth2d<float, access::read> depth [[texture(0)]],
                        texture2d<uint, access::read> gbuf [[texture(1)]],
                        texture2d<float> waves [[texture(2)]],
                        texture2d<float> sky [[texture(3)]],
                        texture2d<float> lightmap [[texture(4)]],
                        texture2d<float, access::write> out [[texture(5)]],
                        texture3d<float> apScatter [[texture(6)]],
                        texture3d<float> apTrans [[texture(7)]],
                        texture2d<float> wavesDetail [[texture(8)]],
                        texture2d<uint, access::write> outAS [[texture(9)]],
                        texture2d<half, access::read> vis [[texture(10)]],
                        texture2d<uint> gi [[texture(11)]],
                        uint2 gid [[thread_position_in_grid]],
                        uint2 lid [[thread_position_in_threadgroup]],
                        ushort lane [[thread_index_in_simdgroup]]) {
    // (No early returns before the refracted rays' exchange below: every thread of a SIMD group takes part in it.)
    bool inside = gid.x < uint(p.size.z) && gid.y < uint(p.size.w);
    uint2 full = uint2(p.size.xy);
    uint sc = p.sample.x, n = sc * sc, start = p.sample.y + p.sample.z * sc;
    // This frame's pixel of the block, else the first of the others that is a water top seen from above (shores).
    uint2 q = uint2(0), g = uint2(0);
    float d = 0.0;
    bool found = false;
    for (uint k = 0; k < n && !found && inside; k++) {
        uint i = (start + k) % n;
        uint2 c = min(gid * sc + uint2(i % sc, i / sc), full - 1u);
        uint2 gc = gbuf.read(c).rg;
        uint fc = (gc.x >> 24) & 7u;
        if (!litIsWater(gc) || (fc != 2u && fc != 7u)) continue;
        float dc = depth.read(c);
        if (!litDepthMatches(dc, gc.y & 0xFFFFu)) continue;
        if (fc == 7u && !litWaterIsTop(depth, c, dc, p.invViewProj, p.size.xy)) continue;
        q = c;
        g = gc;
        d = dc;
        found = true;
    }
    float3 rel = found ? wtRel(p, q, d) : float3(0.0, 1.0, 0.0);
    float dist = length(rel);
    float3 V = -rel / max(dist, 1e-4);
    bool valid = found && V.y > 0.0;   // a water top seen from above
    float rough2 = kWaterRough2;
    float3 nrm = float3(0.0, 1.0, 0.0);
    if (valid) nrm = litWaterNormal(waves, wavesDetail, rel.xz + p.water.yz, dist * p.water.w / max(V.y, 0.02), rough2);
    float3 T = refract(-V, nrm, 1.0 / WATER_ETA);
    if (dot(T, T) < 1e-6) T = -V;

    // The refracted ray: how far it goes through the water to the floor (the water itself isn't in the structure). At half
    // resolution one ray per 2 x 2 texels (a different one each frame), shared: the floor's depth changes slowly, and the
    // refracted rays were half the rays' cost (1.15 of 2.3 ms at water_closeup, measured).
    intersector<instancing> floorRay;
    floorRay.assume_geometry_type(geometry_type::triangle);
    uint ox = p.sample.w & 1u, oy = (p.sample.w >> 1) & 1u;
    bool share = sc == 2u;
    bool lead = !share || ((lid.x & 1u) == ox && (lid.y & 1u) == oy);
    float thick = -1.0;
    if (valid && lead) {
        thick = p.limits.y;
        ray tray(rel - float3(0.0, 0.02, 0.0) + p.camOffset.xyz, T, 0.0, p.limits.y);
        auto th = floorRay.intersect(tray, accel, 0xFF);
        if (th.type != intersection_type::none) thick = th.distance;
    }
    if (share) {
        // The leader's lane: the threadgroup is 8 texels wide, and a SIMD group four of its rows.
        ushort leader = ushort((((lid.y & ~1u) | oy) & 3u) * 8u + ((lid.x & ~1u) | ox));
        float shared = simd_shuffle(thick, leader);
        if (!lead) thick = shared;
    }
    if (!inside) return;
    if (!valid) {
        out.write(float4(0.0, 0.0, 0.0, -1.0), gid);
        outAS.write(uint4(0u), gid);
        return;
    }
    if (thick < 0.0) {
        // The leader had no water (a shore): this texel's own ray.
        thick = p.limits.y;
        ray tray(rel - float3(0.0, 0.02, 0.0) + p.camOffset.xyz, T, 0.0, p.limits.y);
        auto th = floorRay.intersect(tray, accel, 0xFF);
        if (th.type != intersection_type::none) thick = th.distance;
    }
    float4 layer = litWaterLayer(g);
    float skyLevel = float((g.y >> 24) & 15u), blockLevel = float(g.y >> 28);
    bool lm = p.limits.w > 0.5;

    // The reflection: the first surface along the reflected ray, lit as the GI cache lights its bounce hits; the sky's map
    // where it escapes.
    float3 R = reflect(-V, nrm);
    if (R.y < 0.002) R = normalize(float3(R.x, 0.002, R.z));
    // From just over the surface (further with distance: depth precision); not the first stretch (a coarse level's coast
    // can reach over its own water).
    float3 o = rel + float3(0.0, 0.03 + dist * 0.0004, 0.0);
    intersector<triangle_data, instancing> isect;
    isect.set_triangle_front_facing_winding(winding::clockwise);
    ray r(o + p.camOffset.xyz, R, 0.02 + dist * 0.002, p.limits.x);
    auto hit = isect.intersect(r, accel, 0xFF);
    constexpr sampler ss(filter::linear, address::clamp_to_edge);
    float3 col;
    if (hit.type == intersection_type::none || !hit.triangle_front_facing) {
        // Escaped; or the back of a surface: the ray started inside terrain (a coarse level's coast): the sky.
        col = sky.sample(ss, litWaterSkyUV(R), level(0.0)).rgb;
    } else {
        uint pd = GI_HIT(hit);
        uint m = pd & 255u, hf = min((pd >> 8) & 7u, 5u), bl = (pd >> 11) & 15u, cover = (pd >> 15) & 15u;
        float3 hrel = o + R * hit.distance;
        float3 hn = kGiNormal[hf];
        float skyLv = float(15u - min(cover, 15u));
        float skyFall = skyDecode(float3(litWaterBrightness(skyLv))).x;
        // The cache's light at the hit: its irradiance (the sky as its openings let it in, and bounced light) and the
        // share of the sun's disk its cells see; else the open sky's by the sky light level and a ray toward the sun.
        float4 cl = p.sunDir.w > 0.5 ? wtCellLight(gp, check, value, gp.camFrac.xyz + hrel, hf) : float4(0.0, 0.0, 0.0, -1.0);
        float ndl = dot(hn, p.sunDir.xyz);
        float3 skyE;
        float sunVis = 0.0;
        if (cl.w >= 0.0) {
            skyE = cl.rgb * L.scale.rgb;
            sunVis = saturate(cl.w);
        } else {
            skyE = env[1u + hf].rgb * skyFall;
            if (ndl > 0.0 && max(env[0].r, max(env[0].g, env[0].b)) > 0.0) {
                intersector<instancing> shadow;
                shadow.accept_any_intersection(true);
                shadow.assume_geometry_type(geometry_type::triangle);
                ray sr(hrel + hn * 0.03 + p.camOffset.xyz, p.sunDir.xyz, 0.0, 2000.0);
                sunVis = shadow.intersect(sr, accel, 0xFF).type == intersection_type::none ? 1.0 : 0.0;
            }
        }
        float3 blockE, moon = 0.0;
        if (lm) {
            float3 lm00 = skyDecode(litWaterLightmap(lightmap, 0.0, 0.0));
            blockE = skyDecode(litWaterLightmap(lightmap, float(bl), 0.0));
            float moonLum = dot(max(skyDecode(litWaterLightmap(lightmap, 0.0, 15.0)) - lm00, 0.0), float3(0.2126, 0.7152, 0.0722)) * p.moonDir.w;
            moon = moonLum * float3(0.857, 1.006, 1.371) * (0.6 * max(dot(hn, p.moonDir.xyz), 0.0) / max(p.moonDir.y, 0.5) * saturate((skyLv - 12.0) / 3.0) + 0.4 * skyFall);
        } else {
            float b = litWaterBrightness(float(bl));
            blockE = float3(b * b);
        }
        float3 E = (env[0].rgb * (max(ndl, 0.0) * sunVis) + skyE + moon + blockE) * p.limits.z;
        if (bl >= 15u) E = max(E, float3(1.0));   // light sources: full bright, as the relight has them
        col = mats[m * 4u + giFaceClass(hf)].rgb * E;
        // The air along the reflected path.
        SkyAerial ap = skyAerialPerspective(R * hit.distance, sf, apScatter, apTrans);
        col = col * ap.transmittance + skyOvercast(sf, ap.inscatter) + skyNight(sf, R) * (1.0 - ap.transmittance);
    }

    // The light at the surface: the sun's visibility as the relight's sun term has it (traced where the shadow rays went
    // toward the sun, else the open sky), the GI cache's light where it has the pixel.
    float vSun = saturate((skyLevel - 12.0) / 3.0);
    if (p.vis.x > 0.0 && p.vis.y < 0.5) {
        uint s = uint(p.vis.x);
        vSun = float(vis.read(min(q / s, uint2(vis.get_width() - 1, vis.get_height() - 1))).r) * saturate(skyLevel / 2.0);
    }
    float4 giAmb = gi.get_width() > 1u ? giUpsample(gi, giUpsampleTexel(gi, q), q, 2u, rel) : float4(0.0);
    LitWaterLight Lw = litWaterLight(lightmap, lm, skyLevel, blockLevel, vSun, env[0].rgb, env[3].rgb, p.moonDir, p.limits.z, giAmb);
    float3 surfE = Lw.sun * saturate(p.sunDir.y) + Lw.amb;
    // The floor, D blocks under the surface: the sun through the water above it (along its refracted path, focused by the
    // waves), the sky's through the water straight above. Then seen through `thick` blocks of water, with the light the
    // water scatters toward the camera on the way.
    float D = thick * max(-T.y, 0.05);
    float sinSun2 = 1.0 - p.sunDir.y * p.sunDir.y;
    float3 sunUp = normalize(float3(p.sunDir.x, sqrt(max(WATER_ETA * WATER_ETA - sinSun2, 0.0)), p.sunDir.z));
    float caustic = p.sunDir.y > 0.0 ? litWaterCaustic(waves, wavesDetail, p.water.yz, rel + T * thick, D, sunUp) : 1.0;
    float3 floorE = Lw.sun * (saturate(p.sunDir.y) * caustic) * exp(-WATER_ABSORB * (D / max(sunUp.y, 0.2))) + Lw.amb * exp(-WATER_ABSORB * D);
    float3 viewT = exp(-WATER_ABSORB * thick);
    float3 S = litWaterTint(layer.rgb) * (WATER_SCATTER * (1.0 - viewT)) * surfE;
    float3 A = 0.0;
    if (layer.a > 0.02 && layer.a < 0.98) {
        // The floor as vanilla drew it (in vanilla's light, which water dims a sky light level a block) to ours.
        float3 floorV = skyDecode(litWaterVanilla(lightmap, lm, blockLevel, max(skyLevel - floor(D), 0.0)));
        A = viewT * floorE / max(floorV, float3(1e-4));
    } else {
        // No layer to take off (the LOD's deep water, drawn opaque; the far field's): a sandy floor.
        S += WATER_FLOOR * floorE * viewT;
    }
    out.write(float4(max(col, 0.0), thick), gid);
    outAS.write(uint4(litWaterPack(A), litWaterPack(S), 0u, 0u), gid);
}
"""

/// Water's rays (water_trace) each frame, at a half (or a quarter) of the resolution, for the relight's water shading.
final class WaterTrace: @unchecked Sendable {
    static let shared = WaterTrace()
    private var pipe: MTLComputePipelineState?
    private var failed = false
    private var out: MTLTexture?
    private var outAS: MTLTexture?
    private var standIn: MTLTexture?
    private var standInAS: MTLTexture?
    private var colorStandIn: MTLTexture?
    private var frame: UInt32 = 0
    private var traces = 0
    /// This frame's outputs (set by trace, for the relight: Lit.bindDeferred, Lit's own pass).
    private(set) var current: MTLTexture?
    private(set) var currentAS: MTLTexture?

    /// The texture the relight binds: this frame's rays, or a 1 x 1 stand-in (the relight then has none: water3.x 0).
    var texture: MTLTexture? { current ?? standInTexture() }
    var textureAS: MTLTexture? {
        if let currentAS { return currentAS }
        if standInAS == nil {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Uint, width: 1, height: 1, mipmapped: false)
            d.usage = .shaderRead
            standInAS = ctx.device.makeTexture(descriptor: d)
        }
        return standInAS
    }

    private func standInTexture() -> MTLTexture? {
        if let standIn { return standIn }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
        d.usage = .shaderRead
        standIn = ctx.device.makeTexture(descriptor: d)
        return standIn
    }

    /// Lit mode's own pass: a 1 x 1 stand-in for the frame (it draws into it, so the floor isn't bent there).
    var colorStandInTexture: MTLTexture? {
        if let colorStandIn { return colorStandIn }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1, height: 1, mipmapped: false)
        d.usage = .shaderRead
        colorStandIn = ctx.device.makeTexture(descriptor: d)
        return colorStandIn
    }

    private func ensure() -> Bool {
        if failed { return false }
        if pipe != nil { return true }
        let src = waterTraceSource
        do {
            // Lab mode (ShaderLab.swift): after an edit of water_trace.metal the kernel is built again.
            let lib = try ShaderLab.library("water_trace", src) { [self] _ in pipe = nil; failed = false }
            guard let f = lib.makeFunction(name: "water_trace") else { failed = true; return false }
            pipe = try ctx.device.makeComputePipelineState(function: f)
            return true
        } catch {
            log("water: the trace kernel failed: \(error)")
            failed = true
            return false
        }
    }

    /// Lit.relight with water, our sky and the waves' tile and sky map made: this frame's rays, on RtShadows' structure
    /// (taken: it ran before in the frame), the frame's depth and G-buffer. Returns the pixels per traced texel, 0 if it
    /// didn't run (no structure, no GI cache, the switch off).
    func trace(cb: MTLCommandBuffer, depth: MTLTexture, gbuffer: MTLTexture, invViewProj: simd_float4x4, sunDir: SIMD3<Float>,
               moonDir: SIMD4<Float>, water: SIMD4<Float>, radiansPerPixel: Float, exposure: Float, env: MTLBuffer,
               waves: MTLTexture, sky: MTLTexture, lightmap: MTLTexture?, lightmapStandIn: MTLTexture,
               vis: (texture: MTLTexture, scale: Float, moon: Bool), giLight: MTLTexture?, giStandIn: MTLTexture) -> Int {
        current = nil
        currentAS = nil
        let st = RtShadows.shared.takeWaterStructure()
        guard waterRtEnabled, ctx.device.supportsRaytracing, let st, let gi = st.gi, let cache = GiCache.shared,
              Sky.shared.ready, let apScatter = Sky.shared.apScatter, let apTrans = Sky.shared.apTrans, ensure(), let pipe else { return 0 }
        let sc = waterRtScale
        let w = depth.width, h = depth.height, tw = (w + sc - 1) / sc, th = (h + sc - 1) / sc
        if out == nil || out!.width != tw || out!.height != th {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: tw, height: th, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            out = ctx.device.makeTexture(descriptor: d)
            out?.label = "MetalMC water rays"
            let d2 = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg32Uint, width: tw, height: th, mipmapped: false)
            d2.usage = [.shaderRead, .shaderWrite]
            d2.storageMode = .private
            outAS = ctx.device.makeTexture(descriptor: d2)
            outAS?.label = "MetalMC water body"
        }
        guard let out, let outAS else { return 0 }
        frame &+= 1
        struct Params {
            var invViewProj: simd_float4x4
            var size: SIMD4<Float>
            var camOffset: SIMD4<Float>
            var sunDir: SIMD4<Float>
            var moonDir: SIMD4<Float>
            var water: SIMD4<Float>
            var limits: SIMD4<Float>
            var vis: SIMD4<Float>
            var sample: SIMD4<UInt32>
        }
        let off = st.cam - st.origin
        let pat = waterPattern[Int(frame % 16)]
        var p = Params(invViewProj: invViewProj, size: SIMD4(Float(w), Float(h), Float(tw), Float(th)),
                       camOffset: SIMD4(Float(off.x), Float(off.y), Float(off.z), 0), sunDir: SIMD4(sunDir, 1),
                       moonDir: moonDir, water: SIMD4(water.x, water.y, water.z, radiansPerPixel),
                       limits: SIMD4(1024, 48, exposure, lightmap != nil ? 1 : 0), vis: SIMD4(vis.scale, vis.moon ? 1 : 0, 0, 0),
                       sample: SIMD4(UInt32(sc), pat.x % UInt32(sc), pat.y % UInt32(sc), frame))
        // The cache's lookups: its parameters as its kernels have them this frame (the camera's block, the level-0 range,
        // the table's buckets, the structure's origin).
        var gp = GiParams()
        let camFloor = st.cam.rounded(.down)
        gp.origin = SIMD4(Int32(truncatingIfNeeded: Int(st.origin.x)), Int32(Int(st.origin.y)), Int32(truncatingIfNeeded: Int(st.origin.z)), cache.frame)
        gp.camBlock = SIMD4(Int32(truncatingIfNeeded: Int(camFloor.x)), Int32(Int(camFloor.y)), Int32(truncatingIfNeeded: Int(camFloor.z)),
                            Int32(cache.slots / giBucketSlots - 1))
        gp.camFrac = SIMD4(Float(st.cam.x - camFloor.x), Float(st.cam.y - camFloor.y), Float(st.cam.z - camFloor.z), cache.level0Range)
        var skyFrame = Sky.shared.frame
        guard let enc = cb.makeComputeCommandEncoder(descriptor: profComputePass("water rays")) else { return 0 }
        enc.label = "MetalMC water rays"
        enc.setComputePipelineState(pipe)
        enc.setAccelerationStructure(st.tlas, bufferIndex: 0)
        enc.useResources(st.accels, usage: .read)
        enc.setBytes(&p, length: MemoryLayout<Params>.stride, index: 1)
        enc.setBytes(&gp, length: MemoryLayout<GiParams>.stride, index: 2)
        enc.setBuffer(cache.check, offset: 0, index: 3)
        enc.setBuffer(cache.value, offset: 0, index: 4)
        enc.setBuffer(cache.mats, offset: 0, index: 5)
        enc.setBuffer(cache.light, offset: 0, index: 6)
        enc.setBuffer(env, offset: 0, index: 7)
        enc.setBuffer(gi.table, offset: 0, index: 8)
        enc.useResources(gi.buffers, usage: .read)
        enc.setBytes(&skyFrame, length: MemoryLayout<SkyFrameGPU>.stride, index: 9)
        enc.setTexture(depth, index: 0)
        enc.setTexture(gbuffer, index: 1)
        enc.setTexture(waves, index: 2)
        enc.setTexture(WaterRipples.shared.texture ?? waves, index: 8)
        enc.setTexture(sky, index: 3)
        enc.setTexture(lightmap ?? lightmapStandIn, index: 4)
        enc.setTexture(out, index: 5)
        enc.setTexture(apScatter, index: 6)
        enc.setTexture(apTrans, index: 7)
        enc.setTexture(outAS, index: 9)
        enc.setTexture(vis.texture, index: 10)
        enc.setTexture(giLight ?? giStandIn, index: 11)
        // Whole 8 x 8 threadgroups (the kernel's exchange of refracted rays counts on rows of 8 threads).
        enc.dispatchThreadgroups(MTLSize(width: (tw + 7) / 8, height: (th + 7) / 8, depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
        current = out
        currentAS = outAS
        traces += 1
        if traces == 1 { log("water: reflection and refraction rays at \(tw)x\(th) (one per \(sc) x \(sc) pixels of water)") }
        return sc
    }
}

/// water_trace's library: the GI cache's kernels (its hit decode and cell lookups), lit mode's G-buffer header, water's.
private var waterTraceSource: String {
    giShaderSource(zeroCopy: true) + "\n#define LIT_MODE 1\n" + litShaderHeader + "\n" + waterCommonHeader + waterTraceKernel
}

/// A 4 x 4 visiting order that spreads consecutive frames' samples apart (a Bayer order, as RtShadows').
private let waterPattern: [SIMD2<UInt32>] = [0, 10, 2, 8, 5, 15, 7, 13, 1, 11, 3, 9, 4, 14, 6, 12].map { SIMD2(UInt32($0 % 4), UInt32($0 / 4)) }

/// Debug: water_trace's source (the GI cache's kernels, lit mode's G-buffer header, water's), for an offline compile check.
/// Returns its length.
@_cdecl("mmc_debug_water_trace_source")
public func mmc_debug_water_trace_source(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let bytes = Array(waterTraceSource.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}
