import Foundation
import Metal
import simd

// Entities, items and particles in lit mode's light (METALMC_EXP=lit,entities), and rain that looks like rain
// (METALMC_EXP=lit,rain). docs/lighting-design.md, "Entities, particles and rain in our light".
//
// Lit mode relights the terrain. Everything else kept vanilla's lightmap light, which knows nothing of our sun and its
// traced shadows, our sky, the colored block light or the GI cache: mobs too flat by day and gray under torches, rain a
// saturated blue. This file:
//
// 1. Entities (entities): vanilla's entity pipelines that write depth (solid and cutout entities and block entities,
//    armor, items: entityPatchKind) and its opaque particles (opaque_particle: flames, smoke, dust, splashes) get a
//    variant for the level's main pass whose shaders are patched here, as Water.swift patches the translucent terrain's
//    (PipelineBox.installWater, the same hook): the vertex shader passes on its color, its light coordinates and (an
//    entity's) normal, and the fragment shader writes an entity texel into the G-buffer: the albedo (the texture times the
//    vertex color and the color modulator, with the hurt overlay mixed in), the normal, the light levels and a depth key.
//    The relight (litEntityPixel, spliced into litRelightPixel) lights it like the terrain around it: the sun with the
//    traced visibility at the pixel (rt_shadow traces every pixel with depth), the sky by the normal (the GI cache's light
//    where it has some), the moon, block light (the colored light volume at the point), vanilla's fog. Layers blended over
//    it afterwards (an enderman's eyes, a slime's outer cube, glint, rain) are found by the overlay test, as on terrain.
//    Particles have no normal: they're lit from all sides like plants, and full-bright ones (flames) glow like the light
//    sources they come from.
//    Translucent entities and items (players' skins, translucent items, a slime's outer cube) write a texel where the
//    fragment is mostly opaque (alpha 0.5 or more) and read the old one back elsewhere (programmable blending).
// 2. Translucent particles (entities): blended over what's behind them, they can't take a texel of their own. Their
//    shaders take our light at their light levels instead of vanilla's lightmap (mmcForwardLight: the sun's and the sky's
//    light the relight used the frame before, kept by EntityLight.keep and bound in the level's main pass), and leave the
//    G-buffer as it was (they read it back: programmable blending).
// 3. Rain (rain): the weather pipeline's streaks thin (the middle of each texel column of vanilla's rain texture),
//    gray-white and fainter, lit like a translucent particle. Snow keeps its flakes and takes the same light. Streaks
//    over the sky (a pixel without a texel) go into the G-buffer instead of the color and are put on at the end of the
//    frame's light (litRainOverSky): with clouds, after their composite, which would cover them.
// 4. The hand (hand, Java side): the first-person hand goes with the level's features (ExtHand), so the patched entity
//    and item pipelines put it in the G-buffer too.
//
// Without the switches nothing here is compiled in or patched, and every shader is the same text as before.

/// METALMC_EXP=entities (with lit): entities, items and particles lit like the terrain.
let entityLight = litEnabled && experiments.contains("entities")
/// METALMC_EXP=rain (with lit): rain as thin gray-white streaks in our light.
let rainLook = litEnabled && experiments.contains("rain")

/// The rain's look (rain): its streaks' share of a texel column's width (METALMC_RAINWIDTH, 0.05-1), their alpha against
/// vanilla's (METALMC_RAINALPHA, 0-1) and their reflectance (METALMC_RAINALBEDO: linear, of the light around them).
let rainWidth = min(max(Float(ProcessInfo.processInfo.environment["METALMC_RAINWIDTH"] ?? "") ?? 0.35, 0.05), 1)
let rainAlpha = min(max(Float(ProcessInfo.processInfo.environment["METALMC_RAINALPHA"] ?? "") ?? 0.7, 0), 1)
let rainAlbedo = min(max(Float(ProcessInfo.processInfo.environment["METALMC_RAINALBEDO"] ?? "") ?? 0.6, 0), 4)
/// The share of the sun's light (lit from all sides, as a particle) rain takes (METALMC_RAINSUN, 0-1): rain falls from
/// cloud, lit by the sky; in full rain our sky leaves the sun a fifth of its light, which turned the streaks yellow.
let rainSun = min(max(Float(ProcessInfo.processInfo.environment["METALMC_RAINSUN"] ?? "") ?? 0.15, 0), 1)

/// Buffer indices of the forward light (EntityLight.bind) in both stages: free in vanilla's shaders (they use 0-6) and in
/// ours that draw in the level's main pass (16 and up), and below the vertex buffers (30 down).
let entityEnvIndex = 12
let entityFrameIndex = 13

/// What a pipeline's patched variant does.
enum EntityPatchKind {
    /// An entity, block entity or item that writes depth: an entity texel with its normal.
    case entity
    /// A translucent entity or item (players' skins, translucent items, a slime's outer cube): an entity texel where the
    /// fragment is mostly opaque (alpha 0.5 or more), else the texel that was there.
    case translucent
    /// An opaque particle: an entity texel without a normal (lit from all sides).
    case particle
    /// A translucent particle: our light in its own shader, the G-buffer left as it was.
    case forward
    /// Rain and snow: the same, and rain's streaks thin and gray-white.
    case weather
}

/// The pipelines patched (vanilla's names, 26.3), with the switches on.
func entityPatchKind(_ name: String) -> EntityPatchKind? {
    let prefix = "minecraft:pipeline/"
    guard name.hasPrefix(prefix) else { return nil }
    switch name.dropFirst(prefix.count) {
    case "entity_solid", "entity_solid_offset_forward", "entity_cutout", "entity_cutout_cull", "entity_cutout_z_offset",
         "entity_cutout_dissolve", "armor_cutout_no_cull", "armor_decal_cutout_no_cull", "item_cutout":
        return entityLight ? .entity : nil
    case "entity_translucent", "entity_translucent_cull", "item_translucent", "armor_translucent":
        return entityLight ? .translucent : nil
    case "opaque_particle": return entityLight ? .particle : nil
    case "translucent_particle": return entityLight ? .forward : nil
    case "weather": return rainLook ? .weather : nil
    default: return nil
    }
}

// MARK: - The entity texel

/// The G-buffer's entity texel, as the patched fragment shaders write it and litEntityPixel reads it. Terrain's texels
/// have a face code of 1-7 (x bits 29-31), water's code 0 with bit 28; an entity's has code 0, bit 28 clear and bit 27
/// set, so every reader that doesn't know it sees "not lit terrain", as before:
///   x: albedo r | g << 8 | b << 16 (sRGB-encoded 8 bits; blue rounded to even, its low bit 1 for a particle) | the normal's
///      low 3 bits << 24 | 1 << 27
///   y: depth key (bits 5-19 of the depth: 15 bits) | sky light << 15 | block light << 19 (whole levels) | the normal's high
///      9 bits << 23
/// The normal: octahedral (about the y axis), 6 bits a coordinate (0-62, 31 the middle, so the axes are exact): about a
/// degree off at most.
private let entityPackHelper = """

// MetalMC entities (Entities.swift): vanilla's entity, item and particle texels into lit mode's G-buffer (color 1).
static inline uint mmcEntityOct(float3 n) {
    n /= max(abs(n.x) + abs(n.y) + abs(n.z), 1e-6);
    float2 e = n.y >= 0.0 ? n.xz : (1.0 - abs(n.zx)) * select(float2(-1.0), float2(1.0), n.xz >= 0.0);
    uint2 u = uint2(round(clamp(e * 0.5 + 0.5, 0.0, 1.0) * 62.0));
    return u.x | (u.y << 6);
}
static inline uint2 mmcEntityGbuffer(float3 albedo, float3 n, float2 light, float z, bool particle) {
    uint3 a = uint3(round(saturate(albedo) * 255.0));
    uint b = min((a.b + 1u) & ~1u, 254u) | (particle ? 1u : 0u);
    uint o = particle ? 0u : mmcEntityOct(n);
    uint sky = uint(clamp(round(light.y), 0.0, 15.0)), block = uint(clamp(round(light.x), 0.0, 15.0));
    return uint2(a.r | (a.g << 8) | (b << 16) | ((o & 7u) << 24) | (1u << 27),
                 ((as_type<uint>(z) >> 5) & 0x7FFFu) | (sky << 15) | (block << 19) | ((o >> 3) << 23));
}

"""

/// Our light at a vertex's light levels, for what is blended over the level (translucent particles, rain): the sun's and
/// the sky's light the relight used the frame before (lit_env's 8 float4, EntityLight.keep), lit from all sides like a
/// plant, the moon as the relight's, and vanilla's block light from its lightmap. Linear light per unit of albedo, the
/// relight's units.
private let entityForwardHelper = """

// MetalMC (Entities.swift): our light at a vertex's light levels. env: the sun's and the sky's light the relight used last
// frame (lit_env's layout); lf[0]: x the exposure, y 1 if env holds it (else vanilla's light stays); lf[1]: toward the
// sun; lf[2]: toward the moon, w how much it's night; lf[3]: the rain's look (x alpha, y width, z reflectance, w the
// share of the sun's light it takes).
static inline float3 mmcDecode(float3 c) {
    float3 a = abs(c);
    return sign(c) * select(pow((a + 0.055) / 1.055, 2.4), a / 12.92, a <= 0.04045);
}
static inline float3 mmcEncode(float3 c) {
    float3 a = abs(c);
    return sign(c) * select(1.055 * pow(a, 1.0 / 2.4) - 0.055, 12.92 * a, a <= 0.0031308);
}
static inline float3 mmcLightmap(texture2d<float> lm, sampler s, float block, float sky) {
    return lm.sample(s, (float2(block, sky) + 0.5) / 16.0, level(0.0)).rgb;
}
static inline float mmcBrightness(float level) {
    float x = saturate(level / 15.0);
    return x / (4.0 - 3.0 * x);
}
static float3 mmcForwardLight(constant float4* env, constant float4* lf, texture2d<float> lm, sampler s, float2 light, float sunShare) {
    float block = clamp(light.x, 0.0, 15.0), sky = clamp(light.y, 0.0, 15.0);
    // Full bright (flames, emissive particles), like vanilla.
    if (block >= 14.97) return float3(1.0);
    float3 lm00 = mmcDecode(mmcLightmap(lm, s, 0.0, 0.0));
    float3 blockLight = mmcDecode(mmcLightmap(lm, s, block, 0.0));
    float3 vanillaSky = max(mmcDecode(mmcLightmap(lm, s, 0.0, 15.0)) - lm00, 0.0);
    float skyFall = mmcDecode(float3(mmcBrightness(sky))).x;
    float open = saturate((sky - 12.0) / 3.0);
    float3 sun = env[0].rgb * ((0.25 + 0.5 * saturate(lf[1].y)) * open * sunShare);
    float moonLum = dot(vanillaSky, float3(0.2126, 0.7152, 0.0722)) * lf[2].w;
    float3 moon = moonLum * float3(0.857, 1.006, 1.371) * (0.6 * (0.25 + 0.5 * saturate(lf[2].y)) / max(lf[2].y, 0.5) * open + 0.3 * skyFall);
    return (sun + env[7].rgb * skyFall + moon + blockLight) * lf[0].x;
}

"""

/// Water drops among the particles (rain's splashes, drips: blue in vanilla's particle atlas) near white, as water looks
/// (it shows the sky's light: a gray albedo made the splashes dark specks on the sand); `blue`: how much a texel is one
/// (blue well over red), not for full-bright particles (soul fire's flames).
private let entityDropHelper = """
static inline float mmcDropBlue(float3 c) { return saturate((c.b - c.r - 0.2) * 5.0); }
static inline float3 mmcDropGray(float3 c) { return float3(0.84, 0.88, 0.94) * saturate(0.6 + dot(c, float3(0.2126, 0.7152, 0.0722))); }

"""

// MARK: - Patching SPIRV-Cross's MSL

/// The fragment entry point's parameter list (between `main0(` and the body's brace).
private func entityFragmentSignature(_ src: String) -> Range<String.Index>? {
    guard let start = src.range(of: "fragment main0_out main0(") else { return nil }
    guard let brace = src.range(of: ")\n{", range: start.upperBound..<src.endIndex) else { return nil }
    return start.upperBound..<brace.lowerBound
}

/// The vertex entry point's parameter list.
private func entityVertexSignature(_ src: String) -> Range<String.Index>? {
    guard let start = src.range(of: "vertex main0_out main0(") else { return nil }
    guard let brace = src.range(of: ")\n{", range: start.upperBound..<src.endIndex) else { return nil }
    return start.upperBound..<brace.lowerBound
}

/// The name SPIRV-Cross gave the DynamicTransforms block (its ColorModulator) in a parameter list.
private func entityTransformsName(_ sig: Substring) -> String? {
    guard let r = sig.range(of: "constant DynamicTransforms& ") else { return nil }
    let name = sig[r.upperBound...].prefix { $0.isLetter || $0.isNumber || $0 == "_" }
    return name.isEmpty ? nil : String(name)
}

/// Inserts `helper` after `using namespace metal;`.
private func entityInsertHelper(_ s: inout String, _ helper: String) -> Bool {
    guard let using = s.range(of: "using namespace metal;\n") else { return false }
    s.insert(contentsOf: helper, at: using.upperBound)
    return true
}

/// The vertex shader with the outputs its fragment shader needs: the raw vertex color and the light coordinates (in
/// levels) for a texel, and an entity's normal; or our light at the vertex (forward, weather). nil if the source isn't
/// shaped as expected (the pipeline then keeps its unpatched variant).
func entityPatchVertex(_ src: String, _ kind: EntityPatchKind) -> String? {
    guard let outR = waterStructRange(src, "main0_out"), let inR = waterStructRange(src, "main0_in") else { return nil }
    let inBody = src[inR]
    guard waterHasMember(inBody, "UV2"), waterHasMember(inBody, "Color") else { return nil }
    var s = src
    switch kind {
    case .entity, .translucent, .particle:
        let normal = kind != .particle
        if normal && !waterHasMember(inBody, "Normal") { return nil }
        s.insert(contentsOf: "    float4 mmcColor [[user(mmccolor)]];\n    float2 mmcLight [[user(mmclight)]];\n"
                 + (normal ? "    float3 mmcNormal [[user(mmcnormal)]];\n" : ""), at: outR.upperBound)
        return waterBeforeLastReturn(s, "    out.mmcColor = in.Color;\n    out.mmcLight = float2(in.UV2) / 16.0;\n"
                                     + (normal ? "    out.mmcNormal = in.Normal;\n" : ""))
    case .forward, .weather:
        s.insert(contentsOf: "    float4 mmcColor [[user(mmccolor)]];\n    float3 mmcE [[user(mmce)]];\n", at: outR.upperBound)
        guard let sig = entityVertexSignature(s), s[sig].contains("texture2d<float> Sampler2 "), s[sig].contains("sampler Sampler2Smplr ") else { return nil }
        s.insert(contentsOf: ", constant float4* mmcEnv [[buffer(\(entityEnvIndex))]], constant float4* mmcLf [[buffer(\(entityFrameIndex))]]", at: sig.upperBound)
        guard entityInsertHelper(&s, entityForwardHelper) else { return nil }
        // Rain falls from cloud: the sky's light lights it, the sun's (a fifth of it is left in full rain) only a little.
        return waterBeforeLastReturn(s, "    out.mmcColor = in.Color;\n    out.mmcE = mmcForwardLight(mmcEnv, mmcLf, Sampler2, Sampler2Smplr, float2(in.UV2) / 16.0, \(kind == .weather ? "mmcLf[3].w" : "1.0"));\n")
    }
}

/// The fragment shader with the G-buffer as a second output: an entity texel (entity, particle), or the texel that was
/// there (forward, weather: read back, so what's behind stays what the relight sees), whose color then takes our light.
func entityPatchFragment(_ src: String, _ kind: EntityPatchKind) -> String? {
    guard let outR0 = waterStructRange(src, "main0_out") else { return nil }
    let outBody = src[outR0]
    guard let colorName = waterMember(outBody, "[[color(0)]]"), !outBody.contains("[[color(1)]]") else { return nil }
    guard let sig0 = entityFragmentSignature(src) else { return nil }
    let sig = src[sig0]
    guard sig.contains("texture2d<float> Sampler0 "), sig.contains("sampler Sampler0Smplr "), let dt = entityTransformsName(sig) else { return nil }
    let front = sig.contains("bool gl_FrontFacing [[front_facing]]")
    var s = src
    s.insert(contentsOf: "    uint2 mmcGbuf [[color(1)]];\n", at: outR0.upperBound)
    guard let inR = waterStructRange(s, "main0_in"), waterHasMember(s[inR], "texCoord0") else { return nil }
    let overlay = waterHasMember(s[inR], "overlayColor")
    switch kind {
    case .entity, .translucent, .particle:
        let normal = kind != .particle
        var coord = waterMember(s[inR], "[[position]]")
        var add = "    float4 mmcColor [[user(mmccolor)]];\n    float2 mmcLight [[user(mmclight)]];\n" + (normal ? "    float3 mmcNormal [[user(mmcnormal)]];\n" : "")
        if coord == nil {
            coord = "mmcCoord"
            add += "    float4 mmcCoord [[position]];\n"
        }
        s.insert(contentsOf: add, at: inR.upperBound)
        if kind == .translucent {
            // The texel that was there, for the fragments that are mostly see-through (programmable blending).
            guard let sig1 = entityFragmentSignature(s) else { return nil }
            s.insert(contentsOf: ", uint2 mmcPrev [[color(1)]]", at: sig1.upperBound)
        }
        guard entityInsertHelper(&s, entityPackHelper + (kind == .particle ? entityDropHelper : "")) else { return nil }
        // The normal as vanilla lights it: a back face (two-sided parts) takes the light from behind.
        let n = normal ? "(dot(in.mmcNormal, in.mmcNormal) > 1e-8 ? normalize(in.mmcNormal) : float3(0.0, 1.0, 0.0))\(front ? " * (gl_FrontFacing ? 1.0 : -1.0)" : "")" : "float3(0.0)"
        let texel = "mmcEntityGbuffer(mmcA, \(n), in.mmcLight, in.\(coord!).z, \(kind == .particle ? "true" : "false"))"
        // Water drops near white (particles): the albedo, and the forward color with the same light (vanilla's, the color
        // drawn over the albedo), so the overlay test finds the particle plain.
        let drops = kind != .particle ? "" : """

                float mmcB = mmcDropBlue(mmcA) * step(in.mmcLight.x, 14.5);
                if (mmcB > 0.0) {
                    float3 mmcG = mmcDropGray(mmcA);
                    float mmcV = dot(out.\(colorName).rgb, float3(0.2126, 0.7152, 0.0722)) / max(dot(mmcA, float3(0.2126, 0.7152, 0.0722)), 1e-3);
                    out.\(colorName).rgb = mix(out.\(colorName).rgb, mmcG * mmcV, mmcB);
                    mmcA = mix(mmcA, mmcG, mmcB);
                }
        """
        return waterBeforeLastReturn(s, """
            {
                float3 mmcA = Sampler0.sample(Sampler0Smplr, in.texCoord0).rgb * in.mmcColor.rgb * \(dt).ColorModulator.rgb;\(overlay ? "\n        mmcA = mix(in.overlayColor.rgb, mmcA, in.overlayColor.a);" : "")\(drops)
                out.mmcGbuf = \(kind == .translucent ? "out.\(colorName).a >= 0.5 ? \(texel) : mmcPrev" : texel);
            }

        """)
    case .forward, .weather:
        s.insert(contentsOf: "    float4 mmcColor [[user(mmccolor)]];\n    float3 mmcE [[user(mmce)]];\n", at: inR.upperBound)
        guard let sig1 = entityFragmentSignature(s) else { return nil }
        s.insert(contentsOf: ", constant float4* mmcLf [[buffer(\(entityFrameIndex))]], uint2 mmcPrev [[color(1)]]", at: sig1.upperBound)
        guard entityInsertHelper(&s, entityForwardHelper + (kind == .forward ? entityDropHelper : "")) else { return nil }
        let body: String
        if kind == .forward {
            body = """
                    float4 mmcT = Sampler0.sample(Sampler0Smplr, in.texCoord0) * in.mmcColor * \(dt).ColorModulator;
                    mmcT.rgb = mix(mmcT.rgb, mmcDropGray(mmcT.rgb), mmcDropBlue(mmcT.rgb));
                    out.\(colorName).rgb = mmcEncode(mmcDecode(mmcT.rgb) * in.mmcE);
            """
        } else {
            // Rain's texels are blue, snow's white. A rain streak keeps the middle of its texel column (lf[3].y of its width;
            // where a texel is narrower than a pixel, that share of it), gray-white and fainter (lf[3].x), its light
            // reflected (lf[3].z); snow keeps its texture's gray levels. Not over water's texels: the relight takes the
            // water layer off what's drawn to see the floor, which turned the streaks there olive. Over a pixel without a
            // texel (the sky) a streak goes into the G-buffer instead of the color, over-composited with any streak there
            // already, and the frame's light puts it on at the end (litRainOverSky): with the clouds, after their
            // composite, which would cover it.
            body = """
                    float4 mmcT = Sampler0.sample(Sampler0Smplr, in.texCoord0);
                    float mmcRain = saturate((mmcT.b - mmcT.r - 0.15) * 4.0);
                    if (mmcRain > 0.0 && (mmcPrev.x >> 28) == 1u) mmcT.a = 0.0;
                    float mmcU = in.texCoord0.x * float(Sampler0.get_width());
                    float mmcF = fract(mmcU) - 0.5, mmcW = max(fwidth(mmcU), 1e-4), mmcH = 0.5 * mmcLf[3].y;
                    float mmcCover = max(min(mmcF + 0.5 * mmcW, mmcH) - max(mmcF - 0.5 * mmcW, -mmcH), 0.0) / min(mmcW, 1.0);
                    float3 mmcLin = mix(mmcDecode(float3(dot(mmcT.rgb, float3(0.2126, 0.7152, 0.0722)))), mmcLf[3].z * float3(0.95, 0.98, 1.0), mmcRain) * in.mmcE;
                    out.\(colorName) = float4(mmcEncode(mmcLin), mmcT.a * in.mmcColor.a * \(dt).ColorModulator.a * mix(1.0, saturate(mmcCover) * mmcLf[3].x, mmcRain));
                    if (mmcRain > 0.5 && (mmcPrev.x >> 24) == 0u) {
                        float mmcA = saturate(out.\(colorName).a), mmcPa = float(mmcPrev.y & 255u) / 255.0;
                        float3 mmcPr = float3(float(mmcPrev.x & 255u), float((mmcPrev.x >> 8) & 255u), float((mmcPrev.x >> 16) & 255u)) / 255.0;
                        float mmcNa = mmcA + mmcPa * (1.0 - mmcA);
                        float3 mmcNr = (saturate(out.\(colorName).rgb) * mmcA + mmcPr * (mmcPa * (1.0 - mmcA))) / max(mmcNa, 1e-4);
                        uint3 mmcQ = uint3(round(saturate(mmcNr) * 255.0));
                        out.mmcGbuf = uint2(mmcQ.r | (mmcQ.g << 8) | (mmcQ.b << 16), uint(round(mmcNa * 255.0)));
                        out.\(colorName).a = 0.0;
                    }
            """
        }
        return waterBeforeLastReturn(s, """
            out.mmcGbuf = mmcPrev;
            if (mmcLf[0].y > 0.5) {
        \(body)
            }

        """)
    }
}

/// Keeps the patched pipelines' sources, compiles their variants off the render thread and hands each its functions
/// (PipelineBox.installWater: the level's main pass's variant takes them, with the G-buffer written).
final class EntityGbuffer: @unchecked Sendable {
    static let shared = EntityGbuffer()
    private let lock = NSLock()
    private let queue = DispatchQueue(label: "metalmc.entities.pipelines", qos: .userInitiated)
    private(set) var compiled = 0, failed = 0

    /// mmc_pipeline_create, for a pipeline entityPatchKind names (with lit mode and the switch).
    func register(_ box: PipelineBox, kind: EntityPatchKind, vs: String, vsEntry: String, fs: String, fsEntry: String) {
        queue.async { self.compile(box, kind, vs, vsEntry, fs, fsEntry) }
    }

    private func compile(_ box: PipelineBox, _ kind: EntityPatchKind, _ vs: String, _ vsEntry: String, _ fs: String, _ fsEntry: String) {
        guard let pvs = entityPatchVertex(vs, kind), let pfs = entityPatchFragment(fs, kind) else {
            lock.lock(); failed += 1; lock.unlock()
            log("entities: \(box.name)'s shaders aren't shaped as expected; it keeps vanilla's light")
            return
        }
        let opts = MTLCompileOptions()
        opts.languageVersion = .version3_0
        opts.preserveInvariance = true
        do {
            let vlib = try ctx.device.makeLibrary(source: pvs, options: opts)
            let flib = try ctx.device.makeLibrary(source: pfs, options: opts)
            guard let vf = vlib.makeFunction(name: vsEntry), let ff = flib.makeFunction(name: fsEntry) else {
                throw NSError(domain: "metalmc", code: 3, userInfo: [NSLocalizedDescriptionKey: "entry points not found"])
            }
            box.installWater(vs: vf, fs: ff)
            lock.lock(); compiled += 1; let n = compiled; lock.unlock()
            log("entities: \(box.name) \(kind == .forward || kind == .weather ? "takes our light" : "writes its texels into the G-buffer") (\(n) patched)")
        } catch {
            lock.lock(); failed += 1; lock.unlock()
            log("entities: \(box.name)'s patched shaders failed: \(error)")
        }
    }
}

// MARK: - The forward light

/// The sun's and the sky's light the relight used, kept for the next frame's forward-lit pipelines (translucent particles,
/// rain), and bound in the level's main pass.
final class EntityLight: @unchecked Sendable {
    static let shared = EntityLight()
    /// lit_env's 8 float4, copied at each relight (with the atmosphere); zeros until the first.
    private var env: MTLBuffer?
    /// mmcLf: [0] x the exposure, y 1 once env holds the light; [1] toward the sun; [2] toward the moon, w night; [3] rain.
    private var frame = [SIMD4<Float>](repeating: .zero, count: 4)

    /// Lit.relight, once this frame's sun and sky light is in `source` (nil without the atmosphere: the forward-lit
    /// pipelines keep vanilla's light then).
    func keep(cb: MTLCommandBuffer, source: MTLBuffer?, sunDir: SIMD4<Float>, moonDir: SIMD4<Float>, exposure: Float) {
        frame[3] = SIMD4(rainAlpha, rainWidth, rainAlbedo, rainSun)
        guard let source else { frame[0] = SIMD4(exposure, 0, 0, 0); return }
        if env == nil {
            env = ctx.device.makeBuffer(length: 8 * 16, options: .storageModePrivate)
            env?.label = "MetalMC entities' light"
        }
        guard let env, source.length >= 8 * 16, let blit = cb.makeBlitCommandEncoder() else { return }
        blit.label = "MetalMC entities' light"
        blit.copy(from: source, sourceOffset: 0, to: env, destinationOffset: 0, size: 8 * 16)
        blit.endEncoding()
        frame[0] = SIMD4(exposure, 1, 0, 0)
        frame[1] = sunDir
        frame[2] = moonDir
    }

    /// mmc_pass_begin, for the level's main pass (the one with the G-buffer).
    func bind(_ enc: MTLRenderCommandEncoder) {
        if env == nil {
            env = ctx.device.makeBuffer(length: 8 * 16, options: .storageModePrivate)
            env?.label = "MetalMC entities' light"
            frame[0].y = 0
        }
        guard let env else { return }
        enc.setVertexBuffer(env, offset: 0, index: entityEnvIndex)
        enc.setFragmentBuffer(env, offset: 0, index: entityEnvIndex)
        enc.setVertexBytes(frame, length: frame.count * 16, index: entityFrameIndex)
        enc.setFragmentBytes(frame, length: frame.count * 16, index: entityFrameIndex)
    }
}

// MARK: - The relight

/// The relight of an entity texel (spliced into litRelightHeader before litRelightPixel with entities or rain on; empty
/// without, so the relight's text is unchanged then). Called where litRelightPixel finds face code 0.
let entityRelightHeader: String = !(entityLight || rainLook) ? "" : """

// Entities (METALMC_EXP=entities, Entities.swift): an entity's, item's or particle's texel (x bits 27-31: 00001; see
// there), relit like the terrain: the sun with the traced visibility at the pixel, the sky by the normal (the GI cache's
// light where it has some), the moon, block light (the colored light volume at the point). Vanilla's light for the
// overlay test is its lightmap at the texel's levels times its entity face light (light.glsl: two fixed lights, 0.4 to 1;
// none for particles). Particles are lit from all sides, like plants.
\(postEnabled ? "#ifndef LIT_EMIT\n#define LIT_EMIT 6.0\n#endif\n" : "")constant float3 kLitEntityL0 = float3(0.161690, 0.808452, -0.565916);   // normalize(0.2, 1, -0.7): Lighting's level lights
constant float3 kLitEntityL1 = float3(-0.161690, 0.808452, 0.565916);
static bool litIsEntity(uint2 g) { return (g.x >> 27) == 1u; }
// Rain over a pixel without a texel (the sky; METALMC_EXP=rain): the streaks the weather pass left there instead of in the
// color (x: their color, sRGB-encoded 8 bits a channel, bits 24-31 clear; y: their alpha, 1-255), composited at the end
// of the frame's light: here without the clouds, after the clouds' composite with them (it would cover them: rain falls
// below the clouds).
static bool litIsRainSky(uint2 g) { return (g.x >> 24) == 0u && g.y != 0u; }
static float3 litRainOverSky(float3 c, uint2 g) {
    if (!litIsRainSky(g)) return c;
    float3 r = float3(float(g.x & 255u), float((g.x >> 8) & 255u), float((g.x >> 16) & 255u)) / 255.0;
    return mix(c, r, float(g.y & 255u) / 255.0);
}
static float3 litEntityNormal(uint2 g) {
    uint o = ((g.x >> 24) & 7u) | ((g.y >> 23) << 3);
    float2 e = float2(float(o & 63u), float((o >> 6) & 63u)) / 62.0 * 2.0 - 1.0;
    float3 n = float3(e.x, 1.0 - abs(e.x) - abs(e.y), e.y);
    if (n.y < 0.0) n.xz = (1.0 - abs(n.zx)) * select(float2(-1.0), float2(1.0), n.xz >= 0.0);
    return normalize(n);
}
static float3 litEntityPixel(float3 dst, uint2 q, float d, uint2 g, texture2d<half, access::read> vis, texture2d<float> lightmap,
                             constant LitFrame& f, constant float4* env\(litGi ? ", texture2d<float> giStandIn, texture2d<uint> gi" : "")\(clEnabled ? ", texture3d<float> clRGB, texture3d<float> clAux, constant ClFrame& clf, texture2d<half> clSh" : "")) {
    uint view = uint(f.misc.z);
    if (!litIsEntity(g)) return view != 0u ? float3(0.0) : \(cloudsEnabled ? "dst" : "litRainOverSky(dst, g)");
    // Something nearer was drawn over it since: its color stays.
    uint dk = (((as_type<uint>(d) >> 5) & 0x7FFFu) - (g.y & 0x7FFFu)) & 0x7FFFu;
    if (dk > 1u && dk != 0x7FFFu) return view != 0u ? float3(0.2, 0.0, 0.2) : dst;\(litGi ? "\n    uint2 giOwn = giUpsampleTexel(gi, q);" : "")
    bool particle = (g.x & (1u << 16)) != 0u;
    float3 albedo = float3(float(g.x & 255u), float((g.x >> 8) & 255u), float((g.x >> 16) & 254u)) / 255.0;
    float sky = float((g.y >> 15) & 15u), block = float((g.y >> 19) & 15u);
    float3 n = particle ? float3(0.0, 1.0, 0.0) : litEntityNormal(g);

    float2 ndc = (float2(q) + 0.5) / f.size.xy * 2.0 - 1.0;
    float4 h = f.invViewProj * float4(ndc, d, 1.0);
    float3 rel = h.xyz / h.w;
    float fog = 0.0;
    if (f.fogColor.a > 0.0) {
        fog = max(litFogValue(length(rel), f.fog.x, f.fog.y), litFogValue(max(length(rel.xz), abs(rel.y)), f.fog.z, f.fog.w)) * f.fogColor.a;
    }

    // Vanilla's light (what the forward color has) and the overlay test, as on terrain. The normal is a degree off at most,
    // so the test is a little looser.
    bool lm = f.misc.x > 0.5;
    float3 vlight = lm ? litLightmap(lightmap, block, sky) : float3(max(litBrightness(block), litBrightness(sky)));
    float la = particle ? 1.0 : min(1.0, (max(dot(kLitEntityL0, n), 0.0) + max(dot(kLitEntityL1, n), 0.0)) * 0.6 + 0.4);
    float3 V = vlight * la;
    float3 c0 = fog > 0.0 ? (dst - fog * f.fogColor.rgb) / max(1.0 - fog, 1e-3) : dst;
    float3 fwd = albedo * V;
    bool plain = fog > 0.9 || all(abs(c0 - fwd) <= fwd * 0.08 + 3.0 / 255.0);
    bool overlayBright = !plain && litLuma(c0) > litLuma(fwd);
    float3 refl = plain || overlayBright ? albedo : saturate(c0 / max(V, float3(1.0 / 255.0)));

    float3 E;
    float vSunOut = 0.0;\(litGi ? "\n    bool giUsed = false;" : "")\(clEnabled ? "\n    float4 clDbg = float4(0.0);\n    float3 clBL = float3(0.0);" : "")
    if (block >= 15.0) {
        // Full bright (burning mobs, blazes, flames), like vanilla; with the post chain, bright particles glow like the
        // light sources they come from (litEmitterTerm's rule at level 15).
        E = float3(1.0);\(postEnabled ? "\n        if (particle) E *= 1.0 + (LIT_EMIT - 1.0) * smoothstep(0.35, 0.8, litLuma(albedo));" : "")
    } else {
        float3 lm00 = lm ? skyDecode(litLightmap(lightmap, 0.0, 0.0)) : float3(0.0);
        float3 blockLight = lm ? skyDecode(litLightmap(lightmap, block, 0.0)) : float3(litBrightness(block) * litBrightness(block));\(clEnabled ? "\n        blockLight = clBlockLight(blockLight, lm00, block, rel, 6u, n, clf, clRGB, clAux, clSh, q, clDbg, view);\n        clBL = blockLight;" : "")
        float3 vanillaSky = lm ? max(skyDecode(litLightmap(lightmap, 0.0, 15.0)) - lm00, 0.0) : float3(1.0);
        float dayScale = f.misc.y > 0.5 ? litLuma(vanillaSky) : 1.0;
        float skyFall = skyDecode(float3(litBrightness(sky))).x;
        float open = saturate((sky - 12.0) / 3.0);
        float traced = open;
        if (f.size.z > 0.0) {
            uint s = uint(f.size.z);
            traced = float(vis.read(min(q / s, uint2(vis.get_width() - 1, vis.get_height() - 1))).r) * saturate(sky / 2.0);
        }
        bool moonTraced = f.sunDir.w > 0.5;
        float vSun = moonTraced ? open : traced, vMoon = moonTraced ? traced : open;
        vSunOut = vSun;
        float ndlSun = particle ? 0.25 + 0.5 * saturate(f.sunDir.y) : max(dot(n, f.sunDir.xyz), 0.0);
        float ndlMoon = particle ? 0.25 + 0.5 * saturate(f.moonDir.y) : max(dot(n, f.moonDir.xyz), 0.0);
        float3 sun = env[0].rgb * ndlSun * vSun;
        // The sky on the normal: the axis faces' light, weighted by the normal's squared components.
        float3 n2 = n * n;
        float3 skyAmb = (particle ? env[7].rgb : n2.x * env[n.x >= 0.0 ? 1 : 2].rgb + n2.y * env[n.y >= 0.0 ? 3 : 4].rgb + n2.z * env[n.z >= 0.0 ? 5 : 6].rgb) * skyFall;\(litGi ? """

        // The GI cache (Gi.swift) where it has data for the pixel's dominant face: the sky as the openings let it in,
        // plus bounced light (particles: a top's).
        if (gi.get_width() > 1u) {
            float3 an = abs(n);
            uint face = particle ? 2u : (an.x > an.y && an.x > an.z ? (n.x >= 0.0 ? 0u : 1u) : (an.y > an.z ? (n.y >= 0.0 ? 2u : 3u) : (n.z >= 0.0 ? 4u : 5u)));
            float4 cl = giUpsample(gi, giOwn, q, face, rel);
            if (cl.w > 0.0) { skyAmb = cl.rgb; giUsed = true; }
        }
""" : "")
        float hemi = particle ? 0.75 : mix(0.6, n.y >= 0.0 ? 1.0 : 0.2, abs(n.y));
        float moonLum = litLuma(vanillaSky) * f.moonDir.w;
        float3 moon = moonLum * kLitMoonTint * (0.6 * ndlMoon / max(f.moonDir.y, 0.5) * vMoon + 0.4 * hemi * skyFall);
        E = ((sun + skyAmb) * dayScale + moon + blockLight) * f.misc.w;
    }
    float3 lin = skyDecode(refl) * E;
    float3 o = skyEncode(lin);
    if (overlayBright) o = c0 + (o - fwd);
    if (f.size.w <= 1.0) o = saturate(o);
    o = max(o, 0.0);
    if (fog > 0.0) o = mix(o, f.fogColor.rgb, fog);
    if (view != 0u) {
        // The relight's debug views, for entities: 1 albedo, 2 the normal, 3 light levels, 4 white (no AO), 5 the new light
        // alone, 6 sun visibility, 7 the overlay test; 8 the GI cache, 11-12 colored light, as on terrain.
        float3 c = float3(0.0);
        if (view == 1u) c = albedo;
        else if (view == 2u) c = particle ? float3(1.0, 0.0, 1.0) : n * 0.5 + 0.5;
        else if (view == 3u) c = float3(block / 15.0, sky / 15.0, 0.0);
        else if (view == 4u) c = float3(1.0);
        else if (view == 5u) c = skyEncode(skyDecode(float3(0.5)) * E);
        else if (view == 6u) c = float3(vSunOut);
        else if (view == 7u) c = plain ? float3(0.0, 0.8, 0.0) : (overlayBright ? float3(0.9, 0.0, 0.9) : float3(0.9, 0.0, 0.0));\(litGi ? "\n        else if (view == 8u) c = giUsed ? float3(0.0, 0.8, 0.0) : (block >= 15.0 ? float3(0.0, 0.0, 0.9) : float3(0.9, 0.0, 0.0));" : "")\(clEnabled ? "\n        else if (view == 11u) c = skyEncode(skyDecode(float3(0.5)) * clBL * f.misc.w);\n        else if (view == 12u) c = clDbg.rgb;" : "")
        return saturate(c);
    }
    return o;
}

"""

/// litRelightPixel's face code 0 line's result with entities on (the arguments as litRelightPixel has them).
let entityRelightCall = "litEntityPixel(dst, q, d, g, vis, lightmap, f, env\(litGi ? ", giStandIn, gi" : "")\(clEnabled ? ", clRGB, clAux, clf, clSh" : ""))"

// MARK: - Debug

/// Debug: the patched vertex (which 0) or fragment (1) shader for `src` as pipeline `kind` (0 entity, 1 particle,
/// 2 forward, 3 weather, 4 translucent) into `out`; returns its length (0 if the source isn't shaped as expected, the needed length if
/// `len` is too small). For an offline compile check of the patches against dumped MSL.
@_cdecl("mmc_debug_entity_patch")
public func mmc_debug_entity_patch(_ kind: Int32, _ which: Int32, _ src: UnsafePointer<CChar>, _ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let k: EntityPatchKind = [.entity, .particle, .forward, .weather, .translucent][Int(min(max(kind, 0), 4))]
    let s = String(cString: src)
    guard let p = which == 0 ? entityPatchVertex(s, k) : entityPatchFragment(s, k) else { return 0 }
    let bytes = Array(p.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count + 1) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}
