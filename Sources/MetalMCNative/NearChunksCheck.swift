import Foundation
import Metal
import MetalMCCore
import simd

// Offline checks for near chunks (NearChunks.swift), driven by tools/neartest.py through ctypes. Nothing here runs in
// the game.
//
// - mmc_near_debug_synth_chunk: vanilla-like BLOCK vertices for the cube faces of one chunk of a real save (face
//   culling, vanilla's corner order, ambient occlusion from the four samples per corner, directional shade, biome
//   tints, smooth sky light, per-block UV rotations and the grass side overlay), to measure the codec on real terrain.
// - mmc_near_debug_encode / _decode_cpu / _decode_gpu: the codec round trip, and the shader's decoder against the CPU's.
// - mmc_near_debug_arena_check: random allocations and frees against the slab allocator's invariants.
// - mmc_near_debug_render: one section drawn through the near-chunk pipeline and through a transcription of vanilla's
//   terrain.vsh that reads the original 28-byte vertices, into two images to compare.

// MARK: - Synthetic vanilla vertices from a save

private struct SynthWriter {
    let out: UnsafeMutableRawPointer
    let cap: Int
    var count = 0

    mutating func vertex(_ p: SIMD3<Float>, _ rgb: SIMD3<Int32>, _ uv: SIMD2<Float>, _ block: Int32, _ sky: Int32) -> Bool {
        guard count < cap else { return false }
        let o = out + count * 28
        o.storeBytes(of: p.x, toByteOffset: 0, as: Float.self)
        o.storeBytes(of: p.y, toByteOffset: 4, as: Float.self)
        o.storeBytes(of: p.z, toByteOffset: 8, as: Float.self)
        o.storeBytes(of: UInt8(rgb.x), toByteOffset: 12, as: UInt8.self)
        o.storeBytes(of: UInt8(rgb.y), toByteOffset: 13, as: UInt8.self)
        o.storeBytes(of: UInt8(rgb.z), toByteOffset: 14, as: UInt8.self)
        o.storeBytes(of: UInt8(255), toByteOffset: 15, as: UInt8.self)
        o.storeBytes(of: uv.x, toByteOffset: 16, as: Float.self)
        o.storeBytes(of: uv.y, toByteOffset: 20, as: Float.self)
        o.storeBytes(of: Int16(block), toByteOffset: 24, as: Int16.self)
        o.storeBytes(of: Int16(sky), toByteOffset: 26, as: Int16.self)
        count += 1
        return true
    }
}

// Vanilla's FaceInfo corner order per direction (down, up, north, south, west, east): each corner as (x, y, z) in {0, 1}.
private let synthFaces: [(dir: SIMD3<Int32>, shade: Float, corners: [SIMD3<Int32>])] = [
    (SIMD3(0, -1, 0), 0.5, [SIMD3(0, 0, 1), SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 0, 1)]),
    (SIMD3(0, 1, 0), 1.0, [SIMD3(0, 1, 0), SIMD3(0, 1, 1), SIMD3(1, 1, 1), SIMD3(1, 1, 0)]),
    (SIMD3(0, 0, -1), 0.8, [SIMD3(1, 1, 0), SIMD3(1, 0, 0), SIMD3(0, 0, 0), SIMD3(0, 1, 0)]),
    (SIMD3(0, 0, 1), 0.8, [SIMD3(0, 1, 1), SIMD3(0, 0, 1), SIMD3(1, 0, 1), SIMD3(1, 1, 1)]),
    (SIMD3(-1, 0, 0), 0.6, [SIMD3(0, 1, 0), SIMD3(0, 0, 0), SIMD3(0, 0, 1), SIMD3(0, 1, 1)]),
    (SIMD3(1, 0, 0), 0.6, [SIMD3(1, 1, 1), SIMD3(1, 0, 1), SIMD3(1, 0, 0), SIMD3(1, 1, 0)]),
]

/// Debug: vanilla-like BLOCK vertices for chunk `index` (0-1023) of a region file: every cube face of an opaque block
/// next to a non-occluding one, in sections `sectionLo...sectionHi` (0-23, from the world bottom). Solid-layer quads come
/// first, then cutout ones (leaves, the grass side overlay, and plants: grass, ferns and flowers as vanilla's cross
/// models with their random offsets, the generic records' main customer). Positions are section-relative, so several
/// sections overlap in space. The atlas is taken as 1024 x 1024 with 16 x 16 sprites. info: sections, solid quads,
/// cutout quads, tinted quads, plant quads. Returns the vertex count, or -1 if the chunk couldn't be read.
@_cdecl("mmc_near_debug_synth_chunk")
public func mmc_near_debug_synth_chunk(_ path: UnsafePointer<CChar>, _ index: Int32, _ sectionLo: Int32, _ sectionHi: Int32,
                                       _ out: UnsafeMutableRawPointer, _ capVertices: Int32, _ info: UnsafeMutablePointer<Int64>) -> Int32 {
    guard let data = FileManager.default.contents(atPath: String(cString: path)) else { return -1 }
    // Plants map to air in Materials; a marker id keeps them apart here.
    let plant: UInt8 = 250
    var cache: [String: UInt8] = [:]
    for n in ["short_grass", "tall_grass", "fern", "large_fern", "short_dry_grass", "tall_dry_grass", "dandelion", "poppy", "blue_orchid",
              "allium", "azure_bluet", "red_tulip", "orange_tulip", "white_tulip", "pink_tulip", "oxeye_daisy", "cornflower",
              "lily_of_the_valley", "torchflower", "bush", "firefly_bush", "dead_bush", "open_eyeblossom", "closed_eyeblossom"] {
        cache["minecraft:" + n] = plant
    }
    guard let chunk = try? ChunkScan.decodeChunk(region: [UInt8](data), index: Int(index), cache: &cache) else { return -1 }
    let height = Anvil.sectionsY * 16
    var ids = [UInt8](repeating: 0, count: 16 * 16 * height)   // y, z, x
    for (sy, blocks) in chunk.sections {
        for i in 0..<4096 { ids[sy * 4096 + i] = blocks[i] }
    }
    func raw(_ x: Int32, _ y: Int32, _ z: Int32) -> UInt8 {
        guard x >= 0, x < 16, z >= 0, z < 16, y >= 0, y < Int32(height) else { return 0 }
        return ids[Int(y) * 256 + Int(z) * 16 + Int(x)]
    }
    func id(_ x: Int32, _ y: Int32, _ z: Int32) -> Mat {
        let r = raw(x, y, z)
        return r == plant ? .air : (Mat(rawValue: r) ?? .unknown)
    }
    func leaves(_ m: Mat) -> Bool {
        m == .leaves || m == .cherryLeaves || m == .yellowPoplarLeaves || m == .redPoplarLeaves || m == .orangePoplarLeaves
    }
    func solid(_ m: Mat) -> Bool { m.kind == .opaque }
    // Fancy leaves: leaf faces are drawn next to leaves, and nothing is culled by a leaf block.
    func occludes(_ m: Mat) -> Bool { solid(m) && !leaves(m) }
    func shade(_ m: Mat) -> Float { solid(m) ? 0.2 : 1.0 }
    var top = [Int32](repeating: -1, count: 256)
    for z in 0..<16 {
        for x in 0..<16 {
            var y = Int32(height - 1)
            while y >= 0 && !solid(id(Int32(x), y, Int32(z))) { y -= 1 }
            top[z * 16 + x] = y
        }
    }
    // Sky light: 15 above the column's highest block, fading one level per block below it (enough for smooth blends).
    func sky(_ x: Int32, _ y: Int32, _ z: Int32) -> Int32 {
        if solid(id(x, y, z)) { return 0 }
        guard x >= 0, x < 16, z >= 0, z < 16 else { return 15 }
        let t = top[Int(z) * 16 + Int(x)]
        return y > t ? 15 : max(0, 15 - (t - y))
    }
    func sprite(_ m: Mat, _ face: Int) -> SIMD2<Float> {
        let s = Int(m.rawValue) * 3 + (face == 1 ? 0 : (face == 0 ? 1 : 2))
        return SIMD2(Float((s % 64) * 16), Float((s / 64) * 16))
    }
    let grassTint = SIMD3<Int32>(0x91, 0xBD, 0x59), leafTint = SIMD3<Int32>(0x77, 0xAB, 0x2F)
    let lo = max(0, Int(sectionLo)), hi = min(Anvil.sectionsY - 1, Int(sectionHi))
    var w = SynthWriter(out: out, cap: Int(capVertices))
    var solidQuads = 0, cutoutQuads = 0, tinted = 0, plantQuads = 0

    // cutout: 0 solid pass (solid faces), 1 cutout pass (leaves and grass overlays).
    for pass in 0..<2 {
        for sy in lo...max(lo, hi) where hi >= lo {
            for ly in 0..<16 {
                for z in Int32(0)..<16 {
                    for x in Int32(0)..<16 {
                        let y = Int32(sy * 16 + ly)
                        if pass == 1 && raw(x, y, z) == plant {
                            // A cross model: two diagonal planes, each drawn from both sides, offset like vanilla's
                            // BlockState.getOffset (XZ within a quarter block, Y down to -0.2), lit flat, grass-tinted.
                            var h = UInt32(bitPattern: (x &* 3129871) ^ (z &* 116129781) ^ y)
                            h = h &* h &* 42317861 &+ h &* 11
                            let dx = Float((Float((h >> 16) & 15) / 15 - 0.5) * 0.5), dz = Float((Float((h >> 24) & 15) / 15 - 0.5) * 0.5)
                            let dy = Float((Float((h >> 20) & 15) / 15 - 1) * 0.2)
                            let bx = Float(x) + dx, by = Float(ly) + dy, bz = Float(z) + dz
                            let a: Float = 0.1464466, b: Float = 0.8535534
                            let light = sky(x, y, z) << 4
                            let rgb = grassTint &* 229 / 255
                            let s0 = sprite(.grass, 3)
                            let uvs = [SIMD2<Float>(s0.x / 1024, s0.y / 1024), SIMD2<Float>(s0.x / 1024, (s0.y + 16) / 1024),
                                       SIMD2<Float>((s0.x + 16) / 1024, (s0.y + 16) / 1024), SIMD2<Float>((s0.x + 16) / 1024, s0.y / 1024)]
                            for (p0, p1) in [(SIMD2(a, a), SIMD2(b, b)), (SIMD2(a, b), SIMD2(b, a))] {
                                let pts = [SIMD3<Float>(p0.x + bx, 1 + by, p0.y + bz), SIMD3<Float>(p0.x + bx, by, p0.y + bz),
                                           SIMD3<Float>(p1.x + bx, by, p1.y + bz), SIMD3<Float>(p1.x + bx, 1 + by, p1.y + bz)]
                                for order in [[0, 1, 2, 3], [3, 2, 1, 0]] {
                                    for k in order {
                                        guard w.vertex(pts[k], rgb, uvs[k], 0, light) else { return Int32(w.count) }
                                    }
                                    cutoutQuads += 1
                                    plantQuads += 1
                                    tinted += 1
                                }
                            }
                            continue
                        }
                        let m = id(x, y, z)
                        guard solid(m) else { continue }
                        let isLeaves = leaves(m)
                        for (f, face) in synthFaces.enumerated() {
                            let n = SIMD3(x, y, z) &+ face.dir
                            let nm = id(n.x, n.y, n.z)
                            guard !occludes(nm) else { continue }
                            // Layers: leaves are cutout, grass sides get a cutout overlay over their solid dirt side.
                            let overlay = m == .grass && f >= 2
                            if pass == 0 && isLeaves { continue }
                            if pass == 1 && !isLeaves && !overlay { continue }
                            let tint: SIMD3<Int32>? = pass == 1 ? (m == .leaves ? leafTint : (overlay ? grassTint : nil))
                                : (m == .grass && f == 1 ? grassTint : nil)
                            // In-plane axes of this face.
                            let axes = face.dir.x != 0 ? (1, 2) : (face.dir.y != 0 ? (0, 2) : (0, 1))
                            var colors = [SIMD3<Int32>](), lights = [(Int32, Int32)]()
                            for c in face.corners {
                                var d1 = SIMD3<Int32>(repeating: 0), d2 = SIMD3<Int32>(repeating: 0)
                                d1[axes.0] = c[axes.0] == 0 ? -1 : 1
                                d2[axes.1] = c[axes.1] == 0 ? -1 : 1
                                let e1 = n &+ d1, e2 = n &+ d2, dg = n &+ d1 &+ d2
                                let s1 = shade(id(e1.x, e1.y, e1.z)), s2 = shade(id(e2.x, e2.y, e2.z))
                                let s3 = shade(id(dg.x, dg.y, dg.z)), s0 = shade(nm)
                                let level = (s1 + s2 + s3 + s0) * Float(0.25)
                                var g = Int32((level * Float(255)).rounded(.down))
                                g = min(255, Int32(Float(g) * face.shade))
                                let rgb = tint.map { $0 &* g / 255 } ?? SIMD3(repeating: g)
                                colors.append(rgb)
                                // smoothBlend of sky light (16 per level), with the centre standing in for dark neighbors.
                                let center = sky(n.x, n.y, n.z) << 4
                                var sum = center
                                for p in [e1, e2, dg] {
                                    let v = sky(p.x, p.y, p.z) << 4
                                    sum += (v == 0 && center > 32) ? center : v
                                }
                                lights.append((0, sum >> 2))
                            }
                            // UVs: vanilla's default face mapping, rotated per block for the top faces (random variants).
                            let s0 = sprite(overlay && pass == 1 ? .unknown : m, f)
                            var corners = [SIMD2<Float>(0, 0), SIMD2<Float>(0, 16), SIMD2<Float>(16, 16), SIMD2<Float>(16, 0)]
                            if f == 1 {
                                let r = Int((x &* 31 &+ z &* 17 &+ y &* 7) & 3)
                                corners = (0..<4).map { corners[($0 + r) & 3] }
                            }
                            for (k, c) in face.corners.enumerated() {
                                let pos = SIMD3<Float>(Float(x + c.x), Float(ly) + Float(c.y), Float(z + c.z))
                                let t = s0 + corners[k]
                                let u0 = s0.x / 1024, u1 = (s0.x + 16) / 1024, v0 = s0.y / 1024, v1 = (s0.y + 16) / 1024
                                let uv = SIMD2<Float>(u0 + (u1 - u0) * ((t.x - s0.x) / 16), v0 + (v1 - v0) * ((t.y - s0.y) / 16))
                                guard w.vertex(pos, colors[k], uv, lights[k].0, lights[k].1) else { return Int32(w.count) }
                            }
                            if pass == 0 { solidQuads += 1 } else { cutoutQuads += 1 }
                            if tint != nil { tinted += 1 }
                        }
                    }
                }
            }
        }
    }
    info[0] = Int64(chunk.sections.count)
    info[1] = Int64(solidQuads)
    info[2] = Int64(cutoutQuads)
    info[3] = Int64(tinted)
    info[4] = Int64(plantQuads)
    return Int32(w.count)
}

// MARK: - Codec round trip

/// Debug: encodes `vertexCount` BLOCK vertices into `out` (at most `capWords` words). info: quads, aligned, generic,
/// tinted, tint misses, encode nanoseconds. Returns the words written, -1 if the layer was rejected, -2 if out is short.
@_cdecl("mmc_near_debug_encode")
public func mmc_near_debug_encode(_ vertices: UnsafeRawPointer, _ vertexCount: Int32, _ out: UnsafeMutablePointer<UInt32>,
                                  _ capWords: Int32, _ info: UnsafeMutablePointer<Int64>) -> Int32 {
    var words: [UInt32] = []
    var st = NearCodecStats()
    let t0 = DispatchTime.now().uptimeNanoseconds
    let ok = NearCodec.encode(vertices, vertexCount: Int(vertexCount), into: &words, stats: &st)
    let dt = DispatchTime.now().uptimeNanoseconds - t0
    info[0] = Int64(st.quads); info[1] = Int64(st.aligned); info[2] = Int64(st.generic)
    info[3] = Int64(st.tinted); info[4] = Int64(st.tintMisses); info[5] = Int64(dt)
    guard ok else { return -1 }
    guard words.count <= Int(capWords) else { return -2 }
    for (i, v) in words.enumerated() { out[i] = v }
    return Int32(words.count)
}

/// Debug: decodes every vertex of `quadCount` records on the CPU. out: 10 floats per vertex (x y z u v r g b block sky).
@_cdecl("mmc_near_debug_decode_cpu")
public func mmc_near_debug_decode_cpu(_ words: UnsafePointer<UInt32>, _ quadCount: Int32, _ out: UnsafeMutablePointer<Float>) {
    for q in 0..<Int(quadCount) {
        for k in 0..<4 {
            let (p, uv, rgb, light) = NearCodec.decode(words, slot: q, k: k)
            let o = out + (q * 4 + k) * 10
            o[0] = p.x; o[1] = p.y; o[2] = p.z; o[3] = uv.x; o[4] = uv.y
            o[5] = Float(rgb.x); o[6] = Float(rgb.y); o[7] = Float(rgb.z); o[8] = Float(light.x); o[9] = Float(light.y)
        }
    }
}

private var nearDebugLibrary: MTLLibrary?

private func nearDebugLib() -> MTLLibrary? {
    if nearDebugLibrary == nil {
        let opts = MTLCompileOptions()
        opts.languageVersion = .version3_0
        opts.preserveInvariance = true
        do {
            nearDebugLibrary = try ctx.device.makeLibrary(source: nearShaderSource + nearReferenceShaderSource, options: opts)
        } catch {
            log("near chunks debug library: \(error)")
        }
    }
    return nearDebugLibrary
}

/// Debug: decodes every vertex of `quadCount` records with the shader's nearDecode (compute). Same output as
/// mmc_near_debug_decode_cpu. Returns 1 on success.
@_cdecl("mmc_near_debug_decode_gpu")
public func mmc_near_debug_decode_gpu(_ words: UnsafePointer<UInt32>, _ wordCount: Int32, _ quadCount: Int32,
                                      _ out: UnsafeMutablePointer<Float>) -> Int32 {
    guard let lib = nearDebugLib(), let f = lib.makeFunction(name: "near_decode_test"),
          let pipe = try? ctx.device.makeComputePipelineState(function: f),
          let src = ctx.device.makeBuffer(bytes: words, length: max(4, Int(wordCount) * 4), options: [.storageModeShared]),
          let dst = ctx.device.makeBuffer(length: max(4, Int(quadCount) * 4 * 10 * 4), options: [.storageModeShared]),
          let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
    var n = UInt32(quadCount)
    enc.setComputePipelineState(pipe)
    enc.setBuffer(src, offset: 0, index: 0)
    enc.setBuffer(dst, offset: 0, index: 1)
    enc.setBytes(&n, length: 4, index: 2)
    enc.dispatchThreads(MTLSize(width: max(1, Int(quadCount) * 4), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    guard cb.status == .completed else { return 0 }
    memcpy(out, dst.contents(), Int(quadCount) * 4 * 10 * 4)
    return 1
}

// MARK: - Arena

/// Debug: `iterations` random allocations and frees on one slab of `slots` slots, checking after each step that the
/// free list is sorted, merged and disjoint from every live allocation, and that free + used = slots. Returns 0 if all
/// held, else the failing step (1-based). out: allocations made, failed allocations (slab full), largest free range at
/// the end.
@_cdecl("mmc_near_debug_arena_check")
public func mmc_near_debug_arena_check(_ slots: Int32, _ iterations: Int32, _ seed: UInt64, _ out: UnsafeMutablePointer<Int64>) -> Int32 {
    guard let slab = NearArena.Slab(slots: Int(slots)) else { return -1 }
    var rng = seed | 1
    func next() -> UInt64 { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return rng }
    var live: [(Int, Int)] = []
    var made = 0, failed = 0
    for step in 1...Int(iterations) {
        if live.isEmpty || next() % 3 != 0 {
            let n = 1 + Int(next() % 700) + (next() % 50 == 0 ? Int(next() % 20000) : 0)
            if let s = slab.allocate(n) { live.append((s, n)); made += 1 } else { failed += 1 }
        } else {
            let i = Int(next() % UInt64(live.count))
            let (s, n) = live.remove(at: i)
            slab.release(s, n)
        }
        var freeTotal = 0
        for (i, r) in slab.free.enumerated() {
            if r.count <= 0 { return Int32(step) }
            if i > 0 && slab.free[i - 1].start + slab.free[i - 1].count >= r.start { return Int32(step) }   // unsorted or unmerged
            freeTotal += r.count
        }
        let usedTotal = live.reduce(0) { $0 + $1.1 }
        if freeTotal + usedTotal != Int(slots) || slab.used != usedTotal { return Int32(step) }
        if step % 97 == 0 || step == Int(iterations) {
            // Every live allocation must be outside every free range, and live ones must not overlap each other.
            let sorted = live.sorted { $0.0 < $1.0 }
            for i in 1..<max(1, sorted.count) where sorted[i - 1].0 + sorted[i - 1].1 > sorted[i].0 { return Int32(step) }
            for (s, n) in live {
                for r in slab.free where s < r.start + r.count && r.start < s + n { return Int32(step) }
            }
        }
    }
    out[0] = Int64(made)
    out[1] = Int64(failed)
    out[2] = Int64(slab.free.map { $0.count }.max() ?? 0)
    return 0
}

// MARK: - Render comparison

// vanilla's core/terrain.vsh with the multidraw define, transcribed: the 28-byte BLOCK vertex through the vertex fetch
// (RGBA8 unorm color, 16-bit integer light), the section from the instanced stream. Appended to nearShaderSource.
let nearReferenceShaderSource = """

struct NearRefIn {
    float3 position [[attribute(0)]];
    float4 color [[attribute(1)]];
    float2 uv0 [[attribute(2)]];
    short2 uv2 [[attribute(3)]];
};

vertex NearVertex near_ref_vs(NearRefIn in [[stage_in]], uint section [[base_instance]],
                              device const NearSection* sections [[buffer(17)]],
                              constant float4x4& projMat [[buffer(18)]],
                              constant NearTerrain& terrain [[buffer(19)]],
                              constant NearGlobals& globals [[buffer(20)]],
                              texture2d<float> lightmap [[texture(16)]], sampler lightmapSampler [[sampler(12)]]) {
    NearSection sec = sections[section];
    float3 pos = in.position + float3(int3(sec.position) - int3(globals.cameraBlockPos)) + float3(globals.cameraOffset);
    NearVertex o;
    o.position = (projMat * terrain.modelView) * float4(pos, 1.0);
    o.position.y = -o.position.y;
    o.sphericalVertexDistance = length(pos);
    o.cylindricalVertexDistance = max(length(pos.xz), abs(pos.y));
    float2 lightUv = clamp((float2(int2(in.uv2)) / 256.0) + 0.5 / 16.0, float2(0.5 / 16.0), float2(15.5 / 16.0));
    o.vertexColor = in.color * lightmap.sample(lightmapSampler, lightUv, level(0.0));
    o.texCoord0 = in.uv0;
    float dist = length(pos);
    o.chunkVisibility = mix(1.0, sec.visibility, clamp((dist - 16.0) / 16.0, 0.0, 1.0));
    return o;
}
"""


/// The test scene both render checks draw: atlas, lightmap, samplers, Minecraft's uniform blocks, render targets.
private final class NearTestScene {
    let w: Int, h: Int
    let atlas: MTLTexture, lightmap: MTLTexture
    let atlasSampler: MTLSamplerState, lightSampler: MTLSamplerState
    var globals = [UInt8](repeating: 0, count: 48)
    var proj = simd_float4x4()
    var terrain = [UInt8](repeating: 0, count: 80)
    var fog: [Float]
    var section = [UInt8](repeating: 0, count: 16)
    let colorRef: MTLTexture, colorNear: MTLTexture, depth: MTLTexture

    init?(width: Int, height: Int, origin: UnsafePointer<Int32>, camera: UnsafePointer<Double>, yaw: Float, pitch: Float, p: UnsafePointer<Float>) {
        let dev = ctx.device
        w = width
        h = height
        // Atlas: 1024 x 1024, 5 levels, per-texel noise; about one texel in eight is transparent (cutout).
        let ad = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 1024, height: 1024, mipmapped: true)
        ad.mipmapLevelCount = 5
        ad.usage = [.shaderRead]
        guard let atlas = dev.makeTexture(descriptor: ad) else { return nil }
        var seed: UInt32 = 12345
        for level in 0..<5 {
            let s = 1024 >> level
            var texels = [UInt32](repeating: 0, count: s * s)
            for i in 0..<(s * s) {
                seed = seed &* 1664525 &+ 1013904223
                let a: UInt32 = (seed >> 29) == 0 ? 0 : 255
                texels[i] = (seed >> 8) & 0xFFFFFF | a << 24
            }
            atlas.replace(region: MTLRegionMake2D(0, 0, s, s), mipmapLevel: level, withBytes: texels, bytesPerRow: s * 4)
        }
        self.atlas = atlas
        let ld = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: 16, height: 16, mipmapped: false)
        guard let lightmap = dev.makeTexture(descriptor: ld) else { return nil }
        var lm = [UInt32](repeating: 0, count: 256)
        for sky in 0..<16 {
            for block in 0..<16 {
                let r = UInt32(40 + 13 * max(sky, block)), g = UInt32(40 + 12 * sky + 2 * block), b = UInt32(60 + 12 * sky)
                lm[sky * 16 + block] = min(255, r) | min(255, g) << 8 | min(255, b) << 16 | 255 << 24
            }
        }
        lightmap.replace(region: MTLRegionMake2D(0, 0, 16, 16), mipmapLevel: 0, withBytes: lm, bytesPerRow: 64)
        self.lightmap = lightmap
        let sd = MTLSamplerDescriptor()
        sd.minFilter = .linear; sd.magFilter = .linear; sd.mipFilter = .linear
        sd.sAddressMode = .clampToEdge; sd.tAddressMode = .clampToEdge
        guard let atlasSampler = dev.makeSamplerState(descriptor: sd) else { return nil }
        sd.mipFilter = .notMipmapped
        guard let lightSampler = dev.makeSamplerState(descriptor: sd) else { return nil }
        self.atlasSampler = atlasSampler
        self.lightSampler = lightSampler

        // Uniforms laid out as Minecraft's std140 blocks.
        let cam = SIMD3(camera[0], camera[1], camera[2])
        let camBlock = SIMD3<Int32>(Int32(floor(cam.x)), Int32(floor(cam.y)), Int32(floor(cam.z)))
        let camOffset = SIMD3<Float>(Float(Double(camBlock.x) - cam.x), Float(Double(camBlock.y) - cam.y), Float(Double(camBlock.z) - cam.z))
        let rgss = p[3] > 0.5
        globals.withUnsafeMutableBytes { g in
            g.storeBytes(of: camBlock.x, toByteOffset: 0, as: Int32.self)
            g.storeBytes(of: camBlock.y, toByteOffset: 4, as: Int32.self)
            g.storeBytes(of: camBlock.z, toByteOffset: 8, as: Int32.self)
            g.storeBytes(of: camOffset.x, toByteOffset: 16, as: Float.self)
            g.storeBytes(of: camOffset.y, toByteOffset: 20, as: Float.self)
            g.storeBytes(of: camOffset.z, toByteOffset: 24, as: Float.self)
            g.storeBytes(of: Int32(rgss ? 1 : 0), toByteOffset: 44, as: Int32.self)
        }
        // Reverse-Z infinite perspective (depth 1 at the near plane, 0 at infinity), 70 degrees vertical.
        let near: Float = 0.05, fov: Float = 70 * .pi / 180
        let f = 1 / tan(fov / 2), aspect = Float(w) / Float(h)
        proj = simd_float4x4(SIMD4(f / aspect, 0, 0, 0), SIMD4(0, f, 0, 0), SIMD4(0, 0, 0, -1), SIMD4(0, 0, near, 0))
        // View rotation like Minecraft's: pitch about x, then yaw + 180 about y.
        let yr = (yaw + 180) * .pi / 180, pr = pitch * .pi / 180
        let rx = simd_float4x4(SIMD4(1, 0, 0, 0), SIMD4(0, cos(pr), sin(pr), 0), SIMD4(0, -sin(pr), cos(pr), 0), SIMD4(0, 0, 0, 1))
        let ry = simd_float4x4(SIMD4(cos(yr), 0, sin(yr), 0), SIMD4(0, 1, 0, 0), SIMD4(-sin(yr), 0, cos(yr), 0), SIMD4(0, 0, 0, 1))
        terrain.withUnsafeMutableBytes { t in
            var mv = rx * ry
            withUnsafeBytes(of: &mv) { t.baseAddress!.copyMemory(from: $0.baseAddress!, byteCount: 64) }
            t.storeBytes(of: Int32(1024), toByteOffset: 64, as: Int32.self)
            t.storeBytes(of: Int32(1024), toByteOffset: 68, as: Int32.self)
        }
        fog = [0.62, 0.74, 0.95, 1.0, p[0], p[1], 1000, 1200, 1200, 1200]
        section.withUnsafeMutableBytes { s in
            s.storeBytes(of: origin[0], toByteOffset: 0, as: Int32.self)
            s.storeBytes(of: origin[1], toByteOffset: 4, as: Int32.self)
            s.storeBytes(of: origin[2], toByteOffset: 8, as: Int32.self)
            s.storeBytes(of: p[2], toByteOffset: 12, as: Float.self)
        }

        let cd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        cd.usage = [.renderTarget, .shaderRead]
        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: w, height: h, mipmapped: false)
        dd.usage = [.renderTarget]
        dd.storageMode = .private
        guard let colorRef = dev.makeTexture(descriptor: cd), let colorNear = dev.makeTexture(descriptor: cd),
              let depth = dev.makeTexture(descriptor: dd) else { return nil }
        self.colorRef = colorRef
        self.colorNear = colorNear
        self.depth = depth
    }

    func pipeline(_ lib: MTLLibrary, reference: Bool, cutout: Bool) -> MTLRenderPipelineState? {
        let constants = MTLFunctionConstantValues()
        var cut = cutout
        constants.setConstantValue(&cut, type: .bool, index: 0)
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = lib.makeFunction(name: reference ? "near_ref_vs" : "near_vs")
        d.fragmentFunction = try? lib.makeFunction(name: "near_fs", constantValues: constants)
        d.colorAttachments[0].pixelFormat = .rgba8Unorm
        d.depthAttachmentPixelFormat = .depth32Float
        if reference {
            let vd = MTLVertexDescriptor()
            vd.layouts[30].stride = 28
            vd.attributes[0].format = .float3; vd.attributes[0].offset = 0; vd.attributes[0].bufferIndex = 30
            vd.attributes[1].format = .uchar4Normalized; vd.attributes[1].offset = 12; vd.attributes[1].bufferIndex = 30
            vd.attributes[2].format = .float2; vd.attributes[2].offset = 16; vd.attributes[2].bufferIndex = 30
            vd.attributes[3].format = .short2; vd.attributes[3].offset = 24; vd.attributes[3].bufferIndex = 30
            d.vertexDescriptor = vd
        }
        return try? ctx.device.makeRenderPipelineState(descriptor: d)
    }

    /// Draws `quads` quads from `buffer` (BLOCK vertices for the reference, records for near_vs) into one target.
    func draw(_ cb: MTLCommandBuffer, _ pipe: MTLRenderPipelineState, reference: Bool, buffer: MTLBuffer, quads: Int, into target: MTLTexture) -> Bool {
        var idx = [UInt32](repeating: 0, count: max(1, quads * 6))
        for q in 0..<quads {
            let v = UInt32(4 * q)
            idx[6 * q] = v; idx[6 * q + 1] = v + 1; idx[6 * q + 2] = v + 2; idx[6 * q + 3] = v + 2; idx[6 * q + 4] = v + 3; idx[6 * q + 5] = v
        }
        guard let ib = ctx.device.makeBuffer(bytes: idx, length: idx.count * 4, options: [.storageModeShared]) else { return false }
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = target
        rp.colorAttachments[0].loadAction = .clear
        rp.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
        rp.colorAttachments[0].storeAction = .store
        rp.depthAttachment.texture = depth
        rp.depthAttachment.loadAction = .clear
        rp.depthAttachment.clearDepth = 0
        rp.depthAttachment.storeAction = .dontCare
        guard let enc = cb.makeRenderCommandEncoder(descriptor: rp) else { return false }
        enc.setRenderPipelineState(pipe)
        enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: true))
        enc.setCullMode(.back)
        enc.setFrontFacing(.clockwise)
        enc.setVertexBuffer(buffer, offset: 0, index: reference ? 30 : 16)
        enc.setVertexBytes(&section, length: 16, index: 17)
        enc.setVertexBytes(&proj, length: 64, index: 18)
        enc.setVertexBytes(&terrain, length: 80, index: 19)
        enc.setFragmentBytes(&terrain, length: 80, index: 19)
        enc.setVertexBytes(&globals, length: 48, index: 20)
        enc.setFragmentBytes(&globals, length: 48, index: 20)
        enc.setFragmentBytes(&fog, length: 40, index: 21)
        enc.setVertexTexture(lightmap, index: 16)
        enc.setVertexSamplerState(lightSampler, index: 12)
        enc.setFragmentTexture(atlas, index: 17)
        enc.setFragmentSamplerState(atlasSampler, index: 13)
        enc.drawIndexedPrimitives(type: .triangle, indexCount: quads * 6, indexType: .uint32, indexBuffer: ib, indexBufferOffset: 0,
                                  instanceCount: 1, baseVertex: 0, baseInstance: 0)
        enc.endEncoding()
        return true
    }

    /// Reads both images back and compares them. info: pixels that differ, largest channel difference, pixels covered.
    func compare(_ outRef: UnsafeMutablePointer<UInt8>, _ outNear: UnsafeMutablePointer<UInt8>, _ info: UnsafeMutablePointer<Int64>) {
        colorRef.getBytes(outRef, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        colorNear.getBytes(outNear, bytesPerRow: w * 4, from: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0)
        var differ = 0, worst = 0, covered = 0
        for i in 0..<(w * h) {
            var d = 0
            for c in 0..<4 { d = max(d, abs(Int(outRef[4 * i + c]) - Int(outNear[4 * i + c]))) }
            if d > 0 { differ += 1 }
            worst = max(worst, d)
            if (0..<4).contains(where: { outRef[4 * i + $0] != 0 || outNear[4 * i + $0] != 0 }) { covered += 1 }
        }
        info[0] = Int64(differ)
        info[1] = Int64(worst)
        info[2] = Int64(covered)
    }
}

/// Debug: draws `vertexCount` BLOCK vertices (one section at `origin`) twice into width x height RGBA8 images, through
/// a transcription of vanilla's terrain vertex shader reading the vertices as they are (outRef) and through the
/// near-chunk pipeline reading their repacked records (outNear); both use near_fs. The camera sits at `camera` (world
/// position) looking at yaw/pitch degrees; the atlas is 1024 x 1024 with 5 mip levels of noise (some texels transparent),
/// the lightmap a gradient. p: fog start, fog end, section visibility, use RGSS (0/1). Returns 1, or -1 if the layer
/// couldn't be repacked, 0 on a Metal failure. info: pixels that differ, largest channel difference, pixels covered.
@_cdecl("mmc_near_debug_render")
public func mmc_near_debug_render(_ vertices: UnsafeRawPointer, _ vertexCount: Int32, _ cutout: Int32,
                                  _ origin: UnsafePointer<Int32>, _ camera: UnsafePointer<Double>, _ yaw: Float, _ pitch: Float,
                                  _ p: UnsafePointer<Float>, _ width: Int32, _ height: Int32,
                                  _ outRef: UnsafeMutablePointer<UInt8>, _ outNear: UnsafeMutablePointer<UInt8>,
                                  _ info: UnsafeMutablePointer<Int64>) -> Int32 {
    var words: [UInt32] = []
    var st = NearCodecStats()
    guard NearCodec.encode(vertices, vertexCount: Int(vertexCount), into: &words, stats: &st) else { return -1 }
    guard let lib = nearDebugLib(),
          let scene = NearTestScene(width: Int(width), height: Int(height), origin: origin, camera: camera, yaw: yaw, pitch: pitch, p: p),
          let refPipe = scene.pipeline(lib, reference: true, cutout: cutout != 0),
          let nearPipe = scene.pipeline(lib, reference: false, cutout: cutout != 0),
          let vbuf = ctx.device.makeBuffer(bytes: vertices, length: Int(vertexCount) * 28, options: [.storageModeShared]),
          let qbuf = ctx.device.makeBuffer(bytes: words, length: words.count * 4, options: [.storageModeShared]),
          let cb = ctx.queue.makeCommandBuffer() else { return 0 }
    let quads = Int(vertexCount) / 4
    guard scene.draw(cb, refPipe, reference: true, buffer: vbuf, quads: quads, into: scene.colorRef),
          scene.draw(cb, nearPipe, reference: false, buffer: qbuf, quads: quads, into: scene.colorNear) else { return 0 }
    cb.commit()
    cb.waitUntilCompleted()
    guard cb.status == .completed else { return 0 }
    scene.compare(outRef, outNear, info)
    return 1
}

/// Debug: like mmc_near_debug_render, but the near image goes through the path the game uses: the layer is added to the
/// real arena (after a 7-quad filler entry, so it doesn't start at slot 0), drawn by mmc_near_draw inside a render pass
/// opened with mmc_pass_begin, as two records (its first `split` quads, then the rest) with section index 1 of a
/// two-section stream, with Minecraft's uniforms at different offsets of one buffer; then submitted with mmc_submit.
/// Also checks the arena's deferred reuse: the entry is freed right after the draw; an allocation of its size before
/// the submit completes must land elsewhere, and one after it may reuse it. info: pixels that differ, largest channel
/// difference, pixels covered, records drawn, 1 if the early allocation avoided the freed range, 1 if the late one
/// reused it. Returns 1, -1 if the layer couldn't be repacked, -2 if the shaders didn't compile, 0 on a Metal failure.
@_cdecl("mmc_near_debug_render_arena")
public func mmc_near_debug_render_arena(_ vertices: UnsafeRawPointer, _ vertexCount: Int32, _ cutout: Int32, _ split: Int32,
                                        _ origin: UnsafePointer<Int32>, _ camera: UnsafePointer<Double>, _ yaw: Float, _ pitch: Float,
                                        _ p: UnsafePointer<Float>, _ width: Int32, _ height: Int32,
                                        _ outRef: UnsafeMutablePointer<UInt8>, _ outNear: UnsafeMutablePointer<UInt8>,
                                        _ info: UnsafeMutablePointer<Int64>) -> Int32 {
    let r = NearRenderer.shared
    let t0 = Date()
    while !r.ready() {
        if r.failed || Date().timeIntervalSince(t0) > 30 { return -2 }
        usleep(10_000)
    }
    guard let lib = nearDebugLib(),
          let scene = NearTestScene(width: Int(width), height: Int(height), origin: origin, camera: camera, yaw: yaw, pitch: pitch, p: p),
          let refPipe = scene.pipeline(lib, reference: true, cutout: cutout != 0),
          let vbuf = ctx.device.makeBuffer(bytes: vertices, length: Int(vertexCount) * 28, options: [.storageModeShared]),
          let cb = ctx.queue.makeCommandBuffer() else { return 0 }
    let quads = Int(vertexCount) / 4
    guard scene.draw(cb, refPipe, reference: true, buffer: vbuf, quads: quads, into: scene.colorRef) else { return 0 }
    cb.commit()
    cb.waitUntilCompleted()

    // The arena: a filler entry first, then the layer.
    let arena = NearArena.shared
    var filler = [UInt8](repeating: 0, count: 7 * 112)
    filler.withUnsafeMutableBytes { b in
        for q in 0..<7 {
            for k in 0..<4 {
                let o = (q * 4 + k) * 28
                let corner = [(0, 0), (0, 1), (1, 1), (1, 0)][k]
                b.storeBytes(of: Float(q + corner.0), toByteOffset: o, as: Float.self)
                b.storeBytes(of: Float(1), toByteOffset: o + 4, as: Float.self)
                b.storeBytes(of: Float(corner.1), toByteOffset: o + 8, as: Float.self)
                b.storeBytes(of: UInt32.max, toByteOffset: o + 12, as: UInt32.self)
            }
        }
    }
    let fillerId = filler.withUnsafeBytes { arena.add($0.baseAddress!, vertexCount: 28) }
    let id = arena.add(vertices, vertexCount: Int(vertexCount))
    guard fillerId != 0, id != 0 else { return -1 }
    arena.lock.lock()
    let entry = arena.entry(id)!
    arena.lock.unlock()

    // Minecraft's uniforms at 256-byte steps of one buffer, and a two-section stream whose second entry is this one.
    guard let ubo = ctx.device.makeBuffer(length: 2048, options: [.storageModeShared]),
          let sections = ctx.device.makeBuffer(length: 64, options: [.storageModeShared]) else { return 0 }
    let base = ubo.contents()
    withUnsafeBytes(of: &scene.proj) { base.advanced(by: 256).copyMemory(from: $0.baseAddress!, byteCount: 64) }
    scene.terrain.withUnsafeBytes { base.advanced(by: 512).copyMemory(from: $0.baseAddress!, byteCount: 80) }
    scene.globals.withUnsafeBytes { base.advanced(by: 768).copyMemory(from: $0.baseAddress!, byteCount: 48) }
    scene.fog.withUnsafeBytes { base.advanced(by: 1024).copyMemory(from: $0.baseAddress!, byteCount: 40) }
    scene.section.withUnsafeBytes { sections.contents().advanced(by: 32).copyMemory(from: $0.baseAddress!, byteCount: 16) }
    let sectionsH = makeHandle(BufferBox(sections)), uboH = makeHandle(BufferBox(ubo))
    let atlasH = makeHandle(TextureBox(scene.atlas)), atlasS = makeHandle(SamplerBox(scene.atlasSampler))
    let lightH = makeHandle(TextureBox(scene.lightmap)), lightS = makeHandle(SamplerBox(scene.lightSampler))
    let colorH = makeHandle(TextureBox(scene.colorNear)), depthH = makeHandle(TextureBox(scene.depth))
    defer { for h in [sectionsH, uboH, atlasH, atlasS, lightH, lightS, colorH, depthH] { mmc_handle_release(h) } }
    // The stream's slice starts 16 bytes in, so section index 1 is the one at byte 32.
    var res: [Int64] = [sectionsH, 16, uboH, 256, uboH, 512, uboH, 768, uboH, 1024, atlasH, atlasS, lightH, lightS]
    let first = max(0, min(Int(split), quads))
    var records: [Int32] = [Int32(bitPattern: id), 0, Int32(first), 1, Int32(bitPattern: id), Int32(first), Int32(quads - first), 1]
    if first == 0 { records = Array(records[4...]) }

    ctx.cond.lock()
    let submit = ctx.completed + 1
    ctx.cond.unlock()
    var colors: [Int64] = [colorH]
    var clear: [Float] = [0, 0, 0, 0]
    guard mmc_pass_begin(&colors, 1, 1, &clear, depthH, 1, 0, 0, 0, Int32(scene.w), Int32(scene.h)) == 1 else { return 0 }
    let drawn = mmc_near_draw(Int32(cutout != 0 ? 1 : 0), &records, Int32(records.count / 4), submit, &res, 0)
    mmc_pass_end()
    // Freed while the submit that draws it is still being recorded: its slots must not come back yet.
    arena.free(id)
    arena.lock.lock()
    let early = arena.allocate(Int(entry.slots))
    arena.lock.unlock()
    let earlyAvoided = early.map { $0.0 != Int(entry.slab) || $0.1 + Int(entry.slots) <= Int(entry.start) || $0.1 >= Int(entry.start + entry.slots) } ?? true
    mmc_submit(submit)
    _ = mmc_wait_submit(submit, -1)
    arena.lock.lock()
    if let e = early { arena.slabs[e.0].release(e.1, Int(entry.slots)) }
    let late = arena.allocate(Int(entry.slots))
    if let l = late { arena.slabs[l.0].release(l.1, Int(entry.slots)) }
    arena.lock.unlock()
    arena.free(fillerId)
    let lateReused = late.map { $0.0 == Int(entry.slab) && $0.1 == Int(entry.start) } ?? false
    scene.compare(outRef, outNear, info)
    info[3] = Int64(drawn)
    info[4] = earlyAvoided ? 1 : 0
    info[5] = lateReused ? 1 : 0
    return 1
}
