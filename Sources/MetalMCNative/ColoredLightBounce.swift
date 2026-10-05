import Foundation
import Metal
import simd

// Colored block light bounced through the GI cache (with coloredlight and gi; gain METALMC_CLBOUNCE, default 0 = off
// until a debug view shows the channel reaching the image; 1 to try it).
// The cache's cells hold the sky's and the sun's light per unit of the frame's daylight, which block light doesn't
// follow, so the block light has a channel of its own per cell (clBounce: half4 a slot, rgb in absolute light and the
// cell's fingerprint in a, so a slot taken over by another cell reads as empty), filled by the same update: where a
// bounce ray hits a surface, the light volume's colored light there (sampled half a block in front of the hit, as the
// relight samples it) plus the hit cell's own bounced block light, times the hit's color, averaged like the cell's sky
// light. The request resolves it beside the sky light into a half-resolution RG11B10Float texture (in the slot the
// relight's giStandIn had kept free), and the relight adds it to the block light where vanilla's level isn't 0.

/// The bounce's gain (METALMC_CLBOUNCE, default 0: off; 1 to try it).
let clBounceGain: Float = max(0, Float(ProcessInfo.processInfo.environment["METALMC_CLBOUNCE"] ?? "") ?? 0)

/// The cache kernels' part (spliced into Gi.swift's giKernelSource with clEnabled only).
let giClBounceHeader = """

// Colored block light bounced through the cache (ColoredLightBounce.swift): the light volume's colored light at a hit.
struct GiCl {
    float4 volOff;   // xyz: from a hit (relative to the cache's origin) to the volume's (toroidal) texture space; w: 1 if valid
    float4 volRel;   // xyz: from a hit to the volume's cells (relative to its min corner)
    float4 size;     // xyz: the volume's size; w: the bounce's gain
    float4 fire;     // rgb: the fire bucket's color
};
static float3 giClBlock(constant GiCl& c, texture3d<float> rgbVol, texture3d<float> auxVol, float3 hp, uint hf) {
    if (c.volOff.w <= 0.0) return float3(0.0);
    float3 at = hp + kGiNormal[hf] * 0.5;
    float3 v = at + c.volRel.xyz;
    if (any(v < float3(2.0)) || any(v > c.size.xyz - 2.0)) return float3(0.0);
    constexpr sampler s(filter::linear, address::repeat);
    float3 uvw = (at + c.volOff.xyz) / c.size.xyz;
    float4 a = rgbVol.sample(s, uvw, level(0.0));
    float x = auxVol.sample(s, uvw, level(0.0)).x;
    if (a.w < 0.02) return float3(0.0);
    return (a.rgb + c.fire.rgb * x) / a.w * c.size.w;
}
static bool giClOwn(half4 b, uint2 key) { return uint(b.a) == (giPrint(key) & 1023u); }

"""

/// The resolve's part: the block channel at a half-resolution sample, as giInterp takes the sky light (the 2 x 2 cells
/// nearest on the face's plane, bilinear, those without samples left out), at the sample's level.
let giClResolveSource = """

// Colored block light bounced through the cache (ColoredLightBounce.swift): its channel at a half-resolution sample.
static float3 giInterpBlock(constant GiParams& p, const device uint* check, const device half4* value, const device half4* bv,
                            float3 f, int3 air, uint face, uint level) {
    uint2 t = giTangents(face);
    int s = 1 << level;
    int3 c = giCellOf(air, level);
    float3 local = float3(p.camBlock.xyz - giCorner(c, level)) + f;
    float2 q = float2(local[t.x], local[t.y]) / float(s) - 0.5;
    float2 fl = floor(q), w = q - fl;
    int2 o = int2(fl);
    float3 sum = 0.0;
    float ws = 0.0;
    for (int k = 0; k < 4; k++) {
        int3 cc = c;
        cc[t.x] += o.x + (k & 1);
        cc[t.y] += o.y + (k >> 1);
        uint2 key = giKey(cc, face, level);
        int e = giFind(check, giBucketOf(p, key), giPrint(key));
        float wt = ((k & 1) == 0 ? 1.0 - w.x : w.x) * ((k >> 1) == 0 ? 1.0 - w.y : w.y);
        if (wt <= 0.0 || e < 0 || !giSampled(value[e])) continue;
        half4 b = bv[e];
        sum += wt * (giClOwn(b, key) ? float3(b.rgb) : float3(0.0));
        ws += wt;
    }
    return ws > 0.0 ? sum / ws : float3(0.0);
}
static float3 giResolveBlock(constant GiParams& p, const device uint* check, const device half4* value, const device half4* bv,
                             depth2d<float, access::read> depth, uint2 hs) {
    uint2 full = uint2(p.sizes.xy), base = hs * 2u;
    uint best = 0u;
    float bd = -1.0;
    for (uint i = 0; i < 4u; i++) {
        float d = depth.read(min(base + uint2(i & 1u, i >> 1), full - 1));
        if (d > bd) { bd = d; best = i; }
    }
    uint2 q = min(base + uint2(best & 1u, best >> 1), full - 1);
    GiSurface s;
    if (bd <= 0.0 || !giSurfaceAt(p, depth, q, s) || length(s.rel) > p.limits.x) return float3(0.0);
    float3 f = p.camFrac.xyz + s.rel;
    uint level = min(uint(giLevelF(p, length(s.rel))), GI_MAX_LEVEL);
    return giInterpBlock(p, check, value, bv, f, giAirBlock(p.camBlock.xyz, f, s.face), s.face, level);
}

"""

/// Mirrors GiCl.
struct GiClGPU {
    var volOff = SIMD4<Float>.zero
    var volRel = SIMD4<Float>.zero
    var size = SIMD4<Float>.zero
    var fire = SIMD4<Float>.zero
}

private var giClDummy3D: MTLTexture?
private var giClDummyOut: MTLTexture?

/// The update's inputs (GiCache.bindRays): the volume's two textures (7, 8) and where a hit falls in them (buffer 20), from
/// the volume as its last update left it (the cache's frame runs before this frame's), or none.
func giClBind(_ enc: MTLComputeCommandEncoder, origin: SIMD3<Double>) {
    var c = GiClGPU()
    var rgb: MTLTexture?, aux: MTLTexture?
    if clBounceGain > 0, let v = ColoredLight.shared.bounceInputs {
        let o = SIMD3<Double>(origin.x.rounded(), origin.y.rounded(), origin.z.rounded())
        let rel = SIMD3<Float>(Float(o.x - Double(v.org.x)), Float(o.y - Double(v.org.y)), Float(o.z - Double(v.org.z)))
        c.volOff = SIMD4(rel + SIMD3<Float>(Float(v.orgMod.x), Float(v.orgMod.y), Float(v.orgMod.z)), 1)
        c.volRel = SIMD4(rel, 0)
        rgb = v.rgb
        aux = v.aux
    }
    c.size = SIMD4(Float(clSizeX), Float(clSizeY), Float(clSizeZ), clBounceGain)
    c.fire = SIMD4(clBucketColors[clFireBucket], 0)
    if rgb == nil {
        if giClDummy3D == nil {
            let d = MTLTextureDescriptor()
            d.textureType = .type3D
            d.pixelFormat = .rgba16Float
            d.width = 1; d.height = 1; d.depth = 1
            d.usage = .shaderRead
            giClDummy3D = ctx.device.makeTexture(descriptor: d)
        }
        rgb = giClDummy3D
        aux = giClDummy3D
    }
    enc.setTexture(rgb, index: 7)
    enc.setTexture(aux, index: 8)
    enc.setBytes(&c, length: MemoryLayout<GiClGPU>.stride, index: 20)
}

/// The request's block channel output (texture 5): the frame's, or a 1 x 1 stand-in (offline tests).
func giClBindOut(_ enc: MTLComputeCommandEncoder, _ out: MTLTexture?) {
    if let out { enc.setTexture(out, index: 5); return }
    if giClDummyOut == nil {
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg11b10Float, width: 1, height: 1, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        giClDummyOut = ctx.device.makeTexture(descriptor: d)
    }
    enc.setTexture(giClDummyOut, index: 5)
}

/// Offline compile check of the GI cache's kernels as this session's switches make them (with coloredlight: the bounce's
/// splices): 1 if gi_update, gi_request and gi_resolve build (errors to the log).
@_cdecl("mmc_debug_cl_gi_compile")
public func mmc_debug_cl_gi_compile() -> Int32 {
    let len = mmc_debug_gi_shader_source(UnsafeMutablePointer<CChar>.allocate(capacity: 1), 0)
    let buf = UnsafeMutablePointer<CChar>.allocate(capacity: Int(len) + 1)
    defer { buf.deallocate() }
    _ = mmc_debug_gi_shader_source(buf, len + 1)
    let src = "#define GI_ZERO_COPY 1\n" + skyShaderHeader + String(cString: buf)
    do {
        let lib = try ctx.device.makeLibrary(source: src, options: nil)
        for name in ["gi_update", "gi_request", "gi_resolve"] {
            guard let f = lib.makeFunction(name: name) else { log("coloredlight: no \(name)"); return 0 }
            _ = try ctx.device.makeComputePipelineState(function: f)
        }
        return 1
    } catch {
        log("coloredlight: GI kernels failed: \(error)")
        return 0
    }
}
