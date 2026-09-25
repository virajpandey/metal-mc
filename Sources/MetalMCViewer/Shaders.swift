import MetalMCCore
/// Metal Shading Language source, compiled at runtime so SwiftPM needs no metallib build step.
let shaderSource = """
#include <metal_stdlib>
using namespace metal;

struct Uniforms { float4x4 viewProj; float4 cameraPos; float4 fog; float4 sky; };
// Per-section transform: world-space origin and per-axis cell size (1 for full detail; (s, v, s) for LOD rings).
struct Xform { float4 origin; float4 scale; };
struct VOut { float4 position [[position]]; float4 color; float3 wpos; };

// Unit-cube corners per face, CCW seen from outside. Face order matches Mesher.faces: +X -X +Y -Y +Z -Z.
constant float3 kCorners[6][4] = {
    { float3(1,0,0), float3(1,1,0), float3(1,1,1), float3(1,0,1) },
    { float3(0,0,1), float3(0,1,1), float3(0,1,0), float3(0,0,0) },
    { float3(0,1,0), float3(0,1,1), float3(1,1,1), float3(1,1,0) },
    { float3(0,0,0), float3(1,0,0), float3(1,0,1), float3(0,0,1) },
    { float3(1,0,1), float3(1,1,1), float3(0,1,1), float3(0,0,1) },
    { float3(0,0,0), float3(0,1,0), float3(1,1,0), float3(1,0,0) },
};

// Greedy extent scale per face: the normal axis stays 1, the U/V tangent axes take width/height.
static float3 extentScale(uint face, float w, float h) {
    if (face < 2) return float3(1, w, h);   // X faces: U = y, V = z
    if (face < 4) return float3(w, 1, h);   // Y faces: U = x, V = z
    return float3(w, h, 1);                 // Z faces: U = x, V = y
}

// Indexed draw over a shared pattern (4q+0, 4q+1, 4q+2, 4q+0, 4q+2, 4q+3) with baseVertex = 4 * firstQuad,
// so vertex_id = 4 * quad + corner and the two shared corners of each quad are shaded once.
vertex VOut vquad(uint vid [[vertex_id]],
                  uint section [[base_instance]],
                  const device uint* quads [[buffer(0)]],
                  constant Uniforms& u [[buffer(1)]],
                  const device Xform* xforms [[buffer(2)]],
                  constant uint* colors [[buffer(3)]]) {
    uint q = quads[vid >> 2];
    uint corner = vid & 3;
    uint face = (q >> 12) & 7;
    float3 local = float3(q & 15, (q >> 4) & 15, (q >> 8) & 15);
    float3 scale = extentScale(face, float(((q >> 23) & 15) + 1), float(((q >> 27) & 15) + 1));
    Xform xf = xforms[section];
    float3 p = xf.origin.xyz + (local + kCorners[face][corner] * scale) * xf.scale.xyz;

    VOut o;
    o.position = u.viewProj * float4(p, 1.0);
    o.color = unpack_unorm4x8_to_float(colors[((q >> 15) & 255) * 6 + face]);
    o.wpos = p;
    return o;
}

// ---- GPU-driven culling: one thread per section writes its draws into an indirect command buffer ----

struct CullUniforms {
    float4 planes[6];
    float4 cameraPos;
    float fogEnd;
    uint sectionCount;
    uint waterBase;      // first ICB slot of the water range (3 * sectionCount)
    uint bucketsOn;
};

struct SectionGPU {
    float4 minB;
    float4 maxB;
    uint opaqueStart;
    uint waterStart;
    uint waterCount;
    uint pad;
    int faceOffsets[8];  // cumulative face-bucket starts; [6] = opaque count
};

struct ICBContainer { command_buffer cmds [[id(0)]]; };
struct CullStats { atomic_uint drawn; atomic_uint triangles; };

kernel void cullSections(uint i [[thread_position_in_grid]],
                         constant CullUniforms& cu [[buffer(0)]],
                         const device SectionGPU* secs [[buffer(1)]],
                         device ICBContainer& icb [[buffer(2)]],
                         const device uint* pattern [[buffer(3)]],
                         device CullStats& stats [[buffer(4)]]) {
    if (i >= cu.sectionCount) return;
    SectionGPU s = secs[i];
    float3 lo = s.minB.xyz, hi = s.maxB.xyz, c = cu.cameraPos.xyz;
    float3 d = max(max(lo - c, c - hi), float3(0.0));
    if (length(d) > cu.fogEnd) return;
    for (int p = 0; p < 6; p++) {
        float4 pl = cu.planes[p];
        float3 v = select(lo, hi, pl.xyz >= 0.0);
        if (dot(pl.xyz, v) + pl.w < 0.0) return;
    }
    atomic_fetch_add_explicit(&stats.drawn, 1u, memory_order_relaxed);

    uint mask = 63u;
    if (cu.bucketsOn != 0u) {
        mask = (c.x > lo.x ? 1u : 0u) | (c.x < hi.x ? 2u : 0u) | (c.y > lo.y ? 4u : 0u)
             | (c.y < hi.y ? 8u : 0u) | (c.z > lo.z ? 16u : 0u) | (c.z < hi.z ? 32u : 0u);
    }
    uint slot = 0u, tris = 0u;
    int f = 0;
    while (f < 6) {
        if ((mask & (1u << f)) == 0u) { f++; continue; }
        int e = f;
        while (e < 6 && (mask & (1u << e)) != 0u) e++;
        int start = s.faceOffsets[f], end = s.faceOffsets[e];
        if (end > start && slot < 3u) {
            render_command cmd(icb.cmds, i * 3u + slot);
            cmd.draw_indexed_primitives(primitive_type::triangle, uint(end - start) * 6u, pattern, 1u,
                                        (s.opaqueStart + uint(start)) * 4u, i);
            slot++;
            tris += uint(end - start) * 2u;
        }
        f = e;
    }
    if (s.waterCount > 0u) {
        render_command cmd(icb.cmds, cu.waterBase + i);
        cmd.draw_indexed_primitives(primitive_type::triangle, s.waterCount * 6u, pattern, 1u, s.waterStart * 4u, i);
        tris += s.waterCount * 2u;
    }
    atomic_fetch_add_explicit(&stats.triangles, tris, memory_order_relaxed);
}

// Fog distance is computed per pixel, so large greedy quads fog exactly like 1x1 ones.
fragment float4 fmain(VOut in [[stage_in]], constant Uniforms& u [[buffer(1)]]) {
    float dist = distance(in.wpos, u.cameraPos.xyz);
    float f = saturate((dist - u.fog.x) / max(u.fog.y - u.fog.x, 1e-3));
    float3 rgb = mix(in.color.rgb, u.sky.rgb, f);
    return float4(rgb, in.color.a);
}
"""
