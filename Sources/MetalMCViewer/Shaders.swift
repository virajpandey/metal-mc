/// Metal Shading Language source, compiled at runtime so SwiftPM needs no metallib build step.
let shaderSource = """
#include <metal_stdlib>
using namespace metal;

struct Uniforms { float4x4 viewProj; float4 cameraPos; float4 fog; float4 sky; };
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
                  const device float4* origins [[buffer(2)]],
                  constant uint* colors [[buffer(3)]]) {
    uint q = quads[vid >> 2];
    uint corner = vid & 3;
    uint face = (q >> 12) & 7;
    float3 local = float3(q & 15, (q >> 4) & 15, (q >> 8) & 15);
    float3 scale = extentScale(face, float(((q >> 23) & 15) + 1), float(((q >> 27) & 15) + 1));
    float3 p = origins[section].xyz + local + kCorners[face][corner] * scale;

    VOut o;
    o.position = u.viewProj * float4(p, 1.0);
    o.color = unpack_unorm4x8_to_float(colors[((q >> 15) & 255) * 6 + face]);
    o.wpos = p;
    return o;
}

// Fog distance is computed per pixel, so large greedy quads fog exactly like 1x1 ones.
fragment float4 fmain(VOut in [[stage_in]], constant Uniforms& u [[buffer(1)]]) {
    float dist = distance(in.wpos, u.cameraPos.xyz);
    float f = saturate((dist - u.fog.x) / max(u.fog.y - u.fog.x, 1e-3));
    float3 rgb = mix(in.color.rgb, u.sky.rgb, f);
    return float4(rgb, in.color.a);
}
"""
