/// Metal Shading Language source, compiled at runtime so SwiftPM needs no metallib build step.
let shaderSource = """
#include <metal_stdlib>
using namespace metal;

struct VertexIn { packed_float3 pos; uint color; };
struct Uniforms { float4x4 viewProj; float4 cameraPos; float4 fog; float4 sky; };
struct VOut { float4 position [[position]]; float4 color; float dist; };

vertex VOut vmain(uint vid [[vertex_id]],
                  const device VertexIn* verts [[buffer(0)]],
                  constant Uniforms& u [[buffer(1)]]) {
    VertexIn v = verts[vid];
    float3 p = float3(v.pos);
    VOut o;
    o.position = u.viewProj * float4(p, 1.0);
    o.color = unpack_unorm4x8_to_float(v.color);
    o.dist = distance(p, u.cameraPos.xyz);
    return o;
}

fragment float4 fmain(VOut in [[stage_in]], constant Uniforms& u [[buffer(1)]]) {
    float f = saturate((in.dist - u.fog.x) / max(u.fog.y - u.fog.x, 1e-3));
    float3 rgb = mix(in.color.rgb, u.sky.rgb, f);
    return float4(rgb, in.color.a);
}
"""
