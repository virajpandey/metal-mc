// MetalFX temporal denoised scaler cost on this Mac: denoise + upscale from W x H to 2W x 2H (defaults 1728 x 1117 ->
// 3456 x 2234), 40 frames, median GPU time. build: swiftc -O tools/bench/dnbench.swift -o bench_out/dnbench
import Foundation
import Metal
import MetalFX
import simd

let args = CommandLine.arguments
let iw = args.count > 1 ? Int(args[1])! : 1728, ih = args.count > 2 ? Int(args[2])! : 1117
let ow = args.count > 3 ? Int(args[3])! : 3456, oh = args.count > 4 ? Int(args[4])! : 2234
let dev = MTLCreateSystemDefaultDevice()!
print("device: \(dev.name), denoised scaler supported: \(MTLFXTemporalDenoisedScalerDescriptor.supportsDevice(dev))")
let d = MTLFXTemporalDenoisedScalerDescriptor()
d.colorTextureFormat = .rgba16Float
d.depthTextureFormat = .depth32Float
d.motionTextureFormat = .rg16Float
d.diffuseAlbedoTextureFormat = .rgba16Float
d.specularAlbedoTextureFormat = .rgba16Float
d.normalTextureFormat = .rgba16Float
d.roughnessTextureFormat = .r16Float
d.outputTextureFormat = .rgba16Float
d.inputWidth = iw; d.inputHeight = ih; d.outputWidth = ow; d.outputHeight = oh
guard let s = d.makeTemporalDenoisedScaler(device: dev) else { print("no scaler"); exit(1) }
func tex(_ f: MTLPixelFormat, _ w: Int, _ h: Int, _ u: MTLTextureUsage) -> MTLTexture {
    let t = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: f, width: w, height: h, mipmapped: false)
    t.usage = u.union([.shaderRead, .renderTarget]); t.storageMode = .private
    return dev.makeTexture(descriptor: t)!
}
let color = tex(.rgba16Float, iw, ih, s.colorTextureUsage), depth = tex(.depth32Float, iw, ih, s.depthTextureUsage)
let motion = tex(.rg16Float, iw, ih, s.motionTextureUsage), diff = tex(.rgba16Float, iw, ih, s.diffuseAlbedoTextureUsage)
let spec = tex(.rgba16Float, iw, ih, s.specularAlbedoTextureUsage), normal = tex(.rgba16Float, iw, ih, s.normalTextureUsage)
let rough = tex(.r16Float, iw, ih, s.roughnessTextureUsage), out = tex(.rgba16Float, ow, oh, s.outputTextureUsage)
let q = dev.makeCommandQueue()!
do {   // non-trivial inputs: clear to different values
    let cb = q.makeCommandBuffer()!
    for (t, v) in [(color, 0.4), (motion, 0.0), (diff, 0.6), (spec, 0.04), (normal, 0.5), (rough, 0.7)] {
        let rp = MTLRenderPassDescriptor()
        rp.colorAttachments[0].texture = t; rp.colorAttachments[0].loadAction = .clear
        rp.colorAttachments[0].clearColor = MTLClearColor(red: v, green: v * 0.9, blue: v * 0.8, alpha: 1); rp.colorAttachments[0].storeAction = .store
        cb.makeRenderCommandEncoder(descriptor: rp)!.endEncoding()
    }
    let rp = MTLRenderPassDescriptor()
    rp.depthAttachment.texture = depth; rp.depthAttachment.loadAction = .clear; rp.depthAttachment.clearDepth = 0.3; rp.depthAttachment.storeAction = .store
    cb.makeRenderCommandEncoder(descriptor: rp)!.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
}
s.colorTexture = color; s.depthTexture = depth; s.motionTexture = motion; s.diffuseAlbedoTexture = diff
s.specularAlbedoTexture = spec; s.normalTexture = normal; s.roughnessTexture = rough; s.outputTexture = out
s.isDepthReversed = true
s.worldToViewMatrix = matrix_identity_float4x4
s.viewToClipMatrix = simd_float4x4(diagonal: SIMD4(1.2, 1.9, 0, 0))
var times: [Double] = []
for i in 0..<40 {
    s.jitterOffsetX = Float(i % 8) / 8 - 0.5; s.jitterOffsetY = Float((i * 3) % 8) / 8 - 0.5
    s.shouldResetHistory = i == 0
    let cb = q.makeCommandBuffer()!
    s.encode(commandBuffer: cb)
    cb.commit(); cb.waitUntilCompleted()
    if i >= 8 { times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000) }
}
times.sort()
print(String(format: "denoise + upscale %dx%d -> %dx%d: median %.2f ms, min %.2f ms", iw, ih, ow, oh, times[times.count / 2], times[0]))
