import Foundation
import Metal
import MetalFX
import simd

// Temporal anti-aliasing with MetalFX's temporal scaler at 1:1 (lod-independent; config taa). Each frame the level
// is drawn with a sub-pixel jitter of its projection (GameRendererLodMixin); after the level and before the hand
// and HUD, the scaler blends the frame with its history, reprojected with per-pixel motion. Motion comes from the
// depth buffer and the camera (the world is static; entities that move smear a little): a pixel's position is
// rebuilt from depth with this frame's matrices and projected with the previous frame's.
//
// Minecraft's shaders are translated with a vertical flip (flip_vert_y), so texture row 0 is the bottom of
// vanilla's clip space: normalized device y = 2 * v - 1 for texture coordinate v, in both frames.

private let taaShaderSource = """
#include <metal_stdlib>
using namespace metal;

struct TaaParams {
    float4x4 invCur;     // inverse of this frame's projection * view rotation (unjittered)
    float4x4 prev;       // the previous frame's projection * view rotation (unjittered)
    float4 camDelta;     // xyz: this frame's camera position minus the previous one's
    float2 size;         // texture size in pixels
};

// Per-pixel motion in pixels, from this frame's position to where the same world point was in the last frame.
kernel void taa_motion(depth2d<float, access::read> depth [[texture(0)]],
                       texture2d<float, access::write> motion [[texture(1)]],
                       constant TaaParams& p [[buffer(0)]],
                       uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= uint(p.size.x) || gid.y >= uint(p.size.y)) return;
    float d = depth.read(gid);
    float2 uv = (float2(gid) + 0.5) / p.size;
    float2 ndc = uv * 2.0 - 1.0;
    float4 prevClip;
    if (d <= 0.0) {
        // Sky (reverse-Z: 0 is infinitely far): only the camera's rotation moves it.
        float4 dir = p.invCur * float4(ndc, 0.0, 1.0);
        prevClip = p.prev * float4(dir.xyz, 0.0);
    } else {
        float4 rel = p.invCur * float4(ndc, d, 1.0);
        rel.xyz /= rel.w;
        prevClip = p.prev * float4(rel.xyz + p.camDelta.xyz, 1.0);
    }
    float2 prevUV = (prevClip.xy / prevClip.w) * 0.5 + 0.5;
    motion.write(float4((prevUV - uv) * p.size, 0.0, 0.0), gid);
}
"""

/// METALMC_TAASIGN=sx,sy: signs applied to the jitter handed to MetalFX (convention check).
private let taaJitterSign: SIMD2<Float> = {
    let parts = (ProcessInfo.processInfo.environment["METALMC_TAASIGN"] ?? "1,1").split(separator: ",").compactMap { Float($0) }
    return parts.count == 2 ? SIMD2(parts[0], parts[1]) : SIMD2(1, 1)
}()

private struct TaaParams {
    var invCur: simd_float4x4
    var prev: simd_float4x4
    var camDelta: SIMD4<Float>
    var size: SIMD2<Float>
}

final class Taa: @unchecked Sendable {
    static let shared = Taa()

    var scaler: MTLFXTemporalScaler?
    var scalerKey = ""
    var motion: MTLTexture?
    var output: MTLTexture?
    var motionPipe: MTLComputePipelineState?
    var prevViewProj = matrix_identity_float4x4
    var prevCam = SIMD3<Double>(repeating: .nan)
    var lastApply: UInt64 = 0   // uptime of the last frame anti-aliased; after a gap (menus, loading) the history restarts
    var failed = false

    func ensure(color: MTLTexture, depth: MTLTexture) -> Bool {
        if failed { return false }
        let key = "\(color.width)x\(color.height)/\(color.pixelFormat.rawValue)/\(depth.pixelFormat.rawValue)"
        if key == scalerKey, scaler != nil { return true }
        let d = MTLFXTemporalScalerDescriptor()
        d.colorTextureFormat = color.pixelFormat
        d.depthTextureFormat = depth.pixelFormat
        d.motionTextureFormat = .rg16Float
        d.outputTextureFormat = color.pixelFormat
        d.inputWidth = color.width
        d.inputHeight = color.height
        d.outputWidth = color.width
        d.outputHeight = color.height
        guard let s = d.makeTemporalScaler(device: ctx.device) else {
            log("TAA: MetalFX temporal scaler unavailable for \(key)")
            failed = true
            return false
        }
        s.isDepthReversed = true
        let md = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rg16Float, width: color.width, height: color.height, mipmapped: false)
        md.usage = [.shaderRead, .shaderWrite]
        md.storageMode = .private
        let od = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: color.pixelFormat, width: color.width, height: color.height, mipmapped: false)
        od.usage = s.outputTextureUsage
        od.storageMode = .private
        guard let m = ctx.device.makeTexture(descriptor: md), let o = ctx.device.makeTexture(descriptor: od) else { return false }
        if motionPipe == nil {
            do {
                let lib = try ctx.device.makeLibrary(source: taaShaderSource, options: nil)
                motionPipe = try ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "taa_motion")!)
            } catch {
                log("TAA: motion shader failed: \(error)")
                failed = true
                return false
            }
        }
        scaler = s
        motion = m
        output = o
        scalerKey = key
        prevCam = SIMD3(repeating: .nan)   // new history
        log("TAA: MetalFX temporal scaler \(key)")
        return true
    }
}

/// Anti-aliases the level just drawn into `color` (a TextureBox handle) using `depth`. p: this frame's unjittered
/// projection[16] and view rotation[16] (column-major); cam: camera position (world, doubles); jitter: the sub-pixel
/// offset the frame was drawn with, in pixels (x right, y up in vanilla's clip space). Returns 1 if it ran.
@_cdecl("mmc_taa_apply")
public func mmc_taa_apply(_ colorHandle: Int64, _ depthHandle: Int64, _ p: UnsafePointer<Float>, _ cam: UnsafePointer<Double>,
                          _ jitterX: Float, _ jitterY: Float, _ reset: Int32) -> Int32 {
    let t = Taa.shared
    let color = (from(colorHandle) as TextureBox).texture, depth = (from(depthHandle) as TextureBox).texture
    guard ctx.pass == nil, t.ensure(color: color, depth: depth), let scaler = t.scaler, let motion = t.motion,
          let output = t.output, let motionPipe = t.motionPipe else { return 0 }
    func mat(_ o: Int) -> simd_float4x4 {
        simd_float4x4(SIMD4(p[o], p[o + 1], p[o + 2], p[o + 3]), SIMD4(p[o + 4], p[o + 5], p[o + 6], p[o + 7]),
                      SIMD4(p[o + 8], p[o + 9], p[o + 10], p[o + 11]), SIMD4(p[o + 12], p[o + 13], p[o + 14], p[o + 15]))
    }
    let viewProj = mat(0) * mat(16)
    let camera = SIMD3(cam[0], cam[1], cam[2])
    let delta = camera - t.prevCam
    // A new history after a jump (teleport, respawn, dimension change), a gap of frames without anti-aliasing
    // (a menu, the loading screen) or on request.
    let now = DispatchTime.now().uptimeNanoseconds
    let jumped = !(simd_length(delta) < 16) || now - t.lastApply > 100_000_000
    t.lastApply = now
    let prev = jumped ? viewProj : t.prevViewProj
    ctx.endBlit()
    let cb = ctx.ensureCB()
    guard let enc = cb.makeComputeCommandEncoder() else { return 0 }
    var params = TaaParams(invCur: viewProj.inverse, prev: prev,
                           camDelta: jumped ? .zero : SIMD4(Float(delta.x), Float(delta.y), Float(delta.z), 0),
                           size: SIMD2(Float(color.width), Float(color.height)))
    enc.setComputePipelineState(motionPipe)
    enc.setTexture(depth, index: 0)
    enc.setTexture(motion, index: 1)
    enc.setBytes(&params, length: MemoryLayout<TaaParams>.stride, index: 0)
    let tg = MTLSize(width: 16, height: 16, depth: 1)
    enc.dispatchThreadgroups(MTLSize(width: (color.width + 15) / 16, height: (color.height + 15) / 16, depth: 1), threadsPerThreadgroup: tg)
    enc.endEncoding()
    scaler.colorTexture = color
    scaler.depthTexture = depth
    scaler.motionTexture = motion
    scaler.outputTexture = output
    scaler.inputContentWidth = color.width
    scaler.inputContentHeight = color.height
    // MetalFX takes the jitter in pixels with y down the texture; vanilla's clip-space y runs up the texture here
    // (flip_vert_y), so y keeps its sign.
    scaler.jitterOffsetX = jitterX * taaJitterSign.x
    scaler.jitterOffsetY = jitterY * taaJitterSign.y
    scaler.motionVectorScaleX = 1
    scaler.motionVectorScaleY = 1
    scaler.reset = jumped || reset != 0
    scaler.encode(commandBuffer: cb)
    let blit = ctx.blitEncoder()
    blit.copy(from: output, to: color)
    ctx.endBlit()
    t.prevViewProj = viewProj
    t.prevCam = camera
    return 1
}
