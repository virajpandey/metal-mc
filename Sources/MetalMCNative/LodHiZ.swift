import Foundation
import Metal
import simd

// The LOD's occlusion culling (speed, 2026-10-13; lod_cull, LodQuadCull in Lod.swift). Only 10-20% of the level-0 quads the
// LOD issues own a pixel (docs/lod-design.md, "Most of those quads are hidden"), and on this GPU the vertex stage costs per
// quad issued (four vertex invocations), culled later or not: so the quads hidden behind nearer terrain have to be left out
// before the draw. This is the pyramid they're tested against: last frame's occluders' depth, the farthest (reverse-Z: the
// least) over each texel's footprint, half the frame's resolution at its base.
//
// Occluders are opaque lit terrain at its own depth only (the G-buffer's key matches the depth), so glass, water, entities,
// particles and the sky occlude nothing: what shows through them is never culled. The anti-aliasing's resolve writes the
// base as it loads every pixel's depth and G-buffer anyway (taa_resolve, texture 31, p.range.z); a small pass makes the
// levels above. The culling pass reads the pyramid of the frame before, with that frame's projection and camera: a quad is
// culled if, projected as that frame saw it, its nearest depth lies behind the pyramid's farthest over its whole box, grown
// by a margin for the parallax the camera's move since (LodQuadCull.encode). Turning alone changes nothing that's hidden.

private let lodHiZSource = """
#include <metal_stdlib>
using namespace metal;

// A level from the one below: the least of the 3 x 3 texels from 2x, 2y (clamped), a superset of its 2 x 2 footprint, so
// odd sizes stay conservative.
kernel void lod_hiz_down(texture2d<float, access::read> src [[texture(0)]], texture2d<float, access::write> dst [[texture(1)]],
                         uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= dst.get_width() || gid.y >= dst.get_height()) return;
    uint2 mx = uint2(src.get_width() - 1, src.get_height() - 1);
    float m = 1e30;
    for (uint j = 0; j < 3; j++)
        for (uint i = 0; i < 3; i++) m = min(m, src.read(min(gid * 2u + uint2(i, j), mx)).r);
    dst.write(float4(m), gid);
}
"""

/// METALMC_EXP=nohiz: no occlusion culling (the quad culling's other tests stay).
let lodHiZEnabled = !experiments.contains("nohiz")

final class LodHiZ: @unchecked Sendable {
    static let shared = LodHiZ()
    private let lock = NSLock()
    private var pipe: MTLComputePipelineState?
    private var failed = false
    private var tex: [MTLTexture] = []
    private var views: [[MTLTexture]] = []
    private var writing = 0
    /// This frame's projection and camera (the culling pass notes them as it runs; the resolve writes this frame's depth).
    private var noted: (viewProj: simd_float4x4, cam: SIMD3<Double>, frame: UInt64)?
    /// The last pyramid made, with the frame it was made from: what the next frame's culling tests against.
    private var ready: (tex: MTLTexture, viewProj: simd_float4x4, cam: SIMD3<Double>, frame: UInt64)?
    private var frameCounter: UInt64 = 0

    private func ensurePipe() -> MTLComputePipelineState? {
        if let pipe { return pipe }
        if failed { return nil }
        do {
            let lib = try ShaderLab.library("lod_hiz", lodHiZSource) { [self] _ in pipe = nil; failed = false }
            guard let f = lib.makeFunction(name: "lod_hiz_down") else { throw NSError(domain: "metalmc", code: 1) }
            pipe = try ctx.device.makeComputePipelineState(function: f)
        } catch {
            log("LOD occlusion: the pyramid's kernel failed: \(error)")
            failed = true
        }
        return pipe
    }

    /// The culling pass, as it encodes this frame's culling: the projection the level is drawn with and the camera.
    func note(viewProj: simd_float4x4, cam: SIMD3<Double>) {
        lock.lock()
        frameCounter &+= 1
        noted = (viewProj, cam, frameCounter)
        lock.unlock()
    }

    /// The resolve (render thread): the base level to write this frame (texture 31), for a frame of width x height, or
    /// nil (no culling noted this frame, or the kernel failed).
    func target(width: Int, height: Int) -> MTLTexture? {
        guard lodHiZEnabled, ensurePipe() != nil else { return nil }
        lock.lock(); let n = noted; lock.unlock()
        guard n != nil else { return nil }
        let w = max(1, width / 2), h = max(1, height / 2)
        if tex.isEmpty || tex[0].width != w || tex[0].height != h {
            tex = []; views = []
            let levels = Int(log2(Double(max(w, h)))) + 1
            for k in 0..<2 {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: w, height: h, mipmapped: true)
                d.usage = [.shaderRead, .shaderWrite]
                d.storageMode = .private
                guard let t = ctx.device.makeTexture(descriptor: d) else { return nil }
                t.label = "MetalMC LOD occluders \(k)"
                tex.append(t)
                views.append((0..<levels).compactMap { t.makeTextureView(pixelFormat: .r32Float, textureType: .type2D, levels: $0..<($0 + 1), slices: 0..<1) })
            }
            lock.lock(); ready = nil; lock.unlock()
        }
        return views[writing].first
    }

    /// After the resolve wrote the base (same command buffer): the levels above, then the pyramid is the one the next
    /// frame's culling tests against.
    func finish(cb: MTLCommandBuffer) {
        guard let pipe, !tex.isEmpty, let enc = cb.makeComputeCommandEncoder() else { return }
        enc.label = "MetalMC LOD occluders"
        enc.setComputePipelineState(pipe)
        let v = views[writing]
        for k in 1..<v.count {
            enc.setTexture(v[k - 1], index: 0)
            enc.setTexture(v[k], index: 1)
            enc.dispatchThreads(MTLSize(width: v[k].width, height: v[k].height, depth: 1),
                                threadsPerThreadgroup: MTLSize(width: min(16, v[k].width), height: min(16, v[k].height), depth: 1))
        }
        enc.endEncoding()
        lock.lock()
        if let n = noted { ready = (tex[writing], n.viewProj, n.cam, n.frame) }
        noted = nil
        lock.unlock()
        writing = 1 - writing
    }

    private var stand: MTLTexture?
    /// A 1 x 1 stand-in for the culling pass's texture when there's no pyramid (camDelta.w 0: it isn't read).
    func standIn() -> MTLTexture? {
        if let stand { return stand }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: 1, height: 1, mipmapped: false)
        d.usage = .shaderRead
        stand = ctx.device.makeTexture(descriptor: d)
        return stand
    }

    /// The culling pass: the pyramid of the frame before this one (nil if there's none: the frame before made none).
    func previous() -> (tex: MTLTexture, viewProj: simd_float4x4, cam: SIMD3<Double>)? {
        lock.lock(); defer { lock.unlock() }
        guard let r = ready, r.frame == frameCounter else { return nil }
        return (r.tex, r.viewProj, r.cam)
    }
}

/// Debug: the LOD's shader source (lod_vs, lod_cull and the rest) as this session's switches make it, for an offline
/// compile check. Returns its length (the needed length if `len` is too small).
@_cdecl("mmc_debug_lod_shader_source")
public func mmc_debug_lod_shader_source(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let bytes = Array(lodLibrarySource().utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}

/// Debug: the occluders' pyramid's kernel, likewise.
@_cdecl("mmc_debug_lod_hiz_source")
public func mmc_debug_lod_hiz_source(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let bytes = Array(lodHiZSource.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}
