import Foundation
import Metal
import simd

// Occlusion culling for vanilla's chunk sections, with the same box test as the LOD (Lod.swift). Vanilla
// already skips sections outside the frustum and ones its visibility graph rules out (sealed caves), but it
// draws distant surface sections hidden behind hills. After solid terrain, every candidate section's box
// (16 blocks, grown by one) is rasterized in the same render pass with depth testing and no writes; the
// fragment function marks the section visible. When the frame completes, sections that weren't marked are
// reported hidden, and the mod leaves them out of the next frames' draw lists. Every candidate is tested
// again each frame, so a hidden section comes back within 2-3 frames of coming into view. The mod never
// tests or skips sections near the camera.

final class SectionVisSet {
    static let capacity = 1 << 16
    let marks: MTLBuffer      // UInt32 per slot
    let boxes: MTLBuffer      // two float4 per slot (lo, hi), camera-relative
    var keys: [Int64] = []
    var frame: UInt64 = 0
    var camera = SIMD3<Double>(repeating: 0)
    var pending = false       // encoded, not yet harvested (render thread)
    var done = false          // command buffer completed (completion thread, under lock)

    init?() {
        guard let m = ctx.device.makeBuffer(length: Self.capacity * 4, options: [.storageModeShared]),
              let b = ctx.device.makeBuffer(length: Self.capacity * 32, options: [.storageModeShared]) else { return nil }
        m.label = "MetalMC section visibility"
        b.label = "MetalMC section boxes"
        marks = m
        boxes = b
    }
}

final class SectionOcclusion: @unchecked Sendable {
    static let shared = SectionOcclusion()
    let lock = NSLock()
    var sets: [SectionVisSet] = []
    var frame: UInt64 = 0
    var hidden: [Int64] = []          // from the newest completed test
    var hiddenFrame: UInt64 = 0
    var hiddenCamera = SIMD3<Double>(repeating: 0)

    /// Reads completed tests (newest wins) and returns a free set for this frame, or nil if all are in flight.
    func harvestAndAcquire() -> SectionVisSet? {
        lock.lock()
        let ready = sets.filter { $0.pending && $0.done }.sorted { $0.frame < $1.frame }
        lock.unlock()
        for s in ready {
            if s.frame > hiddenFrame {
                let marks = s.marks.contents().bindMemory(to: UInt32.self, capacity: SectionVisSet.capacity)
                var h: [Int64] = []
                for (i, k) in s.keys.enumerated() where marks[i] == 0 { h.append(k) }
                hidden = h
                hiddenFrame = s.frame
                hiddenCamera = s.camera
            }
            s.keys.removeAll(keepingCapacity: true)
            lock.lock(); s.pending = false; s.done = false; lock.unlock()
        }
        if let free = sets.first(where: { !$0.pending }) { return free }
        if sets.count < 4, let s = SectionVisSet() { sets.append(s); return s }
        return nil
    }
}

/// Section keys (SectionPos.asLong) hidden in the newest completed test, written to `out` (at most `max`).
/// Returns the count, or 0 if the camera has moved more than 16 blocks since that test.
@_cdecl("mmc_occ_hidden")
public func mmc_occ_hidden(_ camera: UnsafePointer<Double>, _ out: UnsafeMutablePointer<Int64>, _ max: Int32) -> Int32 {
    let o = SectionOcclusion.shared
    _ = o.harvestAndAcquire()
    let cam = SIMD3(camera[0], camera[1], camera[2])
    guard o.hiddenFrame > 0, simd_distance(cam, o.hiddenCamera) < 16 else { return 0 }
    let n = min(Int(max), o.hidden.count)
    for i in 0..<n { out[i] = o.hidden[i] }
    return Int32(n)
}

/// Tests the boxes of `count` sections (SectionPos.asLong keys) in the open render pass. p: proj[16] and
/// view[16] (column-major, camera-relative); cam: camera position. Leaves ctx.pipe nil so Java re-applies
/// Minecraft's pipeline state.
@_cdecl("mmc_occ_test")
public func mmc_occ_test(_ p: UnsafePointer<Float>, _ cam: UnsafePointer<Double>, _ keys: UnsafePointer<Int64>, _ count: Int32) {
    guard count > 0, let enc = ctx.pass, !ctx.scissorEmpty, let cb = ctx.cb else { return }
    let o = SectionOcclusion.shared
    guard let set = o.harvestAndAcquire(),
          let pipe = LodRenderer.shared.pipeline(colorFormats: ctx.passColorFormats, depth: ctx.passDepthFormat, box: true) else { return }
    func mat(_ off: Int) -> simd_float4x4 {
        simd_float4x4(SIMD4(p[off], p[off + 1], p[off + 2], p[off + 3]), SIMD4(p[off + 4], p[off + 5], p[off + 6], p[off + 7]),
                      SIMD4(p[off + 8], p[off + 9], p[off + 10], p[off + 11]), SIMD4(p[off + 12], p[off + 13], p[off + 14], p[off + 15]))
    }
    var u = LodUniforms(proj: mat(0), view: mat(16), fogColor: .zero, envStart: 0, envEnd: 0, rdStart: 0, rdEnd: 0, discardRadius: 0, sky: 0)
    let cx = cam[0], cy = cam[1], cz = cam[2]
    let n = min(Int(count), SectionVisSet.capacity)
    let boxes = set.boxes.contents().bindMemory(to: SIMD4<Float>.self, capacity: 2 * SectionVisSet.capacity)
    let marks = set.marks.contents().bindMemory(to: UInt32.self, capacity: SectionVisSet.capacity)
    set.keys.removeAll(keepingCapacity: true)
    for i in 0..<n {
        let k = keys[i]
        // SectionPos.asLong: x in bits 42-63, z in bits 20-41, y in bits 0-19 (all signed).
        let sx = Int(k >> 42), sy = Int((k << 44) >> 44), sz = Int((k << 22) >> 42)
        let lo = SIMD3(Float(Double(sx * 16) - cx), Float(Double(sy * 16) - cy), Float(Double(sz * 16) - cz)) - 1
        boxes[2 * i] = SIMD4(lo, 0)
        boxes[2 * i + 1] = SIMD4(lo + 18, 0)
        marks[i] = 0
        set.keys.append(k)
    }
    enc.setRenderPipelineState(pipe)
    enc.setDepthStencilState(ctx.depthState(compare: .greaterEqual, write: false))
    enc.setCullMode(.back)
    enc.setDepthBias(0, slopeScale: 0, clamp: 0)
    enc.setVertexBytes(&u, length: MemoryLayout<LodUniforms>.stride, index: 19)
    enc.setVertexBuffer(set.boxes, offset: 0, index: 22)
    enc.setFragmentBuffer(set.marks, offset: 0, index: 23)
    enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 36, instanceCount: n)
    o.frame += 1
    set.frame = o.frame
    set.camera = SIMD3(cx, cy, cz)
    set.pending = true
    cb.addCompletedHandler { _ in
        o.lock.lock(); set.done = true; o.lock.unlock()
    }
    ctx.statOccSections += n
    ctx.pipe = nil
    ctx.boundPipeState = nil
}
