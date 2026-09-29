import Foundation
import Metal
import simd

// Debug: how fast is the M3's hardware ray tracing on the LOD's geometry? One primitive acceleration structure per
// chosen node (its quads as triangles, camera-relative), an instance structure over them, then a primary ray and a
// sun shadow ray per pixel of a width x height image. The question it answers: could sun shadows (and later light
// bouncing) for all the far terrain be traced at 120 Hz. Not used for drawing.

private let rtProbeSource = """
#include <metal_stdlib>
#include <metal_raytracing>
using namespace metal;
using namespace raytracing;

struct ProbeParams {
    float3 forward; float tanX;
    float3 right; float tanY;
    float3 up; float pad;
    float3 sun; uint width;
    uint height;
};

kernel void rt_probe(instance_acceleration_structure accel [[buffer(0)]],
                     constant ProbeParams& p [[buffer(1)]],
                     device atomic_uint* counts [[buffer(2)]],
                     uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= p.width || gid.y >= p.height) return;
    float2 ndc = float2((float(gid.x) + 0.5) / float(p.width) * 2.0 - 1.0, 1.0 - (float(gid.y) + 0.5) / float(p.height) * 2.0);
    ray r;
    r.origin = float3(0.0);
    r.direction = normalize(p.forward + ndc.x * p.tanX * p.right + ndc.y * p.tanY * p.up);
    r.min_distance = 0.05;
    r.max_distance = 100000.0;
    intersector<triangle_data, instancing> isect;
    isect.accept_any_intersection(false);
    intersection_result<triangle_data, instancing> hit = isect.intersect(r, accel, 0xFF);
    if (hit.type == intersection_type::none) return;
    atomic_fetch_add_explicit(&counts[0], 1u, memory_order_relaxed);
    uint bin = hit.distance < 1.0 ? 2u : (hit.distance < 100.0 ? 3u : (hit.distance < 1000.0 ? 4u : 5u));
    atomic_fetch_add_explicit(&counts[bin], 1u, memory_order_relaxed);
    float3 pos = r.origin + r.direction * hit.distance;
    ray s;
    s.origin = pos - r.direction * 0.02;
    s.direction = p.sun;
    s.min_distance = 0.02;
    s.max_distance = 100000.0;
    isect.accept_any_intersection(true);
    intersection_result<triangle_data, instancing> shadow = isect.intersect(s, accel, 0xFF);
    if (shadow.type != intersection_type::none) atomic_fetch_add_explicit(&counts[1], 1u, memory_order_relaxed);
}
"""

private struct ProbeParams {
    var forward: SIMD3<Float>; var tanX: Float
    var right: SIMD3<Float>; var tanY: Float
    var up: SIMD3<Float>; var pad: Float = 0
    var sun: SIMD3<Float>; var width: UInt32
    var height: UInt32
}

/// out: [triangles, nodes, BLAS build ms (all nodes), TLAS build ms, trace ms (best of 5), primary hit fraction,
/// shadowed fraction of hits, BLAS bytes]. Returns 0 on failure. Primary hits are also binned by distance in the
/// counts buffer (under 1, 100, 1000 blocks, farther): a camera inside geometry hits everything within a few blocks.
///
/// First results (M3 Pro, the 4 km test world, 26.2 M triangles in 129 nodes, 360 degrees): primary + sun shadow
/// ray per pixel 1.9-2.9 ms at 1728 x 1117 and 6.4-7.4 ms at 3456 x 2234; instance structure 0.11 ms; node
/// structures about 2.3 ms each to build; 2.4 GB of acceleration structures before compaction.
@_cdecl("mmc_debug_rt_probe")
public func mmc_debug_rt_probe(_ camX: Double, _ camY: Double, _ camZ: Double, _ yawDeg: Float, _ pitchDeg: Float,
                               _ width: Int32, _ height: Int32, _ out: UnsafeMutablePointer<Double>) -> Int32 {
    let dev = ctx.device
    guard dev.supportsRaytracing else { log("rt probe: no ray tracing on this device"); return 0 }
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return 0 }
    let snap = w.snapshot()
    let chosen = LodRenderer.select(snap.meshes, maxLevel: w.maxLevel, camX: camX, camZ: camZ, splitFactor: lodSplitFactor,
                                    level0Radius: Double(lodLevel0Radius))
    let corners: [[SIMD3<Float>]] = [
        [SIMD3(1, 0, 0), SIMD3(1, 1, 0), SIMD3(1, 1, 1), SIMD3(1, 0, 1)],
        [SIMD3(0, 0, 1), SIMD3(0, 1, 1), SIMD3(0, 1, 0), SIMD3(0, 0, 0)],
        [SIMD3(0, 1, 0), SIMD3(0, 1, 1), SIMD3(1, 1, 1), SIMD3(1, 1, 0)],
        [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 0, 1), SIMD3(0, 0, 1)],
        [SIMD3(1, 0, 1), SIMD3(1, 1, 1), SIMD3(0, 1, 1), SIMD3(0, 0, 1)],
        [SIMD3(0, 0, 0), SIMD3(0, 1, 0), SIMD3(1, 1, 0), SIMD3(1, 0, 0)],
    ]
    guard let queue = dev.makeCommandQueue() else { return 0 }
    var blases: [MTLAccelerationStructure] = []
    var triangles = 0, blasBytes = 0
    var blasMs = 0.0
    for (n, _) in chosen where n.quadCount > 0 {
        let q = n.buffer.contents().bindMemory(to: UInt32.self, capacity: 2 * n.quadCount)
        let scale = Float(1 << n.level)
        let origin = SIMD3(Float(Double(n.x0) - camX), Float(Double(lodWorldMinY) - camY), Float(Double(n.z0) - camZ))
        var verts = [Float](); verts.reserveCapacity(n.quadCount * 12)
        var idx = [UInt32](); idx.reserveCapacity(n.quadCount * 6)
        for i in 0..<n.quadCount {
            let w0 = q[2 * i], w1 = q[2 * i + 1]
            let face = Int((w0 >> 25) & 7)
            if face > 5 { continue }
            let local = SIMD3(Float(w0 & 255), Float((w0 >> 16) & 511), Float((w0 >> 8) & 255))
            let qw = Float(((w1 >> 8) & 255) + 1), qh = Float(((w1 >> 16) & 255) + 1)
            let ext: SIMD3<Float> = face < 2 ? SIMD3(1, qw, qh) : (face < 4 ? SIMD3(qw, 1, qh) : SIMD3(qw, qh, 1))
            let base = UInt32(verts.count / 3)
            for c in corners[face] {
                let p = origin + (local + c * ext) * scale
                verts += [p.x, p.y, p.z]
            }
            idx += [base, base + 1, base + 2, base, base + 2, base + 3]
        }
        guard !idx.isEmpty,
              let vb = dev.makeBuffer(bytes: verts, length: verts.count * 4, options: .storageModeShared),
              let ib = dev.makeBuffer(bytes: idx, length: idx.count * 4, options: .storageModeShared) else { continue }
        let g = MTLAccelerationStructureTriangleGeometryDescriptor()
        g.vertexBuffer = vb
        g.vertexStride = 12
        g.vertexFormat = .float3
        g.indexBuffer = ib
        g.indexType = .uint32
        g.triangleCount = idx.count / 3
        g.opaque = true
        let d = MTLPrimitiveAccelerationStructureDescriptor()
        d.geometryDescriptors = [g]
        let sizes = dev.accelerationStructureSizes(descriptor: d)
        guard let accel = dev.makeAccelerationStructure(size: sizes.accelerationStructureSize),
              let scratch = dev.makeBuffer(length: max(sizes.buildScratchBufferSize, 16), options: .storageModePrivate),
              let cb = queue.makeCommandBuffer(), let enc = cb.makeAccelerationStructureCommandEncoder() else { continue }
        enc.build(accelerationStructure: accel, descriptor: d, scratchBuffer: scratch, scratchBufferOffset: 0)
        guard let sizeBuf = dev.makeBuffer(length: 8, options: .storageModeShared) else { continue }
        enc.writeCompactedSize(accelerationStructure: accel, buffer: sizeBuf, offset: 0, sizeDataType: .ulong)
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        blasMs += (cb.gpuEndTime - cb.gpuStartTime) * 1000
        // Compacted copy (METALMC_EXP=nortcompact keeps the uncompacted one).
        var final = accel
        let compacted = Int(sizeBuf.contents().load(as: UInt64.self))
        if !experiments.contains("nortcompact"), compacted > 0, let small = dev.makeAccelerationStructure(size: compacted),
           let ccb = queue.makeCommandBuffer(), let cenc = ccb.makeAccelerationStructureCommandEncoder() {
            cenc.copyAndCompact(sourceAccelerationStructure: accel, destinationAccelerationStructure: small)
            cenc.endEncoding()
            ccb.commit(); ccb.waitUntilCompleted()
            blasMs += (ccb.gpuEndTime - ccb.gpuStartTime) * 1000
            final = small
        }
        blases.append(final)
        triangles += idx.count / 3
        blasBytes += final.size
    }
    guard !blases.isEmpty else { return 0 }
    // Instance structure: identity transforms (the vertices are already camera-relative).
    var instances = [MTLAccelerationStructureInstanceDescriptor](repeating: MTLAccelerationStructureInstanceDescriptor(), count: blases.count)
    for i in 0..<blases.count {
        instances[i].accelerationStructureIndex = UInt32(i)
        instances[i].mask = 0xFF
        instances[i].options = .opaque
        instances[i].transformationMatrix = MTLPackedFloat4x3(columns: (MTLPackedFloat3Make(1, 0, 0), MTLPackedFloat3Make(0, 1, 0),
                                                                        MTLPackedFloat3Make(0, 0, 1), MTLPackedFloat3Make(0, 0, 0)))
    }
    guard let instBuf = dev.makeBuffer(bytes: instances, length: MemoryLayout<MTLAccelerationStructureInstanceDescriptor>.stride * instances.count,
                                       options: .storageModeShared) else { return 0 }
    let td = MTLInstanceAccelerationStructureDescriptor()
    td.instancedAccelerationStructures = blases
    td.instanceCount = blases.count
    td.instanceDescriptorBuffer = instBuf
    let tsizes = dev.accelerationStructureSizes(descriptor: td)
    guard let tlas = dev.makeAccelerationStructure(size: tsizes.accelerationStructureSize),
          let tscratch = dev.makeBuffer(length: max(tsizes.buildScratchBufferSize, 16), options: .storageModePrivate),
          let tcb = queue.makeCommandBuffer(), let tenc = tcb.makeAccelerationStructureCommandEncoder() else { return 0 }
    tenc.build(accelerationStructure: tlas, descriptor: td, scratchBuffer: tscratch, scratchBufferOffset: 0)
    tenc.endEncoding()
    tcb.commit(); tcb.waitUntilCompleted()
    let tlasMs = (tcb.gpuEndTime - tcb.gpuStartTime) * 1000

    let pipe: MTLComputePipelineState
    do {
        let lib = try dev.makeLibrary(source: rtProbeSource, options: nil)
        pipe = try dev.makeComputePipelineState(function: lib.makeFunction(name: "rt_probe")!)
    } catch { log("rt probe: shader failed: \(error)"); return 0 }
    let yaw = yawDeg * .pi / 180, pitch = pitchDeg * .pi / 180
    // Minecraft: yaw 0 looks south (+z), 90 west (-x), 180 north (-z); positive pitch looks down.
    let forward = normalize(SIMD3(-sin(yaw) * cos(pitch), -sin(pitch), cos(yaw) * cos(pitch)))
    let right = normalize(cross(forward, SIMD3<Float>(0, 1, 0)))
    let up = cross(right, forward)
    let tanY = tan(Float(35) * .pi / 180), tanX = tanY * Float(width) / Float(height)
    var params = ProbeParams(forward: forward, tanX: tanX, right: right, tanY: tanY, up: up,
                             sun: normalize(SIMD3<Float>(0.35, 0.8, 0.25)), width: UInt32(width), height: UInt32(height))
    guard let counts = dev.makeBuffer(length: 24, options: .storageModeShared) else { return 0 }
    var best = Double.infinity
    for _ in 0..<5 {
        memset(counts.contents(), 0, 24)
        guard let cb = queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return 0 }
        enc.setComputePipelineState(pipe)
        enc.setAccelerationStructure(tlas, bufferIndex: 0)
        for b in blases { enc.useResource(b, usage: .read) }
        enc.setBytes(&params, length: MemoryLayout<ProbeParams>.stride, index: 1)
        enc.setBuffer(counts, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: Int(width), height: Int(height), depth: 1), threadsPerThreadgroup: MTLSize(width: 8, height: 8, depth: 1))
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        best = min(best, (cb.gpuEndTime - cb.gpuStartTime) * 1000)
    }
    let c = counts.contents().bindMemory(to: UInt32.self, capacity: 6)
    let pixels = Double(width) * Double(height)
    out[0] = Double(triangles); out[1] = Double(blases.count); out[2] = blasMs; out[3] = tlasMs; out[4] = best
    out[5] = Double(c[0]) / pixels; out[6] = c[0] > 0 ? Double(c[1]) / Double(c[0]) : 0; out[7] = Double(blasBytes)
    return 1
}
