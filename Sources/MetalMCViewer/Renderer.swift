import Foundation
import Metal
import simd

/// Must match `Uniforms` in the shader: 64 + 16 + 16 + 16 = 112 bytes.
struct Uniforms {
    var viewProj: float4x4
    var cameraPos: SIMD4<Float>
    var fog: SIMD4<Float>
    var sky: SIMD4<Float>
}

struct SectionDraw {
    var minB: SIMD3<Float>
    var maxB: SIMD3<Float>
    var opaqueStart: Int
    var opaqueCount: Int
    var faceOffsets: SIMD8<Int32>
    var waterStart: Int
    var waterCount: Int
}

/// Must match `CullUniforms` in the shader (6 x 16 + 16 + 16 = 128 bytes).
struct CullUniforms {
    var planes: (SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>, SIMD4<Float>)
    var cameraPos: SIMD4<Float>
    var fogEnd: Float
    var sectionCount: UInt32
    var waterBase: UInt32
    var bucketsOn: UInt32
}

/// Must match `SectionGPU` in the shader (80 bytes). A tuple keeps faceOffsets 4-byte aligned.
struct SectionGPU {
    var minB: SIMD4<Float>
    var maxB: SIMD4<Float>
    var opaqueStart: UInt32
    var waterStart: UInt32
    var waterCount: UInt32
    var pad: UInt32 = 0
    var faceOffsets: (Int32, Int32, Int32, Int32, Int32, Int32, Int32, Int32)
}

struct EncodeStats {
    var drawn = 0
    var culled = 0
    var triangles = 0
}

enum RendererError: Error {
    case noCommandQueue
    case missingFunction(String)
    case allocation(String)
}

/// Near-terrain renderer. Quads are 4-byte words expanded by the vertex shader; each visible
/// section is one non-indexed draw whose `baseInstance` selects its origin.
/// CPU frustum + fog-distance culling for now; GPU-driven culling comes later in milestone 2.
final class Renderer {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let colorFormat: MTLPixelFormat = .bgra8Unorm
    let depthFormat: MTLPixelFormat = .depth32Float

    private let opaquePipeline: MTLRenderPipelineState
    private let waterPipeline: MTLRenderPipelineState
    private let depthWriteLess: MTLDepthStencilState
    private let depthNoWriteLess: MTLDepthStencilState
    private let depthWriteGreater: MTLDepthStencilState
    private let depthNoWriteGreater: MTLDepthStencilState

    private var quadBuffer: MTLBuffer?
    private var originBuffer: MTLBuffer?
    /// Shared index pattern 4q+{0,1,2,0,2,3}, sized for the largest section.
    private var quadIndexBuffer: MTLBuffer?
    private let colorBuffer: MTLBuffer
    private(set) var sections: [SectionDraw] = []
    private(set) var totalQuads = 0
    private(set) var gpuBytes = 0

    var sky = SIMD4<Float>(0.62, 0.78, 0.95, 1)
    var fogStart: Float = 300
    var fogEnd: Float = 480

    /// Backface culling for opaque geometry. Water stays double-sided.
    var cullBackfaces = true
    var frontFacing: MTLWinding = .counterClockwise
    /// Reverse-Z with an infinite far plane (see `perspectiveReverseZ`).
    var reverseZ = true
    /// Skip whole face-direction buckets that cannot face the camera (before any vertex shading).
    var faceBuckets = true
    /// Cull sections and build draw commands on the GPU (compute kernel -> indirect command buffer).
    var gpuCulling = false

    private let cullPipeline: MTLComputePipelineState
    private let icbArgEncoder: MTLArgumentEncoder
    private var icb: MTLIndirectCommandBuffer?
    private var icbArgBuffer: MTLBuffer?
    private var sectionBuffer: MTLBuffer?
    private let statsBuffer: MTLBuffer

    /// Sections drawn and triangles submitted by the last GPU-culled frame (valid after it completes).
    func readGPUStats() -> (drawn: Int, triangles: Int) {
        let p = statsBuffer.contents().assumingMemoryBound(to: UInt32.self)
        return (Int(p[0]), Int(p[1]))
    }

    func projection(aspect: Float) -> float4x4 {
        reverseZ
            ? perspectiveReverseZ(fovyRadians: 70 * .pi / 180, aspect: aspect, near: 0.1)
            : perspectiveRH(fovyRadians: 70 * .pi / 180, aspect: aspect, near: 0.1, far: 1000)
    }

    var clearDepth: Double { reverseZ ? 0 : 1 }

    init(device: MTLDevice) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else { throw RendererError.noCommandQueue }
        queue = q

        let lib = try device.makeLibrary(source: shaderSource, options: nil)
        guard let vfn = lib.makeFunction(name: "vquad") else { throw RendererError.missingFunction("vquad") }
        guard let ffn = lib.makeFunction(name: "fmain") else { throw RendererError.missingFunction("fmain") }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = colorFormat
        desc.depthAttachmentPixelFormat = depthFormat
        desc.supportIndirectCommandBuffers = true
        opaquePipeline = try device.makeRenderPipelineState(descriptor: desc)

        let blend = desc.colorAttachments[0]!
        blend.isBlendingEnabled = true
        blend.sourceRGBBlendFactor = .sourceAlpha
        blend.destinationRGBBlendFactor = .oneMinusSourceAlpha
        blend.sourceAlphaBlendFactor = .one
        blend.destinationAlphaBlendFactor = .oneMinusSourceAlpha
        waterPipeline = try device.makeRenderPipelineState(descriptor: desc)

        let dd = MTLDepthStencilDescriptor()
        dd.depthCompareFunction = .less
        dd.isDepthWriteEnabled = true
        depthWriteLess = device.makeDepthStencilState(descriptor: dd)!
        dd.isDepthWriteEnabled = false
        depthNoWriteLess = device.makeDepthStencilState(descriptor: dd)!
        dd.depthCompareFunction = .greater
        dd.isDepthWriteEnabled = true
        depthWriteGreater = device.makeDepthStencilState(descriptor: dd)!
        dd.isDepthWriteEnabled = false
        depthNoWriteGreater = device.makeDepthStencilState(descriptor: dd)!

        let colors = Mesher.faceColors
        guard let cb = device.makeBuffer(bytes: colors, length: colors.count * 4, options: .storageModeShared)
        else { throw RendererError.allocation("colors") }
        colorBuffer = cb

        guard let cfn = lib.makeFunction(name: "cullSections") else { throw RendererError.missingFunction("cullSections") }
        cullPipeline = try device.makeComputePipelineState(function: cfn)
        icbArgEncoder = cfn.makeArgumentEncoder(bufferIndex: 2)
        guard let sb = device.makeBuffer(length: 16, options: .storageModeShared) else { throw RendererError.allocation("stats") }
        statsBuffer = sb
    }

    /// Section counts per tier from the last upload: [near, LOD ring 1, LOD ring 2, ...].
    private(set) var sectionsPerTier: [Int] = []
    private(set) var quadsPerTier: [Int] = []

    private static func meshAll(_ world: World) -> [SectionMesh] {
        let cx = world.secX, cy = world.secY, cz = world.secZ
        let count = cx * cy * cz
        var meshes = [SectionMesh?](repeating: nil, count: count)
        world.withView { view in
            meshes.withUnsafeMutableBufferPointer { buf in
                let p = buf.baseAddress!
                DispatchQueue.concurrentPerform(iterations: count) { i in
                    let sx = i % cx, sy = (i / cx) % cy, sz = i / (cx * cy)
                    p[i] = Mesher.meshSection(view: view, sx: sx, sy: sy, sz: sz)
                }
            }
        }
        return meshes.compactMap { m in
            guard let m, !(m.opaque.isEmpty && m.water.isEmpty) else { return nil }
            return m
        }
    }

    /// Meshes the near world and any LOD rings, and packs all quads into one buffer. Each draw slot
    /// has a transform (world-space origin, per-axis cell scale) in a parallel buffer.
    func upload(world: World, lods: [LODLevel] = []) {
        var tiers: [(meshes: [SectionMesh], scale: SIMD3<Float>, offset: SIMD3<Float>)] =
            [(Renderer.meshAll(world), SIMD3(repeating: 1), .zero)]
        for lod in lods {
            tiers.append((Renderer.meshAll(lod.world),
                          SIMD3(Float(lod.scale), Float(lod.vScale), Float(lod.scale)),
                          SIMD3(Float(lod.originX), 0, Float(lod.originZ))))
        }

        var quads: [UInt32] = []
        var origins: [SIMD4<Float>] = []   // pairs: (origin, 0), (scale, 0)
        sections = []
        sectionsPerTier = []
        quadsPerTier = []
        for tier in tiers {
            let q0 = quads.count
            for m in tier.meshes {
                let minB = tier.offset + m.minB * tier.scale
                let maxB = tier.offset + m.maxB * tier.scale
                let d = SectionDraw(minB: minB, maxB: maxB,
                                    opaqueStart: quads.count, opaqueCount: m.opaque.count, faceOffsets: m.faceOffsets,
                                    waterStart: quads.count + m.opaque.count, waterCount: m.water.count)
                quads.append(contentsOf: m.opaque)
                quads.append(contentsOf: m.water)
                origins.append(SIMD4(minB, 0))
                origins.append(SIMD4(tier.scale, 0))
                sections.append(d)
            }
            sectionsPerTier.append(tier.meshes.count)
            quadsPerTier.append(quads.count - q0)
        }

        totalQuads = quads.count
        let qBytes = max(16, quads.count * 4)
        let oBytes = max(16, origins.count * MemoryLayout<SIMD4<Float>>.stride)
        quadBuffer = quads.isEmpty
            ? device.makeBuffer(length: qBytes, options: .storageModeShared)
            : device.makeBuffer(bytes: quads, length: quads.count * 4, options: .storageModeShared)
        originBuffer = origins.isEmpty
            ? device.makeBuffer(length: oBytes, options: .storageModeShared)
            : device.makeBuffer(bytes: origins, length: oBytes, options: .storageModeShared)
        let maxQuads = max(1, sections.map { max($0.opaqueCount, $0.waterCount) }.max() ?? 1)
        var pattern = [UInt32]()
        pattern.reserveCapacity(maxQuads * 6)
        for q in 0..<UInt32(maxQuads) {
            let b = q * 4
            pattern.append(contentsOf: [b, b + 1, b + 2, b, b + 2, b + 3])
        }
        quadIndexBuffer = device.makeBuffer(bytes: pattern, length: pattern.count * 4, options: .storageModeShared)
        gpuBytes = quads.count * 4 + origins.count * MemoryLayout<SIMD4<Float>>.stride + pattern.count * 4

        // GPU-driven path: per-section records plus an ICB with 3 opaque slots and 1 water slot per section.
        let gpuSections = sections.map { s -> SectionGPU in
            let o = s.faceOffsets
            return SectionGPU(minB: SIMD4(s.minB, 0), maxB: SIMD4(s.maxB, 0),
                              opaqueStart: UInt32(s.opaqueStart), waterStart: UInt32(s.waterStart),
                              waterCount: UInt32(s.waterCount),
                              faceOffsets: (o[0], o[1], o[2], o[3], o[4], o[5], o[6], o[7]))
        }
        if !gpuSections.isEmpty {
            sectionBuffer = device.makeBuffer(bytes: gpuSections,
                                              length: gpuSections.count * MemoryLayout<SectionGPU>.stride,
                                              options: .storageModeShared)
            let d = MTLIndirectCommandBufferDescriptor()
            d.commandTypes = .drawIndexed
            d.inheritBuffers = true
            d.inheritPipelineState = true
            icb = device.makeIndirectCommandBuffer(descriptor: d, maxCommandCount: gpuSections.count * 4, options: [])
            icbArgBuffer = device.makeBuffer(length: icbArgEncoder.encodedLength, options: .storageModeShared)
            if let icb, let ab = icbArgBuffer {
                icbArgEncoder.setArgumentBuffer(ab, offset: 0)
                icbArgEncoder.setIndirectCommandBuffer(icb, index: 0)
            }
            gpuBytes += gpuSections.count * MemoryLayout<SectionGPU>.stride
        }
    }

    func makePass(color: MTLTexture, depth: MTLTexture) -> MTLRenderPassDescriptor {
        let p = MTLRenderPassDescriptor()
        p.colorAttachments[0].texture = color
        p.colorAttachments[0].loadAction = .clear
        p.colorAttachments[0].storeAction = .store
        p.colorAttachments[0].clearColor = clearColor
        p.depthAttachment.texture = depth
        p.depthAttachment.loadAction = .clear
        p.depthAttachment.clearDepth = clearDepth
        p.depthAttachment.storeAction = .dontCare
        return p
    }

    var clearColor: MTLClearColor {
        MTLClearColor(red: Double(sky.x), green: Double(sky.y), blue: Double(sky.z), alpha: 1)
    }

    func encode(into cb: MTLCommandBuffer, pass: MTLRenderPassDescriptor,
                viewProj: float4x4, cameraPos: SIMD3<Float>) -> EncodeStats {
        if gpuCulling, icb != nil {
            return encodeGPUDriven(into: cb, pass: pass, viewProj: viewProj, cameraPos: cameraPos)
        }
        var stats = EncodeStats()
        guard let qb = quadBuffer, let ob = originBuffer, let ib = quadIndexBuffer,
              let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return stats }

        var u = Uniforms(viewProj: viewProj, cameraPos: SIMD4(cameraPos, 1),
                         fog: SIMD4(fogStart, fogEnd, 0, 0), sky: sky)
        let frustum = Frustum(viewProj: viewProj)

        var visible: [Int] = []
        visible.reserveCapacity(sections.count)
        for (i, s) in sections.enumerated() {
            if distanceToBox(cameraPos, min: s.minB, max: s.maxB) > fogEnd { continue }
            if !frustum.intersects(min: s.minB, max: s.maxB) { continue }
            visible.append(i)
        }
        stats.drawn = visible.count
        stats.culled = sections.count - visible.count

        enc.setFrontFacing(frontFacing)
        enc.setCullMode(cullBackfaces ? .back : .none)
        enc.setVertexBuffer(qb, offset: 0, index: 0)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setVertexBuffer(ob, offset: 0, index: 2)
        enc.setVertexBuffer(colorBuffer, offset: 0, index: 3)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)

        enc.setRenderPipelineState(opaquePipeline)
        enc.setDepthStencilState(reverseZ ? depthWriteGreater : depthWriteLess)
        let c = cameraPos
        for i in visible {
            let s = sections[i]
            guard s.opaqueCount > 0 else { continue }
            // A +X face lies on a plane at x >= minB.x + 1, so it can only face the camera if c.x > minB.x; etc.
            var mask: UInt8 = 0b111111
            if faceBuckets {
                mask = (c.x > s.minB.x ? 1 : 0) | (c.x < s.maxB.x ? 2 : 0) | (c.y > s.minB.y ? 4 : 0)
                    | (c.y < s.maxB.y ? 8 : 0) | (c.z > s.minB.z ? 16 : 0) | (c.z < s.maxB.z ? 32 : 0)
            }
            // Draw runs of consecutive visible buckets.
            var f = 0
            while f < 6 {
                guard mask & (1 << f) != 0 else { f += 1; continue }
                var e = f
                while e < 6 && mask & (1 << e) != 0 { e += 1 }
                let start = Int(s.faceOffsets[f]), end = Int(s.faceOffsets[e])
                if end > start {
                    enc.drawIndexedPrimitives(type: .triangle, indexCount: (end - start) * 6, indexType: .uint32,
                                              indexBuffer: ib, indexBufferOffset: 0, instanceCount: 1,
                                              baseVertex: (s.opaqueStart + start) * 4, baseInstance: i)
                    stats.triangles += (end - start) * 2
                }
                f = e
            }
        }

        enc.setRenderPipelineState(waterPipeline)
        enc.setDepthStencilState(reverseZ ? depthNoWriteGreater : depthNoWriteLess)
        enc.setCullMode(.none)
        for i in visible {
            let s = sections[i]
            guard s.waterCount > 0 else { continue }
            enc.drawIndexedPrimitives(type: .triangle, indexCount: s.waterCount * 6, indexType: .uint32,
                                      indexBuffer: ib, indexBufferOffset: 0, instanceCount: 1,
                                      baseVertex: s.waterStart * 4, baseInstance: i)
            stats.triangles += s.waterCount * 2
        }

        enc.endEncoding()
        return stats
    }

    /// GPU-driven frame: reset the ICB, let one compute thread per section cull it and write its draws,
    /// then execute the opaque and water ranges. The CPU records a constant handful of commands.
    private func encodeGPUDriven(into cb: MTLCommandBuffer, pass: MTLRenderPassDescriptor,
                                 viewProj: float4x4, cameraPos: SIMD3<Float>) -> EncodeStats {
        var stats = EncodeStats()
        guard let icb, let argBuf = icbArgBuffer, let secBuf = sectionBuffer,
              let qb = quadBuffer, let ob = originBuffer, let ib = quadIndexBuffer else { return stats }
        let n = sections.count
        let fr = Frustum(viewProj: viewProj).planes

        if let blit = cb.makeBlitCommandEncoder() {
            blit.resetCommandsInBuffer(icb, range: 0..<(n * 4))
            blit.fill(buffer: statsBuffer, range: 0..<8, value: 0)
            blit.endEncoding()
        }

        if let ce = cb.makeComputeCommandEncoder() {
            var cu = CullUniforms(planes: (fr[0], fr[1], fr[2], fr[3], fr[4], fr[5]),
                                  cameraPos: SIMD4(cameraPos, 1), fogEnd: fogEnd,
                                  sectionCount: UInt32(n), waterBase: UInt32(n * 3), bucketsOn: faceBuckets ? 1 : 0)
            ce.setComputePipelineState(cullPipeline)
            ce.setBytes(&cu, length: MemoryLayout<CullUniforms>.stride, index: 0)
            ce.setBuffer(secBuf, offset: 0, index: 1)
            ce.setBuffer(argBuf, offset: 0, index: 2)
            ce.setBuffer(ib, offset: 0, index: 3)
            ce.setBuffer(statsBuffer, offset: 0, index: 4)
            ce.useResource(icb, usage: .write)
            let tg = min(64, cullPipeline.maxTotalThreadsPerThreadgroup)
            ce.dispatchThreads(MTLSize(width: n, height: 1, depth: 1),
                               threadsPerThreadgroup: MTLSize(width: tg, height: 1, depth: 1))
            ce.endEncoding()
        }

        guard let enc = cb.makeRenderCommandEncoder(descriptor: pass) else { return stats }
        var u = Uniforms(viewProj: viewProj, cameraPos: SIMD4(cameraPos, 1),
                         fog: SIMD4(fogStart, fogEnd, 0, 0), sky: sky)
        enc.setFrontFacing(frontFacing)
        enc.setCullMode(cullBackfaces ? .back : .none)
        enc.setVertexBuffer(qb, offset: 0, index: 0)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setVertexBuffer(ob, offset: 0, index: 2)
        enc.setVertexBuffer(colorBuffer, offset: 0, index: 3)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.useResource(ib, usage: .read, stages: .vertex)

        enc.setRenderPipelineState(opaquePipeline)
        enc.setDepthStencilState(reverseZ ? depthWriteGreater : depthWriteLess)
        enc.executeCommandsInBuffer(icb, range: 0..<(n * 3))

        enc.setRenderPipelineState(waterPipeline)
        enc.setDepthStencilState(reverseZ ? depthNoWriteGreater : depthNoWriteLess)
        enc.setCullMode(.none)
        enc.executeCommandsInBuffer(icb, range: (n * 3)..<(n * 4))
        enc.endEncoding()

        stats.drawn = -1   // filled in from readGPUStats() after the frame completes
        return stats
    }
}
