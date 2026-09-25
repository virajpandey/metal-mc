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
    var opaqueIndexStart: Int
    var opaqueIndexCount: Int
    var opaqueBaseVertex: Int
    var waterIndexStart: Int
    var waterIndexCount: Int
    var waterBaseVertex: Int
}

struct EncodeStats {
    var drawn = 0
    var culled = 0
    var triangles = 0
}

enum RendererError: Error {
    case noCommandQueue
    case missingFunction(String)
}

/// Milestone 0 renderer: CPU frustum + fog-distance culling, one indexed draw per visible section.
/// GPU-driven culling (object/mesh shaders) replaces this loop in milestone 2.
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

    /// Backface culling for opaque geometry. Water stays double-sided.
    var cullBackfaces = false
    var frontFacing: MTLWinding = .counterClockwise
    /// Reverse-Z with an infinite far plane (see `perspectiveReverseZ`).
    var reverseZ = false

    func projection(aspect: Float) -> float4x4 {
        reverseZ
            ? perspectiveReverseZ(fovyRadians: 70 * .pi / 180, aspect: aspect, near: 0.1)
            : perspectiveRH(fovyRadians: 70 * .pi / 180, aspect: aspect, near: 0.1, far: 1000)
    }

    var clearDepth: Double { reverseZ ? 0 : 1 }

    private var vertexBuffer: MTLBuffer?
    private var indexBuffer: MTLBuffer?
    private(set) var sections: [SectionDraw] = []
    private(set) var totalQuads = 0
    private(set) var gpuBytes = 0

    var sky = SIMD4<Float>(0.62, 0.78, 0.95, 1)
    var fogStart: Float = 300
    var fogEnd: Float = 480

    init(device: MTLDevice) throws {
        self.device = device
        guard let q = device.makeCommandQueue() else { throw RendererError.noCommandQueue }
        queue = q

        let lib = try device.makeLibrary(source: shaderSource, options: nil)
        guard let vfn = lib.makeFunction(name: "vmain") else { throw RendererError.missingFunction("vmain") }
        guard let ffn = lib.makeFunction(name: "fmain") else { throw RendererError.missingFunction("fmain") }

        let desc = MTLRenderPipelineDescriptor()
        desc.vertexFunction = vfn
        desc.fragmentFunction = ffn
        desc.colorAttachments[0].pixelFormat = colorFormat
        desc.depthAttachmentPixelFormat = depthFormat
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
    }

    /// Meshes every 16^3 section in parallel and packs the results into one vertex and one index buffer.
    func upload(world: World) {
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

        var verts: [PackedVertex] = []
        var idx: [UInt32] = []
        sections = []
        for case let m? in meshes where !(m.opaqueIdx.isEmpty && m.waterIdx.isEmpty) {
            var d = SectionDraw(minB: m.minB, maxB: m.maxB,
                                opaqueIndexStart: idx.count, opaqueIndexCount: m.opaqueIdx.count, opaqueBaseVertex: verts.count,
                                waterIndexStart: 0, waterIndexCount: 0, waterBaseVertex: 0)
            verts.append(contentsOf: m.opaqueVerts)
            idx.append(contentsOf: m.opaqueIdx)
            d.waterIndexStart = idx.count
            d.waterIndexCount = m.waterIdx.count
            d.waterBaseVertex = verts.count
            verts.append(contentsOf: m.waterVerts)
            idx.append(contentsOf: m.waterIdx)
            sections.append(d)
        }

        totalQuads = idx.count / 6
        let vBytes = verts.count * MemoryLayout<PackedVertex>.stride
        let iBytes = idx.count * MemoryLayout<UInt32>.stride
        vertexBuffer = vBytes > 0
            ? device.makeBuffer(bytes: verts, length: vBytes, options: .storageModeShared)
            : device.makeBuffer(length: 16, options: .storageModeShared)
        indexBuffer = iBytes > 0
            ? device.makeBuffer(bytes: idx, length: iBytes, options: .storageModeShared)
            : device.makeBuffer(length: 16, options: .storageModeShared)
        gpuBytes = vBytes + iBytes
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
        var stats = EncodeStats()
        guard let vb = vertexBuffer, let ib = indexBuffer,
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
        enc.setVertexBuffer(vb, offset: 0, index: 0)
        enc.setVertexBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)
        enc.setFragmentBytes(&u, length: MemoryLayout<Uniforms>.stride, index: 1)

        enc.setRenderPipelineState(opaquePipeline)
        enc.setDepthStencilState(reverseZ ? depthWriteGreater : depthWriteLess)
        for i in visible {
            let s = sections[i]
            guard s.opaqueIndexCount > 0 else { continue }
            enc.drawIndexedPrimitives(type: .triangle, indexCount: s.opaqueIndexCount, indexType: .uint32,
                                      indexBuffer: ib, indexBufferOffset: s.opaqueIndexStart * 4,
                                      instanceCount: 1, baseVertex: s.opaqueBaseVertex, baseInstance: 0)
            stats.triangles += s.opaqueIndexCount / 3
        }

        enc.setRenderPipelineState(waterPipeline)
        enc.setDepthStencilState(reverseZ ? depthNoWriteGreater : depthNoWriteLess)
        enc.setCullMode(.none)
        for i in visible {
            let s = sections[i]
            guard s.waterIndexCount > 0 else { continue }
            enc.drawIndexedPrimitives(type: .triangle, indexCount: s.waterIndexCount, indexType: .uint32,
                                      indexBuffer: ib, indexBufferOffset: s.waterIndexStart * 4,
                                      instanceCount: 1, baseVertex: s.waterBaseVertex, baseInstance: 0)
            stats.triangles += s.waterIndexCount / 3
        }

        enc.endEncoding()
        return stats
    }
}
