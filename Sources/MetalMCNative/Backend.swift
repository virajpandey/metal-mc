import Foundation
import Metal
import QuartzCore

// The Metal half of the Minecraft backend (Java side: mod/src/client/java/metalmc/backend).
//
// Objects cross the C boundary as retained opaque pointers (`Unmanaged`), so the hot paths
// (draws, binds) cost one pointer dereference, not a registry lookup. Java releases them with
// mmc_handle_release after its own deferred-destruction queue says the GPU is done with them.
//
// Conventions shared with Java (see docs/backend-design.md):
// - Memory layout matches OpenGL: row 0 of every texture is the GL "bottom" row. Vertex shaders are
//   translated with SPIRV-Cross flip_vert_y, front faces are clockwise, and the present blit flips.
// - Uniform i of a pipeline is Metal buffer/texture/sampler index i. Push constants are buffer 24.
//   Vertex buffer slot s is buffer 30 - s.

let pushConstantsIndex = 24
@inline(__always) func vertexBufferIndex(_ slot: Int) -> Int { 30 - slot }

// MARK: - Handles

@inline(__always) func makeHandle(_ o: AnyObject) -> Int64 {
    Int64(Int(bitPattern: Unmanaged.passRetained(o).toOpaque()))
}

@inline(__always) func from<T: AnyObject>(_ h: Int64) -> T {
    Unmanaged<T>.fromOpaque(UnsafeRawPointer(bitPattern: Int(truncatingIfNeeded: h))!).takeUnretainedValue()
}

@_cdecl("mmc_handle_release")
public func mmc_handle_release(_ h: Int64) {
    guard let p = UnsafeRawPointer(bitPattern: Int(truncatingIfNeeded: h)) else { return }
    Unmanaged<AnyObject>.fromOpaque(p).release()
}

func log(_ s: String) {
    FileHandle.standardError.write(("[metalmc-native] " + s + "\n").data(using: .utf8)!)
}

final class BufferBox {
    let buffer: MTLBuffer
    var texelViews: [TexelKey: MTLTexture] = [:]
    init(_ b: MTLBuffer) { buffer = b }
}

struct TexelKey: Hashable {
    let offset: Int
    let length: Int
    let format: UInt
}

final class TextureBox {
    let texture: MTLTexture
    init(_ t: MTLTexture) { texture = t }
}

final class SamplerBox {
    let state: MTLSamplerState
    init(_ s: MTLSamplerState) { state = s }
}

// MARK: - Context (device, queue, the one command stream)

final class MetalContext: @unchecked Sendable {
    static let shared = MetalContext()

    let device: MTLDevice
    let queue: MTLCommandQueue

    var cb: MTLCommandBuffer?
    var blit: MTLBlitCommandEncoder?
    var pass: MTLRenderCommandEncoder?

    // Render pass state.
    var passDepthFormat: MTLPixelFormat = .invalid
    var passWidth = 0
    var passHeight = 0
    var pipe: PipelineBox?
    var boundPipeState: MTLRenderPipelineState?
    // Last object/offset bound per index and stage (0 = vertex, 1 = fragment), to skip redundant sets.
    var boundBuffers = [[ObjectIdentifier?]](repeating: [ObjectIdentifier?](repeating: nil, count: 31), count: 2)
    var boundOffsets = [[Int]](repeating: [Int](repeating: -1, count: 31), count: 2)
    var boundTextures = [[ObjectIdentifier?]](repeating: [ObjectIdentifier?](repeating: nil, count: 32), count: 2)
    var boundSamplers = [[ObjectIdentifier?]](repeating: [ObjectIdentifier?](repeating: nil, count: 16), count: 2)
    var indexBuffer: MTLBuffer?
    var indexType: MTLIndexType = .uint16
    var indexSize = 2
    var scissorEmpty = false
    var lastScissor: MTLScissorRect?

    var pendingDrawables: [CAMetalDrawable] = []

    // Submit completion, for fences and frame pacing.
    let cond = NSCondition()
    var completed: Int64 = 1
    var gpuErrorsLogged = 0
    // GPU time per submit (command buffer gpuStartTime..gpuEndTime), for the benchmark.
    var gpuSeconds: [Double] = []

    // Utility pipelines, built on first use.
    let utilLock = NSLock()
    var blitPipeline: MTLRenderPipelineState?
    var clearPipelines: [UInt64: MTLRenderPipelineState] = [:]
    var copyBytesPipeline: MTLComputePipelineState?
    var depthStates: [Int: MTLDepthStencilState] = [:]

    init() {
        device = MTLCreateSystemDefaultDevice()!
        queue = device.makeCommandQueue(maxCommandBufferCount: 64)!
        queue.label = "MetalMC"
    }

    func ensureCB() -> MTLCommandBuffer {
        if let cb { return cb }
        let made = queue.makeCommandBuffer()!
        cb = made
        return made
    }

    func endBlit() {
        blit?.endEncoding()
        blit = nil
    }

    func blitEncoder() -> MTLBlitCommandEncoder {
        if let blit { return blit }
        precondition(pass == nil, "blit while a render pass is open")
        let made = ensureCB().makeBlitCommandEncoder()!
        blit = made
        return made
    }

    func resetBindings() {
        boundPipeState = nil
        lastScissor = nil
        for st in 0..<2 {
            for i in 0..<31 { boundBuffers[st][i] = nil; boundOffsets[st][i] = -1 }
            for i in 0..<32 { boundTextures[st][i] = nil }
            for i in 0..<16 { boundSamplers[st][i] = nil }
        }
    }

    /// Binds a buffer to a stage unless the same buffer/offset is already there. Same buffer with a new
    /// offset uses the cheaper offset-only setter.
    @inline(__always) func bindBuffer(_ enc: MTLRenderCommandEncoder, _ stage: Int, _ b: MTLBuffer, _ offset: Int, _ index: Int) {
        let id = ObjectIdentifier(b)
        if boundBuffers[stage][index] == id {
            if boundOffsets[stage][index] == offset { return }
            if stage == 0 { enc.setVertexBufferOffset(offset, index: index) } else { enc.setFragmentBufferOffset(offset, index: index) }
        } else {
            if stage == 0 { enc.setVertexBuffer(b, offset: offset, index: index) } else { enc.setFragmentBuffer(b, offset: offset, index: index) }
            boundBuffers[stage][index] = id
        }
        boundOffsets[stage][index] = offset
    }

    @inline(__always) func bindTexture(_ enc: MTLRenderCommandEncoder, _ stage: Int, _ t: MTLTexture, _ index: Int) {
        let id = ObjectIdentifier(t)
        if boundTextures[stage][index] == id { return }
        if stage == 0 { enc.setVertexTexture(t, index: index) } else { enc.setFragmentTexture(t, index: index) }
        boundTextures[stage][index] = id
    }

    @inline(__always) func bindSampler(_ enc: MTLRenderCommandEncoder, _ stage: Int, _ s: MTLSamplerState, _ index: Int) {
        let id = ObjectIdentifier(s)
        if boundSamplers[stage][index] == id { return }
        if stage == 0 { enc.setVertexSamplerState(s, index: index) } else { enc.setFragmentSamplerState(s, index: index) }
        boundSamplers[stage][index] = id
    }

    func depthState(compare: MTLCompareFunction, write: Bool) -> MTLDepthStencilState {
        let key = Int(compare.rawValue) * 2 + (write ? 1 : 0)
        utilLock.lock(); defer { utilLock.unlock() }
        if let s = depthStates[key] { return s }
        let d = MTLDepthStencilDescriptor()
        d.depthCompareFunction = compare
        d.isDepthWriteEnabled = write
        let s = device.makeDepthStencilState(descriptor: d)!
        depthStates[key] = s
        return s
    }
}

let ctx = MetalContext.shared

@_cdecl("mmc_ctx_init")
public func mmc_ctx_init() -> Int32 {
    _ = ctx.device
    // Guard against mismatched raw enum values on the Java side.
    guard MTLPixelFormat.rgba8Unorm.rawValue == 70, MTLPixelFormat.depth32Float.rawValue == 252,
          MTLVertexFormat.float3.rawValue == 30, MTLVertexFormat.uchar4Normalized.rawValue == 9,
          MTLVertexFormat.half.rawValue == 53, MTLBlendFactor.oneMinusSourceAlpha.rawValue == 5,
          MTLCompareFunction.greaterEqual.rawValue == 6 else { return 0 }
    return 1
}

/// out[0] = maxBufferLength, out[1] = recommendedMaxWorkingSetSize, out[2] = apple9 (0/1).
@_cdecl("mmc_ctx_limits")
public func mmc_ctx_limits(_ out: UnsafeMutablePointer<Int64>) {
    out[0] = Int64(ctx.device.maxBufferLength)
    out[1] = Int64(ctx.device.recommendedMaxWorkingSetSize)
    out[2] = ctx.device.supportsFamily(.apple9) ? 1 : 0
}

// MARK: - Buffers

@_cdecl("mmc_buffer_create")
public func mmc_buffer_create(_ size: Int64) -> Int64 {
    // Minecraft sizes uniform buffers to the std140 byte count (e.g. 40), but MSL rounds structs up to
    // their alignment (48), so shaders may read up to 15 bytes past a slice that ends the buffer. Pad.
    let length = (max(Int(size), 16) + 15) / 16 * 16 + 64
    guard let b = ctx.device.makeBuffer(length: length, options: [.storageModeShared]) else { return 0 }
    return makeHandle(BufferBox(b))
}

@_cdecl("mmc_buffer_contents")
public func mmc_buffer_contents(_ h: Int64) -> Int64 {
    let box: BufferBox = from(h)
    return Int64(Int(bitPattern: box.buffer.contents()))
}

@_cdecl("mmc_buffer_label")
public func mmc_buffer_label(_ h: Int64, _ label: UnsafePointer<CChar>) {
    let box: BufferBox = from(h)
    box.buffer.label = String(cString: label)
}

// MARK: - Textures and samplers

/// flags: 1 = render target, 2 = cube map.
@_cdecl("mmc_texture_create")
public func mmc_texture_create(_ format: Int32, _ width: Int32, _ height: Int32, _ layers: Int32, _ mips: Int32, _ flags: Int32) -> Int64 {
    guard let pf = MTLPixelFormat(rawValue: UInt(format)), pf != .invalid else { return 0 }
    let d = MTLTextureDescriptor()
    if flags & 2 != 0 {
        d.textureType = .typeCube
    } else if layers > 1 {
        d.textureType = .type2DArray
        d.arrayLength = Int(layers)
    } else {
        d.textureType = .type2D
    }
    d.pixelFormat = pf
    d.width = max(1, Int(width))
    d.height = max(1, Int(height))
    d.mipmapLevelCount = max(1, Int(mips))
    d.storageMode = .private
    var usage: MTLTextureUsage = [.shaderRead]
    if flags & 1 != 0 { usage.insert(.renderTarget) }
    d.usage = usage
    guard let t = ctx.device.makeTexture(descriptor: d) else { return 0 }
    return makeHandle(TextureBox(t))
}

@_cdecl("mmc_texture_label")
public func mmc_texture_label(_ h: Int64, _ label: UnsafePointer<CChar>) {
    let box: TextureBox = from(h)
    box.texture.label = String(cString: label)
}

@_cdecl("mmc_texture_view")
public func mmc_texture_view(_ h: Int64, _ baseMip: Int32, _ mipCount: Int32) -> Int64 {
    let t = (from(h) as TextureBox).texture
    let slices = t.textureType == .typeCube ? 6 : t.arrayLength
    let levels = Int(baseMip)..<Int(baseMip + max(1, mipCount))
    guard let v = t.makeTextureView(pixelFormat: t.pixelFormat, textureType: t.textureType, levels: levels, slices: 0..<slices) else { return 0 }
    return makeHandle(TextureBox(v))
}

/// address: 0 = repeat, 1 = clamp to edge. filter: 0 = nearest, 1 = linear. maxLod < 0 = unset.
@_cdecl("mmc_sampler_create")
public func mmc_sampler_create(_ addrU: Int32, _ addrV: Int32, _ minF: Int32, _ magF: Int32, _ aniso: Int32, _ maxLod: Float) -> Int64 {
    let d = MTLSamplerDescriptor()
    d.sAddressMode = addrU == 0 ? .repeat : .clampToEdge
    d.tAddressMode = addrV == 0 ? .repeat : .clampToEdge
    d.rAddressMode = .clampToEdge
    d.minFilter = minF == 0 ? .nearest : .linear
    d.magFilter = magF == 0 ? .nearest : .linear
    // Same rule as the Vulkan backend: a max LOD of 0.25 or less means "base level only".
    let lod = maxLod < 0 ? 1000 : maxLod
    d.mipFilter = lod > 0.25 ? .linear : .nearest
    d.lodMaxClamp = max(0.25, lod)
    d.maxAnisotropy = max(1, min(16, Int(aniso)))
    guard let s = ctx.device.makeSamplerState(descriptor: d) else { return 0 }
    return makeHandle(SamplerBox(s))
}

// MARK: - Pipelines

final class PipelineBox {
    let base: MTLRenderPipelineDescriptor
    let depthState: MTLDepthStencilState
    let cull: MTLCullMode
    let fill: MTLTriangleFillMode
    let prim: MTLPrimitiveType
    let fan: Bool
    let depthBias: Float
    let depthSlope: Float
    let vsMask: UInt32
    let fsMask: UInt32
    let name: String
    private var variants: [UInt: MTLRenderPipelineState] = [:]
    private let lock = NSLock()

    init(base: MTLRenderPipelineDescriptor, depthState: MTLDepthStencilState, cull: MTLCullMode, fill: MTLTriangleFillMode,
         prim: MTLPrimitiveType, fan: Bool, depthBias: Float, depthSlope: Float, vsMask: UInt32, fsMask: UInt32, name: String) {
        self.base = base
        self.depthState = depthState
        self.cull = cull
        self.fill = fill
        self.prim = prim
        self.fan = fan
        self.depthBias = depthBias
        self.depthSlope = depthSlope
        self.vsMask = vsMask
        self.fsMask = fsMask
        self.name = name
    }

    /// The pipeline state for a render pass with the given depth attachment format (.invalid = none).
    func state(depthFormat: MTLPixelFormat) -> MTLRenderPipelineState? {
        lock.lock(); defer { lock.unlock() }
        if let s = variants[depthFormat.rawValue] { return s }
        let d = base.copy() as! MTLRenderPipelineDescriptor
        d.depthAttachmentPixelFormat = depthFormat
        if depthFormat == .depth32Float_stencil8 { d.stencilAttachmentPixelFormat = depthFormat }
        do {
            let s = try ctx.device.makeRenderPipelineState(descriptor: d)
            variants[depthFormat.rawValue] = s
            return s
        } catch {
            log("pipeline \(name) (depth \(depthFormat.rawValue)) failed: \(error)")
            return nil
        }
    }
}

func writeError(_ msg: String, _ buf: UnsafeMutablePointer<CChar>, _ len: Int32) {
    let bytes = Array(msg.utf8.prefix(Int(len) - 1))
    for (i, b) in bytes.enumerated() { buf[i] = CChar(bitPattern: b) }
    buf[bytes.count] = 0
}

func compileFunction(_ source: String, _ entry: String) throws -> MTLFunction {
    let opts = MTLCompileOptions()
    opts.languageVersion = .version3_0
    opts.preserveInvariance = true
    let lib = try ctx.device.makeLibrary(source: source, options: opts)
    guard let f = lib.makeFunction(name: entry) else {
        throw NSError(domain: "metalmc", code: 1, userInfo: [NSLocalizedDescriptionKey: "entry point \(entry) not found"])
    }
    return f
}

/// Builds a render pipeline from MSL (translated from SPIR-V on the Java side) plus a packed int
/// description. Layout of `p`:
///   [0] topology (0 point, 1 line, 2 line strip, 3 triangle, 4 triangle strip, 5 fan)
///   [1] cull, [2] wireframe, [3] has depth state, [4] compare, [5] depth write,
///   [6] depth bias constant (float bits), [7] depth bias slope (float bits),
///   [8] vertex-stage uniform mask, [9] fragment-stage uniform mask, [10] precreate depth format,
///   [11] color target count N, then N x 9: format, write mask, blend, rgbOp, alphaOp, srcRGB, dstRGB, srcA, dstA,
///   then vertex buffer count M, M x 3: slot, stride, step rate,
///   then attribute count K, K x 4: location, slot, offset, vertex format.
@_cdecl("mmc_pipeline_create")
public func mmc_pipeline_create(_ name: UnsafePointer<CChar>, _ vsSrc: UnsafePointer<CChar>, _ vsEntry: UnsafePointer<CChar>,
                                _ fsSrc: UnsafePointer<CChar>?, _ fsEntry: UnsafePointer<CChar>?,
                                _ p: UnsafePointer<Int32>, _ count: Int32,
                                _ err: UnsafeMutablePointer<CChar>, _ errLen: Int32) -> Int64 {
    let pipeName = String(cString: name)
    return autoreleasepool { () -> Int64 in
        do {
            let d = MTLRenderPipelineDescriptor()
            d.label = pipeName
            d.vertexFunction = try compileFunction(String(cString: vsSrc), String(cString: vsEntry))
            if let fsSrc, let fsEntry {
                d.fragmentFunction = try compileFunction(String(cString: fsSrc), String(cString: fsEntry))
            }
            var i = 0
            func next() -> Int32 { defer { i += 1 }; return p[i] }
            let topo = next()
            let cull = next() != 0
            let wire = next() != 0
            let hasDepth = next() != 0
            let compare = MTLCompareFunction(rawValue: UInt(next())) ?? .always
            let write = next() != 0
            let bias = Float(bitPattern: UInt32(bitPattern: next()))
            let slope = Float(bitPattern: UInt32(bitPattern: next()))
            let vsMask = UInt32(bitPattern: next())
            let fsMask = UInt32(bitPattern: next())
            let precreate = next()
            let colorCount = Int(next())
            for c in 0..<colorCount {
                let fmt = UInt(next()), mask = UInt(next()), blend = next() != 0
                let rgbOp = UInt(next()), aOp = UInt(next()), sRGB = UInt(next()), dRGB = UInt(next()), sA = UInt(next()), dA = UInt(next())
                let att = d.colorAttachments[c]!
                att.pixelFormat = MTLPixelFormat(rawValue: fmt) ?? .invalid
                att.writeMask = MTLColorWriteMask(rawValue: mask)
                att.isBlendingEnabled = blend
                if blend {
                    att.rgbBlendOperation = MTLBlendOperation(rawValue: rgbOp) ?? .add
                    att.alphaBlendOperation = MTLBlendOperation(rawValue: aOp) ?? .add
                    att.sourceRGBBlendFactor = MTLBlendFactor(rawValue: sRGB) ?? .one
                    att.destinationRGBBlendFactor = MTLBlendFactor(rawValue: dRGB) ?? .zero
                    att.sourceAlphaBlendFactor = MTLBlendFactor(rawValue: sA) ?? .one
                    att.destinationAlphaBlendFactor = MTLBlendFactor(rawValue: dA) ?? .zero
                }
            }
            let vd = MTLVertexDescriptor()
            let bufferCount = Int(next())
            for _ in 0..<bufferCount {
                let slot = Int(next()), stride = Int(next()), step = Int(next())
                let layout = vd.layouts[vertexBufferIndex(slot)]!
                layout.stride = stride
                if step > 0 {
                    layout.stepFunction = .perInstance
                    layout.stepRate = step
                } else {
                    layout.stepFunction = .perVertex
                }
            }
            let attrCount = Int(next())
            for _ in 0..<attrCount {
                let loc = Int(next()), slot = Int(next()), off = Int(next()), fmt = UInt(next())
                let a = vd.attributes[loc]!
                a.bufferIndex = vertexBufferIndex(slot)
                a.offset = off
                a.format = MTLVertexFormat(rawValue: fmt) ?? .invalid
            }
            precondition(i <= Int(count), "pipeline params overrun")
            if bufferCount > 0 { d.vertexDescriptor = vd }

            let prim: MTLPrimitiveType
            switch topo {
            case 0: prim = .point
            case 1: prim = .line
            case 2: prim = .lineStrip
            case 4: prim = .triangleStrip
            default: prim = .triangle
            }
            let box = PipelineBox(
                base: d,
                depthState: hasDepth ? ctx.depthState(compare: compare, write: write) : ctx.depthState(compare: .always, write: false),
                cull: cull ? .back : .none,
                fill: wire ? .lines : .fill,
                prim: prim, fan: topo == 5,
                depthBias: hasDepth ? bias : 0, depthSlope: hasDepth ? slope : 0,
                vsMask: vsMask, fsMask: fsMask, name: pipeName)
            // Warm the variants the game will use so the render thread doesn't compile.
            if box.state(depthFormat: .depth32Float) == nil {
                throw NSError(domain: "metalmc", code: 2, userInfo: [NSLocalizedDescriptionKey: "pipeline state creation failed"])
            }
            if !hasDepth { _ = box.state(depthFormat: .invalid) }
            if precreate != 0, let pf = MTLPixelFormat(rawValue: UInt(precreate)), pf != .depth32Float { _ = box.state(depthFormat: pf) }
            return makeHandle(box)
        } catch {
            writeError("\(error)", err, errLen)
            return 0
        }
    }
}

// MARK: - Render passes

/// colors: `count` texture(-view) handles (0 = unused slot). clearMask bit i: clear color i to
/// clearColors[4i..4i+3]. depthClear != 0: clear depth to depthValue. Area in pixels.
@_cdecl("mmc_pass_begin")
public func mmc_pass_begin(_ colors: UnsafePointer<Int64>, _ count: Int32, _ clearMask: Int32, _ clearColors: UnsafePointer<Float>,
                           _ depth: Int64, _ depthClear: Int32, _ depthValue: Float,
                           _ ax: Int32, _ ay: Int32, _ aw: Int32, _ ah: Int32) -> Int32 {
    autoreleasepool { () -> Int32 in
        ctx.endBlit()
        let cb = ctx.ensureCB()
        let d = MTLRenderPassDescriptor()
        var w = 0, h = 0
        for i in 0..<Int(count) where colors[i] != 0 {
            let t = (from(colors[i]) as TextureBox).texture
            let att = d.colorAttachments[i]!
            att.texture = t
            att.storeAction = .store
            if clearMask & (1 << i) != 0 {
                att.loadAction = .clear
                att.clearColor = MTLClearColor(red: Double(clearColors[4 * i]), green: Double(clearColors[4 * i + 1]),
                                               blue: Double(clearColors[4 * i + 2]), alpha: Double(clearColors[4 * i + 3]))
            } else {
                att.loadAction = .load
            }
            w = t.width
            h = t.height
        }
        ctx.passDepthFormat = .invalid
        if depth != 0 {
            let t = (from(depth) as TextureBox).texture
            d.depthAttachment.texture = t
            d.depthAttachment.storeAction = .store
            if depthClear != 0 {
                d.depthAttachment.loadAction = .clear
                d.depthAttachment.clearDepth = Double(depthValue)
            } else {
                d.depthAttachment.loadAction = .load
            }
            if t.pixelFormat == .depth32Float_stencil8 {
                d.stencilAttachment.texture = t
                d.stencilAttachment.loadAction = .load
                d.stencilAttachment.storeAction = .store
            }
            ctx.passDepthFormat = t.pixelFormat
            if count == 0 || w == 0 {
                w = t.width
                h = t.height
            }
        }
        guard let enc = cb.makeRenderCommandEncoder(descriptor: d) else { return 0 }
        ctx.pass = enc
        ctx.resetBindings()
        ctx.passWidth = w
        ctx.passHeight = h
        ctx.pipe = nil
        ctx.indexBuffer = nil
        enc.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(w), height: Double(h), znear: 0, zfar: 1))
        enc.setFrontFacing(.clockwise)
        setScissor(enc, Int(ax), Int(ay), Int(aw), Int(ah))
        return 1
    }
}

@_cdecl("mmc_pass_end")
public func mmc_pass_end() {
    ctx.pass?.endEncoding()
    ctx.pass = nil
    ctx.pipe = nil
    ctx.indexBuffer = nil
}

func setScissor(_ enc: MTLRenderCommandEncoder, _ x: Int, _ y: Int, _ w: Int, _ h: Int) {
    let x0 = min(max(x, 0), ctx.passWidth), y0 = min(max(y, 0), ctx.passHeight)
    let x1 = min(max(x + w, 0), ctx.passWidth), y1 = min(max(y + h, 0), ctx.passHeight)
    if x1 <= x0 || y1 <= y0 {
        ctx.scissorEmpty = true
        return
    }
    ctx.scissorEmpty = false
    let r = MTLScissorRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    if let last = ctx.lastScissor, last.x == r.x, last.y == r.y, last.width == r.width, last.height == r.height { return }
    enc.setScissorRect(r)
    ctx.lastScissor = r
}

@_cdecl("mmc_rp_scissor")
public func mmc_rp_scissor(_ x: Int32, _ y: Int32, _ w: Int32, _ h: Int32) {
    guard let enc = ctx.pass else { return }
    setScissor(enc, Int(x), Int(y), Int(w), Int(h))
}

@_cdecl("mmc_rp_set_pipeline")
public func mmc_rp_set_pipeline(_ h: Int64) -> Int32 {
    guard let enc = ctx.pass else { return 0 }
    let p: PipelineBox = from(h)
    if ctx.pipe === p { return 1 }
    guard let st = p.state(depthFormat: ctx.passDepthFormat) else {
        ctx.pipe = nil
        return 0
    }
    let prev = ctx.pipe
    if ctx.boundPipeState !== st {
        enc.setRenderPipelineState(st)
        ctx.boundPipeState = st
    }
    if prev == nil || prev!.depthState !== p.depthState { enc.setDepthStencilState(p.depthState) }
    if prev == nil || prev!.cull != p.cull { enc.setCullMode(p.cull) }
    if prev == nil || prev!.fill != p.fill { enc.setTriangleFillMode(p.fill) }
    if prev == nil || prev!.depthBias != p.depthBias || prev!.depthSlope != p.depthSlope {
        enc.setDepthBias(p.depthBias, slopeScale: p.depthSlope, clamp: 0)
    }
    ctx.pipe = p
    return 1
}

@_cdecl("mmc_rp_bind_buffer")
public func mmc_rp_bind_buffer(_ index: Int32, _ h: Int64, _ offset: Int64) {
    guard let enc = ctx.pass, let p = ctx.pipe else { return }
    let b = (from(h) as BufferBox).buffer
    let bit = UInt32(1) << UInt32(index)
    if p.vsMask & bit != 0 { ctx.bindBuffer(enc, 0, b, Int(offset), Int(index)) }
    if p.fsMask & bit != 0 { ctx.bindBuffer(enc, 1, b, Int(offset), Int(index)) }
}

@_cdecl("mmc_rp_bind_texture")
public func mmc_rp_bind_texture(_ index: Int32, _ tex: Int64, _ smp: Int64) {
    guard let enc = ctx.pass, let p = ctx.pipe else { return }
    let t = (from(tex) as TextureBox).texture
    let s = (from(smp) as SamplerBox).state
    let bit = UInt32(1) << UInt32(index)
    if p.vsMask & bit != 0 {
        ctx.bindTexture(enc, 0, t, Int(index))
        ctx.bindSampler(enc, 0, s, Int(index))
    }
    if p.fsMask & bit != 0 {
        ctx.bindTexture(enc, 1, t, Int(index))
        ctx.bindSampler(enc, 1, s, Int(index))
    }
}

func texelTexture(_ box: BufferBox, _ offset: Int, _ length: Int, _ format: MTLPixelFormat, _ bpp: Int) -> MTLTexture? {
    let key = TexelKey(offset: offset, length: length, format: format.rawValue)
    if let t = box.texelViews[key] { return t }
    if box.texelViews.count > 64 { box.texelViews.removeAll() }
    let width = max(1, length / bpp)
    let align = ctx.device.minimumTextureBufferAlignment(for: format)
    var buffer = box.buffer
    var off = offset
    if off % align != 0 {
        // Misaligned slice: bind a private copy (shared memory, so a memcpy is enough).
        guard let copy = ctx.device.makeBuffer(bytes: box.buffer.contents() + off, length: max(length, 16), options: [.storageModeShared]) else { return nil }
        buffer = copy
        off = 0
    }
    let bytesPerRow = (width * bpp + align - 1) / align * align
    let d = MTLTextureDescriptor.textureBufferDescriptor(with: format, width: width, resourceOptions: buffer.resourceOptions, usage: .shaderRead)
    guard let t = buffer.makeTexture(descriptor: d, offset: off, bytesPerRow: bytesPerRow) else { return nil }
    box.texelViews[key] = t
    return t
}

@_cdecl("mmc_rp_bind_texel_buffer")
public func mmc_rp_bind_texel_buffer(_ index: Int32, _ h: Int64, _ offset: Int64, _ length: Int64, _ format: Int32, _ bpp: Int32) {
    guard let enc = ctx.pass, let p = ctx.pipe, let pf = MTLPixelFormat(rawValue: UInt(format)) else { return }
    guard let t = texelTexture(from(h), Int(offset), Int(length), pf, Int(bpp)) else {
        log("texel buffer view failed (format \(format), offset \(offset), length \(length))")
        return
    }
    let bit = UInt32(1) << UInt32(index)
    if p.vsMask & bit != 0 { ctx.bindTexture(enc, 0, t, Int(index)) }
    if p.fsMask & bit != 0 { ctx.bindTexture(enc, 1, t, Int(index)) }
}

@_cdecl("mmc_rp_push_constants")
public func mmc_rp_push_constants(_ ptr: UnsafeRawPointer, _ len: Int32) {
    guard let enc = ctx.pass else { return }
    enc.setVertexBytes(ptr, length: Int(len), index: pushConstantsIndex)
    enc.setFragmentBytes(ptr, length: Int(len), index: pushConstantsIndex)
    ctx.boundBuffers[0][pushConstantsIndex] = nil
    ctx.boundBuffers[1][pushConstantsIndex] = nil
}

@_cdecl("mmc_rp_set_vertex_buffer")
public func mmc_rp_set_vertex_buffer(_ slot: Int32, _ h: Int64, _ offset: Int64) {
    guard let enc = ctx.pass else { return }
    ctx.bindBuffer(enc, 0, (from(h) as BufferBox).buffer, Int(offset), vertexBufferIndex(Int(slot)))
}

/// type: 0 = uint16, 1 = uint32.
@_cdecl("mmc_rp_set_index_buffer")
public func mmc_rp_set_index_buffer(_ h: Int64, _ type: Int32) {
    ctx.indexBuffer = (from(h) as BufferBox).buffer
    ctx.indexType = type == 1 ? .uint32 : .uint16
    ctx.indexSize = type == 1 ? 4 : 2
}

@inline(__always) func drawReady() -> (MTLRenderCommandEncoder, PipelineBox)? {
    guard let enc = ctx.pass, let p = ctx.pipe, !ctx.scissorEmpty else { return nil }
    return (enc, p)
}

/// Triangle fans have no Metal equivalent: expand them to lists. Only the sky and debug views use them.
func drawFan(_ enc: MTLRenderCommandEncoder, _ vertexIndices: [UInt32], _ instanceCount: Int, _ baseVertex: Int, _ baseInstance: Int) {
    guard vertexIndices.count >= 3 else { return }
    var list: [UInt32] = []
    list.reserveCapacity((vertexIndices.count - 2) * 3)
    for t in 1..<(vertexIndices.count - 1) {
        list.append(vertexIndices[0]); list.append(vertexIndices[t]); list.append(vertexIndices[t + 1])
    }
    guard let ib = ctx.device.makeBuffer(bytes: list, length: list.count * 4, options: [.storageModeShared]) else { return }
    enc.drawIndexedPrimitives(type: .triangle, indexCount: list.count, indexType: .uint32, indexBuffer: ib, indexBufferOffset: 0,
                              instanceCount: instanceCount, baseVertex: baseVertex, baseInstance: baseInstance)
}

func fanIndices(first: Int, count: Int) -> [UInt32] {
    guard let ib = ctx.indexBuffer else { return [] }
    let base = ib.contents() + first * ctx.indexSize
    if ctx.indexType == .uint32 {
        let p = base.assumingMemoryBound(to: UInt32.self)
        return (0..<count).map { p[$0] }
    }
    let p = base.assumingMemoryBound(to: UInt16.self)
    return (0..<count).map { UInt32(p[$0]) }
}

@_cdecl("mmc_rp_draw_indexed")
public func mmc_rp_draw_indexed(_ indexCount: Int32, _ instanceCount: Int32, _ firstIndex: Int32, _ baseVertex: Int32, _ baseInstance: Int32) {
    guard let r = drawReady(), let ib = ctx.indexBuffer, indexCount > 0, instanceCount > 0 else { return }
    let (enc, p) = r
    if p.fan {
        drawFan(enc, fanIndices(first: Int(firstIndex), count: Int(indexCount)), Int(instanceCount), Int(baseVertex), Int(baseInstance))
        return
    }
    enc.drawIndexedPrimitives(type: p.prim, indexCount: Int(indexCount), indexType: ctx.indexType, indexBuffer: ib,
                              indexBufferOffset: Int(firstIndex) * ctx.indexSize, instanceCount: Int(instanceCount),
                              baseVertex: Int(baseVertex), baseInstance: Int(baseInstance))
}

/// params: drawCount x {firstIndex, indexCount, vertexOffset} (VkMultiDrawIndexedInfoEXT layout).
@_cdecl("mmc_rp_multi_draw_indexed")
public func mmc_rp_multi_draw_indexed(_ params: UnsafePointer<Int32>, _ instanceCount: Int32, _ firstInstance: Int32, _ drawCount: Int32) {
    for i in 0..<Int(drawCount) {
        mmc_rp_draw_indexed(params[3 * i + 1], instanceCount, params[3 * i], params[3 * i + 2], firstInstance)
    }
}

/// Indirect records are MTLDrawIndexedPrimitivesIndirectArguments (20 bytes), same as Vulkan's.
@_cdecl("mmc_rp_draw_indexed_indirect")
public func mmc_rp_draw_indexed_indirect(_ h: Int64, _ offset: Int64, _ drawCount: Int32) {
    guard let r = drawReady(), let ib = ctx.indexBuffer else { return }
    let (enc, p) = r
    let buf = (from(h) as BufferBox).buffer
    let prim = p.fan ? MTLPrimitiveType.triangle : p.prim
    for i in 0..<Int(drawCount) {
        enc.drawIndexedPrimitives(type: prim, indexType: ctx.indexType, indexBuffer: ib, indexBufferOffset: 0,
                                  indirectBuffer: buf, indirectBufferOffset: Int(offset) + i * 20)
    }
}

@_cdecl("mmc_rp_draw")
public func mmc_rp_draw(_ vertexCount: Int32, _ instanceCount: Int32, _ firstVertex: Int32, _ firstInstance: Int32) {
    guard let r = drawReady(), vertexCount > 0, instanceCount > 0 else { return }
    let (enc, p) = r
    if p.fan {
        drawFan(enc, (0..<UInt32(vertexCount)).map { $0 + UInt32(firstVertex) }, Int(instanceCount), 0, Int(firstInstance))
        return
    }
    enc.drawPrimitives(type: p.prim, vertexStart: Int(firstVertex), vertexCount: Int(vertexCount),
                       instanceCount: Int(instanceCount), baseInstance: Int(firstInstance))
}

/// params: drawCount x {firstVertex, vertexCount}.
@_cdecl("mmc_rp_multi_draw")
public func mmc_rp_multi_draw(_ params: UnsafePointer<Int32>, _ instanceCount: Int32, _ firstInstance: Int32, _ drawCount: Int32) {
    for i in 0..<Int(drawCount) {
        mmc_rp_draw(params[2 * i + 1], instanceCount, params[2 * i], firstInstance)
    }
}

@_cdecl("mmc_rp_draw_indirect")
public func mmc_rp_draw_indirect(_ h: Int64, _ offset: Int64, _ drawCount: Int32) {
    guard let r = drawReady() else { return }
    let (enc, p) = r
    let buf = (from(h) as BufferBox).buffer
    for i in 0..<Int(drawCount) {
        enc.drawPrimitives(type: p.fan ? .triangle : p.prim, indirectBuffer: buf, indirectBufferOffset: Int(offset) + i * 16)
    }
}

// MARK: - Copies

func copyBytesPipeline() -> MTLComputePipelineState {
    ctx.utilLock.lock(); defer { ctx.utilLock.unlock() }
    if let p = ctx.copyBytesPipeline { return p }
    let src = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void copy_bytes(device const uchar* src [[buffer(0)]], device uchar* dst [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint i [[thread_position_in_grid]]) {
        if (i < n) dst[i] = src[i];
    }
    """
    let lib = try! ctx.device.makeLibrary(source: src, options: nil)
    let p = try! ctx.device.makeComputePipelineState(function: lib.makeFunction(name: "copy_bytes")!)
    ctx.copyBytesPipeline = p
    return p
}

@_cdecl("mmc_copy_buffer")
public func mmc_copy_buffer(_ src: Int64, _ srcOff: Int64, _ dst: Int64, _ dstOff: Int64, _ size: Int64) {
    guard size > 0 else { return }
    autoreleasepool {
        let s = (from(src) as BufferBox).buffer, d = (from(dst) as BufferBox).buffer
        if srcOff % 4 == 0 && dstOff % 4 == 0 && size % 4 == 0 {
            ctx.blitEncoder().copy(from: s, sourceOffset: Int(srcOff), to: d, destinationOffset: Int(dstOff), size: Int(size))
            return
        }
        // Blit copies on macOS need 4-byte multiples; fall back to a byte-copy kernel.
        ctx.endBlit()
        let pipe = copyBytesPipeline()
        let enc = ctx.ensureCB().makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipe)
        enc.setBuffer(s, offset: Int(srcOff), index: 0)
        enc.setBuffer(d, offset: Int(dstOff), index: 1)
        var n = UInt32(size)
        enc.setBytes(&n, length: 4, index: 2)
        enc.dispatchThreads(MTLSize(width: Int(size), height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding()
    }
}

@_cdecl("mmc_copy_buffer_to_texture")
public func mmc_copy_buffer_to_texture(_ src: Int64, _ srcOff: Int64, _ bytesPerRow: Int32, _ tex: Int64, _ mip: Int32, _ layer: Int32,
                                       _ x: Int32, _ y: Int32, _ w: Int32, _ h: Int32) {
    guard w > 0, h > 0 else { return }
    autoreleasepool {
        let s = (from(src) as BufferBox).buffer, t = (from(tex) as TextureBox).texture
        ctx.blitEncoder().copy(from: s, sourceOffset: Int(srcOff), sourceBytesPerRow: Int(bytesPerRow),
                               sourceBytesPerImage: Int(bytesPerRow) * Int(h), sourceSize: MTLSize(width: Int(w), height: Int(h), depth: 1),
                               to: t, destinationSlice: Int(layer), destinationLevel: Int(mip),
                               destinationOrigin: MTLOrigin(x: Int(x), y: Int(y), z: 0))
    }
}

@_cdecl("mmc_copy_texture_to_buffer")
public func mmc_copy_texture_to_buffer(_ tex: Int64, _ mip: Int32, _ x: Int32, _ y: Int32, _ w: Int32, _ h: Int32,
                                       _ dst: Int64, _ dstOff: Int64, _ bytesPerRow: Int32) {
    guard w > 0, h > 0 else { return }
    autoreleasepool {
        let t = (from(tex) as TextureBox).texture, d = (from(dst) as BufferBox).buffer
        let opts: MTLBlitOption = t.pixelFormat == .depth32Float_stencil8 ? .depthFromDepthStencil : []
        ctx.blitEncoder().copy(from: t, sourceSlice: 0, sourceLevel: Int(mip), sourceOrigin: MTLOrigin(x: Int(x), y: Int(y), z: 0),
                               sourceSize: MTLSize(width: Int(w), height: Int(h), depth: 1), to: d, destinationOffset: Int(dstOff),
                               destinationBytesPerRow: Int(bytesPerRow), destinationBytesPerImage: Int(bytesPerRow) * Int(h), options: opts)
    }
}

@_cdecl("mmc_copy_texture_to_texture")
public func mmc_copy_texture_to_texture(_ src: Int64, _ dst: Int64, _ mip: Int32, _ dx: Int32, _ dy: Int32, _ sx: Int32, _ sy: Int32,
                                        _ w: Int32, _ h: Int32) {
    guard w > 0, h > 0 else { return }
    autoreleasepool {
        let s = (from(src) as TextureBox).texture, d = (from(dst) as TextureBox).texture
        ctx.blitEncoder().copy(from: s, sourceSlice: 0, sourceLevel: Int(mip), sourceOrigin: MTLOrigin(x: Int(sx), y: Int(sy), z: 0),
                               sourceSize: MTLSize(width: Int(w), height: Int(h), depth: 1), to: d, destinationSlice: 0,
                               destinationLevel: Int(mip), destinationOrigin: MTLOrigin(x: Int(dx), y: Int(dy), z: 0))
    }
}

// MARK: - Clears

/// Clears every mip level (layer 0) of a color and/or depth texture, like vkCmdClear*Image.
@_cdecl("mmc_clear_textures")
public func mmc_clear_textures(_ color: Int64, _ rgba: UnsafePointer<Float>, _ depth: Int64, _ depthValue: Float) {
    autoreleasepool {
        ctx.endBlit()
        let cb = ctx.ensureCB()
        if color != 0 {
            let t = (from(color) as TextureBox).texture
            for level in 0..<t.mipmapLevelCount {
                let d = MTLRenderPassDescriptor()
                d.colorAttachments[0].texture = t
                d.colorAttachments[0].level = level
                d.colorAttachments[0].loadAction = .clear
                d.colorAttachments[0].storeAction = .store
                d.colorAttachments[0].clearColor = MTLClearColor(red: Double(rgba[0]), green: Double(rgba[1]), blue: Double(rgba[2]), alpha: Double(rgba[3]))
                cb.makeRenderCommandEncoder(descriptor: d)?.endEncoding()
            }
        }
        if depth != 0 {
            let t = (from(depth) as TextureBox).texture
            for level in 0..<t.mipmapLevelCount {
                let d = MTLRenderPassDescriptor()
                d.depthAttachment.texture = t
                d.depthAttachment.level = level
                d.depthAttachment.loadAction = .clear
                d.depthAttachment.storeAction = .store
                d.depthAttachment.clearDepth = Double(depthValue)
                cb.makeRenderCommandEncoder(descriptor: d)?.endEncoding()
            }
        }
    }
}

let utilShaderSource = """
#include <metal_stdlib>
using namespace metal;
struct VOut { float4 pos [[position]]; };
struct ClearParams { float4 color; float depth; };
vertex VOut fullscreen_vs(uint vid [[vertex_id]], constant ClearParams& p [[buffer(0)]]) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(uv * 2.0 - 1.0, p.depth, 1.0);
    return o;
}
fragment float4 clear_fs(VOut in [[stage_in]], constant ClearParams& p [[buffer(0)]]) { return p.color; }
vertex VOut blit_vs(uint vid [[vertex_id]]) {
    float2 uv = float2((vid << 1) & 2, vid & 2);
    VOut o;
    o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    return o;
}
// Copies with a vertical flip: texture row 0 is the GL bottom row, drawable row 0 is the top.
fragment float4 blit_fs(VOut in [[stage_in]], texture2d<float, access::read> src [[texture(0)]],
                        constant uint2& size [[buffer(0)]]) {
    uint2 p = uint2(in.pos.xy);
    return src.read(uint2(p.x, size.y - 1 - p.y));
}
"""

var utilLibrary: MTLLibrary?

func utilFunction(_ name: String) -> MTLFunction {
    if utilLibrary == nil { utilLibrary = try! ctx.device.makeLibrary(source: utilShaderSource, options: nil) }
    return utilLibrary!.makeFunction(name: name)!
}

func clearPipeline(color: MTLPixelFormat, depth: MTLPixelFormat) -> MTLRenderPipelineState {
    let key = UInt64(color.rawValue) << 32 | UInt64(depth.rawValue)
    ctx.utilLock.lock(); defer { ctx.utilLock.unlock() }
    if let p = ctx.clearPipelines[key] { return p }
    let d = MTLRenderPipelineDescriptor()
    d.vertexFunction = utilFunction("fullscreen_vs")
    d.fragmentFunction = utilFunction("clear_fs")
    d.colorAttachments[0].pixelFormat = color
    d.depthAttachmentPixelFormat = depth
    let p = try! ctx.device.makeRenderPipelineState(descriptor: d)
    ctx.clearPipelines[key] = p
    return p
}

/// Clears a rectangle of one mip level of a color + depth pair (vkCmdClearAttachments equivalent).
@_cdecl("mmc_clear_region")
public func mmc_clear_region(_ color: Int64, _ rgba: UnsafePointer<Float>, _ depth: Int64, _ depthValue: Float,
                             _ x: Int32, _ y: Int32, _ w: Int32, _ h: Int32, _ mip: Int32) {
    autoreleasepool {
        ctx.endBlit()
        let ct = (from(color) as TextureBox).texture, dt = (from(depth) as TextureBox).texture
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = ct
        d.colorAttachments[0].level = Int(mip)
        d.colorAttachments[0].loadAction = .load
        d.colorAttachments[0].storeAction = .store
        d.depthAttachment.texture = dt
        d.depthAttachment.level = Int(mip)
        d.depthAttachment.loadAction = .load
        d.depthAttachment.storeAction = .store
        guard let enc = ctx.ensureCB().makeRenderCommandEncoder(descriptor: d) else { return }
        let tw = max(1, ct.width >> Int(mip)), th = max(1, ct.height >> Int(mip))
        let x0 = min(max(Int(x), 0), tw), y0 = min(max(Int(y), 0), th)
        let x1 = min(max(Int(x + w), 0), tw), y1 = min(max(Int(y + h), 0), th)
        if x1 > x0 && y1 > y0 {
            enc.setScissorRect(MTLScissorRect(x: x0, y: y0, width: x1 - x0, height: y1 - y0))
            enc.setRenderPipelineState(clearPipeline(color: ct.pixelFormat, depth: dt.pixelFormat))
            enc.setDepthStencilState(ctx.depthState(compare: .always, write: true))
            var params: (Float, Float, Float, Float, Float, Float, Float, Float) = (rgba[0], rgba[1], rgba[2], rgba[3], depthValue, 0, 0, 0)
            enc.setVertexBytes(&params, length: 32, index: 0)
            enc.setFragmentBytes(&params, length: 32, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
        enc.endEncoding()
    }
}

// MARK: - Submit and completion

@_cdecl("mmc_submit")
public func mmc_submit(_ index: Int64) {
    autoreleasepool {
        ctx.endBlit()
        precondition(ctx.pass == nil, "submit with an open render pass")
        let cb = ctx.ensureCB()
        for d in ctx.pendingDrawables { cb.present(d) }
        ctx.pendingDrawables.removeAll()
        cb.addCompletedHandler { cb in
            if cb.status == .error {
                ctx.cond.lock()
                let n = ctx.gpuErrorsLogged
                ctx.gpuErrorsLogged += 1
                ctx.cond.unlock()
                if n < 20 { log("GPU error in submit \(index): \(cb.error.map { "\($0)" } ?? "unknown")") }
            }
            let gpu = cb.gpuEndTime - cb.gpuStartTime
            ctx.cond.lock()
            if ctx.gpuSeconds.count < 1_000_000 { ctx.gpuSeconds.append(gpu) }
            if index > ctx.completed { ctx.completed = index }
            ctx.cond.broadcast()
            ctx.cond.unlock()
        }
        cb.commit()
        ctx.cb = nil
    }
}

/// Waits until submit `index` completed. timeoutNs < 0 waits forever. Returns 1 if completed.
@_cdecl("mmc_wait_submit")
public func mmc_wait_submit(_ index: Int64, _ timeoutNs: Int64) -> Int32 {
    ctx.cond.lock(); defer { ctx.cond.unlock() }
    if ctx.completed >= index { return 1 }
    if timeoutNs == 0 { return 0 }
    let deadline = timeoutNs < 0 ? Date.distantFuture : Date(timeIntervalSinceNow: Double(timeoutNs) / 1e9)
    while ctx.completed < index {
        if !ctx.cond.wait(until: deadline) { return ctx.completed >= index ? 1 : 0 }
    }
    return 1
}

/// Copies up to `max` recorded per-submit GPU times (seconds) into `out`, clears the record, and
/// returns how many were copied.
@_cdecl("mmc_gpu_times_take")
public func mmc_gpu_times_take(_ out: UnsafeMutablePointer<Double>, _ max: Int32) -> Int32 {
    ctx.cond.lock(); defer { ctx.cond.unlock() }
    let n = min(Int(max), ctx.gpuSeconds.count)
    for i in 0..<n { out[i] = ctx.gpuSeconds[i] }
    ctx.gpuSeconds.removeAll(keepingCapacity: true)
    return Int32(n)
}

@_cdecl("mmc_completed_submit")
public func mmc_completed_submit() -> Int64 {
    ctx.cond.lock(); defer { ctx.cond.unlock() }
    return ctx.completed
}

// MARK: - Window surface (CAMetalLayer from SDL_Metal_GetLayer)

final class SurfaceBox {
    let layer: CAMetalLayer
    var drawable: CAMetalDrawable?
    var width = 0
    var height = 0
    init(_ l: CAMetalLayer) { layer = l }
}

@_cdecl("mmc_surface2_create")
public func mmc_surface2_create(_ layerAddress: Int64) -> Int64 {
    guard let raw = UnsafeMutableRawPointer(bitPattern: Int(layerAddress)) else { return 0 }
    let layer = Unmanaged<CAMetalLayer>.fromOpaque(raw).takeUnretainedValue()
    layer.device = ctx.device
    layer.pixelFormat = .bgra8Unorm
    layer.framebufferOnly = true
    layer.maximumDrawableCount = 3
    layer.allowsNextDrawableTimeout = true
    return makeHandle(SurfaceBox(layer))
}

@_cdecl("mmc_surface2_configure")
public func mmc_surface2_configure(_ h: Int64, _ w: Int32, _ hh: Int32, _ displaySync: Int32) {
    let s: SurfaceBox = from(h)
    s.width = Int(w)
    s.height = Int(hh)
    s.layer.drawableSize = CGSize(width: Int(w), height: Int(hh))
    s.layer.displaySyncEnabled = displaySync != 0
}

@_cdecl("mmc_surface2_acquire")
public func mmc_surface2_acquire(_ h: Int64) -> Int32 {
    let s: SurfaceBox = from(h)
    return autoreleasepool { () -> Int32 in
        s.drawable = s.layer.nextDrawable()
        return s.drawable == nil ? 0 : 1
    }
}

/// Encodes a flipped copy of `srcTex` (a texture view) into the acquired drawable. The present
/// itself is scheduled on the command buffer at the next submit.
@_cdecl("mmc_surface2_blit")
public func mmc_surface2_blit(_ h: Int64, _ srcTex: Int64) {
    let s: SurfaceBox = from(h)
    guard let drawable = s.drawable else { return }
    autoreleasepool {
        ctx.endBlit()
        let src = (from(srcTex) as TextureBox).texture
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = drawable.texture
        d.colorAttachments[0].loadAction = .clear
        d.colorAttachments[0].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 1)
        d.colorAttachments[0].storeAction = .store
        guard let enc = ctx.ensureCB().makeRenderCommandEncoder(descriptor: d) else { return }
        ctx.utilLock.lock()
        if ctx.blitPipeline == nil {
            let pd = MTLRenderPipelineDescriptor()
            pd.vertexFunction = utilFunction("blit_vs")
            pd.fragmentFunction = utilFunction("blit_fs")
            pd.colorAttachments[0].pixelFormat = s.layer.pixelFormat
            ctx.blitPipeline = try! ctx.device.makeRenderPipelineState(descriptor: pd)
        }
        let pipe = ctx.blitPipeline!
        ctx.utilLock.unlock()
        let cw = min(drawable.texture.width, src.width), ch = min(drawable.texture.height, src.height)
        enc.setRenderPipelineState(pipe)
        enc.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(cw), height: Double(ch), znear: 0, zfar: 1))
        enc.setScissorRect(MTLScissorRect(x: 0, y: 0, width: cw, height: ch))
        var size = SIMD2<UInt32>(UInt32(cw), UInt32(ch))
        enc.setFragmentBytes(&size, length: 8, index: 0)
        enc.setFragmentTexture(src, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        ctx.pendingDrawables.append(drawable)
    }
}

@_cdecl("mmc_surface2_present")
public func mmc_surface2_present(_ h: Int64) {
    let s: SurfaceBox = from(h)
    s.drawable = nil
}
