import CoreGraphics
import Foundation
import ImageIO
import Metal
import MetalMCCore
import simd

// MARK: - External pipeline (METALMC_EXTPIPE=<dir>, off by default)
//
// A generic plug-in point for a shader pipeline described outside this repository (docs/extpipe-design.md): the
// directory holds pipeline.json, which names
// - render targets: format, size (the screen's, scaled, or fixed), one or two copies (ping-pong), clear value and when
//   to clear, mip levels;
// - programs: MSL files and entry points, which target each fragment output goes to (OptiFine's DRAWBUFFERS), how vertex
//   attributes map to vanilla's vertex elements, which of vanilla's uniform buffers to bind where, blending;
// - the G-buffer: which targets vanilla's level passes draw into instead of the main target, and which program draws
//   each of vanilla's pipelines there (a pipeline with no route isn't drawn);
// - the full-screen passes after the opaque geometry ("deferred"), after the translucent geometry ("composite") and the
//   last one onto the screen ("final"), with Iris's ping-pong rules: a pass reads the current copy of every target,
//   writes the other copy of its outputs, and those then flip;
// - custom textures (PNG files, or the game's block atlas) per stage, constant textures, the texture slots;
// - the uniform blocks' layouts, by the standard OptiFine/Iris uniform names and meanings (gbufferModelView,
//   cameraPosition, frameCounter, sunPosition, ...) and a few per-draw ones (gl_ModelViewMatrix, ...).
// Java computes the standard uniforms each frame (metalmc.backend.MetalExtPipe) and marks the level's sky and main
// passes; Backend.swift's hooks ("ExtPipe hook") hand those passes' pipeline changes, bindings and draws to this file.
// Programs reload when their MSL files change (polled twice a second), the description when pipeline.json does.
// Nothing here knows any particular pack.

let extPipeDir: String? = {
    guard let v = ProcessInfo.processInfo.environment["METALMC_EXTPIPE"], !v.isEmpty else { return nil }
    return (v as NSString).expandingTildeInPath
}()

/// True while a redirected G-buffer pass is the open render encoder (Backend.swift's hooks route vanilla's work here).
var extPassActive = false

/// What Java tells about each of vanilla's pipelines (only with METALMC_EXTPIPE set): its uniforms' names by index, its
/// vertex shader's inputs by location, and its depth test (the routed programs take it over in GL's depth convention).
final class ExtPipelineInfo {
    let uniformNames: [String]
    let attribNames: [Int: String]
    let depthCompare: MTLCompareFunction?
    let depthWrite: Bool
    init(uniformNames: [String], attribNames: [Int: String], depthCompare: MTLCompareFunction?, depthWrite: Bool) {
        self.uniformNames = uniformNames
        self.attribNames = attribNames
        self.depthCompare = depthCompare
        self.depthWrite = depthWrite
    }
}

/// The standard uniforms Java computes each frame, in the order it writes them (MetalExtPipe.STD_NAMES): name, float
/// count, integer. Matrices are column-major. Integers travel as floats (exact below 2^24).
let extStdLayout: [(String, Int, Bool)] = [
    ("gbufferModelView", 16, false), ("gbufferModelViewInverse", 16, false),
    ("gbufferPreviousModelView", 16, false), ("gbufferPreviousProjection", 16, false),
    ("gbufferProjection", 16, false), ("gbufferProjectionInverse", 16, false),
    ("shadowModelView", 16, false), ("shadowModelViewInverse", 16, false),
    ("shadowProjection", 16, false), ("shadowProjectionInverse", 16, false),
    ("cameraPosition", 3, false), ("previousCameraPosition", 3, false),
    ("sunPosition", 3, false), ("moonPosition", 3, false), ("upPosition", 3, false), ("shadowLightPosition", 3, false),
    ("skyColor", 3, false), ("fogColor", 3, false),
    ("eyeBrightness", 2, true), ("eyeBrightnessSmooth", 2, true),
    ("aspectRatio", 1, false), ("blindness", 1, false), ("darknessFactor", 1, false), ("far", 1, false), ("near", 1, false),
    ("fogMode", 1, true), ("fogStart", 1, false), ("fogEnd", 1, false), ("fogDensity", 1, false),
    ("frameCounter", 1, true), ("frameTime", 1, false), ("frameTimeCounter", 1, false),
    ("heldBlockLightValue", 1, true), ("heldBlockLightValue2", 1, true), ("heldItemId", 1, true), ("heldItemId2", 1, true),
    ("isEyeInWater", 1, true), ("moonPhase", 1, true), ("nightVision", 1, false), ("rainStrength", 1, false),
    ("sunAngle", 1, false), ("shadowAngle", 1, false), ("viewHeight", 1, false), ("viewWidth", 1, false), ("wetness", 1, false),
    ("worldTime", 1, true), ("worldDay", 1, true), ("screenBrightness", 1, false), ("eyeAltitude", 1, false),
    ("centerDepthSmooth", 1, false), ("hideGUI", 1, true), ("thunderStrength", 1, false), ("playerMood", 1, false),
]

let extStdIndex: [String: (Int, Int, Bool)] = {
    var m: [String: (Int, Int, Bool)] = [:]
    var at = 0
    for (name, n, isInt) in extStdLayout {
        m[name] = (at, n, isInt)
        at += n
    }
    return m
}()
let extStdCount = extStdLayout.reduce(0) { $0 + $1.1 }

// MARK: - The description (pipeline.json)

struct ExtDesc: Decodable {
    struct Member: Decodable {
        let name: String
        let semantic: String?   // the standard or per-draw name it holds, if not `name`
        let type: String        // mat4 mat3 vec4 vec3 vec2 float int ivec2 ivec3 ivec4 bool
        let offset: Int
    }
    struct Block: Decodable {
        let buffer: Int
        let size: Int
        let scope: String       // "frame" (standard uniforms) or "draw" (per draw / per pass)
        let members: [Member]
    }
    struct Target: Decodable {
        let name: String
        let format: String
        let size: [Int]?        // fixed size; nil: the screen's times `scale`
        let scale: Double?
        let clear: [Double]?    // nil: never cleared (history)
        let clearFog: Bool?     // clear to the fog color (alpha from `clear`, else 1)
        let clearMode: String?  // "frame" (default when `clear` is set) or "once"
        let copies: Int?        // 2 (ping-pong, default for color) or 1
        let mips: Bool?         // a full mip chain
    }
    struct Custom: Decodable {
        let stage: String       // "gbuffers", "deferred", "composite", "final" or "*"
        let name: String        // the texture name it stands in for (noisetex, colortex5, ...)
        let file: String?       // a PNG, relative to the description
        let game: String?       // a game texture (minecraft:textures/atlas/blocks.png)
    }
    struct Constant: Decodable {
        let name: String
        let value: [Double]
    }
    struct Program: Decodable {
        let vertex: String
        let vertexEntry: String
        let fragment: String?
        let fragmentEntry: String?
        let outputs: [String]?              // target of fragment output i ("screen": the frame)
        let attributes: [String: String]?   // vertex attribute location -> vanilla element name, "quad", or "zero"
        let vanillaBuffers: [String: Int]?  // vanilla uniform buffer name -> Metal buffer index
        let blend: [String]?                // OptiFine factors: srcRGB dstRGB srcA dstA
        let mathMode: String?               // "fast" (default) or "safe"
    }
    struct Route: Decodable {
        let pipeline: String    // vanilla pipeline name; a trailing * matches a prefix
        let program: String?    // nil: not drawn
        let alphaTest: Double?
        let renderStage: Int?
        let entityId: Int?
    }
    struct Sampler: Decodable {
        let min: String?
        let mag: String?
        let mip: String?
        let wrap: String?
    }
    struct DepthCopy: Decodable {
        let when: String        // "translucent": before the translucent geometry; "end": after the last G-buffer pass
        let target: String
    }
    struct Gbuffers: Decodable {
        let attachments: [String]
        let depth: String
        let depthCopies: [DepthCopy]?
        let routes: [Route]
        /// The first-person hand into the G-buffer (Java's ExtHand submits it with the level's features, so it's drawn before
        /// the deferred passes and its draws go through `routes`; vanilla's own hand pass is skipped).
        let hand: Bool?
        let samplers: [String: Sampler]?    // game texture name -> the sampler the routed programs get instead of vanilla's
    }
    struct Pass: Decodable {
        let stage: String       // "deferred", "composite" or "final"
        let program: String
        let mipsBefore: [String]?
        let flipAfter: [String]?
        let enabled: Bool?
    }
    /// The shadow pass: before the deferred passes, the terrain draws of the routes listed (as vanilla issued them in the
    /// main pass) again through `program`, from the shadow camera, into `colors` and `depth`. With `expand` (a geometry
    /// shader emulated by vertex expansion) each input triangle becomes that many vertices of a non-indexed draw, and the
    /// program gets {triangles, index mode 2 (quads), first index, base vertex, base instance, render stage} at buffer
    /// `paramsBuffer`; it pulls the vertices itself (buffers 30 and 29 as vanilla bound them, its vanillaBuffers).
    /// `translucentRoutes`: translucent terrain (water) draws, which come after the deferred passes: the shadow pass draws
    /// the last frame's (quads in vertex order, their sections' positions as they were, the current camera).
    struct Shadow: Decodable {
        let program: String
        let colors: [String]
        let depth: String
        let depthCopy: String?
        let expand: Int?
        let paramsBuffer: Int?
        let routes: [String]
        let translucentRoutes: [String]?
        let alphaTest: Double?
        let renderStage: Int?
        let enabled: Bool?
    }
    /// Our LOD's opaque quads in the G-buffer pass (Lod.swift's hook): `program`'s vertex function reads the buffers Lod.swift
    /// binds for its own (lod_vs: quads 18, LodUniforms 19, xforms 20, material colors 21, AO offsets 22; frame block too),
    /// its fragment function writes the G-buffer attachments in order; `seamFragmentEntry` (same library) the variant for
    /// tiles that overlap vanilla's terrain (the seam bitmap at fragment buffer 21). The LOD material ids are the
    /// MMC_MAT_<NAME> macros in every program. Water quads are drawn opaque like the rest (the program sees their
    /// materials); no far field (its levels are drawn as quads) or fades there.
    struct Lod: Decodable {
        let program: String
        let seamFragmentEntry: String?
    }
    let lod: Lod?
    let name: String?
    let mslVersion: String?
    let mathMode: String?
    let consts: [String: Double]?
    let uniformBlocks: [Block]
    let textureSlots: [String: Int]
    let gameSamplerSlots: [String: Int]?
    let targets: [Target]
    let customTextures: [Custom]?
    let constantTextures: [Constant]?
    let programs: [String: Program]
    let gbuffers: Gbuffers
    let passes: [Pass]
    let shadow: Shadow?
}

func extPixelFormat(_ s: String) -> MTLPixelFormat {
    switch s {
    case "rgba8Unorm": return .rgba8Unorm
    case "rgba8Snorm": return .rgba8Snorm
    case "rgba16Unorm": return .rgba16Unorm
    case "rgba16Snorm": return .rgba16Snorm
    case "rgba16Float": return .rgba16Float
    case "rgba32Float": return .rgba32Float
    case "rg8Unorm": return .rg8Unorm
    case "rg16Unorm": return .rg16Unorm
    case "rg16Float": return .rg16Float
    case "rg32Float": return .rg32Float
    case "r8Unorm": return .r8Unorm
    case "r16Unorm": return .r16Unorm
    case "r16Float": return .r16Float
    case "r32Float": return .r32Float
    case "rgb10a2Unorm": return .rgb10a2Unorm
    case "rg11b10Float": return .rg11b10Float
    case "depth32Float": return .depth32Float
    default: return .invalid
    }
}

func extBlendFactor(_ s: String) -> MTLBlendFactor {
    switch s.uppercased() {
    case "ZERO": return .zero
    case "ONE": return .one
    case "SRC_COLOR": return .sourceColor
    case "ONE_MINUS_SRC_COLOR": return .oneMinusSourceColor
    case "DST_COLOR": return .destinationColor
    case "ONE_MINUS_DST_COLOR": return .oneMinusDestinationColor
    case "SRC_ALPHA": return .sourceAlpha
    case "ONE_MINUS_SRC_ALPHA": return .oneMinusSourceAlpha
    case "DST_ALPHA": return .destinationAlpha
    case "ONE_MINUS_DST_ALPHA": return .oneMinusDestinationAlpha
    case "SRC_ALPHA_SATURATE": return .sourceAlphaSaturated
    default: return .one
    }
}

// MARK: - Targets and programs

final class ExtTarget {
    let desc: ExtDesc.Target
    let format: MTLPixelFormat
    let copies: Int
    var tex: [MTLTexture] = []
    var cur = 0
    var clearedOnce = false
    var isDepth: Bool { format == .depth32Float }
    init(_ d: ExtDesc.Target) {
        desc = d
        format = extPixelFormat(d.format)
        copies = d.copies ?? (extPixelFormat(d.format) == .depth32Float ? 1 : 2)
    }
    var current: MTLTexture { tex[cur] }
    var alt: MTLTexture { tex[copies == 2 ? 1 - cur : cur] }
    func flip() { if copies == 2 { cur = 1 - cur } }
}

final class ExtProgram {
    let name: String
    let desc: ExtDesc.Program
    var vfn: MTLFunction?
    var ffn: MTLFunction?        // as written: color(i) is output i (full-screen passes)
    var ffnG: MTLFunction?       // color(i) moved to output i's G-buffer attachment (routed draws)
    var flib: MTLLibrary?        // the fragment library (other entry points of it)
    var written: Set<Int> = []   // the fragment outputs the source declares
    var mtimes: [String: Date] = [:]
    var passPSO: [String: MTLRenderPipelineState] = [:]          // by the attachments' formats
    var routedPSO: [ObjectIdentifier: MTLRenderPipelineState] = [:]   // by vanilla pipeline (its vertex layout)
    var routedFailed: Set<ObjectIdentifier> = []
    var failed = false
    init(name: String, desc: ExtDesc.Program) {
        self.name = name
        self.desc = desc
    }
}

/// Last-modified time of a file, or nil.
func extMtime(_ path: String) -> Date? {
    (try? FileManager.default.attributesOfItem(atPath: path))?[.modificationDate] as? Date
}

// MARK: - The pipeline

final class ExtPipe {
    static var shared: ExtPipe?
    static var loadTried = false
    static var lastDescCheck = Date.distantPast

    let dir: URL
    let desc: ExtDesc
    let descPath: String
    let descMtime: Date?
    var targets: [String: ExtTarget] = [:]
    var programs: [String: ExtProgram] = [:]
    var routedPrograms: Set<String> = []
    var custom: [String: [String: MTLTexture]] = [:]     // stage -> name -> texture
    var customGame: [(stage: String, name: String, game: String)] = []
    var constants: [String: MTLTexture] = [:]
    let dummyColor: MTLTexture
    let dummyDepth: MTLTexture
    var gameSamplers: [String: MTLSamplerState] = [:]
    var frameBlocks: [(ExtDesc.Block, [MTLBuffer])] = []
    var drawBlock: ExtDesc.Block?
    var ring = 0
    let quad: MTLBuffer
    let zeroAttr: MTLBuffer
    var std = [Float](repeating: 0, count: extStdCount)
    var width = 0, height = 0
    var lastReloadCheck = Date()

    // Frame state.
    var frameActive = false
    var pendingRedirect = 0      // 1 sky, 2 main
    var phase = 0                // 0 none, 1 sky pass, 2 opaque, 3 translucent
    var deferredDone = false
    var screen: MTLTexture?
    var blockAtlas: MTLTexture?
    var writtenInStage: Set<String> = []
    var stage = ""
    var stats = (frames: 0, routed: 0, skipped: 0, passes: 0)
    var unrouted: Set<String> = []

    // Draw state inside a G-buffer pass.
    var program: ExtProgram?
    var route: ExtDesc.Route?
    var vanilla: PipelineBox?
    var vanillaBuffers: [String: (MTLBuffer, Int)] = [:]
    var vertexBuffers: [Int: (MTLBuffer, Int)] = [:]
    var gtextureSize = SIMD2<Int32>(0, 0)

    // The shadow pass: the main pass's terrain draws, kept to draw again from the shadow camera.
    struct ShadowRecord {
        let vertices: (MTLBuffer, Int)?
        let instances: (MTLBuffer, Int)?
        let globals: (MTLBuffer, Int)?
        let args: MTLBuffer
        let argsOffset: Int
        let count: Int
        let renderStage: Int
    }
    var shadowRecords: [ShadowRecord] = []
    /// A translucent section draw kept for the next frame's shadow pass: its vertices, first vertex and triangles, and
    /// its entry of the section stream (chunk position + visibility, 16 bytes) copied.
    struct TranslucentDraw {
        let vertices: (MTLBuffer, Int)
        let start: Int32
        let tris: UInt32
        let chunk: SIMD4<Int32>
        let renderStage: Int
    }
    var translucentDraws: [TranslucentDraw] = []
    var lastTranslucent: [TranslucentDraw] = []
    var translucentSeen = false
    var translucentChunks: [MTLBuffer?] = [nil, nil, nil]
    var shadowMode = false
    var shadowPSO: MTLRenderPipelineState?
    var shadowPSOFor: MTLFunction?
    var downsamplePSO: MTLRenderPipelineState?

    static let drawBlockIndex = 1

    init(dir: URL) throws {
        self.dir = dir
        descPath = dir.appendingPathComponent("pipeline.json").path
        descMtime = extMtime(descPath)
        let data = try Data(contentsOf: URL(fileURLWithPath: descPath))
        desc = try JSONDecoder().decode(ExtDesc.self, from: data)
        let dev = ctx.device

        func tiny(_ format: MTLPixelFormat) -> MTLTexture {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: format, width: 1, height: 1, mipmapped: false)
            d.usage = [.shaderRead, .renderTarget]
            d.storageMode = .private
            return dev.makeTexture(descriptor: d)!
        }
        dummyColor = tiny(.rgba8Unorm)
        dummyDepth = tiny(.depth32Float)
        // The full-screen quad: OptiFine's, (0,0)-(1,1) as a triangle strip; position and texcoord are the same numbers.
        let q: [Float] = [0, 0, 1, 0, 0, 1, 1, 1]
        quad = dev.makeBuffer(bytes: q, length: q.count * 4, options: [.storageModeShared])!
        // Missing vertex attributes read zeros (and (0,0,0,1) for four-component floats, GL's default).
        var z = [Float](repeating: 0, count: 16)
        z[7] = 1
        zeroAttr = dev.makeBuffer(bytes: z, length: z.count * 4, options: [.storageModeShared])!

        for t in desc.targets { targets[t.name] = ExtTarget(t) }
        for (name, p) in desc.programs { programs[name] = ExtProgram(name: name, desc: p) }
        for r in desc.gbuffers.routes { if let p = r.program { routedPrograms.insert(p) } }
        for b in desc.uniformBlocks {
            if b.scope == "draw" {
                drawBlock = b
            } else {
                let ring = (0..<3).map { _ in dev.makeBuffer(length: max(16, b.size), options: [.storageModeShared])! }
                frameBlocks.append((b, ring))
            }
        }
        for c in desc.constantTextures ?? [] {
            let v = c.value + [Double](repeating: c.value.last ?? 0, count: max(0, 4 - c.value.count))
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba16Float, width: 1, height: 1, mipmapped: false)
            d.usage = [.shaderRead]
            d.storageMode = .shared
            let t = dev.makeTexture(descriptor: d)!
            var h: [Float16] = v.prefix(4).map { Float16(Float($0)) }
            t.replace(region: MTLRegionMake2D(0, 0, 1, 1), mipmapLevel: 0, withBytes: &h, bytesPerRow: 8)
            constants[c.name] = t
        }
        for c in desc.customTextures ?? [] {
            if let g = c.game {
                customGame.append((c.stage, c.name, g))
            } else if let f = c.file {
                let path = dir.appendingPathComponent(f).path
                if let t = ExtPipe.loadPNG(path) {
                    custom[c.stage, default: [:]][c.name] = t
                } else {
                    log("extpipe: couldn't load \(path)")
                }
            }
        }
        for (name, s) in desc.gbuffers.samplers ?? [:] {
            let d = MTLSamplerDescriptor()
            d.minFilter = s.min == "linear" ? .linear : .nearest
            d.magFilter = s.mag == "linear" ? .linear : .nearest
            d.mipFilter = s.mip == "linear" ? .linear : (s.mip == "nearest" ? .nearest : .notMipmapped)
            let wrap: MTLSamplerAddressMode = s.wrap == "repeat" ? .repeat : .clampToEdge
            d.sAddressMode = wrap
            d.tAddressMode = wrap
            if let st = dev.makeSamplerState(descriptor: d) { gameSamplers[name] = st }
        }
        try compileAll()
    }

    // MARK: Compiling

    func mslOptions(_ p: ExtDesc.Program) -> MTLCompileOptions {
        let o = MTLCompileOptions()
        switch desc.mslVersion ?? "3.1" {
        case "3.0": o.languageVersion = .version3_0
        case "3.2":
            if #available(macOS 15.0, *) { o.languageVersion = .version3_2 } else { o.languageVersion = .version3_1 }
        default: o.languageVersion = .version3_1
        }
        let mode = p.mathMode ?? desc.mathMode ?? "fast"
        if #available(macOS 15.0, *) {
            o.mathMode = mode == "safe" ? .safe : .fast
        } else {
            o.fastMathEnabled = mode != "safe"
        }
        o.preprocessorMacros = ExtPipe.materialMacros
        return o
    }

    /// The LOD's material ids (MetalMCCore's Mat) as MMC_MAT_<NAME> macros, for programs that read our LOD's quads.
    static let materialMacros: [String: NSObject] = {
        var m: [String: NSObject] = [:]
        for c in Mat.allCases { m["MMC_MAT_\(String(describing: c).uppercased())"] = NSNumber(value: c.rawValue) }
        return m
    }()

    static let colorAttr = try! NSRegularExpression(pattern: "\\[\\[color\\((\\d+)\\)\\]\\]")

    /// The fragment outputs a source declares ([[color(n)]]).
    static func colorOutputs(_ src: String) -> Set<Int> {
        var s: Set<Int> = []
        let ns = src as NSString
        for m in colorAttr.matches(in: src, range: NSRange(location: 0, length: ns.length)) {
            if let n = Int(ns.substring(with: m.range(at: 1))) { s.insert(n) }
        }
        return s
    }

    /// The source with output n moved to color(map[n]).
    static func remapColors(_ src: String, _ map: [Int: Int]) -> String {
        let ns = src as NSString
        var out = ""
        var last = 0
        for m in colorAttr.matches(in: src, range: NSRange(location: 0, length: ns.length)) {
            out += ns.substring(with: NSRange(location: last, length: m.range.location - last))
            let n = Int(ns.substring(with: m.range(at: 1))) ?? 0
            out += "[[color(\(map[n] ?? n))]]"
            last = m.range.location + m.range.length
        }
        out += ns.substring(from: last)
        return out
    }

    /// Compiles one program's functions; throws with the compiler's message.
    func compile(_ p: ExtProgram) throws {
        let dev = ctx.device
        let opts = mslOptions(p.desc)
        let vpath = dir.appendingPathComponent(p.desc.vertex).path
        let vsrc = try String(contentsOfFile: vpath, encoding: .utf8)
        var mt: [String: Date] = [vpath: extMtime(vpath) ?? Date()]
        let vlib = try dev.makeLibrary(source: vsrc, options: opts)
        guard let vf = vlib.makeFunction(name: p.desc.vertexEntry) else {
            throw NSError(domain: "extpipe", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(p.name): no vertex entry \(p.desc.vertexEntry)"])
        }
        var ff: MTLFunction?, ffG: MTLFunction?, fl: MTLLibrary?
        var written: Set<Int> = []
        if let fpath0 = p.desc.fragment, let fentry = p.desc.fragmentEntry {
            let fpath = dir.appendingPathComponent(fpath0).path
            let fsrc = try String(contentsOfFile: fpath, encoding: .utf8)
            mt[fpath] = extMtime(fpath) ?? Date()
            written = ExtPipe.colorOutputs(fsrc)
            let flib = try dev.makeLibrary(source: fsrc, options: opts)
            fl = flib
            ff = flib.makeFunction(name: fentry)
            if ff == nil {
                throw NSError(domain: "extpipe", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(p.name): no fragment entry \(fentry)"])
            }
            if routedPrograms.contains(p.name) {
                // Routed draws share one pass over the G-buffer's attachments: move each output to its attachment.
                var map: [Int: Int] = [:]
                for (i, t) in (p.desc.outputs ?? []).enumerated() {
                    if let a = desc.gbuffers.attachments.firstIndex(of: t) { map[i] = a } else if written.contains(i) {
                        throw NSError(domain: "extpipe", code: 1, userInfo: [NSLocalizedDescriptionKey: "\(p.name): output \(i) (\(t)) isn't a G-buffer attachment"])
                    }
                }
                let glib = try dev.makeLibrary(source: ExtPipe.remapColors(fsrc, map), options: opts)
                ffG = glib.makeFunction(name: fentry)
            }
        }
        p.vfn = vf
        p.ffn = ff
        p.ffnG = ffG
        p.flib = fl
        p.written = written
        p.mtimes = mt
        p.passPSO = [:]
        p.routedPSO = [:]
        p.routedFailed = []
        p.failed = false
    }

    func compileAll() throws {
        let t0 = Date()
        let list = Array(programs.values)
        var errors = [String?](repeating: nil, count: list.count)
        let lock = NSLock()
        DispatchQueue.concurrentPerform(iterations: list.count) { i in
            do { try compile(list[i]) } catch {
                lock.lock(); errors[i] = "\(error)"; lock.unlock()
                list[i].failed = true
            }
        }
        let bad = errors.compactMap { $0 }
        for e in bad { log("extpipe: \(e)") }
        log(String(format: "extpipe: %@: %d programs compiled in %.2f s, %d failed", desc.name ?? dir.lastPathComponent,
                   list.count - bad.count, Date().timeIntervalSince(t0), bad.count))
        // A routed program or a pass that failed makes the pipeline unusable until it's fixed (and reloads).
        var needed = Set(desc.passes.filter { $0.enabled ?? true }.map { $0.program }).union(routedPrograms)
        if let s = desc.shadow, s.enabled ?? true { needed.insert(s.program) }
        let missing = needed.filter { programs[$0] == nil || programs[$0]!.failed }
        if !missing.isEmpty {
            throw NSError(domain: "extpipe", code: 2, userInfo: [NSLocalizedDescriptionKey: "programs not usable: \(missing.sorted())"])
        }
    }

    /// Recompiles programs whose files changed; a failed compile keeps the previous version.
    func reloadChanged() {
        for p in programs.values {
            let changed = p.mtimes.contains { path, t in (extMtime(path) ?? t) != t }
            guard changed else { continue }
            let keep = (p.vfn, p.ffn, p.ffnG, p.written, p.passPSO, p.routedPSO, p.mtimes)
            do {
                let t0 = Date()
                try compile(p)
                log(String(format: "extpipe: reloaded %@ (%.0f ms)", p.name, Date().timeIntervalSince(t0) * 1000))
            } catch {
                log("extpipe: reload of \(p.name) failed, keeping the previous one: \(error)")
                (p.vfn, p.ffn, p.ffnG, p.written, p.passPSO, p.routedPSO, _) = keep
                // Don't retry until the files change again.
                for k in p.mtimes.keys { p.mtimes[k] = extMtime(k) ?? p.mtimes[k] }
            }
        }
    }

    // MARK: Images

    static func loadPNG(_ path: String) -> MTLTexture? {
        guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
              let img = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
        let w = img.width, h = img.height
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let alpha = img.alphaInfo
        let raw = img.dataProvider?.data.map { $0 as Data }
        if let raw, img.bitsPerComponent == 8, img.bitsPerPixel == 32,
           alpha == .last || alpha == .noneSkipLast || alpha == .premultipliedLast,
           img.bitmapInfo.intersection(.byteOrderMask) == [] || img.bitmapInfo.intersection(.byteOrderMask) == .byteOrder32Big {
            // Straight RGBA (PNG with alpha decodes unpremultiplied): copy the rows as they are.
            raw.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
                for y in 0..<h {
                    for x in 0..<(w * 4) { rgba[y * w * 4 + x] = p[y * img.bytesPerRow + x] }
                }
            }
            if alpha == .noneSkipLast { for i in 0..<(w * h) { rgba[4 * i + 3] = 255 } }
        } else if let raw, img.bitsPerComponent == 8, img.bitsPerPixel == 24 {
            raw.withUnsafeBytes { (p: UnsafeRawBufferPointer) in
                for y in 0..<h {
                    for x in 0..<w {
                        for c in 0..<3 { rgba[(y * w + x) * 4 + c] = p[y * img.bytesPerRow + x * 3 + c] }
                        rgba[(y * w + x) * 4 + 3] = 255
                    }
                }
            }
        } else {
            // Anything else through Core Graphics (premultiplied; fine for opaque images).
            guard let cs = CGColorSpace(name: CGColorSpace.sRGB),
                  let cg = CGContext(data: &rgba, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: cs,
                                     bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue) else { return nil }
            cg.draw(img, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        // Row 0 is the image's top row, as the game uploads its textures (texture coordinate v = 0 reads it).
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: w, height: h, mipmapped: false)
        d.usage = [.shaderRead]
        d.storageMode = .shared
        guard let t = ctx.device.makeTexture(descriptor: d) else { return nil }
        t.replace(region: MTLRegionMake2D(0, 0, w, h), mipmapLevel: 0, withBytes: rgba, bytesPerRow: w * 4)
        t.label = "extpipe " + (path as NSString).lastPathComponent
        return t
    }

    // MARK: Targets

    func ensureTargets(_ w: Int, _ h: Int, _ cb: MTLCommandBuffer) {
        let resized = w != width || h != height
        width = w
        height = h
        for t in targets.values {
            let tw: Int, th: Int
            if let s = t.desc.size, s.count == 2 {
                tw = s[0]; th = s[1]
            } else {
                let sc = t.desc.scale ?? 1
                tw = max(1, Int((Double(w) * sc).rounded())); th = max(1, Int((Double(h) * sc).rounded()))
            }
            if !t.tex.isEmpty && t.tex[0].width == tw && t.tex[0].height == th { continue }
            guard t.format != .invalid else { log("extpipe: target \(t.desc.name) has an unknown format \(t.desc.format)"); continue }
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: t.format, width: tw, height: th, mipmapped: t.desc.mips ?? false)
            d.usage = [.shaderRead, .renderTarget]
            d.storageMode = .private
            t.tex = (0..<t.copies).compactMap { i in
                let x = ctx.device.makeTexture(descriptor: d)
                x?.label = "extpipe \(t.desc.name)\(t.copies == 2 ? (i == 0 ? " A" : " B") : "")"
                return x
            }
            t.cur = 0
            t.clearedOnce = false
            // New storage holds garbage: clear it once whatever the target's clear mode (history starts from zero).
            for x in t.tex { clearTexture(cb, x, t, initial: true) }
            t.clearedOnce = true
        }
        if resized { log("extpipe: targets at \(w)x\(h)") }
    }

    func clearValue(_ t: ExtTarget) -> MTLClearColor {
        let c = t.desc.clear ?? [0, 0, 0, 0]
        let v = c + [Double](repeating: c.last ?? 0, count: max(0, 4 - c.count))
        if t.desc.clearFog == true, let f = extStdIndex["fogColor"] {
            return MTLClearColor(red: Double(std[f.0]), green: Double(std[f.0 + 1]), blue: Double(std[f.0 + 2]), alpha: v[3])
        }
        return MTLClearColor(red: v[0], green: v[1], blue: v[2], alpha: v[3])
    }

    /// Clears every level of one texture (a pass per level).
    func clearTexture(_ cb: MTLCommandBuffer, _ x: MTLTexture, _ t: ExtTarget, initial: Bool) {
        for level in 0..<x.mipmapLevelCount {
            let d = MTLRenderPassDescriptor()
            if t.isDepth {
                d.depthAttachment.texture = x
                d.depthAttachment.level = level
                d.depthAttachment.loadAction = .clear
                d.depthAttachment.storeAction = .store
                d.depthAttachment.clearDepth = t.desc.clear?.first ?? 1.0
            } else {
                d.colorAttachments[0].texture = x
                d.colorAttachments[0].level = level
                d.colorAttachments[0].loadAction = .clear
                d.colorAttachments[0].storeAction = .store
                d.colorAttachments[0].clearColor = initial && t.desc.clear == nil ? MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0) : clearValue(t)
            }
            cb.makeRenderCommandEncoder(descriptor: d)?.endEncoding()
        }
    }

    /// The per-frame clears: every copy of each target cleared each frame (several targets per pass where they match).
    func frameClears(_ cb: MTLCommandBuffer) {
        var batch: [(MTLTexture, MTLClearColor)] = []
        func flush() {
            guard !batch.isEmpty else { return }
            let d = MTLRenderPassDescriptor()
            for (i, (x, c)) in batch.enumerated() {
                d.colorAttachments[i].texture = x
                d.colorAttachments[i].loadAction = .clear
                d.colorAttachments[i].storeAction = .store
                d.colorAttachments[i].clearColor = c
            }
            cb.makeRenderCommandEncoder(descriptor: d)?.endEncoding()
            batch.removeAll()
        }
        for name in targets.keys.sorted() {
            let t = targets[name]!
            guard t.desc.clear != nil || t.desc.clearFog == true, !t.tex.isEmpty else { continue }
            let mode = t.desc.clearMode ?? "frame"
            if mode == "once" { continue }   // cleared when allocated
            if t.isDepth {
                for x in t.tex { clearTexture(cb, x, t, initial: false) }
                continue
            }
            // (Level 0 only: mip chains are built from it before they're read.)
            for x in t.tex {
                if let first = batch.first?.0, first.width != x.width || first.height != x.height || batch.count == 8 { flush() }
                batch.append((x, clearValue(t)))
            }
        }
        flush()
    }

    // MARK: Uniforms

    func stdValue(_ name: String) -> [Float]? {
        guard let (at, n, _) = extStdIndex[name] else { return nil }
        return Array(std[at..<(at + n)])
    }

    /// Writes `value` into a block member of type `type` at `p`.
    static func put(_ p: UnsafeMutableRawPointer, _ type: String, _ value: [Float]) {
        func f(_ i: Int) -> Float { i < value.count ? value[i] : 0 }
        switch type {
        case "mat4":
            for i in 0..<16 { p.storeBytes(of: f(i), toByteOffset: 4 * i, as: Float.self) }
        case "mat3":
            // std140 (and MSL float3x3): three columns of four floats; the value is a column-major 3x3 or 4x4.
            let src4 = value.count >= 16
            for c in 0..<3 { for r in 0..<3 { p.storeBytes(of: f(src4 ? c * 4 + r : c * 3 + r), toByteOffset: 16 * c + 4 * r, as: Float.self) } }
        case "vec4", "vec3", "vec2", "float":
            let n = type == "float" ? 1 : Int(String(type.last!))!
            for i in 0..<n { p.storeBytes(of: f(i), toByteOffset: 4 * i, as: Float.self) }
        case "int", "bool", "ivec2", "ivec3", "ivec4", "uint":
            let n = type.hasPrefix("ivec") ? Int(String(type.last!))! : 1
            for i in 0..<n { p.storeBytes(of: Int32(f(i).rounded()), toByteOffset: 4 * i, as: Int32.self) }
        default:
            break
        }
    }

    func writeFrameBlocks() {
        ring = (ring + 1) % 3
        for (b, bufs) in frameBlocks {
            let p = bufs[ring].contents()
            memset(p, 0, bufs[ring].length)
            for m in b.members {
                if let v = stdValue(m.semantic ?? m.name) { ExtPipe.put(p + m.offset, m.type, v) }
            }
        }
    }

    static func mat(_ m: simd_float4x4) -> [Float] {
        [m.columns.0.x, m.columns.0.y, m.columns.0.z, m.columns.0.w, m.columns.1.x, m.columns.1.y, m.columns.1.z, m.columns.1.w,
         m.columns.2.x, m.columns.2.y, m.columns.2.z, m.columns.2.w, m.columns.3.x, m.columns.3.y, m.columns.3.z, m.columns.3.w]
    }

    static func mat(_ f: [Float]) -> simd_float4x4 {
        simd_float4x4(columns: (SIMD4(f[0], f[1], f[2], f[3]), SIMD4(f[4], f[5], f[6], f[7]),
                                SIMD4(f[8], f[9], f[10], f[11]), SIMD4(f[12], f[13], f[14], f[15])))
    }

    static func readMat(_ b: MTLBuffer, _ offset: Int) -> simd_float4x4? {
        guard offset + 64 <= b.length else { return nil }
        let p = (b.contents() + offset).assumingMemoryBound(to: Float.self)
        return mat((0..<16).map { p[$0] })
    }

    /// OptiFine's gl_TextureMatrix[1]: lightmap coordinates 0..240 to (uv + 8) / 256.
    static let lightmapMatrix: [Float] = [1 / 256.0, 0, 0, 0, 0, 1 / 256.0, 0, 0, 0, 0, 1 / 256.0, 0, 8 / 256.0, 8 / 256.0, 8 / 256.0, 1]
    /// OptiFine's full-screen pass projection: ortho(0, 1, 0, 1, -1, 1).
    static let quadProjection: [Float] = [2, 0, 0, 0, 0, 2, 0, 0, 0, 0, -1, 0, -1, -1, 0, 1]
    static let identity: [Float] = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]

    /// The per-draw block's values: full-screen passes (vanillaMV nil) or a routed draw.
    func drawValue(_ semantic: String, mv: simd_float4x4?, textureMat: simd_float4x4?, colorMod: SIMD4<Float>?) -> [Float]? {
        switch semantic {
        case "gl_ModelViewMatrix":
            return mv.map { ExtPipe.mat($0) } ?? ExtPipe.identity
        case "gl_ModelViewMatrixInverse":
            return mv.map { ExtPipe.mat($0.inverse) } ?? ExtPipe.identity
        case "gl_ProjectionMatrix":
            if shadowMode { return stdValue("shadowProjection") }
            return mv == nil ? ExtPipe.quadProjection : stdValue("gbufferProjection")
        case "gl_ProjectionMatrixInverse":
            if shadowMode { return stdValue("shadowProjectionInverse") }
            return mv == nil ? ExtPipe.mat(ExtPipe.mat(ExtPipe.quadProjection).inverse) : stdValue("gbufferProjectionInverse")
        case "gl_ModelViewProjectionMatrix":
            guard let mv, let p = stdValue(shadowMode ? "shadowProjection" : "gbufferProjection") else { return ExtPipe.quadProjection }
            return ExtPipe.mat(ExtPipe.mat(p) * mv)
        case "gl_TextureMatrix0":
            return textureMat.map { ExtPipe.mat($0) } ?? ExtPipe.identity
        case "gl_TextureMatrix1":
            return mv == nil ? ExtPipe.identity : ExtPipe.lightmapMatrix
        case "gl_NormalMatrix":
            guard let mv else { return ExtPipe.identity }
            let m3 = simd_float3x3(SIMD3(mv.columns.0.x, mv.columns.0.y, mv.columns.0.z), SIMD3(mv.columns.1.x, mv.columns.1.y, mv.columns.1.z),
                                   SIMD3(mv.columns.2.x, mv.columns.2.y, mv.columns.2.z))
            let n = m3.inverse.transpose
            return [n.columns.0.x, n.columns.0.y, n.columns.0.z, n.columns.1.x, n.columns.1.y, n.columns.1.z, n.columns.2.x, n.columns.2.y, n.columns.2.z]
        case "alphaTestRef":
            if shadowMode { return [Float(desc.shadow?.alphaTest ?? -1)] }
            return [Float(route?.alphaTest ?? -1)]
        case "renderStage":
            if shadowMode { return [Float(desc.shadow?.renderStage ?? 0)] }
            return [Float(route?.renderStage ?? 0)]
        case "entityId", "blockEntityId":
            return [Float(route?.entityId ?? 0)]
        case "atlasSize":
            return [Float(gtextureSize.x), Float(gtextureSize.y)]
        case "entityColor":
            return [0, 0, 0, 0]
        case "colorModulator":
            guard let c = colorMod else { return [1, 1, 1, 1] }
            return [c.x, c.y, c.z, c.w]
        default:
            return stdValue(semantic)
        }
    }

    func drawBytes(mv: simd_float4x4?, textureMat: simd_float4x4?, colorMod: SIMD4<Float>?) -> [UInt8] {
        guard let b = drawBlock else { return [] }
        var bytes = [UInt8](repeating: 0, count: max(16, b.size))
        bytes.withUnsafeMutableBytes { raw in
            for m in b.members {
                if let v = drawValue(m.semantic ?? m.name, mv: mv, textureMat: textureMat, colorMod: colorMod) {
                    ExtPipe.put(raw.baseAddress! + m.offset, m.type, v)
                }
            }
        }
        return bytes
    }

    // MARK: Textures per stage

    func texture(_ name: String, stage: String, attachments: Set<String> = []) -> MTLTexture? {
        if !writtenInStage.contains(name) {
            if let c = custom[stage]?[name] ?? custom["*"]?[name] { return c }
            for g in customGame where (g.stage == stage || g.stage == "*") && g.name == name {
                if g.game.hasSuffix("atlas/blocks.png"), let a = blockAtlas { return a }
            }
        }
        if let t = targets[name], !t.tex.isEmpty {
            if attachments.contains(name) { return t.isDepth ? dummyDepth : dummyColor }   // being drawn into
            return t.current
        }
        if let c = constants[name] { return c }
        return nil
    }

    func bindStageTextures(_ enc: MTLRenderCommandEncoder, stage: String, attachments: Set<String>, skip: Set<String> = []) {
        for (name, slot) in desc.textureSlots where !skip.contains(name) {
            let t = texture(name, stage: stage, attachments: attachments)
                ?? (name.hasPrefix("depthtex") || name.hasPrefix("shadowtex") ? dummyDepth : dummyColor)
            enc.setVertexTexture(t, index: slot)
            enc.setFragmentTexture(t, index: slot)
        }
    }

    // MARK: Pipeline states

    static func blendKey(_ b: [String]?) -> String { b?.joined(separator: ",") ?? "-" }

    func applyBlend(_ att: MTLRenderPipelineColorAttachmentDescriptor, _ blend: [String]?) {
        guard let b = blend, b.count >= 2 else { att.isBlendingEnabled = false; return }
        let src = extBlendFactor(b[0]), dst = extBlendFactor(b[1])
        let sa = b.count >= 4 ? extBlendFactor(b[2]) : src, da = b.count >= 4 ? extBlendFactor(b[3]) : dst
        if src == .one && dst == .zero && sa == .one && da == .zero { att.isBlendingEnabled = false; return }
        att.isBlendingEnabled = true
        att.sourceRGBBlendFactor = src
        att.destinationRGBBlendFactor = dst
        att.sourceAlphaBlendFactor = sa
        att.destinationAlphaBlendFactor = da
        att.rgbBlendOperation = .add
        att.alphaBlendOperation = .add
    }

    /// The vertex descriptor for a program's stage_in attributes: vanilla's elements by name, the quad, or zeros.
    func vertexDescriptor(_ p: ExtProgram, vanilla: PipelineBox?) -> MTLVertexDescriptor? {
        guard let attrs = p.vfn?.vertexAttributes, !attrs.isEmpty else { return nil }
        let vd = MTLVertexDescriptor()
        var usesZero = false, usesQuad = false
        let vbase = vanilla?.base.vertexDescriptor
        for a in attrs where a.isActive {
            let loc = a.attributeIndex
            let want = p.desc.attributes?[String(loc)] ?? "zero"
            let dst = vd.attributes[loc]!
            if want == "quad" {
                dst.format = .float2
                dst.offset = 0
                dst.bufferIndex = 30
                usesQuad = true
                continue
            }
            if want != "zero", let info = vanilla?.ext, let vb = vbase,
               let vloc = info.attribNames.first(where: { $0.value == want })?.key {
                let src = vb.attributes[vloc]!
                if src.format != .invalid {
                    dst.format = src.format
                    dst.offset = src.offset
                    dst.bufferIndex = src.bufferIndex
                    let sl = vb.layouts[src.bufferIndex]!, dl = vd.layouts[src.bufferIndex]!
                    dl.stride = sl.stride
                    dl.stepFunction = sl.stepFunction
                    dl.stepRate = sl.stepRate
                    continue
                }
            }
            usesZero = true
            dst.bufferIndex = 16
            switch a.attributeType {
            case .float: dst.format = .float; dst.offset = 0
            case .float2: dst.format = .float2; dst.offset = 0
            case .float3: dst.format = .float3; dst.offset = 0
            case .int: dst.format = .int; dst.offset = 0
            case .int2: dst.format = .int2; dst.offset = 0
            case .int3: dst.format = .int3; dst.offset = 0
            case .int4: dst.format = .int4; dst.offset = 0
            case .uint: dst.format = .uint; dst.offset = 0
            case .uint2: dst.format = .uint2; dst.offset = 0
            case .uint3: dst.format = .uint3; dst.offset = 0
            case .uint4: dst.format = .uint4; dst.offset = 0
            default: dst.format = .float4; dst.offset = 16
            }
        }
        if usesZero {
            vd.layouts[16]!.stride = 32
            vd.layouts[16]!.stepFunction = .constant
            vd.layouts[16]!.stepRate = 0
        }
        if usesQuad {
            vd.layouts[30]!.stride = 8
            vd.layouts[30]!.stepFunction = .perVertex
        }
        return vd
    }

    func routedPSO(_ p: ExtProgram, _ v: PipelineBox) -> MTLRenderPipelineState? {
        if let s = p.routedPSO[ObjectIdentifier(v)] { return s }
        guard let vf = p.vfn, !p.routedFailed.contains(ObjectIdentifier(v)) else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.label = "extpipe \(p.name) for \(v.name)"
        d.vertexFunction = vf
        d.fragmentFunction = p.ffnG
        d.vertexDescriptor = vertexDescriptor(p, vanilla: v)
        d.depthAttachmentPixelFormat = targets[desc.gbuffers.depth]?.format ?? .depth32Float
        for (i, name) in desc.gbuffers.attachments.enumerated() {
            let att = d.colorAttachments[i]!
            att.pixelFormat = targets[name]?.format ?? .invalid
            let outIndex = (p.desc.outputs ?? []).firstIndex(of: name)
            if let oi = outIndex, p.written.contains(oi) {
                att.writeMask = .all
                applyBlend(att, p.desc.blend)
            } else {
                att.writeMask = []
            }
        }
        do {
            let s = try ctx.device.makeRenderPipelineState(descriptor: d)
            p.routedPSO[ObjectIdentifier(v)] = s
            return s
        } catch {
            log("extpipe: \(p.name) for \(v.name): \(error)")
            p.routedFailed.insert(ObjectIdentifier(v))
            return nil
        }
    }

    func passPSO(_ p: ExtProgram, _ formats: [MTLPixelFormat]) -> MTLRenderPipelineState? {
        let key = formats.map { String($0.rawValue) }.joined(separator: ",")
        if let s = p.passPSO[key] { return s }
        guard let vf = p.vfn else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.label = "extpipe \(p.name)"
        d.vertexFunction = vf
        d.fragmentFunction = p.ffn
        d.vertexDescriptor = vertexDescriptor(p, vanilla: nil)
        for (i, f) in formats.enumerated() where f != .invalid {
            d.colorAttachments[i].pixelFormat = f
            applyBlend(d.colorAttachments[i]!, p.desc.blend)
        }
        do {
            let s = try ctx.device.makeRenderPipelineState(descriptor: d)
            p.passPSO[key] = s
            return s
        } catch {
            log("extpipe: \(p.name): \(error)")
            return nil
        }
    }

    // MARK: Frame

    func beginFrame(_ values: UnsafePointer<Float>, _ count: Int, _ w: Int, _ h: Int, atlas: MTLTexture?) -> Bool {
        if Date().timeIntervalSince(lastReloadCheck) > 0.5 {
            lastReloadCheck = Date()
            reloadChanged()
            pollViewFile()
        }
        if frameActive { endFrame() }   // the last frame never closed its main pass
        let n = min(count, extStdCount)
        for i in 0..<n { std[i] = values[i] }
        blockAtlas = atlas
        let cb = ctx.ensureCB()
        ctx.endBlit()
        ensureTargets(w, h, cb)
        frameClears(cb)
        writeFrameBlocks()
        frameActive = true
        phase = 0
        deferredDone = false
        pendingRedirect = 0
        stage = "gbuffers"
        writtenInStage = []
        shadowRecords.removeAll(keepingCapacity: true)
        if translucentSeen { swap(&lastTranslucent, &translucentDraws) }
        translucentDraws.removeAll(keepingCapacity: true)
        translucentSeen = false
        stats.frames += 1
        if stats.frames % 600 == 1 {
            log("extpipe: frame \(stats.frames): routed draws \(stats.routed), skipped pipelines: \(unrouted.sorted().joined(separator: " "))")
        }
        return true
    }

    func endFrame() {
        frameActive = false
        phase = 0
        pendingRedirect = 0
        extPassActive = false
    }

    /// Opens the G-buffer pass in place of the vanilla pass Java marked (the sky's or the main one). `mainColor` is the
    /// texture the vanilla pass would have drawn into: the final pass draws there.
    func beginGbufferPass(mainColor: MTLTexture?) -> Bool {
        let kind = pendingRedirect
        pendingRedirect = 0
        guard frameActive else { return false }
        if kind == 2 { screen = mainColor }
        if screen == nil { screen = mainColor }
        phase = kind == 1 ? 1 : (phase == 3 ? 3 : 2)
        return openGbufferEncoder()
    }

    func openGbufferEncoder() -> Bool {
        ctx.endBlit()
        let cb = ctx.ensureCB()
        let d = MTLRenderPassDescriptor()
        var formats: [MTLPixelFormat] = []
        for (i, name) in desc.gbuffers.attachments.enumerated() {
            guard let t = targets[name], !t.tex.isEmpty else { formats.append(.invalid); continue }
            let att = d.colorAttachments[i]!
            att.texture = t.current
            att.loadAction = .load
            att.storeAction = .store
            formats.append(t.format)
        }
        guard let depth = targets[desc.gbuffers.depth], !depth.tex.isEmpty else { return false }
        d.depthAttachment.texture = depth.current
        d.depthAttachment.loadAction = .load
        d.depthAttachment.storeAction = .store
        profAttach(d, "extpipe gbuffers (\(phase == 1 ? "sky" : phase == 3 ? "translucent" : "opaque"))")
        guard let enc = cb.makeRenderCommandEncoder(descriptor: d) else { return false }
        enc.label = "extpipe gbuffers"
        ctx.statPasses += 1
        ctx.pass = enc
        ctx.resetBindings()
        ctx.passColorFormats = formats
        ctx.passDepthFormat = depth.format
        ctx.passWidth = depth.current.width
        ctx.passHeight = depth.current.height
        ctx.pipe = nil
        ctx.indexBuffer = nil
        enc.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(ctx.passWidth), height: Double(ctx.passHeight), znear: 0, zfar: 1))
        enc.setFrontFacing(.clockwise)
        setScissor(enc, 0, 0, ctx.passWidth, ctx.passHeight)
        for (b, bufs) in frameBlocks {
            enc.setVertexBuffer(bufs[ring], offset: 0, index: b.buffer)
            enc.setFragmentBuffer(bufs[ring], offset: 0, index: b.buffer)
        }
        enc.setVertexBuffer(zeroAttr, offset: 0, index: 16)
        bindStageTextures(enc, stage: "gbuffers", attachments: Set(desc.gbuffers.attachments + [desc.gbuffers.depth]), skip: ["gtexture", "lightmap"])
        for (name, slot) in desc.textureSlots where name == "gtexture" || name == "lightmap" {
            enc.setFragmentTexture(dummyColor, index: slot)
            enc.setVertexTexture(dummyColor, index: slot)
        }
        program = nil
        route = nil
        vanilla = nil
        vanillaBuffers = [:]
        vertexBuffers = [:]
        extPassActive = true
        return true
    }

    func closeEncoder() {
        ctx.pass?.endEncoding()
        ctx.pass = nil
        ctx.pipe = nil
        ctx.indexBuffer = nil
        extPassActive = false
        program = nil
        route = nil
        vanilla = nil
    }

    /// Between the opaque and the translucent geometry: the deferred passes, then the G-buffer pass again.
    func translucent() -> Bool {
        guard extPassActive, phase == 2 else { return false }
        closeEncoder()
        runDeferred()
        phase = 3
        translucentSeen = true
        stage = "gbuffers"
        return openGbufferEncoder()
    }

    func runDeferred() {
        guard !deferredDone else { return }
        deferredDone = true
        runShadow()
        depthCopies("translucent")
        if ExtPipe.view != nil { captureView(after: "gbuffers", ctx.ensureCB()) }
        runStage("deferred")
    }

    // MARK: Shadow pass

    /// From mmc_rp_draw_indexed_indirect in the opaque G-buffer pass: a terrain draw the shadow pass replays.
    func recordIndirect(_ buf: MTLBuffer, _ offset: Int, _ count: Int) {
        guard let sh = desc.shadow, sh.enabled ?? true, let v = vanilla else { return }
        if phase == 2, sh.routes.contains(v.name) {
            shadowRecords.append(ShadowRecord(vertices: vertexBuffers[0], instances: vertexBuffers[1], globals: vanillaBuffers["Globals"],
                                              args: buf, argsOffset: offset, count: count, renderStage: route?.renderStage ?? 0))
        } else if phase == 3, sh.translucentRoutes?.contains(v.name) ?? false {
            recordTranslucent(buf, offset, count)
        }
    }

    /// Copies what the next frame's shadow pass needs of a translucent multi-draw (the arguments and section stream are
    /// rewritten every frame; the sections' vertex buffers stay).
    func recordTranslucent(_ buf: MTLBuffer, _ offset: Int, _ count: Int) {
        guard let vb = vertexBuffers[0], let (sb, so) = vertexBuffers[1], offset + count * 20 <= buf.length else { return }
        let args = (buf.contents() + offset).assumingMemoryBound(to: UInt32.self)
        // Quads are 4 vertices each from the first vertex; with the shared sequential index buffer a first index is a quad
        // offset, with a sorted one (vanilla sorts translucent quads by index) the section's quads start at its base vertex.
        var sequential: (Int) -> Bool = { _ in false }
        if let ib = ctx.indexBuffer {
            let n = ib.length / ctx.indexSize
            if ctx.indexType == .uint16 {
                let p = ib.contents().assumingMemoryBound(to: UInt16.self)
                sequential = { f in f + 6 <= n && Int(p[f]) == f / 6 * 4 && Int(p[f + 1]) == f / 6 * 4 + 1 && Int(p[f + 4]) == f / 6 * 4 + 3 }
            } else {
                let p = ib.contents().assumingMemoryBound(to: UInt32.self)
                sequential = { f in f + 6 <= n && Int(p[f]) == f / 6 * 4 && Int(p[f + 1]) == f / 6 * 4 + 1 && Int(p[f + 4]) == f / 6 * 4 + 3 }
            }
        }
        let stage = route?.renderStage ?? 0
        for i in 0..<count {
            let a = args + 5 * i
            let indexCount = a[0], instances = a[1], firstIndex = Int(a[2]), baseVertex = Int32(bitPattern: a[3]), baseInstance = Int(a[4])
            guard indexCount >= 6, instances > 0, so + (baseInstance + 1) * 16 <= sb.length else { continue }
            let chunk = (sb.contents() + so + baseInstance * 16).loadUnaligned(as: SIMD4<Int32>.self)
            let start = baseVertex + (firstIndex % 6 == 0 && sequential(firstIndex) ? Int32(firstIndex / 6 * 4) : 0)
            translucentDraws.append(TranslucentDraw(vertices: vb, start: start, tris: indexCount / 3, chunk: chunk, renderStage: stage))
        }
    }

    func runShadow() {
        guard let sh = desc.shadow, sh.enabled ?? true, !(shadowRecords.isEmpty && lastTranslucent.isEmpty), let prog = programs[sh.program],
              let vf = prog.vfn, let ff = prog.ffn, let depth = targets[sh.depth], !depth.tex.isEmpty else { return }
        let colors = sh.colors.compactMap { targets[$0] }.filter { !$0.tex.isEmpty }
        if shadowPSO == nil || shadowPSOFor !== vf {
            // (Again after the program reloads.)
            shadowPSOFor = vf
            let d = MTLRenderPipelineDescriptor()
            d.label = "extpipe shadow"
            d.vertexFunction = vf
            d.fragmentFunction = ff
            d.vertexDescriptor = vertexDescriptor(prog, vanilla: nil)
            for (i, t) in colors.enumerated() { d.colorAttachments[i].pixelFormat = t.format; applyBlend(d.colorAttachments[i]!, prog.desc.blend) }
            d.depthAttachmentPixelFormat = depth.format
            do { shadowPSO = try ctx.device.makeRenderPipelineState(descriptor: d) } catch {
                log("extpipe: shadow program \(sh.program): \(error)")
                shadowPSO = nil
            }
        }
        guard let pso = shadowPSO else { return }
        let cb = ctx.ensureCB()
        ctx.endBlit()
        let d = MTLRenderPassDescriptor()
        for (i, t) in colors.enumerated() {
            d.colorAttachments[i].texture = t.current
            d.colorAttachments[i].loadAction = .clear
            d.colorAttachments[i].clearColor = clearValue(t)
            d.colorAttachments[i].storeAction = .store
        }
        d.depthAttachment.texture = depth.current
        d.depthAttachment.loadAction = .clear
        d.depthAttachment.clearDepth = depth.desc.clear?.first ?? 1.0
        d.depthAttachment.storeAction = .store
        profAttach(d, "extpipe shadow")
        guard let enc = cb.makeRenderCommandEncoder(descriptor: d) else { return }
        enc.label = "extpipe shadow"
        enc.setRenderPipelineState(pso)
        enc.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(depth.current.width), height: Double(depth.current.height), znear: 0, zfar: 1))
        enc.setCullMode(.none)
        enc.setDepthStencilState(ctx.depthState(compare: .lessEqual, write: true))
        for (b, bufs) in frameBlocks {
            enc.setVertexBuffer(bufs[ring], offset: 0, index: b.buffer)
            enc.setFragmentBuffer(bufs[ring], offset: 0, index: b.buffer)
        }
        shadowMode = true
        route = nil
        if let b = drawBlock, let smv = stdValue("shadowModelView") {
            let bytes = drawBytes(mv: ExtPipe.mat(smv), textureMat: nil, colorMod: nil)
            enc.setVertexBytes(bytes, length: bytes.count, index: b.buffer)
            enc.setFragmentBytes(bytes, length: bytes.count, index: b.buffer)
        }
        shadowMode = false
        bindStageTextures(enc, stage: "shadow", attachments: Set(sh.colors + [sh.depth]))
        if let slot = desc.textureSlots["gtexture"], let atlas = blockAtlas {
            enc.setVertexTexture(atlas, index: slot)
            enc.setFragmentTexture(atlas, index: slot)
        }
        if let sslot = desc.gameSamplerSlots?["gtexture"], let s = gameSamplers["gtexture"] {
            enc.setVertexSamplerState(s, index: sslot)
            enc.setFragmentSamplerState(s, index: sslot)
        }
        let expand = sh.expand ?? 0
        let paramsIndex = sh.paramsBuffer ?? 5
        let globalsSlot = prog.desc.vanillaBuffers?["Globals"]
        var draws = 0
        for r in shadowRecords {
            if let (b, o) = r.vertices { enc.setVertexBuffer(b, offset: o, index: 30) }
            if let (b, o) = r.instances { enc.setVertexBuffer(b, offset: o, index: 29) }
            if let slot = globalsSlot, let (b, o) = r.globals { enc.setVertexBuffer(b, offset: o, index: slot) }
            guard r.argsOffset + r.count * 20 <= r.args.length else { continue }
            let args = (r.args.contents() + r.argsOffset).assumingMemoryBound(to: UInt32.self)
            for i in 0..<r.count {
                let a = args + 5 * i
                let indexCount = a[0], instances = a[1], firstIndex = a[2], baseVertex = Int32(bitPattern: a[3]), baseInstance = a[4]
                guard indexCount >= 3, instances > 0 else { continue }
                let tris = indexCount / 3
                // Vanilla's quads share one index pattern ((0,1,2), (2,3,0) per 4 vertices): a first index is a quad offset.
                var params: (UInt32, UInt32, UInt32, Int32, UInt32, UInt32, UInt32, UInt32) =
                    (tris, 2, 0, baseVertex + Int32(firstIndex / 6 * 4), baseInstance, UInt32(r.renderStage), 0, 0)
                enc.setVertexBytes(&params, length: 32, index: paramsIndex)
                if expand > 0 {
                    enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: Int(tris) * expand)
                }
                draws += 1
            }
        }
        // The last frame's translucent sections, with this frame's camera (Globals) and their copied stream entries.
        var waterDraws = 0
        if !lastTranslucent.isEmpty, let globals = shadowRecords.last?.globals {
            let need = lastTranslucent.count * 16
            if (translucentChunks[ring]?.length ?? 0) < need {
                translucentChunks[ring] = ctx.device.makeBuffer(length: max(need * 2, 4096), options: [.storageModeShared])
            }
            if let chunks = translucentChunks[ring] {
                let p = chunks.contents().assumingMemoryBound(to: SIMD4<Int32>.self)
                for (k, t) in lastTranslucent.enumerated() { p[k] = t.chunk }
                enc.setVertexBuffer(chunks, offset: 0, index: 29)
                if let slot = globalsSlot { enc.setVertexBuffer(globals.0, offset: globals.1, index: slot) }
                for (k, t) in lastTranslucent.enumerated() {
                    enc.setVertexBuffer(t.vertices.0, offset: t.vertices.1, index: 30)
                    var params: (UInt32, UInt32, UInt32, Int32, UInt32, UInt32, UInt32, UInt32) =
                        (t.tris, 2, 0, t.start, UInt32(k), UInt32(t.renderStage), 0, 0)
                    enc.setVertexBytes(&params, length: 32, index: paramsIndex)
                    if expand > 0 { enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: Int(t.tris) * expand) }
                    waterDraws += 1
                }
            }
        }
        enc.endEncoding()
        ctx.statPasses += 1
        if stats.frames % 600 == 1 {
            log("extpipe: shadow pass: \(shadowRecords.count) terrain draws, \(draws) sections, \(waterDraws) translucent sections (last frame's)")
        }
        if let c = sh.depthCopy, let t = targets[c], !t.tex.isEmpty, t.format == depth.format,
           t.current.width == depth.current.width, t.current.height == depth.current.height, let blit = cb.makeBlitCommandEncoder() {
            blit.copy(from: depth.current, sourceSlice: 0, sourceLevel: 0, to: t.current, destinationSlice: 0, destinationLevel: 0, sliceCount: 1, levelCount: 1)
            blit.endEncoding()
        }
        if depth.current.mipmapLevelCount > 1 { downsampleDepth(depth.current, cb) }
    }

    /// The depth mip chain (Metal's generateMipmaps doesn't do depth): each level the mean of four texels of the one above,
    /// as GL drivers' glGenerateMipmap does.
    func downsampleDepth(_ t: MTLTexture, _ cb: MTLCommandBuffer) {
        if downsamplePSO == nil {
            let src = """
            #include <metal_stdlib>
            using namespace metal;
            struct VOut { float4 pos [[position]]; };
            vertex VOut extds_vs(uint vid [[vertex_id]]) {
                float2 uv = float2((vid << 1) & 2, vid & 2);
                VOut o; o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0); return o;
            }
            struct DOut { float d [[depth(any)]]; };
            fragment DOut extds_fs(VOut in [[stage_in]], depth2d<float, access::read> src [[texture(0)]]) {
                uint2 p = uint2(in.pos.xy) * 2u;
                uint2 m = uint2(src.get_width() - 1, src.get_height() - 1);
                float a = src.read(min(p, m)), b = src.read(min(p + uint2(1, 0), m));
                float c = src.read(min(p + uint2(0, 1), m)), e = src.read(min(p + uint2(1, 1), m));
                DOut o; o.d = (a + b + c + e) * 0.25; return o;
            }
            """
            guard let lib = try? ctx.device.makeLibrary(source: src, options: nil) else { return }
            let d = MTLRenderPipelineDescriptor()
            d.vertexFunction = lib.makeFunction(name: "extds_vs")
            d.fragmentFunction = lib.makeFunction(name: "extds_fs")
            d.depthAttachmentPixelFormat = t.pixelFormat
            downsamplePSO = try? ctx.device.makeRenderPipelineState(descriptor: d)
        }
        guard let pso = downsamplePSO else { return }
        for level in 1..<t.mipmapLevelCount {
            guard let srcView = t.makeTextureView(pixelFormat: t.pixelFormat, textureType: .type2D, levels: (level - 1)..<level, slices: 0..<1) else { return }
            let d = MTLRenderPassDescriptor()
            d.depthAttachment.texture = t
            d.depthAttachment.level = level
            d.depthAttachment.loadAction = .dontCare
            d.depthAttachment.storeAction = .store
            guard let enc = cb.makeRenderCommandEncoder(descriptor: d) else { return }
            enc.setRenderPipelineState(pso)
            enc.setDepthStencilState(ctx.depthState(compare: .always, write: true))
            enc.setFragmentTexture(srcView, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
            enc.endEncoding()
        }
    }

    func depthCopies(_ when: String) {
        guard let src = targets[desc.gbuffers.depth], !src.tex.isEmpty else { return }
        let copies = (desc.gbuffers.depthCopies ?? []).filter { $0.when == when }
        guard !copies.isEmpty else { return }
        let blit = ctx.blitEncoder()
        for c in copies {
            guard let t = targets[c.target], !t.tex.isEmpty, t.current.width == src.current.width, t.current.height == src.current.height,
                  t.format == src.format else { continue }
            blit.copy(from: src.current, to: t.current)
        }
        ctx.endBlit()
    }

    /// The vanilla pass Java marked ended.
    func endGbufferPass() {
        closeEncoder()
        guard phase >= 2 else { return }   // the sky pass: the main pass comes next
        runDeferred()
        depthCopies("end")
        runStage("composite")
        runStage("final")
        if let v = ExtPipe.view { debugView(v.0, v.1) }
        frameActive = false
        phase = 0
    }

    // MARK: Debug view (METALMC_EXTPIPE_VIEW, or mmc_ext_debug_view): targets on the screen

    /// "<target>[.a][@<pass>][:scale]", comma-separated: each target (its alpha with .a) as it was after that pass (or
    /// "gbuffers": when the G-buffer was done; default: the end of the frame), in a grid over the screen. "frame" is a
    /// cell with the frame itself.
    static var view: (String, Float)? = {
        guard let v = ProcessInfo.processInfo.environment["METALMC_EXTPIPE_VIEW"], !v.isEmpty else { return nil }
        return (v, 1)
    }()
    var viewPSO: [Bool: MTLRenderPipelineState] = [:]
    var viewCaptures: [String: MTLTexture] = [:]
    var frameCopy: MTLTexture?
    var viewFileMtime: Date?

    /// `<dir>/view.txt` (polled with the programs): its first line replaces the debug view while the game runs (an empty
    /// line or no file: the frame), so a session can look at targets without restarting.
    func pollViewFile() {
        let path = dir.appendingPathComponent("view.txt").path
        let m = extMtime(path)
        guard m != viewFileMtime else { return }
        viewFileMtime = m
        let line = m == nil ? "" : ((try? String(contentsOfFile: path, encoding: .utf8)) ?? "")
            .split(separator: "\n").first.map { $0.trimmingCharacters(in: .whitespaces) } ?? ""
        ExtPipe.view = line.isEmpty ? nil : (line, 1)
        viewCaptures = [:]
        log("extpipe: debug view \(line.isEmpty ? "off" : line)")
    }

    struct ViewSpec { let spec: String; let name: String; let alpha: Bool; let pass: String?; let scale: Float }

    static func viewSpecs(_ list: String, _ scale: Float) -> [ViewSpec] {
        list.split(separator: ",").map { item in
            let colon = item.split(separator: ":")
            let s = colon.count > 1 ? (Float(colon[1]) ?? scale) : scale
            let at = colon[0].split(separator: "@")
            var name = String(at[0])
            let alpha = name.hasSuffix(".a")
            if alpha { name = String(name.dropLast(2)) }
            return ViewSpec(spec: String(colon[0]), name: name, alpha: alpha, pass: at.count > 1 ? String(at[1]) : nil, scale: s)
        }
    }

    /// After pass `program` (or "gbuffers" when the G-buffer is done): copies of the viewed targets that ask for it.
    func captureView(after program: String, _ cb: MTLCommandBuffer) {
        guard let (list, scale) = ExtPipe.view else { return }
        for v in ExtPipe.viewSpecs(list, scale) where v.pass == program {
            guard let t = targets[v.name], !t.tex.isEmpty else { continue }
            let src = t.current
            var dst = viewCaptures[v.spec]
            if dst == nil || dst!.width != src.width || dst!.height != src.height || dst!.pixelFormat != src.pixelFormat {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: src.pixelFormat, width: src.width, height: src.height, mipmapped: false)
                d.usage = [.shaderRead]
                d.storageMode = .private
                dst = ctx.device.makeTexture(descriptor: d)
                viewCaptures[v.spec] = dst
            }
            guard let dst, let blit = cb.makeBlitCommandEncoder() else { continue }
            blit.copy(from: src, sourceSlice: 0, sourceLevel: 0, to: dst, destinationSlice: 0, destinationLevel: 0, sliceCount: 1, levelCount: 1)
            blit.endEncoding()
        }
    }

    /// Draws the viewed targets over the screen in a grid: absolute values times the scale; NaN magenta, infinity cyan;
    /// depth as (1 - depth)^(1/4).
    func debugView(_ list: String, _ scale: Float) {
        guard let screen else { return }
        let specs = ExtPipe.viewSpecs(list, scale)
        guard !specs.isEmpty else { return }
        let cols = Int(Double(specs.count).squareRoot().rounded(.up))
        let rows = (specs.count + cols - 1) / cols
        let cb = ctx.ensureCB()
        ctx.endBlit()
        if specs.contains(where: { $0.name == "frame" }) {
            if frameCopy == nil || frameCopy!.width != screen.width || frameCopy!.height != screen.height || frameCopy!.pixelFormat != screen.pixelFormat {
                let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: screen.pixelFormat, width: screen.width, height: screen.height, mipmapped: false)
                d.usage = [.shaderRead]
                d.storageMode = .private
                frameCopy = ctx.device.makeTexture(descriptor: d)
            }
            if let fc = frameCopy, let blit = cb.makeBlitCommandEncoder() {
                blit.copy(from: screen, sourceSlice: 0, sourceLevel: 0, to: fc, destinationSlice: 0, destinationLevel: 0, sliceCount: 1, levelCount: 1)
                blit.endEncoding()
            }
        }
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = screen
        d.colorAttachments[0].loadAction = .clear
        d.colorAttachments[0].clearColor = MTLClearColor(red: 0.2, green: 0.2, blue: 0.2, alpha: 1)
        d.colorAttachments[0].storeAction = .store
        guard let enc = cb.makeRenderCommandEncoder(descriptor: d) else { return }
        let cw = screen.width / cols, ch = screen.height / rows
        for (i, v) in specs.enumerated() {
            let source: MTLTexture?
            var depth = false
            if v.name == "frame" {
                source = frameCopy
            } else if let t = targets[v.name], !t.tex.isEmpty {
                source = v.pass != nil ? viewCaptures[v.spec] : t.current
                depth = t.isDepth
            } else {
                source = nil
            }
            guard let source, let pso = viewPipeline(depth: depth, format: screen.pixelFormat) else { continue }
            // Cell i, counted from the top left of the picture (rows of memory are GL's: row 0 at the bottom).
            let x = (i % cols) * cw, y = (rows - 1 - i / cols) * ch
            enc.setRenderPipelineState(pso)
            enc.setViewport(MTLViewport(originX: Double(x), originY: Double(y), width: Double(cw - 2), height: Double(ch - 2), znear: 0, zfar: 1))
            var p = SIMD4<Float>(Float(x), Float(y), Float(cw - 2), Float(ch - 2))
            var q = SIMD4<Float>(v.scale, v.alpha ? 1 : 0, 0, 0)
            enc.setFragmentBytes(&p, length: 16, index: 0)
            enc.setFragmentBytes(&q, length: 16, index: 1)
            enc.setFragmentTexture(source, index: 0)
            enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        }
        enc.endEncoding()
    }

    func viewPipeline(depth: Bool, format: MTLPixelFormat) -> MTLRenderPipelineState? {
        if let p = viewPSO[depth] { return p }
        let src = """
        #include <metal_stdlib>
        using namespace metal;
        struct VOut { float4 pos [[position]]; };
        vertex VOut extview_vs(uint vid [[vertex_id]]) {
            float2 uv = float2((vid << 1) & 2, vid & 2);
            VOut o; o.pos = float4(uv * 2.0 - 1.0, 0.0, 1.0); return o;
        }
        // p: the cell's origin and size in pixels; q: scale, alpha only.
        fragment float4 extview_fs(VOut in [[stage_in]], texture2d<float> t [[texture(0)]], constant float4& p [[buffer(0)]],
                                   constant float4& q [[buffer(1)]]) {
            constexpr sampler s(filter::nearest);
            float4 c = t.sample(s, (in.pos.xy - p.xy) / p.zw);
            if (q.y > 0.5) c = c.aaaa;
            if (any(isnan(c))) return float4(1.0, 0.0, 1.0, 1.0);
            if (any(isinf(c))) return float4(0.0, 1.0, 1.0, 1.0);
            return float4(abs(c.rgb * q.x), 1.0);
        }
        fragment float4 extview_depth_fs(VOut in [[stage_in]], depth2d<float> t [[texture(0)]], constant float4& p [[buffer(0)]],
                                         constant float4& q [[buffer(1)]]) {
            constexpr sampler s(filter::nearest);
            float d = t.sample(s, (in.pos.xy - p.xy) / p.zw);
            float v = pow(saturate((1.0 - d) * q.x), 0.25);
            return float4(v, v, v, 1.0);
        }
        """
        guard let lib = try? ctx.device.makeLibrary(source: src, options: nil) else { return nil }
        let d = MTLRenderPipelineDescriptor()
        d.vertexFunction = lib.makeFunction(name: "extview_vs")
        d.fragmentFunction = lib.makeFunction(name: depth ? "extview_depth_fs" : "extview_fs")
        d.colorAttachments[0].pixelFormat = format
        let p = try? ctx.device.makeRenderPipelineState(descriptor: d)
        viewPSO[depth] = p
        return p
    }

    func runStage(_ s: String) {
        stage = s
        writtenInStage = []
        let cb = ctx.ensureCB()
        ctx.endBlit()
        for p in desc.passes where p.stage == s && (p.enabled ?? true) { runPass(p, cb) }
    }

    func runPass(_ pass: ExtDesc.Pass, _ cb: MTLCommandBuffer) {
        guard let prog = programs[pass.program], prog.vfn != nil, prog.ffn != nil else { return }
        if let mips = pass.mipsBefore, !mips.isEmpty, let blit = cb.makeBlitCommandEncoder() {
            for n in mips { if let t = targets[n], !t.tex.isEmpty, t.current.mipmapLevelCount > 1 { blit.generateMipmaps(for: t.current) } }
            blit.endEncoding()
        }
        let d = MTLRenderPassDescriptor()
        var formats: [MTLPixelFormat] = []
        var outs: [String] = []
        var w = 0, h = 0
        for (i, name) in (prog.desc.outputs ?? []).enumerated() {
            guard prog.written.contains(i) else { formats.append(.invalid); continue }
            let tex: MTLTexture?
            if name == "screen" { tex = screen } else { tex = targets[name].flatMap { $0.tex.isEmpty ? nil : $0.alt } }
            guard let tex else { formats.append(.invalid); continue }
            d.colorAttachments[i].texture = tex
            d.colorAttachments[i].loadAction = .load   // passes may cover only part of the target
            d.colorAttachments[i].storeAction = .store
            formats.append(tex.pixelFormat)
            outs.append(name)
            w = tex.width
            h = tex.height
        }
        guard w > 0, let pso = passPSO(prog, formats) else { return }
        profAttach(d, "extpipe \(pass.program)")
        guard let enc = cb.makeRenderCommandEncoder(descriptor: d) else { return }
        enc.label = "extpipe \(pass.program)"
        enc.setRenderPipelineState(pso)
        enc.setViewport(MTLViewport(originX: 0, originY: 0, width: Double(w), height: Double(h), znear: 0, zfar: 1))
        enc.setCullMode(.none)
        for (b, bufs) in frameBlocks {
            enc.setVertexBuffer(bufs[ring], offset: 0, index: b.buffer)
            enc.setFragmentBuffer(bufs[ring], offset: 0, index: b.buffer)
        }
        if let b = drawBlock {
            let bytes = drawBytes(mv: nil, textureMat: nil, colorMod: nil)
            enc.setVertexBytes(bytes, length: bytes.count, index: b.buffer)
            enc.setFragmentBytes(bytes, length: bytes.count, index: b.buffer)
        }
        bindStageTextures(enc, stage: pass.stage, attachments: [])
        enc.setVertexBuffer(quad, offset: 0, index: 30)
        enc.setVertexBuffer(zeroAttr, offset: 0, index: 16)
        enc.drawPrimitives(type: .triangleStrip, vertexStart: 0, vertexCount: 4)
        enc.endEncoding()
        ctx.statPasses += 1
        stats.passes += 1
        for n in outs where n != "screen" {
            targets[n]?.flip()
            writtenInStage.insert(n)
        }
        for n in pass.flipAfter ?? [] {
            targets[n]?.flip()
            writtenInStage.insert(n)
        }
        if ExtPipe.view != nil { captureView(after: pass.program, cb) }
    }

    // MARK: Routed draws (Backend.swift's hooks while extPassActive)

    func routeFor(_ name: String) -> ExtDesc.Route? {
        for r in desc.gbuffers.routes {
            if r.pipeline == name { return r }
            if r.pipeline.hasSuffix("*") && name.hasPrefix(String(r.pipeline.dropLast())) { return r }
        }
        return nil
    }

    func setPipeline(_ p: PipelineBox) -> Int32 {
        guard let enc = ctx.pass else { return 0 }
        guard let r = routeFor(p.name), let pn = r.program, let prog = programs[pn], prog.ffnG != nil || prog.ffn == nil,
              let pso = routedPSO(prog, p) else {
            if !unrouted.contains(p.name) { unrouted.insert(p.name) }
            ctx.pipe = nil
            program = nil
            stats.skipped += 1
            return 0
        }
        if ctx.boundPipeState !== pso {
            enc.setRenderPipelineState(pso)
            ctx.boundPipeState = pso
        }
        // GL's depth convention: vanilla's reverse-Z comparisons turned around.
        let cmp: MTLCompareFunction
        switch p.ext?.depthCompare {
        case .some(.greaterEqual): cmp = .lessEqual
        case .some(.greater): cmp = .less
        case .some(.lessEqual): cmp = .greaterEqual
        case .some(.less): cmp = .greater
        case .some(let c): cmp = c
        case .none: cmp = .always
        }
        enc.setDepthStencilState(ctx.depthState(compare: cmp, write: p.ext?.depthWrite ?? false))
        enc.setCullMode(p.cull)
        enc.setTriangleFillMode(p.fill)
        enc.setDepthBias(0, slopeScale: 0, clamp: 0)
        ctx.pipe = p
        program = prog
        route = r
        vanilla = p
        vanillaBuffers = [:]
        gtextureSize = SIMD2(0, 0)
        return 1
    }

    func bindBuffer(_ index: Int, _ b: MTLBuffer, _ offset: Int) {
        guard let enc = ctx.pass, let v = vanilla, let info = v.ext, index < info.uniformNames.count else { return }
        let name = info.uniformNames[index]
        vanillaBuffers[name] = (b, offset)
        if let slot = program?.desc.vanillaBuffers?[name] {
            ctx.bindBuffer(enc, 0, b, offset, slot)
            ctx.bindBuffer(enc, 1, b, offset, slot)
        }
    }

    func bindTexture(_ index: Int, _ t: MTLTexture, _ s: MTLSamplerState) {
        guard let enc = ctx.pass, let v = vanilla, let info = v.ext, index < info.uniformNames.count else { return }
        let game: String
        switch info.uniformNames[index] {
        case "Sampler0": game = "gtexture"
        case "Sampler2": game = "lightmap"
        default: return
        }
        if game == "gtexture" { gtextureSize = SIMD2(Int32(t.width), Int32(t.height)) }
        if let slot = desc.textureSlots[game] {
            ctx.bindTexture(enc, 0, t, slot)
            ctx.bindTexture(enc, 1, t, slot)
        }
        if let sslot = desc.gameSamplerSlots?[game] {
            let st = gameSamplers[game] ?? s
            ctx.bindSampler(enc, 0, st, sslot)
            ctx.bindSampler(enc, 1, st, sslot)
        }
    }

    /// Before each routed draw: the per-draw block from vanilla's transforms as they're bound for it.
    func beforeDraw(_ enc: MTLRenderCommandEncoder) {
        guard drawBlock != nil else { return }
        var mv: simd_float4x4?
        var tm: simd_float4x4?
        var cm: SIMD4<Float>?
        if let (b, off) = vanillaBuffers["DynamicTransforms"], var m = ExtPipe.readMat(b, off) {
            // ModelViewMat, TextureMat, ColorModulator, ModelOffset (std140).
            tm = ExtPipe.readMat(b, off + 64)
            if off + 160 <= b.length {
                let f = (b.contents() + off + 128).assumingMemoryBound(to: Float.self)
                cm = SIMD4(f[0], f[1], f[2], f[3])
                let o = SIMD3(f[4], f[5], f[6])
                if o != .zero {
                    var t = matrix_identity_float4x4
                    t.columns.3 = SIMD4(o.x, o.y, o.z, 1)
                    m = m * t
                }
            }
            mv = m
        } else if let (b, off) = vanillaBuffers["TerrainUniform"], let m = ExtPipe.readMat(b, off) {
            mv = m
            if off + 72 <= b.length {
                let i = (b.contents() + off + 64).assumingMemoryBound(to: Int32.self)
                gtextureSize = SIMD2(i[0], i[1])
            }
        } else if let g = stdValue("gbufferModelView") {
            mv = ExtPipe.mat(g)
        }
        let bytes = drawBytes(mv: mv ?? matrix_identity_float4x4, textureMat: tm, colorMod: cm)
        enc.setVertexBytes(bytes, length: bytes.count, index: drawBlock!.buffer)
        enc.setFragmentBytes(bytes, length: bytes.count, index: drawBlock!.buffer)
        ctx.boundBuffers[0][drawBlock!.buffer] = nil
        ctx.boundBuffers[1][drawBlock!.buffer] = nil
        stats.routed += 1
    }

    // MARK: Our LOD (Lod.swift's hook in mmc_lod_draw)

    var lodPSOs: (key: String, vfn: MTLFunction, ffn: MTLFunction, plain: MTLRenderPipelineState?, seam: MTLRenderPipelineState?)?

    /// The description's LOD program for the opaque G-buffer pass now open (nil: the LOD stays out of it).
    func lodPipelines() -> (plain: MTLRenderPipelineState?, seam: MTLRenderPipelineState?)? {
        guard phase == 2, extPassActive, let l = desc.lod, let prog = programs[l.program], !prog.failed,
              let vf = prog.vfn, let ff = prog.ffn else { return nil }
        let key = ctx.passColorFormats.map { "\($0.rawValue)" }.joined(separator: ",") + "/\(ctx.passDepthFormat.rawValue)"
        if let c = lodPSOs, c.key == key, c.vfn === vf, c.ffn === ff { return (c.plain, c.seam) }
        func make(_ f: MTLFunction?) -> MTLRenderPipelineState? {
            guard let f else { return nil }
            let d = MTLRenderPipelineDescriptor()
            d.label = "extpipe lod"
            d.vertexFunction = vf
            d.fragmentFunction = f
            for (i, fmt) in ctx.passColorFormats.enumerated() where fmt != .invalid { d.colorAttachments[i].pixelFormat = fmt }
            d.depthAttachmentPixelFormat = ctx.passDepthFormat
            do { return try ctx.device.makeRenderPipelineState(descriptor: d) } catch {
                log("extpipe: lod program \(l.program): \(error)")
                return nil
            }
        }
        let plain = make(ff)
        let seam = make(l.seamFragmentEntry.flatMap { prog.flib?.makeFunction(name: $0) })
        lodPSOs = (key, vf, ff, plain, seam)
        return (plain, seam)
    }

    // MARK: Loading

    /// The pipeline for this frame: loads it the first time, and again when pipeline.json changes (a failed reload keeps
    /// the one that works).
    static func current() -> ExtPipe? {
        guard let dirPath = extPipeDir else { return nil }
        if !loadTried {
            loadTried = true
            load(dirPath)
        } else if Date().timeIntervalSince(lastDescCheck) > 1.0 {
            lastDescCheck = Date()
            let path = URL(fileURLWithPath: dirPath).appendingPathComponent("pipeline.json").path
            if let s = shared, extMtime(path) != s.descMtime, !s.frameActive {
                log("extpipe: pipeline.json changed, reloading")
                load(dirPath)
            } else if shared == nil, let m = extMtime(path), m > failedAt {
                load(dirPath)
            }
        }
        return shared
    }

    static var failedAt = Date.distantPast

    static func load(_ dirPath: String) {
        do {
            let p = try ExtPipe(dir: URL(fileURLWithPath: dirPath))
            shared = p
            log("extpipe: loaded \(dirPath): \(p.targets.count) targets, \(p.programs.count) programs, \(p.desc.gbuffers.routes.count) routes, \(p.desc.passes.count) passes")
        } catch {
            log("extpipe: couldn't load \(dirPath): \(error)")
            failedAt = Date()
        }
    }
}

// MARK: - C entry points (metalmc.backend.MetalExtPipe)

/// 1 if METALMC_EXTPIPE names a pipeline that loaded (it loads here the first time).
@_cdecl("mmc_ext_enabled")
public func mmc_ext_enabled() -> Int32 {
    ExtPipe.current() != nil ? 1 : 0
}

/// A `const` of the pipeline's description (sunPathRotation, shadowDistance, ...), or `fallback`.
@_cdecl("mmc_ext_const")
public func mmc_ext_const(_ name: UnsafePointer<CChar>, _ fallback: Float) -> Float {
    guard let v = ExtPipe.shared?.desc.consts?[String(cString: name)] else { return fallback }
    return Float(v)
}

/// What Java knows about one of vanilla's pipelines: uniform names in index order and vertex inputs as "location=name",
/// both newline-separated; its depth compare (MTLCompareFunction raw value, -1 for no depth test) and depth write.
@_cdecl("mmc_ext_pipeline_info")
public func mmc_ext_pipeline_info(_ h: Int64, _ uniforms: UnsafePointer<CChar>, _ attribs: UnsafePointer<CChar>, _ compare: Int32, _ write: Int32) {
    let p: PipelineBox = from(h)
    let names = String(cString: uniforms).split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
    var attr: [Int: String] = [:]
    for line in String(cString: attribs).split(separator: "\n") {
        let parts = line.split(separator: "=", maxSplits: 1)
        if parts.count == 2, let loc = Int(parts[0]) { attr[loc] = String(parts[1]) }
    }
    p.ext = ExtPipelineInfo(uniformNames: names, attribNames: attr,
                            depthCompare: compare < 0 ? nil : MTLCompareFunction(rawValue: UInt(compare)), depthWrite: write != 0)
}

/// Starts a frame of the external pipeline: `values` are the standard uniforms (MetalExtPipe.STD_NAMES order), w x h the
/// main target's size, `atlas` the block atlas's texture view (0 if none). Returns 1 if the level should be redirected.
@_cdecl("mmc_ext_frame_begin")
public func mmc_ext_frame_begin(_ values: UnsafePointer<Float>, _ count: Int32, _ w: Int32, _ h: Int32, _ atlas: Int64) -> Int32 {
    guard ctx.pass == nil, let p = ExtPipe.current() else { return 0 }
    return autoreleasepool { () -> Int32 in
        let a: MTLTexture? = atlas != 0 ? (from(atlas) as TextureBox).texture : nil
        return p.beginFrame(values, Int(count), Int(w), Int(h), atlas: a) ? 1 : 0
    }
}

/// The next render pass is the level's sky pass (1) or main pass (2): it draws into the G-buffer instead.
@_cdecl("mmc_ext_redirect")
public func mmc_ext_redirect(_ kind: Int32) {
    guard let p = ExtPipe.shared, p.frameActive else { return }
    p.pendingRedirect = Int(kind)
}

/// Between the opaque and the translucent geometry of the main pass: runs the deferred passes. Returns 1 if the pass was
/// reopened (Java then binds its pipeline and uniforms again).
@_cdecl("mmc_ext_translucent")
public func mmc_ext_translucent() -> Int32 {
    guard let p = ExtPipe.shared else { return 0 }
    return autoreleasepool { p.translucent() ? 1 : 0 }
}

/// Debug: show target `name` (its current copy, times `scale`) on the screen after the final pass; an empty name stops it.
@_cdecl("mmc_ext_debug_view")
public func mmc_ext_debug_view(_ name: UnsafePointer<CChar>, _ scale: Float) {
    let n = String(cString: name)
    ExtPipe.view = n.isEmpty ? nil : (n, scale)
}

/// 1 if the description wants the first-person hand in its G-buffer this frame (`gbuffers.hand`).
@_cdecl("mmc_ext_hand")
public func mmc_ext_hand() -> Int32 {
    guard let p = ExtPipe.shared else { return 0 }
    return p.frameActive && (p.desc.gbuffers.hand ?? false) ? 1 : 0
}

/// The level is done (safety net for frames whose main pass never came).
@_cdecl("mmc_ext_frame_end")
public func mmc_ext_frame_end() {
    guard let p = ExtPipe.shared, p.frameActive, !extPassActive else { return }
    p.endFrame()
}

// MARK: - Backend.swift hooks

/// mmc_pass_begin: returns nil unless this pass is redirected (then 1 or 0 as mmc_pass_begin would).
func extPassBegin(_ colors: UnsafePointer<Int64>, _ count: Int32) -> Int32? {
    guard let p = ExtPipe.shared, p.pendingRedirect != 0 else { return nil }
    let main: MTLTexture? = count > 0 && colors[0] != 0 ? (from(colors[0]) as TextureBox).texture : nil
    return p.beginGbufferPass(mainColor: main) ? 1 : nil
}

func extPassEnd() {
    autoreleasepool { ExtPipe.shared?.endGbufferPass() }
}

func extSetPipeline(_ p: PipelineBox) -> Int32 {
    ExtPipe.shared?.setPipeline(p) ?? 0
}

func extBindBuffer(_ index: Int32, _ h: Int64, _ offset: Int64) {
    ExtPipe.shared?.bindBuffer(Int(index), (from(h) as BufferBox).buffer, Int(offset))
}

func extBindTexture(_ index: Int32, _ tex: Int64, _ smp: Int64) {
    ExtPipe.shared?.bindTexture(Int(index), (from(tex) as TextureBox).texture, (from(smp) as SamplerBox).state)
}

func extBeforeDraw(_ enc: MTLRenderCommandEncoder) {
    ExtPipe.shared?.beforeDraw(enc)
}

func extVertexBuffer(_ slot: Int32, _ h: Int64, _ offset: Int64) {
    ExtPipe.shared?.vertexBuffers[Int(slot)] = ((from(h) as BufferBox).buffer, Int(offset))
}

func extRecordIndirect(_ buf: MTLBuffer, _ offset: Int, _ count: Int) {
    ExtPipe.shared?.recordIndirect(buf, offset, count)
}

// MARK: - Lod.swift hooks

/// mmc_lod_draw inside the G-buffer pass: the description's LOD pipelines (plain, seam), or nil to draw nothing there.
func extLodPipelines() -> (plain: MTLRenderPipelineState?, seam: MTLRenderPipelineState?)? {
    ExtPipe.shared?.lodPipelines()
}
