import AppKit
import Foundation
import Metal
import QuartzCore

// HDR/EDR output (METALMC_EXP=hdr, lighting milestone 1, off by default). The XDR panel can show light brighter than
// its SDR white (its EDR headroom: up to 16 times at low brightness, less as the brightness goes up, its 1600-nit peak
// over the SDR white), so the sun and the sky around it can be brighter than a white block instead of clipping to the
// same white.
//
// - The level renders into a float main target (RGBA16Float, MainTargetMixin) instead of RGBA8. Vanilla's pipelines
//   declare RGBA8 color targets, so the backend builds each one a variant for the float target on first use (PipelineBox
//   in Backend.swift; warmed when the pipeline is created). The values stay sRGB-encoded, like vanilla's, so blending
//   and every shader behave as before; our sky writes values above 1 where it's brighter than SDR white.
// - The tone curve (skyToneMap in Sky.swift) runs where the level becomes display light: in the sky pass for sky
//   pixels, in the aerial perspective step for everything else (the anti-aliasing's resolve, or a pass of its own;
//   Sky.swift), both before the hand and the GUI. It's the identity up to SDR white, so vanilla's colors are unchanged,
//   and rolls brighter light into the display's current headroom. After it the target holds display light: 1 is SDR
//   white, the headroom the brightest the panel shows now.
// - The hand, the screen effects and the GUI draw over that in SDR as before: the GUI's white is SDR white, not
//   boosted, and a translucent panel over the sun dims it like it dims anything.
// - The present decodes to linear light into an RGBA16Float drawable of an EDR CAMetalLayer (extended linear color
//   space, no metadata, no system tone mapping), so 1 is SDR white and values up to the headroom are shown as they are.
// - Screenshots read an 8-bit copy with the frame rolled into SDR (ScreenshotMixin, mmc_hdr_snapshot).
// - The temporal anti-aliasing keeps a float history when the frame is float (Taa.swift).
//
// The headroom is read from the window's screen every 30 frames on the main thread (Minecraft's render thread on
// macOS) and followed over about a quarter second, so brightness changes don't pump the image. It reads 1 until
// something on the screen asks for EDR, then rises as macOS turns it on; with no headroom the curve is the SDR one.
// Without METALMC_EXP=hdr the target and the drawable stay 8-bit and the sky and aerial perspective use the SDR curve
// (headroom 1): the SDR fallback.

/// METALMC_EXP=hdr: float main target, EDR drawable, tone curve into the display's headroom.
let hdrOutput = experiments.contains("hdr")
/// The float main target (and the anti-aliasing's history) as RG11B10Float by default, 32 bits a pixel instead of
/// RGBA16Float's 64: half the bandwidth in every pass that loads or stores it (the real-terrain flight with sky and HDR:
/// 124.7 → 136.3 fps), for 6-bit mantissas (5 in blue) and no alpha, so the sky and the anti-aliasing dither relative
/// to the value. METALMC_HDRFORMAT=rgba16f: RGBA16Float as before. The drawable stays RGBA16Float.
let hdrPacked = hdrOutput && ProcessInfo.processInfo.environment["METALMC_HDRFORMAT"] != "rgba16f"
/// The main target's format with HDR.
let hdrTargetFormat: MTLPixelFormat = hdrPacked ? .rg11b10Float : .rgba16Float
/// Whether a color target holds HDR's float frame.
@inline(__always) func isFloatFrame(_ f: MTLPixelFormat) -> Bool { f == .rgba16Float || f == .rg11b10Float }
/// METALMC_HDRSPACE: the primaries the linear output is tagged with. "srgb" (default): extended linear sRGB, the same
/// primaries as today's SDR layer (setting a CAMetalLayer's pixel format tags it with a matching color space: BGRA8 gets
/// sRGB, checked on macOS 26), so turning HDR on changes brightness only. "p3": extended linear Display P3 (the game's
/// sRGB values on the panel's wider primaries: more saturated).
private let hdrSpace = ProcessInfo.processInfo.environment["METALMC_HDRSPACE"] ?? "srgb"
/// METALMC_HDRPEAK: the most headroom the tone curve uses (default: all the display reports).
private let hdrPeakCap = max(1, Float(ProcessInfo.processInfo.environment["METALMC_HDRPEAK"] ?? "") ?? 64)

private let hdrShaderSource = skyShaderHeader + """

struct HdrVOut { float4 pos [[position]]; };

vertex HdrVOut hdr_fullscreen_vs(uint vid [[vertex_id]]) {
    HdrVOut o;
    float2 c = float2(float((vid << 1) & 2), float(vid & 2));
    o.pos = float4(c * 2.0 - 1.0, 0.0, 1.0);
    return o;
}

// The present: the frame's sRGB-encoded display values (up to the headroom) to linear light for the EDR layer, flipped
// like the SDR present (texture row 0 is the GL bottom row, drawable row 0 the top).
fragment float4 hdr_present_fs(HdrVOut in [[stage_in]], texture2d<float, access::read> src [[texture(0)]],
                               constant uint2& size [[buffer(0)]]) {
    uint2 p = uint2(in.pos.xy);
    return float4(max(skyDecode(src.read(uint2(p.x, size.y - 1 - p.y)).rgb), 0.0), 1.0);
}

// Screenshots: the frame rolled into SDR the way an SDR display would show it (the tone curve at headroom 1).
fragment float4 hdr_snapshot_fs(HdrVOut in [[stage_in]], texture2d<float, access::read> src [[texture(0)]]) {
    float3 c = max(skyDecode(src.read(uint2(in.pos.xy)).rgb), 0.0);
    return float4(skyEncode(skyToneMap(c, float4(1.0, 0.9, 0.0, 0.0))), 1.0);
}

// Offline check of the tone curve (mmc_debug_hdr_curve): scene-linear inputs to display-linear outputs.
kernel void hdr_debug_curve(device const float4* in [[buffer(0)]], device float4* out [[buffer(1)]],
                            constant float4& tone [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    out[i] = float4(skyToneMap(in[i].rgb, tone), 0.0);
}
"""

final class Hdr: @unchecked Sendable {
    static let shared = Hdr()

    /// The headroom the tone curve uses this frame (smoothed; 1 without HDR or without EDR on the screen).
    private(set) var headroom: Float = 1
    private let lock = NSLock()
    private var reported: Float = 1
    private var potential: Float = 1
    private var logged: Float = 0
    private var polls = 0
    private weak var layer: CAMetalLayer?
    private var library: MTLLibrary?
    private var snapshotPipe: MTLRenderPipelineState?
    private var failed = false

    private func ensureLibrary() -> MTLLibrary? {
        if let library { return library }
        if failed { return nil }
        do {
            library = try ctx.device.makeLibrary(source: hdrShaderSource, options: nil)
        } catch {
            log("hdr: shaders failed: \(error)")
            failed = true
        }
        return library
    }

    /// Makes the window's layer an EDR one: float drawables in extended linear light, no system tone mapping.
    func configure(_ layer: CAMetalLayer) {
        let p3 = hdrSpace == "p3"
        layer.pixelFormat = .rgba16Float
        layer.wantsExtendedDynamicRangeContent = true
        layer.colorspace = CGColorSpace(name: p3 ? CGColorSpace.extendedLinearDisplayP3 : CGColorSpace.extendedLinearSRGB)
        layer.edrMetadata = nil
        if #available(macOS 15.0, *) { layer.toneMapMode = .never }
        self.layer = layer
        log("hdr: EDR layer, \(p3 ? "extended linear Display P3" : "extended linear sRGB")")
    }

    /// Once per presented frame: reads the screen's headroom every 30 frames and moves toward it.
    func poll() {
        guard hdrOutput else { return }
        polls += 1
        if polls % 30 == 1 {
            let read = { [self] in
                let screen = (layer?.delegate as? NSView)?.window?.screen ?? NSScreen.main
                let current = Float(screen?.maximumExtendedDynamicRangeColorComponentValue ?? 1)
                let potential = Float(screen?.maximumPotentialExtendedDynamicRangeColorComponentValue ?? 1)
                lock.lock()
                reported = max(current, 1)
                self.potential = potential
                lock.unlock()
            }
            if Thread.isMainThread { read() } else { DispatchQueue.main.async(execute: read) }
        }
        lock.lock()
        let target = min(reported, hdrPeakCap), pot = potential
        lock.unlock()
        headroom += (target - headroom) * 0.08
        if abs(headroom - target) < 0.005 { headroom = target }
        if abs(target - logged) > 0.1 * max(logged, 1) {
            logged = target
            log(String(format: "hdr: EDR headroom %.2f (the display's most: %.1f)", target, pot))
        }
    }

    /// The present's fragment function (the backend's present blit uses it in place of its SDR one).
    func presentFunction() -> MTLFunction? { ensureLibrary()?.makeFunction(name: "hdr_present_fs") }

    /// Encodes an 8-bit SDR copy of `src` (the float main target) into `dst` (RGBA8, same size) for a screenshot.
    func snapshot(src: MTLTexture, dst: MTLTexture) -> Bool {
        guard ctx.pass == nil, dst.width == src.width, dst.height == src.height, let lib = ensureLibrary() else { return false }
        if snapshotPipe == nil {
            let d = MTLRenderPipelineDescriptor()
            d.label = "MetalMC HDR screenshot"
            d.vertexFunction = lib.makeFunction(name: "hdr_fullscreen_vs")
            d.fragmentFunction = lib.makeFunction(name: "hdr_snapshot_fs")
            d.colorAttachments[0].pixelFormat = dst.pixelFormat
            snapshotPipe = try? ctx.device.makeRenderPipelineState(descriptor: d)
        }
        guard let pipe = snapshotPipe else { return false }
        ctx.endBlit()
        let d = MTLRenderPassDescriptor()
        d.colorAttachments[0].texture = dst
        d.colorAttachments[0].loadAction = .dontCare
        d.colorAttachments[0].storeAction = .store
        guard let enc = ctx.ensureCB().makeRenderCommandEncoder(descriptor: d) else { return false }
        enc.setRenderPipelineState(pipe)
        enc.setFragmentTexture(src, index: 0)
        enc.drawPrimitives(type: .triangle, vertexStart: 0, vertexCount: 3)
        enc.endEncoding()
        ctx.statPasses += 1
        return true
    }

    /// Offline check: the tone curve at `headroom` for n scene-linear RGB inputs (4 floats each) into out.
    func debugCurve(_ input: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ n: Int, headroom: Float) -> Bool {
        guard let lib = ensureLibrary(), let f = lib.makeFunction(name: "hdr_debug_curve"),
              let pipe = try? ctx.device.makeComputePipelineState(function: f),
              let ib = ctx.device.makeBuffer(bytes: input, length: n * 16, options: .storageModeShared),
              let ob = ctx.device.makeBuffer(length: n * 16, options: .storageModeShared),
              let cb = ctx.queue.makeCommandBuffer(), let enc = cb.makeComputeCommandEncoder() else { return false }
        var tone = SIMD4<Float>(max(headroom, 1), skyKnee(headroom), 0, 0)
        enc.setComputePipelineState(pipe)
        enc.setBuffer(ib, offset: 0, index: 0)
        enc.setBuffer(ob, offset: 0, index: 1)
        enc.setBytes(&tone, length: 16, index: 2)
        enc.dispatchThreads(MTLSize(width: n, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: min(n, 64), height: 1, depth: 1))
        enc.endEncoding()
        cb.commit()
        cb.waitUntilCompleted()
        memcpy(out, ob.contents(), n * 16)
        return true
    }
}

/// 1 with METALMC_EXP=hdr (Java makes the main target RGBA16Float then).
@_cdecl("mmc_hdr_enabled")
public func mmc_hdr_enabled() -> Int32 { hdrOutput ? (hdrPacked ? 2 : 1) : 0 }

/// The headroom the tone curve uses this frame (1 without HDR).
@_cdecl("mmc_hdr_headroom")
public func mmc_hdr_headroom() -> Float { Hdr.shared.headroom }

/// Screenshots with HDR: an 8-bit SDR copy of the float main target `src` into `dst` (RGBA8, same size), encoded
/// before the screenshot's own copy. Returns 1 if encoded.
@_cdecl("mmc_hdr_snapshot")
public func mmc_hdr_snapshot(_ src: Int64, _ dst: Int64) -> Int32 {
    Hdr.shared.snapshot(src: (from(src) as TextureBox).texture, dst: (from(dst) as TextureBox).texture) ? 1 : 0
}

/// Offline check (tools, no game): the tone curve for n inputs of 4 floats (rgb, unused) at `headroom`.
@_cdecl("mmc_debug_hdr_curve")
public func mmc_debug_hdr_curve(_ input: UnsafePointer<Float>, _ out: UnsafeMutablePointer<Float>, _ n: Int32, _ headroom: Float) -> Int32 {
    Hdr.shared.debugCurve(input, out, Int(n), headroom: headroom) ? 1 : 0
}
