import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

// Lab mode's `mask LABEL` (docs/lab-mode.md, "The class mask"; Java: metalmc.bench.Lab, metalmc.backend.MetalLab): a
// picture the size of the frame in which every pixel is one flat color for what it shows, worked out from lit mode's
// G-buffer, the depth and the traced sun visibility. A `shot` and a `mask` with the same label line up pixel for pixel
// (the camera holds still in lab mode; the G-buffer's sub-pixel jitter aside), so tools/twindiff.swift can compare our
// frame with its twin from another renderer class by class.
//
// The request runs over the next `frames` frames of the relight (Lit.relight calls frame()): each adds the traced sun
// visibility of the pixel's sample into a float target (the samples rotate over the pixels of each 4 x 4 block, so 16
// frames see them all), and the last classifies every pixel into a byte buffer the CPU turns into the PNG, off the
// render thread. Nothing runs without a request.
//
// The classes, first rule that holds:
//   sky         nothing drawn (depth 0), or vanilla's cloud slab (4 blocks from the cloud height)
//   water       the G-buffer's water flag (the LOD's and the far field's water), or a surface at a source block's height
//               (8/9 of a block) where the G-buffer has no terrain or something nearer was drawn (vanilla's water)
//   unlit       anything else that isn't lit terrain: entities, the hand, particles, glass
//   far         lit terrain past 512 blocks
//   blocklit    lit terrain with block light 8 or more where the sky light is under 10 or the sun is down, and light
//               sources (block light 15)
//   vegetation  faces that aren't axis-aligned (cross plants) and green albedo (leaves, grass, vines): the albedo-hue rule,
//               until the G-buffer flags foliage
//   sunlit      lit terrain whose sun visibility (the mean of the frames' samples) is over 0.5 and whose face is turned to
//               the sun, by day
//   shade       the rest of lit terrain

/// The classes in the order of their codes, with the colors the PNG gets (tools/twindiff.swift reads them back).
let labMaskClasses: [(name: String, r: UInt8, g: UInt8, b: UInt8)] = [
    ("sky", 60, 120, 220), ("water", 0, 200, 200), ("vegetation", 40, 170, 40), ("sunlit", 255, 215, 60),
    ("shade", 120, 70, 180), ("blocklit", 255, 120, 20), ("far", 150, 150, 150), ("unlit", 220, 30, 90)]

private let labMaskSource = """
#include <metal_stdlib>
using namespace metal;

#define LIT_MODE 1
\(litShaderHeader)
// Lab mode's class mask (LabMask.swift; docs/lab-mode.md, "The class mask"). The codes, in the order of labMaskClasses:
#define MASK_SKY 0u
#define MASK_WATER 1u
#define MASK_VEGETATION 2u
#define MASK_SUNLIT 3u
#define MASK_SHADE 4u
#define MASK_BLOCKLIT 5u
#define MASK_FAR 6u
#define MASK_UNLIT 7u
// The rules: lit terrain past MASK_FAR_BLOCKS is far; block light MASK_BLOCK_LEVEL or more where the sky light is under
// MASK_SKY_LOW (or at night) is block-lit; a surface MASK_WATER_TOP of the way up a block (+- MASK_WATER_TOL) where no lit
// terrain shows is vanilla's water (a source block's surface).
#define MASK_FAR_BLOCKS 512.0
#define MASK_BLOCK_LEVEL 8.0
#define MASK_SKY_LOW 10.0
#define MASK_WATER_TOP 0.8889
#define MASK_WATER_TOL 0.02

struct MaskFrame {
    float4x4 invViewProj;   // clip space to camera-relative world (the jittered projection the level was drawn with)
    float4 size;            // xy: the target's size, z: pixels per traced visibility sample (0: no rays), w: 1 if the rays went toward the moon
    float4 sun;             // xyz: toward the sun, w: the cloud slab's bottom, camera-relative (blocks)
    float4 cam;             // y: the camera's height (world)
    float4 step;            // x: 1 on the first frame (the sum starts over), y: 1 on the last (classify), z: frames so far, this one too
};

constant float3 kMaskNormal[6] = { float3(1, 0, 0), float3(-1, 0, 0), float3(0, 1, 0), float3(0, -1, 0), float3(0, 0, 1), float3(0, 0, -1) };

// Vegetation until the G-buffer flags foliage: faces that aren't axis-aligned (cross plants: grass, flowers, ferns, crops)
// and green albedo (leaves, grass blocks' tops, vines): green the largest channel, clearly above blue, and saturated.
static bool maskVegetation(uint code, float3 a) {
    if (code == 7u) return true;
    float mx = max(a.r, max(a.g, a.b)), mn = min(a.r, min(a.g, a.b));
    return a.g >= a.r && a.g > a.b * 1.1 && mx > 0.0 && (mx - mn) / mx > 0.2;
}

kernel void lab_mask(depth2d<float, access::read> depth [[texture(0)]], texture2d<uint, access::read> gbuf [[texture(1)]],
                     texture2d<half, access::read> vis [[texture(2)]], texture2d<float, access::read_write> acc [[texture(3)]],
                     constant MaskFrame& f [[buffer(0)]], device uchar* out [[buffer(1)]], uint2 q [[thread_position_in_grid]]) {
    uint W = uint(f.size.x), H = uint(f.size.y);
    if (q.x >= W || q.y >= H) return;
    float v = 1.0;
    if (f.size.z > 0.0) {
        uint s = uint(f.size.z);
        v = float(vis.read(min(q / s, uint2(vis.get_width() - 1, vis.get_height() - 1))).r);
    }
    float sum = (f.step.x > 0.5 ? 0.0 : acc.read(q).r) + v;
    if (f.step.y < 0.5) {
        acc.write(float4(sum, 0.0, 0.0, 0.0), q);
        return;
    }
    float d = depth.read(q);
    uint2 g = gbuf.read(q).rg;
    uint code = g.x >> 29;
    uint cls;
    if (d <= 0.0) {
        cls = MASK_SKY;   // reverse-Z: nothing drawn
    } else {
        float2 ndc = (float2(q) + 0.5) / f.size.xy * 2.0 - 1.0;
        float4 h = f.invViewProj * float4(ndc, d, 1.0);
        float3 rel = h.xyz / h.w;
        bool shows = litDepthMatches(d, g.y & 0xFFFFu);
        if (code == 0u || !shows) {
            float y = rel.y + f.cam.y;
            if (shows && (g.x >> 28) == 1u) cls = MASK_WATER;                                  // flagged water (lit,water)
            else if (rel.y > f.sun.w - 0.1 && rel.y < f.sun.w + 4.1) cls = MASK_SKY;           // vanilla's clouds
            else if (abs(y - floor(y) - MASK_WATER_TOP) < MASK_WATER_TOL) cls = MASK_WATER;    // vanilla's water
            else cls = MASK_UNLIT;
        } else {
            float3 albedo = float3(float(g.x & 255u), float((g.x >> 8) & 255u), float((g.x >> 16) & 255u)) / 255.0;
            float sky = float((g.y >> 16) & 255u) / 16.0, block = float(g.y >> 24) / 16.0;
            uint fi = code - 1u;
            if (length(rel) > MASK_FAR_BLOCKS) {
                cls = MASK_FAR;
            } else if (block >= 14.97 || (block >= MASK_BLOCK_LEVEL && (sky < MASK_SKY_LOW || f.sun.y < 0.0))) {
                cls = MASK_BLOCKLIT;
            } else if (maskVegetation(code, albedo)) {
                cls = MASK_VEGETATION;
            } else {
                // The sun's visibility as the relight has it: the traced samples' mean, or without rays the open sky.
                float vSun = f.size.z > 0.0 ? sum / max(f.step.z, 1.0) * saturate(sky / 2.0) : saturate((sky - 12.0) / 3.0);
                float ndl = fi < 6u ? dot(kMaskNormal[fi], f.sun.xyz) : 0.25 + 0.5 * f.sun.y;
                cls = f.sun.y > 0.0 && f.size.w < 0.5 && vSun > 0.5 && ndl > 0.0 ? MASK_SUNLIT : MASK_SHADE;
            }
        }
    }
    out[(H - 1u - q.y) * W + q.x] = uchar(cls);   // rows top-down, as the PNG has them (targets hold GL's bottom row first)
}
"""

/// Mirrors MaskFrame in the shader.
private struct LabMaskFrameGPU {
    var invViewProj = matrix_identity_float4x4
    var size = SIMD4<Float>.zero
    var sun = SIMD4<Float>.zero
    var cam = SIMD4<Float>.zero
    var step = SIMD4<Float>.zero
}

final class LabMask: @unchecked Sendable {
    static let shared = LabMask()

    private let lock = NSLock()
    /// The request being worked on: the PNG's path, how many frames, the cloud slab's bottom (world y), frames done.
    private var request: (path: String, frames: Int, cloudHeight: Float, done: Int)?
    /// 2 idle, 0 working, 1 written, -1 failed; the summary (class shares, or the error).
    private var state: Int32 = 2
    private var summary = ""
    private var pipe: MTLComputePipelineState?
    private var failed = false
    private var acc: MTLTexture?
    private var out: MTLBuffer?

    /// Read on the render thread every relight: true while a request takes frames.
    private(set) var pending = false

    /// Starts a request (the render thread, between frames). False if lit mode is off.
    func start(path: String, frames: Int, cloudHeight: Float) -> Bool {
        guard litEnabled else { return false }
        lock.lock(); defer { lock.unlock() }
        request = (path, max(frames, 1), cloudHeight, 0)
        state = 0
        summary = "waiting for the relight"
        pending = true
        return true
    }

    func status() -> (Int32, String) {
        lock.lock(); defer { lock.unlock() }
        return (state, summary)
    }

    private func finish(_ s: Int32, _ text: String) {
        lock.lock(); state = s; summary = text; lock.unlock()
        log("lab mask: " + text)
    }

    private func ensurePipe() -> MTLComputePipelineState? {
        if let pipe { return pipe }
        if failed { return nil }
        do {
            // Lab mode (ShaderLab.swift): lab_mask.metal can be edited live; the next request builds the kernel again.
            let lib = try ShaderLab.library("lab_mask", labMaskSource) { [self] _ in pipe = nil; failed = false }
            guard let fn = lib.makeFunction(name: "lab_mask") else { throw NSError(domain: "metalmc", code: 1) }
            pipe = try ctx.device.makeComputePipelineState(function: fn)
        } catch {
            failed = true
            finish(-1, "the mask's kernel failed: \(error)")
            request = nil
            pending = false
        }
        return pipe
    }

    /// One frame of the request, from Lit.relight (render thread) with this frame's command buffer, the G-buffer, the
    /// depth, the traced visibility (one sample per `visScale` x `visScale` pixels; 0 when there were no rays), whether it
    /// went toward the moon, the jittered projection's inverse and the direction toward the sun.
    func frame(cb: MTLCommandBuffer, gbuffer: MTLTexture, depth: MTLTexture, vis: MTLTexture, visScale: Float, moon: Bool,
               invViewProj: simd_float4x4, sun: SIMD3<Float>) {
        lock.lock()
        guard var r = request else { pending = false; lock.unlock(); return }
        lock.unlock()
        guard let pipe = ensurePipe() else { return }
        let w = gbuffer.width, h = gbuffer.height
        if acc == nil || acc!.width != w || acc!.height != h {
            let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: w, height: h, mipmapped: false)
            d.usage = [.shaderRead, .shaderWrite]
            d.storageMode = .private
            acc = ctx.device.makeTexture(descriptor: d)
            acc?.label = "MetalMC lab mask visibility"
        }
        if out == nil || out!.length != w * h { out = ctx.device.makeBuffer(length: w * h, options: .storageModeShared) }
        guard let acc, let out, let enc = cb.makeComputeCommandEncoder() else { return }
        r.done += 1
        let last = r.done >= r.frames
        let cam = LodRenderer.shared.lastCamera
        let camY = cam.y.isFinite ? Float(cam.y) : 0
        var f = LabMaskFrameGPU()
        f.invViewProj = invViewProj
        f.size = SIMD4(Float(w), Float(h), visScale, moon ? 1 : 0)
        f.sun = SIMD4(sun, r.cloudHeight - camY)
        f.cam = SIMD4(Float(cam.x.isFinite ? cam.x : 0), camY, Float(cam.z.isFinite ? cam.z : 0), 0)
        f.step = SIMD4(r.done == 1 ? 1 : 0, last ? 1 : 0, Float(r.done), 0)
        enc.label = "MetalMC lab mask"
        enc.setComputePipelineState(pipe)
        enc.setTexture(depth, index: 0)
        enc.setTexture(gbuffer, index: 1)
        enc.setTexture(vis, index: 2)
        enc.setTexture(acc, index: 3)
        enc.setBytes(&f, length: MemoryLayout<LabMaskFrameGPU>.stride, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        lock.lock()
        if last {
            request = nil
            pending = false
        } else {
            request = r
        }
        lock.unlock()
        guard last else { return }
        let path = r.path, frames = r.frames
        cb.addCompletedHandler { [self] done in
            if let e = done.error {
                finish(-1, "the frame's command buffer failed: \(e)")
                return
            }
            DispatchQueue.global(qos: .utility).async { [self] in write(out, width: w, height: h, path: path, frames: frames) }
        }
    }

    /// The class codes into the PNG's colors, the shares counted, the file written.
    private func write(_ buffer: MTLBuffer, width w: Int, height h: Int, path: String, frames: Int) {
        let codes = buffer.contents().bindMemory(to: UInt8.self, capacity: w * h)
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        var counts = [Int](repeating: 0, count: labMaskClasses.count)
        let lut = labMaskClasses.map { [$0.r, $0.g, $0.b] }
        rgb.withUnsafeMutableBufferPointer { o in
            for i in 0..<(w * h) {
                let c = Int(min(codes[i], UInt8(labMaskClasses.count - 1)))
                counts[c] += 1
                let p = lut[c]
                o[3 * i] = p[0]; o[3 * i + 1] = p[1]; o[3 * i + 2] = p[2]
            }
        }
        let url = URL(fileURLWithPath: path)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        guard let provider = CGDataProvider(data: Data(rgb) as CFData),
              let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: w * 3,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil) else {
            finish(-1, "can't make \(path)")
            return
        }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else {
            finish(-1, "can't write \(path)")
            return
        }
        let shares = zip(labMaskClasses, counts).map { String(format: "%@ %.1f%%", $0.0.name, 100 * Double($0.1) / Double(w * h)) }
        finish(1, "\(path) (\(w)x\(h), \(frames) frames): " + shares.joined(separator: ", "))
    }
}

/// Lab mode's `mask`: classifies the next `frames` relit frames' pixels into the PNG at `path` (see LabMask). cloudHeight:
/// the bottom of vanilla's cloud slab (world y). Returns 1 if it was taken (lit mode is on), 0 if not.
@_cdecl("mmc_lab_mask")
public func mmc_lab_mask(_ path: UnsafePointer<CChar>, _ frames: Int32, _ cloudHeight: Float) -> Int32 {
    LabMask.shared.start(path: String(cString: path), frames: Int(frames), cloudHeight: cloudHeight) ? 1 : 0
}

/// Debug: the mask kernel's source, for an offline compile check (tools/litflow.swift's compile mode). Returns its length.
@_cdecl("mmc_debug_lab_mask_shader_source")
public func mmc_debug_lab_mask_shader_source(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let bytes = Array(labMaskSource.utf8)
    guard bytes.count < Int(len) else { return Int32(bytes.count) }
    for (i, b) in bytes.enumerated() { out[i] = CChar(bitPattern: b) }
    out[bytes.count] = 0
    return Int32(bytes.count)
}

/// The mask's state: 2 idle, 0 working, 1 written, -1 failed; its summary (the class shares, or the error) into `out`.
@_cdecl("mmc_lab_mask_status")
public func mmc_lab_mask_status(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let (s, text) = LabMask.shared.status()
    writeError(text, out, len)
    return s
}
