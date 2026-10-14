import CoreGraphics
import Foundation
import ImageIO
import Metal
import simd

// Lab mode's `burst LABEL [N] [SAVE]` (docs/lab-mode.md, "The flicker meter"; Java: metalmc.bench.Lab, metalmc.backend.
// MetalLab): what a screenshot can't show. The next N frames the present blit sends to the screen (mmc_surface2_blit, so
// everything: the level, the anti-aliasing's resolve, the post, the sky) are read as they go out, and every pixel's
// perceptual luma (0-255, the frame's stored values: gamma-encoded in SDR and in the HDR frame alike) is tracked through
// them:
//
//   d1  the mean |l(t) - l(t-1)|: any change from frame to frame, flicker and smooth motion (waves, swaying leaves) alike
//   d2  the mean |l(t) - 2 l(t-1) + l(t-2)|: change that reverses. Motion that's smooth over a few frames has almost none
//       (a ripple moving 1 level a frame: d1 1, d2 ~0); a pixel that alternates between two values has twice its d1
//       (a/b/a/b: d1 = |a-b|, d2 = 2|a-b|). So d2 is the flicker, d2/d1 how much of the change is flicker (2 at most).
//
// The camera holds still in lab mode, time and weather are frozen, so with the world's own animation the only motion,
// what d2 sees is the renderer's: temporal noise the anti-aliasing doesn't settle, a jitter it doesn't remove, a pass
// that alternates between frames.
//
// Out (next to `shot`'s screenshots): LABEL-flicker.png (d2 over the dimmed mean frame: blue from 1 level, yellow at 4,
// red from 12), LABEL-motion.png (d1, likewise), LABEL-flicker.bin (the raw numbers, for tools/flickstat.swift: "MMCF",
// width, height, frames as UInt32, then per pixel 4 Float16 mean, standard deviation, d1, d2; rows top-down) and, with
// SAVE, the first SAVE frames as LABEL-fNN.png (an 8-bit copy, clamped to SDR). The summary line has the frame's mean d1
// and d2, d2's 99th percentile and the share of pixels whose d2 is over 2 and over 6 levels.

private let flickerSource = """
#include <metal_stdlib>
using namespace metal;

struct FlickerFrame {
    uint4 size;   // x, y: the frame's size; z: frames so far, this one too (1 on the first); w: 1 on the last
    uint4 save;   // x: 1 if this frame is copied out, y: the copy's index
};

// acc: sum, sum of squares, sum of |d1|, sum of |d2|; hist: l(t-1), l(t-2), 0, 0.
kernel void flicker_frame(texture2d<float, access::read> src [[texture(0)]],
                          texture2d<float, access::read_write> acc [[texture(1)]],
                          texture2d<float, access::read_write> hist [[texture(2)]],
                          constant FlickerFrame& f [[buffer(0)]], device half4* out [[buffer(1)]],
                          device uchar4* copy [[buffer(2)]], uint2 q [[thread_position_in_grid]]) {
    uint W = f.size.x, H = f.size.y;
    if (q.x >= W || q.y >= H) return;
    float4 c = src.read(q);
    float l = 255.0 * saturate(dot(c.rgb, float3(0.2126, 0.7152, 0.0722)));
    uint n = f.size.z;
    float4 a = n == 1u ? float4(0.0) : acc.read(q);
    float4 h = n == 1u ? float4(l, l, 0.0, 0.0) : hist.read(q);
    a.x += l;
    a.y += l * l;
    if (n >= 2u) a.z += abs(l - h.x);
    if (n >= 3u) a.w += abs(l - 2.0 * h.x + h.y);
    acc.write(a, q);
    hist.write(float4(l, h.x, 0.0, 0.0), q);
    uint row = (H - 1u - q.y) * W + q.x;   // rows top-down, as the PNG has them (targets hold GL's bottom row first)
    if (f.save.x != 0u) copy[f.save.y * W * H + row] = uchar4(uchar3(round(255.0 * saturate(c.rgb))), 255);
    if (f.size.w == 0u) return;
    float N = float(n), mean = a.x / N;
    float sd = sqrt(max(a.y / N - mean * mean, 0.0));
    out[row] = half4(half(mean), half(sd), half(a.z / max(N - 1.0, 1.0)), half(a.w / max(N - 2.0, 1.0)));
}
"""

private struct FlickerFrameGPU {
    var size = SIMD4<UInt32>.zero
    var save = SIMD4<UInt32>.zero
}

final class Flicker: @unchecked Sendable {
    static let shared = Flicker()

    private let lock = NSLock()
    /// The request being worked on: the files' path without the suffixes, how many frames, how many to save, frames done.
    private var request: (base: String, frames: Int, save: Int, done: Int)?
    /// 2 idle, 0 working, 1 written, -1 failed; the summary (the numbers, or the error).
    private var state: Int32 = 2
    private var summary = ""
    private var pipe: MTLComputePipelineState?
    private var failed = false
    private var acc: MTLTexture?
    private var hist: MTLTexture?
    private var out: MTLBuffer?
    private var copies: MTLBuffer?

    /// Read on the render thread at every present: true while a request takes frames.
    private(set) var pending = false

    /// Starts a request: the next `frames` presented frames (at least 3), the first `save` of them (at most 16) kept.
    func start(base: String, frames: Int, save: Int) {
        lock.lock(); defer { lock.unlock() }
        request = (base, max(frames, 3), min(max(save, 0), 16, max(frames, 3)), 0)
        state = 0
        summary = "waiting for the present"
        pending = true
    }

    func status() -> (Int32, String) {
        lock.lock(); defer { lock.unlock() }
        return (state, summary)
    }

    private func finish(_ s: Int32, _ text: String) {
        lock.lock(); state = s; summary = text; lock.unlock()
        log("lab burst: " + text)
    }

    private func ensurePipe() -> MTLComputePipelineState? {
        if let pipe { return pipe }
        if failed { return nil }
        do {
            let lib = try ShaderLab.library("flicker", flickerSource) { [self] _ in pipe = nil; failed = false }
            guard let fn = lib.makeFunction(name: "flicker_frame") else { throw NSError(domain: "metalmc", code: 1) }
            pipe = try ctx.device.makeComputePipelineState(function: fn)
        } catch {
            failed = true
            lock.lock(); request = nil; pending = false; lock.unlock()
            finish(-1, "the burst's kernel failed: \(error)")
        }
        return pipe
    }

    private func target(_ t: MTLTexture?, _ w: Int, _ h: Int, _ label: String) -> MTLTexture? {
        if let t, t.width == w, t.height == h { return t }
        let d = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba32Float, width: w, height: h, mipmapped: false)
        d.usage = [.shaderRead, .shaderWrite]
        d.storageMode = .private
        let n = ctx.device.makeTexture(descriptor: d)
        n?.label = label
        return n
    }

    /// One presented frame, from the present blit (render thread, outside any pass) with this frame's command buffer and
    /// the texture it shows.
    func frame(cb: MTLCommandBuffer, src: MTLTexture) {
        lock.lock()
        guard var r = request else { pending = false; lock.unlock(); return }
        lock.unlock()
        guard let pipe = ensurePipe() else { return }
        let w = src.width, h = src.height
        if r.done > 0, let acc, acc.width != w || acc.height != h {
            lock.lock(); request = nil; pending = false; lock.unlock()
            finish(-1, "the frame's size changed mid-burst")
            return
        }
        acc = target(acc, w, h, "MetalMC burst sums")
        hist = target(hist, w, h, "MetalMC burst history")
        if out == nil || out!.length != w * h * 8 { out = ctx.device.makeBuffer(length: w * h * 8, options: .storageModeShared) }
        let copyLen = max(r.save, 1) * w * h * 4
        if r.done == 0 { copies = r.save > 0 ? ctx.device.makeBuffer(length: copyLen, options: .storageModeShared) : out }
        guard let acc, let hist, let out, let copies, let enc = cb.makeComputeCommandEncoder() else { return }
        r.done += 1
        let last = r.done >= r.frames
        var f = FlickerFrameGPU()
        f.size = SIMD4(UInt32(w), UInt32(h), UInt32(r.done), last ? 1 : 0)
        f.save = SIMD4(r.done <= r.save ? 1 : 0, UInt32(r.done - 1), 0, 0)
        enc.label = "MetalMC lab burst"
        enc.setComputePipelineState(pipe)
        enc.setTexture(src, index: 0)
        enc.setTexture(acc, index: 1)
        enc.setTexture(hist, index: 2)
        enc.setBytes(&f, length: MemoryLayout<FlickerFrameGPU>.stride, index: 0)
        enc.setBuffer(out, offset: 0, index: 1)
        enc.setBuffer(copies, offset: 0, index: 2)
        enc.dispatchThreads(MTLSize(width: w, height: h, depth: 1), threadsPerThreadgroup: MTLSize(width: 16, height: 16, depth: 1))
        enc.endEncoding()
        lock.lock()
        if last { request = nil; pending = false } else { request = r }
        lock.unlock()
        guard last else { return }
        let base = r.base, frames = r.frames, save = r.save
        let kept = copies
        cb.addCompletedHandler { [self] done in
            if let e = done.error {
                finish(-1, "the frame's command buffer failed: \(e)")
                return
            }
            DispatchQueue.global(qos: .utility).async { [self] in
                write(out, kept, width: w, height: h, base: base, frames: frames, save: save)
                lock.lock(); if request == nil { self.copies = nil }; lock.unlock()
            }
        }
    }

    /// The heat maps, the raw numbers, the kept frames and the summary.
    private func write(_ buffer: MTLBuffer, _ kept: MTLBuffer, width w: Int, height h: Int, base: String, frames: Int, save: Int) {
        let px = buffer.contents().bindMemory(to: Float16.self, capacity: w * h * 4)
        let url = URL(fileURLWithPath: base)
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        // The raw numbers.
        var head = Data("MMCF".utf8)
        for v in [UInt32(w), UInt32(h), UInt32(frames)] { withUnsafeBytes(of: v.littleEndian) { head.append(contentsOf: $0) } }
        head.append(Data(bytes: buffer.contents(), count: w * h * 8))
        guard (try? head.write(to: URL(fileURLWithPath: base + "-flicker.bin"))) != nil else {
            finish(-1, "can't write \(base)-flicker.bin")
            return
        }
        // The heat maps: a value over the mean frame at a third of its brightness.
        func heat(_ ch: Int, _ path: String) -> Bool {
            var rgb = [UInt8](repeating: 0, count: w * h * 3)
            rgb.withUnsafeMutableBufferPointer { o in
                for i in 0..<(w * h) {
                    let m = Float(px[4 * i]) / 3, v = Float(px[4 * i + ch])
                    var c = SIMD3<Float>(m, m, m)
                    if v >= 1 {
                        // 1 -> blue, 4 -> yellow, 12 -> red.
                        let t = min((v - 1) / 11, 1)
                        let ramp: SIMD3<Float> = t < 0.27 ? mix(SIMD3<Float>(40, 90, 255), SIMD3<Float>(255, 230, 40), t: SIMD3<Float>(repeating: t / 0.27))
                                                          : mix(SIMD3<Float>(255, 230, 40), SIMD3<Float>(255, 30, 20), t: SIMD3<Float>(repeating: (t - 0.27) / 0.73))
                        c = mix(c, ramp, t: SIMD3<Float>(repeating: min(0.35 + t, 1)))
                    }
                    o[3 * i] = UInt8(min(c.x, 255)); o[3 * i + 1] = UInt8(min(c.y, 255)); o[3 * i + 2] = UInt8(min(c.z, 255))
                }
            }
            return writePNG(rgb, w, h, 3, path)
        }
        guard heat(3, base + "-flicker.png"), heat(2, base + "-motion.png") else {
            finish(-1, "can't write the heat maps at \(base)")
            return
        }
        // The kept frames.
        if save > 0 {
            let all = kept.contents().bindMemory(to: UInt8.self, capacity: save * w * h * 4)
            for k in 0..<save {
                let frame = [UInt8](UnsafeBufferPointer(start: all + k * w * h * 4, count: w * h * 4))
                _ = writePNG(frame, w, h, 4, base + String(format: "-f%02d.png", k))
            }
        }
        // The summary.
        var d1 = 0.0, d2 = 0.0, over2 = 0, over6 = 0
        var hist = [Int](repeating: 0, count: 1024)   // d2 in quarter levels
        for i in 0..<(w * h) {
            let a = Double(px[4 * i + 2]), b = Double(px[4 * i + 3])
            d1 += a; d2 += b
            if b > 2 { over2 += 1 }
            if b > 6 { over6 += 1 }
            hist[min(Int(b * 4), 1023)] += 1
        }
        var p99 = 0.0, seen = 0
        for (k, c) in hist.enumerated() {
            seen += c
            if seen >= (w * h * 99) / 100 { p99 = Double(k) / 4; break }
        }
        let n = Double(w * h)
        finish(1, String(format: "%@ (%dx%d, %d frames): d1 %.3f, d2 %.3f, d2 p99 %.2f, d2>2 %.2f%%, d2>6 %.2f%%",
                         base, w, h, frames, d1 / n, d2 / n, p99, 100 * Double(over2) / n, 100 * Double(over6) / n))
    }

    private func writePNG(_ bytes: [UInt8], _ w: Int, _ h: Int, _ comps: Int, _ path: String) -> Bool {
        let info = comps == 4 ? CGImageAlphaInfo.noneSkipLast.rawValue : CGImageAlphaInfo.none.rawValue
        guard let provider = CGDataProvider(data: Data(bytes) as CFData),
              let img = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 8 * comps, bytesPerRow: w * comps,
                                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: info),
                                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(URL(fileURLWithPath: path) as CFURL, "public.png" as CFString, 1, nil)
        else { return false }
        CGImageDestinationAddImage(dest, img, nil)
        return CGImageDestinationFinalize(dest)
    }
}

/// Lab mode's `burst`: the next `frames` presented frames' flicker into `<base>-flicker.png` and the rest (see Flicker).
@_cdecl("mmc_lab_burst")
public func mmc_lab_burst(_ base: UnsafePointer<CChar>, _ frames: Int32, _ save: Int32) -> Int32 {
    Flicker.shared.start(base: String(cString: base), frames: Int(frames), save: Int(save))
    return 1
}

/// The burst's state: 2 idle, 0 working, 1 written, -1 failed; its summary (the numbers, or the error) into `out`.
@_cdecl("mmc_lab_burst_status")
public func mmc_lab_burst_status(_ out: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    let (s, text) = Flicker.shared.status()
    writeError(text, out, len)
    return s
}
