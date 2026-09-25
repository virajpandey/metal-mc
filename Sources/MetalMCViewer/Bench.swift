import CoreGraphics
import CryptoKit
import Foundation
import ImageIO
import Metal
import QuartzCore
import simd
import UniformTypeIdentifiers

struct BenchConfig {
    var width = 1280
    var height = 720
    var frames = 600
    var warmup = 30
    var captures = 5
    var outDir: URL
}

enum BenchError: Error {
    case resource(String)
}

/// Offscreen benchmark: flies a fixed camera path (with a "teleport" every 60 frames),
/// writes per-frame telemetry to frames.csv, saves a few PNG captures, and prints one summary line.
enum Bench {
    static func cameraPose(frame: Int, world: World) -> (eye: SIMD3<Float>, target: SIMD3<Float>) {
        let cx = Float(world.sizeX) / 2, cz = Float(world.sizeZ) / 2
        let segment = frame / 60
        let t = Float(frame % 60) / 60
        let angle = Float(segment) * 2.39996 + t * 0.5
        let radius = min(cx, cz) * (0.3 + 0.5 * Float((segment * 7) % 5) / 4)
        let ref = Float(world.referenceY)
        let eye = SIMD3<Float>(cx + radius * cos(angle), ref + 40 + 20 * sin(Float(segment)), cz + radius * sin(angle))
        return (eye, SIMD3<Float>(cx, ref, cz))
    }

    static func run(renderer: Renderer, world: World, cfg: BenchConfig) throws {
        let dev = renderer.device
        let cd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: renderer.colorFormat,
                                                          width: cfg.width, height: cfg.height, mipmapped: false)
        cd.usage = [.renderTarget]
        cd.storageMode = .private
        let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: renderer.depthFormat,
                                                          width: cfg.width, height: cfg.height, mipmapped: false)
        dd.usage = [.renderTarget]
        dd.storageMode = .private
        let bytesPerRow = cfg.width * 4
        guard let color = dev.makeTexture(descriptor: cd),
              let depth = dev.makeTexture(descriptor: dd),
              let readback = dev.makeBuffer(length: bytesPerRow * cfg.height, options: .storageModeShared)
        else { throw BenchError.resource("offscreen targets") }

        try FileManager.default.createDirectory(at: cfg.outDir, withIntermediateDirectories: true)

        let total = cfg.warmup + cfg.frames
        let captureFrames = Set((0..<cfg.captures).map { cfg.warmup + 30 + ($0 * cfg.frames) / max(1, cfg.captures) })
        let proj = perspectiveRH(fovyRadians: 70 * .pi / 180,
                                 aspect: Float(cfg.width) / Float(cfg.height), near: 0.1, far: 1000)

        var rows = ["frame,cpu_ms,gpu_ms,sections_drawn,sections_culled,triangles"]
        var cpu: [Double] = [], gpu: [Double] = [], tris: [Double] = [], culledPct: [Double] = []
        var holes: [Double] = []
        var hashes: [String] = []

        for f in 0..<total {
            let pose = cameraPose(frame: f, world: world)
            let viewProj = proj * lookAtRH(eye: pose.eye, center: pose.target, up: SIMD3(0, 1, 0))

            let t0 = CACurrentMediaTime()
            guard let cb = renderer.queue.makeCommandBuffer() else { continue }
            let st = renderer.encode(into: cb, pass: renderer.makePass(color: color, depth: depth),
                                     viewProj: viewProj, cameraPos: pose.eye)
            let capture = captureFrames.contains(f)
            if capture, let blit = cb.makeBlitCommandEncoder() {
                blit.copy(from: color, sourceSlice: 0, sourceLevel: 0,
                          sourceOrigin: MTLOrigin(x: 0, y: 0, z: 0),
                          sourceSize: MTLSize(width: cfg.width, height: cfg.height, depth: 1),
                          to: readback, destinationOffset: 0,
                          destinationBytesPerRow: bytesPerRow, destinationBytesPerImage: bytesPerRow * cfg.height)
                blit.endEncoding()
            }
            cb.commit()
            let cpuMs = (CACurrentMediaTime() - t0) * 1000
            cb.waitUntilCompleted()
            let gpuMs = (cb.gpuEndTime - cb.gpuStartTime) * 1000

            if f >= cfg.warmup {
                rows.append(String(format: "%d,%.4f,%.4f,%d,%d,%d", f, cpuMs, gpuMs, st.drawn, st.culled, st.triangles))
                cpu.append(cpuMs)
                gpu.append(gpuMs)
                tris.append(Double(st.triangles))
                let totalSections = max(1, st.drawn + st.culled)
                culledPct.append(100 * Double(st.culled) / Double(totalSections))
            }
            if capture {
                holes.append(100 * holeFraction(buffer: readback, width: cfg.width, height: cfg.height, sky: renderer.sky))
                let digest = SHA256.hash(data: Data(bytes: readback.contents(), count: bytesPerRow * cfg.height))
                hashes.append(String(digest.map { String(format: "%02x", $0) }.joined().prefix(12)))
                let url = cfg.outDir.appendingPathComponent(String(format: "frame_%04d.png", f))
                try writePNG(buffer: readback, width: cfg.width, height: cfg.height, url: url)
            }
        }

        try rows.joined(separator: "\n").write(to: cfg.outDir.appendingPathComponent("frames.csv"),
                                               atomically: true, encoding: .utf8)

        let summary = String(format:
            "BENCH frames=%d res=%dx%d cpu_ms_mean=%.3f cpu_ms_p99=%.3f gpu_ms_mean=%.3f gpu_ms_p99=%.3f gpu_ms_max=%.3f tris_mean=%.0f culled_pct_mean=%.1f hole_pct_max=%.2f captures=%d out=%@",
            cfg.frames, cfg.width, cfg.height,
            mean(cpu), percentile(cpu, 99), mean(gpu), percentile(gpu, 99), gpu.max() ?? 0,
            mean(tris), mean(culledPct), holes.max() ?? 0, holes.count, cfg.outDir.path)
            + " hashes=" + hashes.joined(separator: ",")
        print(summary)
        try (summary + "\n").write(to: cfg.outDir.appendingPathComponent("summary.txt"), atomically: true, encoding: .utf8)
    }

    /// Share of lower-half pixels that still match the sky clear color. The camera always looks
    /// down at terrain, so a high value means missing geometry (holes, inverted faces, bad depth).
    static func holeFraction(buffer: MTLBuffer, width: Int, height: Int, sky: SIMD4<Float>) -> Double {
        let p = buffer.contents().assumingMemoryBound(to: UInt8.self)
        let sr = Int(sky.x * 255 + 0.5), sg = Int(sky.y * 255 + 0.5), sb = Int(sky.z * 255 + 0.5)
        var hits = 0, total = 0
        for y in (height / 2)..<height {
            for x in stride(from: 0, to: width, by: 2) {
                let o = (y * width + x) * 4
                let b = Int(p[o]), g = Int(p[o + 1]), r = Int(p[o + 2])
                if abs(r - sr) <= 2 && abs(g - sg) <= 2 && abs(b - sb) <= 2 { hits += 1 }
                total += 1
            }
        }
        return Double(hits) / Double(max(1, total))
    }

    static func writePNG(buffer: MTLBuffer, width: Int, height: Int, url: URL) throws {
        let info = CGImageAlphaInfo.noneSkipFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        guard let ctx = CGContext(data: buffer.contents(), width: width, height: height, bitsPerComponent: 8,
                                  bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: info),
              let image = ctx.makeImage(),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw BenchError.resource("png \(url.lastPathComponent)") }
        CGImageDestinationAddImage(dest, image, nil)
        if !CGImageDestinationFinalize(dest) { throw BenchError.resource("png finalize") }
    }
}
