import Foundation
import Metal

// Render scale and its governor (docs/frame-budget.md, "Render scale and the governor").
//
// The level (vanilla's terrain and entities, our near chunks, the LOD, the far field, the G-buffer, the relight, the
// shadows, the GI cache's per-pixel work, the water, the sky's aerial perspective) is drawn at a fraction of the panel's
// resolution on each axis and resolved to the panel's by the anti-aliasing (TaaUpscale.swift); the hand, the HUD and
// post-processing stay at the panel's resolution. The Java side (metalmc.render.RenderScale) asks for the scale at the
// start of every frame (mmc_render_scale) and draws the level into a target of that size.
//
// METALMC_SCALE (gradle -PrenderScale, config renderScale), read once:
//   - a number from 0.5 to 1: that scale, always (1, the default: the panel's resolution, no upscaling);
//   - "auto": the governor steers the scale to hold the refresh rate; "auto:MIN:MAX" bounds it (default 0.6 to 1).
//
// The governor. Each frame's GPU time (its command buffer's gpuStartTime..gpuEndTime; one submit per frame) is recorded
// with the scale the frame was drawn at, and only frames drawn at the current scale count; so are the frame intervals
// (between its calls, once a frame). Over the last 32 of them:
//   - over budget, uncapped: the GPU time's 90th percentile above 92% of the frame time (7.67 ms at 120 Hz); the scale
//     drops to what would bring it to 85% (7.08 ms) if all of it scaled with the pixels (it doesn't, so this overshoots
//     less than it seems and a second step follows if needed), on a grid of 0.05;
//   - over budget, paced by the display (vsync: the intervals' median at the refresh), where the GPU slows its clock to fit
//     the work and its time stretches toward the frame's whatever the load: two of the last 32 frames late (4% past the
//     refresh) with the GPU busy (over 80%; else it's the CPU, which the scale can't help): one step down (two when a
//     quarter are late);
//   - under budget, the 90th percentile low enough that one step up (0.05) would still keep it under 85% by the same
//     model, for 1 s without a break: one step up. That rule is the hysteresis: a step up never predicts a step down.
//     Paced, the stretched GPU time may never allow that, so after a quiet spell (no late frame for 3 s) the governor
//     tries a step up and comes back if frames come late, waiting twice as long before the next try (up to 32 s), and 3 s
//     again after a try that held (the offline simulation, tools/govsim.swift, shows the ratchet without it).
//   - at the lowest scale and still over budget, the per-feature quality (`quality`, 1 to 0 in quarters) drops: the GI
//     cache's updates a frame first (its light converges more slowly; nothing pops). It comes back before the scale does.
// A change waits for 32 frames drawn at the new scale before the next. Through the anti-aliasing a step of the scale is
// a slight change of sharpness over a few frames (the history is kept at the panel's resolution), not a pop.
// METALMC_GOVBUDGET: the frame time to hold in ms (default 8.333, 120 Hz).

struct RenderScaleSetting {
    var fixed: Float = 1
    var auto = false
    var min: Float = 0.6
    var max: Float = 1

    static func parse(_ s: String?) -> RenderScaleSetting {
        var r = RenderScaleSetting()
        guard let s = s?.trimmingCharacters(in: .whitespaces).lowercased(), !s.isEmpty else { return r }
        if s.hasPrefix("auto") {
            r.auto = true
            let parts = s.split(separator: ":").dropFirst().compactMap { Float($0) }
            if parts.count >= 1 { r.min = Swift.min(Swift.max(parts[0], 0.5), 1) }
            if parts.count >= 2 { r.max = Swift.min(Swift.max(parts[1], r.min), 1) }
            r.fixed = r.max
        } else if let v = Float(s), v.isFinite {
            r.fixed = Swift.min(Swift.max(v, 0.5), 1)
        }
        return r
    }
}

let renderScaleSetting = RenderScaleSetting.parse(ProcessInfo.processInfo.environment["METALMC_SCALE"])

/// The drawable's width in pixels (the panel's resolution, which the upscaled level, the hand and the HUD are at), from the
/// surface's configuration (mmc_surface2_configure); 0 before it.
nonisolated(unsafe) var renderOutputWidth = 0

/// Texture detail under a render scale: the level's pass is narrower than the output, so texture coordinates' screen
/// derivatives are larger by the ratio and the GPU would pick mips that much blurrier (0.42 of a level at 0.75), detail no
/// upscaler gets back. Our terrain shaders (the near chunks, the LOD) multiply their gradients by this, the pass's width
/// over the output's (1 without a scale), so they choose mips as if drawn at the output's resolution: a mip bias of
/// log2(scale), what temporal upscalers ask of the renderer (the jitter then resolves the finer texels over frames).
/// METALMC_EXP=nomipbias leaves it at 1.
func renderMipScale() -> Float {
    guard !renderNoMipBias, renderOutputWidth > 0, ctx.passWidth > 0, ctx.passWidth < renderOutputWidth else { return 1 }
    return max(0.5, Float(ctx.passWidth) / Float(renderOutputWidth))
}
private let renderNoMipBias = experiments.contains("nomipbias")

final class RenderScaleGovernor: @unchecked Sendable {
    static let shared = RenderScaleGovernor()

    let budgetMs: Double
    static let step: Float = 0.05
    static let window = 32
    /// The offline simulation's instances don't log.
    private let quiet: Bool

    private let lock = NSLock()
    /// The setting in force: METALMC_SCALE, or in lab mode <shader dir>/render_scale.txt (same syntax, read every 2 s), so
    /// one session can compare scales.
    private(set) var setting: RenderScaleSetting
    private var labChecks = 0
    private var labText: String?
    /// Completed frames' GPU times (ms) drawn at the current scale (completion handlers, any thread).
    private var samples: [Double] = []
    /// The last completed command buffer's GPU end (seconds), for each frame's own time.
    private var lastGpuEnd: Double = 0
    /// The scale handed out for the frame being recorded (render thread), and with which every submit is tagged.
    private(set) var scale: Float
    /// Per-feature quality, 1 (full) to 0, in quarters: what the governor gives up after the scale.
    private(set) var quality: Float = 1
    private var underSince: UInt64 = 0
    /// Frame intervals (ms, render thread: from one call of next() to the next) since the last change.
    private var intervals: [Double] = []
    private var lastFrame: UInt64 = 0
    /// The last late frame, the last change, and the paced mode's probes: a step up tried after `probeWait` seconds without
    /// a late frame (3, doubling to 32 after each that brought late frames, back to 3 after one that held).
    private var lastLate: UInt64 = 0, lastChange: UInt64 = 0
    private var probing = false
    private var probeWait = 3.0
    private var frames = 0
    private var changes = 0
    private(set) var totalChanges = 0
    /// Frames drawn at each scale step (index: scale / step), for the summary.
    private var histogram = [Int](repeating: 0, count: 21)
    private var lastLog: UInt64 = 0

    init(setting s: RenderScaleSetting = renderScaleSetting,
         budgetMs: Double = Double(ProcessInfo.processInfo.environment["METALMC_GOVBUDGET"] ?? "") ?? 1000.0 / 120.0, quiet: Bool = false) {
        setting = s
        self.budgetMs = budgetMs
        self.quiet = quiet
        // The governor starts in the middle of its range: a heavy view at the panel's resolution would drop frames for
        // the half second its first step takes.
        scale = s.auto ? RenderScaleGovernor.quantize(Swift.min(s.max, Swift.max(s.min, 0.8))) : s.fixed
    }

    static func quantize(_ s: Float) -> Float { (s / step).rounded() * step }

    /// The submit's tag: the scale its frame was drawn at.
    var tag: Float { scale }

    /// Completion handler of a submit (any thread): its command buffer's GPU start and end (seconds) and the scale it was
    /// drawn at. Frames overlap on the GPU (the next one's vertex work starts under this one's fragment work), so a command
    /// buffer's span counts some of its neighbors' work: 19 ms spans at 12 ms frames in game. The frame's own GPU time is
    /// what it added after the frame before it ended: end - max(start, the previous end).
    func gpuFrame(start: Double, end: Double, tag: Float) {
        lock.lock()
        let own = end - Swift.max(start, lastGpuEnd)
        lastGpuEnd = Swift.max(lastGpuEnd, end)
        lock.unlock()
        gpuFrame(Swift.max(own, 0), tag: tag)
    }

    /// A frame's own GPU time (seconds) and the scale it was drawn at (the offline simulation calls this directly).
    func gpuFrame(_ seconds: Double, tag: Float) {
        lock.lock()
        if setting.auto && tag == scale {
            samples.append(seconds * 1000)
            if samples.count > RenderScaleGovernor.window { samples.removeFirst(samples.count - RenderScaleGovernor.window) }
        }
        lock.unlock()
    }

    /// Render thread, once a frame (mmc_render_scale): the scale to draw this frame's level at.
    func next() -> Float {
        labOverride()
        return next(now: DispatchTime.now().uptimeNanoseconds)
    }

    /// next() at time `now` (nanoseconds; the offline simulation's clock).
    func next(now: UInt64) -> Float {
        let s = setting
        frames += 1
        histogram[Swift.min(20, Int((scale / RenderScaleGovernor.step).rounded()))] += 1
        guard s.auto else { return scale }
        let lateMs = 1.04 * budgetMs
        if lastFrame != 0 {
            let iv = Double(now - lastFrame) / 1e6
            intervals.append(iv)
            if intervals.count > RenderScaleGovernor.window { intervals.removeFirst(intervals.count - RenderScaleGovernor.window) }
            if iv > lateMs { lastLate = now }
        }
        lastFrame = now
        lock.lock()
        let window = samples
        lock.unlock()
        defer { logSummary(now) }
        guard window.count >= RenderScaleGovernor.window, intervals.count >= RenderScaleGovernor.window / 2 else { underSince = 0; return scale }
        let p90 = window.sorted()[Int(0.9 * Double(window.count - 1))]
        let high = 0.92 * budgetMs, target = 0.85 * budgetMs
        // Paced by the display (vsync: most frames take one refresh), the GPU slows its clock to the work and its time
        // stretches toward the frame's, so it says little about overload: there, too much is frames actually late (two of
        // the last 32 past the refresh) while the GPU is busy (not the CPU's hitches). Uncapped, the GPU runs flat out and
        // its time is the measure.
        let iv = intervals.sorted()
        let intP50 = iv[iv.count / 2], intP90 = iv[Int(0.9 * Double(iv.count - 1))]
        let paced = abs(intP50 - budgetMs) < 0.03 * budgetMs
        let late = intervals.reduce(0) { $0 + ($1 > lateMs ? 1 : 0) }
        let over = paced ? late >= 2 && p90 > 0.8 * budgetMs : p90 > high
        let why = String(format: "GPU p90 %.2f ms, frame interval p50 %.2f p90 %.2f ms, %d late, %@", p90, intP50, intP90, late,
                         paced ? "paced" : "not paced")
        if over {
            underSince = 0
            // A probe that brought late frames: back down, and twice as long before the next.
            if probing { probeWait = Swift.min(probeWait * 2, 32) }
            probing = false
            if scale > s.min + 1e-3 {
                let next: Float
                if paced {
                    // The GPU's time is stretched by its clock here: one step, two when a quarter of the frames are late.
                    next = scale - RenderScaleGovernor.step * (late >= 8 ? 2 : 1)
                } else {
                    // All of the GPU's time scaling with the pixels: the scale that brings p90 to the target.
                    let want = scale * Float((target / p90).squareRoot())
                    next = Swift.min(scale - RenderScaleGovernor.step, (want / RenderScaleGovernor.step).rounded(.down) * RenderScaleGovernor.step)
                }
                change(scale: RenderScaleGovernor.quantize(Swift.max(s.min, next)), quality: quality, why: why, now: now)
            } else if quality > 0 {
                change(scale: scale, quality: Swift.max(0, quality - 0.25), why: why, now: now)
            }
            return scale
        }
        // A probe that held for a second: back to the short wait.
        if probing && now - lastChange >= 1_000_000_000 {
            probing = false
            probeWait = 3
        }
        guard quality < 1 || scale < s.max - 1e-3 else { underSince = 0; return scale }
        // One step up (quality first): would p90 stay under the target if all of it scaled with the pixels? After 1 s of it.
        let up = Swift.min(s.max, scale + RenderScaleGovernor.step)
        let grow = quality < 1 ? 1.25 : Double((up / scale) * (up / scale))
        if p90 * grow < target {
            if underSince == 0 { underSince = now }
            if now - underSince >= 1_000_000_000 {
                underSince = 0
                stepUp(up, why: why, now: now)
            }
            return scale
        }
        underSince = 0
        // Paced, the GPU's time says little (its clock follows the work), so the model may never let it up: after a while
        // with no late frame, try a step up, and come back if frames come late (twice as long a wait before the next try).
        let wait = UInt64(probeWait * 1e9)
        if paced && !probing && now - lastLate >= wait && now - lastChange >= wait {
            probing = true
            stepUp(up, why: why + ", a probe", now: now)
        }
        return scale
    }

    private func stepUp(_ up: Float, why: String, now: UInt64) {
        if quality < 1 { change(scale: scale, quality: Swift.min(1, quality + 0.25), why: why, now: now) }
        else { change(scale: RenderScaleGovernor.quantize(up), quality: quality, why: why, now: now) }
    }

    /// Lab mode: <shader dir>/render_scale.txt, read every 240 frames, overrides METALMC_SCALE while it's there; and
    /// quadcull.txt (1 or 0) switches the LOD's quad culling (Lod.swift, METALMC_EXP=quadcull), to trace it against the
    /// render scale in one session.
    private func labOverride() {
        guard let dir = ShaderLab.dir else { return }
        labChecks += 1
        guard labChecks % 240 == 1 else { return }
        if let q = (try? String(contentsOfFile: dir + "/quadcull.txt", encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines),
           q == "1" || q == "0", lodQuadCullOn != (q == "1") {
            lodQuadCullOn = q == "1"
            log("render scale: lab override: the LOD's quad culling \(lodQuadCullOn ? "on" : "off")")
        }
        let text = (try? String(contentsOfFile: dir + "/render_scale.txt", encoding: .utf8))?.trimmingCharacters(in: .whitespacesAndNewlines)
        guard text != labText else { return }
        labText = text
        let next = text.map { RenderScaleSetting.parse($0) } ?? renderScaleSetting
        let old = scale
        lock.lock()
        setting = next
        if !next.auto { scale = next.fixed } else { scale = Swift.min(next.max, Swift.max(next.min, RenderScaleGovernor.quantize(scale))) }
        samples.removeAll(keepingCapacity: true)
        lock.unlock()
        underSince = 0
        log(String(format: "render scale: lab override %@ (%@), scale %.2f -> %.2f", text ?? "removed", next.auto ? "the governor" : "fixed", old, scale))
    }

    private func change(scale next: Float, quality q: Float, why: String, now: UInt64) {
        let old = scale, oldQ = quality
        lastChange = now
        lock.lock()
        scale = next
        quality = q
        samples.removeAll(keepingCapacity: true)
        lock.unlock()
        intervals.removeAll(keepingCapacity: true)
        changes += 1
        totalChanges += 1
        if quiet { return }
        if q != oldQ { applyQuality() }
        log(String(format: "render scale: %.2f -> %.2f, quality %.2f -> %.2f (%@; budget %.2f ms)", old, next, oldQ, q, why, budgetMs))
    }

    /// The per-feature quality knobs: the GI cache's cell updates a frame (a quarter of the setting at quality 0).
    private func applyQuality() {
        if litGi, let gi = GiCache.shared {
            gi.budget = Swift.max(1024, Int(Double(giBudget) * Double(0.25 + 0.75 * quality)))
        }
    }

    private func logSummary(_ now: UInt64) {
        if quiet { return }
        if lastLog == 0 { lastLog = now }
        guard now - lastLog >= 20_000_000_000 else { return }
        lastLog = now
        let total = Swift.max(1, histogram.reduce(0, +))
        let rows = histogram.enumerated().filter { $0.element > 0 }.map {
            String(format: "%.2f: %.0f%%", Float($0.offset) * RenderScaleGovernor.step, 100 * Double($0.element) / Double(total))
        }
        log("render scale: now \(String(format: "%.2f", scale)), quality \(String(format: "%.2f", quality)), \(changes) changes; frames at each scale over the last 20 s: \(rows.joined(separator: ", "))")
        histogram = [Int](repeating: 0, count: 21)
        changes = 0
    }
}

/// Offline simulation of the governor (tools/govsim.swift), no GPU: a frame's GPU time is (fixed + perPixel x s^2) x
/// (1 + noise x a normal deviate), s the scale it was drawn at, reported `latency` frames later as a completed command
/// buffer would be. Uncapped (vsync 0) a frame takes its GPU time; with vsync (1) a whole number of refreshes, and with
/// `dvfs` (vsync only) the GPU's reported time stretches toward the frame as a clock that follows the work would make it
/// (to `dvfs` times the frames it fills, never less than the work). `phases`: per-pixel cost multipliers held for
/// `phaseFrames` frames each (a heavy view, a light one, ...). Writes each frame's scale to `scales` and returns the share
/// of frames late (over the refresh) in [0, 1]; out[0] the number of changes, out[1] the mean scale.
@_cdecl("mmc_debug_governor_sim")
public func mmc_debug_governor_sim(_ fixedMs: Double, _ perPixelMs: Double, _ noise: Double, _ vsync: Int32, _ dvfs: Double,
                                   _ latency: Int32, _ phases: UnsafePointer<Double>, _ phaseCount: Int32, _ phaseFrames: Int32,
                                   _ scales: UnsafeMutablePointer<Float>, _ out: UnsafeMutablePointer<Double>) -> Double {
    let gov = RenderScaleGovernor(setting: RenderScaleSetting.parse("auto"), budgetMs: 1000.0 / 120.0, quiet: true)
    var rng = SystemRandomNumberGenerator()
    func gauss() -> Double {
        let u1 = Double.random(in: 1e-12..<1, using: &rng), u2 = Double.random(in: 0..<1, using: &rng)
        return (-2 * log(u1)).squareRoot() * cos(2 * Double.pi * u2)
    }
    let refresh = 1000.0 / 120.0
    var now: UInt64 = 1_000_000_000
    var pending: [(ms: Double, tag: Float)] = []
    var late = 0, sum = 0.0
    let total = Int(phaseCount) * Int(phaseFrames)
    for i in 0..<total {
        let s = gov.next(now: now)
        scales[i] = s
        sum += Double(s)
        let work = max(0.5, (fixedMs + perPixelMs * phases[i / Int(phaseFrames)] * Double(s * s)) * (1 + noise * gauss()))
        var frameMs = work, reported = work
        if vsync != 0 {
            let refreshes = max(1.0, (work / refresh).rounded(.up))
            frameMs = refreshes * refresh
            if dvfs > 0 { reported = max(work, min(dvfs * frameMs, frameMs)) }
        }
        if frameMs > refresh * 1.01 { late += 1 }
        pending.append((reported, gov.tag))
        if pending.count > Int(latency) {
            let f = pending.removeFirst()
            gov.gpuFrame(f.ms / 1000, tag: f.tag)
        }
        now += UInt64(frameMs * 1e6)
    }
    out[0] = Double(gov.totalChanges)
    out[1] = sum / Double(max(1, total))
    return Double(late) / Double(max(1, total))
}

/// Offline check (tools/taautest.swift): the shaders the render scale's mip bias is in compile under this METALMC_EXP: the
/// near chunks' library and every LOD pipeline for a main pass's formats (lit mode's with its G-buffer). Returns how many
/// failed (each logged).
@_cdecl("mmc_debug_terrain_compile")
public func mmc_debug_terrain_compile() -> Int32 {
    var failed: Int32 = 0
    let opts = MTLCompileOptions()
    opts.languageVersion = .version3_0
    opts.preserveInvariance = true
    do { _ = try ctx.device.makeLibrary(source: nearShaderSource, options: opts) } catch {
        log("near chunks: shaders failed: \(error)")
        failed += 1
    }
    LodRenderer.shared.lastFormats = (litEnabled ? [.rg11b10Float, litGbufferFormat] : [.bgra8Unorm], .depth32Float)
    let n = mmc_debug_lod_pipelines()
    if n != 0 { failed += max(n, 1) }
    return failed
}

/// The render scale for the frame that starts now (Java's metalmc.render.RenderScale, at the start of every frame): a
/// fixed METALMC_SCALE, or the governor's with METALMC_SCALE=auto. 1: the panel's resolution.
@_cdecl("mmc_render_scale")
public func mmc_render_scale() -> Float {
    RenderScaleGovernor.shared.next()
}
