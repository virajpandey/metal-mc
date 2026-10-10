import Foundation

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
//   - over budget: uncapped, the GPU time's 90th percentile above 92% of the frame time (7.67 ms at 120 Hz); paced by
//     the display (vsync, the intervals' median at the refresh), where the GPU slows its clock to fit the work and its
//     time stretches toward the frame's, the intervals' 90th percentile past the refresh by 4% (frames actually late)
//     with the GPU busy (over 80%; else the CPU is late, which the scale can't help). The scale drops to what would bring
//     the GPU time to 85% (7.08 ms) if all of it scaled with the pixels (it doesn't, so this overshoots less than it seems
//     and a second step follows if needed), on a grid of 0.05;
//   - under budget, the 90th percentile low enough that one step up (0.05) would still keep it under 85% by the same
//     rule, for 1 s without a break: one step up. That rule is the hysteresis: a step up never predicts a step down.
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

final class RenderScaleGovernor: @unchecked Sendable {
    static let shared = RenderScaleGovernor()

    let budgetMs: Double = Double(ProcessInfo.processInfo.environment["METALMC_GOVBUDGET"] ?? "") ?? 1000.0 / 120.0
    static let step: Float = 0.05
    static let window = 32

    private let lock = NSLock()
    /// The setting in force: METALMC_SCALE, or in lab mode <shader dir>/render_scale.txt (same syntax, read every 2 s), so
    /// one session can compare scales.
    private(set) var setting = renderScaleSetting
    private var labChecks = 0
    private var labText: String?
    /// Completed frames' GPU times (ms) drawn at the current scale (completion handlers, any thread).
    private var samples: [Double] = []
    /// The scale handed out for the frame being recorded (render thread), and with which every submit is tagged.
    private(set) var scale: Float
    /// Per-feature quality, 1 (full) to 0, in quarters: what the governor gives up after the scale.
    private(set) var quality: Float = 1
    private var underSince: UInt64 = 0
    /// Frame intervals (ms, render thread: from one call of next() to the next) since the last change.
    private var intervals: [Double] = []
    private var lastFrame: UInt64 = 0
    private var frames = 0
    private var changes = 0
    /// Frames drawn at each scale step (index: scale / step), for the summary.
    private var histogram = [Int](repeating: 0, count: 21)
    private var lastLog: UInt64 = 0

    init() {
        let s = renderScaleSetting
        // The governor starts in the middle of its range: a heavy view at the panel's resolution would drop frames for
        // the half second its first step takes.
        scale = s.auto ? RenderScaleGovernor.quantize(Swift.min(s.max, Swift.max(s.min, 0.8))) : s.fixed
    }

    static func quantize(_ s: Float) -> Float { (s / step).rounded() * step }

    /// The submit's tag: the scale its frame was drawn at.
    var tag: Float { scale }

    /// Completion handler of a submit (any thread): its GPU time and the scale it was drawn at.
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
        let s = setting
        frames += 1
        histogram[Swift.min(20, Int((scale / RenderScaleGovernor.step).rounded()))] += 1
        guard s.auto else { return scale }
        let now = DispatchTime.now().uptimeNanoseconds
        if lastFrame != 0 {
            intervals.append(Double(now - lastFrame) / 1e6)
            if intervals.count > RenderScaleGovernor.window { intervals.removeFirst(intervals.count - RenderScaleGovernor.window) }
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
        // stretches toward the frame's, so it says little about overload: there, too much is frames actually late (the
        // 90th percentile past the refresh) while the GPU is busy (not the CPU's hitches). Uncapped, the GPU runs flat
        // out and its time is the measure.
        let iv = intervals.sorted()
        let intP50 = iv[iv.count / 2], intP90 = iv[Int(0.9 * Double(iv.count - 1))]
        let paced = abs(intP50 - budgetMs) < 0.03 * budgetMs
        let over = paced ? intP90 > 1.04 * budgetMs && p90 > 0.8 * budgetMs : p90 > high
        let why = String(format: "GPU p90 %.2f ms, frame interval p50 %.2f p90 %.2f ms, %@", p90, intP50, intP90, paced ? "paced" : "not paced")
        if over {
            underSince = 0
            if scale > s.min + 1e-3 {
                // All of the GPU's time scaling with the pixels: the scale that brings p90 to the target.
                let want = scale * Float((target / p90).squareRoot())
                let next = Swift.max(s.min, Swift.min(scale - RenderScaleGovernor.step,
                                                      (want / RenderScaleGovernor.step).rounded(.down) * RenderScaleGovernor.step))
                change(scale: RenderScaleGovernor.quantize(next), quality: quality, why: why)
            } else if quality > 0 {
                change(scale: scale, quality: Swift.max(0, quality - 0.25), why: why)
            }
            return scale
        }
        // One step up (quality first): would p90 stay under the target if all of it scaled with the pixels?
        let up = Swift.min(s.max, scale + RenderScaleGovernor.step)
        let grow = quality < 1 ? 1.25 : Double((up / scale) * (up / scale))
        if (quality < 1 || up > scale + 1e-3) && p90 * grow < target {
            if underSince == 0 { underSince = now }
            if now - underSince >= 1_000_000_000 {
                underSince = 0
                if quality < 1 { change(scale: scale, quality: Swift.min(1, quality + 0.25), why: why) }
                else { change(scale: RenderScaleGovernor.quantize(up), quality: quality, why: why) }
            }
        } else {
            underSince = 0
        }
        return scale
    }

    /// Lab mode: <shader dir>/render_scale.txt, read every 240 frames, overrides METALMC_SCALE while it's there.
    private func labOverride() {
        guard let dir = ShaderLab.dir else { return }
        labChecks += 1
        guard labChecks % 240 == 1 else { return }
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

    private func change(scale next: Float, quality q: Float, why: String) {
        let old = scale, oldQ = quality
        lock.lock()
        scale = next
        quality = q
        samples.removeAll(keepingCapacity: true)
        lock.unlock()
        intervals.removeAll(keepingCapacity: true)
        changes += 1
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

/// The render scale for the frame that starts now (Java's metalmc.render.RenderScale, at the start of every frame): a
/// fixed METALMC_SCALE, or the governor's with METALMC_SCALE=auto. 1: the panel's resolution.
@_cdecl("mmc_render_scale")
public func mmc_render_scale() -> Float {
    RenderScaleGovernor.shared.next()
}
