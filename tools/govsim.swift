// Offline simulation of the render scale's governor (Sources/MetalMCNative/RenderScale.swift, mmc_debug_governor_sim), no
// GPU: synthetic frames whose GPU time is a fixed part plus a part that scales with the pixels (s^2), with noise, reported
// two frames late as a command buffer completes; uncapped, or at 120 Hz with vsync, with or without a GPU clock that
// slows to fit the work (its reported time stretches toward the frame's). Prints, per scenario, the scales it settled on,
// how often it changed, and the share of frames late.
// usage: swiftc -O tools/govsim.swift -o .build/govsim; .build/govsim .build/release/libMetalMCNative.dylib
import Foundation

let args = CommandLine.arguments
guard args.count >= 2, let lib = dlopen(args[1], RTLD_NOW) else { print("usage: govsim <dylib>"); exit(1) }
typealias SimF = @convention(c) (Double, Double, Double, Int32, Double, Int32, UnsafePointer<Double>, Int32, Int32,
                                 UnsafeMutablePointer<Float>, UnsafeMutablePointer<Double>) -> Double
guard let sym = dlsym(lib, "mmc_debug_governor_sim") else { print("missing mmc_debug_governor_sim"); exit(1) }
let sim = unsafeBitCast(sym, to: SimF.self)

func run(_ name: String, fixed: Double, perPixel: Double, noise: Double = 0.04, vsync: Bool, dvfs: Double = 0,
         phases: [Double] = [1], phaseFrames: Int = 2400) {
    var scales = [Float](repeating: 0, count: phases.count * phaseFrames)
    var out = [Double](repeating: 0, count: 2)
    let late = sim(fixed, perPixel, noise, vsync ? 1 : 0, dvfs, 2, phases, Int32(phases.count), Int32(phaseFrames), &scales, &out)
    // Each phase's scale over its last half (settled), and the scale's range there.
    var settled: [String] = []
    for p in 0..<phases.count {
        let tail = scales[(p * phaseFrames + phaseFrames / 2)..<((p + 1) * phaseFrames)]
        let mean = tail.reduce(0, +) / Float(tail.count)
        settled.append(String(format: "x%.2f: %.2f (%.2f-%.2f)", phases[p], mean, tail.min()!, tail.max()!))
    }
    // The ideal fixed scale for each phase: the largest step with (fixed + perPixel x phase x s^2) under 85% of 8.33 ms.
    print(String(format: "%-40@ late %5.1f%%, %3.0f changes, mean scale %.2f | settled %@", name as NSString, late * 100, out[0], out[1],
                 settled.joined(separator: ", ")))
}

print("GPU time = fixed + perPixel x s^2 ms (the full look at the panel's resolution is about 5 + 9 = 14 ms)")
run("uncapped, fits at 0.75 (3 + 7 s^2)", fixed: 3, perPixel: 7, vsync: false)
run("vsync, fits at 0.75", fixed: 3, perPixel: 7, vsync: true)
run("vsync + clock stretch 0.9, fits at 0.75", fixed: 3, perPixel: 7, vsync: true, dvfs: 0.9)
run("uncapped, never fits (5 + 9 s^2)", fixed: 5, perPixel: 9, vsync: false)
run("vsync, never fits", fixed: 5, perPixel: 9, vsync: true)
run("uncapped, heavy/light views every 20 s", fixed: 3, perPixel: 7, vsync: false, phases: [1.4, 0.6, 1.4, 0.6])
run("vsync, heavy/light views every 20 s", fixed: 3, perPixel: 7, vsync: true, phases: [1.4, 0.6, 1.4, 0.6])
run("vsync + stretch, heavy/light every 20 s", fixed: 3, perPixel: 7, vsync: true, dvfs: 0.9, phases: [1.4, 0.6, 1.4, 0.6])
run("uncapped, noisy (12%)", fixed: 3, perPixel: 7, noise: 0.12, vsync: false)
run("vsync, noisy (12%)", fixed: 3, perPixel: 7, noise: 0.12, vsync: true)
run("uncapped, light (fits at 1)", fixed: 2, perPixel: 4, vsync: false)
run("vsync + stretch, light (fits at 1)", fixed: 2, perPixel: 4, vsync: true, dvfs: 0.9)
