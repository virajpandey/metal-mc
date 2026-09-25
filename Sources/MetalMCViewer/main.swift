import Foundation
import Metal
import QuartzCore

let args = CommandLine.arguments

func argValue(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}

do {
    guard let device = MTLCreateSystemDefaultDevice() else {
        print("ERROR no Metal device")
        exit(1)
    }
    let chunks = Int(argValue("--chunks") ?? "") ?? 32

    if let path = argValue("--dump") {
        try Anvil.dump(path: path)
        exit(0)
    }

    let t0 = CACurrentMediaTime()
    let world: World
    let label: String
    if let path = argValue("--world") {
        world = try Anvil.load(path: path)
        label = URL(fileURLWithPath: path).lastPathComponent
    } else {
        world = World.procedural(chunksX: chunks, chunksZ: chunks)
        label = "procedural"
    }
    let t1 = CACurrentMediaTime()
    let renderer = try Renderer(device: device)
    renderer.upload(world: world)
    let t2 = CACurrentMediaTime()

    print(String(format: "SETUP gpu=\"%@\" world=%@ size=%dx%dx%d_blocks load_ms=%.0f mesh_upload_ms=%.0f sections=%d quads=%d gpu_mb=%.1f",
                 device.name, label, world.sizeX, world.sizeY, world.sizeZ, (t1 - t0) * 1000, (t2 - t1) * 1000,
                 renderer.sections.count, renderer.totalQuads, Double(renderer.gpuBytes) / 1_048_576))

    if args.contains("--bench") {
        var cfg = BenchConfig(outDir: URL(fileURLWithPath: argValue("--out") ?? "bench_out"))
        if let n = Int(argValue("--frames") ?? "") { cfg.frames = n }
        try Bench.run(renderer: renderer, world: world, cfg: cfg)
        exit(0)
    }

    MainActor.assumeIsolated {
        runInteractive(renderer: renderer, world: world)
    }
} catch {
    print("ERROR \(error)")
    exit(1)
}
