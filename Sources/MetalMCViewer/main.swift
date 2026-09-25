import Foundation
import Metal
import QuartzCore
import MetalMCCore

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
    // Defaults verified by the image harness: CCW front faces with culling, reverse-Z depth.
    renderer.cullBackfaces = !args.contains("--no-cull")
    renderer.frontFacing = args.contains("--cw") ? .clockwise : .counterClockwise
    renderer.reverseZ = !args.contains("--standard-z")
    renderer.faceBuckets = !args.contains("--no-buckets")
    Mesher.greedy = !args.contains("--no-greedy")
    renderer.gpuCulling = args.contains("--gpu-cull")

    // Far-terrain LOD rings (procedural worlds only for now).
    let lodLevels = Int(argValue("--lod") ?? "") ?? 0
    var lods: [LODLevel] = []
    if lodLevels > 0 && label == "procedural" {
        let tl = CACurrentMediaTime()
        let vmax = Int(argValue("--lod-vmax") ?? "") ?? 8
        lods = LOD.buildProcedural(nearSize: world.sizeX, height: world.sizeY, levels: lodLevels, maxVScale: vmax)
        let reach = (world.sizeX / 2) << lodLevels
        renderer.fogEnd = Float(reach) * 0.95
        renderer.fogStart = renderer.fogEnd * 0.55
        print(String(format: "LOD levels=%d scales=%@ reach_blocks=%d build_ms=%.0f",
                     lodLevels, lods.map { "\($0.scale)x\($0.vScale)" }.joined(separator: ","), reach,
                     (CACurrentMediaTime() - tl) * 1000))
    }
    renderer.upload(world: world, lods: lods)
    if !lods.isEmpty {
        print("LOD sections_per_tier=\(renderer.sectionsPerTier.map(String.init).joined(separator: ",")) quads_per_tier=\(renderer.quadsPerTier.map(String.init).joined(separator: ","))")
    }
    let t2 = CACurrentMediaTime()

    print(String(format: "SETUP gpu=\"%@\" world=%@ size=%dx%dx%d_blocks load_ms=%.0f mesh_upload_ms=%.0f sections=%d quads=%d gpu_mb=%.1f",
                 device.name, label, world.sizeX, world.sizeY, world.sizeZ, (t1 - t0) * 1000, (t2 - t1) * 1000,
                 renderer.sections.count, renderer.totalQuads, Double(renderer.gpuBytes) / 1_048_576))

    if args.contains("--bench") {
        var cfg = BenchConfig(outDir: URL(fileURLWithPath: argValue("--out") ?? "bench_out"))
        if let n = Int(argValue("--frames") ?? "") { cfg.frames = n }
        if let g = argValue("--golden") { cfg.golden = URL(fileURLWithPath: g) }
        cfg.writeGolden = args.contains("--write-golden")
        if let c = argValue("--compare") { cfg.compareDir = URL(fileURLWithPath: c) }
        cfg.pan = args.contains("--pan") || lodLevels > 0
        print("MODE cull=\(renderer.cullBackfaces) front=\(renderer.frontFacing == .clockwise ? "cw" : "ccw") reverse_z=\(renderer.reverseZ) buckets=\(renderer.faceBuckets) greedy=\(Mesher.greedy) gpu_cull=\(renderer.gpuCulling)")
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
