import Foundation
import Metal
import MetalMCCore

// Keeps the LOD current while playing (single-player). The integrated server writes chunks to region files
// when they unload (the player moved away) and on autosave. A background thread notices changed region
// files and rebuilds only the affected nodes and their parents, and meshes the finest level around the
// player's current position.
//
// Per region it caches the level-2 quadrant (128 x 128 x 96 voxels), so parents can be rebuilt without
// re-reading sibling regions from disk.

let lodQuadrantVoxels = lodNodeVoxels / 2

/// A meshed node on the GPU.
final class LodMeshNode {
    let level: Int
    let x0: Int, z0: Int
    let buffer: MTLBuffer
    let quadCount: Int
    let faceStart: [Int]        // 7 prefix offsets of the face buckets +X -X +Y -Y +Z -Z
    var size: Int { lodNodeVoxels << level }

    init?(node: LodNode) {
        guard !node.quads.isEmpty,
              let b = ctx.device.makeBuffer(bytes: node.quads, length: node.quads.count * 4, options: [.storageModeShared]) else { return nil }
        level = node.level
        x0 = node.x0
        z0 = node.z0
        buffer = b
        quadCount = node.quads.count / 2
        var starts = [0]
        for c in node.faceCounts { starts.append(starts.last! + c) }
        faceStart = starts
    }
}

struct LodNodeKey: Hashable {
    let level: Int
    let x: Int
    let z: Int
}

final class LodWorld: @unchecked Sendable {
    let regionDir: URL
    let maxLevel: Int
    let fineRadius: Int

    private let lock = NSLock()
    private var meshes: [LodNodeKey: LodMeshNode] = [:]
    private(set) var generation = 0

    // Update-thread state.
    private var fileStamps: [Int64: (Date, Int)] = [:]
    private var quadrants: [Int64: [UInt8]] = [:]     // region -> level-2 quadrant
    private var center: (x: Int, z: Int)
    private var meshedCenter: (x: Int, z: Int)?
    private var requestedCenter: (x: Int, z: Int)
    private var vanillaRadius = 0                      // blocks; regions this close to the player are drawn by vanilla
    private var deferred: [Int64: String] = [:]        // changed regions waiting until the player leaves them
    private let queue = DispatchQueue(label: "metalmc.lod.update", qos: .utility)
    private var running = true
    var status = "starting"

    init(regionDir: URL, maxLevel: Int, fineRadius: Int, centerX: Int, centerZ: Int) {
        self.regionDir = regionDir
        self.maxLevel = maxLevel
        self.fineRadius = fineRadius
        center = (centerX, centerZ)
        requestedCenter = (centerX, centerZ)
    }

    func snapshot() -> (meshes: [LodNodeKey: LodMeshNode], generation: Int) {
        lock.lock(); defer { lock.unlock() }
        return (meshes, generation)
    }

    func setCenter(x: Int, z: Int, vanillaRadius: Int) {
        lock.lock(); requestedCenter = (x, z); self.vanillaRadius = vanillaRadius; lock.unlock()
    }

    /// True if any part of the region is within `radius` blocks (horizontally) of (cx, cz).
    private func regionNear(_ x: Int, _ z: Int, cx: Int, cz: Int, radius: Int) -> Bool {
        let size = lodNodeVoxels << 1
        let x0 = x * size, z0 = z * size
        let dx = Double(max(x0 - cx, 0, cx - (x0 + size))), dz = Double(max(z0 - cz, 0, cz - (z0 + size)))
        return (dx * dx + dz * dz).squareRoot() <= Double(radius)
    }

    func stop() {
        lock.lock(); running = false; lock.unlock()
    }

    func start() {
        queue.async { [self] in
            while true {
                lock.lock(); let go = running; lock.unlock()
                if !go { return }
                let t0 = Date()
                let did = poll()
                if did { log("LOD: update \(status) in \(String(format: "%.1f", Date().timeIntervalSince(t0))) s") }
                Thread.sleep(forTimeInterval: 2.0)
            }
        }
    }

    // MARK: - Update pass

    private func install(_ key: LodNodeKey, _ node: LodNode?) {
        let mesh = node.flatMap { LodMeshNode(node: $0) }
        lock.lock()
        if let mesh { meshes[key] = mesh } else { meshes[key] = nil }
        generation += 1
        lock.unlock()
    }

    private func dropLevel1(outside cx: Int, _ cz: Int) {
        lock.lock()
        for k in meshes.keys where k.level == 1 && !withinFine(regionX: k.x, regionZ: k.z, cx: cx, cz: cz) {
            meshes[k] = nil
            generation += 1
        }
        lock.unlock()
    }

    private func withinFine(regionX: Int, regionZ: Int, cx: Int, cz: Int) -> Bool {
        let size = lodNodeVoxels << 1
        let x0 = regionX * size, z0 = regionZ * size
        let dx = Double(max(x0 - cx, 0, cx - (x0 + size))), dz = Double(max(z0 - cz, 0, cz - (z0 + size)))
        return (dx * dx + dz * dz).squareRoot() <= Double(fineRadius)
    }

    /// One update pass. Returns true if anything was rebuilt.
    private func poll() -> Bool {
        let files = ((try? FileManager.default.contentsOfDirectory(atPath: regionDir.path)) ?? []).filter { $0.hasSuffix(".mca") }
        var changed: [(Int, Int, String)] = []
        var seen = Set<Int64>()
        for f in files {
            let parts = f.split(separator: ".")
            guard parts.count == 4, let x = Int(parts[1]), let z = Int(parts[2]) else { continue }
            let path = regionDir.appendingPathComponent(f).path
            guard let attrs = try? FileManager.default.attributesOfItem(atPath: path),
                  let mtime = attrs[.modificationDate] as? Date, let size = attrs[.size] as? Int else { continue }
            let k = LodBuild.key(x, z)
            seen.insert(k)
            if let old = fileStamps[k], old.0 == mtime, old.1 == size { continue }
            fileStamps[k] = (mtime, size)
            changed.append((x, z, path))
        }

        lock.lock(); let want = requestedCenter; let near = vanillaRadius; lock.unlock()
        // Regions vanilla is drawing around the player change constantly (autosave). Rebuild them only once
        // the player has moved away; the first pass (no mesh yet) builds everything.
        if meshedCenter != nil && near > 0 {
            var keep: [(Int, Int, String)] = []
            for c in changed {
                if regionNear(c.0, c.1, cx: want.x, cz: want.z, radius: near + 64) {
                    deferred[LodBuild.key(c.0, c.1)] = c.2
                } else {
                    keep.append(c)
                }
            }
            for (k, path) in deferred {
                let (x, z) = LodBuild.unkey(k)
                if !regionNear(x, z, cx: want.x, cz: want.z, radius: near + 64) {
                    keep.append((x, z, path))
                    deferred[k] = nil
                }
            }
            changed = keep
        }
        let moved = meshedCenter.map { abs($0.x - want.x) > 128 || abs($0.z - want.z) > 128 } ?? true
        if changed.isEmpty && !moved { return false }
        center = want

        // 1. Changed regions: level-1 grid -> cached level-2 quadrant, and a level-1 mesh if near the player.
        var changedKeys = Set<Int64>()
        let results = rebuildRegions(changed, meshFine: true)
        for (k, quadrant, node) in results {
            changedKeys.insert(k)
            quadrants[k] = quadrant
            let (x, z) = LodBuild.unkey(k)
            install(LodNodeKey(level: 1, x: x, z: z), node)
        }

        // 2. The player moved: mesh level-1 nodes that came into range (from disk), drop the ones that left.
        if moved {
            var need: [(Int, Int, String)] = []
            snapshotLock: do {
                lock.lock()
                let have = Set(meshes.keys.filter { $0.level == 1 }.map { LodBuild.key($0.x, $0.z) })
                lock.unlock()
                for k in seen where !have.contains(k) && !changedKeys.contains(k) {
                    let (x, z) = LodBuild.unkey(k)
                    if withinFine(regionX: x, regionZ: z, cx: center.x, cz: center.z) {
                        need.append((x, z, regionDir.appendingPathComponent("r.\(x).\(z).mca").path))
                    }
                }
            }
            for (k, _, node) in rebuildRegions(need, meshFine: true) {
                let (x, z) = LodBuild.unkey(k)
                install(LodNodeKey(level: 1, x: x, z: z), node)
            }
            dropLevel1(outside: center.x, center.z)
            meshedCenter = center
        }

        // 3. Parents of changed regions, level by level.
        var dirty = Set(changedKeys.map { LodBuild.unkey($0) }.map { LodNodeKey(level: 1, x: $0.0, z: $0.1) })
        var level = 2
        while level <= maxLevel && !dirty.isEmpty {
            let parents = Array(Set(dirty.map { LodNodeKey(level: level, x: $0.x >> 1, z: $0.z >> 1) }))
            var built = [LodNode?](repeating: nil, count: parents.count)
            built.withUnsafeMutableBufferPointer { out in
                let outp = out.baseAddress!
                DispatchQueue.concurrentPerform(iterations: parents.count) { i in
                    var g = self.grid(level: parents[i].level, x: parents[i].x, z: parents[i].z)
                    g.fillUnreachable()
                    let m = LodBuild.mesh(g, maxMerge: 64)
                    let size = lodNodeVoxels << parents[i].level
                    outp[i] = LodNode(level: parents[i].level, x0: parents[i].x * size, z0: parents[i].z * size, quads: m.quads, faceCounts: m.faceCounts)
                }
            }
            for (i, p) in parents.enumerated() { install(p, built[i]) }
            dirty = Set(parents)
            level += 1
        }
        lock.lock()
        let count = meshes.count
        let quads = meshes.values.reduce(0) { $0 + $1.quadCount }
        lock.unlock()
        status = "\(changed.count) changed regions, moved \(moved): \(count) nodes, \(quads) quads"
        return true
    }

    /// Reads regions in parallel: returns (key, level-2 quadrant, level-1 node if within the fine radius).
    private func rebuildRegions(_ list: [(Int, Int, String)], meshFine: Bool) -> [(Int64, [UInt8], LodNode?)] {
        var out = [(Int64, [UInt8], LodNode?)?](repeating: nil, count: list.count)
        let c = center
        out.withUnsafeMutableBufferPointer { buf in
            let outp = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: list.count) { i in
                let (x, z, path) = list[i]
                guard let g = LodBuild.regionGrid(path: path) else {
                    outp[i] = (LodBuild.key(x, z), [UInt8](repeating: 0, count: lodQuadrantVoxels * lodQuadrantVoxels * (lodWorldHeight >> 2)), nil)
                    return
                }
                var q = LodGrid(level: 2)
                g.downsample(into: &q, qx: 0, qz: 0)   // the quadrant occupies the low corner
                let quadrant = Self.extractQuadrant(q)
                var node: LodNode?
                if meshFine && self.withinFine(regionX: x, regionZ: z, cx: c.x, cz: c.z) {
                    var filled = g
                    filled.fillUnreachable()
                    let m = LodBuild.mesh(filled, maxMerge: 16)
                    let size = lodNodeVoxels << 1
                    node = LodNode(level: 1, x0: x * size, z0: z * size, quads: m.quads, faceCounts: m.faceCounts)
                }
                outp[i] = (LodBuild.key(x, z), quadrant, node)
            }
        }
        return out.compactMap { $0 }
    }

    /// The 128 x 128 x 96 corner of a level-2 grid that one region's downsample fills.
    static func extractQuadrant(_ g: LodGrid) -> [UInt8] {
        let q = lodQuadrantVoxels, h = g.height
        var out = [UInt8](repeating: 0, count: q * q * h)
        g.v.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0..<h {
                    for z in 0..<q {
                        let s = (y * lodNodeVoxels + z) * lodNodeVoxels
                        let d = (y * q + z) * q
                        for x in 0..<q { dst[d + x] = src[s + x] }
                    }
                }
            }
        }
        return out
    }

    /// Dense grid for a node at `level` >= 2, assembled from cached region quadrants (downsampled further
    /// for levels above 2). Missing regions are air.
    func grid(level: Int, x: Int, z: Int) -> LodGrid {
        if level == 2 {
            var g = LodGrid(level: 2)
            let q = lodQuadrantVoxels, h = g.height
            for qz in 0...1 {
                for qx in 0...1 {
                    guard let quad = quadrants[LodBuild.key(2 * x + qx, 2 * z + qz)] else { continue }
                    quad.withUnsafeBufferPointer { src in
                        g.v.withUnsafeMutableBufferPointer { dst in
                            for y in 0..<h {
                                for zz in 0..<q {
                                    let s = (y * q + zz) * q
                                    let d = (y * lodNodeVoxels + qz * q + zz) * lodNodeVoxels + qx * q
                                    for xx in 0..<q { dst[d + xx] = src[s + xx] }
                                }
                            }
                        }
                    }
                }
            }
            return g
        }
        var g = LodGrid(level: level)
        for qz in 0...1 {
            for qx in 0...1 {
                let child = grid(level: level - 1, x: 2 * x + qx, z: 2 * z + qz)
                child.downsample(into: &g, qx: qx, qz: qz)
            }
        }
        return g
    }
}
