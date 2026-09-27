import Compression
import Foundation
import Metal
import MetalMCCore

// Keeps the LOD current while playing. Two sources feed it: region files (single-player; the integrated
// server writes chunks when they unload and on autosave) and live chunks handed over by the mod as the
// client loads and unloads them (LodLive.swift; the only source in multiplayer). A background thread
// notices changed regions and rebuilds only the affected nodes and their parents, and meshes the finest
// level around the player's current position. Live chunks replace the region file's version of a chunk.
//
// Per region it caches the level-2 quadrant (128 x 128 x 96 voxels), so parents can be rebuilt without
// re-reading sibling regions from disk. Level 0 (full resolution, a quarter region per node) is built from
// region files only, in a ring around the player just past vanilla's render distance.

let lodQuadrantVoxels = lodNodeVoxels / 2
/// Level-0 nodes are meshed within this many blocks of the player (lod.detail, METALMC_LOD0 overrides; 0 for
/// none). 768 took the far-band fidelity error from 6.9 to 5.3 for 0.6 ms at the mid orbit (native res).
var lodLevel0Radius = Int(ProcessInfo.processInfo.environment["METALMC_LOD0"] ?? "") ?? 768

/// Sets the level-0 radius before the LOD opens (METALMC_LOD0 still wins, for experiments).
@_cdecl("mmc_lod_set_detail")
public func mmc_lod_set_detail(_ radius: Int32) {
    if ProcessInfo.processInfo.environment["METALMC_LOD0"] == nil { lodLevel0Radius = max(0, Int(radius)) }
}

/// A region's level-2 quadrant (128 x 128 columns of 96 voxels), run-length encoded per column:
/// `offsets[c] ..< offsets[c + 1]` indexes (material, count) byte pairs from the bottom up.
struct LodQuadrant {
    var offsets: [UInt32]
    var runs: [UInt8]

    init(offsets: [UInt32], runs: [UInt8]) {
        self.offsets = offsets
        self.runs = runs
    }

    init(grid g: LodGrid) {
        let q = lodQuadrantVoxels, h = g.height
        offsets = [UInt32](repeating: 0, count: q * q + 1)
        runs = []
        runs.reserveCapacity(q * q * 8)
        g.v.withUnsafeBufferPointer { src in
            for z in 0..<q {
                for x in 0..<q {
                    offsets[z * q + x] = UInt32(runs.count)
                    var y = 0
                    while y < h {
                        let m = src[(y * lodNodeVoxels + z) * lodNodeVoxels + x]
                        var n = 1
                        while y + n < h && n < 255 && src[((y + n) * lodNodeVoxels + z) * lodNodeVoxels + x] == m { n += 1 }
                        runs.append(m); runs.append(UInt8(n))
                        y += n
                    }
                }
            }
        }
        offsets[q * q] = UInt32(runs.count)
    }

    /// Writes the quadrant into quadrant (qx, qz) of a level-2 grid.
    func expand(into g: inout LodGrid, qx: Int, qz: Int) {
        let q = lodQuadrantVoxels
        g.v.withUnsafeMutableBufferPointer { dst in
            runs.withUnsafeBufferPointer { r in
                for z in 0..<q {
                    for x in 0..<q {
                        var y = 0
                        var i = Int(offsets[z * q + x])
                        let end = Int(offsets[z * q + x + 1])
                        let col = (qz * q + z) * lodNodeVoxels + qx * q + x
                        while i < end {
                            let m = r[i], n = Int(r[i + 1])
                            if m != 0 { for k in 0..<n { dst[(y + k) * lodNodeVoxels * lodNodeVoxels + col] = m } }
                            y += n
                            i += 2
                        }
                    }
                }
            }
        }
    }

    var bytes: Int { offsets.count * 4 + runs.count }
}

/// A meshed node on the GPU.
final class LodMeshNode {
    let level: Int
    let x0: Int, z0: Int
    let buffer: MTLBuffer
    let aoOffsets: MTLBuffer    // UInt32 per quad (LodMesh.aoOffsets)
    let ao: MTLBuffer           // packed rim ambient occlusion (at least one word)
    let quadCount: Int
    let start: [Int]            // prefix offsets of the (tile, face) buckets, 16 * 6 + 1 entries
    let tileY: [Int]            // per tile: min and max voxel y (min > max if the tile is empty)
    let tileYCore: [Int]        // the same without the node's edge skirts
    let sectionMask: [UInt32]   // levels 0-1: the chunk sections each tile has quads in (LodMesh.sectionMask)
    // Occlusion results, render thread only: the last frame each tile's box was tested, and the last
    // frame it was found visible.
    var tileTested = [UInt64](repeating: 0, count: 16)
    var tileVisible = [UInt64](repeating: 0, count: 16)
    var size: Int { lodNodeVoxels << level }

    init?(node: LodNode) {
        let m = node.mesh
        let ao = m.ao.isEmpty ? [UInt32(0)] : m.ao
        guard !m.quads.isEmpty,
              let b = ctx.device.makeBuffer(bytes: m.quads, length: m.quads.count * 4, options: [.storageModeShared]),
              let o = ctx.device.makeBuffer(bytes: m.aoOffsets, length: m.aoOffsets.count * 4, options: [.storageModeShared]),
              let a = ctx.device.makeBuffer(bytes: ao, length: ao.count * 4, options: [.storageModeShared]) else { return nil }
        level = node.level
        x0 = node.x0
        z0 = node.z0
        buffer = b
        aoOffsets = o
        self.ao = a
        quadCount = m.quads.count / 2
        var starts = [0]
        for c in m.counts { starts.append(starts.last! + c) }
        start = starts
        tileY = m.tileY
        tileYCore = m.tileYCore
        sectionMask = m.sectionMask
    }
}

struct LodNodeKey: Hashable {
    let level: Int
    let x: Int
    let z: Int
}

final class LodWorld: @unchecked Sendable {
    let id: Int                 // unique per opened world: calls from threads of a previous dimension or save miss it
    let dimension: String       // e.g. minecraft:overworld
    /// The End: islands over the void. Air columns are sky (chunks are marked with lodChunkMarker instead)
    /// and the air under an island isn't a cave to fill.
    let floating: Bool
    let hasSkyLight: Bool       // open surfaces get sky light 15 (vanilla's lightmap then gives the dimension's color)
    var downsampleRule: LodDownsampleRule { floating ? .top : lodDownsampleRule }
    let regionDir: URL?
    let cacheDir: URL?          // where fully generated regions' quadrants are kept between sessions (single-player)
    let live: LodLiveStore
    let maxLevel: Int
    let fineRadius: Int

    private let lock = NSLock()
    private var meshes: [LodNodeKey: LodMeshNode] = [:]
    let far = LodFarStore()   // generated columns where no chunk exists (LodFar.swift)
    private(set) var generation = 0

    // Update-thread state.
    private var fileStamps: [Int64: (Date, Int)] = [:]
    private var quadrants: [Int64: LodQuadrant] = [:]   // region -> run-length-encoded level-2 quadrant
    private var center: (x: Int, z: Int)
    private var meshedCenter: (x: Int, z: Int)?
    private var requestedCenter: (x: Int, z: Int)
    private var vanillaRadius = 0                      // blocks; regions this close to the player are drawn by vanilla
    private var deferred = Set<Int64>()                // changed regions waiting until the player leaves them
    // The queue's own priority is a floor (a block can only raise it), so it's background and later passes
    // raise themselves to utility.
    private let queue = DispatchQueue(label: "metalmc.lod.update", qos: .background)
    /// The first pass (every region of the save, 20+ s on the 8 km test world) runs at background priority: macOS
    /// keeps it on the efficiency cores, so the game keeps the performance cores while the player starts playing.
    /// Joining that world and flying at once gave 24 frames over 8.33 ms in the first minute instead of 155 at
    /// utility priority, but the LOD appeared after 27 s instead of 11. Later passes (the player moved: 1-3 s)
    /// run at utility priority; at background they took 9-11 s, too slow to keep level 0 around a flying
    /// player. METALMC_LODQOS=utility or background for one priority throughout.
    private static let qosSetting = ProcessInfo.processInfo.environment["METALMC_LODQOS"] ?? ""
    private var running = true
    private var paused = false   // the player is in another dimension: keep the meshes, stop updating
    var status = "starting"
    private(set) var firstPassDone = false   // the first build of every level has finished

    init(id: Int, dimension: String, floating: Bool, hasSkyLight: Bool, regionDir: URL?, storeDir: URL?, cacheDir: URL? = nil,
         maxLevel: Int, fineRadius: Int, centerX: Int, centerZ: Int) {
        self.cacheDir = cacheDir
        if let cacheDir { try? FileManager.default.createDirectory(at: cacheDir, withIntermediateDirectories: true) }
        self.id = id
        self.dimension = dimension
        self.floating = floating
        self.hasSkyLight = hasSkyLight
        self.regionDir = regionDir
        live = LodLiveStore(saveDir: storeDir)
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

    /// Pauses updates (the player left for another dimension) or resumes them. A pass in progress finishes.
    func setPaused(_ p: Bool) {
        lock.lock(); paused = p; lock.unlock()
    }

    func start() {
        queue.async { [self] in
            if live.saveDir != nil { log("LOD: loaded \(live.load()) saved regions from \(live.saveDir!.path)") }
            schedule(after: 0)
        }
    }

    /// One update pass per block (so each gets its own priority; the parallel work inside inherits it), every 2 s.
    private func schedule(after seconds: Double) {
        let background = Self.qosSetting == "background" || (Self.qosSetting != "utility" && !firstPassDone)
        queue.asyncAfter(deadline: .now() + seconds, qos: background ? .background : .utility, flags: background ? [] : .enforceQoS) { [self] in
            lock.lock(); let go = running, idle = paused; lock.unlock()
            if !go { return }
            if idle { schedule(after: 0.5); return }
            let t0 = Date()
            let did = poll()
            firstPassDone = true
            if did { log("LOD: update \(status) in \(String(format: "%.1f", Date().timeIntervalSince(t0))) s") }
            schedule(after: 2.0)
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
        var changed: [(Int, Int)] = []
        var seen = Set<Int64>()
        if let regionDir {
            let files = ((try? FileManager.default.contentsOfDirectory(atPath: regionDir.path)) ?? []).filter { $0.hasSuffix(".mca") }
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
                changed.append((x, z))
            }
        }
        // Regions that received live chunks.
        let changedSet = Set(changed.map { LodBuild.key($0.0, $0.1) })
        for k in live.takeDirty() where !changedSet.contains(k) { changed.append(LodBuild.unkey(k)) }
        for k in live.regionKeys { seen.insert(k) }

        lock.lock(); let want = requestedCenter; let near = vanillaRadius; lock.unlock()
        // Regions vanilla is drawing around the player change constantly (autosave). Rebuild them only once
        // the player has moved away; the first pass (no mesh yet) builds everything.
        if meshedCenter != nil && near > 0 {
            var keep: [(Int, Int)] = []
            for c in changed {
                if regionNear(c.0, c.1, cx: want.x, cz: want.z, radius: near + 64) {
                    deferred.insert(LodBuild.key(c.0, c.1))
                } else {
                    keep.append(c)
                }
            }
            for k in deferred {
                let (x, z) = LodBuild.unkey(k)
                if !regionNear(x, z, cx: want.x, cz: want.z, radius: near + 64) {
                    keep.append((x, z))
                    deferred.remove(k)
                }
            }
            changed = keep
        }
        let moved = meshedCenter.map { abs($0.x - want.x) > 128 || abs($0.z - want.z) > 128 } ?? true
        far.lock.lock(); var farKeys = far.dirty; far.dirty.removeAll(); let holes = far.holeRegions; far.lock.unlock()
        // Partly explored regions under newly generated nodes are read again, so their levels 0-1 fill the
        // unexplored columns (even near the player, where changed regions otherwise wait).
        if !farKeys.isEmpty && !holes.isEmpty {
            var have = Set(changed.map { LodBuild.key($0.0, $0.1) })
            for k in farKeys {
                let span = 1 << (k.level - 1)
                for r in holes where !have.contains(r) {
                    let (rx, rz) = LodBuild.unkey(r)
                    if rx >= k.x * span && rx < (k.x + 1) * span && rz >= k.z * span && rz < (k.z + 1) * span {
                        changed.append((rx, rz))
                        have.insert(r)
                    }
                }
            }
        }
        if changed.isEmpty && !moved && farKeys.isEmpty { return false }
        center = want

        var phases: [(String, Double)] = []
        var phaseStart = Date()
        func phase(_ name: String) { phases.append((name, Date().timeIntervalSince(phaseStart))); phaseStart = Date() }
        // 1. Changed regions: level-1 grid -> cached level-2 quadrant, and a level-1 mesh if near the player.
        var changedKeys = Set<Int64>()
        let results = rebuildRegions(changed, meshFine: true)
        far.lock.lock()
        for r in results {
            if r.hasData { far.realRegions.insert(r.key) } else { far.realRegions.remove(r.key) }
            if r.full { far.fullRegions.insert(r.key) } else { far.fullRegions.remove(r.key) }
            if r.hasData && !r.full { far.holeRegions.insert(r.key) } else { far.holeRegions.remove(r.key) }
        }
        far.lock.unlock()
        for r in results {
            changedKeys.insert(r.key)
            quadrants[r.key] = r.quadrant
            let (x, z) = LodBuild.unkey(r.key)
            install(LodNodeKey(level: 1, x: x, z: z), r.node)
        }

        phase("regions")
        // 2. The player moved: mesh level-1 nodes that came into range (from disk), drop the ones that left.
        if moved {
            var need: [(Int, Int)] = []
            snapshotLock: do {
                lock.lock()
                let have = Set(meshes.keys.filter { $0.level == 1 }.map { LodBuild.key($0.x, $0.z) })
                lock.unlock()
                for k in seen where !have.contains(k) && !changedKeys.contains(k) {
                    let (x, z) = LodBuild.unkey(k)
                    if withinFine(regionX: x, regionZ: z, cx: center.x, cz: center.z) { need.append((x, z)) }
                }
            }
            for r in rebuildRegions(need, meshFine: true) {
                let (x, z) = LodBuild.unkey(r.key)
                install(LodNodeKey(level: 1, x: x, z: z), r.node)
            }
            dropLevel1(outside: center.x, center.z)
            meshedCenter = center
        }

        phase("level 1")
        // 2b. Level 0 around the player: nodes that came into range, and those of changed regions.
        if lodLevel0Radius > 0, regionDir != nil { updateLevel0(seen: seen, changed: changedKeys) }

        phase("level 0")
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
                    let size = lodNodeVoxels << parents[i].level
                    if parents[i].level >= lodFarMinLevel {
                        _ = self.far.fill(&g, key: parents[i])
                    } else {
                        // Level 2: regions without any chunks are empty in the quadrants it's built from.
                        _ = self.far.fillFromAncestors(&g, x0: parents[i].x * size, z0: parents[i].z * size, maxLevel: self.maxLevel)
                    }
                    self.fillHidden(&g)
                    let m = self.meshNode(g)
                    outp[i] = LodNode(level: parents[i].level, x0: parents[i].x * size, z0: parents[i].z * size, mesh: m)
                }
            }
            for (i, p) in parents.enumerated() { install(p, built[i]) }
            farKeys.subtract(parents)
            dirty = Set(parents)
            level += 1
        }

        phase("parents")
        // 4. Nodes whose generated columns arrived (and weren't just rebuilt above).
        if !farKeys.isEmpty {
            let keys = Array(farKeys)
            var built = [LodNode?](repeating: nil, count: keys.count)
            built.withUnsafeMutableBufferPointer { out in
                let outp = out.baseAddress!
                DispatchQueue.concurrentPerform(iterations: keys.count) { i in
                    let k = keys[i]
                    let real = self.hasRealData(level: k.level, x: k.x, z: k.z)
                    var g = real ? self.grid(level: k.level, x: k.x, z: k.z) : LodGrid(level: k.level)
                    let filled = self.far.fill(&g, key: k)
                    if lodFarDebug {
                        self.far.lock.lock(); let hs = self.far.columns[k]?.height ?? []; self.far.lock.unlock()
                        log("LOD: far node \(k.level) \(k.x) \(k.z): real \(real), filled \(filled) columns, heights \(hs.min() ?? 0)...\(hs.max() ?? 0)")
                    }
                    self.fillHidden(&g)
                    let m = self.meshNode(g)
                    let size = lodNodeVoxels << k.level
                    outp[i] = LodNode(level: k.level, x0: k.x * size, z0: k.z * size, mesh: m)
                }
            }
            for (i, k) in keys.enumerated() { install(k, built[i]) }
        }
        phase("generated")
        live.saveUnsaved()
        lock.lock()
        let count = meshes.count
        let quads = meshes.values.reduce(0) { $0 + $1.quadCount }
        lock.unlock()
        let cacheMB = quadrants.values.reduce(0) { $0 + $1.bytes } / 1_000_000
        let timing = phases.filter { $0.1 >= 0.05 }.map { "\($0.0) \(String(format: "%.1f", $0.1))" }.joined(separator: ", ")
        status = "\(changed.count) changed regions, moved \(moved), \(farKeys.count) generated: \(count) nodes, \(quads) quads, quadrant cache \(cacheMB) MB, \(live.chunkCount) live chunks [\(timing) s]"
        return true
    }

    private func level0Near(_ nx: Int, _ nz: Int, radius: Int) -> Bool {
        let size = lodNodeVoxels
        let x0 = nx * size, z0 = nz * size
        let dx = Double(max(x0 - center.x, 0, center.x - (x0 + size))), dz = Double(max(z0 - center.z, 0, center.z - (z0 + size)))
        return (dx * dx + dz * dz).squareRoot() <= Double(radius)
    }

    /// Meshes the level-0 nodes within `lodLevel0Radius` that are missing or whose region changed, and drops
    /// the ones more than 256 blocks past it. A few at a time: each needs about 75 MB while it's built.
    private func updateLevel0(seen: Set<Int64>, changed: Set<Int64>) {
        guard let regionDir else { return }
        lock.lock()
        let have = Set(meshes.keys.filter { $0.level == 0 }.map { LodBuild.key($0.x, $0.z) })
        lock.unlock()
        var need: [(Int, Int)] = []
        for k in seen {
            let (rx, rz) = LodBuild.unkey(k)
            for q in 0..<4 {
                let nx = 2 * rx + (q & 1), nz = 2 * rz + (q >> 1)
                if level0Near(nx, nz, radius: lodLevel0Radius) && (!have.contains(LodBuild.key(nx, nz)) || changed.contains(k)) {
                    need.append((nx, nz))
                }
            }
        }
        let batch = 6
        var i = 0
        while i < need.count {
            let part = Array(need[i..<min(need.count, i + batch)])
            var built = [LodNode?](repeating: nil, count: part.count)
            built.withUnsafeMutableBufferPointer { out in
                let outp = out.baseAddress!
                DispatchQueue.concurrentPerform(iterations: part.count) { j in
                    let (nx, nz) = part[j]
                    let path = regionDir.appendingPathComponent("r.\(nx >> 1).\(nz >> 1).mca").path
                    guard var g = LodBuild.regionQuarterGrid(path: path, qx: nx & 1, qz: nz & 1, floating: self.floating) else { return }
                    _ = self.far.fillFromAncestors(&g, x0: nx * lodNodeVoxels, z0: nz * lodNodeVoxels, maxLevel: self.maxLevel)
                    self.fillHidden(&g, deepRadius: 16, deepDepth: 8)
                    let m = self.meshNode(g)
                    outp[j] = LodNode(level: 0, x0: nx * lodNodeVoxels, z0: nz * lodNodeVoxels, mesh: m)
                }
            }
            for (j, (nx, nz)) in part.enumerated() { install(LodNodeKey(level: 0, x: nx, z: nz), built[j]) }
            i += batch
        }
        lock.lock()
        for k in meshes.keys where k.level == 0 && !level0Near(k.x, k.z, radius: lodLevel0Radius + 256) {
            meshes[k] = nil
            generation += 1
        }
        lock.unlock()
    }

    struct RegionResult {
        var key: Int64
        var quadrant: LodQuadrant
        var node: LodNode?
        var hasData: Bool   // any real chunk
        var full: Bool      // every column has data
    }

    /// Reads regions in parallel (region file, then live chunks on top): returns each region's level-2
    /// quadrant, level-1 node if within the fine radius, and whether it has real chunks (all or any).
    private func rebuildRegions(_ list: [(Int, Int)], meshFine: Bool) -> [RegionResult] {
        var out = [RegionResult?](repeating: nil, count: list.count)
        let c = center
        let liveKeys = Set(live.regionKeys)
        out.withUnsafeMutableBufferPointer { buf in
            let outp = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: list.count) { i in
                let (x, z) = list[i]
                let key = LodBuild.key(x, z)
                let near = meshFine && self.withinFine(regionX: x, regionZ: z, cx: c.x, cz: c.z)
                // Unchanged since an earlier session: the region's quadrant from the cache instead of decoding it.
                if !near, !liveKeys.contains(key), let cached = self.cachedQuadrant(x: x, z: z) {
                    outp[i] = RegionResult(key: key, quadrant: cached, node: nil, hasData: true, full: true)
                    return
                }
                let fromFile = self.regionDir.flatMap { LodBuild.regionGrid(path: $0.appendingPathComponent("r.\(x).\(z).mca").path, floating: self.floating) }
                var g = fromFile ?? LodGrid(level: 1)
                let hasLive = self.live.overlay(regionX: x, regionZ: z, into: &g)
                guard fromFile != nil || hasLive else {
                    outp[i] = RegionResult(key: LodBuild.key(x, z), quadrant: LodQuadrant(grid: LodGrid(level: 2)), node: nil, hasData: false, full: false)
                    return
                }
                // Every generated column has something (bedrock) in its bottom voxel.
                var full = true
                g.v.withUnsafeBufferPointer { v in for c in 0..<(lodNodeVoxels * lodNodeVoxels) where v[c] == 0 { full = false; break } }
                if !full { _ = self.far.fillFromAncestors(&g, x0: x * 512, z0: z * 512, maxLevel: self.maxLevel) }
                var q = LodGrid(level: 2)
                g.downsample(into: &q, qx: 0, qz: 0, rule: self.downsampleRule)   // the quadrant occupies the low corner
                let quadrant = LodQuadrant(grid: q)
                // Only a fully generated region's quadrant depends on nothing but its file (a partly explored one is
                // filled from generated terrain, a region with live chunks has edits the file may not have).
                if fromFile != nil && !hasLive && full { self.saveQuadrant(x: x, z: z, quadrant) }
                var node: LodNode?
                if near {
                    var filled = g
                    self.fillHidden(&filled)
                    let m = self.meshNode(filled)
                    let size = lodNodeVoxels << 1
                    node = LodNode(level: 1, x0: x * size, z0: z * size, mesh: m)
                }
                outp[i] = RegionResult(key: LodBuild.key(x, z), quadrant: quadrant, node: node, hasData: true, full: full)
            }
        }
        return out.compactMap { $0 }
    }

    /// Fills air that can't be seen (LodGrid.fillUnreachable), without the deep-air step in floating dimensions:
    /// the air under an island is open to the void, not a cave.
    func fillHidden(_ g: inout LodGrid, deepRadius: Int = lodDeepFill ? 8 : -1, deepDepth: Int = 4) {
        g.fillUnreachable(deepRadius: floating ? -1 : deepRadius, deepDepth: deepDepth)
    }

    /// Greedy mesh of a node grid, with sky light under cover (always in floating dimensions: their undersides need it).
    func meshNode(_ g: LodGrid) -> LodMesh {
        LodBuild.mesh(g, maxMerge: 64, skyCover: floating || lodSkyCover)
    }

    // MARK: - Region cache

    /// Bump when what a quadrant holds changes (materials, downsampling), so older cache files are rebuilt.
    static let cacheVersion: UInt8 = 1

    private func cacheURL(_ x: Int, _ z: Int) -> URL? { cacheDir?.appendingPathComponent("r.\(x).\(z).lq") }

    /// The region file's stamp as stored in cache files: modification time (ms) and size.
    private func stamp(_ x: Int, _ z: Int) -> (Int64, Int64)? {
        guard let s = fileStamps[LodBuild.key(x, z)] else { return nil }
        return (Int64((s.0.timeIntervalSince1970 * 1000).rounded()), Int64(s.1))
    }

    /// A fully generated region's quadrant from an earlier session, if its region file hasn't changed since.
    /// File: "MMCQ", version, region file stamp (ms, bytes), body length, then LZFSE of the offsets (UInt32) and runs.
    func cachedQuadrant(x: Int, z: Int) -> LodQuadrant? {
        guard let url = cacheURL(x, z), let st = stamp(x, z), let d = try? Data(contentsOf: url), d.count > 29 else { return nil }
        let raw = [UInt8](d)
        func i64(_ o: Int) -> Int64 { var v: Int64 = 0; for k in 0..<8 { v |= Int64(raw[o + k]) << (8 * k) }; return v }
        guard raw[0...3] == Array("MMCQ".utf8)[0...3], raw[4] == Self.cacheVersion, i64(5) == st.0, i64(13) == st.1 else { return nil }
        let len = Int(i64(21))
        let q = lodQuadrantVoxels
        guard len >= (q * q + 1) * 4 else { return nil }
        var body = [UInt8](repeating: 0, count: len)
        let got = raw.withUnsafeBufferPointer { compression_decode_buffer(&body, len, $0.baseAddress! + 29, $0.count - 29, nil, COMPRESSION_LZFSE) }
        guard got == len else { return nil }
        var offsets = [UInt32](repeating: 0, count: q * q + 1)
        for k in 0..<offsets.count {
            offsets[k] = UInt32(body[4 * k]) | UInt32(body[4 * k + 1]) << 8 | UInt32(body[4 * k + 2]) << 16 | UInt32(body[4 * k + 3]) << 24
        }
        let runs = Array(body[(offsets.count * 4)...])
        guard Int(offsets[q * q]) == runs.count else { return nil }
        return LodQuadrant(offsets: offsets, runs: runs)
    }

    func saveQuadrant(x: Int, z: Int, _ quadrant: LodQuadrant) {
        guard let url = cacheURL(x, z), let st = stamp(x, z) else { return }
        var body = [UInt8]()
        body.reserveCapacity(quadrant.offsets.count * 4 + quadrant.runs.count)
        for o in quadrant.offsets { body += [UInt8(o & 255), UInt8((o >> 8) & 255), UInt8((o >> 16) & 255), UInt8(o >> 24)] }
        body += quadrant.runs
        var packed = [UInt8](repeating: 0, count: body.count + 1024)
        let n = body.withUnsafeBufferPointer { compression_encode_buffer(&packed, packed.count, $0.baseAddress!, $0.count, nil, COMPRESSION_LZFSE) }
        guard n > 0 else { return }
        func le(_ v: Int64) -> [UInt8] { (0..<8).map { UInt8((v >> (8 * $0)) & 255) } }
        let out = Data(Array("MMCQ".utf8) + [Self.cacheVersion] + le(st.0) + le(st.1) + le(Int64(body.count)) + packed[0..<n])
        try? out.write(to: url, options: .atomic)
    }

    /// True if any region under node (level, x, z) has real chunks.
    func hasRealData(level: Int, x: Int, z: Int) -> Bool {
        let span = 1 << (level - 1)   // regions per node side
        far.lock.lock(); defer { far.lock.unlock() }
        if far.realRegions.isEmpty { return false }
        if span * span > far.realRegions.count {
            for k in far.realRegions {
                let (rx, rz) = LodBuild.unkey(k)
                if rx >> (level - 1) == x && rz >> (level - 1) == z { return true }
            }
            return false
        }
        for rz in (z * span)..<((z + 1) * span) {
            for rx in (x * span)..<((x + 1) * span) where far.realRegions.contains(LodBuild.key(rx, rz)) { return true }
        }
        return false
    }

    /// Nodes of levels 3 and up that the quadtree can draw around the player and that lack real chunks
    /// somewhere, nearest first, not yet generated or requested. Marks them requested.
    func farWanted(max: Int) -> [LodNodeKey] {
        lock.lock(); let c = requestedCenter; lock.unlock()
        guard maxLevel >= lodFarMinLevel else { return [] }
        func dist(_ x0: Int, _ z0: Int, _ size: Int) -> Double {
            let dx = Double(Swift.max(x0 - c.x, 0, c.x - (x0 + size))), dz = Double(Swift.max(z0 - c.z, 0, c.z - (z0 + size)))
            return (dx * dx + dz * dz).squareRoot()
        }
        var found: [(Double, LodNodeKey)] = []
        far.lock.lock(); defer { far.lock.unlock() }
        for level in lodFarMinLevel...maxLevel {
            let size = lodNodeVoxels << level
            // A node is drawn when its parent splits (the parent's nearest point within 2 x this node's size)
            // or, at the top level, anywhere within the LOD distance.
            let reach = level == maxLevel ? Double(size) : 2.0 * Double(size)
            let psize = size * 2
            let cx = Int((Double(c.x) / Double(size)).rounded(.down)), cz = Int((Double(c.z) / Double(size)).rounded(.down))
            let r = Int(reach) / size + 2
            for nz in (cz - r)...(cz + r) {
                for nx in (cx - r)...(cx + r) {
                    let key = LodNodeKey(level: level, x: nx, z: nz)
                    if far.columns[key] != nil || far.requested.contains(key) { continue }
                    if far.cached.contains(key) {
                        // Generated in an earlier session: load it instead of asking for it.
                        far.cached.remove(key)
                        if let cols = far.load(key) {
                            far.columns[key] = cols
                            far.dirty.insert(key)
                            continue
                        }
                    }
                    let d = level == maxLevel ? dist(nx * size, nz * size, size) : dist((nx >> 1) * psize, (nz >> 1) * psize, psize)
                    if d > reach { continue }
                    let span = 1 << (level - 1)
                    var covered = true
                    check: for rz in (nz * span)..<((nz + 1) * span) {
                        for rx in (nx * span)..<((nx + 1) * span) where !far.fullRegions.contains(LodBuild.key(rx, rz)) { covered = false; break check }
                    }
                    if covered { continue }
                    found.append((dist(nx * size, nz * size, size), key))
                }
            }
        }
        found.sort { $0.0 < $1.0 }
        let out = found.prefix(max).map { $0.1 }
        for k in out { far.requested.insert(k) }
        return Array(out)
    }

    /// Dense grid for a node at `level` >= 2, assembled from cached region quadrants (downsampled further
    /// for levels above 2). Missing regions are air.
    func grid(level: Int, x: Int, z: Int) -> LodGrid {
        if level == 2 {
            var g = LodGrid(level: 2)
            for qz in 0...1 {
                for qx in 0...1 {
                    quadrants[LodBuild.key(2 * x + qx, 2 * z + qz)]?.expand(into: &g, qx: qx, qz: qz)
                }
            }
            return g
        }
        var g = LodGrid(level: level)
        for qz in 0...1 {
            for qx in 0...1 {
                // Children with no real chunks are air: skip them (at the top levels most of a node can be
                // generated terrain, and assembling empty subtrees down to level 2 costs gigabytes of grids).
                if !hasRealData(level: level - 1, x: 2 * x + qx, z: 2 * z + qz) { continue }
                let child = grid(level: level - 1, x: 2 * x + qx, z: 2 * z + qz)
                child.downsample(into: &g, qx: qx, qz: qz, rule: downsampleRule)
            }
        }
        return g
    }
}
