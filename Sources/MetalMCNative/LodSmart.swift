import Foundation
import MetalMCCore
import simd

// Smart LOD selection (METALMC_EXP=smartlod): levels by how far a coarser level's surface would lie from the finer
// one's on screen, not by distance alone, within a per-frame quad budget.
//
// The distance rule (LodRenderer.select) spends detail evenly: every node splits at the same distance, so flat
// plains and the open sea get as many quads as cliffs, towers and forest edges. Here each node records, per tile, how
// far its parent's surface lies from its own over that tile (the error of not splitting the parent there, measured
// when the node is meshed). Selection then works per tile, like METALMC_EXP=tilesel: a tile splits into the finer
// level's 2 x 2 tiles when that error, projected to pixels at the tile's distance, is over METALMC_LODERR, the
// worst tiles first, until the quads the view would draw reach METALMC_LODBUDGET. Tile-edge skirts (levels 1 and up)
// cover the steps where a node draws some tiles and a finer level the others.

/// METALMC_EXP=smartlod: per-tile selection by projected error with a quad budget (see above).
let lodSmart = experiments.contains("smartlod")
/// Projected error (pixels at the panel's resolution, at the center of a 70-degree view) over which a tile splits
/// (METALMC_LODERR, -PlodErr). Offline on the 4 km test world (tools/smartlod.py), 2 px drew 5-6% fewer quads than the
/// distance rule, left a mean error under 0.08 px, and at most 2 px where the distance rule left up to 5.7.
let lodSmartErrorPx = Double(ProcessInfo.processInfo.environment["METALMC_LODERR"] ?? "") ?? 2.0
/// Quads the error-driven splits may bring the view up to (METALMC_LODBUDGET, -PlodBudget): the chosen tiles' quads
/// in the view frustum facing the camera, before occlusion culling (so more than the frame draws, most at ground
/// level). The level-0 ring and the levels leading to it are always drawn and count toward it. The distance rule's
/// views on the test world counted 1.7-3.0 M, so this binds only in busier ones.
let lodSmartBudget = Int(ProcessInfo.processInfo.environment["METALMC_LODBUDGET"] ?? "") ?? 3_000_000
/// A tile split last frame stays split down to this share of the threshold, and outranks new splits by its inverse
/// under the budget, so selection doesn't flip back and forth at the threshold.
let lodSmartHysteresis = 0.7
/// A tile split last frame may stay split up to this much over the budget (new splits stop at the budget itself).
/// Without it, a camera hovering with the view swaying 3 degrees under a binding budget changed 16 tiles a frame:
/// tiles at the frustum's edges come and go, the quads counted move with them, and the last splits that fit swap.
let lodSmartBudgetSlack = 0.1
/// METALMC_EXP=smartmax ranks tiles by the largest error over them instead of the RMS error.
let lodSmartUseMax = experiments.contains("smartmax")
/// Panel pixels per radian at the center of the view for proj[1][1] = 1: half of the 2234-pixel panel height.
let lodSmartHalfPanel = 1117.0

/// Per tile of a node's grid (levels 1 and up): how far the next coarser level's surface lies from this one's over
/// the tile, as it would be built by downsampling this grid (`rule`, as the parent is). Two floats per tile (tile =
/// tz * 4 + tx): the RMS and the largest error over its columns, in blocks. A column's error is the largest height
/// difference between the coarse column's top (the top of its highest non-air voxel, water included) and the tops
/// of the 2 x 2 fine columns under it; a coarse column with nothing left in it counts as height 0 (a tower or a
/// thin tree that vanished). Columns with no data are skipped.
///
/// Flat ground scores its rounding (0 or one voxel), the sea 0 (every level's surface is the same), cliffs and
/// tree edges their height where the coarse column rounds them away. The parents' grids are downsampled from their
/// children's before those are filled (LodWorld.grid), and filling only turns hidden air under the surface to stone,
/// so column tops are the same either way. Level 0 isn't measured: level 1 splits into it by distance only (level 0
/// has no tile-edge skirts, so it's drawn by whole nodes).
func lodTileErrors(_ g: LodGrid, rule: LodDownsampleRule) -> [Float] {
    let n = lodNodeVoxels, h = g.height, layer = n * n
    let ph = h / 2   // the parent's height (LodGrid.downsample)
    guard g.level >= 1, ph >= 1 else { return [] }
    var out = [Float](repeating: 0, count: 2 * lodTilesPerSide * lodTilesPerSide)
    g.v.withUnsafeBufferPointer { v in
        // Highest non-air voxel per column, top down: the scan stops once every column has one.
        var top = [Int32](repeating: -1, count: layer)
        var left = layer
        var y = h - 1
        while y >= 0 && left > 0 {
            let base = y * layer
            for i in 0..<layer where top[i] < 0 && v[base + i] != 0 {
                top[i] = Int32(y)
                left -= 1
            }
            y -= 1
        }
        let s = Double(1 << g.level)
        var sum = [Double](repeating: 0, count: 16), count = [Int](repeating: 0, count: 16), worst = [Double](repeating: 0, count: 16)
        for cz in 0..<(n / 2) {
            for cx in 0..<(n / 2) {
                let i00 = 2 * cz * n + 2 * cx, i01 = i00 + 1, i10 = i00 + n, i11 = i10 + 1
                let tops = (top[i00], top[i01], top[i10], top[i11])
                let highest = max(max(tops.0, tops.1), max(tops.2, tops.3))
                if highest < 0 { continue }
                // The coarse column's top: its highest non-air voxel, from the fine column's top down.
                var cy = min(Int(highest) >> 1, ph - 1), coarse = -1
                while cy >= 0 {
                    var c = SIMD8<UInt8>(repeating: 0)
                    for dy in 0..<2 {
                        let row = (2 * cy + dy) * layer
                        c[dy << 2] = v[row + i00]; c[dy << 2 | 1] = v[row + i01]
                        c[dy << 2 | 2] = v[row + i10]; c[dy << 2 | 3] = v[row + i11]
                    }
                    if lodReduce(c, rule: rule) != 0 { coarse = cy; break }
                    cy -= 1
                }
                let coarseTop = Double(coarse + 1) * 2 * s
                @inline(__always) func diff(_ t: Int32) -> Double { t < 0 ? 0 : abs(coarseTop - Double(t + 1) * s) }
                let e = max(max(diff(tops.0), diff(tops.1)), max(diff(tops.2), diff(tops.3)))
                let tile = (2 * cz / lodTileVoxels) * lodTilesPerSide + 2 * cx / lodTileVoxels
                sum[tile] += e * e
                count[tile] += 1
                worst[tile] = max(worst[tile], e)
            }
        }
        for t in 0..<16 where count[t] > 0 {
            out[2 * t] = Float((sum[t] / Double(count[t])).squareRoot())
            out[2 * t + 1] = Float(worst[t])
        }
    }
    return out
}

/// Smart selection's view of one mesh set, rebuilt when the set changes (render thread): nodes by index (children
/// before parents), each one's children, the saturated split errors and the roots, so that a frame's selection only
/// walks arrays.
final class LodSmartGraph {
    let nodes: [LodMeshNode]
    let keys: [LodNodeKey]
    let children: [Int32]       // 4 per node (quarter qz * 2 + qx), -1 where there's none
    /// 16 per node (tile tz * 4 + tx): the error of drawing the tile instead of the finer levels under it, in blocks.
    /// That's the tile's own split error (its child's tileError over that quarter: the RMS, or the largest with
    /// METALMC_EXP=smartmax), saturated with the child tiles' errors, so a tile whose own step is small still splits
    /// when a finer level under it would. -1: no finer node to split into, or level 1 (level 0 is reached by distance
    /// only); NaN: the child has no error data (the distance rule decides).
    let errors: [Float]
    let roots: [Int32]          // every node whose parent doesn't exist, not just the top level (as in `select`)
    let index: [LodNodeKey: Int32]
    // Copies of what each frame reads per tile, so it doesn't go through the nodes (class and array accesses).
    let origin: [SIMD4<Double>] // per node: x0, z0 (world blocks), tile size, node size
    let tileY: [Float]          // 2 per tile: bottom and top of its quads (world blocks); bottom > top if it has none
    let tileFaces: [Int32]      // 6 per tile: its opaque and water quads facing +X -X +Y -Y +Z -Z

    init(_ meshes: [LodNodeKey: LodMeshNode], maxLevel: Int) {
        let sorted = meshes.sorted { ($0.key.level, $0.key.x, $0.key.z) < ($1.key.level, $1.key.x, $1.key.z) }
        let nodes = sorted.map { $0.value }, keys = sorted.map { $0.key }
        var index: [LodNodeKey: Int32] = [:]
        index.reserveCapacity(keys.count)
        for (i, k) in keys.enumerated() { index[k] = Int32(i) }
        var children = [Int32](repeating: -1, count: 4 * keys.count)
        var errors = [Float](repeating: -1, count: 16 * keys.count)
        var roots: [Int32] = []
        var origin = [SIMD4<Double>](repeating: .zero, count: keys.count)
        var tileY = [Float](repeating: 0, count: 32 * keys.count)
        var tileFaces = [Int32](repeating: 0, count: 96 * keys.count)
        for (i, n) in nodes.enumerated() {
            let voxel = Double(1 << n.level)
            origin[i] = SIMD4(Double(n.x0), Double(n.z0), Double(lodTileVoxels) * voxel, Double(n.size))
            for t in 0..<16 {
                let j = 16 * i + t, yMin = n.tileY[2 * t], yMax = n.tileY[2 * t + 1]
                tileY[2 * j] = yMin > yMax ? 1 : Float(Double(lodWorldMinY) + Double(yMin) * voxel)
                tileY[2 * j + 1] = yMin > yMax ? 0 : Float(Double(lodWorldMinY) + Double(yMax) * voxel)
                let base = lodBucketIndex(t, 0, 0), sts = lodSubtilesPerTile
                for f in 0..<6 {
                    tileFaces[6 * j + f] = Int32(n.start[base + (f + 1) * sts] - n.start[base + f * sts]
                                                 + n.start[base + (f + 7) * sts] - n.start[base + (f + 6) * sts])
                }
            }
        }
        for (i, k) in keys.enumerated() {
            if k.level == maxLevel || index[LodNodeKey(level: k.level + 1, x: k.x >> 1, z: k.z >> 1)] == nil { roots.append(Int32(i)) }
            if k.level < 1 { continue }
            for q in 0..<4 { children[4 * i + q] = index[LodNodeKey(level: k.level - 1, x: 2 * k.x + q % 2, z: 2 * k.z + q / 2)] ?? -1 }
            if k.level < 2 { continue }
            for t in 0..<16 {
                let tx = t % 4, tz = t / 4
                let c = Int(children[4 * i + (tz / 2) * 2 + tx / 2])
                if c < 0 { continue }
                let ce = nodes[c].tileError
                if ce.count < 32 { errors[16 * i + t] = .nan; continue }
                var own: Float = 0, sq: Float = 0, below: Float = 0
                for j in 0...1 {
                    for m in 0...1 {
                        let ct = ((tz % 2) * 2 + j) * 4 + (tx % 2) * 2 + m
                        sq += ce[2 * ct] * ce[2 * ct]
                        own = max(own, ce[2 * ct + 1])
                        let b = errors[16 * c + ct]   // children come first
                        if b > below { below = b }    // NaN and -1 never compare greater
                    }
                }
                errors[16 * i + t] = max(lodSmartUseMax ? own : (sq / 4).squareRoot(), below)
            }
        }
        self.nodes = nodes
        self.keys = keys
        self.index = index
        self.children = children
        self.errors = errors
        self.roots = roots
        self.origin = origin
        self.tileY = tileY
        self.tileFaces = tileFaces
    }
}

/// Smart selection's state across frames (render thread only, apart from background builds): the graph of the current
/// mesh set, and the tiles split last frame (for hysteresis).
final class LodSmartState {
    /// The game's. Its graphs are rebuilt on a background queue: offline, a rebuild with the nodes' data cold in the
    /// caches took 0.3 ms (p90) for 162 nodes and once 3.5 ms, and the mesh set changes with every node the LOD
    /// installs. The frame keeps the last graph until the new one is ready (a frame or two; the nodes it still draws
    /// stay valid, as fading ones do).
    static let shared = LodSmartState(background: true)
    let background: Bool
    private(set) var graph: LodSmartGraph?
    private var worldId = -1
    private var generation = -1
    var split: [UInt16] = []                      // per graph node: its tiles split last frame
    var stats = [Int64](repeating: 0, count: 6)   // the last frame's selectSmart stats
    var frames = 0
    var nanos: UInt64 = 0                         // selection time since the last log line
    var worstNanos: UInt64 = 0                    // the slowest selection since then
    // Background builds, under buildLock.
    private let buildLock = NSLock()
    private var built: (graph: LodSmartGraph, worldId: Int, generation: Int)?
    private var building = false
    private var buildNanos: UInt64 = 0            // the last background build
    private static let queue = DispatchQueue(label: "metalmc.lod.smart", qos: .userInitiated)

    init(background: Bool = false) { self.background = background }

    /// Microseconds the last background build took.
    var lastBuildMicros: UInt64 {
        buildLock.lock(); defer { buildLock.unlock() }
        return buildNanos / 1000
    }

    /// Drops the graph, and the nodes it holds, when the LOD is closed. A build still running lands for a world that
    /// no longer draws and is dropped on the next frame.
    func reset() {
        graph = nil
        worldId = -1
        generation = -1
        split = []
        buildLock.lock(); built = nil; buildLock.unlock()
    }

    /// The graph of world `worldId`'s mesh set `generation`, rebuilt when that changed (in the background, if this
    /// state builds there and has a graph of the same world to use meanwhile). Last frame's splits carry over by node
    /// key.
    func graph(worldId: Int, generation: Int, meshes: [LodNodeKey: LodMeshNode], maxLevel: Int) -> LodSmartGraph {
        if background {
            buildLock.lock(); let done = built; built = nil; buildLock.unlock()
            if let done, done.worldId == worldId, !(self.worldId == worldId && self.generation >= done.generation) {
                adopt(done.graph, worldId: worldId, generation: done.generation)
            }
        }
        if let g = graph, self.worldId == worldId, self.generation == generation { return g }
        if background, let g = graph, self.worldId == worldId {
            buildLock.lock()
            if !building {
                building = true
                Self.queue.async { [self] in
                    let t0 = DispatchTime.now().uptimeNanoseconds
                    let ng = LodSmartGraph(meshes, maxLevel: maxLevel)
                    buildLock.lock()
                    built = (ng, worldId, generation)
                    building = false
                    buildNanos = DispatchTime.now().uptimeNanoseconds - t0
                    buildLock.unlock()
                }
            }
            buildLock.unlock()
            return g
        }
        adopt(LodSmartGraph(meshes, maxLevel: maxLevel), worldId: worldId, generation: generation)
        return graph!
    }

    private func adopt(_ g: LodSmartGraph, worldId: Int, generation: Int) {
        var carried = [UInt16](repeating: 0, count: g.nodes.count)
        if let old = graph, self.worldId == worldId {
            for (i, m) in split.enumerated() where m != 0 {
                if let j = g.index[old.keys[i]] { carried[Int(j)] = m }
            }
        }
        graph = g
        self.worldId = worldId
        self.generation = generation
        split = carried
    }
}

/// The column-major 4 x 4 matrix at p[o ..< o + 16] (mmc_lod_draw's parameters).
func lodMatrix(_ p: UnsafePointer<Float>, _ o: Int) -> simd_float4x4 {
    simd_float4x4(SIMD4(p[o], p[o + 1], p[o + 2], p[o + 3]), SIMD4(p[o + 4], p[o + 5], p[o + 6], p[o + 7]),
                  SIMD4(p[o + 8], p[o + 9], p[o + 10], p[o + 11]), SIMD4(p[o + 12], p[o + 13], p[o + 14], p[o + 15]))
}

/// Frustum side planes (camera-relative, as in mmc_lod_draw) of proj * view. Points behind the camera fail them too.
func lodFrustumPlanes(_ m: simd_float4x4) -> [SIMD4<Float>] {
    let r0 = SIMD4(m.columns.0.x, m.columns.1.x, m.columns.2.x, m.columns.3.x)
    let r1 = SIMD4(m.columns.0.y, m.columns.1.y, m.columns.2.y, m.columns.3.y)
    let r3 = SIMD4(m.columns.0.w, m.columns.1.w, m.columns.2.w, m.columns.3.w)
    return [r3 + r0, r3 - r0, r3 + r1, r3 - r1]
}

/// Tiles of quarter q = qz * 2 + qx of a node (tile t = tz * 4 + tx; the quarter covers tx, tz in 2q ..< 2q + 2).
let lodQuarterTiles: [UInt16] = [0x0033, 0x00CC, 0x3300, 0xCC00]

extension LodRenderer {
    /// Smart LOD selection (METALMC_EXP=smartlod). Same result as `select`: nodes with the tiles to draw.
    ///
    /// The roots draw all their tiles. A tile of level 2 or up with a finer node under it is a candidate to split into
    /// that node's 2 x 2 tiles over it:
    /// - Forced, nearest first, where the level-0 ring needs it: a tile within `level0Radius` of the camera, measured
    ///   to the level-1 node it lies in (the distance rule's level-0 reach), and a level-1 node's quarters within it
    ///   (level 0 by whole nodes: it has no tile-edge skirts). These always split, as the distance rule has them, and
    ///   count toward the budget.
    /// - Otherwise by projected error: the tile's saturated error (LodSmartGraph.errors) over its 3D distance, times
    ///   `pixelsPerRadian`, against `errorPx`. The worst splits go first; one that would take the view's quads over
    ///   `budget` is skipped. A child without error data falls back to the distance rule (`splitFactor`) at tile
    ///   granularity. Tiles split last frame (`split` on entry) keep splitting down to lodSmartHysteresis x the
    ///   threshold, outrank new ones by its inverse and may go lodSmartBudgetSlack over the budget; `split` gets this
    ///   frame's.
    /// A tile's cost is its quads in buckets that can face the camera, 0 outside the view frustum (`planes`,
    /// camera-relative): off-screen tiles split for free, so turning around finds them refined already.
    /// `stats`, if given, gets [quads counted, forced splits, error splits, splits over the budget, fallback splits].
    static func selectSmart(_ g: LodSmartGraph, cam: SIMD3<Double>, planes: [SIMD4<Float>], pixelsPerRadian: Double,
                            errorPx: Double, budget: Int, level0Radius: Double, splitFactor: Double, split: inout [UInt16],
                            stats: UnsafeMutablePointer<Int64>? = nil) -> [(LodMeshNode, UInt16)] {
        let count = g.nodes.count
        if split.count != count { split = [UInt16](repeating: 0, count: count) }
        let prev = split
        var masks = [UInt16](repeating: 0, count: count)
        var splitNow = [UInt16](repeating: 0, count: count)
        let keys = g.keys, children = g.children, errors = g.errors, origin = g.origin, tileY = g.tileY, faces = g.tileFaces
        let p0 = planes[0], p1 = planes[1], p2 = planes[2], p3 = planes[3]
        @inline(__always) func outside(_ pl: SIMD4<Float>, _ lo: SIMD3<Float>, _ hi: SIMD3<Float>) -> Bool {
            let v = SIMD3(pl.x >= 0 ? hi.x : lo.x, pl.y >= 0 ? hi.y : lo.y, pl.z >= 0 ? hi.z : lo.z)
            return pl.x * v.x + pl.y * v.y + pl.z * v.z + pl.w < 0
        }
        // Tile box (world blocks): x, z from the grid; y from the tile's quads (the world's height if it has none).
        @inline(__always) func box(_ i: Int, _ t: Int) -> (lo: SIMD3<Double>, hi: SIMD3<Double>, empty: Bool) {
            let o = origin[i], j = 16 * i + t
            let x0 = o.x + Double(t % lodTilesPerSide) * o.z, z0 = o.y + Double(t / lodTilesPerSide) * o.z
            let y0 = tileY[2 * j], y1 = tileY[2 * j + 1]
            if y0 > y1 { return (SIMD3(x0, Double(lodWorldMinY), z0), SIMD3(x0 + o.z, Double(lodWorldMinY + lodWorldHeight), z0 + o.z), true) }
            return (SIMD3(x0, Double(y0), z0), SIMD3(x0 + o.z, Double(y1), z0 + o.z), false)
        }
        func cost(_ i: Int, _ t: Int) -> Int {
            let b = box(i, t)
            if b.empty { return 0 }
            let lo = SIMD3<Float>(b.lo - cam), hi = SIMD3<Float>(b.hi - cam)
            if outside(p0, lo, hi) || outside(p1, lo, hi) || outside(p2, lo, hi) || outside(p3, lo, hi) { return 0 }
            // Opaque and water buckets of faces that can face the camera, as mmc_lod_draw picks them.
            let f = 6 * (16 * i + t)
            var q: Int32 = 0
            if 0 > lo.x { q += faces[f] }
            if 0 < hi.x { q += faces[f + 1] }
            if 0 > lo.y { q += faces[f + 2] }
            if 0 < hi.y { q += faces[f + 3] }
            if 0 > lo.z { q += faces[f + 4] }
            if 0 < hi.z { q += faces[f + 5] }
            return Int(q)
        }
        @inline(__always) func horizontal(_ x0: Double, _ z0: Double, _ size: Double) -> Double {
            let dx = max(x0 - cam.x, 0, cam.x - (x0 + size)), dz = max(z0 - cam.z, 0, cam.z - (z0 + size))
            return (dx * dx + dz * dz).squareRoot()
        }
        // Candidates: a max-heap on priority. Forced splits rank above every error (1e30 minus their distance).
        let quarterK: UInt8 = 1, forcedK: UInt8 = 2, fallbackK: UInt8 = 4
        struct Candidate { var priority: Double; var node: Int32; var t: Int8; var kind: UInt8 }
        var heap: [Candidate] = []
        heap.reserveCapacity(1024)
        func push(_ c: Candidate) {
            heap.append(c)
            var i = heap.count - 1
            while i > 0 {
                let p = (i - 1) / 2
                if heap[p].priority >= heap[i].priority { break }
                heap.swapAt(p, i)
                i = p
            }
        }
        func pop() -> Candidate? {
            guard let last = heap.popLast() else { return nil }
            if heap.isEmpty { return last }
            let top = heap[0]
            heap[0] = last
            var i = 0
            while true {
                let l = 2 * i + 1, r = l + 1
                var m = i
                if l < heap.count && heap[l].priority > heap[m].priority { m = l }
                if r < heap.count && heap[r].priority > heap[m].priority { m = r }
                if m == i { break }
                heap.swapAt(i, m)
                i = m
            }
            return top
        }
        // Candidates of node `i` for the tiles in `mask` (the ones it was just given).
        func consider(_ i: Int, _ mask: UInt16) {
            let level = keys[i].level, o = origin[i]
            if level < 1 { return }
            let size = o.w
            if level == 1 {
                // Level 0 by whole nodes, as the distance rule has it: quarters whose four tiles this node draws, within
                // the level-0 radius of the node.
                let d = horizontal(o.x, o.y, size)
                if d >= level0Radius { return }
                for q in 0..<4 where children[4 * i + q] >= 0 && mask & lodQuarterTiles[q] == lodQuarterTiles[q] {
                    push(Candidate(priority: 1e30 - d, node: Int32(i), t: Int8(q), kind: quarterK | forcedK))
                }
                return
            }
            let tileSize = o.z, l1 = Double(lodNodeVoxels << 1)
            for t in 0..<16 where mask & (1 << UInt16(t)) != 0 {
                let tx = t % 4, tz = t / 4
                if children[4 * i + (tz / 2) * 2 + tx / 2] < 0 { continue }
                let tx0 = o.x + Double(tx) * tileSize, tz0 = o.y + Double(tz) * tileSize
                // The level-0 ring's reach, as the distance rule has it: measured to the level-1 node the tile lies in
                // (tiles of level 3 and up are whole level-1 nodes already).
                let r0 = horizontal((tx0 / l1).rounded(.down) * l1, (tz0 / l1).rounded(.down) * l1, max(l1, tileSize))
                if r0 < level0Radius {
                    push(Candidate(priority: 1e30 - r0, node: Int32(i), t: Int8(t), kind: forcedK))
                    continue
                }
                let err = errors[16 * i + t]
                var priority: Double
                var kind: UInt8 = 0
                if err.isNaN {
                    // No error data under this tile: the distance rule, per tile (splits within its reach).
                    priority = errorPx * (splitFactor * size / 2) / max(1, horizontal(tx0, tz0, tileSize))
                    kind = fallbackK
                } else if err < 0 {
                    continue
                } else {
                    let b = box(i, t)
                    let dv = max(b.lo - cam, max(SIMD3(repeating: 0), cam - b.hi))
                    priority = Double(err) / max(1, (dv * dv).sum().squareRoot()) * pixelsPerRadian
                }
                if prev[i] & (1 << UInt16(t)) != 0 { priority /= lodSmartHysteresis }
                if priority < errorPx { continue }
                push(Candidate(priority: priority, node: Int32(i), t: Int8(t), kind: kind))
            }
        }
        var total = 0
        for r in g.roots {
            let i = Int(r)
            masks[i] = 0xFFFF
            for t in 0..<16 { total += cost(i, t) }
        }
        for r in g.roots { consider(Int(r), 0xFFFF) }
        var forcedSplits = 0, errorSplits = 0, overBudget = 0, fallbackSplits = 0
        while let c = pop() {
            let i = Int(c.node), t = Int(c.t)
            let quarter = c.kind & quarterK != 0
            let q = quarter ? t : ((t / 4) / 2) * 2 + (t % 4) / 2
            let ci = Int(children[4 * i + q])
            // The parent's tiles handed over, and the child's tiles that take them.
            let parentTiles: UInt16 = quarter ? lodQuarterTiles[q] : 1 << UInt16(t)
            let childTiles: UInt16 = quarter ? 0xFFFF : lodQuarterTiles[((t / 4) % 2) * 2 + (t % 4) % 2]
            if ci < 0 || masks[i] & parentTiles != parentTiles { continue }
            var delta = 0
            for ct in 0..<16 where childTiles & (1 << UInt16(ct)) != 0 { delta += cost(ci, ct) }
            for pt in 0..<16 where parentTiles & (1 << UInt16(pt)) != 0 { delta -= cost(i, pt) }
            let forced = c.kind & forcedK != 0
            let kept = !quarter && prev[i] & parentTiles != 0
            let limit = kept ? Int(Double(budget) * (1 + lodSmartBudgetSlack)) : budget
            if !forced && delta > 0 && total + delta > limit { overBudget += 1; continue }
            total += delta
            masks[i] &= ~parentTiles
            masks[ci] |= childTiles
            if forced { forcedSplits += 1 } else if c.kind & fallbackK != 0 { fallbackSplits += 1 } else { errorSplits += 1 }
            if !quarter { splitNow[i] |= parentTiles }
            consider(ci, childTiles)
        }
        split = splitNow
        if let stats {
            stats[0] = Int64(total); stats[1] = Int64(forcedSplits); stats[2] = Int64(errorSplits)
            stats[3] = Int64(overBudget); stats[4] = Int64(fallbackSplits)
        }
        var out: [(LodMeshNode, UInt16)] = []
        for i in 0..<count where masks[i] != 0 { out.append((g.nodes[i], masks[i])) }
        return out
    }

    /// Smart selection for a frame of mmc_lod_draw: the graph rebuilt when the mesh set changed, split state kept.
    static func selectSmartFrame(_ w: LodWorld, meshes: [LodNodeKey: LodMeshNode], generation: Int, cam: SIMD3<Double>,
                                 projView: simd_float4x4, zoom: Double) -> [(LodMeshNode, UInt16)] {
        let s = LodSmartState.shared
        let t0 = DispatchTime.now().uptimeNanoseconds
        let g = s.graph(worldId: w.id, generation: generation, meshes: meshes, maxLevel: w.maxLevel)
        // Pixels per radian at the view's center for vanilla's 70 degrees, times the zoom the distance rule counts
        // (the spyglass, a narrow field of view; not sprinting's small change).
        let ppr = lodSmartHalfPanel * 1.4281 * zoom
        let chosen = selectSmart(g, cam: cam, planes: lodFrustumPlanes(projView), pixelsPerRadian: ppr, errorPx: lodSmartErrorPx,
                                 budget: lodSmartBudget, level0Radius: Double(lodLevel0Radius), splitFactor: lodSplitFactor,
                                 split: &s.split, stats: &s.stats)
        let dt = DispatchTime.now().uptimeNanoseconds - t0
        s.frames += 1
        s.nanos += dt
        s.worstNanos = max(s.worstNanos, dt)
        if s.frames % 1000 == 0 {
            log("LOD smart: \(s.stats[0] / 1000) K quads counted of \(lodSmartBudget / 1000) K, splits: \(s.stats[1]) forced, \(s.stats[2]) by error, \(s.stats[4]) by distance (no error data), \(s.stats[3]) over the budget; \(s.nanos / 1000 / 1000) us per selection, worst \(s.worstNanos / 1000) us; last graph build \(s.lastBuildMicros) us (background)")
            s.nanos = 0
            s.worstNanos = 0
        }
        return chosen
    }
}

// MARK: - Offline checks (tools/smartlod.py)

/// A view for the offline selection checks: proj * view (camera-relative) for yaw / pitch in degrees (Minecraft's:
/// yaw 0 looks toward +z, 90 toward -x; pitch 90 looks straight down), vertical field of view `fovDeg`, the panel's
/// aspect.
func lodDebugProjView(yawDeg: Double, pitchDeg: Double, fovDeg: Double) -> simd_float4x4 {
    let yaw = yawDeg * .pi / 180, pitch = pitchDeg * .pi / 180
    let f = SIMD3<Float>(Float(-sin(yaw) * cos(pitch)), Float(-sin(pitch)), Float(cos(yaw) * cos(pitch)))
    let worldUp = SIMD3<Float>(0, 1, 0)
    var r = simd_cross(f, worldUp)
    if simd_length(r) < 1e-4 { r = SIMD3(Float(cos(yaw)), 0, Float(sin(yaw))) }
    r = simd_normalize(r)
    let u = simd_cross(r, f)
    // View: rows r, u, -f (the camera looks down -z).
    let view = simd_float4x4(rows: [SIMD4(r.x, r.y, r.z, 0), SIMD4(u.x, u.y, u.z, 0), SIMD4(-f.x, -f.y, -f.z, 0), SIMD4(0, 0, 0, 1)])
    let t = Float(1 / tan(fovDeg * .pi / 360)), aspect: Float = 3456.0 / 2234.0
    let near: Float = 0.05, far: Float = 100_000
    let proj = simd_float4x4(rows: [SIMD4(t / aspect, 0, 0, 0), SIMD4(0, t, 0, 0),
                                    SIMD4(0, 0, (far + near) / (near - far), 2 * far * near / (near - far)), SIMD4(0, 0, -1, 0)])
    return proj * view
}

/// Debug: every node's tile errors (lodTileErrors) of the active world: out[35 * i ...] = level, x, z, then RMS and
/// max per tile (x1000, blocks). Returns the count (at most `max`).
@_cdecl("mmc_debug_lod_tile_errors")
public func mmc_debug_lod_tile_errors(_ out: UnsafeMutablePointer<Int64>, _ max: Int32) -> Int32 {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return 0 }
    var i = 0
    for (k, n) in w.snapshot().meshes.sorted(by: { ($0.key.level, $0.key.x, $0.key.z) < ($1.key.level, $1.key.x, $1.key.z) }) where i < Int(max) {
        let o = out + 35 * i
        o[0] = Int64(k.level); o[1] = Int64(k.x); o[2] = Int64(k.z)
        for j in 0..<32 { o[3 + j] = n.tileError.count == 32 ? Int64((n.tileError[j] * 1000).rounded()) : -1 }
        i += 1
    }
    return Int32(i)
}

/// Debug: one frame's selection on the active world for a camera at (x, y, z) looking along yaw / pitch (degrees) with
/// a vertical field of view `fovDeg`: mode 0 is the distance rule (`select`, split factor `splitFactor`), 1 is smart
/// selection with `errorPx` and `budget` (fresh state), 2 the same keeping the split state across calls (a camera
/// path, for hysteresis). Writes (level, node x, node z, tile, facing quads in view, all quads, saturated error
/// x1000 or -1, horizontal distance, 3D distance to the tile's box) per chosen tile, at most `max`; stats[0...4] as
/// in selectSmart (smart modes), stats[5] = selection microseconds, stats[6] = microseconds to build the graph (0 when
/// the mesh set hasn't changed since the last call). Returns the count.
@_cdecl("mmc_debug_lod_select_smart")
public func mmc_debug_lod_select_smart(_ camX: Double, _ camY: Double, _ camZ: Double, _ yawDeg: Double, _ pitchDeg: Double,
                                       _ fovDeg: Double, _ mode: Int32, _ errorPx: Double, _ budget: Int64, _ splitFactor: Double,
                                       _ out: UnsafeMutablePointer<Int64>, _ max: Int32, _ stats: UnsafeMutablePointer<Int64>) -> Int32 {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return 0 }
    let snap = w.snapshot()
    let state = lodSmartDebugState
    let tg = DispatchTime.now().uptimeNanoseconds
    let g = state.graph(worldId: w.id, generation: snap.generation, meshes: snap.meshes, maxLevel: w.maxLevel)
    stats[6] = Int64((DispatchTime.now().uptimeNanoseconds - tg) / 1000)
    let pv = lodDebugProjView(yawDeg: yawDeg, pitchDeg: pitchDeg, fovDeg: fovDeg)
    let planes = lodFrustumPlanes(pv)
    let ppr = lodSmartHalfPanel / tan(fovDeg * .pi / 360)
    if mode == 1 { state.split = [] }
    let t0 = DispatchTime.now().uptimeNanoseconds
    let chosen = mode == 0
        ? LodRenderer.select(snap.meshes, maxLevel: w.maxLevel, camX: camX, camZ: camZ, splitFactor: splitFactor,
                             level0Radius: Double(lodLevel0Radius), zoom: 1)
        : LodRenderer.selectSmart(g, cam: SIMD3(camX, camY, camZ), planes: planes, pixelsPerRadian: ppr, errorPx: errorPx,
                                  budget: Int(budget), level0Radius: Double(lodLevel0Radius), splitFactor: splitFactor,
                                  split: &state.split, stats: stats)
    stats[5] = Int64((DispatchTime.now().uptimeNanoseconds - t0) / 1000)
    // Per chosen tile: its facing quads in view (the smart cost) and its saturated error.
    var i = 0
    for (n, mask) in chosen {
        let gi = g.index[LodNodeKey(level: n.level, x: n.x0 >> (8 + n.level), z: n.z0 >> (8 + n.level))].map { Int($0) }
        let voxel = Float(1 << n.level)
        let tile = Double(lodTileVoxels << n.level)
        for t in 0..<16 where mask & (1 << UInt16(t)) != 0 && i < Int(max) {
            var facing = 0
            let yMin = n.tileY[2 * t], yMax = n.tileY[2 * t + 1]
            let x0 = Double(n.x0) + Double(t % 4) * tile, z0 = Double(n.z0) + Double(t / 4) * tile
            if yMin <= yMax {
                let lo = SIMD3<Float>(Float(x0 - camX), Float(Double(lodWorldMinY) - camY) + Float(yMin) * voxel, Float(z0 - camZ))
                let hi = SIMD3<Float>(lo.x + Float(tile), Float(Double(lodWorldMinY) - camY) + Float(yMax) * voxel, lo.z + Float(tile))
                var inView = true
                for pl in planes {
                    let v = SIMD3(pl.x >= 0 ? hi.x : lo.x, pl.y >= 0 ? hi.y : lo.y, pl.z >= 0 ? hi.z : lo.z)
                    if pl.x * v.x + pl.y * v.y + pl.z * v.z + pl.w < 0 { inView = false }
                }
                if inView {
                    let fv = [0 > lo.x, 0 < hi.x, 0 > lo.y, 0 < hi.y, 0 > lo.z, 0 < hi.z]
                    for f in 0..<6 where fv[f] {
                        facing += n.start[lodBucketIndex(t, f + 1, 0)] - n.start[lodBucketIndex(t, f, 0)]
                        facing += n.start[lodBucketIndex(t, f + 7, 0)] - n.start[lodBucketIndex(t, f + 6, 0)]
                    }
                }
            }
            let dx = Swift.max(x0 - camX, 0, camX - (x0 + tile)), dz = Swift.max(z0 - camZ, 0, camZ - (z0 + tile))
            let y0 = Double(lodWorldMinY) + Double(yMin <= yMax ? yMin : 0) * Double(voxel)
            let y1 = Double(lodWorldMinY) + (yMin <= yMax ? Double(yMax) * Double(voxel) : Double(lodWorldHeight))
            let dy = Swift.max(y0 - camY, 0, camY - y1)
            let o = out + 9 * i
            o[0] = Int64(n.level); o[1] = Int64(n.x0 >> (8 + n.level)); o[2] = Int64(n.z0 >> (8 + n.level)); o[3] = Int64(t)
            o[4] = Int64(facing)
            o[5] = Int64(n.start[lodBucketIndex(t + 1, 0, 0)] - n.start[lodBucketIndex(t, 0, 0)])
            let et = gi.map { g.errors[16 * $0 + t] } ?? -1
            o[6] = et.isNaN || et < 0 ? -1 : Int64((et * 1000).rounded())
            o[7] = Int64((dx * dx + dz * dz).squareRoot())
            o[8] = Int64((dx * dx + dy * dy + dz * dz).squareRoot())
            i += 1
        }
    }
    return Int32(i)
}

/// State of the offline selection checks, apart from the game's.
let lodSmartDebugState = LodSmartState()

/// Debug: lodTileErrors of a made-up level-1 grid of stone, all columns solid from the bottom up to voxel `a` except:
/// kind 0 none (flat ground); kind 1 a one-voxel stone tower 20 voxels above it at (5, 5); kind 2 columns x >= `edge`
/// up to voxel `b` (a cliff). out[0 ..< 32] = RMS and max per tile, x1000 (blocks).
@_cdecl("mmc_debug_lod_tile_errors_synthetic")
public func mmc_debug_lod_tile_errors_synthetic(_ kind: Int32, _ a: Int32, _ b: Int32, _ edge: Int32, _ out: UnsafeMutablePointer<Int64>) {
    var g = LodGrid(level: 1)
    let n = lodNodeVoxels, stone = Mat.stone.rawValue
    for z in 0..<n {
        for x in 0..<n {
            let top = kind == 2 && x >= Int(edge) ? Int(b) : (kind == 1 && x == 5 && z == 5 ? Int(a) + 20 : Int(a))
            for y in 0...min(top, g.height - 1) { g.v[g.index(x, y, z)] = stone }
        }
    }
    let e = lodTileErrors(g, rule: .hybrid)
    for (j, v) in e.enumerated() { out[j] = Int64((v * 1000).rounded()) }
}

/// Debug: checks the game's background graph builds on the active world. A first graph is built at once; for a later mesh
/// set (the level-0 nodes dropped) the old graph is kept until the new one lands, then adopted with last frame's splits
/// carried over by node key. Returns 0 if all of that held, else the step that failed (1 old graph not kept, 2 new
/// graph never landed, 3 wrong graph, 4 splits not carried).
@_cdecl("mmc_debug_lod_smart_background_check")
public func mmc_debug_lod_smart_background_check() -> Int32 {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return -1 }
    let snap = w.snapshot()
    let s = LodSmartState(background: true)
    let g1 = s.graph(worldId: w.id, generation: snap.generation, meshes: snap.meshes, maxLevel: w.maxLevel)
    for (i, k) in g1.keys.enumerated() where k.level >= 2 { s.split[i] = 0xFFFF }
    let fewer = snap.meshes.filter { $0.key.level > 0 }
    if s.graph(worldId: w.id, generation: snap.generation + 1, meshes: fewer, maxLevel: w.maxLevel) !== g1 { return 1 }
    var g = g1
    for _ in 0..<200 where g === g1 {
        Thread.sleep(forTimeInterval: 0.005)
        g = s.graph(worldId: w.id, generation: snap.generation + 1, meshes: fewer, maxLevel: w.maxLevel)
    }
    if g === g1 { return 2 }
    if g.nodes.count != fewer.count { return 3 }
    for (i, k) in g.keys.enumerated() where s.split[i] != (k.level >= 2 ? 0xFFFF : 0) { return 4 }
    log("LOD smart: background graph landed (\(g1.nodes.count) -> \(g.nodes.count) nodes) in \(s.lastBuildMicros) us")
    return 0
}

/// Debug: builds the active world's LodSmartGraph `iterations` times; returns the mean microseconds per build.
@_cdecl("mmc_debug_lod_smart_graph_us")
public func mmc_debug_lod_smart_graph_us(_ iterations: Int32) -> Double {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w, iterations > 0 else { return 0 }
    let snap = w.snapshot()
    let t0 = DispatchTime.now().uptimeNanoseconds
    var roots = 0
    for _ in 0..<Int(iterations) { roots += LodSmartGraph(snap.meshes, maxLevel: w.maxLevel).roots.count }
    let us = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1000 / Double(iterations)
    log("LOD smart: graph of \(snap.meshes.count) nodes (\(roots / Int(iterations)) roots) built in \(String(format: "%.1f", us)) us")
    return us
}

/// Debug: the cost of lodTileErrors against meshing, on one region file at level 1 (or downsampled to `level`): out[0] =
/// fill + mesh microseconds, out[1] = tile error microseconds, out[2 ...] = the 32 tile errors x1000.
@_cdecl("mmc_debug_lod_tile_error_cost")
public func mmc_debug_lod_tile_error_cost(_ path: UnsafePointer<CChar>, _ level: Int32, _ out: UnsafeMutablePointer<Int64>) {
    guard var g = LodBuild.regionGrid(path: String(cString: path)) else { return }
    while g.level < Int(level) {
        var p = LodGrid(level: g.level + 1)
        g.downsample(into: &p, qx: 0, qz: 0)
        g = p
    }
    let t0 = DispatchTime.now().uptimeNanoseconds
    g.fillUnreachable()
    _ = LodBuild.mesh(g, maxMerge: 64, skyCover: lodSkyCover)
    let t1 = DispatchTime.now().uptimeNanoseconds
    let e = lodTileErrors(g, rule: lodDownsampleRule)
    let t2 = DispatchTime.now().uptimeNanoseconds
    out[0] = Int64((t1 - t0) / 1000); out[1] = Int64((t2 - t1) / 1000)
    for (j, v) in e.enumerated() { out[2 + j] = Int64((v * 1000).rounded()) }
}
