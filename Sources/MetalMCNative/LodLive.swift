import Compression
import Foundation
import MetalMCCore

// Live chunk ingestion. The mod hands over every overworld chunk the client loads or unloads, already
// classified into LOD materials. That keeps the LOD current without waiting for the server to save, and
// it is the only data source in multiplayer, where there are no region files. On a server the store is
// saved to disk per server, so the LOD survives restarts and grows as the player explores.
//
// Each chunk is kept at level-1 resolution (8 x 8 columns of 192 voxels), run-length encoded per column:
// about 1-2 KB per chunk instead of 12 KB.

let lodChunkVoxels = 8                                   // level-1 voxels per chunk side
let lodColumnHeight = lodWorldHeight >> 1                // level-1 voxels per column

/// One chunk's level-1 voxels, run-length encoded: 64 UInt16 column offsets (plus the end) into
/// (material, count) byte pairs, bottom up.
struct LodChunkColumns {
    var data: [UInt8]

    init(data: [UInt8]) { self.data = data }

    /// Encodes 8 x 8 x 192 voxels laid out (y, z, x).
    init(voxels v: UnsafeBufferPointer<UInt8>) {
        let n = lodChunkVoxels, h = lodColumnHeight
        var offsets = [UInt16](repeating: 0, count: n * n + 1)
        var runs: [UInt8] = []
        runs.reserveCapacity(n * n * 12)
        for c in 0..<(n * n) {
            offsets[c] = UInt16(runs.count)
            var y = 0
            while y < h {
                let m = v[y * n * n + c]
                var k = 1
                while y + k < h && k < 255 && v[(y + k) * n * n + c] == m { k += 1 }
                runs.append(m); runs.append(UInt8(k))
                y += k
            }
        }
        offsets[n * n] = UInt16(runs.count)
        var out = [UInt8](repeating: 0, count: offsets.count * 2)
        for (i, o) in offsets.enumerated() { out[2 * i] = UInt8(o & 255); out[2 * i + 1] = UInt8(o >> 8) }
        data = out + runs
    }

    /// Replaces this chunk's columns in a level-1 grid (the chunk's corner at voxel (x0, z0)).
    func write(into g: inout LodGrid, x0: Int, z0: Int) {
        let n = lodChunkVoxels, h = lodColumnHeight, header = (n * n + 1) * 2
        data.withUnsafeBufferPointer { d in
            g.v.withUnsafeMutableBufferPointer { dst in
                for c in 0..<(n * n) {
                    let col = (z0 + c / n) * lodNodeVoxels + x0 + c % n
                    var i = header + (Int(d[2 * c]) | Int(d[2 * c + 1]) << 8)
                    let end = header + (Int(d[2 * c + 2]) | Int(d[2 * c + 3]) << 8)
                    var y = 0
                    while i < end && y < h {
                        let m = d[i], k = Int(d[i + 1])
                        for j in 0..<k where y + j < h { dst[(y + j) * lodNodeVoxels * lodNodeVoxels + col] = m }
                        y += k
                        i += 2
                    }
                }
            }
        }
    }
}

final class LodLiveStore: @unchecked Sendable {
    private let lock = NSLock()
    private var regions: [Int64: [Int: LodChunkColumns]] = [:]   // region -> chunk index (z * 32 + x) -> columns
    private var dirty = Set<Int64>()
    private var unsaved = Set<Int64>()
    let saveDir: URL?

    init(saveDir: URL?) {
        self.saveDir = saveDir
    }

    func put(cx: Int, cz: Int, _ columns: LodChunkColumns) {
        let rk = LodBuild.key(cx >> 5, cz >> 5)
        lock.lock()
        regions[rk, default: [:]][(cz & 31) * 32 + (cx & 31)] = columns
        dirty.insert(rk)
        unsaved.insert(rk)
        lock.unlock()
    }

    /// Regions that received chunks since the last call.
    func takeDirty() -> Set<Int64> {
        lock.lock(); defer { lock.unlock() }
        let d = dirty
        dirty.removeAll()
        return d
    }

    var regionKeys: [Int64] {
        lock.lock(); defer { lock.unlock() }
        return Array(regions.keys)
    }

    var chunkCount: Int {
        lock.lock(); defer { lock.unlock() }
        return regions.values.reduce(0) { $0 + $1.count }
    }

    /// Writes every live chunk of a region over a level-1 grid. Returns true if the region has any.
    func overlay(regionX: Int, regionZ: Int, into g: inout LodGrid) -> Bool {
        lock.lock()
        let chunks = regions[LodBuild.key(regionX, regionZ)]
        lock.unlock()
        guard let chunks, !chunks.isEmpty else { return false }
        for (ci, cols) in chunks { cols.write(into: &g, x0: (ci % 32) * lodChunkVoxels, z0: (ci / 32) * lodChunkVoxels) }
        return true
    }

    // MARK: - Persistence (multiplayer): one file per region: "MMCL", version 2, the body's length (u32), then
    // the body compressed with LZFSE. The body is (chunk index u16, length u32, column data)*.

    func saveUnsaved() {
        guard let saveDir else { return }
        lock.lock()
        let keys = unsaved
        unsaved.removeAll()
        let snapshot = keys.map { ($0, regions[$0] ?? [:]) }
        lock.unlock()
        try? FileManager.default.createDirectory(at: saveDir, withIntermediateDirectories: true)
        for (k, chunks) in snapshot {
            var body: [UInt8] = []
            for (ci, cols) in chunks.sorted(by: { $0.key < $1.key }) {
                let n = cols.data.count
                body += [UInt8(ci & 255), UInt8(ci >> 8), UInt8(n & 255), UInt8((n >> 8) & 255), UInt8((n >> 16) & 255), UInt8(n >> 24)]
                body += cols.data
            }
            var packed = [UInt8](repeating: 0, count: body.count + 4096)
            let packedCount = body.withUnsafeBufferPointer { src in
                compression_encode_buffer(&packed, packed.count, src.baseAddress!, src.count, nil, COMPRESSION_LZFSE)
            }
            guard packedCount > 0 else { log("LOD: couldn't compress a saved region"); continue }
            let n = body.count
            let out: [UInt8] = Array("MMCL".utf8) + [2, UInt8(n & 255), UInt8((n >> 8) & 255), UInt8((n >> 16) & 255), UInt8(n >> 24)]
                + packed[0..<packedCount]
            let (x, z) = LodBuild.unkey(k)
            let url = saveDir.appendingPathComponent("r.\(x).\(z).lod")
            do { try Data(out).write(to: url, options: .atomic) } catch { log("LOD: couldn't save \(url.path): \(error)") }
        }
    }

    /// Loads every saved region; returns how many were read.
    func load() -> Int {
        guard let saveDir, let files = try? FileManager.default.contentsOfDirectory(atPath: saveDir.path) else { return 0 }
        var loaded = 0
        for f in files where f.hasSuffix(".lod") {
            let parts = f.split(separator: ".")
            guard parts.count == 4, let x = Int(parts[1]), let z = Int(parts[2]),
                  let d = FileManager.default.contents(atPath: saveDir.appendingPathComponent(f).path) else { continue }
            let raw = [UInt8](d)
            guard raw.count >= 9, raw[0..<4].elementsEqual("MMCL".utf8), raw[4] == 2 else { continue }
            let n = Int(raw[5]) | Int(raw[6]) << 8 | Int(raw[7]) << 16 | Int(raw[8]) << 24
            var b = [UInt8](repeating: 0, count: n)
            let got = raw.withUnsafeBufferPointer { src in
                compression_decode_buffer(&b, n, src.baseAddress! + 9, src.count - 9, nil, COMPRESSION_LZFSE)
            }
            guard got == n else { log("LOD: couldn't read \(f)"); continue }
            var chunks: [Int: LodChunkColumns] = [:]
            var i = 0
            while i + 6 <= b.count {
                let ci = Int(b[i]) | Int(b[i + 1]) << 8
                let n = Int(b[i + 2]) | Int(b[i + 3]) << 8 | Int(b[i + 4]) << 16 | Int(b[i + 5]) << 24
                i += 6
                guard ci < 1024, n >= (lodChunkVoxels * lodChunkVoxels + 1) * 2, i + n <= b.count else { break }
                chunks[ci] = LodChunkColumns(data: Array(b[i..<(i + n)]))
                i += n
            }
            let k = LodBuild.key(x, z)
            lock.lock()
            regions[k] = chunks
            dirty.insert(k)
            lock.unlock()
            loaded += 1
        }
        return loaded
    }
}

/// Material id for a block name ("minecraft:stone"), as used by the region-file reader.
@_cdecl("mmc_lod_classify")
public func mmc_lod_classify(_ name: UnsafePointer<CChar>) -> Int32 {
    Int32(Materials.classify(String(cString: name)).rawValue)
}

/// Biome tint class for a biome name ("minecraft:savanna").
@_cdecl("mmc_lod_tint_index")
public func mmc_lod_tint_index(_ name: UnsafePointer<CChar>) -> Int32 {
    Int32(lodTintIndex(String(cString: name)))
}

/// One loaded chunk: `blocks` is 16 x 16 x 384 material ids laid out (y from the world bottom, z, x);
/// `tints` is the 4 x 4 surface biome tint grid (z * 4 + x). Downsampled like the region reader: in each
/// 2 x 2 x 2 group the last non-air block in (y, z, x) order wins.
@_cdecl("mmc_lod_ingest")
public func mmc_lod_ingest(_ cx: Int32, _ cz: Int32, _ blocks: UnsafePointer<UInt8>, _ tints: UnsafePointer<UInt8>) {
    let r = LodRenderer.shared
    r.lock.lock(); let w = r.world; r.lock.unlock()
    guard let w else { return }
    let n = lodChunkVoxels, h = lodColumnHeight
    var v = [UInt8](repeating: 0, count: n * n * h)
    for by in 0..<lodWorldHeight {
        let vy = by >> 1
        for bz in 0..<16 {
            let row = (vy * n + (bz >> 1)) * n
            let src = (by * 16 + bz) * 16
            let trow = (bz >> 2) * 4
            for bx in 0..<16 {
                let m = blocks[src + bx]
                if m != 0 { v[row + (bx >> 1)] = lodTinted(m, tints[trow + (bx >> 2)]) }
            }
        }
    }
    let cols = v.withUnsafeBufferPointer { LodChunkColumns(voxels: $0) }
    if lodLiveCheck, let dir = w.regionDir { liveCheck(dir: dir, cx: Int(cx), cz: Int(cz), live: v) }
    w.live.put(cx: Int(cx), cz: Int(cz), cols)
}

// MARK: - METALMC_EXP=livecheck: compare each live chunk with the region file's copy of it.

let lodLiveCheck = experiments.contains("livecheck")
private let liveCheckLock = NSLock()
private var liveCheckStats = (chunks: 0, missing: 0, voxels: 0, mismatched: 0, badChunks: 0)
private let liveCheckScratch = ChunkScan.Scratch()

private func liveCheck(dir: URL, cx: Int, cz: Int, live: [UInt8]) {
    let path = dir.appendingPathComponent("r.\(cx >> 5).\(cz >> 5).mca").path
    liveCheckLock.lock(); defer { liveCheckLock.unlock() }
    guard let data = FileManager.default.contents(atPath: path) else { liveCheckStats.missing += 1; return }
    let r = [UInt8](data)
    let index = (cz & 31) * 32 + (cx & 31)
    var cache: [String: UInt8] = [:]
    guard Anvil.be32(r, index * 4) != 0,
          let chunk = r.withUnsafeBufferPointer({ try? ChunkScan.decodeChunk(region: $0, index: index, cache: &cache, scratch: liveCheckScratch) }),
          !chunk.sections.isEmpty else { liveCheckStats.missing += 1; return }
    // The region reader's downsampling, for this one chunk.
    let n = lodChunkVoxels
    var file = [UInt8](repeating: 0, count: n * n * lodColumnHeight)
    var tint = [UInt8](repeating: 0, count: 16)
    if chunk.surfaceBiomes.count == 16 { for i in 0..<16 { tint[i] = lodTintIndex(chunk.surfaceBiomes[i]) } }
    for (sy, blocks) in chunk.sections.sorted(by: { $0.sy < $1.sy }) {
        for by in 0..<16 {
            let vy = (sy * 16 + by) >> 1
            for bz in 0..<16 {
                for bx in 0..<16 {
                    let m = blocks[(by << 8) | (bz << 4) | bx]
                    if m != 0 { file[(vy * n + (bz >> 1)) * n + (bx >> 1)] = lodTinted(m, tint[(bz >> 2) * 4 + (bx >> 2)]) }
                }
            }
        }
    }
    var bad = 0
    for i in 0..<file.count where file[i] != live[i] {
        if bad < 3 && liveCheckStats.badChunks < 5 {
            let x = i % n, z = (i / n) % n, y = i / (n * n)
            log("livecheck: chunk (\(cx), \(cz)) voxel (\(x), \(y), \(z)): file \(file[i]) live \(live[i])")
        }
        bad += 1
    }
    liveCheckStats.chunks += 1
    liveCheckStats.voxels += file.count
    liveCheckStats.mismatched += bad
    if bad > 0 { liveCheckStats.badChunks += 1 }
    if liveCheckStats.chunks % 200 == 0 {
        log("livecheck: \(liveCheckStats.chunks) chunks compared, \(liveCheckStats.badChunks) differ, \(liveCheckStats.mismatched) of \(liveCheckStats.voxels) voxels differ, \(liveCheckStats.missing) not in the region file")
    }
}

/// Debug: feeds every full chunk of a region file through mmc_lod_ingest, the path live chunks take.
/// Returns the number of chunks ingested.
@_cdecl("mmc_debug_ingest_region")
public func mmc_debug_ingest_region(_ path: UnsafePointer<CChar>) -> Int32 {
    guard let data = FileManager.default.contents(atPath: String(cString: path)) else { return 0 }
    let r = [UInt8](data)
    var cache: [String: UInt8] = [:]
    let scratch = ChunkScan.Scratch()
    var blocks = [UInt8](repeating: 0, count: 16 * 16 * lodWorldHeight)
    var tints = [UInt8](repeating: 0, count: 16)
    var count: Int32 = 0
    for i in 0..<1024 where Anvil.be32(r, i * 4) != 0 {
        guard let chunk = r.withUnsafeBufferPointer({ try? ChunkScan.decodeChunk(region: $0, index: i, cache: &cache, scratch: scratch) }),
              !chunk.sections.isEmpty else { continue }
        for k in 0..<blocks.count { blocks[k] = 0 }
        for (sy, b) in chunk.sections { for k in 0..<4096 { blocks[sy * 4096 + k] = b[k] } }
        for k in 0..<16 { tints[k] = chunk.surfaceBiomes.count == 16 ? lodTintIndex(chunk.surfaceBiomes[k]) : 0 }
        mmc_lod_ingest(Int32(chunk.cx), Int32(chunk.cz), blocks, tints)
        count += 1
    }
    return count
}
