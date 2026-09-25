import Foundation
import MetalMCCore

/// Loads a whole world save into a `World` (the decoder lives in MetalMCCore).
extension Anvil {
    static func load(path: String) throws -> World {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let worldURL = URL(fileURLWithPath: path)
        guard let dir = regionDirectory(worldURL) else { throw AnvilError.noRegionDir(path) }

        let names = try FileManager.default.contentsOfDirectory(atPath: dir.path)
            .filter { $0.hasPrefix("r.") && $0.hasSuffix(".mca") }
            .sorted()
        var regions: [[UInt8]] = []
        for name in names {
            let d = try Data(contentsOf: dir.appendingPathComponent(name))
            if d.count >= 8192 { regions.append([UInt8](d)) }
        }

        var jobs: [(region: Int, index: Int)] = []
        for (ri, r) in regions.enumerated() {
            for i in 0..<1024 where be32(r, i * 4) != 0 { jobs.append((ri, i)) }
        }

        var results = [ChunkResult?](repeating: nil, count: jobs.count)
        var errors = [String?](repeating: nil, count: jobs.count)
        results.withUnsafeMutableBufferPointer { rb in
            errors.withUnsafeMutableBufferPointer { eb in
                let rp = rb.baseAddress!, ep = eb.baseAddress!
                DispatchQueue.concurrentPerform(iterations: jobs.count) { j in
                    do {
                        rp[j] = try decodeChunk(region: regions[jobs[j].region], index: jobs[j].index)
                    } catch {
                        ep[j] = "\(error)"
                    }
                }
            }
        }

        var chunks: [ChunkResult] = []
        var statusCounts: [String: Int] = [:]
        var unknown: [String: Int] = [:]
        var errorCount = 0
        var firstError = ""
        var unparsed = 0
        var unparsedSample = ""
        for j in 0..<jobs.count {
            if let r = results[j] {
                statusCounts[r.status, default: 0] += 1
                for (k, v) in r.unknown { unknown[k, default: 0] += v }
                unparsed += r.unparsed
                if unparsedSample.isEmpty { unparsedSample = r.unparsedSample }
                if r.status.hasSuffix("full") { chunks.append(r) }
            } else if let e = errors[j] {
                errorCount += 1
                if firstError.isEmpty { firstError = e }
            }
        }
        guard !chunks.isEmpty else {
            throw AnvilError.noChunks("jobs=\(jobs.count) errors=\(errorCount) first_error=\(firstError) statuses=\(statusCounts)")
        }

        let minCX = chunks.map(\.cx).min()!, maxCX = chunks.map(\.cx).max()!
        let minCZ = chunks.map(\.cz).min()!, maxCZ = chunks.map(\.cz).max()!
        let world = World(secX: maxCX - minCX + 1, secY: sectionsY, secZ: maxCZ - minCZ + 1,
                          originX: minCX * 16, originZ: minCZ * 16, minY: minY, referenceY: seaLevel - minY)
        for c in chunks {
            for s in c.sections {
                world.setSection(sx: c.cx - minCX, sy: s.sy, sz: c.cz - minCZ, data: s.data)
            }
        }

        let ms = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e6
        let topUnknown = unknown.sorted { $0.value > $1.value }.prefix(25)
            .map { "\($0.key.replacingOccurrences(of: "minecraft:", with: "")):\($0.value)" }
            .joined(separator: ",")
        let statuses = statusCounts.sorted { $0.value > $1.value }
            .map { "\($0.key.replacingOccurrences(of: "minecraft:", with: "")):\($0.value)" }
            .joined(separator: ",")
        print(String(format: "LOAD regions=%d chunk_slots=%d full_chunks=%d errors=%d sections=%d chunk_range=x[%d..%d]z[%d..%d] load_ms=%.0f",
                     regions.count, jobs.count, chunks.count, errorCount, world.sectionCount,
                     minCX, maxCX, minCZ, maxCZ, ms))
        print("LOAD statuses=\(statuses)")
        if errorCount > 0 { print("LOAD first_error=\(firstError)") }
        print("LOAD unknown_blocks=\(topUnknown.isEmpty ? "none" : topUnknown)")
        print("LOAD unparsed_palette_entries=\(unparsed)\(unparsed > 0 ? " sample=\(unparsedSample)" : "")")

        var histogram = [Int](repeating: 0, count: Mat.allCases.count)
        world.storage.withUnsafeBufferPointer { s in
            for b in s where Int(b) < histogram.count { histogram[Int(b)] += 1 }
        }
        let mats = Mat.allCases.filter { $0 != .air && histogram[Int($0.rawValue)] > 0 }
            .sorted { histogram[Int($0.rawValue)] > histogram[Int($1.rawValue)] }
            .map { "\($0):\(histogram[Int($0.rawValue)])" }
            .joined(separator: ",")
        print("LOAD materials=\(mats)")
        return world
    }

    /// Prints the NBT structure of the first fully generated chunk (debug aid for format changes).
    static func dump(path: String) throws {
        guard let dir = regionDirectory(URL(fileURLWithPath: path)) else { throw AnvilError.noRegionDir(path) }
        let name = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { $0.hasSuffix(".mca") }.sorted()[0]
        let r = [UInt8](try Data(contentsOf: dir.appendingPathComponent(name)))
        for i in 0..<1024 where be32(r, i * 4) != 0 {
            let offset = Int(be32(r, i * 4) >> 8) * 4096
            let length = Int(be32(r, offset))
            let deflate = Data(r[(offset + 7)..<(offset + 4 + length - 4)])
            let raw = [UInt8](try (deflate as NSData).decompressed(using: .zlib) as Data)
            let root = try NBTReader.parse(raw)
            guard root["Status"]?.stringValue?.hasSuffix("full") == true else { continue }
            print("DUMP region=\(name) index=\(i) root_keys=\(root.keys)")
            print("DUMP DataVersion=\(root["DataVersion"]?.intValue ?? -1)")
            if let secs = root["sections"]?.listValue {
                print("DUMP sections=\(secs.count)")
                for s in secs.prefix(12) {
                    let bs = s["block_states"]
                    let pal = bs?["palette"]?.listValue ?? []
                    let first = pal.first
                    print("DUMP  Y=\(s["Y"].map { "\($0)" } ?? "nil") keys=\(s.keys) bs_keys=\(bs?.keys ?? []) palette=\(pal.count) first_entry_keys=\(first?.keys ?? []) first=\(first.map { "\($0)" }?.prefix(160) ?? "")")
                }
            }
            return
        }
    }
}
