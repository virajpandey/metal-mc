import Foundation

enum AnvilError: Error {
    case noRegionDir(String)
    case badChunk(String)
    case unsupportedCompression(UInt8)
    case noChunks(String)
}

/// Reads a Minecraft Java world save (Anvil region files) into a `World`.
/// Supports the 26.x layout (dimensions/minecraft/overworld/region) and the older region/ folder.
enum Anvil {
    static let minY = -64
    static let sectionsY = 24   // y -64 ..< 320
    static let seaLevel = 63

    struct ChunkResult {
        var cx: Int
        var cz: Int
        var status: String
        var sections: [(sy: Int, data: [UInt8])] = []
        var unknown: [String: Int] = [:]
        var unparsed = 0
        var unparsedSample = ""
    }

    static func regionDirectory(_ world: URL) -> URL? {
        for rel in ["dimensions/minecraft/overworld/region", "region"] {
            let u = world.appendingPathComponent(rel)
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: u.path, isDirectory: &isDir), isDir.boolValue { return u }
        }
        return nil
    }

    @inline(__always)
    static func be32(_ b: [UInt8], _ o: Int) -> UInt32 {
        UInt32(b[o]) << 24 | UInt32(b[o + 1]) << 16 | UInt32(b[o + 2]) << 8 | UInt32(b[o + 3])
    }

    /// Block-state palette entry name. Older saves use `{Name, Properties}` compounds. 26.x (DataVersion 5023)
    /// uses `{id, properties}`, stores property-less states as plain strings, and wraps those as
    /// `{"": "..."}` when a list mixes types.
    static func paletteName(_ entry: NBT) -> String? {
        if let s = entry.stringValue { return s }
        if let s = entry["id"]?.stringValue { return s }
        if let s = entry["Name"]?.stringValue { return s }
        if let s = entry[""]?.stringValue { return s }
        return nil
    }

    static func bitsNeeded(_ n: Int) -> Int {
        var b = 1
        while (1 << b) < n { b += 1 }
        return b
    }

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

    static func decodeChunk(region r: [UInt8], index i: Int) throws -> ChunkResult {
        let offset = Int(be32(r, i * 4) >> 8) * 4096
        guard offset >= 8192, offset + 5 <= r.count else { throw AnvilError.badChunk("offset \(offset)") }
        let length = Int(be32(r, offset))
        let ctype = r[offset + 4]
        let start = offset + 5, end = offset + 4 + length
        guard length > 1, end <= r.count else { throw AnvilError.badChunk("length \(length)") }

        let raw: [UInt8]
        switch ctype {
        case 2:
            // zlib = 2-byte header + raw deflate + 4-byte Adler-32; Foundation's .zlib wants raw deflate.
            guard end - start > 6 else { throw AnvilError.badChunk("short zlib") }
            let deflate = Data(r[(start + 2)..<(end - 4)])
            raw = [UInt8](try (deflate as NSData).decompressed(using: .zlib) as Data)
        case 3:
            raw = Array(r[start..<end])
        default:
            throw AnvilError.unsupportedCompression(ctype)
        }

        let root = try NBTReader.parse(raw)
        guard let cx = root["xPos"]?.intValue, let cz = root["zPos"]?.intValue else {
            throw AnvilError.badChunk("no xPos/zPos; keys=\(root.keys.prefix(15))")
        }
        var res = ChunkResult(cx: cx, cz: cz, status: root["Status"]?.stringValue ?? "none")
        guard res.status.hasSuffix("full") else { return res }
        guard let secs = root["sections"]?.listValue else {
            throw AnvilError.badChunk("no sections; keys=\(root.keys.prefix(15))")
        }

        var cache: [String: UInt8] = [:]
        for s in secs {
            guard let y = s["Y"]?.intValue else { continue }
            let sy = y - minY / 16
            guard sy >= 0, sy < sectionsY,
                  let states = s["block_states"],
                  let palette = states["palette"]?.listValue, !palette.isEmpty else { continue }

            var ids = [UInt8](repeating: 0, count: palette.count)
            var anySolid = false
            for (pi, entry) in palette.enumerated() {
                guard let name = paletteName(entry) else {
                    res.unparsed += 1
                    if res.unparsedSample.isEmpty {
                        let fields = entry.keys.map { k -> String in
                            if let v = entry[k]?.stringValue { return "\(k)=\(v)" }
                            if let v = entry[k]?.intValue { return "\(k)=\(v)" }
                            return "\(k)=<\(entry[k]!.keys.prefix(4).joined(separator: "|"))>"
                        }
                        res.unparsedSample = fields.joined(separator: " ")
                    }
                    continue
                }
                let id: UInt8
                if let c = cache[name] {
                    id = c
                } else {
                    id = Materials.classify(name).rawValue
                    cache[name] = id
                }
                if id == Mat.unknown.rawValue { res.unknown[name, default: 0] += 1 }
                ids[pi] = id
                if id != 0 { anySolid = true }
            }
            guard anySolid else { continue }

            var data = [UInt8](repeating: 0, count: 4096)
            if palette.count == 1 {
                data = [UInt8](repeating: ids[0], count: 4096)
            } else {
                guard let longs = states["data"]?.longArrayValue, !longs.isEmpty else { continue }
                let bits = max(4, bitsNeeded(palette.count))
                let perLong = 64 / bits
                let mask = (UInt64(1) << UInt64(bits)) - 1
                for idx in 0..<4096 {
                    let li = idx / perLong
                    if li >= longs.count { break }
                    let v = Int((UInt64(bitPattern: longs[li]) >> UInt64((idx % perLong) * bits)) & mask)
                    data[idx] = v < ids.count ? ids[v] : 0
                }
            }
            res.sections.append((sy, data))
        }
        return res
    }
}
