import Foundation

public enum AnvilError: Error {
    case noRegionDir(String)
    case badChunk(String)
    case unsupportedCompression(UInt8)
    case noChunks(String)
}

/// Minecraft Java world saves (Anvil region files): region lookup and per-chunk decoding into
/// 16x16x16 material sections.
/// Supports the 26.x layout (dimensions/minecraft/overworld/region) and the older region/ folder.
public enum Anvil {
    public static let minY = -64
    public static let sectionsY = 24   // y -64 ..< 320
    public static let seaLevel = 63

    public struct ChunkResult {
        public var cx: Int
        public var cz: Int
        public var status: String
        public var sections: [(sy: Int, data: [UInt8])] = []
        public var unknown: [String: Int] = [:]
        public var unparsed = 0
        public var unparsedSample = ""
    }

    public static func regionDirectory(_ world: URL) -> URL? {
        for rel in ["dimensions/minecraft/overworld/region", "region"] {
            let u = world.appendingPathComponent(rel)
            var isDir: ObjCBool = false
            if FileManager.default.fileExists(atPath: u.path, isDirectory: &isDir), isDir.boolValue { return u }
        }
        return nil
    }

    @inline(__always)
    public static func be32(_ b: [UInt8], _ o: Int) -> UInt32 {
        UInt32(b[o]) << 24 | UInt32(b[o + 1]) << 16 | UInt32(b[o + 2]) << 8 | UInt32(b[o + 3])
    }

    /// Block-state palette entry name. Older saves use `{Name, Properties}` compounds. 26.x (DataVersion 5023)
    /// uses `{id, properties}`, stores property-less states as plain strings, and wraps those as
    /// `{"": "..."}` when a list mixes types.
    public static func paletteName(_ entry: NBT) -> String? {
        if let s = entry.stringValue { return s }
        if let s = entry["id"]?.stringValue { return s }
        if let s = entry["Name"]?.stringValue { return s }
        if let s = entry[""]?.stringValue { return s }
        return nil
    }

    public static func bitsNeeded(_ n: Int) -> Int {
        var b = 1
        while (1 << b) < n { b += 1 }
        return b
    }

    public static func decodeChunk(region r: [UInt8], index i: Int) throws -> ChunkResult {
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
