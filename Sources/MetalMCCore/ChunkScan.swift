import Compression
import Foundation

/// Fast chunk decoding for LOD ingestion: walks the NBT bytes once, reading only position, status,
/// block-state palettes and data, and the surface biome grid. Everything else (lighting, entities,
/// heightmaps, structures) is skipped by length without building the NBT tree, which is where
/// `NBTReader` spends most of its time. Produces the same `Anvil.ChunkResult` as `Anvil.decodeChunk`.
public enum ChunkScan {
    struct Cursor {
        let p: UnsafePointer<UInt8>
        let n: Int
        var i = 0

        @inline(__always) mutating func u8() throws -> UInt8 {
            guard i < n else { throw NBTError.truncated }
            defer { i += 1 }
            return p[i]
        }
        @inline(__always) mutating func i16() throws -> Int {
            guard i + 2 <= n else { throw NBTError.truncated }
            defer { i += 2 }
            return Int(Int16(bitPattern: UInt16(p[i]) << 8 | UInt16(p[i + 1])))
        }
        @inline(__always) mutating func i32() throws -> Int {
            guard i + 4 <= n else { throw NBTError.truncated }
            defer { i += 4 }
            let v = UInt32(p[i]) << 24 | UInt32(p[i + 1]) << 16 | UInt32(p[i + 2]) << 8 | UInt32(p[i + 3])
            return Int(Int32(bitPattern: v))
        }
        @inline(__always) func i64(at o: Int) -> Int64 {
            var v: UInt64 = 0
            for k in 0..<8 { v = v << 8 | UInt64(p[o + k]) }
            return Int64(bitPattern: v)
        }
        /// Tag name or string payload as (offset, length); no allocation.
        @inline(__always) mutating func str() throws -> (Int, Int) {
            let len = Int(UInt16(bitPattern: Int16(try i16())))
            guard i + len <= n else { throw NBTError.truncated }
            defer { i += len }
            return (i, len)
        }
        func equals(_ s: (Int, Int), _ lit: StaticString) -> Bool {
            guard s.1 == lit.utf8CodeUnitCount else { return false }
            let b = lit.utf8Start
            for k in 0..<s.1 where p[s.0 + k] != b[k] { return false }
            return true
        }
        func string(_ s: (Int, Int)) -> String {
            String(decoding: UnsafeBufferPointer(start: p + s.0, count: s.1), as: UTF8.self)
        }

        mutating func skip(_ type: UInt8) throws {
            switch type {
            case 1: i += 1
            case 2: i += 2
            case 3, 5: i += 4
            case 4, 6: i += 8
            case 7: let len = try i32(); i += len
            case 8: _ = try str()
            case 9:
                let et = try u8(); let len = try i32()
                switch et {
                case 0: break
                case 1: i += len
                case 2: i += 2 * len
                case 3, 5: i += 4 * len
                case 4, 6: i += 8 * len
                default: for _ in 0..<len { try skip(et) }
                }
            case 10:
                while true {
                    let t = try u8()
                    if t == 0 { break }
                    _ = try str()
                    try skip(t)
                }
            case 11: let len = try i32(); i += 4 * len
            case 12: let len = try i32(); i += 8 * len
            default: throw NBTError.badTag(type)
            }
            guard i <= n else { throw NBTError.truncated }
        }
    }

    /// Palette entry name: a plain string, or a compound with `id`, `Name` or `""`.
    static func paletteEntry(_ c: inout Cursor, _ type: UInt8) throws -> String {
        if type == 8 { return c.string(try c.str()) }
        var name = ""
        while true {
            let t = try c.u8()
            if t == 0 { break }
            let key = try c.str()
            if t == 8 && (c.equals(key, "id") || c.equals(key, "Name") || c.equals(key, "")) {
                name = c.string(try c.str())
            } else {
                try c.skip(t)
            }
        }
        return name
    }

    /// Reads a `{palette, data}` compound: palette names and the offset/count of the long array.
    static func paletted(_ c: inout Cursor) throws -> (names: [String], data: (Int, Int)) {
        var names: [String] = []
        var data = (0, 0)
        while true {
            let t = try c.u8()
            if t == 0 { break }
            let key = try c.str()
            if t == 9 && c.equals(key, "palette") {
                let et = try c.u8(); let len = try c.i32()
                names.reserveCapacity(len)
                for _ in 0..<len { names.append(try paletteEntry(&c, et)) }
            } else if t == 12 && c.equals(key, "data") {
                let len = try c.i32()
                data = (c.i, len)
                c.i += 8 * len
            } else {
                try c.skip(t)
            }
        }
        return (names, data)
    }

    /// Decodes one chunk's NBT (already decompressed). `cache` maps block names to material ids and can
    /// be shared across the chunks of a region.
    public static func decode(_ raw: [UInt8], cache: inout [String: UInt8]) throws -> Anvil.ChunkResult {
        try raw.withUnsafeBufferPointer { try decode($0, cache: &cache) }
    }

    public static func decode(_ buf: UnsafeBufferPointer<UInt8>, cache: inout [String: UInt8]) throws -> Anvil.ChunkResult {
        do {
            var c = Cursor(p: buf.baseAddress!, n: buf.count)
            guard try c.u8() == 10 else { throw NBTError.badRoot(buf[0]) }
            _ = try c.str()
            var cx: Int?, cz: Int?, status = "none"
            struct Sec { var y: Int; var blocks: ([String], (Int, Int))?; var biomes: ([String], (Int, Int))? }
            var secs: [Sec] = []
            while true {
                let t = try c.u8()
                if t == 0 { break }
                let key = try c.str()
                if t == 3 && c.equals(key, "xPos") { cx = try c.i32() }
                else if t == 3 && c.equals(key, "zPos") { cz = try c.i32() }
                else if t == 8 && c.equals(key, "Status") { status = c.string(try c.str()) }
                else if t == 9 && c.equals(key, "sections") {
                    let et = try c.u8(); let len = try c.i32()
                    guard et == 10 else { for _ in 0..<len { try c.skip(et) }; continue }
                    for _ in 0..<len {
                        var s = Sec(y: Int.min)
                        while true {
                            let st = try c.u8()
                            if st == 0 { break }
                            let sk = try c.str()
                            if st == 1 && c.equals(sk, "Y") { s.y = Int(Int8(bitPattern: try c.u8())) }
                            else if st == 10 && c.equals(sk, "block_states") { s.blocks = try paletted(&c) }
                            else if st == 10 && c.equals(sk, "biomes") { s.biomes = try paletted(&c) }
                            else { try c.skip(st) }
                        }
                        secs.append(s)
                    }
                } else {
                    try c.skip(t)
                }
            }
            guard let cx, let cz else { throw AnvilError.badChunk("no xPos/zPos") }
            var res = Anvil.ChunkResult(cx: cx, cz: cz, status: status)
            guard status.hasSuffix("full") else { return res }

            // Surface biomes from the section nearest Y = 6 (world y 96-111).
            if let bs = secs.filter({ $0.biomes != nil }).min(by: { abs($0.y - 6) < abs($1.y - 6) })?.biomes, !bs.0.isEmpty {
                if bs.0.count == 1 {
                    res.surfaceBiomes = [String](repeating: bs.0[0], count: 16)
                } else if bs.1.1 > 0 {
                    let bits = Anvil.bitsNeeded(bs.0.count), perLong = 64 / bits
                    let mask = (UInt64(1) << UInt64(bits)) - 1
                    res.surfaceBiomes = (0..<16).map { idx in
                        let li = idx / perLong
                        guard li < bs.1.1 else { return bs.0[0] }
                        let v = Int((UInt64(bitPattern: c.i64(at: bs.1.0 + 8 * li)) >> UInt64((idx % perLong) * bits)) & mask)
                        return v < bs.0.count ? bs.0[v] : bs.0[0]
                    }
                }
            }

            for s in secs {
                guard s.y != Int.min, let (names, data) = s.blocks, !names.isEmpty else { continue }
                let sy = s.y - Anvil.minY / 16
                guard sy >= 0, sy < Anvil.sectionsY else { continue }
                var ids = [UInt8](repeating: 0, count: names.count)
                var anySolid = false
                for (pi, name) in names.enumerated() {
                    let id: UInt8
                    if let hit = cache[name] { id = hit } else { id = Materials.classify(name).rawValue; cache[name] = id }
                    if id == Mat.unknown.rawValue { res.unknown[name, default: 0] += 1 }
                    ids[pi] = id
                    if id != 0 { anySolid = true }
                }
                guard anySolid else { continue }
                var out = [UInt8](repeating: 0, count: 4096)
                if names.count == 1 {
                    out = [UInt8](repeating: ids[0], count: 4096)
                } else {
                    guard data.1 > 0 else { continue }
                    let bits = max(4, Anvil.bitsNeeded(names.count)), perLong = 64 / bits
                    let mask = (UInt64(1) << UInt64(bits)) - 1
                    out.withUnsafeMutableBufferPointer { o in
                        var idx = 0
                        for li in 0..<data.1 {
                            var word = UInt64(bitPattern: c.i64(at: data.0 + 8 * li))
                            for _ in 0..<perLong {
                                if idx >= 4096 { break }
                                let v = Int(word & mask)
                                o[idx] = v < ids.count ? ids[v] : 0
                                word >>= UInt64(bits)
                                idx += 1
                            }
                        }
                    }
                }
                res.sections.append((sy, out))
            }
            return res
        }
    }

    /// Reusable decompression buffer for one thread (grows as needed).
    public final class Scratch {
        var buffer: UnsafeMutablePointer<UInt8>
        var capacity: Int
        public init(capacity: Int = 1 << 20) {
            self.capacity = capacity
            buffer = .allocate(capacity: capacity)
        }
        deinit { buffer.deallocate() }
        func grow() {
            buffer.deallocate()
            capacity *= 2
            buffer = .allocate(capacity: capacity)
        }
    }

    /// Like `decodeChunk(region:index:cache:)`, but inflates with the Compression framework straight into
    /// `scratch` and scans in place (no intermediate Data or array copies).
    public static func decodeChunk(region r: UnsafeBufferPointer<UInt8>, index i: Int, cache: inout [String: UInt8],
                                   scratch: Scratch) throws -> Anvil.ChunkResult {
        func be32(_ o: Int) -> Int { Int(UInt32(r[o]) << 24 | UInt32(r[o + 1]) << 16 | UInt32(r[o + 2]) << 8 | UInt32(r[o + 3])) }
        let offset = (be32(i * 4) >> 8) * 4096
        guard offset >= 8192, offset + 5 <= r.count else { throw AnvilError.badChunk("offset \(offset)") }
        let length = be32(offset)
        let ctype = r[offset + 4]
        let start = offset + 5, end = offset + 4 + length
        guard length > 1, end <= r.count else { throw AnvilError.badChunk("length \(length)") }
        switch ctype {
        case 2:
            // zlib = 2-byte header + raw deflate + 4-byte Adler-32; COMPRESSION_ZLIB wants raw deflate.
            guard end - start > 6 else { throw AnvilError.badChunk("short zlib") }
            let src = r.baseAddress! + start + 2
            let srcLen = end - start - 6
            while true {
                let n = compression_decode_buffer(scratch.buffer, scratch.capacity, src, srcLen, nil, COMPRESSION_ZLIB)
                if n == 0 { throw AnvilError.badChunk("inflate failed") }
                if n < scratch.capacity { return try decode(UnsafeBufferPointer(start: scratch.buffer, count: n), cache: &cache) }
                scratch.grow()   // output may have been truncated
            }
        case 3:
            return try decode(UnsafeBufferPointer(start: r.baseAddress! + start, count: end - start), cache: &cache)
        default:
            throw AnvilError.unsupportedCompression(ctype)
        }
    }

    /// Decompresses and decodes chunk `index` of a region file's bytes.
    public static func decodeChunk(region r: [UInt8], index i: Int, cache: inout [String: UInt8]) throws -> Anvil.ChunkResult {
        let offset = Int(Anvil.be32(r, i * 4) >> 8) * 4096
        guard offset >= 8192, offset + 5 <= r.count else { throw AnvilError.badChunk("offset \(offset)") }
        let length = Int(Anvil.be32(r, offset))
        let ctype = r[offset + 4]
        let start = offset + 5, end = offset + 4 + length
        guard length > 1, end <= r.count else { throw AnvilError.badChunk("length \(length)") }
        let raw: [UInt8]
        switch ctype {
        case 2:
            guard end - start > 6 else { throw AnvilError.badChunk("short zlib") }
            let deflate = Data(r[(start + 2)..<(end - 4)])
            raw = [UInt8](try (deflate as NSData).decompressed(using: .zlib) as Data)
        case 3:
            raw = Array(r[start..<end])
        default:
            throw AnvilError.unsupportedCompression(ctype)
        }
        return try decode(raw, cache: &cache)
    }
}
