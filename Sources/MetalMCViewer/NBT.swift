import Foundation

/// Minecraft's Named Binary Tag format (big-endian). Byte arrays keep only their length,
/// since nothing in the renderer reads them.
indirect enum NBT {
    case byte(Int8)
    case short(Int16)
    case int(Int32)
    case long(Int64)
    case float(Float)
    case double(Double)
    case byteArray(Int)
    case string(String)
    case list([NBT])
    case compound([String: NBT])
    case intArray([Int32])
    case longArray([Int64])

    subscript(key: String) -> NBT? {
        if case .compound(let d) = self { return d[key] }
        return nil
    }

    var intValue: Int? {
        switch self {
        case .byte(let v): return Int(v)
        case .short(let v): return Int(v)
        case .int(let v): return Int(v)
        case .long(let v): return Int(v)
        default: return nil
        }
    }

    var stringValue: String? {
        if case .string(let s) = self { return s }
        return nil
    }

    var listValue: [NBT]? {
        if case .list(let l) = self { return l }
        return nil
    }

    var longArrayValue: [Int64]? {
        if case .longArray(let a) = self { return a }
        return nil
    }

    var keys: [String] {
        if case .compound(let d) = self { return d.keys.sorted() }
        return []
    }
}

enum NBTError: Error {
    case truncated
    case badRoot(UInt8)
    case badTag(UInt8)
}

struct NBTReader {
    private let b: [UInt8]
    private var p = 0

    private init(_ bytes: [UInt8]) { b = bytes }

    static func parse(_ bytes: [UInt8]) throws -> NBT {
        var r = NBTReader(bytes)
        let t = try r.u8()
        guard t == 10 else { throw NBTError.badRoot(t) }
        _ = try r.str()
        return try r.payload(10)
    }

    private func need(_ n: Int) throws {
        if n < 0 || p + n > b.count { throw NBTError.truncated }
    }

    private mutating func u8() throws -> UInt8 {
        try need(1)
        let v = b[p]
        p += 1
        return v
    }

    private mutating func u16() throws -> UInt16 {
        try need(2)
        let v = UInt16(b[p]) << 8 | UInt16(b[p + 1])
        p += 2
        return v
    }

    private mutating func u32() throws -> UInt32 {
        try need(4)
        let v = UInt32(b[p]) << 24 | UInt32(b[p + 1]) << 16 | UInt32(b[p + 2]) << 8 | UInt32(b[p + 3])
        p += 4
        return v
    }

    private mutating func u64() throws -> UInt64 {
        let hi = try u32()
        let lo = try u32()
        return UInt64(hi) << 32 | UInt64(lo)
    }

    private mutating func str() throws -> String {
        let n = Int(try u16())
        try need(n)
        let s = String(decoding: b[p..<(p + n)], as: UTF8.self)
        p += n
        return s
    }

    private mutating func count() throws -> Int {
        let raw = try u32()
        return max(0, Int(Int32(bitPattern: raw)))
    }

    private mutating func payload(_ t: UInt8) throws -> NBT {
        switch t {
        case 1:
            let v = try u8()
            return .byte(Int8(bitPattern: v))
        case 2:
            let v = try u16()
            return .short(Int16(bitPattern: v))
        case 3:
            let v = try u32()
            return .int(Int32(bitPattern: v))
        case 4:
            let v = try u64()
            return .long(Int64(bitPattern: v))
        case 5:
            let v = try u32()
            return .float(Float(bitPattern: v))
        case 6:
            let v = try u64()
            return .double(Double(bitPattern: v))
        case 7:
            let n = try count()
            try need(n)
            p += n
            return .byteArray(n)
        case 8:
            return .string(try str())
        case 9:
            let et = try u8()
            let n = try count()
            if et == 0 || n == 0 { return .list([]) }
            var items: [NBT] = []
            items.reserveCapacity(n)
            for _ in 0..<n { items.append(try payload(et)) }
            return .list(items)
        case 10:
            var d: [String: NBT] = [:]
            while true {
                let ct = try u8()
                if ct == 0 { break }
                let name = try str()
                d[name] = try payload(ct)
            }
            return .compound(d)
        case 11:
            let n = try count()
            try need(n * 4)
            var a = [Int32]()
            a.reserveCapacity(n)
            for _ in 0..<n { a.append(Int32(bitPattern: try u32())) }
            return .intArray(a)
        case 12:
            let n = try count()
            try need(n * 8)
            var a = [Int64]()
            a.reserveCapacity(n)
            for _ in 0..<n { a.append(Int64(bitPattern: try u64())) }
            return .longArray(a)
        default:
            throw NBTError.badTag(t)
        }
    }
}
