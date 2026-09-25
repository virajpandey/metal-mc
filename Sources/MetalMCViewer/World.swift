import Foundation
import MetalMCCore

/// Sparse sectioned voxel world. Each 16^3 section is either absent (all air) or
/// 4,096 material IDs stored contiguously in `storage`, indexed y*256 + z*16 + x
/// (the same order Minecraft uses inside a section).
final class World {
    let secX: Int
    let secY: Int
    let secZ: Int
    /// World-space block coordinates of local (0, 0, 0).
    let originX: Int
    let originZ: Int
    let minY: Int
    /// Local Y used to aim benchmark cameras (sea level).
    let referenceY: Int

    private(set) var table: [Int32]
    private(set) var storage: [UInt8] = []
    private(set) var sectionCount = 0

    var sizeX: Int { secX * 16 }
    var sizeY: Int { secY * 16 }
    var sizeZ: Int { secZ * 16 }

    init(secX: Int, secY: Int, secZ: Int, originX: Int, originZ: Int, minY: Int, referenceY: Int) {
        self.secX = secX
        self.secY = secY
        self.secZ = secZ
        self.originX = originX
        self.originZ = originZ
        self.minY = minY
        self.referenceY = referenceY
        table = [Int32](repeating: -1, count: secX * secY * secZ)
    }

    func setSection(sx: Int, sy: Int, sz: Int, data: [UInt8]) {
        precondition(data.count == 4096)
        let ti = (sy * secZ + sz) * secX + sx
        if table[ti] >= 0 {
            let start = Int(table[ti]) * 4096
            storage.replaceSubrange(start..<(start + 4096), with: data)
        } else {
            table[ti] = Int32(sectionCount)
            storage.append(contentsOf: data)
            sectionCount += 1
        }
    }

    func withView<R>(_ body: (WorldView) throws -> R) rethrows -> R {
        try table.withUnsafeBufferPointer { t in
            try storage.withUnsafeBufferPointer { s in
                try body(WorldView(table: t.baseAddress!, storage: s.baseAddress,
                                   secX: secX, secY: secY, secZ: secZ))
            }
        }
    }

    // MARK: Procedural test terrain (milestone 0)

    static let proceduralWaterLevel = 60

    static func height(_ x: Float, _ z: Float, seed: Float) -> Int {
        var h: Float = 64
        h += 14 * sin(x * 0.021 + seed) * cos(z * 0.017 + seed * 0.5)
        h += 7 * sin(x * 0.053 + z * 0.041 + seed * 1.3)
        h += 3 * cos(x * 0.11 - z * 0.093)
        return Int(h)
    }

    static func procedural(chunksX: Int, chunksZ: Int, height: Int = 128, seed: Float = 0) -> World {
        let secY = height / 16
        let world = World(secX: chunksX, secY: secY, secZ: chunksZ, originX: 0, originZ: 0,
                          minY: 0, referenceY: proceduralWaterLevel)
        let water = proceduralWaterLevel
        let columns = chunksX * chunksZ
        var results = [[(Int, [UInt8])]](repeating: [], count: columns)
        results.withUnsafeMutableBufferPointer { buf in
            let out = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: columns) { c in
                let sx = c % chunksX, sz = c / chunksX
                var heights = [Int](repeating: 0, count: 256)
                for lz in 0..<16 {
                    for lx in 0..<16 {
                        let h = World.height(Float(sx * 16 + lx), Float(sz * 16 + lz), seed: seed)
                        heights[lz * 16 + lx] = min(height - 1, max(1, h))
                    }
                }
                var column: [(Int, [UInt8])] = []
                for sy in 0..<secY {
                    var data = [UInt8](repeating: 0, count: 4096)
                    var any = false
                    for ly in 0..<16 {
                        let y = sy * 16 + ly
                        for lz in 0..<16 {
                            for lx in 0..<16 {
                                let h = heights[lz * 16 + lx]
                                var m = Mat.air
                                if y < h - 4 { m = .stone }
                                else if y < h { m = .dirt }
                                else if y == h { m = h <= water + 1 ? .sand : .grass }
                                else if y <= water { m = .water }
                                if m != .air {
                                    data[(ly << 8) | (lz << 4) | lx] = m.rawValue
                                    any = true
                                }
                            }
                        }
                    }
                    if any { column.append((sy, data)) }
                }
                out[c] = column
            }
        }
        for c in 0..<columns {
            for (sy, data) in results[c] {
                world.setSection(sx: c % chunksX, sy: sy, sz: c / chunksX, data: data)
            }
        }
        return world
    }
}

/// Pointer-based read view used by the mesher (no ARC traffic in hot loops).
struct WorldView {
    let table: UnsafePointer<Int32>
    let storage: UnsafePointer<UInt8>?
    let secX: Int
    let secY: Int
    let secZ: Int

    var sizeX: Int { secX * 16 }
    var sizeY: Int { secY * 16 }
    var sizeZ: Int { secZ * 16 }

    @inline(__always)
    func hasSection(_ sx: Int, _ sy: Int, _ sz: Int) -> Bool {
        table[(sy * secZ + sz) * secX + sx] >= 0
    }

    @inline(__always)
    func block(_ x: Int, _ y: Int, _ z: Int) -> UInt8 {
        if x < 0 || y < 0 || z < 0 || x >= secX * 16 || y >= secY * 16 || z >= secZ * 16 { return 0 }
        let si = table[((y >> 4) * secZ + (z >> 4)) * secX + (x >> 4)]
        if si < 0 { return 0 }
        return storage![(Int(si) << 12) | ((y & 15) << 8) | ((z & 15) << 4) | (x & 15)]
    }
}
