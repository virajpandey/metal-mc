import Foundation

enum Block: UInt8 {
    case air = 0, stone, dirt, grass, sand, water
}

/// Dense voxel world. Milestone 0 fills it procedurally; milestone 1 will load Anvil region files instead.
final class World {
    let sizeX: Int
    let sizeY: Int
    let sizeZ: Int
    private(set) var blocks: [UInt8]

    static let waterLevel = 60

    init(chunksX: Int, chunksZ: Int, height: Int = 128, seed: Float = 0) {
        sizeX = chunksX * 16
        sizeZ = chunksZ * 16
        sizeY = height
        blocks = [UInt8](repeating: 0, count: sizeX * sizeY * sizeZ)
        generate(seed: seed)
    }

    @inline(__always)
    func block(_ x: Int, _ y: Int, _ z: Int) -> UInt8 {
        if x < 0 || y < 0 || z < 0 || x >= sizeX || y >= sizeY || z >= sizeZ { return Block.air.rawValue }
        return blocks[(y * sizeZ + z) * sizeX + x]
    }

    static func height(_ x: Float, _ z: Float, seed: Float) -> Int {
        var h: Float = 64
        h += 14 * sin(x * 0.021 + seed) * cos(z * 0.017 + seed * 0.5)
        h += 7 * sin(x * 0.053 + z * 0.041 + seed * 1.3)
        h += 3 * cos(x * 0.11 - z * 0.093)
        return Int(h)
    }

    private func generate(seed: Float) {
        let sx = sizeX, sy = sizeY, sz = sizeZ
        let water = World.waterLevel
        blocks.withUnsafeMutableBufferPointer { buf in
            let p = buf.baseAddress!
            DispatchQueue.concurrentPerform(iterations: sz) { z in
                for x in 0..<sx {
                    let h = min(sy - 1, max(1, World.height(Float(x), Float(z), seed: seed)))
                    for y in 0..<sy {
                        var b = Block.air
                        if y < h - 4 { b = .stone }
                        else if y < h { b = .dirt }
                        else if y == h { b = h <= water + 1 ? .sand : .grass }
                        else if y <= water { b = .water }
                        p[(y * sz + z) * sx + x] = b.rawValue
                    }
                }
            }
        }
    }
}
