import Foundation
import MetalMCCore

/// One far-terrain ring: a voxel grid where each cell covers `scale` x `scale` x `scale` blocks.
/// Cells inside the next-finer ring are left empty, so every ring emits skirt walls on both its
/// inner and outer edges, which hides most cracks between levels.
struct LODLevel {
    let world: World
    /// Horizontal cell size in blocks.
    let scale: Int
    /// Vertical cell size in blocks. Kept small at every distance so hills don't flatten out.
    let vScale: Int
    /// World-space block coordinates of the level grid's (0, 0, 0).
    let originX: Int
    let originZ: Int
}

/// Milestone 5 prototype: heightfield LOD rings built straight from the terrain function
/// (the Distant Horizons-style "sample the noise" approach), meshed by the regular greedy mesher
/// and drawn with the same 4-byte quads plus a per-section scale.
enum LOD {
    /// Builds `levels` rings around a procedural near world of `nearSize` blocks. Ring k uses
    /// 2^k-block cells and covers Chebyshev radius (nearSize/2) * 2^(k-1) ..< (nearSize/2) * 2^k
    /// around the near world's center, so every ring is a 512 x 512-cell grid with a 256 x 256 hole.
    static func buildProcedural(nearSize: Int, height: Int, levels: Int, maxVScale: Int = 2,
                                seed: Float = 0) -> [LODLevel] {
        guard levels > 0 else { return [] }
        let center = nearSize / 2
        let water = World.proceduralWaterLevel
        var out: [LODLevel] = []

        for k in 1...levels {
            let s = 1 << k
            let v = min(s, maxVScale)
            let rInner = (nearSize / 2) << (k - 1)
            let rOuter = rInner * 2
            let cells = 2 * rOuter / s
            let secXZ = cells / 16
            let secY = max(1, (height / v + 15) / 16)
            let originX = center - rOuter, originZ = center - rOuter
            let holeLo = (rOuter - rInner) / s, holeHi = (rOuter + rInner) / s   // hole in cell units
            let world = World(secX: secXZ, secY: secY, secZ: secXZ, originX: originX, originZ: originZ,
                              minY: 0, referenceY: water)
            let waterCell = water / v

            let columns = secXZ * secXZ
            var results = [[(Int, [UInt8])]](repeating: [], count: columns)
            results.withUnsafeMutableBufferPointer { buf in
                let outp = buf.baseAddress!
                DispatchQueue.concurrentPerform(iterations: columns) { c in
                    let sx = c % secXZ, sz = c / secXZ
                    var heights = [Int](repeating: -1, count: 256)   // -1 = hole
                    for lz in 0..<16 {
                        for lx in 0..<16 {
                            let cx = sx * 16 + lx, cz = sz * 16 + lz
                            if cx >= holeLo && cx < holeHi && cz >= holeLo && cz < holeHi { continue }
                            let wx = Float(originX + cx * s) + Float(s) / 2
                            let wz = Float(originZ + cz * s) + Float(s) / 2
                            let h = min(height - 1, max(1, World.height(wx, wz, seed: seed)))
                            heights[lz * 16 + lx] = h / v
                        }
                    }
                    var column: [(Int, [UInt8])] = []
                    for sy in 0..<secY {
                        var data = [UInt8](repeating: 0, count: 4096)
                        var any = false
                        for ly in 0..<16 {
                            let cy = sy * 16 + ly
                            for lz in 0..<16 {
                                for lx in 0..<16 {
                                    let hc = heights[lz * 16 + lx]
                                    if hc < 0 { continue }
                                    var m = Mat.air
                                    if cy < hc { m = .stone }
                                    else if cy == hc { m = hc <= waterCell ? .sand : .grass }
                                    else if cy <= waterCell { m = .water }
                                    if m != .air {
                                        data[(ly << 8) | (lz << 4) | lx] = m.rawValue
                                        any = true
                                    }
                                }
                            }
                        }
                        if any { column.append((sy, data)) }
                    }
                    outp[c] = column
                }
            }
            for c in 0..<columns {
                for (sy, data) in results[c] {
                    world.setSection(sx: c % secXZ, sy: sy, sz: c / secXZ, data: data)
                }
            }
            out.append(LODLevel(world: world, scale: s, vScale: v, originX: originX, originZ: originZ))
        }
        return out
    }
}
