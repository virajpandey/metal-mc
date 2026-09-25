import simd

/// 16 bytes per vertex; matches `VertexIn` in the Metal shader (packed_float3 + uint).
struct PackedVertex {
    var x: Float
    var y: Float
    var z: Float
    var color: UInt32
}

struct SectionMesh {
    var minB: SIMD3<Float>
    var maxB: SIMD3<Float>
    var opaqueVerts: [PackedVertex] = []
    var opaqueIdx: [UInt32] = []
    var waterVerts: [PackedVertex] = []
    var waterIdx: [UInt32] = []
}

/// Naive face-culling mesher: emits a quad for every solid face that touches air (or water).
/// Greedy meshing and a compact vertex format come later.
enum Mesher {
    struct Face {
        let dx: Int, dy: Int, dz: Int
        let corners: [SIMD3<Float>]
        let shade: Float
    }

    static let faces: [Face] = [
        Face(dx: 1, dy: 0, dz: 0, corners: [SIMD3(1, 0, 0), SIMD3(1, 1, 0), SIMD3(1, 1, 1), SIMD3(1, 0, 1)], shade: 0.80),
        Face(dx: -1, dy: 0, dz: 0, corners: [SIMD3(0, 0, 1), SIMD3(0, 1, 1), SIMD3(0, 1, 0), SIMD3(0, 0, 0)], shade: 0.80),
        Face(dx: 0, dy: 1, dz: 0, corners: [SIMD3(0, 1, 0), SIMD3(0, 1, 1), SIMD3(1, 1, 1), SIMD3(1, 1, 0)], shade: 1.00),
        Face(dx: 0, dy: -1, dz: 0, corners: [SIMD3(0, 0, 0), SIMD3(1, 0, 0), SIMD3(1, 0, 1), SIMD3(0, 0, 1)], shade: 0.50),
        Face(dx: 0, dy: 0, dz: 1, corners: [SIMD3(1, 0, 1), SIMD3(1, 1, 1), SIMD3(0, 1, 1), SIMD3(0, 0, 1)], shade: 0.70),
        Face(dx: 0, dy: 0, dz: -1, corners: [SIMD3(0, 0, 0), SIMD3(0, 1, 0), SIMD3(1, 1, 0), SIMD3(1, 0, 0)], shade: 0.70),
    ]

    static func baseColor(_ b: UInt8) -> SIMD4<Float> {
        switch Block(rawValue: b) ?? .air {
        case .stone: return SIMD4(0.50, 0.50, 0.53, 1)
        case .dirt: return SIMD4(0.47, 0.33, 0.21, 1)
        case .grass: return SIMD4(0.37, 0.63, 0.26, 1)
        case .sand: return SIMD4(0.86, 0.80, 0.56, 1)
        case .water: return SIMD4(0.18, 0.38, 0.85, 0.62)
        case .air: return SIMD4(1, 0, 1, 1)
        }
    }

    static func pack(_ c: SIMD4<Float>) -> UInt32 {
        func q(_ v: Float) -> UInt32 { UInt32(max(0, min(255, (v * 255).rounded()))) }
        return q(c.x) | (q(c.y) << 8) | (q(c.z) << 16) | (q(c.w) << 24)
    }

    static func meshSection(world: World, sx: Int, sy: Int, sz: Int) -> SectionMesh {
        let x0 = sx * 16, y0 = sy * 16, z0 = sz * 16
        var mesh = SectionMesh(minB: SIMD3(Float(x0), Float(y0), Float(z0)),
                               maxB: SIMD3(Float(x0 + 16), Float(y0 + 16), Float(z0 + 16)))
        let air = Block.air.rawValue
        let water = Block.water.rawValue

        for y in y0..<(y0 + 16) {
            for z in z0..<(z0 + 16) {
                for x in x0..<(x0 + 16) {
                    let b = world.block(x, y, z)
                    if b == air { continue }
                    let isWater = b == water
                    let base = baseColor(b)
                    for f in faces {
                        let n = world.block(x + f.dx, y + f.dy, z + f.dz)
                        let visible = isWater ? (n == air) : (n == air || n == water)
                        if !visible { continue }
                        let c = pack(SIMD4(base.x * f.shade, base.y * f.shade, base.z * f.shade, base.w))
                        if isWater {
                            emit(&mesh.waterVerts, &mesh.waterIdx, x, y, z, f, c)
                        } else {
                            emit(&mesh.opaqueVerts, &mesh.opaqueIdx, x, y, z, f, c)
                        }
                    }
                }
            }
        }
        return mesh
    }

    @inline(__always)
    private static func emit(_ v: inout [PackedVertex], _ i: inout [UInt32],
                             _ x: Int, _ y: Int, _ z: Int, _ f: Face, _ c: UInt32) {
        let b = UInt32(v.count)
        for p in f.corners {
            v.append(PackedVertex(x: Float(x) + p.x, y: Float(y) + p.y, z: Float(z) + p.z, color: c))
        }
        i.append(b); i.append(b + 1); i.append(b + 2)
        i.append(b); i.append(b + 2); i.append(b + 3)
    }
}
