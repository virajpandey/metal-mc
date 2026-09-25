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

/// Naive face-culling mesher: emits a quad for every opaque face that touches air or water,
/// and every water face that touches air. Greedy meshing and a compact format come in milestone 2.
///
/// All lookup tables are read through raw pointers: shared Swift arrays in the inner loop cause
/// atomic refcount traffic that serializes the worker threads.
enum Mesher {
    /// Plain-old-data face description (no arrays, so no refcounting).
    struct Face {
        let dx: Int32, dy: Int32, dz: Int32
        let c0: SIMD3<Float>, c1: SIMD3<Float>, c2: SIMD3<Float>, c3: SIMD3<Float>
        let shade: Float
    }

    static let faces: [Face] = [
        Face(dx: 1, dy: 0, dz: 0, c0: SIMD3(1, 0, 0), c1: SIMD3(1, 1, 0), c2: SIMD3(1, 1, 1), c3: SIMD3(1, 0, 1), shade: 0.80),
        Face(dx: -1, dy: 0, dz: 0, c0: SIMD3(0, 0, 1), c1: SIMD3(0, 1, 1), c2: SIMD3(0, 1, 0), c3: SIMD3(0, 0, 0), shade: 0.80),
        Face(dx: 0, dy: 1, dz: 0, c0: SIMD3(0, 1, 0), c1: SIMD3(0, 1, 1), c2: SIMD3(1, 1, 1), c3: SIMD3(1, 1, 0), shade: 1.00),
        Face(dx: 0, dy: -1, dz: 0, c0: SIMD3(0, 0, 0), c1: SIMD3(1, 0, 0), c2: SIMD3(1, 0, 1), c3: SIMD3(0, 0, 1), shade: 0.50),
        Face(dx: 0, dy: 0, dz: 1, c0: SIMD3(1, 0, 1), c1: SIMD3(1, 1, 1), c2: SIMD3(0, 1, 1), c3: SIMD3(0, 0, 1), shade: 0.70),
        Face(dx: 0, dy: 0, dz: -1, c0: SIMD3(0, 0, 0), c1: SIMD3(0, 1, 0), c2: SIMD3(1, 1, 0), c3: SIMD3(1, 0, 0), shade: 0.70),
    ]

    static func pack(_ c: SIMD4<Float>) -> UInt32 {
        func q(_ v: Float) -> UInt32 { UInt32(max(0, min(255, (v * 255).rounded()))) }
        return q(c.x) | (q(c.y) << 8) | (q(c.z) << 16) | (q(c.w) << 24)
    }

    /// Packed color per (material * 6 + face), so the inner loop does no float math.
    static let faceColors: [UInt32] = Materials.colors.flatMap { base in
        faces.map { f in pack(SIMD4(base.x * f.shade, base.y * f.shade, base.z * f.shade, base.w)) }
    }

    /// MaterialKind raw value per material ID.
    static let kinds: [UInt8] = Materials.kinds.map(\.rawValue)

    static func meshSection(view: WorldView, sx: Int, sy: Int, sz: Int) -> SectionMesh {
        let x0 = sx * 16, y0 = sy * 16, z0 = sz * 16
        var mesh = SectionMesh(minB: SIMD3(Float(x0), Float(y0), Float(z0)),
                               maxB: SIMD3(Float(x0 + 16), Float(y0 + 16), Float(z0 + 16)))
        guard view.hasSection(sx, sy, sz) else { return mesh }

        let air = MaterialKind.air.rawValue
        let water = MaterialKind.water.rawValue
        let opaque = MaterialKind.opaque.rawValue

        faces.withUnsafeBufferPointer { fp in
            faceColors.withUnsafeBufferPointer { cp in
                kinds.withUnsafeBufferPointer { kp in
                    for y in y0..<(y0 + 16) {
                        for z in z0..<(z0 + 16) {
                            for x in x0..<(x0 + 16) {
                                let b = Int(view.block(x, y, z))
                                if b == 0 { continue }
                                let isWater = kp[b] == water
                                for fi in 0..<6 {
                                    let f = fp[fi]
                                    let nk = kp[Int(view.block(x + Int(f.dx), y + Int(f.dy), z + Int(f.dz)))]
                                    let visible = isWater ? (nk == air) : (nk != opaque)
                                    if !visible { continue }
                                    let c = cp[b * 6 + fi]
                                    if isWater {
                                        emit(&mesh.waterVerts, &mesh.waterIdx, x, y, z, f, c)
                                    } else {
                                        emit(&mesh.opaqueVerts, &mesh.opaqueIdx, x, y, z, f, c)
                                    }
                                }
                            }
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
        let o = SIMD3<Float>(Float(x), Float(y), Float(z))
        let q0 = o + f.c0, q1 = o + f.c1, q2 = o + f.c2, q3 = o + f.c3
        v.append(PackedVertex(x: q0.x, y: q0.y, z: q0.z, color: c))
        v.append(PackedVertex(x: q1.x, y: q1.y, z: q1.z, color: c))
        v.append(PackedVertex(x: q2.x, y: q2.y, z: q2.z, color: c))
        v.append(PackedVertex(x: q3.x, y: q3.y, z: q3.z, color: c))
        i.append(b); i.append(b + 1); i.append(b + 2)
        i.append(b); i.append(b + 2); i.append(b + 3)
    }
}
