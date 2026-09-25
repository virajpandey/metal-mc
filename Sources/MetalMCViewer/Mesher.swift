import simd

/// One quad = one 32-bit word, expanded to 6 vertices by the vertex shader (vertex pulling).
///
///   bits  0-3   x within section        bits 12-14  face (0..5, order of `Mesher.faces`)
///   bits  4-7   y within section        bits 15-22  material ID
///   bits  8-11  z within section        bits 23-26  width - 1  (greedy extent along U)
///                                        bits 27-30  height - 1 (greedy extent along V)
///
/// The section origin comes from a per-draw `[[base_instance]]` lookup, so quads never
/// store world positions. 4 bytes per quad, down from 88 (4 x 16-byte vertices + 6 x 4-byte indices).
enum QuadFormat {
    @inline(__always)
    static func pack(x: Int, y: Int, z: Int, face: Int, material: Int, w: Int = 1, h: Int = 1) -> UInt32 {
        UInt32(x) | UInt32(y) << 4 | UInt32(z) << 8 | UInt32(face) << 12 | UInt32(material) << 15
            | UInt32(w - 1) << 23 | UInt32(h - 1) << 27
    }
}

struct SectionMesh {
    var minB: SIMD3<Float>
    var maxB: SIMD3<Float>
    /// Opaque quads sorted by face direction (+X, -X, +Y, -Y, +Z, -Z).
    var opaque: [UInt32] = []
    /// Cumulative start offsets of each face bucket within `opaque`; lane 6 = total count.
    var faceOffsets = SIMD8<Int32>(repeating: 0)
    var water: [UInt32] = []
}

/// Face-culling mesher: emits a quad for every opaque face that touches air or water,
/// and every water face that touches air. All lookup tables are read through raw pointers;
/// shared Swift arrays in the inner loop cause atomic refcount traffic that serializes threads.
enum Mesher {
    /// Face order is shared with the shader's corner table: +X, -X, +Y, -Y, +Z, -Z.
    struct Face {
        let dx: Int32, dy: Int32, dz: Int32
        let shade: Float
    }

    static let faces: [Face] = [
        Face(dx: 1, dy: 0, dz: 0, shade: 0.80),
        Face(dx: -1, dy: 0, dz: 0, shade: 0.80),
        Face(dx: 0, dy: 1, dz: 0, shade: 1.00),
        Face(dx: 0, dy: -1, dz: 0, shade: 0.50),
        Face(dx: 0, dy: 0, dz: 1, shade: 0.70),
        Face(dx: 0, dy: 0, dz: -1, shade: 0.70),
    ]

    static func pack(_ c: SIMD4<Float>) -> UInt32 {
        func q(_ v: Float) -> UInt32 { UInt32(max(0, min(255, (v * 255).rounded()))) }
        return q(c.x) | (q(c.y) << 8) | (q(c.z) << 16) | (q(c.w) << 24)
    }

    /// Packed color per (material * 6 + face). Uploaded to the GPU as a lookup table.
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
        var buckets: [[UInt32]] = [[], [], [], [], [], []]

        faces.withUnsafeBufferPointer { fp in
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
                                let q = QuadFormat.pack(x: x - x0, y: y - y0, z: z - z0, face: fi, material: b)
                                if isWater { mesh.water.append(q) } else { buckets[fi].append(q) }
                            }
                        }
                    }
                }
            }
        }
        var offset: Int32 = 0
        for fi in 0..<6 {
            mesh.faceOffsets[fi] = offset
            mesh.opaque.append(contentsOf: buckets[fi])
            offset += Int32(buckets[fi].count)
        }
        mesh.faceOffsets[6] = offset
        return mesh
    }
}
