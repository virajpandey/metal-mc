import simd

/// One quad = one 32-bit word, expanded to 4 vertices by the vertex shader (vertex pulling).
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

/// Face-culling mesher with optional greedy merging. For each face direction and each of the 16
/// slices along its normal, it builds a 16x16 mask of visible faces (by material), then merges runs
/// of equal material into rectangles up to 16x16. Lookup tables are read through raw pointers;
/// shared Swift arrays in the inner loop cause atomic refcount traffic that serializes threads.
enum Mesher {
    /// Face order is shared with the shader's corner table: +X, -X, +Y, -Y, +Z, -Z.
    /// `n`, `u`, `v` are the normal and tangent axes (0 = x, 1 = y, 2 = z); U/V match the shader's extentScale.
    struct Face {
        let dx: Int32, dy: Int32, dz: Int32
        let n: Int32, u: Int32, v: Int32
        let shade: Float
    }

    static let faces: [Face] = [
        Face(dx: 1, dy: 0, dz: 0, n: 0, u: 1, v: 2, shade: 0.80),
        Face(dx: -1, dy: 0, dz: 0, n: 0, u: 1, v: 2, shade: 0.80),
        Face(dx: 0, dy: 1, dz: 0, n: 1, u: 0, v: 2, shade: 1.00),
        Face(dx: 0, dy: -1, dz: 0, n: 1, u: 0, v: 2, shade: 0.50),
        Face(dx: 0, dy: 0, dz: 1, n: 2, u: 0, v: 1, shade: 0.70),
        Face(dx: 0, dy: 0, dz: -1, n: 2, u: 0, v: 1, shade: 0.70),
    ]

    /// Set to false to emit one quad per block face (for A/B comparisons).
    nonisolated(unsafe) static var greedy = true

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
        let origin = SIMD3<Int>(sx * 16, sy * 16, sz * 16)
        var mesh = SectionMesh(minB: SIMD3<Float>(origin), maxB: SIMD3<Float>(origin &+ 16))
        guard view.hasSection(sx, sy, sz) else { return mesh }

        let air = MaterialKind.air.rawValue
        let water = MaterialKind.water.rawValue
        let opaque = MaterialKind.opaque.rawValue
        let merge = greedy
        var buckets: [[UInt32]] = [[], [], [], [], [], []]
        var maskO = [UInt8](repeating: 0, count: 256)
        var maskW = [UInt8](repeating: 0, count: 256)

        faces.withUnsafeBufferPointer { fp in
            kinds.withUnsafeBufferPointer { kp in
                for fi in 0..<6 {
                    let f = fp[fi]
                    let n = Int(f.n), ua = Int(f.u), va = Int(f.v)
                    for d in 0..<16 {
                        var anyO = false, anyW = false
                        for v in 0..<16 {
                            for u in 0..<16 {
                                var l = SIMD3<Int>(0, 0, 0)
                                l[n] = d; l[ua] = u; l[va] = v
                                let p = origin &+ l
                                let b = Int(view.block(p.x, p.y, p.z))
                                var mo: UInt8 = 0, mw: UInt8 = 0
                                if b != 0 {
                                    let nk = kp[Int(view.block(p.x + Int(f.dx), p.y + Int(f.dy), p.z + Int(f.dz)))]
                                    if kp[b] == water {
                                        if nk == air { mw = UInt8(b) }
                                    } else if nk != opaque {
                                        mo = UInt8(b)
                                    }
                                }
                                maskO[v * 16 + u] = mo
                                maskW[v * 16 + u] = mw
                                anyO = anyO || mo != 0
                                anyW = anyW || mw != 0
                            }
                        }
                        if anyO { emitMask(&maskO, fi: fi, f: f, d: d, merge: merge, into: &buckets[fi]) }
                        if anyW { emitMask(&maskW, fi: fi, f: f, d: d, merge: merge, into: &mesh.water) }
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

    /// Turns one 16x16 slice mask into quads, merging equal-material rectangles when `merge` is set.
    /// Consumes (zeroes) the mask.
    private static func emitMask(_ mask: inout [UInt8], fi: Int, f: Face, d: Int, merge: Bool,
                                 into out: inout [UInt32]) {
        let n = Int(f.n), ua = Int(f.u), va = Int(f.v)
        for v in 0..<16 {
            var u = 0
            while u < 16 {
                let m = mask[v * 16 + u]
                if m == 0 { u += 1; continue }
                var w = 1, h = 1
                if merge {
                    while u + w < 16 && mask[v * 16 + u + w] == m { w += 1 }
                    grow: while v + h < 16 {
                        for k in 0..<w where mask[(v + h) * 16 + u + k] != m { break grow }
                        h += 1
                    }
                }
                for dv in 0..<h {
                    for k in 0..<w { mask[(v + dv) * 16 + u + k] = 0 }
                }
                var l = SIMD3<Int>(0, 0, 0)
                l[n] = d; l[ua] = u; l[va] = v
                out.append(QuadFormat.pack(x: l.x, y: l.y, z: l.z, face: fi, material: Int(m), w: w, h: h))
                u += w
            }
        }
    }
}
