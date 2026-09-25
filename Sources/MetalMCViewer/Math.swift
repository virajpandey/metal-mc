import simd

/// Right-handed perspective projection with Metal's clip-space depth range [0, 1].
func perspectiveRH(fovyRadians: Float, aspect: Float, near: Float, far: Float) -> float4x4 {
    let ys = 1 / tan(fovyRadians * 0.5)
    let xs = ys / aspect
    let zs = far / (near - far)
    return float4x4(columns: (
        SIMD4<Float>(xs, 0, 0, 0),
        SIMD4<Float>(0, ys, 0, 0),
        SIMD4<Float>(0, 0, zs, -1),
        SIMD4<Float>(0, 0, zs * near, 0)
    ))
}

/// Right-handed, reverse-Z, infinite-far projection: depth 1 at the near plane, approaching 0 at infinity.
/// Pair with a `.greater` depth test and a clear depth of 0. Much better precision at long view distances.
func perspectiveReverseZ(fovyRadians: Float, aspect: Float, near: Float) -> float4x4 {
    let ys = 1 / tan(fovyRadians * 0.5)
    let xs = ys / aspect
    return float4x4(columns: (
        SIMD4<Float>(xs, 0, 0, 0),
        SIMD4<Float>(0, ys, 0, 0),
        SIMD4<Float>(0, 0, 0, -1),
        SIMD4<Float>(0, 0, near, 0)
    ))
}

func lookAtRH(eye: SIMD3<Float>, center: SIMD3<Float>, up: SIMD3<Float>) -> float4x4 {
    let f = simd_normalize(center - eye)
    let s = simd_normalize(simd_cross(f, up))
    let u = simd_cross(s, f)
    return float4x4(columns: (
        SIMD4<Float>(s.x, u.x, -f.x, 0),
        SIMD4<Float>(s.y, u.y, -f.y, 0),
        SIMD4<Float>(s.z, u.z, -f.z, 0),
        SIMD4<Float>(-simd_dot(s, eye), -simd_dot(u, eye), simd_dot(f, eye), 1)
    ))
}

/// Six clip planes extracted from a view-projection matrix (Metal depth range 0..1).
struct Frustum {
    let planes: [SIMD4<Float>]

    init(viewProj m: float4x4) {
        func row(_ i: Int) -> SIMD4<Float> {
            SIMD4(m.columns.0[i], m.columns.1[i], m.columns.2[i], m.columns.3[i])
        }
        let r0 = row(0), r1 = row(1), r2 = row(2), r3 = row(3)
        planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2]
    }

    func intersects(min lo: SIMD3<Float>, max hi: SIMD3<Float>) -> Bool {
        for p in planes {
            let v = SIMD3<Float>(p.x >= 0 ? hi.x : lo.x,
                                 p.y >= 0 ? hi.y : lo.y,
                                 p.z >= 0 ? hi.z : lo.z)
            if p.x * v.x + p.y * v.y + p.z * v.z + p.w < 0 { return false }
        }
        return true
    }
}

func distanceToBox(_ p: SIMD3<Float>, min lo: SIMD3<Float>, max hi: SIMD3<Float>) -> Float {
    let d = simd_max(simd_max(lo - p, p - hi), SIMD3<Float>(repeating: 0))
    return simd_length(d)
}

func percentile(_ xs: [Double], _ p: Double) -> Double {
    guard !xs.isEmpty else { return 0 }
    let s = xs.sorted()
    let i = min(s.count - 1, Int((p / 100) * Double(s.count - 1) + 0.5))
    return s[i]
}

func mean(_ xs: [Double]) -> Double {
    xs.isEmpty ? 0 : xs.reduce(0, +) / Double(xs.count)
}
