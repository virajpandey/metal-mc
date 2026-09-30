// Offline LOD render, no game needed: opens a world through libMetalMCNative (mmc_lod_open, as tools/lodtest.py does),
// waits for the build, picks nodes for each camera with the LOD's own selection (mmc_debug_lod_select) and draws their
// quads (mmc_debug_lod_node_raw) with a simplified copy of lod_vs / lodShade: flat colors (mmc_debug_lod_colors), no
// textures, AO, fog or seam, the no-lightmap light curve, water blended at 0.706 after the opaque quads with no depth
// writes, tile-edge skirts only toward tiles the node doesn't draw. Good for A/B of mesher changes (METALMC_EXP applies).
//   swiftc -O tools/lodrender.swift -o bench_out/lodrender
//   bench_out/lodrender .build/release/libMetalMCNative.dylib fixtures/claudeworld-merged 8192 bench_out/lr/a \
//       highnorth:8:260:8:180:12 ocean:-400:160:-350:180:25
// Views are name:x:y:z:yaw:pitch[:fov[:w:h]] (Minecraft's yaw and pitch; 70 degrees, 1728 x 1117 by default); the
// world opens around the first. Writes <prefix>-<name>.png, .rgba (raw, for diffs) and .holes (pixels where water was
// drawn over nothing opaque: sky under the sea). Env: HILITE=1 colors node-edge skirts under water magenta (cyan if
// water) and tile-edge skirts yellow; PICK_<name>="x,y ..." prints the opaque quad under each pixel; RAY_<name>="x,y ..."
// prints every drawn quad the pixel's ray crosses, nearest first.
import Foundation
import Metal
import simd
import ImageIO
import UniformTypeIdentifiers

let args = CommandLine.arguments
guard args.count >= 5, let lib = dlopen(args[1], RTLD_NOW) else { print("dlopen failed"); exit(1) }
func sym<T>(_ name: String, _ t: T.Type) -> T { unsafeBitCast(dlsym(lib, name)!, to: t) }
typealias OpenF = @convention(c) (UnsafePointer<CChar>, Int32, Int32, Int32) -> Int32
typealias StatusF = @convention(c) (UnsafeMutablePointer<Int64>) -> Void
typealias SelectF = @convention(c) (Double, Double, UnsafeMutablePointer<Int64>, Int32) -> Int32
typealias RawF = @convention(c) (Int32, Int32, Int32, UnsafeMutablePointer<UInt32>, Int32, UnsafeMutablePointer<Int32>) -> Int32
typealias ColorsF = @convention(c) (UnsafeMutablePointer<Float>) -> Void
let lodOpen = sym("mmc_lod_open", OpenF.self)
let lodStatus = sym("mmc_lod_status", StatusF.self)
let lodSelect = sym("mmc_debug_lod_select", SelectF.self)
let lodRaw = sym("mmc_debug_lod_node_raw", RawF.self)
let lodColors = sym("mmc_debug_lod_colors", ColorsF.self)

let world = args[2], far = Int32(args[3])!, prefix = args[4]
let views = args[5...].map { $0.split(separator: ":").map(String.init) }
let cx0 = Int32(Double(views[0][1])!), cz0 = Int32(Double(views[0][3])!)
print("open", lodOpen(world, far, cx0, cz0))
var st = [Int64](repeating: 0, count: 4)
var last: Int64 = -1, stable = 0
let t0 = Date()
while stable < 24 {
    lodStatus(&st)
    stable = st[0] == 2 && st[3] == 1 && st[2] == last ? stable + 1 : 0
    last = st[2]
    Thread.sleep(forTimeInterval: 0.25)
}
print("built: nodes \(st[1]) quads \(st[2]) in \(Int(Date().timeIntervalSince(t0))) s")

var colors = [Float](repeating: 0, count: 256 * 3 * 4)
lodColors(&colors)

let dev = MTLCreateSystemDefaultDevice()!
let queue = dev.makeCommandQueue()!
let src = """
#include <metal_stdlib>
using namespace metal;
struct U { float4x4 vp; float alpha; float pad0, pad1, pad2; };
struct VOut { float4 pos [[position]]; float4 color; };
constant float3 kCorners[6][4] = {
    { float3(1,0,0), float3(1,1,0), float3(1,1,1), float3(1,0,1) },
    { float3(0,0,1), float3(0,1,1), float3(0,1,0), float3(0,0,0) },
    { float3(0,1,0), float3(0,1,1), float3(1,1,1), float3(1,1,0) },
    { float3(0,0,0), float3(1,0,0), float3(1,0,1), float3(0,0,1) },
    { float3(1,0,1), float3(1,1,1), float3(0,1,1), float3(0,0,1) },
    { float3(0,0,0), float3(0,1,0), float3(1,1,0), float3(1,0,0) },
};
constant float kShade[6] = { 0.6, 0.6, 1.0, 0.5, 0.8, 0.8 };
constant float kWaterLight[16] = { 1.0, 0.908, 0.822, 0.747, 0.676, 0.609, 0.544, 0.482,
                                   0.423, 0.367, 0.315, 0.265, 0.218, 0.174, 0.133, 0.094 };
static float3 extentScale(uint face, float w, float h) {
    if (face < 2) return float3(1, w, h);
    if (face < 4) return float3(w, 1, h);
    return float3(w, h, 1);
}
static float3 quadCorner(uint2 q, uint corner, float4 xs) {
    uint face = (q.x >> 25) & 7;
    float3 local = float3(q.x & 255, (q.x >> 16) & 511, (q.x >> 8) & 255);
    float w = float(((q.y >> 8) & 255) + 1), h = float(((q.y >> 16) & 255) + 1);
    uint m = q.y & 255;
    bool water = (m >= 128u && m < 160u) || m == 5u;
    float3 rel = xs.xyz + (local + kCorners[face][corner] * extentScale(face, w, h)) * xs.w;
    if (water && kCorners[face][corner].y > 0.5) rel.y -= xs.w < 1.5 ? 1.0 / 9.0 : 10.0 / 9.0;
    return rel;
}
vertex VOut vs(uint vid [[vertex_id]], const device uint2* quads [[buffer(0)]], constant U& u [[buffer(1)]],
               constant float4& xs [[buffer(2)]], constant float4* colors [[buffer(3)]], constant uint& bucket [[buffer(5)]]) {
    uint2 q = quads[vid >> 2];
    uint face = (q.x >> 25) & 7;
    uint m = q.y & 255;
    bool water = (m >= 128u && m < 160u) || m == 5u;
    uint faceClass = face == 2 ? 0u : (face == 3 ? 2u : 1u);
    uint depth = water ? 0u : (q.x >> 28) & 15;
    float blockLevel = float((q.y >> 24) & 15u);
    float light = blockLevel > 0.0 ? 1.0 : kWaterLight[depth];
    VOut o;
    o.pos = u.vp * float4(quadCorner(q, vid & 3, xs), 1.0);
    o.color = float4(colors[m * 3 + faceClass].rgb * kShade[face] * light, u.alpha);
    if (u.pad0 > 0.5) {
        // Highlight: node-edge skirts under water magenta (water-material ones cyan), tile-edge skirts yellow.
        uint x = q.x & 255, z = (q.x >> 8) & 255;
        bool nodeEdge = (face == 0 && x == 255) || (face == 1 && x == 0) || (face == 4 && z == 255) || (face == 5 && z == 0);
        bool wet = ((q.x >> 28) & 15) != 0;
        if (bucket >= 12) o.color = float4(1, 1, 0, 1);
        else if (nodeEdge && water) o.color = float4(0, 1, 1, u.alpha);
        else if (nodeEdge && wet) o.color = float4(1, 0, 1, 1);
    }
    return o;
}
struct OpaqueOut { float4 c [[color(0)]]; float hit [[color(1)]]; };
fragment OpaqueOut fs(VOut in [[stage_in]]) { OpaqueOut o; o.c = in.color; o.hit = 1.0; return o; }
// Water: marks pixels where it's drawn over nothing opaque (sky under the sea).
struct WaterOut { float4 c [[color(0)]]; float hole [[color(2)]]; };
fragment WaterOut fsWater(VOut in [[stage_in]], float hit [[color(1)]]) { WaterOut o; o.c = in.color; o.hole = hit > 0.5 ? 0.0 : 1.0; return o; }
struct IOut { float4 pos [[position]]; uint id [[flat]]; };
vertex IOut vsId(uint vid [[vertex_id]], const device uint2* quads [[buffer(0)]], constant U& u [[buffer(1)]],
                 constant float4& xs [[buffer(2)]], constant uint& node [[buffer(4)]]) {
    IOut o;
    o.pos = u.vp * float4(quadCorner(quads[vid >> 2], vid & 3, xs), 1.0);
    o.id = (node << 22) | (vid >> 2);
    return o;
}
fragment uint4 fsId(IOut in [[stage_in]]) { return uint4(in.id, 0, 0, 0); }
"""
let library = try! dev.makeLibrary(source: src, options: nil)
func pipe(_ blend: Bool) -> MTLRenderPipelineState {
    let d = MTLRenderPipelineDescriptor()
    d.vertexFunction = library.makeFunction(name: "vs")
    d.fragmentFunction = library.makeFunction(name: blend ? "fsWater" : "fs")
    d.colorAttachments[0].pixelFormat = .rgba8Unorm
    d.colorAttachments[1].pixelFormat = .r32Float
    d.colorAttachments[2].pixelFormat = .r8Unorm
    if blend { d.colorAttachments[1].writeMask = [] } else { d.colorAttachments[2].writeMask = [] }
    d.depthAttachmentPixelFormat = .depth32Float
    if blend {
        d.colorAttachments[0].isBlendingEnabled = true
        d.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        d.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        d.colorAttachments[0].sourceAlphaBlendFactor = .one
        d.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
    }
    return try! dev.makeRenderPipelineState(descriptor: d)
}
let opaquePipe = pipe(false), waterPipe = pipe(true)
let idPipe: MTLRenderPipelineState = {
    let d = MTLRenderPipelineDescriptor()
    d.vertexFunction = library.makeFunction(name: "vsId")
    d.fragmentFunction = library.makeFunction(name: "fsId")
    d.colorAttachments[0].pixelFormat = .r32Uint
    d.depthAttachmentPixelFormat = .depth32Float
    return try! dev.makeRenderPipelineState(descriptor: d)
}()
let dsd = MTLDepthStencilDescriptor(); dsd.depthCompareFunction = .less; dsd.isDepthWriteEnabled = true
let depthWrite = dev.makeDepthStencilState(descriptor: dsd)!
dsd.isDepthWriteEnabled = false
let depthNoWrite = dev.makeDepthStencilState(descriptor: dsd)!
let colorBuf = dev.makeBuffer(bytes: colors, length: colors.count * 4)!
let maxQuads = 4_000_000
var idx = [UInt32](repeating: 0, count: maxQuads * 6)
for q in 0..<maxQuads { let b = UInt32(4 * q); idx[6 * q] = b; idx[6 * q + 1] = b + 1; idx[6 * q + 2] = b + 2; idx[6 * q + 3] = b; idx[6 * q + 4] = b + 2; idx[6 * q + 5] = b + 3 }
let indexBuf = dev.makeBuffer(bytes: idx, length: idx.count * 4)!
let faceNames = ["+X", "-X", "+Y", "-Y", "+Z", "-Z"]

struct NodeData { let buf: MTLBuffer; let starts: [Int32]; let level: Int; let x: Int; let z: Int }
var cache: [String: NodeData] = [:]
func node(_ l: Int, _ x: Int, _ z: Int) -> NodeData? {
    let k = "\(l),\(x),\(z)"
    if let n = cache[k] { return n }
    var starts = [Int32](repeating: 0, count: 16 * 16 * 16 + 1)
    var q = [UInt32](repeating: 0, count: 2 * maxQuads)
    let c = Int(lodRaw(Int32(l), Int32(x), Int32(z), &q, Int32(maxQuads), &starts))
    if c <= 0 { return nil }
    let n = NodeData(buf: dev.makeBuffer(bytes: q, length: 8 * c)!, starts: starts, level: l, x: x, z: z)
    cache[k] = n
    return n
}
@inline(__always) func bucketIndex(_ t: Int, _ k: Int, _ st: Int) -> Int { (t * 16 + k) * 16 + st }

for v in views {
    let name = v[0]
    let cam = SIMD3<Double>(Double(v[1])!, Double(v[2])!, Double(v[3])!)
    let yaw = Double(v[4])! * .pi / 180, pitch = Double(v[5])! * .pi / 180
    let fov = (v.count > 6 ? Double(v[6])! : 70) * .pi / 180
    let W = v.count > 8 ? Int(v[7])! : 1728, H = v.count > 8 ? Int(v[8])! : 1117
    var sel = [Int64](repeating: 0, count: 5 * 4096)
    let ns = Int(lodSelect(cam.x, cam.z, &sel, 4096))
    var chosen: [(NodeData, UInt16)] = []
    for i in 0..<ns {
        if let n = node(Int(sel[5 * i]), Int(sel[5 * i + 1]), Int(sel[5 * i + 2])) { chosen.append((n, UInt16(sel[5 * i + 3]))) }
    }
    // View: Minecraft yaw 0 = +Z, 90 = -X; pitch > 0 looks down.
    let f = SIMD3<Float>(Float(-sin(yaw) * cos(pitch)), Float(-sin(pitch)), Float(cos(yaw) * cos(pitch)))
    let r = normalize(cross(f, SIMD3<Float>(0, 1, 0)))
    let up = cross(r, f)
    let view = simd_float4x4(rows: [SIMD4(r.x, r.y, r.z, 0), SIMD4(up.x, up.y, up.z, 0), SIMD4(-f.x, -f.y, -f.z, 0), SIMD4(0, 0, 0, 1)])
    let near: Float = 0.5, farP: Float = 60000, aspect = Float(W) / Float(H)
    let ys = 1 / Float(tan(fov / 2)), xsc = ys / aspect
    let proj = simd_float4x4(rows: [SIMD4(xsc, 0, 0, 0), SIMD4(0, ys, 0, 0), SIMD4(0, 0, farP / (near - farP), near * farP / (near - farP)), SIMD4(0, 0, -1, 0)])
    var u = (proj * view, Float(1), Float(ProcessInfo.processInfo.environment["HILITE"] == nil ? 0 : 1), Float(0), Float(0))
    // Draw ranges per chosen node: opaque (with tile-edge skirts toward tiles the node doesn't draw), then water.
    func ranges(_ n: NodeData, _ mask: UInt16, water: Bool) -> [(Int, Int, Int)] {
        var out: [(Int, Int, Int)] = []
        for t in 0..<16 where mask & (1 << UInt16(t)) != 0 {
            var ks: [Int] = water ? Array(6..<12) : Array(0..<6)
            if !water {
                let tx = t % 4, tz = t / 4
                for (e, (dx, dz)) in [(-1, 0), (1, 0), (0, -1), (0, 1)].enumerated() {
                    let ntx = tx + dx, ntz = tz + dz
                    if ntx < 0 || ntz < 0 || ntx >= 4 || ntz >= 4 { continue }
                    if mask & (1 << UInt16(ntz * 4 + ntx)) != 0 { continue }
                    ks.append(12 + e)
                }
            }
            for k in ks {
                let a = Int(n.starts[bucketIndex(t, k, 0)]), b = Int(n.starts[bucketIndex(t, k, 15) + 1])
                if b > a { out.append((a, b, k)) }
            }
        }
        return out
    }
    func xform(_ n: NodeData) -> SIMD4<Float> {
        let size = 256 << n.level
        return SIMD4<Float>(Float(Double(n.x * size) - cam.x), Float(Double(-64) - cam.y), Float(Double(n.z * size) - cam.z), Float(1 << n.level))
    }
    let dd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .depth32Float, width: W, height: H, mipmapped: false)
    dd.usage = .renderTarget; dd.storageMode = .private
    let dtex = dev.makeTexture(descriptor: dd)!
    let td = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .rgba8Unorm, width: W, height: H, mipmapped: false)
    td.usage = [.renderTarget, .shaderRead]; td.storageMode = .shared
    let tex = dev.makeTexture(descriptor: td)!
    let rp = MTLRenderPassDescriptor()
    rp.colorAttachments[0].texture = tex; rp.colorAttachments[0].loadAction = .clear; rp.colorAttachments[0].storeAction = .store
    rp.colorAttachments[0].clearColor = MTLClearColor(red: 0.47, green: 0.65, blue: 1.0, alpha: 1)
    rp.depthAttachment.texture = dtex; rp.depthAttachment.loadAction = .clear; rp.depthAttachment.clearDepth = 1; rp.depthAttachment.storeAction = .dontCare
    let hd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Float, width: W, height: H, mipmapped: false)
    hd.usage = .renderTarget; hd.storageMode = .private
    let hitTex = dev.makeTexture(descriptor: hd)!
    let hod = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r8Unorm, width: W, height: H, mipmapped: false)
    hod.usage = .renderTarget; hod.storageMode = .shared
    let holeTex = dev.makeTexture(descriptor: hod)!
    rp.colorAttachments[1].texture = hitTex; rp.colorAttachments[1].loadAction = .clear; rp.colorAttachments[1].storeAction = .dontCare
    rp.colorAttachments[1].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    rp.colorAttachments[2].texture = holeTex; rp.colorAttachments[2].loadAction = .clear; rp.colorAttachments[2].storeAction = .store
    rp.colorAttachments[2].clearColor = MTLClearColor(red: 0, green: 0, blue: 0, alpha: 0)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeRenderCommandEncoder(descriptor: rp)!
    enc.setCullMode(.back)
    enc.setFrontFacing(.counterClockwise)
    enc.setVertexBuffer(colorBuf, offset: 0, index: 3)
    var drawn = 0
    for pass in 0..<2 {
        enc.setRenderPipelineState(pass == 0 ? opaquePipe : waterPipe)
        enc.setDepthStencilState(pass == 0 ? depthWrite : depthNoWrite)
        u.1 = pass == 0 ? 1 : 0.706
        enc.setVertexBytes(&u, length: MemoryLayout.size(ofValue: u), index: 1)
        for (n, mask) in chosen {
            var xs = xform(n)
            enc.setVertexBytes(&xs, length: 16, index: 2)
            enc.setVertexBuffer(n.buf, offset: 0, index: 0)
            for (a, b, k) in ranges(n, mask, water: pass == 1) {
                var kk = UInt32(k)
                enc.setVertexBytes(&kk, length: 4, index: 5)
                enc.drawIndexedPrimitives(type: .triangle, indexCount: (b - a) * 6, indexType: .uint32, indexBuffer: indexBuf, indexBufferOffset: a * 24)
                drawn += b - a
            }
        }
    }
    enc.endEncoding()
    cb.commit(); cb.waitUntilCompleted()
    var px = [UInt8](repeating: 0, count: W * H * 4)
    tex.getBytes(&px, bytesPerRow: W * 4, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0)
    for i in 0..<(W * H) { px[4 * i + 3] = 255 }
    let cs = CGColorSpace(name: CGColorSpace.sRGB)!
    let ctxImg = CGContext(data: &px, width: W, height: H, bitsPerComponent: 8, bytesPerRow: W * 4, space: cs, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
    let img = ctxImg.makeImage()!
    let url = URL(fileURLWithPath: "\(prefix)-\(name).png")
    let dst = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dst, img, nil)
    CGImageDestinationFinalize(dst)
    FileManager.default.createFile(atPath: "\(prefix)-\(name).rgba", contents: Data(px))
    var holes = [UInt8](repeating: 0, count: W * H)
    holeTex.getBytes(&holes, bytesPerRow: W, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0)
    let holeCount = holes.reduce(0) { $0 + ($1 > 127 ? 1 : 0) }
    FileManager.default.createFile(atPath: "\(prefix)-\(name).holes", contents: Data(holes))
    print("\(name): \(ns) nodes, \(drawn) quads, \(holeCount) px of water over nothing (\(String(format: "%.3f", Double(holeCount) * 100 / Double(W * H)))%) -> \(url.path)")

    // Ray picks (env RAY_<name> = "x,y x,y ..."): every drawn quad (opaque and water) the pixel's ray crosses, nearest first.
    if let raySpec = ProcessInfo.processInfo.environment["RAY_\(name)"] {
        for p in raySpec.split(separator: " ") {
            let c = p.split(separator: ",").map { Double($0)! }
            let ndcX = (c[0] + 0.5) / Double(W) * 2 - 1, ndcY = 1 - (c[1] + 0.5) / Double(H) * 2
            let dirF = normalize(f + r * Float(ndcX / Double(xsc)) + up * Float(ndcY / Double(ys)))
            let dir = SIMD3<Double>(Double(dirF.x), Double(dirF.y), Double(dirF.z))
            var hits: [(Double, String)] = []
            for (n, mask) in chosen {
                let xs = xform(n)
                let s = Double(xs.w), o = SIMD3<Double>(Double(xs.x), Double(xs.y), Double(xs.z))
                let qp = n.buf.contents().bindMemory(to: UInt32.self, capacity: 8)
                for water in [false, true] {
                    for (a, b, _) in ranges(n, mask, water: water) {
                        for qi in a..<b {
                            let w0 = qp[2 * qi], w1 = qp[2 * qi + 1]
                            let face = Int((w0 >> 25) & 7), axis = face / 2
                            let normalSign: Double = face % 2 == 0 ? 1 : -1
                            if dir[axis] * normalSign >= 0 { continue }   // back face
                            let lx = Double(w0 & 255), lz = Double((w0 >> 8) & 255), ly = Double((w0 >> 16) & 511)
                            let qw = Double((w1 >> 8) & 255) + 1, qh = Double((w1 >> 16) & 255) + 1
                            var lo = SIMD3<Double>(lx, ly, lz), hi = lo + 1
                            switch axis {
                            case 0: hi.y = ly + qw; hi.z = lz + qh
                            case 1: hi.x = lx + qw; hi.z = lz + qh
                            default: hi.x = lx + qw; hi.y = ly + qh
                            }
                            let plane = o[axis] + (face % 2 == 0 ? hi[axis] : lo[axis]) * s
                            let m = Int(w1 & 255)
                            let isWater = (m >= 128 && m < 160) || m == 5
                            var planeY = plane
                            if isWater && face == 2 { planeY -= s < 1.5 ? 1.0 / 9.0 : 10.0 / 9.0 }
                            let t = (axis == 1 ? planeY : plane) / dir[axis]
                            if t <= 0 { continue }
                            let pt = dir * t
                            var inside = true
                            for k in 0..<3 where k != axis {
                                let pLo = o[k] + lo[k] * s, pHi = o[k] + hi[k] * s - (isWater && k == 1 && face != 3 ? (s < 1.5 ? 1.0 / 9.0 : 10.0 / 9.0) : 0)
                                if pt[k] < pLo || pt[k] > pHi { inside = false }
                            }
                            if !inside { continue }
                            var bucket = -1
                            for t2 in 0..<16 { for k in 0..<16 where Int(n.starts[bucketIndex(t2, k, 0)]) <= qi && qi < Int(n.starts[bucketIndex(t2, k, 15) + 1]) { bucket = k } }
                            let size = 256 << n.level, sb = 1 << n.level
                            hits.append((t, String(format: "t %.1f L%d node (%d,%d) bucket %d local (%d,%d,%d) world (%d,%d,%d) face %@ w %d h %d mat %d depth %d light %d hit (%.1f,%.1f,%.1f)",
                                                   t, n.level, n.x, n.z, bucket, Int(lx), Int(ly), Int(lz), n.x * size + Int(lx) * sb, Int(ly) * sb - 64, n.z * size + Int(lz) * sb,
                                                   faceNames[face], Int(qw), Int(qh), m, Int(w0 >> 28), Int(w1 >> 24), pt.x + cam.x, pt.y + cam.y, pt.z + cam.z)))
                        }
                    }
                }
            }
            hits.sort { $0.0 < $1.0 }
            print("ray \(p):")
            for h in hits.prefix(6) { print("   ", h.1) }
        }
    }
    guard let pickSpec = ProcessInfo.processInfo.environment["PICK_\(name)"] else { continue }
    let itd = MTLTextureDescriptor.texture2DDescriptor(pixelFormat: .r32Uint, width: W, height: H, mipmapped: false)
    itd.usage = .renderTarget; itd.storageMode = .shared
    let itex = dev.makeTexture(descriptor: itd)!
    let ip = MTLRenderPassDescriptor()
    ip.colorAttachments[0].texture = itex; ip.colorAttachments[0].loadAction = .clear; ip.colorAttachments[0].storeAction = .store
    ip.colorAttachments[0].clearColor = MTLClearColor(red: Double(UInt32.max), green: 0, blue: 0, alpha: 0)
    ip.depthAttachment.texture = dtex; ip.depthAttachment.loadAction = .clear; ip.depthAttachment.clearDepth = 1; ip.depthAttachment.storeAction = .dontCare
    let cb2 = queue.makeCommandBuffer()!
    let e2 = cb2.makeRenderCommandEncoder(descriptor: ip)!
    e2.setCullMode(.back); e2.setFrontFacing(.counterClockwise)
    e2.setRenderPipelineState(idPipe); e2.setDepthStencilState(depthWrite)
    u.1 = 1
    e2.setVertexBytes(&u, length: MemoryLayout.size(ofValue: u), index: 1)
    for (i, (n, mask)) in chosen.enumerated() {
        var xs = xform(n)
        var ni = UInt32(i)
        e2.setVertexBytes(&xs, length: 16, index: 2)
        e2.setVertexBytes(&ni, length: 4, index: 4)
        e2.setVertexBuffer(n.buf, offset: 0, index: 0)
        for (a, b, _) in ranges(n, mask, water: false) {
            e2.drawIndexedPrimitives(type: .triangle, indexCount: (b - a) * 6, indexType: .uint32, indexBuffer: indexBuf, indexBufferOffset: a * 24)
        }
    }
    e2.endEncoding(); cb2.commit(); cb2.waitUntilCompleted()
    var ids = [UInt32](repeating: 0, count: W * H)
    itex.getBytes(&ids, bytesPerRow: W * 4, from: MTLRegionMake2D(0, 0, W, H), mipmapLevel: 0)
    for p in pickSpec.split(separator: " ") {
        let c = p.split(separator: ",").map { Int($0)! }
        let id = ids[c[1] * W + c[0]]
        if id == UInt32.max { print("pick \(c): nothing opaque"); continue }
        let (n, _) = chosen[Int(id >> 22)]
        let qi = Int(id & 0x3FFFFF)
        let qp = n.buf.contents().bindMemory(to: UInt32.self, capacity: 2 * (qi + 1))
        let w0 = qp[2 * qi], w1 = qp[2 * qi + 1]
        var bucket = -1, tile = -1
        for t in 0..<16 { for k in 0..<16 where Int(n.starts[bucketIndex(t, k, 0)]) <= qi && qi < Int(n.starts[bucketIndex(t, k, 15) + 1]) { bucket = k; tile = t } }
        let lx = Int(w0 & 255), lz = Int((w0 >> 8) & 255), ly = Int((w0 >> 16) & 511), face = Int((w0 >> 25) & 7)
        let s = 1 << n.level, size = 256 << n.level
        print("pick \(c): L\(n.level) node (\(n.x),\(n.z)) tile \(tile) bucket \(bucket) local (\(lx),\(ly),\(lz)) world (\(n.x * size + lx * s),\(ly * s - 64),\(n.z * size + lz * s)) face \(faceNames[face]) w \((w1 >> 8 & 255) + 1) h \((w1 >> 16 & 255) + 1) mat \(w1 & 255) depth \(w0 >> 28) light \(w1 >> 24)")
    }
}
