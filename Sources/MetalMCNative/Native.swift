import Foundation
import Metal
import QuartzCore

/// C ABI entry points for the Java side (java.lang.foreign). Every function is `@_cdecl` and uses
/// only C-compatible types. Metal objects live in `Registry` and cross the boundary as Int64 handles
/// (0 = failure). Pointers from Java arrive as raw addresses.

/// Bumped whenever the ABI changes, so Java can refuse a stale dylib.
@_cdecl("mmc_abi_version")
public func mmc_abi_version() -> Int32 { 2 }

/// Writes the default Metal device's name (UTF-8, NUL-terminated) into `buf`.
@_cdecl("mmc_device_name")
public func mmc_device_name(_ buf: UnsafeMutablePointer<CChar>, _ len: Int32) -> Int32 {
    guard len > 0, let device = MTLCreateSystemDefaultDevice() else { return 0 }
    let bytes = Array(device.name.utf8.prefix(Int(len) - 1))
    for (i, b) in bytes.enumerated() { buf[i] = CChar(bitPattern: b) }
    buf[bytes.count] = 0
    return 1
}

/// 1 if the default device supports the Apple9 family (M3/M4: mesh-shader ICBs, etc.).
@_cdecl("mmc_supports_apple9")
public func mmc_supports_apple9() -> Int32 {
    guard let device = MTLCreateSystemDefaultDevice() else { return 0 }
    return device.supportsFamily(.apple9) ? 1 : 0
}

// MARK: - Handle registry

final class Registry: @unchecked Sendable {
    static let shared = Registry()
    private var next: Int64 = 1
    private var objects: [Int64: AnyObject] = [:]
    private let lock = NSLock()

    func add(_ object: AnyObject) -> Int64 {
        lock.lock(); defer { lock.unlock() }
        let h = next
        next += 1
        objects[h] = object
        return h
    }

    func get<T>(_ handle: Int64) -> T? {
        lock.lock(); defer { lock.unlock() }
        return objects[handle] as? T
    }

    func remove(_ handle: Int64) {
        lock.lock(); defer { lock.unlock() }
        objects[handle] = nil
    }
}

@_cdecl("mmc_release")
public func mmc_release(_ handle: Int64) {
    Registry.shared.remove(handle)
}

// MARK: - Surface (CAMetalLayer from SDL_Metal_GetLayer)

final class Surface {
    let layer: CAMetalLayer
    let device: MTLDevice
    let queue: MTLCommandQueue

    init(layer: CAMetalLayer, device: MTLDevice, queue: MTLCommandQueue) {
        self.layer = layer
        self.device = device
        self.queue = queue
    }
}

/// Wraps an existing CAMetalLayer. `displaySync` = 0 lets presentation run faster than the display.
@_cdecl("mmc_surface_create")
public func mmc_surface_create(_ layerAddress: Int64, _ displaySync: Int32) -> Int64 {
    guard layerAddress != 0,
          let raw = UnsafeMutableRawPointer(bitPattern: Int(layerAddress)),
          let device = MTLCreateSystemDefaultDevice(),
          let queue = device.makeCommandQueue() else { return 0 }
    let layer = Unmanaged<CAMetalLayer>.fromOpaque(raw).takeUnretainedValue()
    layer.device = device
    layer.pixelFormat = .bgra8Unorm
    layer.framebufferOnly = true
    layer.displaySyncEnabled = displaySync != 0
    layer.maximumDrawableCount = 3
    let scale = layer.contentsScale
    layer.drawableSize = CGSize(width: layer.bounds.width * scale, height: layer.bounds.height * scale)
    return Registry.shared.add(Surface(layer: layer, device: device, queue: queue))
}

/// Drawable size in pixels, packed as (width << 32) | height.
@_cdecl("mmc_surface_size")
public func mmc_surface_size(_ handle: Int64) -> Int64 {
    guard let s: Surface = Registry.shared.get(handle) else { return 0 }
    return Int64(s.layer.drawableSize.width) << 32 | Int64(s.layer.drawableSize.height)
}

/// Clears and presents `frames` frames as fast as presentation allows. Returns wall seconds from the
/// first frame until the last one completed on the GPU (negative on failure).
@_cdecl("mmc_surface_run_frames")
public func mmc_surface_run_frames(_ handle: Int64, _ frames: Int32, _ frameOffset: Int32) -> Double {
    guard let s: Surface = Registry.shared.get(handle) else { return -1 }
    let t0 = CACurrentMediaTime()
    var last: MTLCommandBuffer?
    for i in 0..<Int(frames) {
        autoreleasepool {
            guard let drawable = s.layer.nextDrawable(), let cb = s.queue.makeCommandBuffer() else { return }
            let pass = MTLRenderPassDescriptor()
            pass.colorAttachments[0].texture = drawable.texture
            pass.colorAttachments[0].loadAction = .clear
            pass.colorAttachments[0].storeAction = .store
            let t = Double(i + Int(frameOffset)) / 120.0
            pass.colorAttachments[0].clearColor = MTLClearColor(red: 0.45 + 0.35 * sin(t), green: 0.30, blue: 0.62, alpha: 1)
            cb.makeRenderCommandEncoder(descriptor: pass)?.endEncoding()
            cb.present(drawable)
            cb.commit()
            last = cb
        }
    }
    last?.waitUntilCompleted()
    return CACurrentMediaTime() - t0
}
