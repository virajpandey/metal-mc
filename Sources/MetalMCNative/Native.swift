import Metal

/// C ABI entry points for the Java side (java.lang.foreign). Every function is `@_cdecl` and uses
/// only C-compatible types. Returns follow C conventions: 1 = success, 0 = failure.

/// Bumped whenever the ABI changes, so Java can refuse a stale dylib.
@_cdecl("mmc_abi_version")
public func mmc_abi_version() -> Int32 { 1 }

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
