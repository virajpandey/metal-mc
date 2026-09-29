import CoreGraphics
import Foundation
let opts = CGWindowListOption(arrayLiteral: .optionOnScreenOnly, .excludeDesktopElements)
let list = (CGWindowListCopyWindowInfo(opts, kCGNullWindowID) as? [[String: Any]]) ?? []
for w in list {
    let owner = w[kCGWindowOwnerName as String] as? String ?? "?"
    let name = w[kCGWindowName as String] as? String ?? ""
    let layer = w[kCGWindowLayer as String] as? Int ?? 0
    let b = w[kCGWindowBounds as String] as? [String: Any] ?? [:]
    let alpha = w[kCGWindowAlpha as String] as? Double ?? 1
    print("layer \(layer) alpha \(alpha) \(owner) '\(name)' \(b["Width"] ?? 0)x\(b["Height"] ?? 0) at \(b["X"] ?? 0),\(b["Y"] ?? 0)")
}
