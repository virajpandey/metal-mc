import ApplicationServices
import Foundation
let pid = pid_t(CommandLine.arguments[1])!
print("trusted for accessibility: \(AXIsProcessTrusted())")
let app = AXUIElementCreateApplication(pid)
func texts(_ e: AXUIElement, _ depth: Int) {
    if depth > 6 { return }
    for attr in [kAXTitleAttribute, kAXValueAttribute, kAXDescriptionAttribute] {
        var v: CFTypeRef?
        if AXUIElementCopyAttributeValue(e, attr as CFString, &v) == .success, let s = v as? String, !s.isEmpty { print(String(repeating: " ", count: depth) + s) }
    }
    var kids: CFTypeRef?
    if AXUIElementCopyAttributeValue(e, kAXChildrenAttribute as CFString, &kids) == .success, let arr = kids as? [AXUIElement] {
        for k in arr { texts(k, depth + 1) }
    }
}
texts(app, 0)
