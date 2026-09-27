import AppKit

/// Brings the game's window to the front (benchmarks). A window the user can't see is paced to the display's
/// refresh by the window server, which caps a benchmark at 120 fps whatever the renderer does. Call on the
/// main thread (Minecraft's render thread on macOS). Returns 1 if the app is active afterwards.
@_cdecl("mmc_activate_app")
public func mmc_activate_app() -> Int32 {
    let app = NSApplication.shared
    app.unhide(nil)
    app.activate()
    if !app.isActive { app.activate(ignoringOtherApps: true) }
    for w in app.windows where w.isVisible || w.isMiniaturized {
        if w.isMiniaturized { w.deminiaturize(nil) }
        w.makeKeyAndOrderFront(nil)
        w.orderFrontRegardless()
    }
    return app.isActive ? 1 : 0
}
