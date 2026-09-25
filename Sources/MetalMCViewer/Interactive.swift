import AppKit
import MetalKit
import QuartzCore
import simd

/// MTKView that tracks held keys (macOS virtual key codes).
final class InputView: MTKView {
    var keys = Set<UInt16>()
    override var acceptsFirstResponder: Bool { true }
    override func keyDown(with event: NSEvent) { keys.insert(event.keyCode) }
    override func keyUp(with event: NSEvent) { keys.remove(event.keyCode) }
}

@MainActor
final class ViewController: NSObject, MTKViewDelegate {
    private let renderer: Renderer
    private let view: InputView
    private var pos: SIMD3<Float>
    private var yaw: Float = 0
    private var pitch: Float = -0.4
    private let speed: Float = 40
    private var last = CACurrentMediaTime()
    private var lastTitle = CACurrentMediaTime()
    private var frames = 0
    private var lastCpuMs = 0.0
    var lastGpuMs = 0.0

    init(renderer: Renderer, view: InputView, world: World) {
        self.renderer = renderer
        self.view = view
        pos = SIMD3(Float(world.sizeX) / 2, Float(world.referenceY + 50), Float(world.sizeZ) / 2 + 60)
    }

    private var forward: SIMD3<Float> {
        SIMD3(cos(pitch) * sin(yaw), sin(pitch), -cos(pitch) * cos(yaw))
    }

    private func update(_ dt: Float) {
        let k = view.keys
        let turn: Float = 1.8 * dt
        if k.contains(123) { yaw -= turn }                       // left arrow
        if k.contains(124) { yaw += turn }                       // right arrow
        if k.contains(126) { pitch = min(1.5, pitch + turn) }    // up arrow
        if k.contains(125) { pitch = max(-1.5, pitch - turn) }   // down arrow

        let right = SIMD3<Float>(cos(yaw), 0, sin(yaw))
        var move = SIMD3<Float>(repeating: 0)
        if k.contains(13) { move += forward }                    // W
        if k.contains(1) { move -= forward }                     // S
        if k.contains(2) { move += right }                       // D
        if k.contains(0) { move -= right }                       // A
        if k.contains(14) || k.contains(49) { move.y += 1 }      // E or space
        if k.contains(12) { move.y -= 1 }                        // Q
        if simd_length(move) > 0 { pos += simd_normalize(move) * speed * dt }
    }

    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {}

    func draw(in view: MTKView) {
        let now = CACurrentMediaTime()
        update(Float(min(0.1, now - last)))
        last = now

        guard let pass = view.currentRenderPassDescriptor,
              let drawable = view.currentDrawable,
              let cb = renderer.queue.makeCommandBuffer() else { return }

        let size = view.drawableSize
        let proj = renderer.projection(aspect: Float(size.width / max(1, size.height)))
        let viewM = lookAtRH(eye: pos, center: pos + forward, up: SIMD3(0, 1, 0))

        let t0 = CACurrentMediaTime()
        let st = renderer.encode(into: cb, pass: pass, viewProj: proj * viewM, cameraPos: pos)
        lastCpuMs = (CACurrentMediaTime() - t0) * 1000

        cb.addCompletedHandler { [weak self] buffer in
            let g = (buffer.gpuEndTime - buffer.gpuStartTime) * 1000
            DispatchQueue.main.async { MainActor.assumeIsolated { self?.lastGpuMs = g } }
        }
        cb.present(drawable)
        cb.commit()

        frames += 1
        if now - lastTitle > 0.5 {
            let fps = Double(frames) / (now - lastTitle)
            view.window?.title = String(format: "MetalMC  %.0f fps  cpu %.2f ms  gpu %.2f ms  sections %d/%d  tris %.1fM",
                                        fps, lastCpuMs, lastGpuMs, st.drawn, st.drawn + st.culled,
                                        Double(st.triangles) / 1e6)
            frames = 0
            lastTitle = now
        }
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }
}

private var retainedDelegate: AppDelegate?
private var retainedController: ViewController?

@MainActor
func runInteractive(renderer: Renderer, world: World) {
    let app = NSApplication.shared
    app.setActivationPolicy(.regular)
    let delegate = AppDelegate()
    retainedDelegate = delegate
    app.delegate = delegate

    let rect = NSRect(x: 0, y: 0, width: 1280, height: 720)
    let window = NSWindow(contentRect: rect, styleMask: [.titled, .closable, .resizable, .miniaturizable],
                          backing: .buffered, defer: false)
    window.title = "MetalMC"

    let view = InputView(frame: rect, device: renderer.device)
    view.colorPixelFormat = renderer.colorFormat
    view.depthStencilPixelFormat = renderer.depthFormat
    view.clearColor = renderer.clearColor
    view.clearDepth = renderer.clearDepth
    view.preferredFramesPerSecond = 120

    let controller = ViewController(renderer: renderer, view: view, world: world)
    retainedController = controller
    view.delegate = controller

    window.contentView = view
    window.center()
    window.makeKeyAndOrderFront(nil)
    window.makeFirstResponder(view)
    app.activate(ignoringOtherApps: true)
    app.run()
}
