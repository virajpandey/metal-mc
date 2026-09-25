# M3 backend spike: reconnaissance (Minecraft 26.3)

All findings come from the 26.3 client jars and Loom's decompiled sources. Nothing here has run yet.

## What a Metal backend must implement

| Interface (`com.mojang.renderpearl`) | Methods | Covers |
|---|---|---|
| `api.device.GpuBackend` | ~5 | name, library unload, `createWindow`, device creation |
| `backend.api.GpuDeviceBackend` | 14 | surfaces, command encoders, textures/views, buffers, `compilePipeline`, timestamp query pools, `DeviceInfo` |
| `backend.api.CommandEncoderBackend` | 17 | submit, render passes, clears, buffer/texture writes and copies, fences, timestamps |
| `backend.api.RenderPassBackend` | 18 | pipeline, uniforms, push constants, scissor, vertex/index buffers, draw / multiDraw / indirect variants |
| `backend.api.GpuSurfaceBackend` | 4 | `blitFromTexture`, `present`, present modes, `isSuboptimal` |
| `backend.api.BackendRenderPipeline` | 2 | asynchronous pipeline compile |

There are also concrete resource types: `GpuTexture`, `GpuTextureView`, `GpuBuffer`, `GpuQueryPool`, `GpuFence`, and `TransientMemory`.

**Size reference:** Mojang's OpenGL backend is 28 files and about 5,050 lines. Its Vulkan backend is 22 files and about 5,000 lines. A Metal backend will likely be 4–6K lines of Java plus a Swift/Objective-C bridge.

## Facts already confirmed

- Backend selection: `PreferredGraphicsApi` returns `{gl, vulkan}` or `{vulkan, gl}` from a hard-coded list, so one mixin can put a Metal backend first.
- `FrontendGpuDevice` has a public constructor, which keeps Mojang's validation layer in front of any backend.
- Shaders reach backends as SPIR-V (`SpvModule`). The game's bundled `lwjgl-spvc` 3.4.3 (`libspirv-cross.dylib`, arm64) contains `CompilerMSL`, so SPIR-V to MSL translation needs no extra native library.
- `Minecraft.windowSurface()` is public. Windows come from SDL3, which has a documented Metal-view path.
- Presentation: vanilla picks IMMEDIATE, then MAILBOX, then FIFO when vsync is off. On this Mac, MoltenVK offers `[IMMEDIATE, FIFO]`, yet windowed frames are still paced to 120 Hz (see `results/baseline-26.3`). A native `CAMetalLayer` should test `displaySyncEnabled = NO` in a window.

## Plan

1. **Title screen.** Stub `MetalGpuBackend`, create the device, surface and swapchain through `CAMetalLayer`, and clear to a color. Translate and compile the GUI pipelines (SPIR-V to MSL). Exit criterion: the 26.3 title screen renders on Metal.
2. **World.** Vanilla terrain geometry, correct depth against entities, and one live block update.
3. **Benchmark** against the baseline with the same mod, route and settings (windowed and fullscreen).

Open questions for step 1:

- Does anything outside Renderpearl downcast to `GlDevice` or `VulkanDevice`? A static scan says no; runtime will tell.
- Which JNI / FFM bridge shape fits the per-draw call rate of `RenderPassBackend`?
- Can the SDL3 window accept a `CAMetalLayer` without the OpenGL or Vulkan flags `createWindow` normally passes?
