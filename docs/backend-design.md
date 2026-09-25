# Metal backend design (M3 step 1b onward)

**Status:** step 1a works. Minecraft 26.3 tries `MetalGpuBackend` first. Java calls the Swift dylib through FFM, creates a real `MTLDevice` (Apple M3 Pro, Apple9), then declines, and the game falls back to OpenGL cleanly.

**Why step 1b is big:** once `createDevice` succeeds, the game immediately creates textures and buffers, compiles every core pipeline, and records render passes. The title screen therefore needs most of the ~60 backend methods (see `backend-spike.md`), so this design comes first.

## Principles

1. **Java holds handles; Swift holds Metal objects.** Every Metal object lives in a Swift registry and is referenced from Java by a 64-bit handle. Java never sees Objective-C pointers, and freeing a handle releases the object.
2. **Zero-copy uploads through unified memory.** Buffers use `.storageModeShared`. The bridge returns `contents()` as an address, and Java writes into it through `MemorySegment.ofAddress(ptr).reinterpret(size)`. That makes `writeToBuffer` a Java-side `copyFrom` with no native call. Transient per-frame memory is a ring over one shared buffer.
3. **Batch per render pass, not per call.** `RenderPassBackend` calls (setPipeline, setVertexBuffer, draw…) append compact records to a native command stream. The stream is a shared-memory byte buffer written from Java, and Swift replays it into one `MTLRenderCommandEncoder` at `submitRenderPass`. That costs one FFM call per pass instead of one per draw. Per-call FFM downcalls are cheap (tens of ns), but batching also keeps Metal encoding on one native thread.
4. **Shaders go SPIR-V → MSL in Java, compile in Swift.** Use the bundled `lwjgl-spvc` (`CompilerMSL` confirmed) with explicit resource bindings, then `MTLDevice.makeLibrary(source:)` asynchronously, mapped onto `BackendRenderPipeline.Pending`. Cache the translated MSL on disk, keyed by a SPIR-V hash.
5. **Clip-space and winding fixes live in the translator,** not scattered through code: Y flip, depth range (the frontend already reports `isZZeroToOne`, so Metal gets true), and front-face winding.
6. **Surface: SDL3 window with `SDL_WINDOW_METAL`,** `SDL_Metal_CreateView`, and `SDL_Metal_GetLayer` producing a `CAMetalLayer`. IMMEDIATE sets `displaySyncEnabled = false`. Test in a window whether native Metal escapes the 120 Hz pacing MoltenVK shows.
   - **Measured (2026-09-25, spike 1b):** native `CAMetalLayer` clear+present in an 854×480 SDL window (drawable 1708×960) gives 119.1 fps with `displaySyncEnabled=true` and 120.0 fps with `false`. So the windowed 120 Hz cap is the macOS compositor, not MoltenVK. Windowed comparisons between backends are therefore capped; use fullscreen for all backend benchmarks.
7. **Terrain `drawIndexedIndirect` maps to indirect draws, with ICBs later.** Vanilla already issues 20-byte indirect records per section, so this can reuse the M2 path.

## Order of work (each step boots the game or falls back cleanly)

1. `DeviceInfo`, limits and features filled from the real `MTLDevice`. Keep declining until surfaces exist.
2. Surface and present: clear to a color and present. Frame pacing test in a window.
3. Buffers, textures and views with the zero-copy upload path.
4. Pipeline translation (SPIR-V → MSL) with the disk cache. First target: the GUI and blit pipelines.
5. Render passes via the command stream. **Checkpoint: title screen.**
6. World: terrain (indirect), entities, depth. **Checkpoint: claudeworld renders, and a block edit updates.**
7. Benchmark against `results/baseline-26.3` (same mod, route and settings).

## Risks to watch

- Threading: pipeline compiles happen on worker threads, while encoding stays on the render thread.
- `GpuFence`/`GpuQueryPool` semantics vs `MTLSharedEvent` and counter sample buffers (Apple GPUs only sample at stage boundaries).
- Texture formats Metal lacks (for example some packed or 3-channel formats) need conversions at upload.
- Anything outside Renderpearl that assumes GL or Vulkan: the static scan found none, but runtime will tell.
