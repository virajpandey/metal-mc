# MetalMC native Metal backend vs vanilla OpenGL and Vulkan (Minecraft 26.3, Apple M3 Pro)

First results from the native Metal backend (`mod/src/client/java/metalmc/backend`, `Sources/MetalMCNative/Backend.swift`), collected 2026-09-25 with the same benchmark, world, route, and settings as [the baseline](../baseline-26.3/README.md). Every configuration in the table below ran in the same unattended session, interleaved, with the world reset from the pristine fixture before each run.

## Fullscreen (4112×2580)

| Run | Backend | FPS | Mean ms | p50 | p95 | p99 | Max | Frames >8 ms | Frames >12 ms |
|---|---|---|---|---|---|---|---|---|---|
| metal_fullscreen1 | Metal (MetalMC) | 403.7 | 2.48 | 2.44 | 3.27 | 4.12 | 27.4 | 17 | 2 |
| vulkan_ctl1 | Vulkan (MoltenVK 1.4.2) | 251.6 | 3.98 | 4.10 | 6.11 | 7.27 | 34.2 | 58 | 3 |
| metal_fullscreen2 | Metal | 393.6 | 2.54 | 2.50 | 3.32 | 4.03 | 27.2 | 20 | 2 |
| metal_fullscreen3 | Metal | 407.0 | 2.46 | 2.42 | 3.31 | 4.04 | 27.1 | 16 | 3 |
| opengl_ctl1 | OpenGL (`4.1 Metal - 90.5`) | 179.2 | 5.58 | 5.38 | 7.09 | 7.81 | 32.5 | 79 | 7 |
| vulkan_ctl2 | Vulkan | 249.4 | 4.01 | 4.11 | 6.17 | 7.21 | 30.2 | 55 | 4 |
| metal_fullscreen4 | Metal | 404.6 | 2.47 | 2.43 | 3.26 | 4.07 | 27.9 | 13 | 3 |

**Result:** the Metal backend averages **402 fps** (4 runs, 394–407) against **250 fps** for Vulkan through MoltenVK (2 runs) and **179 fps** for OpenGL. That is 1.61× Vulkan and 2.24× OpenGL. Mean frame time drops from 3.99 ms (Vulkan) to 2.49 ms, and p99 from about 7.2 ms to about 4.1 ms. In absolute terms Metal also has the fewest slow frames: 13–20 frames over 8 ms per run, against 55–58 for Vulkan and 79 for OpenGL, while drawing about 60% more frames than Vulkan.

The relative stutter metric (`stutters_gt2x_median` in the .txt files) is higher for Metal (95–108) because its median is 2.4 ms, so its ">2× median" bar is only 4.9 ms. The absolute columns above are the fair comparison.

## Where the time goes

`metal_gpu_ms_*` (in runs 3 and 4) is each frame's command buffer, measured from `gpuStartTime` to `gpuEndTime`: a mean of 2.70 ms and a p99 of 3.69 ms. That is longer than the 2.46 ms wall time per frame because consecutive frames overlap on the GPU. **The GPU is saturated: at this resolution the backend is GPU-bound, not CPU-bound.** Further gains have to come from the GPU side (render-pass load/store traffic, the present blit, shader cost), not from the Java→native call path.

## Late drawable acquisition (2026-09-26)

The conclusion above, that the backend is GPU-bound, was only half right. A profile that also samples native code (`jdk.NativeMethodSample`) showed the render thread at RD 12 blocked in `CAMetalLayer.nextDrawable` for about two thirds of each frame (88.6% of its native samples, which JFR takes half as often as Java samples). Minecraft acquires the surface at the start of its frame. It renders into its own target, though, and only the final copy writes the drawable. Acquiring at the start held each of the three drawables for the whole frame, so the pool ran dry.

`mmc_surface2_acquire` now does nothing, and the drawable is fetched in `mmc_surface2_blit`, right before the copy (`METALMC_EXP=earlyacquire` restores the old behavior). A/B runs, back to back, fullscreen, noon:

| Run | Early acquire | Late acquire |
|---|---|---|
| 4 km world, RD 12 | 359.3 fps, p99 4.45 ms, 110 stutters | **402.2 fps**, p99 4.13 ms, 65 stutters |
| 8 km world, RD 12 + LOD 8192 | 301.7 fps | **311.7 fps** |
| 4 km world, RD 32 | 173.2 fps (earlier the same night) | **179.5 fps** |

"Stutters" counts frames longer than 2× the median. The GPU time per command buffer goes up (2.44 → 2.82 ms at RD 12) because more frames now overlap on the GPU.

Repeat runs at RD 12, alternating: early 357.8 and 359.3 fps; late 407.7, 407.9 and 402.2 fps. That's +13%.

**Runs paced by the display.** Some runs, with either setting, came out at almost exactly the display's 120 Hz (120.0, 135.2, 144.0 fps), with GPU time per frame near 7 ms and almost no stutters. Macs pace a fullscreen window through the compositor at the refresh rate when it can't present directly: the display idled to sleep (bench runs now use `caffeinate -di`), or something overlays the window. These runs are excluded from comparisons.

**Tried and dropped: lazy clears.** Minecraft clears its targets with separate passes (6 per frame at RD 12). Folding each clear into the load action of the next pass that renders to that target removed all of them (6.0 → 0.1 per frame). The 11-scene tour matched. It didn't change frame rate or GPU time (clear passes: 415.0 fps, 2.93 ms; lazy: 396.6–417.5 fps, 2.85–3.09 ms). Apple GPUs appear to make full clears cheap already, so the code was reverted.

**Two other suspects, ruled out.**
- The profile charged 17.5% of the render thread to vanilla's `ChunkSectionsToRender$DrawIndirect.render`. Native timing of the indexed-indirect draws (`per_frame_indirect_cpu_ms`) shows only 0.11–0.13 ms per frame for about 1,600 draws at RD 12, and 0.58 ms for about 8,100 at RD 32. JFR attributes critical downcalls to their Java caller, which inflated that figure.
- A standalone microbenchmark (`scratchpad icbbench`: 1,600 draws × 500 quads) measured Metal's own cost. Encoding 1,600 indirect draws costs 0.10–0.14 ms of CPU. An indirect command buffer written by a compute kernel costs 0.02 ms and was also 15–25% faster on the GPU at that draw size. So ICBs would save about 0.1 ms of CPU at RD 12. They'd need the terrain's textures in argument buffers, which isn't worth it yet.

## Occlusion culling for chunk sections (2026-09-26)

After the late-acquire fix, RD 12 fullscreen runs at the rate macOS presents frames: skipping the present copy entirely (`METALMC_EXP=noblit`) didn't raise fps (405.5 vs 413.6). Higher render distances are GPU-bound on terrain geometry. At RD 32 the main pass is 4.3 ms of a 5.5 ms frame (vertex 2.6, fragment 1.7). Vanilla culls sections outside the frustum and those its visibility graph rules out (sealed caves), but it still draws distant surface sections hidden behind hills.

`metalmc.terrain.SectionOcclusion` and `SectionOcclusion.swift` reuse the LOD's box test:
- **Test.** After solid terrain (and the LOD), each candidate section's box (16 blocks, grown by 1) is rasterized in the same pass with depth testing and no writes. A fragment function with early fragment tests marks the section visible.
- **Skip.** Sections found hidden get an empty mesh when vanilla builds its draw lists. They're dropped before any draw group or section data is created; skipping the draw itself left empty groups and crashed vanilla's `writeDataBatchedMultiple`.
- **Test cadence.** Hidden sections are re-tested every frame, so they come back 2–3 frames after coming into view. Visible ones are re-tested every 8th frame, staggered by key. Testing every visible box every frame made the high orbit 44% slower, because each visible box runs the marking shader over every pixel it covers.
- **Near sections.** Sections within 48 blocks of the camera are never tested or skipped.
- **Switch.** `occlusionCulling` in the config, `-PocclusionCulling=0` for A/B.

4 km world, RD 32, fullscreen, noon, back to back:

| Run | Culling off | Culling on |
|---|---|---|
| Ground level (`-PbenchY=ground`) | 247.3 fps, p99 6.49 ms, 6,093 draws | **347.6 fps (+41%)**, p99 5.07 ms, 2,060 draws |
| High orbit (y = 150, 25° down) | 179.0 fps, p99 8.15 ms, 8,096 draws | **204.3 fps (+14%)**, p99 7.13 ms, 5,631 draws |

Correctness checks, all against culling-off runs:
- **Bench screenshots:** start and mid-orbit (the camera is moving at mid-orbit) differ on 0.01–0.23% of pixels, all animals near the player.
- **11-scene rendering tour:** differs only where content is random (rain, particles, F3 text, Nether and End mobs).
- **8-view LOD tour:** 0.01–0.31% of pixels differ, again animals.

## Correctness check

`metal_fullscreen1-start.png` and `vulkan_ctl1-start.png` are the benchmark's own screenshots at the same pose. They read the main render target back through `copyTextureToBuffer`, so the readback path is exercised too. Downscaled to 1028×645, 95.1% of pixels agree within 15/255 on every channel. The remaining 4.9% are one region: the player's arm, which sits in a different animation pose in each run (arm bob depends on sub-tick frame timing). Terrain, water, foliage, the village, clouds, sky, fog, and the hotbar match.

No pipeline failed to translate or compile (errors would be logged as `Couldn't compile Metal pipeline` or `Couldn't translate`), and the native side logged no GPU errors.

## Render distance scaling (fullscreen, same session, vanilla terrain path)

| Render distance | OpenGL | Vulkan (MoltenVK) | Metal (MetalMC) | Metal vs Vulkan | Metal draws/frame |
|---|---|---|---|---|---|
| 12 | 179 fps | 250 fps | 402 fps | +61% | 1,129 |
| 24 | 112 fps | 214 fps | 238 fps | +11% | 3,655 |
| 32 | 74 fps | 154 fps | 170 fps | +10% | 5,640 |

At RD 12 the frame is split roughly evenly between CPU and GPU work, so Metal's lower overhead shows fully. At RD 24–32 both modern backends are GPU-bound on vanilla's terrain. Per-pass timing at RD 32 (`-PbenchTrace=1`): the main world pass takes 4.7 ms, of which **3.2 ms is the vertex/tiling stage and 1.5 ms is fragments**. Far terrain is geometry-bound: most far triangles cover few pixels, so vertex work dominates. The fixes are fewer and smaller vertices (face culling, compact formats, LOD), not faster pixels.

## Face-direction culling (Sodium's idea, in `metalmc.terrain`)

At mesh-build time each opaque/cutout section layer's quads are sorted into 7 buckets (±X, ±Y, ±Z, and unaligned). At draw time only buckets that can face the camera are drawn, with adjacent visible buckets merged into one draw. Back faces were already culled by the rasterizer, so the image is unchanged, but the GPU no longer transforms them. It's implemented as mixins on vanilla's `SectionCompiler`, `CompiledSectionMesh`, and `LevelRenderer.extractSectionDrawGroups`, so it helps every backend.

| RD 32, fullscreen | Before | With face culling | Change |
|---|---|---|---|
| Metal fps | 170.0 | 187.9 | +10.5% |
| Metal GPU ms per frame (command buffer) | 8.27 | 6.57 | −21% |
| Vulkan fps | 154.0 | 168.2 | +9.2% |
| Metal draws per frame | 5,640 | 8,248 | +46% |

At RD 12 (CPU-bound) fps is unchanged (395), with GPU time down 10%. The rendering tour with face culling on matches the tour without it in all 11 scenes (only animated or random content differs), and the RD 12 benchmark frame shows no holes. The mod now enables it by default; `-PfacingCulling=0` turns it off for vanilla comparisons.

**Tried and parked: indirect command buffers.** Encoding the per-section indirect draws into a Metal ICB (one `executeCommandsInBuffer` instead of one encoder call per section) fails for vanilla terrain. Metal refuses ICB support for pipelines whose shaders take directly bound textures or samplers ("Vertex/Fragment shader cannot be used with indirect command buffers"), and the terrain samples the lightmap in the vertex stage and the atlas in the fragment stage. Doing this needs textures in argument buffers. It's behind `-PmetalExp=icb` and off by default. It would cut CPU time, not GPU time, so it matters at low render distances.

## Rendering coverage tour

`-PbenchTour=1` runs a scripted tour (`mod/src/client/java/metalmc/bench/Tour.java`) and takes a screenshot at the end of each step. I ran it once on Metal and once on Vulkan, windowed (1708×960), then diffed the screenshots pairwise at 854×480. The composites in `tour/` are Metal | Vulkan | mask, with differences over 24/255 in red.

| Step | What it exercises | Pixels differing >24/255 | Cause of the difference |
|---|---|---|---|
| 00 day | terrain, water, clouds, sky, fog, hand | 4.33% | arm (random skin, animation phase) |
| 01 sunset | sunrise/sunset triangle fan (emulated on Metal), sky colors | 3.76% | arm |
| 02 night | stars, moon, night lighting | 0.93% | arm |
| 03 rain | rain and splash particles | 6.47% | random rain streaks, arm |
| 04 scene | glass, stained glass, water, lava, torch, chest, enchanting table, leaves, flower, end portal, bed, banner, sign text, pig, charged creeper, zombie with glinting helmet, villager, glowing sheep (outline post effect), armor stand | 4.40% | arm, idle animations |
| 05 particles | flame and villager particles | 5.97% | random particles, arm |
| 06 f3 | debug overlay text, chunk-border and hitbox lines | 5.98% | FPS and backend-name text, particles, arm |
| 07 pause | pause menu with blur post effect | 4.43% | arm |
| 08 inventory | inventory screen, item rendering, glint on items | 1.31% | player model pose |
| 09 nether | Nether terrain, fog, lava | 0.57% | — |
| 10 end | End sky, obsidian pillars, crystals, boss bar, held glinting sword | 0.46% | — |

Every difference traces to something that isn't deterministic between runs. None looks like a rendering error. The crops `tour/crop_scene_metal.png` and `tour/crop_scene_vulkan.png` show the scene at full resolution, and they are indistinguishable.

A first version of the tour showed chunk borders in Vulkan's pause and End screenshots but not in Metal's. That was a bug in the tour script, not in the renderer. Debug-overlay statuses persist in `debug-profile.json`, and `toggleStatus` is a 3-state machine, so the same toggles gave different end states depending on the previous run. The tour now sets the statuses explicitly, and runs use a fixed `--username` so the skin is the same each time.

## Windowed

`metal_first` (windowed, 1708×960 drawable) runs at 119.9 fps. That is the macOS compositor's 120 Hz cap, the same one Vulkan hits and the same one a bare `CAMetalLayer` clear/present loop hits (see `docs/backend-design.md`). Windowed numbers don't compare backends on this machine.

## What is not covered yet

- Timestamp queries (vanilla's GPU-utilization line) read as unavailable on Metal.
- Performance was measured on only this route (overworld, daytime, render distance 12). The tour checks correctness, not speed, in the other scenes. Snow, chunk-heavy flight, and long sessions haven't been exercised yet.
- Four runs from one session on one machine. This is a strong result for this workload, not a general claim.
