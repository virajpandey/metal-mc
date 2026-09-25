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

## Correctness check

`metal_fullscreen1-start.png` and `vulkan_ctl1-start.png` are the benchmark's own screenshots at the same pose. They read the main render target back through `copyTextureToBuffer`, so the readback path is exercised too. Downscaled to 1028×645, 95.1% of pixels agree within 15/255 on every channel. The remaining 4.9% are one region: the player's arm, which sits in a different animation pose in each run (arm bob depends on sub-tick frame timing). Terrain, water, foliage, the village, clouds, sky, fog, and the hotbar match.

No pipeline failed to translate or compile (errors would be logged as `Couldn't compile Metal pipeline` or `Couldn't translate`), and the native side logged no GPU errors.

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
