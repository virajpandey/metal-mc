# Vanilla Minecraft 26.3 baseline on Apple M3 Pro (2026-09-25)

These are the numbers any Metal backend has to beat. They come from the `mod/` benchmark and were collected unattended.

## Setup

| Item | Value |
|---|---|
| Machine | Apple M3 Pro, macOS 26.6.2, plugged in, Low Power Mode off |
| Game | Minecraft 26.3, Fabric Loader 0.19.5, Fabric API 0.161.0, dev client (offline profile) |
| World | `fixtures/claudeworld`, reset from a pristine copy before every run (a run grows it from 14 to 34 MB with new chunks) |
| Settings | render distance 12, simulation distance 8, vsync off, max FPS 260 (unlimited), sound muted, `pauseOnLostFocus:false` |
| Path | 20 s warm-up at the start pose, then a 60 s orbit at radius 140 blocks, y = 150, looking 25° down at the center |
| Metric | wall time between consecutive `Minecraft.renderFrame` calls |

## Fullscreen (4112×2580): the comparison that counts

| Run | Backend | FPS | Mean ms | p50 | p95 | p99 | Max | Stutters (>2× median) | Frames >12 ms |
|---|---|---|---|---|---|---|---|---|---|
| opengl_fullscreen1 | OpenGL (`4.1 Metal - 90.5`) | 181.2 | 5.52 | 5.32 | 6.97 | 7.69 | 13.9 | 19 | 9 |
| opengl_fullscreen2 | OpenGL | 177.2 | 5.64 | 5.48 | 7.14 | 8.57 | 17.7 | 21 | 10 |
| vulkan_fullscreen1 | Vulkan (MoltenVK 1.4.2) | 251.9 | 3.97 | 4.09 | 6.07 | 7.20 | 13.2 | 45 | 1 |
| vulkan_fullscreen2 † | Vulkan | 250.5 | 3.99 | 4.04 | 7.52 | 8.44 | 15.2 | 395 | 7 |
| vulkan_fullscreen3 | Vulkan | 253.8 | 3.94 | 4.06 | 6.03 | 7.15 | 15.6 | 55 | — |

**Result:** in fullscreen, Vulkan through MoltenVK averages 252 fps against OpenGL's 179 fps: about 41% more frames and 29% lower mean frame time. p99 is similar or slightly better on Vulkan.

† **Flagged, not excluded.** 359 of this run's 395 stutters fall between 20 and 40 s. The camera path is identical in every run, and runs 1 and 3 are calm at that point, so terrain doesn't explain it. It didn't recur in run 3. The cause is still unidentified (a background process, or a MoltenVK/compositor episode).

**Stutter caveat:** the ">2× median" threshold is relative. Vulkan's lower median gives it a stricter bar (about 8 ms) than OpenGL's (about 10.6 ms), so the absolute ">12 ms" column is also shown.

## Windowed (854×480): presentation, not rendering

| Run | Backend | FPS | Mean ms | p99 | Stutters | Present mode |
|---|---|---|---|---|---|---|
| opengl_try1 | OpenGL | 230.8 | 4.33 | 8.34 | 82 | (not logged) |
| vulkan_try1 | Vulkan | 120.0 | 8.34 | 10.48 | 1 | (not logged) |
| vulkan_try2 | Vulkan | 119.9 | 8.34 | 10.41 | 2 | IMMEDIATE (MoltenVK offers IMMEDIATE and FIFO) |

- Windowed Vulkan stays at the 120 Hz display refresh even in IMMEDIATE present mode. Fullscreen removes the cap. So windowed Vulkan results measure macOS window-compositor pacing, not rendering speed. Anyone comparing backends on a Mac needs to know this, including readers of Titanium-style benchmarks.
- OpenGL loses only about 20% going from 0.41 MP (windowed) to 10.6 MP (fullscreen), about 26× the pixels. That suggests OpenGL on macOS is mostly CPU-bound here, which fits its per-section draw path on GL 4.1.

## GPU timing: not usable yet

Vanilla's `TimerQuery` only runs while the F3 "GPU utilization" line is on (`-PbenchGpu=1`), and that line cost about 18% FPS in the OpenGL run (231 → 190). On OpenGL it reads 0, so the GL backend reports no GPU time. On windowed Vulkan it read 79% (about 6.6 ms/frame). That figure probably includes time waiting for the compositor, since fullscreen Vulkan frames take 3.9 ms in total, so it is not treated as GPU work. The next step is a Metal System Trace (`xctrace`), which sees both backends' GPU work at the Metal level.

## Scope

These results cover one machine, one route, 60 s runs, and 2–3 runs per configuration, taken at night with no user activity. They are a baseline for this workload, not a general claim about Minecraft on macOS.
