# LOD v1: render distance 12 plus far terrain to 2 km, on Metal

First in-game results for the far-terrain LOD (`docs/lod-design.md`), 2026-09-25.

## Setup

| Item | Value |
|---|---|
| World | `fixtures/claudeworld-big`: `claudeworld` pregenerated to 257 × 257 chunks (4.1 km square, 66,049 chunks, 684 MB) with `-PbenchPregen=128` |
| Route | Baseline orbit (radius 140 blocks at y = 150, 25° down) after a 40 s warm-up, so the LOD build (about 26 s in-game) is done before timing starts |
| Display | Fullscreen 4112 × 2580, M3 Pro |
| Backend | Metal, face culling on |

## Results

| Run | Visible distance | FPS | Mean ms | p99 ms | Metal GPU ms/frame |
|---|---|---|---|---|---|
| Vanilla RD 12 (`big_rd12`) | 192 blocks | 391.3 | 2.56 | — | 2.45 |
| Vanilla RD 32 (`big_rd32`) | 512 blocks | 186.4 | 5.37 | — | 6.53 |
| RD 12 + LOD, first version (`lod_rd12`) | 2,048 blocks | 162.0 | 6.17 | 8.16 | 10.58 |
| **RD 12 + LOD, with node frustum culling and face buckets (`lod2_rd12`)** | **2,048 blocks** | **245.3** | **4.08** | — | **6.56** |

| **RD 12 + LOD, per-tile culling (`lod_tiles`)** | **2,048 blocks** | **281.7** | **3.55** | — | **5.54** |

**Per-tile culling** (later the same day): each node is split into 4 × 4 tiles of 64 voxels, with greedy merges kept inside tiles and per-tile height ranges. Each tile is culled against the frustum, by facing, and when it lies entirely inside vanilla's range. That took LOD from 244 to 282 fps (GPU 6.6 → 5.5 ms/frame), again pixel-identical (0.04% of pixels differ).

**Result:** RD 12 plus LOD draws terrain out to 2 km, 4× vanilla RD 32's distance and 16× the area, and, with per-tile culling, runs **51% faster than RD 32** (282 fps vs 186).

The culling step (skipping nodes outside the view frustum and face directions that face away from the camera) took the LOD from 162 to 245 fps. The screenshot is pixel-identical to the unculled version: 0.01% of pixels differ, all on the animated arm.

## 8 km

`fixtures/claudeworld-huge` is the same seed pregenerated to 513 × 513 chunks (8.2 km square, 263,169 chunks, 2.3 GB, 324 region files) with `-PbenchPregen=256`, resumed after two memory-limited attempts. With `-PlodFar=8192`, the LOD builds 128 nodes on 5 levels (11.8 M quads, 95 MB of GPU memory, 106 MB of RLE quadrant cache) from 324 regions in 12.6 s standalone and 18 s in-game.

| Run | Terrain visible to | FPS | Mean ms | Metal GPU ms/frame |
|---|---|---|---|---|
| RD 12 + LOD 8192 (`lod8k2_rd12`, noon, clear weather) | **8,192 blocks** | **270.3** | 3.70 | 5.86 |

That's 16× vanilla RD 32's view distance at 45% more FPS. `rd12-lod8192-start.png`: coastlines, islands, and snow-capped mountains to the horizon. Two fixes made this work:

- **Quadtree selection.** A split node now draws its own tiles for any missing child quarter. The old rule, "split only if all four children exist", drew the coarsest (32-block) level next to the camera whenever the world didn't fill a coarse node.
- **Clear-weather haze.** The haze now stretches to the LOD distance. Vanilla's linear 0 → 1,024-block haze hid everything past 1 km.

`-PbenchNoon=1` sets clear weather and noon, frozen, because the pregenerated fixture was saved during a rainstorm.

**Showcase tour** (`-PbenchTour=lod`, `tour-8k/`): horizontal views from 3 blocks above the ground and from y = 260 in four directions. These are the hardest views for LOD, since the seam and the horizon are at eye level. At ground level the vanilla-to-LOD transition behind hills and villages isn't noticeable. From y = 260 the terrain runs to an 8 km horizon, with vanilla's clouds correctly layered above it. The ground shots still show the last rain streaks, because the tour switches the weather to clear when it starts.

## Deep-cave filling (2026-09-25)

New counters showed where LOD time goes. The CPU side takes 0.06–0.08 ms per frame. The GPU cost is the quads: 786–896 K per frame. A mesh audit (`mmc_debug_lod_mesh_stats`) found that 49% of level-1 quads were cave walls more than 16 blocks underground. They sit in cave networks connected to a surface entrance somewhere, so the sealed-cave fill kept them. The LOD now also fills air more than 4 voxels below the lowest open ground within 8 voxels (see `docs/lod-design.md`).

A/B runs use the same build, fullscreen, RD 12, noon and clear weather. The baseline is `METALMC_EXP=nodeepfill`.

| Run | FPS | LOD quads/frame | Metal GPU ms/frame |
|---|---|---|---|
| 4 km world, LOD 2048, before (`lodm_2k_old`) | 276.4 | 786 K | 5.52 |
| 4 km world, LOD 2048, deep fill (`lodm_2k_deep`) | **325.5** | 455 K | 4.30 |
| 8 km world, LOD 8192, before (`lodm_8k`) | 267.2 | 896 K | 5.82 |
| 8 km world, LOD 8192, deep fill (`lodm_8k_deep`) | **319.5** | 567 K | 4.69 |
| 4 km world, vanilla RD 32, no LOD (`van_rd32_noon`) | 184.9 | none | 6.45 |

At 8 km, total LOD quads dropped from 11.8 M to 7.7 M and GPU memory from 95 to 62 MB. The main pass went from 2.55 to 2.00 ms (vanilla alone is 1.05 ms). All 8 showcase-tour views and the bench screenshots match the baseline. The only differing pixels (0.01–0.4%) are animals near the player that moved between runs.

## Occlusion culling (2026-09-25)

Each candidate tile's bounding box is rasterized right after the LOD draws, in the same pass, with depth test and no writes. Tiles whose box was fully hidden are skipped on the next frames (see `docs/lod-design.md`). `-PbenchY=ground` flies the same orbit 2 blocks above the terrain, looking nearly level. That's the usual in-game view, and there hills hide most far terrain.

8 km world, LOD 8192, RD 12, noon, fullscreen:

| Run | FPS | LOD quads/frame | LOD draws/frame | Metal GPU ms/frame |
|---|---|---|---|---|
| Ground, no LOD (`lodg_off`) | 381.8 | none | none | 2.48 |
| Ground, LOD, no occlusion (`lodg_8k_noocc`) | 336.9 | 579 K | 430 | 4.00 |
| Ground, LOD, occlusion (`lodg_8k_final`) | **362.1** | 170 K | 79 | 2.95 |
| High orbit, LOD, no occlusion (`lodm_8k_deep`) | 319.5 | 567 K | 552 | 4.69 |
| High orbit, LOD, occlusion (`lodm_8k_final`) | 318.1 | 424 K | 377 | 4.44 |

- **At ground level** the 8 km LOD now costs 5% of frame rate relative to no LOD at all.
- **From the high orbit** (150 blocks up, looking 25° down) only about a quarter of the LOD is hidden, and the test roughly pays for itself: repeat runs gave 318–325 fps.
- **Variants measured along the way:**
  - Running the test without skipping any tiles (`occnocull`) costs 11 fps.
  - Drawing every box face, rather than only the faces toward the camera, was 4% slower than no occlusion from the high orbit.
  - Shrinking the LOD vertex outputs (half-precision color, fog distances instead of positions) changed nothing (316.6 fps) and was reverted. Vertex count, not vertex size, is what costs.
- **Correctness:** all 8 showcase-tour views match the no-occlusion run. The high views are pixel-identical. The ground views differ on 0.06–0.32% of pixels, all of them animals near the player. The moving-camera bench screenshots match too.

## Build cost

The LOD is built in the background when the world opens: 81 non-empty regions in 26 s in-game (18 s standalone). The result is 74 nodes on 3 levels, 7.7 M quads, 61 MB of GPU memory. Filling sealed caves and not emitting faces under water halved the quad count.

## What it looks like

`rd12-lod2048-start.png`: vanilla chunks near the camera (textured village, cherry trees, crater) continue into flat-colored LOD terrain (rivers, lakes, forests as tree clusters, mountains, ocean) that fades into vanilla's own fog at 2 km.

**Colors:** the first screenshot used hand-picked material colors, which were too bright and saturated. `rd12-lod2048-texture-colors.png` uses colors computed from Minecraft's own block textures: the mean of each material's opaque texels, with the plains grass, foliage, and water tints (`LodColors.swift`, generated by `lod_colors.py`). The seam between textured vanilla chunks and flat LOD is now hard to spot. Frame rate is unchanged (244 fps).

## Known issues

- Grass, foliage, and water use the plains tint everywhere; the LOD doesn't read biomes yet. Surfaces are flat colors, so the seam still shows as a loss of texture detail up close.
- The LOD is static, built once from the save.
- Quads within vanilla's render distance still cost vertex work (they are collapsed in the vertex shader).
- One run per configuration so far.
