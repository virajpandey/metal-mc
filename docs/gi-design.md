# Bounce light: a world-space irradiance cache (design + prototype; in the frame with lit mode, `METALMC_EXP=lit,gi`)

The goal is the part of SEUS PTGI that is hardest to fake: indirect light that is noise-free, stable while the camera
moves, reaches the whole LOD, and fits a locked 120 Hz at the panel's native 3456 x 2234 (8.33 ms a frame). The budget
set for it is 1.5 ms of GPU per frame, everything included.

`Sources/MetalMCNative/Gi.swift` has the kernels and an offline test; `tools/gicache.py` drives the test. In the game it
runs with lit mode (`METALMC_EXP=lit,gi`): RtShadows runs its frame and the relight takes its light in place of its sky
term (lighting-design.md, "Lit mode with the GI cache"); `gi` without `lit` changes nothing. Numbers marked *measured*
come from the offline tests on the M3 Pro; *estimate* means reasoned, not measured.

## Why a cache on the world

Minecraft is axis-aligned block faces that rarely change. Indirect light is low-frequency and changes slowly, so it can be
computed on the world, where it stays put, instead of per pixel, where it has to be recomputed (and denoised) every frame:

- **Stable in motion by construction.** A cell's light doesn't depend on the camera, so nothing swims or reprojects.
- **Noise-free by accumulation.** A cell averages its last 64 updates of 4 samples each; the screen only interpolates.
- **Cost set by a budget, not by pixels.** Rays are traced per cell (16,384 cells a frame by default), not per pixel.
- **Multiple bounces for free.** A bounce ray reads the cell it lands on, which already holds that cell's light, so every
  update adds a bounce.
- **Reaches the LOD.** Cells grow with distance (level L covers 2^L blocks), so a cell covers about 10 pixels everywhere.

This is close to NVIDIA's spatial hash radiance cache (SHaRC) and to Lumen's surface cache, specialized to block faces:
the key is a block face, so no surface parameterization is needed and walls a block thick don't leak.

## Data structure

- **Cells.** A cell is (the air block in front of a face, the face's direction, a level). At level 0 it is one block face;
  at level L it holds the faces of that direction whose air block lies in a 2^L cube aligned with the LOD's voxels (y
  counted from the world bottom). The level is picked from the distance to the camera: 0 inside `METALMC_GIRANGE` (160
  blocks, where a block is about 10 px on the panel), then one more per doubling of distance (level 1 to 320, level 2 to
  640, level 5 to 5 km).
- **Key.** 64 bits: x and z in cells (24 bits each, modulo 2^24: two cells 16 M cells apart never share a view), y (9),
  face (3), level (4). Unwrapped against the acceleration structure's origin when a cell is updated.
- **Table.** Open addressing in 16-slot buckets. A slot's 32-bit fingerprint (a second hash of the key; 0 = empty) is what
  lookups compare, and a bucket's 16 fingerprints are one 64-byte line: a lookup is four 16-byte loads. Inserting claims
  the first empty slot with a 32-bit compare-and-swap (the M3 has no 64-bit one); every thread tries the empty slots in
  the same order and a failed exchange reveals what took the slot, so two threads inserting the same key end in the same
  slot. Evictions leave holes, so lookups scan the whole bucket.
- **Per slot (36 bytes):** fingerprint 4, key and the cell's first-seen surface point 16, last frame seen 4, light 8 (RGB
  irradiance from the sky and bounced light as halfs, and 1 + the share of the sun's disk the cell sees; 0 means no
  samples, so cleared or evicted memory never reads as a black cell), update count and flags 4.
- **Memory.** 2 M slots by default (`METALMC_GICAP=21`): 72 MB. *Measured:* the test view at the panel's resolution needed
  100 K cells for what it shows and 240 K more that its bounce rays reached, 340 K in all (16% of the table), with no
  bucket ever full; after turning around, 463 K. Cells unseen for 10 s are evicted. What is visible scales with pixels,
  not with the world, so the table's size doesn't depend on the LOD distance.

## Each frame (two compute encoders)

The first encoder makes the frame's light (`gi_light`), clears the counters (`gi_begin`) and runs the request with the
resolve (`gi_request`); the second runs the schedule (`gi_schedule`, `gi_args`) and the update (`gi_update`). The relight
reads only what the first wrote, so it waits for that one alone and the update runs beside the frame's later passes; the
resolve shows the cells as the last frame's update left them (a cell's new light shows a frame later).

1. **Request and resolve** (one thread per 4 x 4 pixels, one pass over the depth buffer):
   - **Request** (one pixel of the block, a different one each frame, Bayer order): rebuild the surface from the depth
     buffer (position, and the face from the neighbors' depths, as `rt_shadow` does), find or create its cell (near a
     level boundary the next level's too), mark it seen. A new cell starts from its parent's light (the next level's
     cell holding it) counted as two updates, so moving closer refines the light instead of starting black. The first
     thread to touch a cell in a frame lists it for an update; a cell covering 7 x 7 pixels or more holds a whole aligned
     4 x 4 block and is touched every frame. When more cells are visible than their part of the list holds, a random
     share of them (from last frame's count) is listed, a different one each frame: appending them all filled the list
     with the same cells every frame, the first in dispatch order: 23% of the resolve's samples had light after 128
     frames, against 74% now (the rest is sky).
   - **Resolve** (the block's 2 x 2 half-resolution samples): per 2 x 2 pixels, the nearest surface; bilinear across
     the 2 x 2 nearest cells on that surface's plane (cells that don't exist or have no samples are left out, so light
     doesn't cross a wall), blended toward the next level in the last quarter of each level's range. Output: one RG32Uint
     texel per sample, the irradiance as RG11B10Float's bits (packed in the shader, truncated as the M3's texture unit
     converts: the self-test checks it) and a code word (the surface's plane as a half, its face, and whether its cells
     have samples).
2. **Schedule** (a slice of the table, 1/64 to 1/8 of it, sized so the cells it lists fill their quarter of the budget):
   evict cells unseen for 10 s; list cells that aren't visible every 8th time the sweep passes them, and cells with no
   samples yet whenever it does.
3. **Update** (the list, up to 16,384 cells: three quarters for visible cells, one quarter for the rest, so light off
   screen keeps flowing). Per cell, 4 samples, one per thread (the cell's 4 threads sit side by side in a SIMD group and
   sum their samples with shuffles), each:
   - a point on the face (level 0: on the face's plane; coarser: a probe ray from the cell's far side back along the
     normal finds the surface, passing through solid with back faces culled; if the faces don't cover that part of the
     cell, the cell's first-seen point instead);
   - a ray toward a point of the sun's disk (visibility, stored apart from the irradiance);
   - one cosine-weighted bounce ray: the sky where it escapes, else the light leaving the surface it reaches, from that
     surface's cell (its irradiance plus its sun share times the sun, plus block light, times its albedo); if that cell
     has no samples yet, one more ray for its sun and half the open sky.

   Samples follow the R2 low-discrepancy sequence per cell (so an average of many updates is stratified), and a cell
   averages its last 64 updates (a running mean until then). Bounce rays of visible cells create the cells they reach
   (behind the camera, around corners), which then keep light bouncing off them; bounce rays of other cells create
   nothing, or the cache grows into everything rays can reach (the first version reached 928 K cells in 128 frames and
   filled buckets).
4. **Upsample, in the lighting pass** (`giUpsample`): each pixel takes its own block's sample if it's on the same face and
   plane (one read, issued as soon as the pixel is known to be terrain: its address depends on nothing else), else the
   nearest of the 8 around that is (planes up to a block and a half apart count: the riser of a one-block step is often a
   pixel wide; their code words from four 2 x 2 gathers, then one read of the chosen sample). If none matches, the
   cells' own lookups (`giIrradiance`; lit mode's relight leaves its own sky term there), except where the code says the
   cells have no samples yet.

## What shading does with it

Direct sunlight stays the per-pixel ray-traced shadow (RtShadows): sharp, and the cache can't hold shadow edges finer
than a block. Direct block light stays the flood fill vanilla and the LOD already carry (noise-free, occluded around
corners). The cache adds what those can't: sky light that follows the real openings (a room with a small window is dark
inside; vanilla's flood fill lights it by the distance from the opening, not by how much sky the opening shows) and
bounced light, sun, sky and block light alike:

    color = albedo * (sun * shadow + cache irradiance * ao + block light) + emission

- **Contact detail.** Cells are a block wide and bilinear across a face, so corners and crevices need the per-vertex AO
  vanilla and the LOD already compute (or a screen-space AO), multiplied into the cache's term only.
- **Entities** aren't in the cache; they can take the cell under them, or the six cells around their position (one per
  face direction) as a crude irradiance volume (*design only*).
- **Where there's no data** (sky, past the cached range, cells not sampled yet), the lighting pass keeps its own ambient:
  vanilla's sky light from the lightmap. A turn of 180 degrees leaves 28% of surface pixels there for the first frame,
  0% after 32 frames (*measured*); new cells start from their parents, and cells behind the camera that bounce rays
  reached already have light, so most of the view has data at once.

## Measurements (offline, M3 Pro)

Real terrain: `r.0.0` of the claudeworld-merged fixture at level 0 (its four quarters) and its 8 neighbors at level 1,
3.45 M triangles, meshed as the LOD does; per-tile structures as RtShadows builds them. Camera 140 blocks up at the
region's north edge looking south 20 degrees down, sun at -45 degrees, 3456 x 2234, `METALMC_GIREPEAT=4` (each stage run
4 times per frame after a warm-up, so the GPU clock is where a busy frame would put it):
`python3 tools/gicache.py region <r.0.0.mca> <dir>`.

| Stage | GPU ms | What it scales with |
|---|---|---|
| request | 0.15-0.21 | pixels / 16 |
| schedule | 0.02-0.03 | the slice of the table |
| update | 0.34-0.47 | budget: 13-16 K cells, 119-162 K rays (250-370 M rays a second) |
| resolve | 0.42-0.50 | pixels / 4 |
| **the cache's stages** | **1.04-1.17** | |
| lighting pass: upsample and fallback lookups | 0.30-0.46 more | pixels (0.28-0.35 without the fallback lookups) |

The last row is what the cache adds to a deferred lighting pass that reads depth and a G-buffer anyway (the test kernel
timed with and without the cache's work). In all, **1.35-1.6 ms, at the 1.5 ms budget.** With G-buffer normals the request
and the resolve drop their normal reconstruction (4 extra depth reads and unprojections per sample): *estimate* 20-30%
less for those two.

How the resolve got there (the same view):

| Resolve | GPU ms |
|---|---|
| full resolution, 4 lookups per pixel, RGBA16F out | 1.55-2.3 (0.6 of it the reconstruction and the 62 MB write) |
| full resolution, one thread per 2 x 2 sharing lookups by face | 1.54 (divergent: edge blocks run the body up to 4 times) |
| half resolution | **0.42-0.50** |

and the lighting pass's share of it (the cache's work only): 4 bilinear taps of packed samples 0.42-0.86 ms; a gather of
the codes and one filtered read where all four match 0.65-0.91 (in this world so many pixels lie near a face edge that
the fast path rarely applies); the nearest matching sample **0.30-0.46**. The high ends are just after a turn, when many
cells have no samples yet: taking their lookups anyway cost up to 1.1 ms before the code word flagged them.

Quality:

- **Noise** (the synthetic house: a room lit through a 4 x 3 window, the hardest case; mean absolute 8-bit error of the
  irradiance view against a 16-sample, 255-update reference; `bench_out/results/gi-v1/house-noise-*.png`): 2 samples over 32
  updates 5.53 (frame-to-frame change 2.4%); **4 over 64 (default) 2.86 (1.5%)**. Outdoors the frame-to-frame change is
  0.05-0.07%, in the glowstone room 0.22%.
- **Spatial reuse biases enclosed rooms** (`METALMC_GISPATIAL`, off): blending the coplanar neighbors' averages into each
  update halves the flicker but the error went to 8-20, because the room got brighter: the irradiance view's mean went
  up 27% (the glowstone room 7%, outdoors 0%). Along a floor it moves light from cells that see the sky to cells the
  room reflects more, and multiple bounces amplify it. Any spatial filtering belongs at read time, not in the
  transport.
- **Edits** (`python3 tools/gicache.py edit`: a hole in the house's east wall and one in its roof after 96 frames): the
  converged cells went from 602 to 416 (the rest restarted), and the 83 whose faces vanished were evicted by the
  re-check in the first frame (listed by the invalidation itself: before that, only the sweep reached them, 18 within
  96 frames, and their neighbors kept reading their stale light); 1 frame later the new sun patches are lit and their
  bounce is noisy; after 32 frames (0.27 s at 120 Hz) the room is settled (`bench_out/results/gi-v1/edit-*.png`).
- **Real terrain** (`bench_out/results/gi-v1/region-lit-128.png` vs `region-nocache.png`, vanilla's sky light as flat ambient):
  shaded faces pick up warm bounce from sunlit ground, crevices and the ravine darken; `region-cells.png` shows the cells
  (about 10 px) and the level rings; `region-turn-1.png` and `region-turn-32.png` are one frame and 32 frames after a
  180-degree turn.

Acceleration structures with the per-triangle data the bounce rays need (material, face, block light, sky cover: 4
bytes): 41.9 -> 64.6 bytes per triangle compacted, +54% (*measured*, 3.45 M triangles, +75 MB). At the game's 26 M
triangles that would be about +590 MB (*estimate*). The zero-copy alternative, integrated: one geometry per quad range in
each tile's structure, and the hit's geometry and primitive index look up the quad in the node's own buffer (bound by
GPU address in a per-instance table, `GiTile`): 41.8 bytes per triangle, the same as without the data (*measured*,
3.41 M triangles, 136.0 MB either way), and the dependent read per bounce hit costs nothing measurable (update 0.41 ms
against 0.46 with the data, back to back).

## Cutting the frame cost (2026-10-02)

In the game (native, the full look with the cache, a flight over real terrain, medians of traced frames) the cache cost
about 2.1 ms against its 1.5 ms budget: light, begin and request 0.38 ms; schedule and update 0.68; the resolve 0.52;
and the upsample, in the anti-aliasing resolve with lit mode's relight, 0.55 (2.35 ms against 1.80 without the cache).
None of the changes below changes the light: what the relight takes from the cache is the same to the bit on one saved
table, or, where a change moves when cells update, it differs from the old build's by no more than two runs of the old
build differ from each other.

**How it was measured** (offline, M3 Pro; `python3 tools/gicache.py bench <r.0.0.mca> <dir>`): the region view above at
3456 x 2234 in lit mode's light (sun at -45 degrees), the cache's frame as the game encodes it, each frame after 96 MB of
other traffic (the cache's reads start cold, as in a frame) and the shadow rays' stand-in (`gi_test_shadow`), then the
anti-aliasing resolve's load loop with the relight's use of the cache (`gi_test_taa`, a proxy: Taa.swift's 32 x 32
tiles, a row of pixels per SIMD group, the depth and a G-buffer stand-in read per pixel). Per encoder with the game
profile's timestamps, the encoders one after another (a fence: each time is its own; without one, a frame's encoders
overlap and so do the end of one frame and the start of the next, which made the first per-encoder numbers
meaningless), and the frames with the cache less the frames without it, alternating, the encoders overlapping as in the
game: what the cache costs the frame. Kernel variants ran against the built-in kernels in one process, alternating, on
one converged table (`exp`); `tools/litflow.swift` (the game's own calls on the real-terrain fixture with the sky, at
the panel's resolution) gave the anti-aliasing resolve with and without the cache's light (`Lit.debugTimeTaa`) and whole
frames with and without the cache.

| GPU ms, *measured* (medians; ranges over runs) | before | after |
|---|---|---|
| light, begin, request | 0.225-0.229 | one pass with the resolve: |
| resolve (half resolution) | 0.478-0.482 | **0.58-0.59** for both |
| schedule, update (rays) | 0.453-0.460 | **0.30-0.35** |
| the cache's encoders, first start to last end | 1.16-1.18 | 0.89-0.93 |
| the upsample in the anti-aliasing resolve (the proxy with it less without) | 0.39 | 0.28-0.29 |
| **what the cache costs the frame** | **+1.61-1.62** | **+1.27-1.28** (+1.27 flying 0.17 blocks a frame) |
| litflow: the anti-aliasing resolve with the cache's light less without | +0.35-0.45 | +0.23 |
| litflow: the frame with the cache less without (the relight in its own pass / in the anti-aliasing resolve) | +1.41 / +1.51 | +1.06 / +1.13 |

The bench's "before" is the old kernels with the new one-texel output (which the bench needs); litflow's "before" is the
old build. What changed, and what each was worth (*measured*, the bench unless noted):

- **The light and its code word in one texel** (RG32Uint, where an RG11B10Float texture and an R32Uint one were). The
  upsample's common case (93% of the surface pixels) is one read whose address depends only on the pixel, and the
  relight issues it right after its depth test; it read the code word, then the light the code picked. The shader packs
  the irradiance into RG11B10Float's bits itself, truncating as the M3's texture unit converts (rounding to nearest
  differed from the hardware's conversion in 16,592 of 25,856 test values; truncating, in none: `selftest` checks it),
  so the bits are the old ones: what the relight takes was the same to the bit as the old build's on a saved table.
- **The update in lanes:** each of a cell's 4 samples on a thread of its own (`giLanes`; the 4 threads sit side by side
  in a SIMD group, take what decides the samples from the first lane and sum by shuffles), so each thread's chain of
  dependent rays is a quarter as long and four times as many rays are in flight (16 K cells are only a few hundred SIMD
  groups on the whole GPU). With one atomic per SIMD group for the request's list (it was one per listed cell) and a
  plain read of a cell's stamp before the exchange (a cell takes about six request samples a frame; only the first
  needs the exchange): the update 0.460 -> 0.343, the frame -0.12. The sums' order changed (by shuffles), so the
  light converged from empty differs from the old build's by 1.25% (mean absolute difference of the relight's input),
  as two runs of the old build differ (1.24%).
- **The request and the resolve in one pass** over the depth buffer (`gi_request`; `gi_resolve` stays for the views and
  tests). Both read all of it (31 MB at the panel's resolution: every cache line, though the request needs one pixel in
  16) and the same visible cells: 0.711 -> 0.623 together, the frame -0.06. The resolve now shows the cells as the last
  frame's update left them, so the update has an encoder of its own and runs beside the frame's later passes: the relight
  waits for the first encoder only (in the game it waited for all of the cache's work, one encoder outside traced
  frames). A cell's new light shows a frame later; the light converged from empty: 1.26% from the old build's (noise
  1.24%).
- **The upsample's neighbors in four gathers:** when its own sample doesn't match (an edge, a face too small for a
  sample), the 8 neighbors' code words come from four 2 x 2 gathers, not 8 reads (the anti-aliasing resolve runs a row
  of pixels per SIMD group, so nearly every group has a pixel that needs them): the proxy 0.983 -> 0.913, the frame
  -0.07; the same bits.
- **The lookups:** a bucket's four 16-byte loads issued together and the first match taken after them (an early exit
  after each made a SIMD group wait for them one after another), and the bilinear's four corners' scans, then their
  values, with no branch between: the request and resolve 0.622 -> 0.580, the frame -0.05; the same bits but for a few
  pixels one RG11B10 step apart (none over 1/32; likely the compiler fusing multiply-adds differently).
- **The relight takes the cache's sun and sky light:** with the atmosphere the cache runs `lit_env` on the frame's sky
  tables for its own light, and the relight ran it again in a compute encoder of its own; now it reads the cache's
  (the same kernel on the same tables in the same frame: the same values).

Measured and dropped:

- The G-buffer's face in the resolve and the request instead of the depth buffer's neighbors: the resolve +0.035 ms,
  the request +0.134 (its 8 bytes a pixel against the depth reads it saves), and the resolve's samples changed at
  edges.
- The resolve's lookups shared across a SIMD group (a lane looks up a cell its neighbors need too): +0.06 (only a
  quarter of the lookups were shared). Shared by a thread's four samples instead (their cells looked up once, all
  together: 4 lookups where 16 were, most often): +0.42, the same bits (likely the four samples' state held at once:
  more registers, fewer threads in flight). The resolve's 2 x 2 depths in one gather: no change.
- The upsample's scan of the 8 neighbors stopping at one on the pixel's own plane (the same choice): +0.065 in the
  anti-aliasing resolve.
- Intersector hints for the update (geometry type, opacity; the hit's face from the quad, not the triangle data): no
  change. Updating converged cells less often (every 2nd, 4th, 8th time): nothing saved at this budget (the list fills
  anyway), and the light follows the sun more slowly.
- Fewer rays: `METALMC_GIBUDGET=8192`, or `METALMC_GISPP=2` with `METALMC_GIHISTORY=128` (the same samples per average),
  halve the update's rays (143 K a frame to 72 K): the update 0.35 -> 0.22 or 0.25 ms, the frame's cost +1.29 -> +1.16
  or +1.21. But the converged light moves from the defaults' by 1.53-1.56% (mean absolute difference, against
  0.87-1.24% between two default runs), and 384 frames after a 20-degree jump of the sun it is 2.85-2.92% from the
  light settled there, against 2.52%: more noise and more lag. Not taken.

In the game (not run here): traced frames show the cache as two encoders, "GI cache: light, begin, request and resolve"
and "GI cache: schedule, update (rays)"; the flight and the fidelity tour in lighting-design.md ("Lit mode with the GI
cache") give its cost (with `gi` less without) and the look (against the old build's: the same light, within its
noise). By the offline ratios per stage the 2.1 ms there should come to about 1.5-1.6 ms (*estimate*), short of the
1.0 ms asked for. What is left is set by what the cache does more than by how: about 140 K rays a frame, the depth
buffer read once (31 MB at the panel's resolution), and the half-resolution light written once and read once in the
anti-aliasing resolve (15 MB each). Half the rays (above) would save about 0.2-0.3 ms more there (*estimate*: the
update costs about twice the offline one in the game), with their noise and lag.

## Emissive light (torches, lava, glowstone)

- **Now:** direct block light is vanilla's flood fill (per face, `giBlockLight`, vanilla's lightmap curve); the cache
  bounces it (surfaces lit by a torch light others). Emissive faces (lava, glowstone) seen by bounce rays add nothing
  themselves, since their light already reaches neighbors through the flood fill; `look.y` turns that emission on for
  the far LOD, where levels 5 and up carry no block light (a lava lake 6 km away should still glow on its rim).
- **Next (design):** many-light direct lighting with shadows, cache-driven. Each cell keeps a small reservoir (a light
  index and its weight, 8 bytes: the slot's unused fourth key word and 4 more) of the light that mattered most to it,
  resampled from the LOD's existing light lists (`LodGrid.emitters` at level 0, `lights` above) with the neighbors'
  reservoirs, as ReGIR does on a grid (Boksansky et al. 2021); shading then traces one shadow ray per pixel to its cell's
  chosen light instead of taking the flood fill. Torch shadows at the cost of one ray per pixel (*estimate* 0.3-0.5 ms at
  quarter resolution, like RtShadows).

## Invalidation on block edits

`GiCache.invalidate(x:y:z:)` queues an edited block. Next frame `gi_invalidate` restarts the averages of level-0 cells
within 6 blocks (kept at one update, so new samples take over at once) and of the 3 x 3 x 3 cells around it at every
coarser level (kept at four), and lists the level-0 cells next to it to re-check their surface in that frame's update,
which probes for the face and evicts the cell if it's gone. Light further away follows through the bounces within a
second or two. The acceleration structures are rebuilt by whatever rebuilds the tile (in the game, a node rebuild when
the chunk reaches the LOD).

## Far terrain (height-field rings, design)

Past the ray-traced range the far field's rings hold one column per cell. There a cache of per-column sky visibility is
cheap: march the ring's max pyramid in 8 azimuths for the horizon angle (the same traversal the far field's rays use),
visibility = the average of 1 - sin(horizon); bounce = (1 - visibility) x the neighbors' mean albedo x their mean
irradiance. Stored per column when a ring is refilled (2 bytes), read by the far field's shading. *Estimate:* 8 x tens of
steps per column over 2048^2 columns per ring, spread over the refill (which already runs one ring per frame at most).

## Alternatives for this game on this GPU

| Technique | For | Against | Verdict |
|---|---|---|---|
| **World-space face cache** (this) | stable in motion (world-space); noise-free once converged; cost set by budget; multiple bounces free; reaches the LOD; 1-block walls don't leak | lags lighting changes by 0.3-4 s (edits are invalidated; day and night lag); a block per cell needs AO for contact detail; entities not in it | the base |
| DDGI probes | proven (1-2 ms a frame on an RTX 2080 Ti at 1080p, its author's report); entities and particles sample it | a probe grid over 3D air, most of it sky; 1-block walls need probes every block or leak (visibility tests only go so far); relocating probes out of blocks | worse fit: the world is surfaces |
| Screen-space GI | cheap (about 0.5 ms at half resolution, *estimate*) | misses what's off screen and behind; unstable as things disocclude; needs a denoiser | only for contact detail, on top |
| Voxel cone tracing | Minecraft is voxels; no RT hardware needed | leaks through thin walls (mip averaging); 3D clipmaps only reach so far; several cones per pixel (*estimate* 1-3 ms at 1080p on desktop GPUs) | out at the panel's resolution |
| ReSTIR GI | sharp, correct indirect detail | per-pixel paths plus a denoiser: 8.9 ms on an RTX 3090 at 1080p (paper) | later, at quarter resolution, reading this cache at its second bounce |
| Radiance cascades | smooth, no denoiser (2D) | 3D is an open research problem; screen-space versions miss off-screen light | watch |

## Prototype

- `Sources/MetalMCNative/Gi.swift`: `GiCache` (table, pipelines, `encodeFrame` for the game and the stages separately),
  the kernels `gi_light`, `gi_request`, `gi_schedule`, `gi_args`, `gi_update`, `gi_resolve`, `gi_invalidate`,
  `gi_count`, the lighting pass's `giUpsample` (`giUpsampleHeader`, shared with lit mode's relight) and the debug view
  `gi_debug_view` (lit, irradiance, cells, samples, without the cache, bounce only, and lit mode's relight without and
  with the cache, and 9: what the relight takes from the cache, as floats for comparisons), plus offline-test kernels
  and entry points (`mmc_debug_gi_scene`, `_run`, `_set_block`, `_reset`, `_selftest`, `_lit_light`; `_profile`, the
  game's frame with its timestamps; `_view`, `_save`, `_load`, `_diff`, for pictures of one table compared to the bit;
  `_shader_source`, `_reload`, for kernel variants timed against the built-in ones in one process; and
  `mmc_debug_gi_frame`, `mmc_debug_lit_gi` for lit mode's A/B).
- The light (`GiLight`, buffer 15): the sun, the sky (the atmosphere's sky view table or a gradient), the ground below the
  horizon, the open sky on each face direction, and the scale the cells' light is read at. The offline test fills it with
  the prototype's light (scale 1); lit mode makes it each frame (`gi_light`) from the relight's own light, per unit of
  daylight (lighting-design.md, "Lit mode with the GI cache").
- Switches: `METALMC_EXP=lit,gi` (in the frame with lit mode; `gi` alone does nothing in the game), `METALMC_GICAP`
  (log2 slots, 21), `METALMC_GIBUDGET` (cells a frame, 16384), `METALMC_GISPP` (samples an update, 4),
  `METALMC_GIHISTORY` (updates averaged, 64), `METALMC_GIRANGE` (level-0 range, 160), `METALMC_GISPATIAL` (0, off:
  biased), `METALMC_GIREPEAT` (offline timing only); in the game `-PgiCap`, `-PgiBudget`, `-PgiSpp`, `-PgiHistory`,
  `-PgiRange`.
- Offline test: `python3 tools/gicache.py selftest | synth <dir> | litsynth <dir> | edit <dir> | region <r.X.Z.mca> <dir>`
  (`METALMC_DYLIB` for another build; zero-copy structures as the game builds them, `GICACHE_PRIMDATA=1` for per-triangle
  data). The synthetic scene is a LOD grid built in code (a house with a window and a door,
  a white floor and a red wall, a closed room lit by glowstone, a hill with a tunnel); the region test meshes a fixture
  region as the LOD does. `selftest` checks 1,500 key round trips (both signs, both sides of 2^23, the world's bottom and
  top, every face and level), ray-winding cases, and the resolve's RG11B10 packing against the texture unit's (25,856
  values). Costs: `bench <r.X.Z.mca> <dir>` (the game's frame per encoder, and what it costs the frame; "Cutting the
  frame cost" below), `cmp` (two pictures of what the relight takes, to the bit and statistically), `src`, `exp` (kernel
  variants against the built-in ones: times, pictures on one table, the light converged from empty against two built-in
  runs) and `lag` (how fast the light follows a jump of the sun).
- Found on the way: the LOD's faces, counterclockwise seen from outside, are back faces to Metal's intersector with a
  counterclockwise front winding: `clockwise` is right (`gi_test_rays`).

## Integration (with lit mode, `METALMC_EXP=lit,gi`; lighting-design.md, "Lit mode with the GI cache")

1. **Triangle data: done, zero copy.** RtShadows' tile structures have one geometry per quad range (`giTileMesh`), and a
   per-instance table points into the node buffers (`GiTile`, `giHitQuad`). `primitiveDataBuffer` (`giTileGeometry`,
   +54% structure memory) remains for offline comparisons.
2. **Per frame: done.** In `RtShadows.trace`, after the shadow rays, on the same depth, jittered projection, origin and
   instance structure: `GiCache.shared.encodeFrame`, into a half-resolution RG32Uint texture (light and code word) the
   relight takes, with the sun and sky light the cache made from the atmosphere (`lit_env`'s output: the relight's
   own, so it doesn't run it again).
3. **Shading: done, in lit mode's relight** (its G-buffer has the albedo): `giUpsample` replaces the sky term where it
   has data, times AO.
4. **Edits.** Block changes near the player (a client-side block update hook) call `GiCache.invalidate`; a chunk the LOD
   rebuilds could invalidate its changed columns in bulk. Not done.
5. **Sky: done.** With our sky, the sky view table and the relight's own sun and sky light; without, lit mode's daylight
   curve.
6. **Dimensions.** One cache per dimension (keys are world positions), cleared on a new world; the Nether has no sky.
   Not done (lit mode runs in the overworld only; the one cache isn't cleared on a world change).
7. **Gradle: done** (`-PgiCap`, `-PgiBudget`, `-PgiSpp`, `-PgiHistory`, `-PgiRange`).

The checks to run in the game are in lighting-design.md ("Lit mode with the GI cache", "Not done, and what to look at in
game").

## Next

- Split each cell's light by source (sky, sun, block; 3 x RGB halfs) and scale at read time. Lit mode already reads the
  cells per unit of the frame's daylight (sun plus sky), so a change of daylight shows at once and only the change of
  the sun-to-sky ratio lags by the history; the split would make that exact too and carry bounced block light, which
  lit mode leaves out (its scale doesn't follow the daylight). 16 bytes more per slot (104 MB at 2 M slots, from 72).
- Guide bounce rays toward bright cells a cell has found (a direction reservoir per cell): the remaining noise is small
  bright sources seen through small openings.
- The far-field column cache.
- Measure in the game: the frame cost with the LOD's real structures (26 M triangles), and how many cells a flight keeps
  alive.
