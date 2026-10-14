# Frame budget: where the 8.33 ms go (native 3456 x 2234, 120 Hz)

The goal is a locked 120 Hz at the panel's resolution with the full look: lit mode, ray-traced sun shadows, the sky,
the GI cache, and later reflections and water. Locked means the slowest frames fit too, so the 99th percentile matters
more than the mean. The method is the same for every part of the frame:

1. **Measure.** Each part's GPU time comes from a traced flight: per-pass timestamps at stage boundaries, three frames
   at each eighth of the real-terrain flight (21 frames), median per pass (`-PbenchTrace=1`, `tools/bench/passes.py`).
   Passes overlap on the GPU, so a pass's "total" can include waiting on the one before it; the vertex and fragment stage
   times are the cleaner numbers. Where one pass holds several parts, switches that skip a part split it (the
   difference is that part's cost).
2. **Work out the floor.** What the part has to cost from first principles: bytes moved (at the M3 Pro's 150 GB/s, each
   byte per pixel of the panel's 7.72 M pixels costs at least 0.05 ms), lookups and rays, vertex invocations (about
   2.6 billion a second on this GPU, measured), and divergence (a SIMD group of 32 pixels pays for its slowest).
3. **Cut where the gap is biggest**, and measure again.

## The frame (2026-10-02, the real-terrain flight at y 150, uncapped, TAA on)

Medians per pass. "Lit" is `-PmetalExp=lit,nearchunks,rtshadows,sky`, "+GI" adds `gi`.

| Part | Default | Lit | Notes |
|---|---|---|---|
| Sky at quarter resolution (compute) | | 0.23-0.38 | the range is overlap with the clear before it |
| Sky pass (full resolution) | 0.05 | 0.21 | the quarter-resolution sky filtered up: 4 B/px written (0.2 ms floor) |
| **Main pass** | **4.55** (vertex 1.68, fragment 3.02) | **4.88** (vertex 1.73, fragment 3.25) | vanilla terrain, the LOD's quads, the far field, entities, water |
| - the LOD's quads, vertex work | | 1.33 | `-PmetalExp=...,lodskip` takes the vertex stage from 1.73 to 0.40 ms: 0.9 M quads, 3.5 M vertex invocations |
| - the far field's march | | 1.78 | `ffskip` takes the fragment stage from 3.25 to 1.47 ms |
| - everything else's fragments | | 1.47 | of which the G-buffer's 8 B/px store about 0.34 (lit minus default) |
| Ray-traced shadows | | 0.32 | one ray per 4 x 4 pixels, rotating; the anti-aliasing accumulates them |
| GI cache (+GI) | | 1.37 | its stages split in traced frames since |
| Anti-aliasing resolve | 0.74 | 2.01, 1.80 with 32 x 32 tiles | with the relight (about 0.47) and the aerial perspective (about 0.43) in its load (offline, litflow) |
| Anti-aliasing copy (+ sharpening) | | 0.38 | at its floor: 4 B/px read, 4 B/px written |
| Hand, HUD | 0.06, 0.24 | 0.06, 0.24 | each loads and stores the whole frame |
| Present (copy to the screen, flipped) | 0.56 | 0.35-0.59 | 0.64 with HDR (its drawable is 8 B/px) |

Frame times: default 150.2 fps (6.66 ms, p99 10.4); lit 117.1 fps (8.54 ms), 120.5 (8.30) with 32 x 32 tiles; lit with GI
95.3 fps (10.49 ms), 98.3 (10.17) with 32 x 32 tiles; lit with HDR +0.33 ms (the present).

**Baseline after the 2026-10-02 work** (2026-10-05 02:55-03:31, screen unlocked, same flight, uncapped unless noted;
builders were coding in other worktrees at the same time, so these lean slightly pessimistic):

| Run | fps | p99 (ms) | Main pass vertex / fragment (ms) | Before (2026-10-02) |
|---|---|---|---|---|
| Default | 184.3 (warm-up run 187.7) | 7.42 | 2.43 / 2.53 | 150.2 fps |
| Lit | 126.0 | 10.17 | 1.66 / 2.89 | 117.1, 120.5 with 32 x 32 tiles |
| Lit + `quadcull` | 127.4 | 10.10 | 1.46 / 2.98 | |
| Lit + GI | 103.1 | 12.02 | 1.83 / 2.97 | 95.3, 98.3 |
| Default, vsync (120 Hz) | 119.8 | 9.86 | | |
| Lit, vsync (120 Hz) | 119.2 | 10.72 | | |

In the lit + GI frame the GI cache's stages took 0.77 (light, begin, request and resolve) and 0.47 ms (schedule and
the rays); the resolve with relight and aerial perspective 2.32 ms.

## What the measurements ruled in and out

- **The anti-aliasing resolve's tiles** (done): each 16 x 16 threadgroup loaded its border again, and in lit mode every
  load is the relight, the shadows' shade and the aerial perspective: 27% more loads than pixels, and the last few
  loads ran on 3 of 8 SIMD groups. 32 x 32 tiles: 13%. Resolve 2.01 -> 1.80 ms in lit mode, the GI frame 10.49 -> 10.17 ms.
- **Culling the LOD's quads before the vertex stage** (measured, `METALMC_EXP=cullstats`): of the 0.9-1.07 M level-0
  quads a frame draws, 2-6% face away from the camera, 5-6% are off screen, and 20-23% are so thin that no pixel
  center falls in their bounding box: about 30% in all, about 0.4 ms of vertex work. The other 50% that own no pixel
  are hidden behind nearer terrain at a fine grain, which only an occlusion test per quad (two-pass, against a depth
  pyramid) could find.
- **Moving the quad/march boundary inward** (`-Plod0`, fidelity tour): the near band stays (4.58, 4.60, 4.61 for 768, 512,
  384), but the far band against the level-0 answer key goes from 3.33 to 4.89 (512) and 5.62 (384): the far field's
  2-block columns are visibly coarser than level 0's quads at 512-768 blocks. The boundary stays at 768 unless the far
  field gets a 1-block ring.
- **Pass folding** (done, on by default; `METALMC_EXP=nofold` turns it off): clears and the anti-aliasing's copy become
  the next pass's load action or first draw instead of passes that store the whole frame for the next to load back. Two
  clears and the copy fold every frame; lit mode 118.3 -> 119.9 fps (about 0.1 ms: the copy's own reading and writing
  stays, only the store and load around it go); the frame is unchanged (the flight's screenshots differ only where the
  clouds drifted). The first version folded nothing: Minecraft uploads uniforms between a clear and the pass it folds
  into, and the upload's blit encoder ran what was waiting. Uploads now leave it alone.
- **The relight is bandwidth-bound, not math-bound** (offline, `tools/litflow.swift ... sky time`): handing its linear
  light straight to the aerial perspective (skipping an encode and a decode, six powers per pixel) changed the resolve
  from 1.640 to 1.630 ms, nothing. The relight's 0.46 ms is its G-buffer read: 8 bytes per pixel, 13% more for the
  tiles' borders, is 70 MB, 0.46 ms at 150 GB/s. With the G-buffer's store in the main pass (0.22-0.34 ms), the deferred
  relight costs about 0.75 ms in memory traffic; only a smaller G-buffer (hard: the albedo alone is 24 bits) or keeping
  it on chip would cut that. The change was reverted.

- **The far field's march, cheaper** (done, merged 2026-10-02; docs/far-field-design.md has the details): cheaper steps,
  climbing only into untested parents, water stored at its drawn surface in the pyramid, and a per-ring horizon
  profile (rays skip the rings whose terrain they pass over). Offline 19-28% less march time with an exact image (a few
  hundred pixels change where rays used to hit the step cap). In game (default settings, traced): the main pass's
  fragment stage 3.02 -> 2.53 ms (confirmed unlocked on 2026-10-05; the 2.63 first seen came from a run with the
  lock screen up). Fidelity unchanged (far band 3.35 against 3.33-3.34, near band 4.57 against 4.58), and
  the horizon tour (LOD 262144) shows no holes (0.004-0.7% of pixels changed against last night's, water animation).
- **The plain resolve keeps 16 x 16 tiles** (done): with 32 x 32 tiles it took 0.85 ms against 0.74 (its 14 KB of
  threadgroup memory against 4 costs more than the smaller border saves); only the heavy variants (sky, lit) use 32.
- **The GI cache, cheaper** (done, merged 2026-10-02; docs/gi-design.md has the details): the light and its code word in
  one texel per half-resolution sample (the upsample's common case is one read), the fallback's neighbors by gathers,
  each cell's 4 samples on 4 lanes (four times the rays in flight), request and resolve fused into one pass over the depth,
  faster hash lookups, and the relight reusing the cache's sun and sky light. Offline the cache's cost per frame went
  from +1.61 to +1.27 ms (litflow, the game's call sequence: +1.51 -> +1.13, and +1.18 on the merged build); the light
  matches within run-to-run noise. In game: the 2.1 ms measured tonight should come to about 1.5-1.6 (to measure).
  Left for Viraj to decide: half the rays (`-PgiBudget=8192`, or `-PgiSpp=2 -PgiHistory=128`) saves another 0.2-0.3 ms,
  and moves the converged light by about 1.5% (two runs of the defaults differ by about 1%).
- **LOD quad culling** (built, off by default: `METALMC_EXP=quadcull`; docs/lod-design.md, "Quad culling"): a compute
  pass leaves out the level-0 quads that can't make a fragment (off screen, no pixel center in the bounding box, both
  triangles facing away), and the survivors are drawn with indexed indirect draws. Exact: no pixel of color, depth or
  G-buffer changes in any view checked. Offline it takes the vertex stage down 0.08-0.40 ms depending on how much of
  the view is cullable (15-36%), for a pass that costs 0.17-0.24 ms: 0.03-0.30 ms net. The in-game flight (about 30%
  cullable, cullstats) should land near the top of that range; it becomes the default only if the in-game A/B shows it.
- **Water that reflects** (built, a look feature, off by default: `METALMC_EXP=water` with lit mode and the sky;
  docs/lighting-design.md): the sky reflected with Fresnel, the sun's glint (shadowed by the traced visibility), small
  waves that roughen into the distance, for the far field's and the LOD's water (vanilla's near water not yet: it needs a
  translucent-pipeline variant). Cost offline: about 0.35 ms a frame with a third of the screen water (0.08 ms with only
  the far field's sea), bit for bit the old frame without the switch. Next: terrain in the reflections (screen space).
- **The screen locks itself after 15 minutes with no input** (2026-10-02 at 05:15:37; the screen saver's idle time and
  password are set by the Mac's managed policy). loginwindow checks every 15 minutes and starts the screen saver unless
  something holds a display-sleep assertion; the game holds one while it runs and `caffeinate -d` does too (its log says
  so: "PMNoDisplaySleepEnabled so do not launch screen saver"), so it locked in an 8-minute gap between two runner
  starts. runq.sh holds `caffeinate -dims` while it runs, and a separate keep-awake covers the gaps on test nights. The
  lock screen over the fullscreen game paces it at 120 Hz like a dialog does (the GPU's frame time drops below the
  frame time: the command buffers stop overlapping), so timed flights are void until it's unlocked; runq.sh holds them
  while it's locked. Fidelity tours (screenshots read from the render target) and offline tests are unaffected. A
  fidelity tour between flights also makes the next flight rebuild the flight world's LOD cache (the two fixtures share
  the world's name): keep flights together, or give the first one more -PbenchExtraWait.

## Next levers, by size (ideas with reasoning; none measured yet)

- **The far field's grazing rays.** In the flight's view the far band (768 blocks to the horizon) is only about 190
  pixel rows: a camera 80 blocks above the terrain sees land 768 m away 6 degrees down and land 4 km away 1.1 degrees
  down, while a pixel is 0.031 degrees. So the march's 1.78 ms is spent on well under a tenth of the screen, thousands of
  cycles per pixel: rays near the horizon skim the terrain through every ring (each ring starts its march over), and
  the sky just above the horizon pays for proving that nothing is hit. A per-frame horizon profile per ring (for each
  azimuth, the highest elevation angle any terrain in that ring reaches from the camera, conservative over the azimuth
  bin and the cell, from a coarse level of the max pyramid) would let a ray skip every ring it passes over and skip the
  march outright where it's above the horizon of all of them: exact, and one table read per ring.
- **Overlapping compute with the main pass's vertex stage.** For about 1.7 ms each frame the GPU is bound by the
  rate it invokes vertices, a fixed-function limit, while its ALUs have room. Work that doesn't depend on this frame's
  depth (the GI cache's rays for last frame's requests, the sky's tables) could run at the same time on a second
  queue, synchronized with events.
- **Fewer pixels shaded, reconstructed temporally** (measured offline, 2026-10-02: tools/litflow.swift's lit frame with
  the sky and the GI cache, the merged build, rendered at a fraction of the panel's resolution on each axis):

  | Scale | Pixels | Main pass | Relight | Resolve (with relight) | Frame with GI | Frame without GI |
  |---|---|---|---|---|---|---|
  | 1.0 | 7.72 M | 3.48 | 0.70 | 1.79 | 7.06 | 5.91 |
  | 0.75 | 4.34 M | 2.52 | 0.42 | 1.03 | 4.85 (-2.21) | 3.93 (-1.98) |
  | 0.667 | 3.43 M | 2.32 | 0.33 | 0.83 | 4.32 (-2.74) | 3.47 (-2.44) |

  The reconstruction would run at the panel's resolution (the anti-aliasing core, about 0.74 ms, plus the upsampling
  filter), so at 0.75 the realistic saving is about 1.4 ms (lit) to 1.6 ms (lit with GI), the biggest single lever
  measured.
  Every per-pixel cost here scales with the panel's 7.7 M pixels, and
  at 254 pixels per inch the panel resolves finer than the eye does at a normal viewing distance (a pixel is about 0.7
  arcminutes at 50 cm). Rendering the level at 0.75 scale (56% of the pixels) and reconstructing native resolution in
  the anti-aliasing resolve would save about 2 ms of the lit frame. MetalFX's temporal scaler cost 9 ms here (Taa.swift),
  so it would be our own resolve. It changes the image (softer in motion), so it's a fidelity trade to measure, not a
  given; and the hand and the HUD must stay at native resolution, which means splitting vanilla's main target.
  (Built 2026-10-10: the next section.)

## Render scale, temporal upscaling and the governor (2026-10-10)

The level is drawn at a fraction of the panel's resolution on each axis and resolved to the panel's by our own
anti-aliasing; the hand, the HUD, the screen effects and post-processing stay at the panel's resolution. A governor can
steer the scale (and then per-feature quality) to hold the refresh rate.

Switches: `METALMC_SCALE` (gradle `-PrenderScale`, config `renderScale`): a number from 0.5 to 1 (1, the default: off,
the old path bit for bit), or `auto` for the governor (`auto:MIN:MAX`, default 0.6 to 1). Needs TAA (`-Ptaa=true`).
In lab mode `<shader dir>/render_scale.txt` (same syntax, read every 2 s) overrides it, so one session compares scales.
Tuning: `METALMC_TAAU_K` (the reconstruction's sharpness, 4.5), `METALMC_TAAU_SHARPEN` (the copy's sharpening, 0.6),
`METALMC_GOVBUDGET` (the governor's frame time, 8.333 ms).

**Splitting vanilla's main target** (`metalmc.render.RenderScale`, `GameRendererScaleMixin`). vanilla's level reaches
the main target only through `GameRenderer.mainRenderTarget()`: LevelRenderer's frame graph imports it as "main", clears
it, composites into it, and makes the sky renderer with it. The hand, the screen effects, the GUI and vanilla's post
effects use GameRenderer's own field. So from `renderLevel` to the anti-aliasing (`render3dHud`'s head), the accessor
answers a target of our own at the scaled size (same formats as the main target, HDR's float included), and our passes
before the anti-aliasing take it too. Everything sized from it follows without a change of its own: the G-buffer (the
level pass's second color target), the relight, the traced shadows, the GI cache's per-pixel request and resolve, the
water's rays, the sky's quarter-resolution view and its aerial perspective, the far field's march and the LOD (whose
quad culling tests the scaled pixel centers, so it culls more). The sky renderer keeps the target it was made with, so a
redirect points its pass at the frame's level target. The frame's start clears the main target; with a scale the clear
is left out (the upscaler's copy writes every pixel of its color, and vanilla clears its depth before the hand), since
with pass folding it would otherwise run as a pass of its own. Frames that can't be scaled draw at full resolution as
before: vanilla's post effects (spectating a creeper: they want the level's depth at full size), the external pipeline,
no TAA.

**The upscaler** (TAAU, `TaaUpscale.swift`, kernel `taa_upscale`): the anti-aliasing's resolve with two changes, after
the published temporal upsamplers (Karis's TAA, "High Quality Temporal Supersampling", 2014; the survey by Yang, Liu and
Salvi, 2020). Each output pixel takes this frame's samples near it, the 3 x 3 rendered texels around the nearest one,
each weighted by exp(-k d^2) for its distance d from the output pixel's center in output pixels (a sample sits at its
texel's center minus the frame's jitter); and the current frame's share is W / (W + (1 / b - 1) E[W]), W the weights'
sum, E[W] its mean over all jitters (the samples per output pixel times pi / k) and b the anti-aliasing's blend (0.1
still, up to 0.3 in motion): a pixel with a sample at its center takes more of this frame, one with none near takes
little, and on average it's b as before. The history, its reprojection, the Catmull-Rom history sample, the variance clip
(over the 3 x 3 texels) and the copy with contrast-adaptive sharpening are the anti-aliasing's, at the panel's
resolution, sharing its history (post-processing reads it unchanged). The jitter covers 16 Halton positions instead of
8. Each threadgroup loads a 32 x 32 tile of the rendered frame and its 1-texel border (the load's heavy part, the relight,
the water, the shadows' shade and the aerial perspective, runs once per rendered texel, 13% border as before), then
resolves every output pixel whose nearest sample this frame is one of the tile's texels: the tiles partition the output
exactly at any scale.

**The governor** (`RenderScale.swift`). Each frame's GPU time (its command buffer's, one submit per frame) is recorded
with the scale it was drawn at; only frames at the current scale count, so a change is judged on frames drawn after it.
Over the last 32: too slow is, uncapped, the GPU time's 90th percentile above 92% of the frame time; paced by the display
(vsync), where the GPU slows its clock to fit the work and its time stretches toward the frame's whatever the load, it's
the frame intervals' 90th percentile 4% past the refresh with the GPU over 80% busy. Then the scale drops to what would
bring the GPU time to 85% if all of it scaled with the pixels, on a grid of 0.05 (at least one step). A step up (0.05)
needs the same model to predict under 85% after it, held for 1 s: the model is the hysteresis, a step up never predicts a
step down. At the lowest scale the per-feature quality drops in quarters (the GI cache's cell updates a frame: its light
converges more slowly, nothing pops) and comes back before the scale does. Through the anti-aliasing a step of the scale
is a slight change of sharpness over a few frames, not a pop: the history stays at the panel's resolution. Features merged
later read `RenderScaleGovernor.shared.quality` (1 to 0) for their own steps (cloud and god-ray steps).

**Offline check** (`tools/taautest.swift`): the real kernels on a test image (a zone plate whose frequency rises past what
the rendered frame can hold, and a hard edge), a still view, scored against the image box-filtered over each output
pixel (RMS error; `bench_out/agents/speed/taautest/` in the main checkout):

| | all | where the drawn frame can hold the image | past it |
|---|---|---|---|
| TAA at the panel's resolution (8 jitters) | 0.122 | 0.071 | 0.132 |
| bilinear upscale of 0.75, no jitter | 0.238 | 0.110 | 0.249 |
| TAAU 0.75 (32 frames) | 0.103 | 0.033 | 0.109 |
| TAAU 0.75, the jitter's sign flipped | 0.215 | 0.223 | 0.214 |
| TAAU 0.667, 0.6 | 0.099, 0.100 | 0.037, 0.041 | 0.103, 0.104 |
| TAAU 0.75 after 8 frames | 0.152 | 0.061 | 0.160 |

A still view's frame-to-frame change once converged (the RMS between the last two frames, 32 frames in, k 4.5):
native TAA 0.102, TAAU 0.75 0.127, 0.667 0.108, 0.6 0.087, over the whole plate (not split by band; its far corner is
past even the output's Nyquist, where any resolve flickers). (That table at k 3. At 0.75 with k 4.5: 0.109 / 0.041 / 0.114; k 6: 0.119 / 0.046 / 0.125. Against this box-filtered
reference a smoother kernel always scores better, while in game a sharper one looks closer to native TAA, which keeps
more of the frame's high frequencies (0.071 in band here): k 4.5, the default, keeps most of k 3's margin over native TAA
and measured sharper in game.) The flipped sign doubling the error checks the sample positions; "past it" is mostly past the output's own Nyquist too
(the plate's frequency reaches 1.4 cycles a pixel in the far corner), where every method aliases. Convergence takes the
16 jitter positions (8 frames is half-way). The governor's rules are checked offline too (`tools/govsim.swift`: synthetic
frames, GPU time = fixed + per-pixel x s^2 with noise, reported two frames late, uncapped or vsync with a GPU clock that
stretches its time toward the frame): it settles where the load fits (0.75 for 3 + 7 s^2 ms) without oscillating
uncapped, keeps late frames under 1% wherever the load can fit, and climbs back after heavy views also under vsync with
the stretched clock (the GPU-time rule alone ratcheted down to 0.6 there and stayed).

### The full look per pass, at the panel's resolution and scaled (2026-10-10, lab session 1)

Lab mode, 3456 x 2234 fullscreen, uncapped, `lit,nearchunks,rtshadows,sky,gi,water,coloredlight,post,leaflight,wave`
(no clouds yet), LOD 8192, `trace 8` per row's scale at mountain_view (the heaviest of the nine: a high view over 30 km of
LOD), medians. Other helpers were building and running games in turns, so these are noisy; compute passes' totals overlap
their neighbors and don't add up (`bench_out/agents/speed/traces1/` in the main checkout, `passes.py`).

| Pass | 1.0 | 0.75 | 0.6 |
|---|---|---|---|
| Sky (quarter resolution, then the full-resolution pass) | 0.04 + 0.24 | 0.07 + 0.47 | 0.04 + 0.33 |
| **Main pass: vertex / fragment** | **3.15 / 5.78** | **3.14 / 2.58** | **3.90 / 2.00** |
| Ray-traced shadows | 0.69 | 0.46 | 0.34 |
| GI cache: light, begin, request and resolve | 1.52 | 1.28 | 1.04 |
| GI cache: schedule and rays | 0.47 | 0.45 | 0.51 |
| Water's rays | 0.77 | 0.47 | 0.33 |
| Resolve with the relight, water and aerial perspective (TAA, then TAAU) | 2.92 | 2.32 | 1.85 |
| Post: bloom, exposure, composite | 1.42 + 0.04 + 0.67 | 1.33 + 0.05 + 0.68 | 1.31 + 0.04 + 0.71 |
| The copy into the frame (fragment) | 0.42 | 0.48 | 0.50 |
| Present (fragment) | 0.37 | 0.41 | 0.42 |

- What scales with the pixels does: the main pass's fragment stage -55% at 0.75 (-65% at 0.6), shadows, water, the GI
  cache's per-pixel stage, the resolve's load. The upscaler's own per-output work (reconstruction, history, clip, at the
  panel's 7.7 M pixels) is about 1.1 ms (from the resolve's two rows: 2.92 = L + 0.75 natively, 2.32 = 0.56 L + C).
- **What doesn't**: the main pass's vertex stage (the LOD's quads, 3.1-3.9 ms in this view), post (2.1 ms, at the panel's
  resolution by design), the copy and present (0.9), the upscaler's per-output work (1.1), the GI cache's rays (0.5).
  That's about 8.5 ms: in this view 120 Hz can't be had by scaling alone, at any scale. In game, uncapped, the governor
  (`auto`) went to 0.6 with GI quality 0 and held 88 fps at mountain_view, 93 at noon_overview, 111 at forest, 114 at
  water_closeup.
- Command buffers' GPU spans overlap: 19 ms spans at 12 ms frames (mountain_view at 0.6). A frame's own GPU time is
  end - max(start, the previous frame's end); the governor uses that. The bench's `metal_gpu_ms` is the span and
  overstates per-frame GPU time whenever frames overlap.
- In 2-3 of 8 frames vanilla redraws its animated block atlas (a 2048 x 2048 texture and its four mips, render passes on
  the game tick), and those passes' fragment spans reach 5 ms. That is most likely waiting for the frame before to stop
  reading the atlas (the GPU busy with that frame meanwhile), not work; whether it costs a frame's overlap once a tick
  is for the flights' frame times to show.

**Lab session 2** (the same, with clouds; `bench_out/agents/speed/traces2/`, `boards2/`): mountain_view's main pass
3.24 / 3.79 ms at 1.0 and 3.19 / 2.57 at 0.75; the clouds' march 0.56 -> 0.35 ms (their half-resolution buffers follow
the level: 1728 x 1117 -> 1296 x 838); the upscaler with the clouds' composite in its load 2.62 ms against the plain
resolve's 3.15.

- **The LOD's quad culling at 0.75** (`quadcull`, exact): 70.9% of mountain_view's 1.34 M level-0 quads issued (67.5% in
  the forest), the vertex stage 3.19 -> 2.91 ms, for a culling pass of 0.37 ms in a command buffer of its own (0.21 in
  the forest); status 69 -> 71 fps in mountain_view, 84 -> 84 in the forest. Worth having on with a scale, not the lever
  that makes 120 Hz.
- **The governor** (uncapped, `auto`, with each frame's own GPU time now): agrees with the frame interval (13.9 ms GPU
  at 13.8 ms frames) and goes straight to 0.6 and quality 0 in the heavy views: mountain_view 82 fps, noon_overview 82,
  forest 103 (with clouds).
- **The look, native against 0.75, back to back** (the same pose seconds apart; clouds drift and leaves wave between
  them): sharpness (the scaled shot's gradient energy over native's) 0.70-0.82 with the reconstruction's k 3 and the
  copy's sharpening 0.5 (forest 0.76, mountain_view 0.73, noon 0.80, torch_cave 0.82, water_closeup 0.70). At 100% the
  0.75 frame is softer in fine texture (leaves, distant block faces), with no artifacts of its own: edges are clean,
  water, clouds and the sky hold up. k 4.5 with sharpening 0.7 took the forest to 0.81; sharpening alone (0.8 at k 3)
  didn't move it (0.755). In theory the still image converges to the scene convolved with the reconstruction's Gaussian
  (sigma 0.41 output pixels at k 3), against native TAA's box over each pixel (sigma 0.29): k 6 would match it, at the
  price of a blockier current frame where the history is rejected (disocclusions, waving leaves).
- **A system dialog blocked the timed flights** from at least 12:15: macOS's "“claude” would like to access data from
  other apps" (UserNotificationCenter, Don't Allow / Allow), which no helper may answer. run_mc.sh holds timed runs
  until the screen is clear, since a dialog over the fullscreen game paces it at 120 Hz and voids uncapped numbers. The
  vsync flights below ran anyway (a run paced at 120 Hz by vsync isn't changed by that), flagged in the ledger; the
  uncapped flights wait for the dialog to go. The lab sessions' uncapped fps may have run under it too (when it appeared
  isn't known).
- The mip bias (the terrain's texture gradients scaled to the output's resolution) didn't change the forest's score:
  there the leaves' texels are magnified (2 rendered pixels each), so it's the reconstruction, not the mips. It matters
  for minified textures (distant terrain), where it keeps the native mip.

### Flights (2026-10-10, vsync only, noisy)

The real-terrain flight (`claudeworld-merged`, y 150, 20 blocks/s, LOD 32768, 60 s timed), 3456 x 2234 fullscreen at
120 Hz, TAA on, `lit,nearchunks,rtshadows,sky,gi,water,coloredlight,post,leaflight,wave,clouds` (before god rays and the
grade), from a frozen copy of the branch at af7d06c. Other helpers built and ran in turns between the runs, and the
system dialog above was on screen: **noisy**, one run each. Frames over 8.33 and 16.67 ms are the bench's
`frames_over_8ms` and `frames_over_16ms`; own GPU time is each frame's (above), the span is the command buffers'.

| Run (vsync on) | Frames | fps | ms p50 | ms p99 | Over 8.33 ms | Over 16.67 ms | Own GPU p50 / p99 | Span p50 |
|---|---|---|---|---|---|---|---|---|
| Native (1.0) | 3948 | 66.0 | 15.29 | 17.86 | 3947 | 344 | 15.08 / 17.12 | 24.01 |
| 0.75 | 4901 | 81.9 | 12.39 | 15.01 | 4895 | 9 | 12.03 / 14.52 | 18.06 |
| `auto` (0.6-1.0) | 2669 | 45.0 | 10.70 | 101.57 | 2586 | 362 | 10.46 / 21.00 | 14.33 |

- **0.75 against native**: +24% fps, p99 -2.9 ms, frames over 16.67 ms 344 -> 9. Neither reaches 120 Hz (every frame
  over 8.33 ms): the fixed costs above.
- **`auto` is void**: about 35 s of it ran at about 10 fps (100 ms frames), the window throttled while the dialog had
  focus. Before that the governor stepped 0.8 -> 0.65 -> 0.6 and quality 0 in the warm-up and stayed there: at 120 Hz
  with this look no scale fits 8.33 ms, so a 120 Hz target sends it to its floor. Its own GPU p50 at 0.6 was 10.5 ms.
- **Uncapped (vsync off): not run.** The dialog paces an uncapped fullscreen game at 120 Hz, so run_mc.sh holds those
  runs until the screen is clear.

To re-run on a quiet machine (no other runs, the dialog answered), from the repository's root, one flight each:

```
for s in 1 0.75 auto:0.7:1; do for vs in false true; do
  BENCH_FIXTURE=claudeworld-merged BENCH_TIMEOUT=900 BENCH_VSYNC=$vs bash tools/bench/bench_lod.sh "fl-${s//:/_}-$vs" 32768 \
    -PbenchY=150 -PbenchFly=20 -PbenchExtraWait=600 -PbenchHitches=1 -PbenchTrace=1 -Ptaa=true \
    -PmetalExp=lit,nearchunks,rtshadows,sky,gi,water,coloredlight,post,leaflight,wave,clouds,godrays -PrenderScale=$s
done; done
```

Then read `fps_mean`, `ms_p50`, `ms_p99`, `frames_over_8ms`, `frames_over_16ms` and `metal_gpu_own_ms_p50`/`p99` from
the `METALMC_BENCH` line of each `bench_out/run_fl-*.log`, and the governor's `render scale: now ...` lines for `auto`.
Three runs of each, alternating, would make them more than one sample.

## 2026-10-13: the flicker fix's cost, and the water pass

The resolve (anti-aliasing + relight + aerial perspective; the water pass counts in it, same encoder), lab trace of 10
frames, the Claude edition's switches, native resolution:

| Build | forest | water_closeup |
|---|---|---|
| main 34ac2e6 (before) | 5.95 ms | 3.51 ms |
| flicker fix, first cut (visibility read three times a pixel) | 8.44 | 6.43 |
| visibility read once (`visPx`) | 7.27 | |
| water in its own pass (`taa_water`) | **5.51** | **4.56** |

Profiled live in the forest (0.2% water), each change undone in the shader files: the depth-aware visibility upsample
0.45 ms, refraction's code 0.23 (register pressure: no pixel ran it), the layered waves 0.20, the tone-weighted clip 0.06.
Water's code in the resolve's load cost every pixel occupancy; in its own pass (texture 30, p.range.y) the forest's
resolve is under the old build's. water_closeup's +1.05 is the new water: refraction (0.8), the seabed's shadow ray,
five wave layers, layered caustics, the reflection's history. Whole frame (median command buffer span): 28.8-29.1 ms,
the old build 27.7-29.1.
