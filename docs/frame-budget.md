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
- **Fewer pixels shaded, reconstructed temporally.** Every per-pixel cost here scales with the panel's 7.7 M pixels, and
  at 254 pixels per inch the panel resolves finer than the eye does at a normal viewing distance (a pixel is about 0.7
  arcminutes at 50 cm). Rendering the level at 0.75 scale (56% of the pixels) and reconstructing native resolution in
  the anti-aliasing resolve would save about 2 ms of the lit frame. MetalFX's temporal scaler cost 9 ms here (Taa.swift),
  so it would be our own resolve. It changes the image (softer in motion), so it's a fidelity trade to measure, not a
  given; and the hand and the HUD must stay at native resolution, which means splitting vanilla's main target.
