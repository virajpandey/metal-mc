# Lighting design (Metal ray tracing)

The plan (decided 2026-09-29): our own lighting on the M3's hardware ray tracing, aimed at a locked 120 Hz and lighting the
whole LOD out to the horizon, with SEUS's look as the reference. No shader-pack code is used; SEUS stills will serve as an
answer key, the way vanilla at render distance 32 does for the LOD. Milestone 1 is sky and atmosphere, a tone curve
designed for the XDR panel's HDR, and ray-traced sun shadows. This file covers what exists so far.

## Ray-traced sun shadows (prototype, `METALMC_EXP=rtshadows`)

`Sources/MetalMCNative/RtShadows.swift`, hooked from `GameRendererLodMixin` right before the temporal anti-aliasing.

- **What casts shadows: the LOD's own geometry.** Level 0 of the LOD is the blocks around the player, so one set of
  acceleration structures covers the near terrain vanilla draws and everything out to the LOD's edge. Each drawn tile
  gets a primitive structure (its opaque quads as triangles, in node-local blocks), built in the background as tiles
  appear, nearest first, eight at a time, then compacted (about half the memory). Per tile, not per node: a node draws
  only the tiles its finer children don't, and a node-wide structure would cast shadows from the coarse version of
  terrain drawn in full detail beside it. Water doesn't cast shadows.
- **Instances.** An instance structure over the drawn tiles, relative to a fixed origin that moves when the camera
  gets 1 km from it. It's rebuilt only when the set of tile structures changes, which in a 20 blocks/s flight is 0-35
  times in 240 frames. The instance list itself is cached until the drawn tiles change.
- **Tracing.** A compute pass after the level is drawn, at half resolution (one ray per 2 x 2 pixels). Each sample
  rebuilds its camera-relative position from the depth buffer and its normal from the neighboring depths (on each axis
  the neighbor on the same surface), snapped to the nearest axis since nearly everything is axis-aligned. Faces clearly
  turned away from the sun (N·L below -0.05) are in shadow; the rest cast one ray toward a point on the sun's disk
  (radius about 0.6°, a different point each frame so the anti-aliasing softens the edges), from just off the surface,
  up to 4 km. The sample stores the final shade factor: 1 in sunlight, 1 - strength in shadow, fading out between 6 and
  20 km where the haze takes over.
- **Applying.** The anti-aliasing multiplies each pixel's color by the shade of its 2 x 2 block as it loads the color,
  which it does anyway (a separate full-screen pass cost more; it remains for runs without anti-aliasing).
- **Strength and sun.** Vanilla's sun direction is (-sin a, cos a, 0) for its sun angle a (the sky renderer rotates -90°
  about y, then by a about x, from straight up). Strength 0.42 while the sun is more than about 6° up, fading to none as
  it sets, weaker in rain; overworld only.
- **Fixed on the way.**
  - The LOD's water didn't write depth, so at far water pixels the depth buffer held the sea floor and whole water
    pixels were darkened by the floor's shadows (dark bands across the sea). With shadows on, water now writes depth,
    and the LOD's occlusion boxes are tested before the water so its surfaces don't hide the floors under them.
  - Vanilla's clouds write depth; their undersides face away from the sun and turned dark gray. Pixels inside the
    cloud layer (`levelRenderState.cloudHeight`, 4 blocks thick) get no shadow.
  - Treating every face edge-on to the sun as shadowed darkened all vertical faces at noon on top of vanilla's face
    shading. Only faces clearly turned away count now; edge-on faces get a ray and darken only where something is in
    the way (under trees, overhangs).
- **Cost** (native 3456×2234, TAA on, the real-terrain flight, uncapped): 159.6 → 137.4 fps, about 1.0 ms per frame, of
  which about 0.7 ms is the rays. Mean frame time stays inside 8.33 ms, but p99 goes from 8.1 to 9.6 ms, so it stays
  off by default until the frame has more room (see the LOD vertex cost in lod-design.md).
- **Not yet:** entities and particles don't cast shadows (they're not in the acceleration structures); water surfaces
  don't show shadows' effect on what's under them; no moonlight.
