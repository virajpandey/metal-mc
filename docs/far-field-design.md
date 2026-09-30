# Far field: a height-field ray march past the voxel LOD (prototype working, off by default)

## Why

Measurements on real terrain (native 3456×2234, flying at y 150; see lod-design.md):

- The LOD's cost is its vertex invocations, four per quad, at about 2.6 billion a second on the M3 Pro. Smaller vertex
  outputs, mesh shaders, and culling quads after they're issued (collapsing them in the vertex shader or culling them in
  the mesh shader) all saved nothing. Only quads that are never issued are saved.
- 85-90% of the issued quads own no pixel: about 80% at level 0 (192-768 blocks) and 95% at level 1 (768 m-2.3 km).
  They're hidden at a fine grain, since terrain seen at grazing angles stacks many voxel tops and steps into each row of
  pixels. Coarse culling can't reach that (the sub-tile horizon test removed 11%).
- Cube voxels stop working past about 32 km: a level-8 voxel is 256 blocks tall, a level-9 one 512, more than the world's
  height. The LOD can't reach the horizon, where land 262 km away is still a pixel above it for a camera 140 blocks up.

A ray march over a height field costs per pixel instead of per quad, has exact visibility (each pixel finds the first
column its ray enters), keeps full vertical precision at any distance, and reaches the horizon for the price of one
more coarse ring per doubling of distance. At that distance terrain really is a height field: a pixel covers dozens of
blocks, and overhangs, caves and tree trunks under canopies are invisible.

## Shape

- **Near: the voxel LOD** (level 0, maybe level 1) as today, drawn as quads.
- **Far: rings of a height field around the camera** (clipmaps). Ring r holds W x W columns of 2^(r+1) blocks: height
  of the top solid voxel (half precision or 16-bit blocks), top material and side material (for the texture/color
  tables the LOD already has), water surface and floor where there's water. Addressed toroidally, so moving the camera
  rewrites only the columns that scrolled in. Filled from the LOD's own grids for explored terrain (the mesher can emit a
  node's columns as it meshes it) and from the seed-sampled far terrain past that (FarTerrain already samples heights and
  biomes; for rings past 32 km it samples coarser, with vertical precision kept, since a ring's cell is a horizontal size
  only). Rings to about 256 km: 8 rings of 512², a few MB.
- **A max pyramid per ring** (each level the maximum height of 2 x 2 cells below), so rays skip empty space:
  the standard maximum-mipmap height-field traversal, typically tens of steps per ray.
- **The pass:** a full-screen draw inside the main pass right after the LOD's quads, at the far plane with depth test, so
  only pixels nothing nearer covered run it (early depth test), writing color and depth (a conservative depth output:
  the hit is always nearer than the far plane). Rays start where the quad LOD's coverage ends. Each hit shades like the LOD:
  material color and texture detail from the block coordinate at the hit, the face (top or side of the column it hit),
  vanilla's lightmap, fog, water blended over its floor.
- **Shadows** from the same height field (a second march toward the sun) past the ray-traced range.

## Expected

- Levels 1+ as quads today: about 0.5-0.6 M of the 1.8 M quads in view, roughly 0.8 ms of vertex work. The march over
  the band they covered: on the order of 2 M pixels x tens of steps, estimated 0.3-0.6 ms (to be measured).
- The horizon: land out to where it's under a pixel above the horizon line.
- Later, moving the quad/march boundary nearer (to 400-500 blocks, where trees and overhangs begin to be too small to
  matter) would take most of level 0's hidden quads with it; its fidelity has to be measured against vanilla first.

## Steps

1. Emit per-node column data (height, top/side material, water) from the mesher; assemble one ring from level-1 nodes;
   a debug view that draws the ring's heights.
2. The march for that ring alone, flat-colored, compared against the level-1 quads it replaces (fidelity score and GPU time).
3. Shading parity (texture detail, lightmap, water, fog, AO from neighbor heights), then more rings, then rings past the
   LOD from seed sampling.
4. Turn off level 1+ quads where the march covers them; measure the frame; then the horizon.

## Prototype (2026-09-30, `METALMC_FARFIELD=<level>`, `-PfarField`, `Sources/MetalMCNative/FarField.swift`)

Steps 1-2 and most of 3's shading, for LOD levels from `<level>` up (1 in the measurements below).

- **Columns.** Every node at those levels keeps one 32-bit word per column (LodBuild.farColumns): the top solid voxel's
  height, voxels of water above it, the top material, and the water's material (or the material under the top). That's
  256 KB per node, filled on the GPU from the node's buffer.
- **Rings.** Per level, a 2048 x 2048 window of that level's columns around the camera (one slice of a 2D array
  texture; `METALMC_FFWIDTH`) with a max pyramid of column tops in blocks. Each level's ring reaches 1024 of its cells
  from the camera, about as far as the quads use that level; 1024-cell rings switched to the next level at half that
  distance, and against the answer key below scored 5.87 instead of 4.87. A ring is refilled (clear, one dispatch per
  node, mip passes) when the camera moves 64 of its cells or the columns of a node in its window change, at most one
  ring per frame, from a separate command buffer committed ahead of the frame's. Nodes are rebuilt whenever their
  region is saved (every few seconds while flying over fresh chunks), nearly always with the same columns, so rings
  are keyed on a hash of each node's columns rather than the node object (which refilled a ring about once a second).
- **The quads' area.** Rays must not find terrain the quads draw (levels below the far field's): a 64-block bitmap of
  the chosen quad tiles is rebuilt every frame (cheap on the CPU) and tested where a ray reaches a column; covered
  columns count as empty. Baking it into the rings instead forced a full refill whenever a tile near the player changed
  (every frame or two in flight: p99 15 ms).
- **The shell.** The march is drawn as a 64-sided prism around the camera at the horizontal distance of the nearest
  far-field tile, from the world bottom to the highest terrain in the rings (or the camera), plus its floor. Every hit
  lies outside it, so rays start there, the early depth test skips every pixel with anything drawn nearer (the
  fragment writes depth with `[[depth(less)]]`), and rays that leave through its top pass over all the terrain and never
  start. A full-screen triangle at the depth of the nearest possible hit instead ran the march on every pixel with LOD
  terrain beyond that plane: 120 fps instead of 168.
- **The march.** Per ring, from the pyramid's top (rays over everything leave a ring in one step), the standard
  max-mipmap traversal: skip a cell if the ray stays above its maximum, else descend; at a column, a side hit where the
  ray enters it below its top, else a top hit. Then the next ring from where the ray leaves this one's window.
- **Shading like the LOD's:** face shade, vanilla's lightmap, texture detail from a mip picked for the pixel's
  footprint (no smooth derivatives at column edges), water over its depth-lit floor at the LOD's water alpha, the
  water surface 10/9 block below the voxel top, fog, and ambient occlusion like the LOD's per-voxel-corner steps: a top
  face's corners count taller neighbors, a side face darkens toward its foot (answer key 4.87 -> 4.47). A side hit below a water column's surface is terrain seen through
  water (rays that pass under the quads' water reach the first far-field column that way), not another water surface:
  shading it as water drew a bright strip where the quads end (far band error 4.92 -> 4.65).

Results (Viraj's world, native 3456×2234, flying at y 150, TAA on, uncapped; `-PmetalExp=ffsteps` colors hits by steps):

| | fps | p99 | main pass | vertex | fragment | LOD quads/frame |
|---|---|---|---|---|---|---|
| quads (ffpD) | 158.7 | 7.78 ms | 4.14 ms | 2.70 | 1.44 | 1.79 M |
| far field from level 1 (ffpE) | 168.4 | 7.72 ms | 3.68 ms | 1.36 | 2.31 | 0.88 M |

With 2048-cell rings and AO (commit 0e68840), under heavy background CPU (Microsoft Defender scanning at 30-90%):
163.5 fps vs 159.9 with quads, main pass 3.86 vs 4.10 ms, p99 8.70 vs 8.14 ms; at 120 Hz with vsync both hold 119.8 fps
and drop 14 vs 10 frames in the 60 s flight (the same flight dropped 11 a minute with quads on a quiet night earlier).
The far field is faster on average but not yet a clear win; its payoffs are the horizon and level 0's hidden quads.

Fidelity in the band past vanilla's 512 blocks (fog off, 4 km world). The answer key (`nL0`) is the quad LOD with
level 0 (every block) out to 2 km (`-Plod0=2048`: 40 M quads, 158 s to build); past 2 km it's the same level-1 quads as
the quad run, which flatters the quads there.

| | vs answer key | vs quads |
|---|---|---|
| quads (nC35) | 3.12 | – |
| far field, 1024-cell rings (ff1d) | 5.87 | 4.55 |
| 2048-cell rings (ff1e) | 4.87 | 2.97 |
| + ambient occlusion (ff1f) | 4.47 | 2.46 |

The near band against vanilla is unchanged (4.62-4.64). Past the rings' finest data a height field can't show
overhangs or the gaps under canopies (a tree is a column down to the ground).

**The horizon (commit c8847c0).** Column heights are stored in blocks, not voxels; generated columns get far-field words
built from the generator's exact height, water and materials (LodGrid.farExact: a level-8 grid is one voxel tall, with
no room for water); and with the far field on, the top level is generated out to the LOD distance instead of one node
size. At LOD distance 262144 the fidelity tour's high views show terrain to the horizon line (116 generated nodes, 7.6 M
columns, 82 s at 33 us per column per thread, cached after).

Known artifacts (Viraj, 2026-09-30):
- Spires where real terrain meets generated terrain: real columns at coarse levels come from voxels rounded up to 8-32
  blocks, generated ones are exact. Fix: carry each column's true top through the downsampling.
- Trees as pillars: one height per column can't hold a crown over air. Fix: a canopy layer (ground height plus canopy top
  and bottom).
- Thin tall things (a tower) become full-height pillars; each ring's cells are twice the previous ring's, so similar
  objects change thickness across a ring edge. Better: pick detail by on-screen error, not distance alone.
- Not the far field's: pale "curtains" under the sea along LOD node borders (the quad LOD's underwater skirts, drawn as
  opaque floor stand-ins, seen through the translucent water); both renderers show them.

Next: an absolute reference for that band (level 0 out to 2 km as the answer key), the horizon (rings past the LOD
from the world generator, with heights in blocks instead of voxels so coarse rings keep full vertical precision),
ambient occlusion at column feet, and the quad/march boundary moved inward (level 0's hidden quads).
