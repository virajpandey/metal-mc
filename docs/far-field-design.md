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
built from the generator's exact height, water and materials (LodGrid.farExact, now LodGrid.far's cells: a level-8 grid
is one voxel tall, with no room for water); and with the far field on, the top level is generated out to the LOD distance instead of one node
size. At LOD distance 262144 the fidelity tour's high views show terrain to the horizon line (116 generated nodes, 7.6 M
columns, 82 s at 33 us per column per thread, cached after).

Known artifacts (Viraj, 2026-09-30):
- Spires where real terrain meets generated terrain: real columns at coarse levels come from voxels rounded up to 8-32
  blocks, generated ones are exact. Fixed offline by columns v2 (below); needs an in-game look.
- Trees as pillars: one height per column can't hold a crown over air. Fixed offline by the canopy layer (below).
- Thin tall things (a tower) become full-height pillars; each ring's cells are twice the previous ring's, so similar
  objects change thickness across a ring edge. Better: pick detail by on-screen error, not distance alone. (Mean
  heights shrink a lone tower to its share of the cell instead.)
- Not the far field's: pale "curtains" under the sea along LOD node borders (the quad LOD's underwater skirts, drawn as
  opaque floor stand-ins, seen through the translucent water); both renderers show them.

## Columns v2: true heights and a canopy layer (2026-09-30, offline; in-game checks pending)

Two words per column instead of one (LodBuild.farColumns, LodFarCell.words; the rings are RG32Uint):

- **Word 0, the ground**, as before: top in blocks above the world bottom, water depth, top material, water or side
  material. **Word 1, the canopy:** a slab of leaves from its underside to its top (blocks above the world bottom) and
  its material; 0 for none. The pyramid holds the higher of the ground (or water) and the canopy.
- **The march** tests a column as the ground box plus the slab: a ray enters through a side (below the ground's top, or
  into the slab), meets the slab's top or the ground's top on its way down, or the slab's underside on its way up; between
  the ground and the slab it passes under the trees to the next column. Ground under a canopy gets 3 sky-light levels
  less (vanilla's leaves dim sky light a level a block); the slab's underside is lit the same way, its sides darken
  toward the underside, its top gets corner occlusion from taller crowns around it.

**True heights (spires).** Every column carries a LodFarCell at block precision from the region file up: the dry
ground's top, the water surface and the bed under it, the canopy's top and underside (the lowest leaves: a trunk under
them is too thin to see; snow on leaves belongs to the tree), the shares of the cell with data, under water and under
trees, and the materials. regionGrid builds one per 2 x 2 blocks from the chunk's blocks; LodGrid.downsample merges 2 x 2
at every level; the region's level-2 quadrant keeps them (LZFSE, about 50 KB a region; quadrant cache version 6); live
chunks drop the file's cells for their columns, which then come from the voxels. Generated columns build theirs from the
generator's heights (LodFarStore.generatedCell), canopy included (a crown depth per biome: 3-10 blocks of leaves).

Which statistic: each height is the mean over the part of the cell it describes, which composes exactly from level to
level. Measured offline (`tools/fartest.py stats`, 100 regions of claudeworld-merged): per level, 16 cameras at world y
150 and 260 around the regions, each at the distance the far field draws that level (1024 x 2^(L-1) to 1024 x 2^L
blocks), every pixel-wide azimuth (70 degree field of view on 2234 px) sampled per block against level-1 truth:

| level | voxel tops: silhouette px (signed) / >5% depth-off px / top - truth | cells, mean, canopy dithered | cells, mean, canopy wherever any | cells, max, canopy wherever any |
|---|---|---|---|---|
| 2 | 1.20 (+1.08) / 3.7% / +1.46 | 0.50 (-0.32) / 1.5% / -0.28 | 0.43 (-0.08) / 1.3% / +0.37 | 0.72 (+0.72) / 2.1% / +1.29 |
| 3 | 1.49 (+1.36) / 6.4% / +3.51 | 0.70 (-0.56) / 2.9% / -0.26 | 0.56 (-0.20) / 2.3% / +0.93 | 0.79 (+0.79) / 3.2% / +2.80 |
| 4 | 1.80 (+1.68) / 8.2% / +7.45 | 0.75 (-0.67) / 3.6% / -0.22 | 0.55 (-0.26) / 2.8% / +1.64 | 0.79 (+0.79) / 3.5% / +4.96 |
| 5 | 1.99 (+1.96) / 7.7% / +15.50 | 0.65 (-0.60) / 2.9% / -0.15 | 0.47 (-0.29) / 2.2% / +2.57 | 0.70 (+0.70) / 3.1% / +8.21 |
| 6 | 2.10 (+2.08) / 5.5% / +33.31 | 0.55 (-0.53) / 1.6% / +0.13 | 0.41 (-0.30) / 1.0% / +3.92 | 0.55 (+0.55) / 1.5% / +13.18 |

The maximum keeps peaks on the silhouette but stands everything up (+1 to +13 blocks: it's what made the voxel tops
spires). A canopy wherever a cell has any trees gives the best silhouettes but paints the land as forest: 27% of the
cells at level 2, 63% at level 6, against 16% under trees in fact. So the default is the mean, and a partial canopy
(cover below 100%) is drawn in a share of cells equal to its cover, picked by a hash: 16.1-16.5% at every level, heights
unbiased (within 0.3 blocks), silhouettes 0.5-0.75 px (a third of the voxel tops'), a half to a third of their pixels
more than 5% off in depth. It's also what the generator does for sparse woods, so real and generated forests match.

Against the generator (`tools/fartest.py seam`, the cached generated nodes of claudeworld at the same places as real
columns, and the real/generated boundary cells):

| level | real - generated top, before: voxel tops (mean, p95) | after: cells (mean, p95) | boundary step: before / after / the generated terrain's own |
|---|---|---|---|
| 3 | +3.65, 15 | -0.75, 9 | 4.82 / 2.53 / 1.80 |
| 4 | +8.29, 25 | -0.51, 9 | 8.89 / 3.19 / 2.84 |
| 5 | +17.22, 42 | -0.59, 10 | 16.63 / 4.45 / 4.24 |
| 6 | +37.56, 71 | +1.85, 15 | 35.16 / 7.74 / 6.46 |

The seam now steps about as much as the terrain does anywhere else. The generated heights themselves read low at coarse
levels: FarTerrain stops its search for the ground within a quarter voxel (2-64 blocks at levels 3-8) and returns the
bottom of that interval (the real ground was 0.8, 1.7, 3.1 and 6.7 blocks above them at levels 3-6), so generated cells
take the interval's middle (after: +0.33, +0.23, -0.24, +0.40).

**A stepping bug (pre-existing).** The march recomputed its cell from the ray a small step past each boundary. For rays
nearly along x or z crossing a boundary toward -x or -z, that step was under a float's precision, so floor() put the ray
back in the cell it had just left; it spent its 192 steps there and skipped the rest of its ring, and a wedge of the next
ring's terrain (cliff sides, brown and black) was drawn near the camera whenever the view looked close to -x or -z. Now
the ray's position is carried separately and put half a cell past each boundary it crosses
(`bench_out/results/far-field-v2/ (local, not in git) steps-l3-epsstep-0.png` before, `steps-l3-fixed-0.png` after). It may be some of the brown walls
seen on the horizon.

Offline renders (`tools/fartest.py render`: the world built by LodWorld from the fixture and the cached generated nodes,
the march drawn by FarField into a PNG, every column by the march, no texture detail or fog), in `bench_out/results/far-field-v2/ (local, not in git) `:
`seam-l5-{old,new}-{0,1}` (far field from level 5, the real terrain's east and south edges from 1 km out: the old
voxel tops stand as 32-block plateaus with cliff sides above the generated terrain; the cells meet it) and
`forest-l1-{old,new}-0` (level 1 over a spruce forest: the ground and water show under and between crowns). Mean march
steps per hit pixel in forest views: 30.4 and 34.9 with the canopy, 29.2 and 33.8 without, 27.9 and 33.1 before these
changes. The first pass over the fixture's 100 regions took as long with cells as without (run to run noise), and the
in-memory quadrant cache grew from 36 to 41 MB.

Switches (METALMC_EXP, `-PmetalExp`): `ffvoxeltops` (real columns' heights from the voxels as before), `ffpillars`
(canopy drawn as columns down to the ground as before), `ffepsstep` (the old stepping). All three together are the first
far field, but for generated columns' crown variation and search-interval middle.

Costs: the rings take twice the memory (RG32: 32 MB per 2048-cell ring, 256 MB for 8 rings), node column buffers 512 KB
instead of 256 KB, and every region's quadrant is rebuilt once (cache version 6).

In-game checks still to run: the fidelity tour and the flight bench, new against all three switches; a look at the
horizon from high up toward the real terrain's edges (spires gone, forests as crowns) and along -x and -z (no wedge);
that rays passing under forest canopies at grazing angles don't run out of steps (holes); memory with 8 rings.

Next: an absolute reference for that band (level 0 out to 2 km as the answer key), the horizon (rings past the LOD
from the world generator, with heights in blocks instead of voxels so coarse rings keep full vertical precision),
ambient occlusion at column feet, and the quad/march boundary moved inward (level 0's hidden quads).

## Night of 2026-10-01: in game at the panel's resolution, towers, rings updated in place

**The answer key again.** The fidelity runs from 2026-09-30 evening on were partly at the wrong resolution: a crash during
startup (HDR's, since fixed) makes Minecraft's next start windowed and drop its fullscreen mode, and without a mode the
exclusive fullscreen takes the desktop's ("More Space" on this Mac: 4112 x 2658, scaled down by the system).
bench_lod.sh now sets the mode (1728 x 1117 @ 120, 3456 x 2234 pixels) and startedCleanly before every run. The answer
key was rendered again with the current build (`nL0c`, level 0 out to 2 km). Fidelity tour, far band, against it
(lower is better):

| | far band | near band (vs vanilla) |
|---|---|---|
| quads (`m3Q`) | 3.03 | 4.58 |
| far field (`m3F`) | 4.26 | 4.60 |
| smart LOD (`m3S`) | 3.15 | 4.58 |
| near chunks (`m3N`) | 3.02 | 4.58 |

The far field is brighter than the answer key by about 2.7 levels (RGB +2.8 +2.7 +2.2); by surface (`biasclass`): light
gray +22, dark gray +14 (6.6% of the band, half the bias), dirt +3.6, leaves +1.6 (the quads: +3.2, +2.4, +2.0, +0.5).
The worst views look north over the ice spikes: the answer key shows the spikes (light blue-gray, standing over the
snow), the far field's level-1 cells average them into the snow under them (white).

**Towers (columns v3, `LodFarCell.addTower`).** A cell's dry ground can be two heights: a base over the cell and a high
part standing in a rectangle of it (a spire, a peak, an ice spike, the top of a cliff the cell straddles), instead of
one mean over the whole cell. At every merge from the blocks up, the children's ground is taken as parts (each child's
base over its quadrant, its high part in its rectangle); the parts split in two by height where that explains the most
of their spread (Otsu's split), and the high side is the tower if it stands at least a quarter of the cell (and 2
blocks) over the rest and its rectangle is at most three quarters of the cell. High parts far apart (a rectangle more
than twice their share plus an eighth) would stand the whole cell up, so then only the most prominent one is the tower.
The rectangle is kept in eighths of the cell; `ground` stays the mean, so everything else reads the cell as before. The
level-2 quadrant cache keeps the towers (24 bytes a cell instead of 18; cache version 7). Word 1 with bit 31 set is a
tower (its top, rectangle, top block) instead of a canopy (trees covering at least half the cell keep the canopy); the
march tests it as a box from the bottom to its top beside the base that word 0 holds, and shades its sides with its own
block where that differs from the base's (a stone peak on grass). `METALMC_EXP=ffnotowers` turns them off.

Offline (`tools/fartest.py render`, the view over the spires at 2.3 km, which ring 1 draws at level 2): 1.1% of the
pixels change, all on rough terrain (cliff faces near the camera, the ice spikes); zoomed in (`FARTEST_FOV=12`) the
spires keep their shape and get narrower where they're narrower. The spires themselves are real: from 600 blocks
closer, at level 1, the same gray columns stand there.

**Rings updated in place.** A ring is now toroidal: cell (x, z) of its level is texel (x mod W, z mod W), and the march
works in ring coordinates (the world's cells less a multiple of W, so they stay small). A window that moves 64 cells
clears and writes the 64-cell strip it moved onto instead of all W x W cells, a node that changes writes its own cells,
a node that goes clears its cells. Offline, along a five-view route: the first view matches a full fill but for 106 of
1.9 M pixels (coarse pyramid cells at the window's edge now hold neighbors from across the torus: more steps, never a
different hit), a view after a round trip and one reached from two directions are identical; a move writes 262 K cells
instead of 8.4 M. `METALMC_EXP=fflog` logs every update with its time, to line them up with long frames.

In game (the same night, native resolution; flights on the real-terrain world, 60 s, uncapped unless noted):

| | far band | fps | p99 | frames over 8.33 ms | dropped at 120 Hz (vsync) |
|---|---|---|---|---|---|
| quads (`m3`/`m4`) | 3.03 | 158.5 / 158.6 | 7.80 / 7.78 ms | 20 / 20 | 6 |
| far field, rings refilled whole (`m3F`, `m3ff`) | 4.26 | 155.1 | 9.94 ms | 453 | 4 |
| rings updated in place, no towers (`m4Fn`, `m4ffn`) | 4.25 | 153.9 | 8.49 ms | 118 | |
| rings in place and towers (`m4F`, `m4ff`) | 4.22 | 154.3 | 8.53 ms | 134 | 6 |

Updating rings in place cut the long frames by 3.4x. What's left isn't the updates: 47 of them in the flight's 59 s,
and 10 of the 134 long frames within 50 ms of one (`METALMC_EXP=fflog`). The far field's frames are slower in the tail
on the GPU (p95 of the frame's GPU time 12.56 against the quads' 11.42 ms), from the march in heavy views; its LOD CPU
time is lower (0.41 against 0.51 ms). At a locked 120 Hz the two drop about as many frames. Towers cost nothing
measurable and gain a little in the far band, more where it doesn't reach (cliffs, peaks and ice spikes past 2 km).

**On by default (2026-10-01, end of the night).** Two shading fixes against the level-0 answer key, then the switch:

- A side face's foot now gets two occluders' ambient occlusion (0.76), as vanilla's smooth lighting and the quads'
  voxels give a step's bottom corners (the ground in front and the corner beside it), not one (0.88): far band 4.24 →
  3.80, the bias +2.9 → +2.1 RGB (`m6A`; `METALMC_EXP=ffsideao1` for one).
- What's left is mostly the occlusion of detail finer than a cell: the cells' mean heights smooth away the steps the
  quads' voxels keep, and with them corners that darken. A land gain (`METALMC_FFGAIN`, applied as occlusion so lit mode
  relights the same color) fitted at 0.93: far band 3.32 with the bias at +0.6 RGB, the quads' +0.8 (`m6A96` 3.43,
  `m6A93` 3.32; without the side-foot fix 0.92 gave 3.39).
- Against the quads (3.05) that's 9% more error in the far band, all of it in the views from 150 blocks up and higher
  (mid-north 6.07 vs 5.38, high-north 4.67 vs 4.02, mostly snowy terrain whose step sides and spruces the 2-block cells
  thin out); from the ground the far field is closer to the answer key (east 0.46 vs 0.81, south 1.04 vs 1.61).
- At a locked 120 Hz the two drop as many frames (4-6 in 60 s), the far field's CPU time is lower, and it's the only
  one of the two that reaches the horizon: so it's on by default. `METALMC_FARFIELD=0` (`-PfarField=0`) draws quads at
  every level as before; `tools/bench/verify_all.sh` asks for that explicitly for its quads runs.

Verified with the new defaults (`verify_all.sh m9`, after the projection's far plane was pushed past the LOD, so the
quads past 2 km are drawn instead of clipped): fidelity near band 4.58 (quads) and 4.60 (far field) against vanilla,
far band 3.28 and 3.32 against the older answer key `nL0` (3.03 and 3.34 against `nL0c`); the real-terrain flight
uncapped 109.8 fps (quads, 3.55 M quads a frame) against 153.1 (far field, 0.87 M; p99 8.39 ms, 115 frames over
8.33 ms), and at 120 Hz 110.8 fps (the quads can't hold it) against 119.8 with 5 dropped frames in 60 s. The quads'
earlier numbers (158 fps) came from drawing nothing past 2 km.

**The march, cheaper (2026-10-01 morning).** Offline (`tools/fartest.py render` with `FARTEST_REPEAT=12`,
`FARTEST_SHELL=512`, the panel's resolution, the flight's world) the march over the flight's view (y 150, 10° down)
cost 4.0 ms of GPU time with no near geometry in front of it, 29.4 steps per hit pixel. Two changes:

- Each ring's march starts at pyramid level 6 instead of the top (11): from the top every ray that meets terrain first
  descends 11 levels, while a ray over the terrain climbs a level a step anyway. Flight view 4.02 → 3.51 ms, from 260
  blocks up 6.04 → 5.50, near the ground 2.94 → 2.38 (level 4: 3.35, 5.39, 2.86); 6 of 7.7 M pixels differ (rays at the
  step cap). `METALMC_FFSTART` sets it.
- A top face's corner occlusion read 12 neighbors for 8 distinct ones; now each once (the image is identical).

## Where the march's time goes, and a cheaper march (2026-10-02, offline)

New tools first:

- `tools/fartest.py serve` loads the world once and runs commands from a directory (JSON): views × shader variants
  (`fartest.py shader`'s output, edited; `fartest.py compile` checks they compile, with and without the counters),
  rounds interleaved; each variant's median GPU time, and how its image differs from the first variant's (pixels, by how
  much, where, and a picture marking them; `mmc_debug_far_compare`). It holds the GPU lock (tools/bench/gpulock.inc, as
  gpuwait.sh takes it) for the world's load and while commands run, not between.
- Counters (a variant run with stats, `FF_STATS`): per pixel the steps by kind (advances over level 0, descents,
  level-0 column tests and their misses, climbs undone by the next step), the rings, the steps in each ring, the hit's
  ring and step; logged as means and as what 8 × 4 pixel groups (a SIMD group) pay, their slowest pixel, by eighths of
  the image. The shader runs with early fragment tests then (it writes a buffer, which would otherwise run it on pixels
  the depth test skips).
- **The game's nearer geometry, stood in for** (`FARTEST_NEAR`, `near` in a command). With nothing in front of the
  march, as the numbers above were taken, rays toward terrain inside the shell start under it and "hit" at once (7
  steps each): half the flight view's pixels and a third of its steps, work the game never does (its quads and chunks
  cover those pixels and the early depth test skips them). A first draw from 16 blocks out (with the built-in shaders)
  fills the depth and the timed draws test against it, so only pixels whose terrain lies past the shell run the march.
  "Near mode" below.

Where the time went (near mode, the march as it was; ms of GPU, median of 11 draws, 3 rounds; the parts from variants of
its shader):

| | flight (y 150, 10° down) | 260 up (12° down) | near the ground (y 80, 3° down) |
|---|---|---|---|
| the march | 2.50-2.59 | 5.00-5.29 | 1.15-1.22 |
| the shell rasterized, every ray discarded | 0.31 | 0.33 | 0.29 |
| shading without the march (each ray "hits" its first column) | 0.41 | 0.39 | 0.40 |
| the traversal alone (the first level-0 dip is the hit, no shading) | 1.97 | 3.32 | 0.88 |
| + the column tests, no shading | 2.30 | 4.59 | 1.09 |

So the traversal is two thirds of the time, the column tests (with the steps after their misses) a fifth to a quarter,
the shading under a tenth. Steps per 8 × 4 group 19.7, 39.3 and 8.5; per hit pixel 36.5, 35.6 and 41.0. By eighths of
the image from the top, the flight view's steps were 22% in the eighth just over the horizon (sky: rays that pass over
everything, 13.8 steps a group) and 71% in the one under it (the far band's terrain, 48 a group); from 260 up 23/46/25%
in the three around the horizon. After the changes below: 13% and 84% (5.2 and 37 a group). A pixel's own steps barely matter: a group pays its slowest pixel, about a third more.

What a step costs (probes, 260 up): one more texture read per step +12% (of the same texel, so cached: +6%), the same
read on the step's critical path +24%, 24 more ALU ops +8.5% (+10% on the critical path), the traversal run twice per
pixel (the second finds every texel cached) 2.03-2.1x. So it's neither memory- nor latency-bound: the steps' issue
(instructions, and the branches a SIMD group runs for all its lanes) is the cost; the levers are fewer instructions per
step and fewer steps in the groups that pay most.

What changed (exact: the image is the same but for rays at the step cap, which now get further):

- **A cheaper step.** The ray's level-0 cell is kept as integers inside the window (its cell at level l is it shifted
  l bits), set once per move, instead of floors, divides and the window's clamp at the step's level every step; a
  crossing sets the crossed axis's cell to the one past the boundary and the other from the ray (as the half-cell nudge
  did, so rays along x or z still don't fall back into the cell they left). 260 up: 5.33 → 4.96 ms at the same steps.
- **Up a level only into a parent not tested from here.** After a step at level l the march went up to l + 1; a ray
  going down that's still in the same parent leaves it as low as before and fails its test again (a step to fail, one
  back down). It goes up only when the boundary it crosses is the parent's too (an even one at level l), or when the
  ray rises. Climbs undone by the next step: 3.9 → 1.5 per pixel (flight), 8.5 → 2.2 (260 up).
- **Water in the pyramid at its drawn surface.** The pyramid held a water column's word top; the march draws the
  surface 10/9 block lower, so rays skimming the sea dipped to level 0 in every column under them and missed the
  surface (a descent, a column test and the climb back per cell). It holds the block under (the surface rounded up)
  now: from 260 up, level-0 tests 2.1 → 1.0 per pixel and their misses 1.3 → 0.2.
- **The horizon profile** (the coordinator's idea, sized against these counters). A compute pass per frame (from
  pyramid level 5, one thread per cell, 1024 azimuth bins per ring) finds per ring and direction the steepest rise at
  which the camera sees any of the ring's terrain past the shell; a ray rising faster passes the ring without marching
  it: most of the sky over the horizon, and rays crossing near rings toward far terrain. 0.014 ms; its own command
  buffer, committed ahead of the frame's as the rings' fills are, so it isn't in the main pass's GPU time.
  `METALMC_EXP=ffnoprofile` turns it off.
- **The shell's top from what the rings hold.** The highest terrain was taken from the nodes the rings should hold;
  until a ring's refill its old nodes are still drawn, and a shell top under them would cut them off.

Before and after (ms of GPU as above; before and after from two sessions, the pyramid change needing its own build):

| | flight | 260 up | near the ground |
|---|---|---|---|
| near mode, before | 2.53 | 5.04 | 1.21 |
| near mode, after | 2.05 (-19%) | 3.67 (-27%) | 0.87 (-28%) |
| near mode, after without the profile | 2.22 | 3.85 | 1.09 |
| nothing in front, before | 3.59 | 5.71 | 2.51 |
| nothing in front, after | 2.91 (-19%) | 4.24 (-26%) | 1.94 (-22%) |
| steps per 8 × 4 group (near mode) | 19.7 → 12.6 | 39.3 → 26.8 | 8.5 → 4.1 |
| steps per hit pixel (near mode) | 36.5 → 27.8 | 35.6 → 24.6 | 41.0 → 33.3 |
| rays at a ring's step cap (nothing in front) | 34 → 28 | 417 → 241 | 166 → 49 |

With nothing in front 2, 175 and 116 of the 7.7 M pixels differ: as many as the rays no longer at the step cap, which
used to give up on the rest of a ring (dots over the sea at grazing angles, and under the horizon, that drew a cliff
side or black instead of the water) and now find their hit (`bench_out/agents/far/img/final-vs-orig-*-diff.png`,
local). With the nearer geometry stood in for, a few more at the edge of its depth (5, 207 and 143). Each change was
checked the same way: the cheaper step and the parent rule differ only at the cap, the profile and the last trims are
identical.

Tried and dropped (near mode unless noted):

- One loop over all rings instead of one per ring: a group pays its slowest lane either way (13.0 against 12.8 steps a
  group), and the single loop's extra live state made it 8-10% slower.
- The column test's details after the loop (in the loop only whether there's a hit): 9-16% slower, twice.
- Two pyramid levels a step (l and l - 1 read together, descending two a step): 42% fewer steps a group from 260 up,
  each dearer: no faster than the cheaper step with the parent rule.
- Heights read from a buffer instead of the texture: 0-5%.
- A beam pass (per 8 × 4 tile, a compute pass marching the tile's rays together, as a box, against the pyramid, so each
  pixel starts where its tile may first meet terrain). An oracle (each group starting at its nearest hit) cut the far
  band's steps 45%, and the real beams 47-68% of all steps, yet the pass cost 0.27-0.37 ms (0.20-0.28 with corners
  only) and the fragment work fell only 8-12%: the steps it saves are the cheap ones (coherent, coarse levels); the
  ones near the hits remain. Net -3.5% (flight), -10% (260 up), +3% (ground).
- The shell's walls topped per sector at the steepest terrain seen (from the nodes' tops): 21% fewer pixels marched in
  the flight view, but they were the sky's cheapest groups (2% of the steps); no measurable gain and a CPU cost a frame.
- The horizon profile per band of distance as well (to start a ring's march where its terrain may first be met): -3.7%
  from 260 up, ±2% elsewhere.

Not the 2x hoped for. What's left is mostly the traversal near the hits (the band under the horizon, three quarters of
the flight view's steps): the coherent coarse steps are gone (the profile, and the beams showed the approach is
cheap), and a group still pays its slowest pixel. The levers left: fewer instructions per step (a walk within a level
that steps its boundary times instead of recomputing them; its rounding needs the care the half-cell nudge took), and
divergence, which a fragment shader can't rebalance. In the game: the quads' area (the cover test) was empty offline,
so its level-0 tests weren't measured; lit mode compiles and shades as before.
