# Far field: a height-field ray march past the voxel LOD (design, not built yet)

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
