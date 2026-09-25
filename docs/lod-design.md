# Far-terrain LOD (Voxy-style, clean-room)

Goal: draw real terrain far past vanilla's render distance on the Metal backend, cheaply enough that a short vanilla render distance plus LOD beats a long vanilla render distance on both frame time and view distance.

Voxy (All Rights Reserved) and its public discussion were studied for techniques only. No Voxy code is used, and Distant Horizons is not used as a reference. `~/research_notes/voxy-techniques.md` has the research notes.

## Data model (v1)

- **Levels.** A level-L voxel covers 2^L × 2^L × 2^L blocks, for L ≥ 1. Level 0 (full-resolution blocks) is left to vanilla.
- **Nodes.** A node is a 256 × 256-voxel column spanning the full world height (y −64 to 320). A level-1 node is exactly one region file (512 × 512 blocks). A level-(L+1) node merges 2 × 2 level-L nodes at half resolution.
- **Voxels.** Each voxel is one byte, a material id from `MetalMCCore.Mat`, the same flat-color material table the standalone engine uses.
- **Downsampling keeps the top layer.** In each 2 × 2 × 2 group, the highest non-air child wins, so grass, sand and snow survive at a distance and thin features such as trees stay as blobs.
- **Unreachable air is filled.** Before meshing, air and water that can't be reached from the sky or the node's sides is filled with stone. Sealed caves are invisible from LOD distances, and their walls were half the mesh. Cave entrances stay open.

## Mesh

- **8-byte quads.** `word0 = x | z<<8 | y<<16 | face<<24` (voxel coordinates within the node), `word1 = material | (w−1)<<8 | (h−1)<<16`. That's 8 bytes per quad, against vanilla's 4 × 28 bytes.
- **Greedy merging.** Merges are capped at 16 voxels per side at level 1, so quads can be dropped individually near vanilla's edge, and at 64 voxels above that.
- **Water.** LOD water is opaque: only water faces toward air are emitted, and nothing under the water is.
- **Skirts.** The node's sides count as air, so every node emits walls along its edges. These hide cracks where neighboring nodes are at different levels.

## Rendering

- **Where it draws.** In Minecraft's main world pass, right after solid terrain (`LevelRendererLodMixin`), encoded natively into the same `MTLRenderCommandEncoder`. It uses the game's projection and reverse-Z depth, so it depth-tests against vanilla. Afterwards Minecraft's pipeline state is re-applied.
- **Selection.** A quadtree per frame on the CPU: a node splits into its four children when the camera is within 2 × the child size and all four children exist.
- **Seam.** Quads whose center lies within `(renderDistance − 1) × 16` blocks horizontally are collapsed in the vertex shader. There's no fragment `discard`, which would turn off Apple GPUs' hidden-surface removal for the whole pipeline.
- **Far plane and fog.** The camera's far plane is pushed to 1.5 × the LOD distance; reverse-Z float depth keeps precision. Vanilla's render-distance fog moves from the chunk edge to the LOD edge. The LOD shader reproduces vanilla's fog formula, so it blends into the same sky.

## Known gaps in v1

- The LOD is static: built once from the save when the world opens. It doesn't update on block changes or exploration.
- Level-1 nodes are only meshed within 1.5 km of the starting position.
- Colors come from a flat 31-material table, not block textures or biome tint.
- Everything is drawn one call per node with CPU selection. GPU-driven selection with indirect command buffers is next, and possible because this pipeline binds no textures.
