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
- **Tiles.** Each node is meshed as 4 × 4 tiles of 64 voxels, and greedy merges don't cross tile edges along x and z. The quads are stored tile-major, then by face direction, with each tile's vertical range. The draw culls per tile: view frustum, "entirely inside vanilla's range", and face directions that can't face the camera. Adjacent visible face ranges merge into one draw.
- **Selection.** A quadtree per frame on the CPU: a node splits into its four children when the camera is within 2 × the child size and all four children exist.
- **Seam.** Quads whose center lies within `(renderDistance − 1) × 16` blocks horizontally are collapsed in the vertex shader. There's no fragment `discard`, which would turn off Apple GPUs' hidden-surface removal for the whole pipeline.
- **Far plane and fog.** The camera's far plane is pushed to 1.5 × the LOD distance; reverse-Z float depth keeps precision. Vanilla's render-distance fog moves from the chunk edge to the LOD edge. The LOD shader reproduces vanilla's fog formula, so it blends into the same sky.
- **Environmental haze.** The overworld's clear-weather haze (`FOG_END_DISTANCE`) defaults to a linear 0 → 1,024 blocks, which fully hides anything past 1 km; vanilla never draws that far, so it never shows. With LOD active, that default haze is stretched to the LOD distance. Shorter fogs (rain, water, lava, blindness, the Nether's 96 blocks) are unchanged.

## Ingestion speed and memory

- **Chunk decoding.** `ChunkScan` walks each chunk's NBT once and reads only position, status, block palettes and data, and the surface biome grid; everything else is skipped by length. It inflates with the Compression framework straight into a reusable per-region buffer. It is verified identical to the general NBT decoder (0 mismatches on 3,072 chunks) and takes 0.14–0.21 s per region versus 1.6–2.3 s.
- **Cache memory.** The cached level-2 quadrants are run-length encoded per column: 31 MB instead of 157 MB for 100 regions.

## Streaming (v2, `LodWorld.swift`)

- **Change detection.** A background thread polls the region files every 2 s (modification time and size). The integrated server writes chunks when they unload and on autosave, so explored and edited terrain reaches the LOD through the same path as the initial build.
- **Parents without re-reading siblings.** Each region caches its level-2 quadrant (128 × 128 × 96 voxels, 1.5 MB). A parent node is rebuilt from the cached quadrants of its children.
- **The finest level follows the player.** Level-1 nodes are meshed within 1.5 km of the player, re-centered when they move more than 128 blocks, and dropped when they leave that range.
- **Deferred regions.** Regions inside vanilla's render distance (plus 64 blocks) change constantly with autosave. Rebuilding them is deferred until the player has moved away. Before this, 2–4 regions were rebuilt every 2 s; now it's zero while the player stays in the area.
- **Swapping nodes.** Each node owns its Metal buffer, so updates replace nodes atomically while the render thread draws.
- **Single-player only.** Multiplayer would need to ingest chunks as the client receives them (Voxy's approach).

## Colors

- **Base materials** use the average color of each material's block texture (`LodColors.swift`, from `tools/lod_colors.py`).
- **Grass, leaves and water are biome-tinted** (`LodBiomes.swift`). The chunk decoder reads each chunk's 4 × 4 surface biome grid (y 96–111, above cave biomes). Tinted voxels use material ids 64 + t, 96 + t and 128 + t for 20 tint classes (vanilla's grass/foliage/water colors), so the tint survives downsampling. `results/lod-v1/far-orbit-biome-tints.png` shows savanna hills matching vanilla's yellow-olive grass across the seam.

## Known gaps
- Everything is drawn one call per node with CPU selection. GPU-driven selection with indirect command buffers is next, and possible because this pipeline binds no textures.
