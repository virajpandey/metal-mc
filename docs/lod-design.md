# Far-terrain LOD (Voxy-style, clean-room)

Goal: draw real terrain far past vanilla's render distance on the Metal backend, cheaply enough that a short vanilla render distance plus LOD beats a long vanilla render distance on both frame time and view distance.

Voxy (All Rights Reserved) and its public discussion were studied for techniques only. No Voxy code is used, and Distant Horizons is not used as a reference. `~/research_notes/voxy-techniques.md` has the research notes.

## Data model (v1)

- **Levels.** A level-L voxel covers 2^L × 2^L × 2^L blocks, for L ≥ 1. Level 0 (full-resolution blocks) is left to vanilla.
- **Nodes.** A node is a 256 × 256-voxel column spanning the full world height (y −64 to 320). A level-1 node is exactly one region file (512 × 512 blocks). A level-(L+1) node merges 2 × 2 level-L nodes at half resolution.
- **Voxels.** Each voxel is one byte, a material id from `MetalMCCore.Mat`, the same flat-color material table the standalone engine uses.
- **Downsampling keeps the top layer.** In each 2 × 2 × 2 group, the highest non-air child wins, so grass, sand and snow survive at a distance and thin features such as trees stay as blobs.
- **Deep air is filled.** Before meshing, air and water more than 4 voxels below the lowest open ground within 8 voxels is filled with stone. "Open ground" is the height from which a column is open to the sky, and the lowest one in the neighborhood is taken. This removes cave networks under hills even when they connect to a surface entrance somewhere. They were 49% of all quads, measured on six level-1 regions of the 8 km world (`mmc_debug_lod_mesh_stats`). Entrances stay open to that depth, and arches and overhangs stay open because the ground beside them is open. `METALMC_EXP=nodeepfill` turns it off for comparisons.
- **Unreachable air is filled.** Then air and water that can't be reached from the sky or the node's sides is filled with stone as well (sealed caves).

## Mesh

- **8-byte quads.** `word0 = x | z<<8 | y<<16 | face<<24` (voxel coordinates within the node), `word1 = material | (w−1)<<8 | (h−1)<<16`. That's 8 bytes per quad, against vanilla's 4 × 28 bytes.
- **Greedy merging.** Merges are capped at 16 voxels per side at level 1, so quads can be dropped individually near vanilla's edge, and at 64 voxels above that.
- **Water.** LOD water is opaque: only water faces toward air are emitted, and nothing under the water is.
- **Skirts.** The node's sides count as air, so every node emits walls along its edges. These hide cracks where neighboring nodes are at different levels.

## Rendering

- **Where it draws.** In Minecraft's main world pass, right after solid terrain (`LevelRendererLodMixin`), encoded natively into the same `MTLRenderCommandEncoder`. It uses the game's projection and reverse-Z depth, so it depth-tests against vanilla. Afterwards Minecraft's pipeline state is re-applied.
- **Tiles.** Each node is meshed as 4 × 4 tiles of 64 voxels, and greedy merges don't cross tile edges along x and z. The quads are stored tile-major, then by face direction, with each tile's vertical range. The draw culls per tile: view frustum, "entirely inside vanilla's range", and face directions that can't face the camera. Adjacent visible face ranges merge into one draw.
- **Selection.** A quadtree per frame on the CPU: a node splits into its four children when the camera is within 2 × the child size and all four children exist.
- **Occlusion culling.** Right after the LOD draws, every candidate tile's bounding box is rasterized in the same render pass. The boxes are grown by one voxel, and only their faces toward the camera are drawn, with depth testing on and no depth or color writes. At that point the depth buffer holds vanilla's solid terrain and the LOD. The fragment function has `[[early_fragment_tests]]` and marks the tile visible in a shared buffer. When the command buffer completes, the CPU reads the marks. Tiles whose box was fully hidden in the newest completed test are skipped. Every candidate tile is tested again each frame, so a tile that comes into view reappears 2–3 frames later.
  - Tiles whose box contains the camera are never tested, so they are always drawn.
  - After the camera jumps more than 16 blocks, results tested before the jump are ignored.
  - There's no render pass break and no depth readback, which matters on a tile-based GPU.
  - Translucent terrain, entities and clouds are drawn later in the pass, so they never act as occluders.
  - `METALMC_EXP=noocc` turns it off, and `occnocull` runs the test without skipping anything.
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
- **Live chunks** (`LodLive.swift`, `LiveIngest.java`). Every overworld chunk the client loads is handed to the LOD, and so is every chunk it unloads, which captures edits made while the chunk was loaded. The client thread only copies the chunk's non-empty block sections (`PalettedContainer.copy()`) and samples its 4 × 4 surface biomes at y = 96. A low-priority worker thread maps block states to LOD materials, with a per-state cache and the native classifier, and passes 16 × 16 × 384 material bytes to native code. There they are downsampled exactly like the region reader and stored per chunk at level-1 resolution (8 × 8 columns of 192 voxels, run-length encoded, about 2 KB). When a region is rebuilt, its live chunks replace the region file's version. In single-player this keeps the LOD current without waiting for the server to save.
- **Multiplayer.** A server has no region files, so the LOD is built from live chunks only. The store is saved per server under `<game dir>/metalmc/lod/<address>/overworld/`, one file per region compressed with LZFSE (about 0.8 KB per chunk), and loaded when the player rejoins. The LOD shows the terrain the player has seen on that server, as in Voxy. Config: `lod.live`, `lod.multiplayer`.

## Colors

- **Base materials** use the average color of each material's block texture (`LodColors.swift`, from `tools/lod_colors.py`).
- **Grass, leaves and water are biome-tinted** (`LodBiomes.swift`). The chunk decoder reads each chunk's 4 × 4 surface biome grid (y 96–111, above cave biomes). Tinted voxels use material ids 64 + t, 96 + t and 128 + t for 20 tint classes (vanilla's grass/foliage/water colors), so the tint survives downsampling. `results/lod-v1/far-orbit-biome-tints.png` shows savanna hills matching vanilla's yellow-olive grass across the seam.

## Where the time goes

Per-frame counters from `-PbenchTrace=1` runs: `per_frame_lod_draws`, `per_frame_lod_kquads` and `per_frame_lod_cpu_ms`.

- **The CPU side is cheap.** Selection, culling and encoding take 0.06–0.08 ms per frame for 420–570 draws. GPU-driven selection with indirect command buffers would save almost nothing, so it isn't a priority.
- **The GPU cost is quads.** At 8 km, the LOD submitted 896 K quads per frame and added 0.95 ms of vertex time and 0.55 ms of fragment time to the main pass (1.05 → 2.55 ms). On a tile-based GPU, vertex shading and tiling scale with primitive count, and fragment cost includes per-tile primitive processing. Cutting the quads drawn is what pays.
- **Deep-cave filling** took the 8 km LOD from 896 K to 567 K quads per frame and 267 to 320 fps. It also cut total quads from 11.8 M to 7.7 M and GPU memory from 95 to 62 MB. The screenshots are pixel-identical apart from moving animals.
- **Occlusion culling** matters most at ground level, where hills hide most far terrain. On an 8 km LOD at ground level it cut quads per frame from 579 K to 170 K and draws from 430 to 79. That took the LOD from 337 to 362 fps, against 382 with no LOD at all. From 150 blocks up, only about a quarter of the LOD is hidden. The test costs about as much as it saves there (318–325 fps with it, 320 without). Drawing every box face instead of only camera-facing ones doubled the test's cost and made that view 4% slower.

## Known gaps
- **Edges of explored areas at node borders.** Inside a node, side faces toward a column with no data (an unexplored or ungenerated chunk) are kept only within 8 voxels of the top of the terrain. That leaves a short skirt instead of the terrain's cross-section down to the world bottom. The deep-cave fill ignores such columns, so caves next to them are filled too. On the multiplayer test strip this removed 24% of quads (442,407 → 337,911). Node borders keep full-depth skirts, since they hide cracks between LOD levels where a cliff meets the border. Where explored terrain ends exactly at a node border, the cross-section still shows.
- **Edits while a chunk stays loaded** reach the LOD when the chunk unloads (or when the save changes, in single-player), not immediately. They're mostly inside vanilla's range anyway.
- **Screen-size selection.** Levels switch by distance: a node splits when the camera is closer than its size. That gives voxels of about 4–8 px at 4K. Voxy switches by projected size, 64 px per 32-voxel section. That's finer than ours, so ours already trades detail for speed.
