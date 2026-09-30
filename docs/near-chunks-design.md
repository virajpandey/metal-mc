# Near chunks in our own format (rewrite swing 1, first step)

**Status (2026-09-30):** built and checked offline, not yet run in the game. Everything is behind `METALMC_EXP=nearchunks` (`-PmetalExp=nearchunks`), off by default. `METALMC_EXP=nearchunks,nearslim` also leaves the repacked layers out of vanilla's vertex heaps, which is where the memory saving comes from.

Code:
- `Sources/MetalMCNative/NearChunks.swift`: codec, arena, shaders, draw.
- `Sources/MetalMCNative/NearChunksCheck.swift`: offline checks, never run in the game.
- `mod/src/client/java/metalmc/terrain/NearChunks.java` and `NearData.java`: the Java side.
- `metalmc/backend/MetalNearChunks.java`: the bridge to the native code.
- Four new mixins in `metalmc/terrain/mixin/*Near*`.
- Marked hook points in `SectionCompilerMixin`, `LevelRendererFacingMixin`, `MetalRenderPass.uniformValue` and `MetalCommandEncoder.currentSubmit`.
- `tools/neartest.py`: the offline checks.

## What it does

Vanilla's section compiler keeps meshing: it knows every block model, fluid and tint.

1. **Repack on the mesh worker.** When `SectionCompiler.compile` returns, right after our facing sort (so the quad order is final), the section's SOLID and CUTOUT layers are repacked into 32-byte quad records in one GPU arena (`mmc_near_add`).
   - The two layers are repacked together or not at all. The grass side overlay (cutout) only passes the depth test at exactly the depth of the dirt side under it (solid), so both must come out of the same vertex shader.
   - The entry ids ride on the `CompiledSectionMesh`. They're freed when vanilla closes the mesh, the same moment vanilla frees the mesh's own heap allocations.
   - This happens on the worker, not where the backend copies vertex data into the heaps. That copy runs on the render thread (`uploadTerrainBuffersToGpu`), and encoding there would cost milliseconds per frame while chunks stream in.
2. **Divert the draws.** While vanilla extracts its section draws, each SOLID/CUTOUT draw of a repacked layer goes into a per-layer record list instead of vanilla's indirect draws. A record is {entry, first quad, quad count, section index}. The facing split still applies, so only the buckets that can face the camera are listed.
   - This happens only when the near-chunk shaders are compiled and terrain is on the multi-draw-indirect path.
   - A layer the codec refused stays in vanilla's list.
3. **Draw.** At the end of `ChunkSectionsToRender.DrawIndirect.render` for each layer, in the same render pass, `mmc_near_draw` draws the list from the arena with `near_vs`/`near_fs`.
   - It uses the uniform buffers vanilla just bound for its own draw of that layer (Projection, TerrainUniform, Globals, Fog, the block atlas and its sampler, the lightmap).
   - Positions and fade come from vanilla's per-section instanced stream (`chunkSectionInfos`, via `baseInstance`).
   - The order stays vanilla solid, near solid, vanilla cutout, near cutout, so the cutout overlays still land on the solid faces under them.
   - Translucent terrain stays vanilla's: it needs per-quad sorting, and water is most of it.

The shader is `core/terrain.vsh` + `terrain.fsh` (with `fog.glsl`, `sample_lightmap.glsl` and `texture_sampling.glsl`) transcribed operation for operation:
- camera-relative integer positions, `(ProjMat * ModelViewMat) * pos`, the `flip_vert_y` flip;
- the per-vertex lightmap sample;
- the chunk fade (16-32 blocks);
- `sampleNearest`/RGSS with `textureGrad`;
- the alpha cutout at 0.5 (a function constant, CUTOUT only);
- spherical plus cylindrical fog.

Colors go through `unpack_unorm4x8_to_float`, the same conversion as the vertex fetch of vanilla's RGBA8 color. A division by 255 rounded one pixel differently in the checks. The position output is `[[invariant]]`, so the solid and cutout pipelines agree on depth.

## The format

A quad is one 32-byte record, and the draw's vertex id picks the record and corner: `vid >> 2`, `vid & 3`. The quad index buffer is vanilla's pattern (0 1 2 2 3 0), so the triangle split, and with it the interpolation of AO and light, is vanilla's.

**Aligned record** (bit 0 of word 0 clear): a rectangle on the 1/16-block grid in a plane of constant x, y or z.
- Origin: 3 × 9 bits, −1 to 30.9 blocks, section-relative.
- Two extents, 9 bits each.
- Which corner vertex 0 is, and which way the vertices go round (3 bits). This keeps vanilla's vertex order.
- UVs of vertices 0 and 2 (17 bits each, u·65536), plus 1 bit saying how vertices 1 and 3 mix them.
- A tint (RGB8) and four 8-bit gray levels. Vanilla's vertex color is exactly `ARGB.multiply(gray(ao·shade), tint) = floor(T·g/255)` per channel.
  - The encoder finds a T and g that reproduce the four colors exactly. It tries the gray levels vanilla's lighter produces for whole faces first, then every level.
  - Gray colors are a white tint.
- Four block/sky light pairs, 8 bits each.

**Generic record** (bit 0 set): used when any of the aligned conditions fails.
- Cases: plants with random offsets, cross models, rotated elements, fluid tops, colors the solver can't split.
- Layout: vertex 0 is in the record, and vertices 1-3 follow the section's records, 80 bytes a quad in all.
- Each generic vertex is 16 bytes:
  - position: 3 × 18 bits, 1/8192 block over −8..24;
  - color: RGB8;
  - light: block and sky, 8 bits each;
  - UV: u and v, 17 bits each.

**Refused layers:** the encoder refuses a layer (it stays vanilla's) if any vertex has
- alpha other than 255,
- light outside 0-255,
- a UV outside 0..1,
- a position outside the generic range, or NaN,
- or a vertex count that isn't a multiple of 4.

**What's exact:**
- colors and light (the 8-bit values);
- vertex order;
- aligned positions;
- UVs on the 1/65536 grid: every whole texel of a power-of-two atlas up to 65536 wide, every 1/16 texel up to 4096 wide, and 1.0 itself.

**What's rounded:**
- generic positions, to 1/8192 block (at most 6.1e-5 block);
- UVs off that grid (flowing fluids' rotated ones), to within 7.6e-6.

## The arena

- **Slabs.** A list of 64 MiB `MTLBuffer` slabs of 32-byte slots, each with a first-fit free list (sorted, merged on free).
- **Adding a layer.** A worker encodes into a scratch array, allocates under the lock, copies without it, then publishes the entry (id = slot + 7-bit generation, always a positive Java int).
- **Freeing.** Freeing tags the range with the newest submit that drew near chunks. The range goes back to its slab only once `ctx.completed` passes that submit, so a worker never overwrites memory an in-flight frame reads.
- **Stale ids.** Draws look entries up at draw time, so an entry freed between extraction and drawing is skipped rather than drawn from reused memory.
- **Not done yet.** Slabs are never given back to the system.

## Checked offline (`python3 tools/neartest.py`, all passing)

Run with `NEARTEST_LIB=<worktree>/.build-wt/release/libMetalMCNative.dylib`.

1. **Allocator.** 3 seeds × 200,000 random allocations and frees on a 262,144-slot slab. After every step: the free list is sorted, merged and non-empty, and free + used = total. Every 97 steps: no live allocation overlaps another or a free range.

2. **Codec on real terrain.** 42 random chunks from 12 regions of `fixtures/claudeworld-merged`, turned into vanilla-like BLOCK vertices by `mmc_near_debug_synth_chunk`:
   - cube faces with vanilla's culling (fancy leaves) and FaceInfo corner order;
   - AO from the four samples per corner and cardinal shade;
   - grass and leaf tints, smooth sky light;
   - random per-block UV rotations on top faces;
   - the grass side overlay;
   - grass, ferns and flowers as cross models with vanilla-style random offsets.

   | Measure | Result |
   |---|---|
   | Quads | 506,054 |
   | Aligned records | 99.22% (32 B) |
   | Generic records (the plants) | 0.78% (80 B) |
   | Average size | **32.37 bytes a face against vanilla's 112 (3.46× smaller)** |
   | Tinted aligned quads | 42,238, every one split exactly (0 misses) |
   | Aligned positions, UVs, colors, light | exact |
   | Generic positions | at most 6.03e-5 block off |
   | GPU decode (`near_decode_test`) vs CPU decoder | equal, value for value, on every vertex |

3. **Hand-made shapes (28 quads).** Slab top and side (partial faces with weighted AO), a stair step, a torch (1/16 insets, emissive light), offset cross plants, lava with a sloped top, 0.001 insets and flow-rotated UVs in a 32×32 sprite, a plane rotated 22.5°, tinted partial faces with arbitrary gray levels, planes at x = −1 and y = 17 (the grid's edges), and both UV layouts plus a skewed mapping.

   | Measure | Result |
   |---|---|
   | Aligned records | 42.9% |
   | Generic position error | at most 5.79e-5 block |
   | UV error | at most 5.72e-6 (the lava's rotated UVs) |
   | Colors and light | exact |

4. **Refusals.** The encoder refuses alpha 128, light 300, a NaN position, a position of 40, a UV of 1.5, and 6 vertices.

5. **Rendering: pipeline equivalence.** Each scene is drawn twice at 640×400: once through a transcription of vanilla's `terrain.vsh` reading the original 28-byte vertices through the vertex fetch, and once through `near_vs` reading the records. Both use `near_fs` with the same fog, fade, cutout and RGSS. The textures are a noise atlas with transparent texels and a lightmap gradient.

   | Scene | Pixels differing |
   |---|---|
   | Real section, solid | 0 |
   | Real section, cutout with 0.55 fade | 0 |
   | Real section, RGSS close-up with fog | 0 |
   | Hand shapes, vanilla's shader fed the rounded vertices | 0 in every view |

6. **Rendering: what rounding generic positions and UVs costs,** against the original vertices:

   | View | Pixels differing | Largest difference |
   |---|---|---|
   | Plants, close-up | 0 of 11,555 | 0 |
   | Overview | 102 of 2,847 | 232/255 |
   | Lava, close-up | 1,784 of 4,914 | 12/255 |

   - The overview differences are along the lines where the crossed plant planes intersect, where depth order is decided by a few ULPs, and in distant sub-texel sampling on the noise texture.
   - The lava differences are on texel edges, from the rotated flow UVs.

7. **The game's own draw path.** `mmc_near_debug_render_arena` runs:
   - the real arena, with a filler entry first so the layer doesn't start at slot 0;
   - `mmc_pass_begin` and `mmc_near_draw` with two records per layer and section index 1 of a two-section stream;
   - uniforms at different offsets of one buffer;
   - `mmc_submit`.

   Results:
   - Images are identical to the reference except 1 pixel off by one level in one of four scenes. The game's pipeline comes from its own library build.
   - A freed range isn't handed out again before its submit completes, and is reused after.

8. **Encode speed.** 33 ns a quad untinted and 50 ns tinted (142,000 quads). A 3,000-quad layer is about 0.1-0.15 ms on a mesh worker. Vanilla's own meshing of such a layer costs several times that. An earlier version that built Swift SIMD vectors lane by lane took 150-230 ns.

9. **Builds and shaders.** Swift release build and Java `compileJava compileClientJava` are clean. `mslcheck` passes on the shader source (`near_vs`, `near_fs`, `near_decode_test`) and on the check library (plus `near_ref_vs`). The mixin targets were checked against the 26.3 bytecode with `javap`:
   - `DrawIndirect.render` descriptor and its `chunkSectionInfos` field;
   - `CompiledSectionMesh.<init>` and `close`;
   - `LevelRenderer.usingMultiDrawIndirectForTerrain`;
   - the two `UberGpuBuffer.addAllocation` calls in `addSectionBuffersToUberBuffer`, the vertex one first;
   - the three `List.add` calls in `extractSectionDrawGroups`.

   There's no refmap (the game is unobfuscated), so the injectors themselves are only exercised at startup (`defaultRequire` is 1).

**Memory, estimated.** At render distance 12, vanilla keeps one 128 MiB vertex heap per layer (about 1.2 M quads each; `docs/research/vanilla-26.3-chunk-rendering.md`).
- The same faces take 32-37 bytes each here, against 112: about 32 MB per million quads against 112 MB.
- By default vanilla still uploads its own copy (dual storage), so memory goes up. With `nearslim`, vanilla's heaps get a 28-byte placeholder per repacked layer instead.

## In-game checks for Viraj's main agent (not run: nothing here launched the game)

1. **Boot and log.**
   ```
   BENCH_TIMEOUT=300 bash tools/bench/bench_lod.sh nearA 0 -PmetalExp=nearchunks
   BENCH_TIMEOUT=300 bash tools/bench/bench_lod.sh nearA0 0
   ```
   - `bench_out/run_nearA.log` should show `near chunks: shaders compiled` and, after 1,200 frames, `near chunks: N layers repacked, M left to vanilla; live … quads in … MB (… B/quad vs 112) …; per frame …: … draws, … quads, … ms CPU, 0 records missing`.
   - It should show no `couldn't draw the … layer` warning and no GPU errors.
   - Compare `fps_mean`, `metal_gpu_ms_mean` and `frames_over_8ms` with `nearA0`.
   - Check that "left to vanilla" stays small. If it doesn't, find which blocks refuse (probably alpha ≠ 255 or light > 255 from some model) and decide whether the codec should hold them.

2. **Fidelity A/B against vanilla's own terrain** (TAA is off in bench runs, so frames are deterministic):
   ```
   BENCH_TIMEOUT=900 bash tools/bench/fidelity.sh nearV 12 0
   BENCH_TIMEOUT=900 bash tools/bench/fidelity.sh nearF 12 0 -PmetalExp=nearchunks
   swiftc -O tools/bench/imgdiff.swift -o bench_out/imgdiff
   for f in mod/run/screenshots/nearV-tour-*.png; do bench_out/imgdiff "$f" "${f/nearV/nearF}" "bench_out/nearAB-${f##*/}" 1; done
   ```
   - Expected: well under 0.1% of pixels changing by more than 1 level. The exceptions are texel edges on lava tops and plant intersection lines.
   - Look at the heatmaps for whole sections missing or doubled (the diversion) and for grass-side overlays that flicker (depth mismatch).
   - Then repeat with the LOD: `fidelity.sh nearFL 12 8192 -PlodGenerate=0 -PmetalExp=nearchunks` against `nearVL`. The LOD's seam logic reads vanilla's section lists, which the diversion leaves unchanged.

3. **Flight and streaming.**
   ```
   BENCH_FIXTURE=claudeworld-merged BENCH_TIMEOUT=900 bash tools/bench/bench_lod.sh nearFly 32768 -PbenchY=150 -PbenchFly=20 -PbenchExtraWait=600 -PbenchHitches=1 -PmetalExp=nearchunks
   ```
   Compare with the same run without the flag. Check `frames_over_8ms` and the hitch lines, the log's near-chunk CPU per frame (expect about 0.1-0.2 ms at RD 12, like vanilla's own indirect encoding), and that live quads stay bounded as sections unload.

4. **GPU cost.** Add `-PbenchTrace=1` to runs 1-3. The main pass's time should match vanilla's terrain within noise. Vanilla at RD 12 is about 0.6 ms. The vertex cost is per vertex invocation (`docs/lod-design.md`), and the record fetch is 32 bytes a quad against 112.

5. **Memory with `nearslim`.**
   ```
   bench_lod.sh nearS 0 -PmetalExp=nearchunks,nearslim
   ```
   - Compare the process footprint (`footprint <pid>` or Activity Monitor) with `nearA` and `nearA0` at the same spot.
   - The fidelity tour with `nearchunks,nearslim` must match `nearF` pixel for pixel.

6. **By hand, in a window.**
   - Break and place blocks next to you: synchronous compiles go through the same repack.
   - Water and lava next to glass.
   - Leaves, both the fancy and fast leaves options.
   - Grass sides up close (the overlay).
   - The wireframe debug view: near chunks switch to lines too.
   - Changing the render distance, F3+A (all sections recompile), F3+T (atlas reload).
   - Leaving and rejoining the world: the log's live-quad count should drop to about zero between worlds.
   - The Nether and the End (different cardinal shading).
   - A resource pack with a large atlas: UVs stay exact up to 65,536 texels on whole texels.

## Known gaps and next steps

- **The draw list is still built on the CPU.** It costs one draw call per section layer run, the same encoder cost as vanilla's non-ICB multi-draw path.
  - Next, a compute pass should build the draws from a GPU-resident section table (frustum and occlusion culling on the GPU) into an indirect command buffer. Our pipeline can support ICBs once its textures move to an argument buffer.
  - That's where "one GPU-built draw list" and near chunks as level −1 of the LOD come in.
- **Only the MDI path is diverted.** Terrain drawn without multi-draw-indirect (the `DrawSeparate` path) stays vanilla's.
  - With `nearslim` that path would draw placeholders, so slimmed layers are dropped from vanilla's lists and holes appear instead.
  - `nearslim` also must not be combined with switching multi-draw-indirect off at runtime.
- **Generic records are rounded.** They use 1/8192-block positions and 16-bit UVs, which is visually lossless but not bit-exact.
  - An exact generic form (float positions) would cost 128 bytes a quad, more than vanilla's 112.
  - A dedicated "cross plant" form (two xz points + y range + offsets) could bring plants to 32 bytes if they turn out common on the surface.
- **Translucent terrain** (water, stained glass, ice) is vanilla's. Repacking it needs the sorted index buffers vanilla rebuilds per camera move.
- **The baked AO and light.** The records carry vanilla's per-vertex values. The lighting swing is expected to replace them with light computed from volumes, which would take an aligned record down to about 8-16 bytes (see the bit budget in the research doc).
- **Arena slabs are never returned to the system.** An emptied slab stays allocated.
