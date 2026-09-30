# Vanilla 26.3 chunk (terrain) rendering: what a replacement near-chunk renderer has to match

This report was written on 2026-09-30 from the decompiled Loom sources (`minecraft-clientOnly-7e9a32a5b8-26.3-sources.jar`, `minecraft-common-…-sources.jar`) and the GLSL in the client jar. It is a static read of the code: no game was run for it. Anything marked *measured* comes from our own existing bench logs (`results/metal-26.3/*.txt`, vanilla terrain with `lod=false` on our Metal backend). Anything marked *estimate* is arithmetic based on the code.

## 0. Key numbers

- **Vertex format.** `DefaultVertexFormat.BLOCK` is **28 bytes per vertex**. Every quad is 4 vertices and 6 indices, so one block face costs **112 bytes** of vertex data. Faces are **never merged**.
- **Indices.** Solid and cutout geometry uses a shared, auto-generated quad index buffer, so it stores no indices of its own. Translucent geometry adds a per-section sorted index buffer: 12 bytes per quad with `SHORT` indices, 24 bytes with `INT`.
- **Draw data.** Each visible section gets a 16-byte `ChunkSectionInfo`. Each visible section-layer gets a 20-byte indirect command (`DynamicGpuData.IndexedDraw`).
- **GPU buffers.** Each layer has its own 128 MiB vertex heaps (`UberGpuBuffer` with a `TlsfAllocator`). Index heaps are 32 MiB, and there is a 98 MiB staging buffer (`SectionRenderDispatcher` constructor). One 128 MiB heap holds about 1.198 M quads.
- **Draw calls.** Terrain uses multi-draw-indirect: **one `drawIndexedIndirect` per layer per heap group**.
  - *Measured* on our Metal backend with our section-occlusion culling off: 3.0 indirect calls and about 1,605–1,684 section sub-draws per frame at render distance 12, taking 0.12–0.15 ms of CPU time to encode.
  - At render distance 32: 4.3 calls and 6,096 sub-draws at ground level (0.43 ms), and 5.1 calls and 8,096 sub-draws at the high orbit (0.58 ms).
  - At RD 32 the main pass takes 4.3 ms of a 5.5 ms frame: 2.6 ms vertex, 1.7 ms fragment (`results/metal-26.3/README.md`). Terrain at high render distance is vertex-bound.
- **Depth.** Depth is reverse-Z: the pipeline compares with `GREATER_THAN_OR_EQUAL` and the buffer is cleared to 0.0.

## 1. Vertex formats (`com.mojang.blaze3d.vertex.DefaultVertexFormat`)

| Stream | Attribute | Format | Bytes | Content |
|---|---|---|---|---|
| binding 0 (`BLOCK`) | `Position` | `RGB32_FLOAT` | 12 (offset 0) | Block position relative to the section origin, as a float: 0..16 plus the model's coordinates (elements may span −1..2 blocks) plus the block's random offset. |
| | `Color` | `RGBA8_UNORM` | 4 (offset 12) | AO × directional shade × biome tint. Alpha is 255. |
| | `UV0` | `RG32_FLOAT` | 8 (offset 16) | Normalized block-atlas UV (static, including for animated sprites). |
| | `UV2` | `RG16_SINT` | 4 (offset 24) | Light: (block, sky), each 0..240. Smooth light uses steps of 4; flat light is `level<<4`. |
| binding 1 (`CHUNK_DATA_INSTANCED`, MDI only) | `ChunkPosition` | `RGB32_SINT` | 12 | Section origin in world blocks. |
| | `ChunkVisibility` | `R32_FLOAT` | 4 | Fade-in factor, 0..1. |

- **No normal.** There is no normal and no padding: the old 32-byte `BLOCK` layout (…`Normal`, padding) is gone. Directional shade is baked into `Color`.
- **Writer.** `BufferBuilder.addVertex(...)` has a fast path for `blockFormat`, and `VertexConsumer.putBlockBakedQuad` writes the 4 vertices of each `BakedQuad`.
- **Topology.** `PrimitiveTopology.QUADS` becomes 6 indices per quad through `RenderSystem.sharedSequentialQuad`, in the order `(i, i+1, i+2), (i+2, i+3, i)`. The diagonal is fixed; there is no AO-based flip.
  - `IndexType.least(n)` picks `SHORT` when the vertex count is below 65,536 and `INT` otherwise.
  - The shared index buffer is sized to the largest section index count (`requestIndexCount(largestIndexCount)`).
- **Merging.** `SectionCompiler` visits each of the 4,096 blocks and emits every surviving `BakedQuad`, so there is no greedy meshing.
- **Faces that cost more than one quad:**
  - A grass-block side is 2 coplanar quads: dirt in `SOLID`, plus the tinted `grass_block_side_overlay` in `CUTOUT`.
  - Water top and side faces usually get a back face as well (`FluidRenderer.addFace(..., addBackFace)`).

## 2. Meshing path

### 2.1 Scheduling

1. **Collect dirty sections.** `LevelExtractor.extractLevel` goes through `levelRenderer.visibleSections()`. For each section that `SectionUpdateTracker` marks dirty, it adds a `SectionUpdateRenderState` holding a snapshot, `RenderRegionCache.createRegion`.
2. **Compile.** `LevelRenderer.compileSections` runs after the frame graph executes. It calls `RenderSection.compileSync` when `prioritizeChunkUpdates` asks for it (NEARBY means `distSqr < 768`, about 27.7 blocks from the section center, or the player changed the section). Otherwise it calls `compileAsync`, which queues a `CompileTask` in `SectionTaskDynamicQueue` on `Util.backgroundExecutor()`.
3. **Worker buffers.** Each worker takes a `SectionBufferBuilderPack` with initial capacities of 4 MiB for `SOLID`, 4 MiB for `CUTOUT` and 768 KiB for `TRANSLUCENT`. The number of packs is `min(cores, 0.3 × maxHeap / 8.75 MiB)`.
4. **Snapshot.** The snapshot is a `RenderSectionRegion`: 3×3×3 `SectionCopy`s, each a `PalettedContainer<BlockState>.copy()` plus a copy of the chunk's block-entity map.
   - Light is **not** copied; it is read live from `LevelLightEngine`.
   - Tint comes from `ClientLevel.getBlockTint`: a `BlockTintCache` per resolver (GRASS, FOLIAGE, DRY_FOLIAGE, WATER), with `biomeBlendRadius` 0–7 (default 2, a 5×5 blend).

### 2.2 `SectionCompiler.compile` (`net.minecraft.client.renderer.chunk`)

For every non-air block in the section:

- **Visibility graph.** `VisGraph.setOpaque` runs if the block `isSolidRender()`. It is a flood fill over a 4,096-cell BitSet that produces a `VisibilitySet` of face-to-face visibility for the occlusion BFS.
- **Block entities.** If the block has a block entity, it is added to `results.blockEntities`. Only a list is kept; nothing is meshed.
- **Fluids.** A non-empty fluid state goes to `FluidRenderer.tesselate` (§2.6).
- **Models.** If `RenderShape.MODEL`, `ModelBlockRenderer.tesselateBlock` is called with `blockModelSet.get(state)` and seed `state.getSeed(pos)`. `RenderShape` has only `INVISIBLE` and `MODEL`. Output is routed per quad to `quad.materialInfo().layer()`.
  - `ModelBlockRenderer.forceOpaque` sends `LeavesBlock` to `SOLID` when `cutoutLeaves` is off. In that mode `LeavesBlock.skipRendering` also culls faces between neighboring leaves.
- **Finishing up.**
  - `TRANSLUCENT` gets `MeshData.sortQuads` (§2.7).
  - `BlockModelLighter` caches, per thread, 100 light values and 100 shade values, keyed by position.
  - An empty result becomes `CompiledSectionMesh.EMPTY`.

### 2.3 Block models to quads

- **Quad data.** A `BakedQuad` holds 4 `Vector3fc` positions, 4 packed float UVs, a `Direction` and `MaterialInfo(sprite, layer, tintIndex, shadeDirectionOverride, lightEmission)`.
- **Layer.** The layer is decided per quad at bake time. `FaceBakery.computeMaterialTransparency` checks the sprite's actual pixels inside the face's UV rectangle (`SpriteContents.computeTransparency(u0, v0, u1, v1)`), unless the material has `forceTranslucent`. `ChunkSectionLayer.byTransparency` then maps any translucent pixel to `TRANSLUCENT`, else any transparent pixel to `CUTOUT`, else `SOLID`. There is no per-block RenderType table any more.
- **Parts and variants.** `BlockStateModel.collectParts(random, parts)` handles multipart and weighted variants, drawing from `RandomSource` seeded with `BlockState.getSeed(pos)`.
  - Stone has 4 variants (mirrored and rotated by 180°). `grass_block` has 4 y-rotations.
  - Adjacent identical blocks therefore have different UV orientations.
- **Offsets.** `BlockState.getOffset(pos)` adds random XZ or XYZ offsets (flowers, short grass, bamboo and so on).
- **Culling.**
  - A quad with a cullface is emitted if `Block.shouldRenderFace(state, neighbor, dir)`. That function checks face-occlusion `VoxelShape`s and `skipRendering` (glass on glass, the same fluid, and so on), using a 256-entry per-thread `OCCLUSION_CACHE`.
  - Quads from `getQuads(null)` (no cullface) are always emitted.
- **Model JSON.** Elements may use `rotation`, `light_emission` (0–15) and `shade_direction_override`. `shade:false` no longer exists; the override replaces it. Models may set `ambientocclusion:false`.
- **Per-quad pipeline** in `ModelBlockRenderer`:
  1. `BlockModelLighter.prepareQuadAmbientOcclusion` or `prepareQuadFlat` fills `QuadInstance`: 4 colors and 4 light values.
  2. `putQuadWithTint` calls `QuadInstance.multiplyColor(tint)` when `tintIndex != -1`.
  3. `putBlockBakedQuad` writes the quad. Light comes from `getLightCoordsWithEmission(vertex, lightEmission)`: sky and block light are each raised to at least the element's emission.

### 2.4 Ambient occlusion and light (`net.minecraft.client.renderer.block.BlockModelLighter`)

- **AO path.** AO is used when the AO option is on, `state.getLightEmission() == 0` (so light-emitting blocks are always flat-lit) and the first part has `useAmbientOcclusion()`.
- **Samples.** For each quad, the lighter reads the 4 edge neighbors in the plane in front of the face (the `faceCubic` face moves the base position one block out). A diagonal corner is sampled only if one of its two adjacent edge blocks, one step further along the face normal, `isLightPermeable()`; otherwise edge 0's value stands in.
- **Shade and corner brightness.**
  - Shade is `getShadeBrightness`: 0.2 if the block's collision shape is a full block, else 1.0. Glass-like `TransparentBlock`, barrier, light, mud, soul sand, snow layer and structure void override it.
  - Each corner's brightness is the mean of 4 shades (edge, edge, diagonal, center), so full faces get 5 levels: 0.2, 0.4 … 1.0.
- **Corner light.** `LightCoordsUtil.smoothBlend(n1, n2, n3, center)`:
  - If the center has more than 2 sky or block light, a neighbor that is 0 is replaced by the center, and a neighbor with sky 0 takes the center's sky.
  - The result is `(a+b+c+d) >> 2`, masked per channel (0..240).
- **Partial faces.** When a face does not cover the whole block side (`facePartial`), the corner values are bilinearly re-weighted by the face's min/max extents (`SizeInfo`, `smoothWeightedBlend`).
- **Directional shade.** The color is then scaled by `CardinalLighting.byFace(shadeDirectionOverride ?: dir)`:
  - `DEFAULT`: down 0.5, up 1.0, north/south 0.8, west/east 0.6.
  - `NETHER`: 0.9 / 0.9 / 0.8 / 0.6.
  - The dimension type selects which one applies.
- **Flat path.** One light value per face, taken from the neighbor in the face's direction (from the block itself for non-cubic faces), with color `gray(cardinal shade)`.
- **Light coordinates** (`LightCoordsUtil.getLightCoords`):
  - `emissiveRendering()` blocks (magma and similar) return `FULL_BRIGHT` 0xF000F0.
  - Otherwise block light is raised to the block's own `getLightEmission()`.

### 2.5 Biome tint

- **Lookup.** `BlockColors.getTintSources(state)` returns a list of `BlockTintSource`, indexed by `tintIndex`. `colorInWorld(state, level, pos)` is computed once per block and tint index (`ModelBlockRenderer.computeTintColor` caches it).
- **Result.** One color per block, identical on all 4 vertices, multiplied into the vertex color. Grass and leaves get biome-blended colors. Water uses its fluid model's `tintSource`. Blocks with fixed colors or state-dependent colors (redstone wire, stems) use the same mechanism.

### 2.6 Fluids (`FluidRenderer`, `FluidModel`)

- **Layer.** A `FluidModel` has still, flowing and optional overlay materials; its layer comes from sprite transparency. Water is `TRANSLUCENT`; lava is `SOLID`.
- **Top face.**
  - Corner heights are averaged over neighbors (`calculateAverageHeight`) and inset by 0.001.
  - UVs are rotated by the flow angle, `atan2(flow)`, when the fluid is flowing.
  - A back face is added if `shouldRenderBackwardUpFace`.
- **Bottom and sides.**
  - The bottom face is inset 0.001 when drawn.
  - Sides are inset 0.001 from the block edge and use V ranges scaled by height.
  - Next to `HalfTransparentBlock`/`LeavesBlock` the sides use the `water_overlay` sprite with no back face; otherwise they are double-sided.
- **Lighting.** No AO. Light is `max(light(pos), light(pos.above()))` (the bottom face uses the block below). Tint is one color per block, scaled by cardinal shade (sides use `up × north` or `up × west`).

### 2.7 Animated textures, layers and translucency sorting

- **Animated sprites never touch meshes.** `TextureAtlas` keeps a `SpriteContents.AnimationState` per animated sprite; `drawToAtlas` renders the current frame into every atlas mip level on the GPU (`ANIMATE_SPRITE_BLIT`, `ANIMATE_SPRITE_INTERPOLATE`). The 26.3 jar has 1,405 block PNGs, 105 of them with a `.mcmeta`. A replacement only has to sample the live block atlas.
- **Layers.** `ChunkSectionLayer` has three layers, grouped by `ChunkSectionLayerGroup` into OPAQUE (SOLID, CUTOUT) and TRANSLUCENT.

  | Layer | Alpha discard | Blending |
  |---|---|---|
  | `SOLID` | none | none |
  | `CUTOUT` | below 0.5 (`ALPHA_CUTOUT` in `RenderPipelines.CUTOUT_TERRAIN`) | none |
  | `TRANSLUCENT` | below 0.1 | `BlendFunction.TRANSLUCENT` |

  All three use `DepthStencilState.DEFAULT` (GEQUAL, depth write on), so **translucent terrain writes depth**.
- **Sorting within a section.**
  - `MeshData.sortQuads` keeps each quad's centroid in a `CompactVectorArray` (12 bytes per quad of CPU heap, kept for re-sorting). The centroid is the midpoint of vertices 0 and 2, relative to the section.
  - `VertexSorting.byDistance(camera − sectionOrigin)` does a mergesort on squared distance, farthest first, and writes a custom index buffer.
- **Re-sorting.** `LevelRenderer.scheduleTranslucentSectionResort` queues a `ResortTransparencyTask`:
  - for every section within 32 blocks (octree `isClose`) whenever the camera's block position changes;
  - otherwise when `TranslucencyPointOfView` (the camera's section offset clamped to {−1, 0, 1}³) changes, or it is axis-aligned and the block position changed;
  - with a round-robin budget of `max(visible/8, 15)` sections per frame.
- **Across sections.** `visibleSections` comes out of the octree near to far. `prepareChunkRenders*` reverses the translucent draw groups and draws, which gives back-to-front order.
- **With OIT.** When `improvedTransparency` (OIT) is on, `respectTranslucentOrder` is false and no re-sorts are scheduled.

### 2.8 Upload

1. The worker calls `UberGpuBuffer.addAllocation`, which copies into the shared `StagingBuffer` (98 MiB; persistently mapped when `writeToBufferIsSlow`). If staging is full, the worker spins (`Thread.onSpinWait`).
2. After each frame the render thread runs `uploadTerrainBuffersToGpu` under `copyLock`:
   - It allocates with TLSF in the layer's heaps: 128 MiB vertex heaps aligned to 28 bytes, 32 MiB index heaps aligned to 8. TLSF granularity is 32 bytes, plus padding when the alignment isn't a power of two.
   - It creates a new heap when an allocation doesn't fit and frees one fully empty heap per call.
3. `checkSectionMesh` publishes the new `CompiledSectionMesh` only once every layer's vertex and index data has been uploaded, so there is no partial swap.
4. `RenderSection.uploadedTime` is set once, at the first mesh. It drives the fade-in.

## 3. Draw path

- **Preparation.** In `LevelRenderer.render`, `prepareChunkRendersIndirect` runs before the frame graph when MDI is available: `maxDrawIndirectDrawCount > 0`, `nonZeroFirstInstance`, no `multiDrawIndirectHasKnownIssues`, and `Minecraft.multiDrawIndirect` (default true). Otherwise `prepareChunkRenders` runs.
- **Main pass.** `executeSolid` runs `renderGroup(OPAQUE)` (SOLID, then CUTOUT), then the solid features: entities and block entities.
- **Translucency, classic path.** In the same render pass: translucent features, **translucent terrain**, `executeTranslucentAfterTerrain`, clouds, weather, world border.
- **Translucency, OIT path.** `executeOit` draws terrain once per `OitStage`:
  1. `DEPTH_BOUNDS`: alpha only, into an `RGBA32F` target.
  2. `TRANSMITTANCE`: 2 × `RGBA16F`, wavelet rank 2 with 8 coefficients.
  3. `ACCUMULATE`: `RGBA16F`.

  Then a full-screen `OIT_COMPOSITE` runs, so translucent terrain is rasterized **3 times**.

**`LevelRenderer.extractSectionDrawGroups`** runs per frame on the render thread, under `copyLock`:

- **Per section.** For each visible section, it adds one `ChunkSectionInfo(x, y, z, visibility)` (16 bytes) if any layer has data.
- **Per layer.** For each of the 3 layers:
  - `getSectionDraw`, plus two HashMap lookups (vertex and index `TlsfAllocator.Allocation`);
  - a hash of (heap buffer, index buffer, index type), used to group draws;
  - one `IndexedDraw(indexCount, 1, firstIndex, baseVertex = offset/28, baseInstance = sectionInfoIndex)`.
- **Buffers.** Commands go to `DynamicGpuData.writeChunkSectionCommands`, a mapped ring of 20-byte `DrawIndexedIndirect` records. Section infos go to `writeChunkSectionsInstanced`, a non-mapped instanced vertex stream at binding 1.
- **Grouping.** Solid and cutout draws group by hash. Translucent draws start a new group whenever the hash changes, which keeps the order.

**`ChunkSectionsToRender.DrawIndirect.render`:**

- Per layer it sets the pipeline once and binds binding 1.
- Per group it binds the vertex heap and the index buffer (the shared quad index buffer when the layer has no custom one), then calls `drawIndexedIndirect(slice, count)`, split at `maxDrawIndirectDrawCount`.
- `baseInstance` selects the section's `ChunkPosition`/`ChunkVisibility`. There are no per-section uniform changes.
- The fallback, `DrawSeparate`, issues one `drawIndexed` per section-layer and binds a 16-byte `ChunkSection` UBO for each.

**Bindings.**

- `TerrainUniform` (`ModelViewMat` mat4 plus `TextureSize` ivec2)
- `Globals`
- `Projection`
- `Fog`
- `Sampler0`: the block atlas, with `chunkLayerSampler` (LINEAR/LINEAR, clamp; anisotropy when filtering is ANISOTROPIC; 0–4 mip levels, default 4)
- `Sampler2`: the 16×16 `RGBA8` `Lightmap`, re-rendered each frame by `lightmap.fsh`, sampled with a linear clamp

**Draws per frame.**

- **MDI calls.** One call per layer per heap group: normally 3 (SOLID, CUTOUT, TRANSLUCENT), plus one for each additional 128 MiB heap in use (the vertex heap holds about 1.2 M quads), and ×3 for translucent under OIT.
- **Sub-draws.** One per visible non-empty section-layer.
- **View area.** It holds (2·RD+1)² × 24 sections in the overworld: 15,000 at RD 12 and 101,400 at RD 32.
- **Measured on our backend** (vanilla section lists, our `occlusionCulling` off):
  - RD 12: 3.0 calls and about 1.6–1.7 k sub-draws, 0.12–0.15 ms of encoding (`early_rd12`, `late_a1`, `lazy_c2`).
  - RD 32 at ground level: 4.3 calls, 6.1 k sub-draws, 0.43 ms (`nocc_g32b`).
  - RD 32 at the high orbit: 5.1 calls, 8.1 k sub-draws, 0.58 ms (`nocc_h32b`, `late_rd32`). The extra calls are presumably from additional 128 MiB heap groups (this is inferred, not traced).
  - With our section occlusion culling on, sub-draws drop to 2.1 k at ground level and 5.6 k at the high orbit.
- **On Metal.** Our `mmc_rp_draw_indexed_indirect` (`Sources/MetalMCNative/Backend.swift`) turns 16 or more records into a single `MTLIndirectCommandBuffer` execute. Below that it loops with one `drawIndexedPrimitives(indirectBuffer:)` per record.

**CPU work per frame, per visible section:**

- the octree frustum visit (`SectionOcclusionGraph.addSectionsInFrustum` → `Octree.visitNodes`);
- a dirty-state lookup;
- a walk of its block-entity list (block entities are skipped while visibility < 0.3);
- 3 × (EnumMap get + 2 HashMap gets + a record allocation) in `extractSectionDrawGroups`;
- a translucency point-of-view check for nearby sections and the round-robin sweep.

The occlusion BFS (`SectionOcclusionGraph`, smart cull beyond 60 blocks) rebuilds fully on a background thread and propagates incrementally on the render thread as sections compile.

## 4. Terrain shaders (`core/terrain.vsh`, `core/terrain.fsh`)

**Vertex shader, per vertex:**

- **Position.** `pos = Position + (ChunkPosition − CameraBlockPos) + CameraOffset`.
  - `Globals` supplies `CameraBlockPos` (the camera's integer floor) and `CameraOffset` (floor − camera, as a float). Doing the subtraction in integers avoids float error far from the origin.
  - `gl_Position = ProjMat * ModelViewMat * pos`. `ModelViewMat` is rotation only, from `TerrainUniform`.
- **Fog distances.** `sphericalVertexDistance = |pos|` and `cylindricalVertexDistance = max(|pos.xz|, |pos.y|)`.
- **Lighting per vertex.** `vertexColor = Color * texture(Sampler2, clamp(UV2/256 + 1/32, 1/32, 31/32))`. Because the lightmap is sampled per vertex and then interpolated across the two triangles, AO and light gradients follow the fixed (0,1,2)/(2,3,0) split.
- **Fade.** `chunkVisibility = mix(1, ChunkVisibility, clamp((|pos| − 16)/16, 0, 1))`: sections within 16 blocks never fade, and the fade is fully applied beyond 32.
  - `ChunkVisibility = min(1, (now − uploadedTime)/chunkSectionFadeInTime)`. The default fade time is 0.75 s (range 0–2 s).
- **Without MDI**, `ChunkPosition` and `ChunkVisibility` come from the `ChunkSection` UBO instead of the vertex stream (`#ifndef MULTIDRAW_TERRAIN`).

**Fragment shader, per pixel:**

1. **Texel sample.** `sampleNearest(Sampler0, uv, 1/TextureSize)` is a texel-center-snapped `textureGrad` that anti-aliases pixel edges according to on-screen texel size. With `UseRgss`, it is instead RGSS: 4 rotated-grid taps at two mip levels, blended with the nearest-texel sample.
2. **Color.** The sample is multiplied by `vertexColor`.
3. **Fade.** `color = mix(FogColor·(1,1,1,a), color, chunkVisibility)`.
4. **Alpha test.** Discard if `a < ALPHA_CUTOUT`.
5. **Fog.** `apply_fog` takes the maximum of the linear environmental fog (spherical distance) and the linear render-distance fog (cylindrical distance), then mixes toward `FogColor.rgb` by `fog × FogColor.a`.
6. **OIT variants.** `OIT_ALPHA_ONLY` skips the lightmap and color; `OIT_ACCUMULATE` pre-multiplies the fog color by alpha.

Inputs: `Sampler0` (atlas), `Sampler2` (lightmap), UBOs `Globals`, `Projection`, `Fog`, `TerrainUniform`, and optionally `ChunkSection`. There are no normals, lights or time inputs; animation lives in the atlas.

## 5. Bytes per face, and what a replacement must reproduce

| Design | Stored per rendered face | Vertex shader work per face | Notes |
|---|---|---|---|
| **Vanilla 26.3** | 112 bytes of vertices; translucent adds 12 bytes of indices (24 with `INT`) and a 12-byte CPU centroid | 4 vertices, 112 bytes fetched | Grass side = 224 bytes; a double-sided water face = 248 bytes. Minimum GPU reservation is 3 × 128 MiB of vertex heap + 32 MiB of index heap once all 3 layers exist. |
| Sodium-style vertex, about 20 bytes | about 80 bytes | 4 vertices | About 29% less, and still one record per face per vertex. |
| 8 bytes per face, vertex pulling (`vid >> 2`) | 8 bytes, plus about 8 KiB of side data per non-empty section | 4 vertices, one fetch | Needs per-section volumes (next list). About 14× less face data than vanilla. |
| 16 bytes per face, vertex pulling | 16 bytes | 4 vertices | Holds vanilla's per-corner results directly (bit budget below). About 7× less. |
| 8-byte merged rectangle (our LOD) | 8/k bytes, where k is faces merged | 4 per rectangle | Exactness requires per-pixel reconstruction of UV, variant, AO, light and tint (below). |

**Side data for the 8-byte variant:**

- an 18³ light volume, 1 byte per cell (sky and block nibbles): 5.7 KiB;
- 18³ × 3 flag bits (full collision shape, `isLightPermeable`, `isSolidRender`): 2.2 KiB;
- a tint table.

**Bit budget for one vanilla-identical full-cube face** (*estimate*):

| Field | Bits |
|---|---|
| position (4+4+4) | 12 |
| direction | 3 |
| sprite id | 16 |
| variant UV transform (rotation, mirror) | 3 |
| element emission | 4 |
| shade override | 3 |
| AO (4 × 3) | 12 |
| smooth light (4 × (6 + 6); steps of 4 in 0..240 give 61 levels) | 48 |
| tint (in-face palette index, or a per-block tint lookup) | 8–24 |
| **total** | **about 110–125, so 16 bytes** |

Dropping the stored AO and light (computing them in the shader from the volumes) brings this to about 45 bits, which fits in 8 bytes.

Non-cube quads (slabs, stairs, cross models, rotated elements, offset plants, sloped fluid tops) need arbitrary positions and UVs. They need a fallback per-vertex format (about 16–20 bytes per vertex, or 64–80 bytes per quad).

**Greedy merging** (our LOD's 8-byte rectangles) can only be bit-identical if the shader rebuilds, per block cell:

- the atlas UV, wrapped inside the sprite rectangle, with `textureGrad` using per-block derivatives so mip selection and `sampleNearest`/RGSS behave identically at block seams;
- the random variant rotation or mirror;
- the tint;
- AO and light as **per-vertex Gouraud values over vanilla's triangle split**, not bilinear.

In practice merged quads suit near-identical visuals at distance, not exact matching.

### Exhaustive checklist for a visually identical near-chunk mesher

1. **All `BakedQuad`s from `BlockStateModel.collectParts`**: multipart, weighted variants with the exact `RandomSource`/`getSeed` sequence, x/y rotations, uvlock, mirrored models, element rotations with rescale, sub-block UV rectangles, coordinates outside 0..1.
2. **Random offsets** (`BlockState.getOffset`), applied to positions but not to AO sampling positions.
3. **Face culling identical to `Block.shouldRenderFace`**: occlusion shapes, `skipRendering`, cutoutLeaves culling between leaves. Only quads with a cullface are culled.
4. **Coplanar overlays at bit-identical depth.** The grass side overlay in CUTOUT sits over dirt in SOLID and relies on GEQUAL passing at equal depth, so positions must be computed with the same float math.
5. **Layer per quad from sprite pixels in its UV rectangle** (or `forceTranslucent`). Alpha cutoff is 0.5 for cutout and 0.1 for translucent. Solid has no alpha test.
6. **cutoutLeaves off**: leaves in SOLID (opaque, showing the texture's RGB under alpha-0 pixels), with leaf-to-leaf faces culled.
7. **Smooth lighting and AO exactly as `BlockModelLighter`**: `faceCubic`/`facePartial` weighting, the `isLightPermeable` diagonal rule, the center substitution in `smoothBlend`, shade-brightness overrides, AO off for emitting blocks and for `ambientocclusion:false`, and the flat path when the AO option is off.
8. **Per-vertex light and color**, interpolated across the fixed triangulation (0,1,2)(2,3,0), with the lightmap sampled per vertex through a linear filter.
9. **`CardinalLighting` per dimension**, and per-element `shade_direction_override`.
10. **Emission**: `emissiveRendering` → `FULL_BRIGHT`; element `light_emission` → max(sky, block, emission); a block's own emission raises its block light.
11. **Tint through `BlockTintSource`** (biome resolvers with blend radius 0–7, fixed and state-dependent colors). It is per block and multiplied into RGBA8, where quantization should match (`ARGB.multiply`, `ARGB.scaleRGB`, `ARGB.gray`).
12. **Fluids**: corner heights, flow-rotated UVs, 0.001 insets, back faces, overlay sprite next to glass and leaves, light as max(self, above), no AO, water TRANSLUCENT and lava SOLID.
13. **The live block atlas**: animated and interpolated sprites, mip chain with `alphaCutoffBias`, anisotropy option, `sampleNearest`/RGSS sampling and `TextureSize`.
14. **Translucency**: sorted per quad within a section and back to front across sections, with the same re-sort cadence and translucent depth writes. When `improvedTransparency` is on, it must feed the 3-stage wavelet OIT (`OIT_TERRAIN*` pipelines) and the boat water mask (`executeOitWaterMask` / `rendertype_water_mask`).
15. **Fog** (spherical environmental plus cylindrical render-distance) and **chunk fade-in**: per-section first-upload time, the 16–32-block ramp, a mix toward fog color, and `chunkSectionFadeInTime`.
16. **Section visibility semantics used elsewhere.**
    - Entities are hidden until their section is compiled and its visibility is at least 0.3 (`isSectionCompiledAndVisible`), and block entities are skipped below 0.3.
    - The octree, the `VisGraph`-based occlusion and frustum culling feed `visibleSections`.
    - `playerCompiledSectionCallback` must also keep working.
17. **Update latency.** Nearby or player-caused edits compile synchronously (`prioritizeChunkUpdates`), and meshes swap atomically once all layers are uploaded.
18. **Transient blocks.** `ClientboundAddTransientBlockPacket` blocks (1 s time to live) are drawn as moving blocks until the section recompiles (`removeTransientBlocksInSection`).
19. **Not in chunk meshes, but must still draw correctly against the replacement's depth**: block entities (chests, signs, banners, beds, shulker boxes, skulls, bells, conduits, decorated pots, enchanting tables, lecterns, campfires, spawners, end portal/gateway, beacon beams, copper golem statues and so on), the crumbling overlay, the block outline, moving pistons, falling blocks, item frames, particles, clouds, weather and the world border.
20. **Reverse-Z** (GEQUAL, clear 0.0), camera-relative integer positions, the wireframe debug pipelines, and ordering relative to solid features (terrain is drawn before entities in the same pass).

## 6. Changes in 26.3 that matter for replacing the chunk renderer

- **Renderpearl.** `com.mojang.renderpearl` is the GPU API: `api` (GpuDevice, CommandEncoder, RenderPass, RenderPipeline, BindGroupLayout, GpuFormat), `frontend` (validation, `GlslCompiler` → SPIR-V) and `backend` (opengl, vulkan). Terrain pipelines are declarative `RenderPipeline`s (`TERRAIN_SNIPPET`, `MULTIDRAW_TERRAIN_SNIPPET`, `OitPipelineSet`), and the `RENDERPEARL_DEPTH_IS_ZERO_TO_ONE` define exists. Our Metal backend plugs in at `backend/api`.
- **Terrain meshes live in shared per-layer uber buffers** (`UberGpuBuffer` + TLSF, 128 MiB heaps) instead of per-section VertexBuffers. They are drawn with **multi-draw-indirect** plus an instanced per-section stream that needs `nonZeroFirstInstance`, with `DrawSeparate` as the fallback.
- **The `BLOCK` vertex is 28 bytes** with no normal. Previously it was 32 bytes including the normal and padding.
- **Three layers, decided per quad from texture content.** The older five RenderType chunk layers (including cutout_mipped and tripwire) and the per-block render-type map are gone.
- **Reverse-Z depth.**
- **Order-independent transparency** (`improvedTransparency`, default off) draws translucent terrain in 3 stages.
- **Chunk section fade-in** (`chunkSectionFadeInTime`).
- **Pixel-art texture sampling** in the shader (`texture_sampling.glsl`, RGSS option).
- **Sprite animation on the GPU.**
- **`CardinalLighting`** per dimension type.
- **Model JSON**: `shade_direction_override` and `light_emission`.
- **Rendering pipeline structure.** Render state is extracted in `LevelExtractor` → `LevelRenderState`, and passes go through `FrameGraphBuilder`. Server-driven transient blocks are new.

For our replacement:

- **Where to hook.** The natural cut points are `SectionCompiler.compile` (mesh production), the `UberGpuBuffer`/`addSectionBuffersToUberBuffer` path (storage), and `ChunkSectionsToRender.renderGroup`/`renderOit` (drawing). Everything else can keep running: `VisGraph`, the occlusion graph, `visibleSections`, fade timing and block-entity lists.
