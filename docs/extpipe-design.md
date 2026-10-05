# External pipeline (`METALMC_EXTPIPE=<dir>`, `-PextPipe=<dir>`)

**Status (2026-10-05):** a generic plug-in point for a shader pipeline described outside this repository: OptiFine/Iris
style programs (translated to MSL elsewhere), their render targets and passes, drawn from vanilla's own level draws on
our backend. Off by default; nothing changes without the variable. Run in game at 3456 x 2234 with a large path-traced
OptiFine pack translated privately (its gbuffers programs for terrain, water, sky, entities, particles and weather, 29
full-screen passes at half-resolution with exact ping-pong and flips, the shadow pass with its geometry shader as vertex
expansion and an 8192^2 shadow and voxel atlas): its sky, clouds, sun shadows, water and path-traced GI from block
lights show as the pack intends, at 11-20 fps; with a `lod` program our LOD's terrain to the horizon goes through the
pack's G-buffer and is shaded by it too. Such a pack needs safe math (`"mathMode": "safe"`): with fast math its
TAA history filled with NaNs and the frame went black. Code: `Sources/MetalMCNative/ExtPipe.swift` (with
hooks marked "ExtPipe hook" in `Backend.swift`), `metalmc.backend.MetalExtPipe` (uniforms, bridge),
`metalmc.extpipe.BlockIds` and the mixins in `metalmc.extpipe.mixin`; `tools/extpipe_check.swift` runs a description
offline. No pack's code, shaders or constants live here: a description and its MSL stay wherever their license allows.

## What a frame does

1. **Frame start** (`LevelRenderer.render`, head): Java computes the standard uniforms (below) and calls
   `mmc_ext_frame_begin`. The native side polls the programs' MSL files (reloading the ones that changed, keeping the old
   one if a reload fails) and `pipeline.json` itself, (re)allocates the targets at the main target's size, clears the
   targets that clear every frame (both copies), and writes the frame uniform blocks.
2. **G-buffer passes**: vanilla's sky pass (`SkyRenderer.render`) and its main pass ("Main" in
   `LevelRenderer.addMainPass`) are marked just before they open (`mmc_ext_redirect`); `mmc_pass_begin` then opens the
   description's G-buffer instead: its attachment targets (current copies, loaded) and its depth target (GL depth,
   cleared to 1 each frame). Inside it:
   - `setPipeline`: vanilla's pipeline name is looked up in the routes (exact, or a prefix ending in `*`). No route, or a
     route without a program: the pipeline's draws are skipped. Otherwise the program's pipeline state for that vanilla
     pipeline (its vertex layout) is built once and set, with vanilla's depth test turned to GL's convention
     (`GEQUAL` -> `LEQUAL`) and vanilla's culling.
   - Bindings: vanilla's `Sampler0`/`Sampler2` go to the `gtexture`/`lightmap` slots (with the description's sampler
     for `gtexture` if it gives one); vanilla uniform buffers the program asks for by name (`vanillaBuffers`) are bound at
     its indices; vertex buffers stay where vanilla binds them (Metal buffer 30 - slot).
   - Draws: the per-draw block is written first, from vanilla's transforms as bound for that draw (`DynamicTransforms`'
     model-view with its model offset and texture matrix, or `TerrainUniform`'s model-view and atlas size), the frame's
     GL projection, OptiFine's lightmap texture matrix, and the route's alpha-test reference and render stage.
   - Each program's fragment output `i` goes to `outputs[i]`; for routed programs the MSL's `[[color(i)]]` is moved to
     that target's G-buffer attachment when it compiles. Attachments a program doesn't write keep their contents (write
     mask off).
3. **Deferred** (`executeClassicTransparency`, head, `mmc_ext_translucent`): the G-buffer encoder closes, the depth
   copies marked `translucent` are made, the `deferred` passes run, and the G-buffer opens again for the translucent
   geometry; Java re-applies the pass's pipeline and uniforms. (With improved transparency there's no such point: the
   deferred passes run at the end of the main pass, after the translucent geometry.)
4. **Composite and final** (the main pass ends): the `composite` passes, then `final`, whose `screen` output is vanilla's
   main color target. Vanilla's later passes (outlines, the hand, the GUI) draw over it as usual.

## The shadow pass (`shadow` in the description)

Right before the deferred passes, the terrain draws of the routes it lists, as vanilla issued them in the main pass
(multi-draw-indirect records, read on the CPU: unified memory), are drawn again through the shadow program from the
shadow camera (`gl_ModelViewMatrix` = `shadowModelView`, `gl_ProjectionMatrix` = `shadowProjection`), culling off,
into `colors` and `depth` (cleared each time). A geometry shader runs as vertex expansion (`expand` vertices per input
triangle, a non-indexed draw): the program gets `{triangles, index mode 2, 0, base vertex, base instance, render stage,
0, 0}` (uint/int) at `paramsBuffer` per section draw, vanilla's vertex buffers at 30 and 29 and its `vanillaBuffers`,
and pulls the vertices itself; a section draw's first index (vanilla's facing buckets) becomes a quad offset in the base
vertex. Then `depthCopy` gets the depth, and a depth target with mips gets its chain (each level the mean of four texels,
a render pass per level: Metal's mipmap generation doesn't do depth formats).

Only what the camera's draws hold is there: sections outside the view frustum (or culled) cast no shadow and aren't in a
voxel volume a pack builds from this pass, and with facing culling only the faces turned to the camera are.
`-PextPipeNoCull=1` (`METALMC_EXTPIPE_NOCULL=1`) turns vanilla's view-frustum culling off, so the main pass, and with it
the shadow pass, has every section in range (at the cost of drawing the ones behind the camera).

Translucent terrain (`translucentRoutes`, water) is drawn after the deferred passes, so the shadow pass draws the last
frame's: when vanilla draws it, each section's indirect arguments and its section-stream entry (chunk position) are
copied on the CPU, and the next frame's shadow pass draws its quads from the base vertex (vanilla's translucent index
buffers are sorted, so a first index isn't a quad offset there) with that frame's camera (`Globals`).

A pack's shadow pass that voxelizes geometry (SEUS PTGI: each triangle's corners moved into their face's voxel along
the tangent and bitangent) depends on OptiFine's `at_tangent.w`: `cross(at_tangent.xyz, normal) * w` is the direction of
increasing v. With the other sign the corners leave the face, faces land in the wrong voxel or none, and GI rays start
inside solid voxels (black faces).

## Our LOD in the G-buffer (`lod` in the description)

`mmc_lod_draw` runs after vanilla's solid terrain, inside the G-buffer pass; with a `lod` entry it draws there: every LOD
quad (water too, all opaque) with `program`, whose vertex function reads the buffers Lod.swift binds for its own `lod_vs`
(quads 18, `LodUniforms` 19, xforms 20, material colors 21, AO offsets 22) plus the description's frame block, and
whose fragment function writes the G-buffer attachments in order (`seamFragmentEntry`, from the same library: the
variant for tiles that overlap vanilla's sections, with the seam bitmap at fragment buffer 21). Depth test less-equal
(GL's convention). No far field there (its levels are drawn as quads), no occlusion boxes or fades. Every program gets
the LOD's material ids as `MMC_MAT_<NAME>` macros (MetalMCCore's `Mat`). Run with the LOD on
(`LAB_FAR=8192`, `-Plod=...`): vanilla's far plane is pushed past the LOD, so `gbufferProjection` covers it too.

## Full-screen passes

Each pass draws OptiFine's quad ((0,0)-(1,1), the same numbers as texture coordinates; attributes named `quad`) with
`gl_ModelViewMatrix` = identity and `gl_ProjectionMatrix` = ortho(0,1,0,1,-1,1). Iris's ping-pong rules: a pass reads
the current copy of every target and writes the other copy of its outputs, which then flip; `flipAfter` flips without a
write. Every written attachment loads (passes may cover only part of a target, relying on what's outside). `mipsBefore`
builds the named targets' mip chains first. Custom textures override a texture name per stage (`gbuffers`, `deferred`,
`composite`, `final`, `*`) until a pass of that stage writes or flips that name. Targets whose `clear` is absent are
history: they carry their latest copy into the next frame.

## The description (`<dir>/pipeline.json`)

| key | what |
|---|---|
| `uniformBlocks` | `{buffer, size, scope: frame\|draw, members: [{name, semantic?, type, offset}]}`; a member holds `semantic` (or its `name`): a standard uniform (frame) or a per-draw value |
| `textureSlots`, `gameSamplerSlots` | texture name -> Metal texture index; game texture name -> sampler index |
| `targets` | `{name, format, size?\|scale?, clear?, clearFog?, clearMode?: frame\|once, copies?, mips?}` |
| `customTextures` | `{stage, name, file?\|game?}` (PNG relative to the description, or `minecraft:textures/atlas/blocks.png`) |
| `constantTextures` | `{name, value: [r,g,b,a]}` (1x1, e.g. default normal and specular maps) |
| `programs` | name -> `{vertex, vertexEntry, fragment?, fragmentEntry?, outputs, attributes?, vanillaBuffers?, blend?, mathMode?}` |
| `gbuffers` | `{attachments, depth, depthCopies: [{when: translucent\|end, target}], routes: [{pipeline, program?, alphaTest?, renderStage?, entityId?}], samplers?}` |
| `passes` | `[{stage, program, mipsBefore?, flipAfter?, enabled?}]` in order |
| `shadow` | `{program, colors, depth, depthCopy?, expand?, paramsBuffer?, routes, translucentRoutes?, alphaTest?, renderStage?, enabled?}` (above) |
| `lod` | `{program, seamFragmentEntry?}`: our LOD's quads in the G-buffer (above) |
| `consts` | pack constants Java needs for the standard uniforms (`sunPathRotation`, `shadowDistance`, `shadowIntervalSize`, `wetnessHalflife`, `drynessHalflife`, `eyeBrightnessHalflife`) |

Vertex attributes: `attributes` maps a vertex function's attribute location to a vanilla vertex element name
(`Position`, `Color`, `UV0`, `UV1`, `UV2`, `Normal`, ...), `quad`, or `zero`; locations not listed (or that vanilla's
format lacks) read zeros ((0,0,0,1) for four floats). A vertex function without `[[stage_in]]` gets no descriptor and
can pull vanilla's vertices itself (buffers 30 and 29, `[[vertex_id]]`, `[[base_vertex]]`, `[[instance_id]]`).

Block ids: a `block.properties` next to the description (OptiFine's format, preprocessed) gives terrain vertices their
block's id (`metalmc.extpipe.BlockIds`): bits 0-7 in the color's alpha, 8-14 in the block light's high byte, bit 14 of
the sky light set; light values keep their low bytes. Only the pipeline's own terrain programs read them.

## Standard uniforms (Java, `MetalExtPipe`)

OptiFine's names and meanings, as Iris computes them: `gbufferModelView` (vanilla's view rotation; positions are
camera-relative), `gbufferProjection` (vanilla's projection with view bobbing, turned to GL's depth range with near 0.05
and vanilla's far plane), their inverses and previous-frame values, `cameraPosition`, `previousCameraPosition`, the
shadow camera (`shadowModelView`: 100 back along the sun or moon direction with `sunPathRotation`, snapped to
`shadowIntervalSize`; `shadowProjection`: ortho +-`shadowDistance`, near 0.05, far 256), `sunPosition`, `moonPosition`,
`upPosition`, `shadowLightPosition` (view space, length 100), `sunAngle`, `shadowAngle`, `skyColor`, `fogColor`,
`fogStart`, `fogEnd`, `eyeBrightness`, `eyeBrightnessSmooth`, `isEyeInWater`, `rainStrength`, `wetness`,
`thunderStrength`, `moonPhase`, `worldTime`, `worldDay`, `frameCounter` (wraps at 720720), `frameTime`,
`frameTimeCounter` (wraps at 3600 s), `viewWidth`, `viewHeight`, `aspectRatio`, `near`, `far`, `nightVision`,
`blindness`, `darknessFactor`, `heldBlockLightValue(2)`, `screenBrightness`, `eyeAltitude`, `hideGUI`. Per draw:
`gl_ModelViewMatrix(Inverse)`, `gl_ProjectionMatrix(Inverse)`, `gl_ModelViewProjectionMatrix`, `gl_TextureMatrix0`,
`gl_TextureMatrix1`, `gl_NormalMatrix`, `alphaTestRef`, `renderStage`, `entityId`, `atlasSize`, `entityColor`,
`colorModulator`.

## Running and debugging

- `-PextPipe=<dir>` (needs the Metal backend; classic transparency; no `METALMC_EXP` look switches: near chunks would
  take terrain out of vanilla's draws, lit mode and our anti-aliasing would work on the finished frame). Without a
  `lod` entry our LOD, far field and section occlusion test skip the G-buffer pass (`-Plod=0` saves their work). In a lab
  session: `LAB_EXP=none LAB_FAR=0 bash tools/bench/lab.sh <label> -PextPipe=<dir> -PocclusionCulling=0 -Ptaa=false`
  (`LAB_FAR=8192` with a `lod` entry).
- `-PbenchTour=shader`: noon overview and ground view, afternoon, sunset, a pool of water built in the sky, a closed room
  lit by torches and glowstone, night; screenshots in `mod/run/screenshots/<label>-tour-*.png`.
- `METALMC_EXTPIPE_VIEW=<target>[.a][@<pass>][:scale],...` (`-PextPipeView=`) shows targets in a grid over the screen
  instead of the frame: each as it was after that pass (`@gbuffers`: when the G-buffer was done; none: at the end of the
  frame), its alpha with `.a`, absolute values times the scale; NaN magenta, infinity cyan, depth as (1 - depth)^(1/4);
  `frame` is a cell with the frame itself. While the game runs, the first line of `<dir>/view.txt` replaces it (polled
  with the programs; empty or no file: the frame).
- `tools/extpipe_check.swift`: loads and compiles a description and runs frames with an empty G-buffer offline
  (`EXTCHECK_VIEWS=` for the same views).
