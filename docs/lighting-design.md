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

## Sky and atmosphere (prototype, `METALMC_EXP=sky`, 2026-09-30)

`Sources/MetalMCNative/Sky.swift`, with `metalmc.sky.Sky` and the mixins in `metalmc.sky.mixin` on the Java side. Off by
default; everything below applies only with the experiment on.

- **Model.** Hillaire's (EGSR 2020): Rayleigh and Mie scattering and ozone absorption over an Earth-sized planet (his
  coefficients; ground at sea level, atmosphere 100 km thick), precomputed on the GPU into four tables:
  - transmittance, 256 × 64 (Bruneton's parameterization);
  - multiple scattering, 32 × 32, one 64-thread threadgroup per texel summing 64 directions (his f_ms series);
  - sky view, 192 × 108, from the zenith down to the planet's horizon, rows squeezed toward the horizon;
  - aerial perspective, 32 × 64 × 32: columns the angle from the sun around the vertical, rows the sine of the
    elevation squeezed toward the horizontal (terrain a few km away and beyond is all within a degree or two of it),
    slices the distance, quadratic out to 400 km. RGB in-scatter and RGB transmittance, in two textures.
- **Flat world, round planet.** A camera-relative position p (blocks, which are metres) is the point (0, R + altitude, 0)
  + p / 1000 km from the planet's center, the altitude being the camera's height above sea level. Terrain at the horizon
  and the sky just above it are the same rays through the same air, so far terrain fades into exactly the sky behind it.
  Rays toward terrain below sea level carry on through air at the ground's density.
- **Below the planet's horizon** (0.3° down from 87 m up) the sky is the horizon's color, like vanilla's fog color there.
  The physical sky, air down to a black planet, showed as a dark band wherever terrain ended short of the horizon.
- **Sun.** Vanilla's direction, (−sin a, cos a, 0) for its sun angle a. The disk is 0.6° in radius (the ray-traced
  shadows' penumbra; `METALMC_SUNSIZE` in degrees; the real sun's is 0.27°, vanilla's square about 17°), seen through the
  transmittance, limb-darkened (Hestroffer and Magnan's power law at 680, 550 and 440 nm) and normalized so its mean is
  the sun's illuminance. The terminator is soft over the disk's size.
- **Units and exposure.** Scene-linear light where vanilla's white is 1 (vanilla's colors decoded from sRGB). The sun's
  illuminance is 12 in those units (`METALMC_SKYEXPOSURE`), which puts the afternoon zenith at about sRGB (71, 102, 151),
  a deeper blue than vanilla's, with a near-white horizon. Eye adaptation from the sun's height (no histogram): up to 3.5
  stops as the sun goes from about 15° up to 9° under, so dusk and dawn keep their colors; vanilla's lightmap dims the
  terrain over the same span. `METALMC_SKYHAZE` scales the distances aerial perspective sees (2: twice as hazy).
- **Night.** The moon and stars stay vanilla's (drawn over our sky in the same pass). A faint blue-gray glow stands in
  for the moon's scattering and fades in as the sun goes down (full at 12° under); at night it also replaces the haze on
  far terrain.
- **Rain** (vanilla's rain brightness): 40 times the Mie haze at full rain (visibility from about 200 km to 10), a flatter
  Mie phase (g 0.8 → 0.3), 80% less sunlight, the sky and the haze 75% toward gray, no sun disk. Quantized to 1/32, so a
  rain transition rebuilds the two atmosphere tables 32 times, not every frame.
- **Per frame.**
  1. At the start of `GameRenderer.renderLevel` (`SkyGameRendererMixin`): whether the sky replaces vanilla's this frame
     (the overworld's sky, the camera in air, no blinding effect, no boss fog; anything else keeps vanilla's sky and
     fog). Then the sky view and aerial perspective tables are rebuilt if the sun moved 0.01°, the altitude changed 0.5%,
     or the rain or exposure changed; the other two when the rain did. Vanilla's distance fog, and the LOD's and the far
     field's, which read the same fog data, is pushed out of reach; its render-distance start and end are kept for the
     fade.
  2. At the start of `SkyRenderer.render`, before its pass opens: the sky at a quarter of the resolution on each axis,
     tone mapped and encoded (the sky has no edges, so filtering it up looks the same).
  3. In the sky pass, in place of the sky disc (`SkyRendererMixin`): that sky filtered up, with the sun's disk worked
     out per pixel near the sun, dithered on 8-bit targets. Vanilla's sunrise fan and sun aren't drawn.
  4. After the level (`GameRendererLodMixin`, after the ray-traced shadows): every pixel that isn't sky, from the depth
     buffer, gets aerial perspective (color × transmittance + in-scatter, in linear light), the render distance's edge
     fades into the sky's own color (cylindrical distance, like vanilla's fog), and the tone curve makes it display light.
     With anti-aliasing on, this happens in its resolve as it loads each pixel, after the shadows' shade (a second
     variant of the resolve, chosen by a function constant, so the default one is unchanged); without, in a pass of its
     own that first applies any shadows left for the anti-aliasing. The anti-aliasing's write back into an 8-bit frame is
     then dithered: its 10-bit history averages the sky's own dither away, and smooth gradients banded.
- **Shared with other passes.** `skyShaderHeader` (Sky.swift) has the frame parameters (`SkyFrame`) and the functions:
  `skyAerialPerspective(rel, …)` gives the in-scatter and transmittance for a camera-relative position,
  `skyApplyAerial(linearColor, rel, …)` applies them, `skyLevelColor` does the whole per-pixel job, `skyLuminance` is the
  sky in a direction, and `skyToneMap`, `skyEncode` and `skyDecode` are the output side.

### Integration call sites (not edited here)

The full-screen application already covers every terrain source, so these are optimizations and fidelity steps, not
needed for it to work:

- **The LOD** (`Lod.swift`, `lodShade`, where it returns `mix(color, u.fogColor.rgb, fog * u.fogColor.a)`): prepend
  `skyShaderHeader` to the LOD's shader source, bind this frame's `SkyFrame` (`Sky.shared.frame` with the draw's matrix
  and size) and `Sky.shared.apScatter`/`apTrans` at free slots (fragment buffer 16 and textures 25 and 26 are unused by
  the LOD today), and return `skyEncode(skyApplyAerial(skyDecode(color), in.rel, f, apScatter, apTrans))` instead. The
  LOD's fog is already off while the sky is on.
- **The far field** (`FarField.swift`, the fragment that ends `out.color = float4(mix(color, u.fogColor.rgb, …), 1.0)`):
  the same with its `rel` (buffer 16 and textures 25 and 26 are free there too).
- **Vanilla's chunks** would need their core shaders (translated GLSL) to call the same function, or keep the
  full-screen application.
- If either applies it in its own shader, the full-screen application must skip its pixels (a stencil bit, or a flag
  in the G-buffer once there is one), or they get it twice. The payoff is small with anti-aliasing on (the work rides in
  its resolve) and larger without.
- The deferred-lighting rewrite (rewrite-plan.md, swing 2) is the natural single place: apply it once per pixel there.

## HDR/EDR output (prototype, `METALMC_EXP=hdr`, 2026-09-30)

`Sources/MetalMCNative/Hdr.swift`, hooks marked "HDR hook" in Backend.swift and Taa.swift, and `MainTargetMixin`,
`RenderTargetMixin` and `ScreenshotMixin`. Useful with the sky on; on its own it changes nothing visible (vanilla's colors
stay at most SDR white) and only costs the float target.

- **The level in float.** The main target's color is RGBA16Float instead of RGBA8 (`MainTargetMixin`, and
  `RenderTargetMixin` for window resizes). Its values stay sRGB-encoded like vanilla's, so blending and every shader
  behave as before; the sky writes values above 1 where it's brighter than SDR white. Vanilla's pipelines declare RGBA8
  color targets, so the backend builds each a variant with the pass's format (`PipelineBox.state`, keyed by the
  substituted formats), warmed when the pipeline is created.
- **The tone curve** (`skyToneMap`): scene-linear light to display-linear light, where 1 is the display's SDR white and
  H its current EDR headroom. The identity up to a knee (1 when H ≥ 1.25, 0.9 at H = 1), so vanilla's colors and the GUI
  are untouched, then an exponential shoulder with a matching slope that approaches H. It acts on the largest channel
  (keeps the hue) and moves toward the same curve per channel the further past H the light is, by (m − k)/(m + 8H), so
  the sun's disk (thousands of times over) comes out white and brighter than its glow while a sunset glow keeps its
  orange. At H = 1 it is the SDR curve, which is what runs without `hdr` (the SDR fallback, into the 8-bit target).
- **Where it runs.** Where the level becomes display light: in the sky pass for sky pixels, and in the aerial
  perspective step (the anti-aliasing's resolve or its own pass) for everything else. That is after the level and before
  the hand, the screen effects and the GUI, which then draw over display light: the GUI's white is SDR white, not
  boosted, and a translucent panel over the sun dims it like anything else.
- **The layer.** RGBA16Float drawables, `wantsExtendedDynamicRangeContent`, extended linear sRGB (`METALMC_HDRSPACE=p3`
  for extended linear Display P3), no EDR metadata and `toneMapMode = .never` (macOS 15+), so values up to the headroom
  are shown as they are. The present decodes sRGB to linear light. Setting a CAMetalLayer's pixel format tags it with a
  matching color space (checked on macOS 26: BGRA8 gets sRGB, RGBA16Float extended linear sRGB), so today's SDR layer
  is color-matched as sRGB and extended linear sRGB keeps the same colors.
- **Headroom.** `NSScreen.maximumExtendedDynamicRangeColorComponentValue` of the window's screen, read every 30 frames on
  the main thread (Minecraft's render thread on macOS), followed over about a quarter second, capped by
  `METALMC_HDRPEAK`. It reads 1 until something on the screen asks for EDR, then rises as macOS turns EDR on. Offline on
  this machine, with nothing asking for EDR: current 1.0, potential 16.0, reference 0 (the default preset has no
  reference mode).
- **Anti-aliasing.** With a float frame the history is RGBA16Float and nothing is clamped to 1 (RGB10A2 would clip the
  highlights). CAS leaves values past 2 unsharpened.
- **Screenshots.** Vanilla reads the main target as 8-bit RGBA; `ScreenshotMixin` hands it an 8-bit copy rolled into
  SDR (the tone curve at H = 1). That covers F2, the world icon and the benchmark's screenshots.
- **What the level-versus-GUI split still lacks.** Post effects (the spectator shaders, the menu blur) run through
  vanilla's 8-bit intermediate targets, so they clip what they process. Translucent vanilla things blended over the sky
  (a cloud's edge over a bright sky) get the tone curve a second time in the aerial perspective step, which compresses
  those highlights a little more. The cleanest split, once the deferred rewrite is in: the level in its own float target,
  tone mapped into the main target just before the hand, which also takes vanilla's 8-bit post chains out of the float
  path.

## Verified offline (2026-09-30, no game)

- `swift build -c release`: clean, no new warnings. Every kernel compiles with the runtime compiler (`mslcheck` on
  the sky, HDR and anti-aliasing shaders expanded from the Swift strings). The Java compiles; every mixin target was
  checked against the 26.3 client jar with `javap` (method names, descriptors, and the `createTexture` calls' ordinals).
- **Sky pictures** (`tools/skytest.swift`, which renders through the library's `mmc_debug_sky_render`: an
  equirectangular view over a flat plain at sea level with rock ridges 2, 8, 30, 100 and 250 km away, from 87 m up, with
  the game's aerial perspective and SDR tone curve), at noon, afternoon (35°), golden hour (8°), sunset (1°), twilight
  (−4°), blue hour (−9°), night (−30°) and rain:
  - noon and afternoon: blue zenith (afternoon sRGB 71, 102, 151), near-white horizon, a white glow around the sun; the
    ridges go from gray at 2 km to pale blue at 30 km, barely there at 100 km, gone at 250 km;
  - golden hour: a white disk brighter than its warm glow, peach horizon; sunset: lavender-blue upper sky, pink over the
    sun, an orange band on the horizon (sRGB 255, 181, 81 at the sun), ridges dark against it; twilight: a purple sky
    over a red-orange band; night: near-black with the faint glow; rain: an even gray overcast, the plain fading into
    haze within a few km;
  - no NaN or infinity in any view (scene-linear output checked), and no lookup-table rows show: the largest second
    difference of the sky's encoded luminance up a clear column is under 0.62 of an 8-bit step at every time of day.
- **The game's call sequence** (`tools/skyflow.swift`, through the C entry points with the experiments set, no window):
  the tables, the quarter-resolution sky, the sky pass, a vanilla-style pipeline that declares RGBA8 drawing terrain into
  the RGBA16Float target (accepted: the variant works), aerial perspective as its own pass and inside the
  anti-aliasing, the HDR screenshot copy, and a present into an offscreen EDR layer. Near terrain keeps its exact color
  (0.400, 0.520, 0.301 in and out); far terrain takes the sunlit haze. The two paths agree to within 0.04 at edges (the
  anti-aliasing's sharpening); the quarter-resolution sky matches the per-pixel one to 0.12 of an 8-bit step on average.
  Also run with `sky` alone (the RGBA8 target, the SDR curve, the anti-aliasing's dither, the 8-bit layer tagged sRGB as
  before).
- **Tone curve** (`mmc_debug_hdr_curve`): at H = 1, 1.5, 3, 8 and 16 it is monotonic, never past H, and the identity up
  to the knee; at H = 8, for example: 2 → 1.93, 5 → 4.05, 10 → 6.07, 100 → 8.0.
- **GPU time on this M3 Pro** (`mmc_debug_sky_time`, encoded as the game encodes them, medians of 30 after 10 warm-up
  rounds, two runs agreeing within 0.01 ms; other builds were running on the machine):
  - tables on their own: transmittance 0.02-0.04 ms, multiple scattering 0.09-0.22 (only when the rain changes), sky
    view 0.07-0.15, aerial perspective 0.05-0.12; the per-frame pair together 0.067;
  - at 3456 × 2234: the sky pass working out every pixel 0.48 ms, and 0.185 filtering up the quarter-resolution sky
    (making it included); the same pass drawing a constant costs about 0.1-0.14, so the sky itself is about 0.05-0.09;
  - the aerial perspective adds 0.28 ms to the anti-aliasing's resolve (1.05 → 1.33), or costs 0.38 as its own pass;
  - so with anti-aliasing on, about 0.4-0.45 ms a frame over vanilla's sky pass.

## In game (2026-10-01, native 3456 x 2234, TAA on)

- **Runs.** The render tour (day, sunset, night, rain, the scene, particles, F3, pause, inventory, the Nether, the End) with
  `-PmetalExp=sky` and with `sky,hdr`: no errors once the render pass accepted vanilla's RGBA8 pipelines on the float
  main target (FrontendRenderPassMixin; before that, HDR crashed at the first GUI draw). The pause menu's blur works on
  the float target.
- **Cost** (the real-terrain flight, uncapped): without the sky 158.5 fps, p99 7.80 ms, 20 frames over 8 ms; with it
  147.5 fps, p99 8.35 ms, 93 over 8 ms (about 0.5 ms a frame); with sky and HDR 128.8 fps, p99 9.75 ms, 1,327 over
  8 ms (about 1 ms more: the float main target's bandwidth in every pass, the float history, the float drawable). HDR
  can't hold 120 Hz as it is. The main target and history as RG11B10Float (32 bits, which the render API has) would
  halve their bandwidth; its 6-bit mantissas on sRGB-encoded values band smooth gradients unless dithered.
- **Found and fixed: no haze past 2 km.** At the far field's horizon (LOD 262144) the distant terrain showed no haze
  against the pale sky above it (`m3L1`). `METALMC_EXP=skyhazedebug` (each pixel's distance, haze and fade as colors,
  the sky magenta) showed the far terrain magenta: depth 0, taken for sky. 26.3's Camera.update builds the perspective
  from depthFar as soon as it sets it, so CameraMixin's push of the far plane (after update returned) never reached the
  projection: the level kept vanilla's far plane (the cloud range, 2048 blocks). The quads past it were clipped and the
  far field's march, which isn't, wrote depth 0. The mixin now changes the far plane as it's passed in (`m5L1`: the
  distant terrain fades into the haze; `m5hz`: no magenta under the horizon). Fidelity inside 2 km is unchanged (quads:
  far band 3.05, near 4.59).

## Still to check in game

- The frame with it on (native, TAA on, the real-terrain flight), against off, and with HDR:
  - `BENCH_FIXTURE=claudeworld-merged bash tools/bench/bench_lod.sh skyoff 32768 -PbenchY=150 -PbenchFly=20 -PbenchExtraWait=600 -PbenchHitches=1 -Ptaa=true`
  - the same with the label `skyon` and `-PmetalExp=sky`, `skyhdr` and `-PmetalExp=sky,hdr`, `hdronly` and
    `-PmetalExp=hdr` (`BENCH_NOBUILD=1` after the first);
  - `-PbenchTrace=1` on one of each for per-pass times, and `BENCH_VSYNC=true` for dropped frames at a real 120 Hz.
  Expected: about 0.4-0.45 ms more with the sky (the tables, the sky pass and the resolve), and more with HDR: the float
  history (about 0.3 ms, per the note in Taa.swift), the float target's bandwidth in every pass and the float present
  (not measured offline).
- The look, on the fidelity tour's views at several times of day with the LOD reaching the horizon:
  - `bash tools/bench/fidelity.sh skyNoon 12 262144 -PmetalExp=sky -PfidelityTime=6000`
  - the same with `-PfidelityTime=12000` (sunset), `13000` (dusk), `18000` (midnight) and `23500` (dawn), and with
    `-PfarField=1`.
  Things to look for: far terrain fading into the sky with no line at the horizon; the render distance's edge (try
  `-Plod=0`, vanilla's chunks only) fading into the horizon's color, not a band; clouds (they get the haze from their
  depth, but their own fade at the cloud range still goes to vanilla's fog color); water; the moon and stars over the
  sky; the sun's disk and glow; banding in the sky with and without `-Ptaa=true`.
- By hand: `/weather rain` and back (the sky and haze gray out and return, no flashes as the tables rebuild); under
  water, in lava, in powder snow and with blindness, vanilla's sky and fog must come back; the Nether and the End keep
  vanilla's; sunrise and sunset in motion (`/time add`); below sea level in a cave, vanilla's dark disc still covers the
  lower sky.
- HDR by hand (`-PmetalExp=sky,hdr`): the log's "hdr: EDR headroom" line should rise above 1 within a second; the sun and
  the sky around it brighter than a white block; the GUI (hotbar, chat, menus) exactly as bright as without HDR; F2 and
  the benchmark's screenshots look right (8-bit, SDR); resizing the window keeps HDR, with no "pipeline … failed" lines;
  on an SDR external display the curve falls back to SDR; colors the same as without HDR (`-PhdrSpace=p3` for the more
  saturated alternative).
- Mixins: no errors applying `metalmc_sky.client.mixins.json`; with no experiment set, the frame and the fidelity scores
  are unchanged (on the default path only the anti-aliasing's clamp is now written as `clamp(0, 1)`, and its resolve
  has an unused sky variant).

## Lit mode (deferred relighting) (prototype, `METALMC_EXP=lit`, 2026-10-01)

`Sources/MetalMCNative/Lit.swift`; hooks marked "Lit hook" in Backend.swift and Taa.swift, `LIT_MODE` blocks in the
LOD's, the far field's and the near chunks' shaders, raw visibility in RtShadows.swift; on the Java side
`metalmc.backend.MetalLit`, `LevelRendererLitMixin` and a call in `GameRendererLodMixin`. Step 1 of the lighting
milestone. Off by default: without `lit` none of it is compiled into a shader or attached to a pass. Meant to run with
`nearchunks` (the near terrain in our shader; vanilla's own section draws aren't relit) and `rtshadows` (the sun's
visibility); `sky` gives it the atmosphere's light and `hdr` lets sunlit values past SDR white.

- **What it does.** The terrain shaders we own write a G-buffer beside their color in the level's main pass. After the
  level, and after the ray-traced shadows, every pixel where that terrain is still what shows gets
  `albedo x (sun x max(N.L, 0) x visibility + sky x sky light x AO + moon + block light x AO)` in linear light, light
  sources full bright. The forward color stays vanilla's: it's what shows wherever something nearer was drawn over the
  terrain (entities, water, clouds, particles), and what translucent layers blend with.
- **The G-buffer: one RG32Uint target, 8 bytes a pixel.** At 3456 x 2234 that's 61.8 MB stored at the end of the main
  pass and read once by the relight, the bandwidth of two RGBA8 targets in one attachment (exact bit packing, one
  extra target for every pipeline in the pass to declare); RGBA32Uint would be 123.5 MB each way.
  - x: albedo (sRGB-encoded 8 bits a channel: texture or material color with its tint, before light, face shade and AO),
    AO (5 bits), face (3 bits: 0 not lit terrain, the clear value; 1-6 +X -X +Y -Y +Z -Z; 7 not axis-aligned, like cross
    plants).
  - y: a 16-bit depth key, sky light and block light (8 bits each: levels x 16, vanilla's light coordinates as the shader
    has them, so smooth lighting's gradients survive). A light source is a face with block light 15.
  - The depth key is bits 4-19 of the depth the terrain wrote, and matches the final depth within one step: the depth
    unit doesn't always store the fragment's `position.z` to the last bit (offline, with an exact key 1.2% of a frame's
    LOD terrain pixels differed in their last few bits, scattered over the land; with the tolerance none do). That
    tolerance is about 4e-6 of the distance. A 16-bit key takes about 3 in 65,536 other depths for terrain (measured: 2
    of 55,154 pixels of a quad drawn over terrain), which then shows the relit terrain for a frame. The exact depth would
    take the format to 16 bytes.
- **The pass it goes to.** `LevelRendererLitMixin` marks LevelRenderer's main pass (the lambda in `addMainPass` that
  opens the "Solid"/"Main" pass, where vanilla's terrain, the near chunks, the LOD and the far field are drawn; overworld
  only; the injection isn't required, so a moved lambda only logs "lit: no G-buffer in this frame's level pass").
  `mmc_pass_begin` adds the G-buffer as color target 1, cleared. Vanilla's pipelines in that pass get a variant with the
  extra target and no writes (`PipelineBox.state`, warmed when they're created, like HDR's), the way the far field's
  drawPipe already handled extra targets.
- **Writers.** The LOD's opaque quads (lod_vs sends unlit colors with the depth field and block light in the alpha byte,
  and the fragment shader works vanilla's light out again with the lightmap bound to it too, so the forward color is
  the same); the far field's march (its explicit depth); the near chunks (the lightmap sampled per pixel at the
  interpolated light coordinates instead of per vertex; the tint is the vertex color over its gray level, AO the gray
  level over vanilla's face shade, the face from the record's plane axis toward the camera, or a generic quad's corners).
  Water (the LOD's quads, which now write depth in lit mode, and the far field's water columns) isn't lit terrain: it
  keeps its forward light.
- **Overlays.** Things drawn over terrain without writing depth (rain, block cracks, entity shadows; translucent layers
  with improved transparency, off in the bench setup) leave its depth as it was, so the relight also predicts the
  forward color from the G-buffer (albedo x vanilla's lightmap at its levels x face shade x AO) and compares (6% + 2.5/255).
  Something that darkened the terrain is relit with it (the color over vanilla's light takes the new light); something
  that brightened it keeps its own light and gets the surface's change added. Offline: 0.00% of plain terrain pixels
  flagged, 100% of a translucent band that doesn't write depth caught.
- **The light.**
  - With our sky (`lit_env`, a 64-thread kernel each frame, 512 directions of the upper hemisphere through the sky view
    table, the ground below the horizon gray and lit by sun and sky): the sun through the transmittance at the camera's
    altitude with the sky's eye adaptation, and the sky's light on each face direction, scaled so the sun at the zenith
    plus about 12% for the sky is 1: vanilla's sunlit top at noon.
  - Without it: a daylight curve from vanilla's sun angle (0.82 sun, 0.18 sky on a top under a high sun; the sun warms
    and fades near the horizon; the sky's eye adaptation), scaled by vanilla's own daylight (its lightmap's sky light,
    which follows rain and thunder).
  - The moon (opposite the sun), calibrated to vanilla's night: a top under a high moon gets vanilla's night sky light,
    60% from the moon, 40% from the night sky's glow, bluer than vanilla's.
  - Block light: vanilla's lightmap at the block light level with no sky light (its torch color and flicker, the
    dimension's ambient, night vision, the darkness effect), times AO.
  - Sky light: the face direction's sky light times vanilla's sky light curve, decoded to linear light (sky light 14
    gets 57% of the open sky, 12 gets 21%), times AO.
  - Sun and moon visibility: the traced visibility of the pixel's 4 x 4 block where there are rays (toward the moon while
    the sun is down), else open sky (sky light 12 or less counts as shade). `METALMC_LITEXPOSURE` scales it all.
- **Where it runs.** With anti-aliasing, in the resolve as it loads each pixel (a `taaLit` variant of it, before the
  sky's aerial perspective), like the sky's work: it reads every pixel's color and depth anyway, and a pass of its own
  has to load and store the whole target. Without anti-aliasing, that pass (programmable blending).
- **Shadows in lit mode.** RtShadows stores the raw visibility (1 lit, 0 shadowed, fading to 1 between 6 and 20 km) and
  doesn't darken the color: neither its own pass nor the anti-aliasing multiplies it in. It traces whenever the sun or
  the moon is up.
- **Debug views** (`METALMC_LITVIEW`, `-PlitView`): 1 albedo, 2 face, 3 light levels, 4 AO, 5 the light alone, 6 sun
  visibility, 7 the overlay test.

### Verified offline (2026-10-01, no game)

- `swift build -c release`: clean (the two warnings in RtShadows.apply were there before). The Java compiles.
- `tools/litflow.swift` (through the C entry points as the game calls them, on `claudeworld-merged`, LOD to 2048, camera
  at (8, 150, 8) looking west 22 degrees down, 1728 x 1117, vanilla's lightmap formula for the time of day, mean of 16
  frames per picture where the shadows' rotating samples need accumulating; `bench_out/results/lit/`):
  - the G-buffer covers 76% of the frame; with the depth key, 55% of those are still what shows (the rest is water over
    its floor and the "entity"), 2 of 55,154 pixels under a vanilla-style quad that writes depth were taken for terrain;
  - the overlay test: 0.00% of plain terrain flagged, 100% of the translucent band caught;
  - mean 8-bit luma of the terrain, forward (vanilla's light) against relit:

    | Light | Noon tops | Noon sides | Dusk (sun 8 degrees) tops | sides | Midnight tops | sides |
    |---|---|---|---|---|---|---|
    | vanilla daylight curve | 97.5 -> 98.0 | 52.0 -> 40.4 | 73.3 -> 39.4 | 39.0 -> 25.4 | 38.5 -> 34.4 | 20.4 -> 9.8 |
    | atmosphere (`sky`) | 99.8 -> 99.6 | 60.4 -> 48.0 | 75.8 -> 30.9 | 49.1 -> 33.8 | 38.5 -> 34.4 | 20.4 -> 9.9 |

    So noon's sunlit tops match vanilla's, shaded faces are darker (and bluer), and dusk is darker than vanilla's
    lightmap keeps it (a top under a sun 8 degrees up gets sin 8 degrees, 14%, of the sunlight it gets at noon, less
    through the longer air; the eye adaptation gives back about half a stop): a tuning question for the game.
  - the relight inside the anti-aliasing's resolve gives the same picture as its own pass: mean 8-bit luma difference
    1.16 (vanilla daylight), 1.22 (atmosphere), 1.20 (HDR) over the frame, from the resolve's sharpening and blend;
  - pictures: `<mode>-forward-<time>.png` and `<mode>-lit-<time>.png` (modes `sdr`, `sky`, `hdr`; morning, noon, dusk,
    midnight), `<mode>-lit-morning-taa.png`, `<mode>-view-<name>.png` (the debug views), `<mode>-depth-mismatch.png`
    (magenta: the key doesn't match, which is the water; green: the entity).
- `tools/neartest.py` with `METALMC_EXP=lit,nearchunks`: through the game's draw path (arena, `mmc_pass_begin`,
  `mmc_near_draw`), the near chunks write the G-buffer for every pixel they draw, every depth key matches, and the
  overlay test finds 100% of a real section's pixels plain (the albedo, face, AO and light levels give back the forward
  color; 22 of 2,847 flagged on the hand-made shapes). Their forward color is within 10 levels of vanilla's on real
  terrain (32 on the hand-made shapes): the lightmap is sampled per pixel at the interpolated light coordinates, where
  vanilla samples it per vertex.
- Without `lit`: `tools/neartest.py` ("all checks passed") and `tools/skyflow.swift` (sdr and hdr) pass as before.

### Costs (offline, M3 Pro, 3456 x 2234)

Other work on the machine (the desktop's compositor and apps drawing to the XDR panel) shared the GPU during these runs and
stretched most command buffers 2-6 times, so the numbers are the fastest of 60 runs (A/B runs alternating), which is the
closest to each pass's own cost; the medians were noise. `litflow ... time` with `LITFLOW_TIMEONLY=1` repeats them.

| | RGBA8 target | RGBA16Float target (`hdr`) |
|---|---|---|
| main pass (sky clear + LOD), the G-buffer's share | +0.31-0.35 ms (4.01-4.04 against 3.69) | +0.39 ms in one round, noise in the other |
| the relight as a pass of its own (no anti-aliasing) | 0.62-0.63 ms | 0.75-0.97 ms |
| the relight inside the anti-aliasing's resolve, against the resolve alone | +0.46-0.64 ms (1.77-1.97 against 1.31-1.33) | +0.13 ms (2.55-2.56 against 2.42-2.44) |

So with anti-aliasing, about 0.8-1.0 ms a frame on an 8-bit target and about 0.5 on a float one, plus the sky's light
kernel with `sky` (one 64-thread threadgroup) and the ray-traced visibility, which costs what the shadows did. The LOD
alone stands in for the main pass: vanilla's draws in it get the extra target too, and pay for it in tile memory (not
measured). In-game per-pass times (`-PbenchTrace=1`) are the numbers to go by.

### Not done, and what to look at in game

- Water isn't relit: next to relit land it keeps vanilla's light (most visible at dusk and night). The way: water
  surfaces in the G-buffer with their own albedo, relit by the change in their own light, the floor behind them left as
  it is. (Water now reflects the sky and the sun, `METALMC_EXP=lit,water`, below; its own color is still vanilla's.)
- Vanilla's section draws (`nearchunks` off, or a layer the codec refused) aren't relit; translucent terrain never is.
- Entities aren't lit (and don't cast shadows); a mob's feet and the ground under them are lit differently.
- The sun crosses the zenith (vanilla's path), so at noon every side face is in shade, darker than vanilla's flat 0.6
  and 0.8. A tilted sun path (sky, shadows and relight together) would fix that look.
- The quarter-resolution visibility puts each 4 x 4 block's sample on all 16 pixels; the anti-aliasing averages the
  rotating samples, so edges between a sunlit top and a face turned away soften over a pixel or two.
- The mesh-shader LOD path (`meshshader`) has the lit code but wasn't run offline.
- In game, with timings (`BENCH_NOBUILD=1` after the first):
  - `BENCH_FIXTURE=claudeworld-merged bash tools/bench/bench_lod.sh litoff 32768 -PbenchY=150 -PbenchFly=20 -PbenchExtraWait=600 -PbenchHitches=1 -PbenchTrace=1 -Ptaa=true -PmetalExp=nearchunks,rtshadows,sky`
  - the same with label `liton` and `-PmetalExp=lit,nearchunks,rtshadows,sky`, and `lithdr` with
    `-PmetalExp=lit,nearchunks,rtshadows,sky,hdr`; `litnosky` with `-PmetalExp=lit,nearchunks,rtshadows`.
- The look: `bash tools/bench/fidelity.sh litNoon 12 32768 -PmetalExp=lit,nearchunks,rtshadows,sky -Ptaa=true`, and with
  `-PfidelityTime=13000` and `18000`, and `-PlitView=7` (the overlay test: everything that's terrain should be green;
  red or magenta only under rain, cracks, entity shadows).
- By hand: no "lit: no G-buffer" line in the log; no "pipeline ... failed" lines (vanilla's variants with the G-buffer);
  break a block (its cracks stay), stand next to a mob (its shadow stays), `/weather rain` (streaks stay visible over
  terrain), walk into a cave (no sun inside; torches light it like vanilla), night (moonlight close to vanilla's on
  tops, darker sides), the Nether and the End (not relit: vanilla's look, no G-buffer attached).

### In game (2026-10-01, native 3456 x 2234, TAA on, the far field on)

- **It runs.** The render tour with `-PmetalExp=lit,nearchunks,rtshadows,sky` (`m6Lt`), with `hdr` too (`m6Lh`), and the
  LOD tour to the horizon (`m6LL`, LOD 262144): no errors, "relit 1200 of the last 1200 frames (atmosphere, traced
  visibility)". The level-pass mixin applies (the G-buffer is attached).
- **The look.** Noon: the same brightness as the forward frame on sunlit tops, slopes away from the sun and shaded faces
  darker, the shadows placed (the ravine under the camera falls into shade). Night: vanilla's, a little bluer. Rain: the
  streaks stay over relit terrain, the light flat. Dusk is much darker than vanilla's, with warm light on the faces that
  catch the low sun: a tuning question (`-PlitExposure`).
- **The overlay test** (`-PlitView=7`, `m6Lv`): terrain green; water, mobs, the hand and the sky not terrain; flowers
  over grass flagged as overlays and kept as drawn.
- **Cost** (the real-terrain flight, far field on): near chunks, ray-traced shadows and the sky without lit mode
  132.8 fps (p99 9.89 ms, 1,174 frames over 8.33 ms); with it 115.4 fps (p99 10.79 ms, 4,565 over): about 1.1 ms, the
  G-buffer's writes and the relight. With the shadows' 1 ms and the sky's 0.5 ms the whole look doesn't hold 120 Hz yet.

### Next: the GI cache as the sky term (gi-design.md, "Integration" steps 1-3)

Done: see "Lit mode with the GI cache" below.

## HDR's main target packed (2026-10-01)

HDR's float main target cost about a millisecond over the sky alone, mostly bandwidth: every pass that loads or stores
it moves 64 bits a pixel. It's now RG11B10Float by default (`METALMC_HDRFORMAT=rgba16f` for RGBA16Float), and so is the
anti-aliasing's history; the drawable stays RGBA16Float for EDR. The real-terrain flight with sky and HDR (far field
on): 124.7 → 136.3 fps, frames over 8.33 ms 2,166 → 964. Its 6-bit mantissas (5 in blue) banded the sky's glow around
the sun in rings with an absolute dither (`m9Tp`); the sky pass and the anti-aliasing now dither relative to the value
(1/64 of it, 1/32 in blue), and the sunset is as smooth as with RGBA16Float (`m10Tp`). No alpha: the render tour's
GUI, inventory, pause blur, particles, the Nether and the End look as before.

## Lit mode with the GI cache (prototype, `METALMC_EXP=lit,gi`, 2026-10-01)

The world-space irradiance cache (gi-design.md, `Sources/MetalMCNative/Gi.swift`) wired into lit mode as its sky term:
the sky's light as the real openings let it in, plus bounced light. Meant to run as
`-PmetalExp=lit,nearchunks,rtshadows,sky,gi` (or without `sky`). With `gi` and no `lit`, or `lit` and no `gi`, nothing
changes: the GI parts of the relight's and the anti-aliasing's shaders are spliced into their text only with both
(`litGi`), RtShadows builds its structures as before, and nothing else runs.

- **What a bounce ray needs from its hit, zero copy.** RtShadows builds each tile's structure with one geometry per
  quad range of the node's own buffer: the opaque faces' buckets, then the tile-edge skirts' (`giTileMesh`: the same
  triangles in the same order as without the cache). With the instance structure comes a table of each instance's node
  buffer, by GPU address, and its ranges' first quads (`GiTile`, 16 bytes an instance, buffer 17); the encoder declares
  the node buffers resident. A hit's instance, geometry and primitive index give its quad (`start[geometry] + primitive
  / 2`), and the quad's two words give the material, face, block light and sky cover the primitive data held
  (`giHitQuad`). Structure memory on the region test (3.41 M triangles): 41.8 bytes a triangle with zero copy, the
  same as without the cache (136.0 MB; the second geometry costs nothing measurable after compaction), against 64.5
  with per-triangle data (209.7 MB, +73.8 MB). The other route stays for comparisons (`GICACHE_PRIMDATA=1`); it wasn't
  needed.
- **Per frame.** In `RtShadows.trace`, after the shadow rays: `GiCache.shared.encodeFrame` on the same depth, jittered
  projection, origin and instance structure, in one compute encoder, into half-resolution RG11B10Float light and R32Uint
  code words that the relight takes like the visibility (`takeLitGi`). The cache (2 M cells, 72 MB) and its kernels are
  made on the first `lit,gi` frame.
- **Its light is the relight's.** Each frame `gi_light` makes the cache's light (`GiLight`) from lit mode's own sun and
  sky light (the relight's `env`): with our sky, Lit's `lit_env` run on this frame's sky tables into the cache's buffer,
  and bounce rays that leave the terrain read the sky view table (`skyLuminance`, scaled to the relight's units by
  `lit_env`'s own scale, which it writes to `env[0].w` with `gi` only); without it, `litDaylightEnv`'s values and an even
  sky whose light on a top face is the curve's sky (and its ground below the horizon). Which of the two follows the
  relight's choice of the frame before (one frame late when our sky turns on or off). Block light isn't in the cells: the
  relight's block light stays vanilla's flood fill, and bouncing it would need a cell channel of its own (below). No
  cells for vanilla's clouds (they write depth), as `rt_shadow` skips them.
- **Time of day.** The cells hold light per unit of the frame's daylight (the sun's irradiance plus the open sky's on a
  top face, per channel, `env`'s units), and the resolve multiplies by this frame's: a change of brightness or color
  (the time of day, rain, eye adaptation) shows in the first frame; what follows over the cells' history (64 updates)
  is only the change in the pattern of light, as the sun moves and the sun-to-sky ratio changes. Measured (below): after
  a jump from noon to dusk the shaded faces' mean luma is within 3% of where it settles from the first frame on, and
  within 1.6% after 64 frames (half a second). Without the scale the cells would start from noon's light and take their
  history to come down (*estimate*: 0.5 s for cells updated every frame, seconds for the rest, longer for bounced light,
  which reads other cells' lagging light).
- **Shading.** In `litRelightPixel`, where `giUpsample` finds a sample on the pixel's face and plane, the sky term
  (`env[1 + face] x` the sky light level's curve `x AO`) becomes the cache's light `x AO`; it is in `env`'s units, so
  the daylight curve's scale (without our sky) and the exposure apply as before. Everything else is unchanged: sun x
  N.L x traced visibility, moon, block light, the overlay test. Faces that aren't axis-aligned (plants) take the top
  face's sample around them. Where no sample matches (3.9% of the relit terrain in the test view: faces a pixel or two
  wide, far hills) the sky term stays; the cells' own lookups (`giIrradiance`) aren't made there (they need the table
  bound, 0.1-0.5 ms more offline). Debug view 8 (`-PlitView=8`): green where the cache's light applied, red where it
  didn't, blue for light sources.

### Verified offline (2026-10-01, no game)

- `swift build -c release`: clean (the two warnings in `RtShadows.apply` were there before). The Java compiles
  (`build.gradle` has `-PgiBudget`, `-PgiRange`, `-PgiCap`, `-PgiSpp`, `-PgiHistory` now). Every kernel compiles at run
  time in both routes.
- `python3 tools/gicache.py selftest`, both routes: 1,500 key round trips and the 4 ray cases right (with zero copy the
  hit's face comes from its quad). The region's pictures (`region/`, zero copy) have the materials in their places.
- **Without `gi` nothing changed.** `tools/litflow.swift` in `sdr` mode (lit mode without the cache) through the same
  driver on the unmodified library and this one: every number it prints is the same (the G-buffer and overlay checks,
  the luma tables, the anti-aliasing's match), and 16 of its 17 pictures are the same to the bit. The 17th, the first
  capture after the warm-up (`sdr-lit-morning.png`), differs in 384 of 1.93 M pixels, by at most 7 levels, all near the
  horizon (rows 297-324): a far tile's shadow structure finished building in the background a frame apart in the two
  runs, by the look of it (they build asynchronously, nearest first); not proven.
- **The synthetic house** (`python3 tools/gicache.py litsynth <dir>`, lit mode's daylight curve through `gi_light`, the
  relight's formula without AO or lightmap; 1152 x 745, 128 frames): the room lit through its 4 x 3 window, mean 8-bit
  luma at exposure 4: lit 26.3, lit+gi 45.9 at mid-morning (the LOD's sky cover leaves the room black but for the
  floor by the window; the cache lights it from the sunlit patch on the floor, warm, the red wall's bounce on the
  floor), 7.3 and 23.9 at dusk; outside 74.5 and 74.0 at exposure 1 (in the open the cache's sky matches the sky term's).
  Frame-to-frame change of the cache's light: 1.17% in the room (1.14% at dusk), 0.03% outside.
- **The game's call sequence** (`LITFLOW_GI=1 .build/litflow ... sky|sdr [time]`: claudeworld-merged, LOD to 2048,
  camera at (8, 150, 8) looking west 22 degrees down over hills, a ravine and the sea, 1728 x 1117; the cache and the
  relight as the game runs them, lit and lit+gi in the same frames through `mmc_debug_lit_gi`; 16-frame means), mean
  8-bit luma of the relit terrain:

  | Light | | Shaded faces (no direct sun) | Sunlit tops | All relit terrain |
  |---|---|---|---|---|
  | atmosphere (`sky`) | morning | 41.1 -> 35.1 | 77.8 -> 77.5 | 76.2 -> 74.5 |
  | | noon | 47.6 -> 36.1 | 99.9 -> 99.4 | 81.1 -> 76.7 |
  | | dusk (sun 8 degrees up) | 33.2 -> 30.9 | 31.9 -> 30.7 | 32.4 -> 30.8 |
  | daylight curve (`sdr`) | morning | | 82.2 -> 80.8 | 77.2 -> 74.5 |
  | | noon | | 98.6 -> 97.6 | 77.8 -> 72.2 |
  | | dusk | 25.7 -> 19.1 (faces turned away) | 41.1 -> 37.0 | 35.5 -> 30.6 |

  Sunlit tops barely change; shaded faces get darker where terrain hides part of the sky (terrace risers, the ravine,
  inside the forest's canopy), and greener from the grass in front of them, where the sky term gave every face with sky
  light 15 the same open sky and a gray ground. The cache's light covered 96.1% of the relit terrain (94.6% at dusk).
  Frame-to-frame change of single frames on shaded faces: 0.16-0.23% without the cache (the sky's dither), 0.23-0.27%
  with it (`sdr`, no dither: 0.00% and 0.05%). A jump from noon to dusk (`sky`): the shaded faces' mean luma 31.1 in the
  first frame after it, 31.1, 31.1, 31.0 at frames 2, 8 and 16, 30.9 at 32, 30.7 at 63, settling at 30.2 (the sky term
  without the cache: 33.2); `sdr`: 19.0 to 18.8 over 63 frames, settling at 18.5.
- Pictures (`bench_out/results/gi-lit/`): `skygi-lit-<time>.png` and `skygi-lit-gi-<time>.png` (morning, noon, dusk),
  `skygi-view-gi-coverage-<time>.png` (view 8), the same with `sdrgi-`; `synth/room-lit.png`, `room-lit-gi.png`,
  `room-dusk-*`, `outside-*`; `region/region-*.png` (the cache's own views on real terrain, zero copy).

### Costs (offline, M3 Pro, 3456 x 2234)

Other agents' GPU work shared the machine during these runs; the numbers are medians, and A/B runs were made back to
back.

- The cache's stages on real terrain (`METALMC_GIREPEAT=4 python3 tools/gicache.py region <r.0.0.mca> <dir>`, 128 frames,
  converged): zero copy request 0.17, schedule 0.03, update 0.41, resolve 0.46: 1.06 ms; with per-triangle data, just
  before, 0.20, 0.03, 0.46, 0.47: 1.16 ms. The dependent read per bounce hit costs nothing measurable. (A first zero-copy
  run while other work had the GPU: 1.69 ms.) The relight's upsample: 0.32-0.50 ms more (the region test's
  lighting-pass share without lookups).
- In the frame (`LITFLOW_GI=1 .build/litflow ... sky time`, whole frames with and without the cache's frame,
  alternating, 60 each): with the relight in the anti-aliasing's resolve, median +1.11 ms (fastest +1.76); with it as a
  pass of its own, fastest +1.21 ms (the medians were noise). About 1.1-1.8 ms, against the cache's 1.5 ms budget: with
  the shadows' 1 ms, the sky's 0.5 and lit mode's 1.1, the whole look is further from 120 Hz.

### In game (2026-10-01 morning, native 3456 x 2234, TAA on, the far field on)

- **It runs.** The render tour with `-PmetalExp=lit,nearchunks,rtshadows,sky,gi` (`m12G`) and its coverage view
  (`-PlitView=8`, `m12Gv`): no errors; "gi: cache of 2097152 cells, 72 MB, hit quads from the LOD's buffers (zero copy)",
  then about 16,300 cells updated a frame and 45,000 visible cells read.
- **The look outdoors.** Shaded faces darker than lit mode's open-sky term where the terrain hides the sky (the ravine
  under the camera, forest floors, the far hills' sides); the sunlit tops as before; at sunset the faces near sunlit
  ground get some warm bounce. Subtle outdoors, as expected: the room lit through a window (offline) is where it shows.
- **Cost:** the real-terrain flight with lit, near chunks, shadows and the sky: 115.9 fps without the cache, 93.5 with
  it (p99 13.2 ms): about 2.1 ms, more than the 1.1-1.8 ms offline. Not for 120 Hz yet.

### Not done, and what to look at in game

- Not run in game. Not run offline with `hdr`, or on the mesh-shader LOD path.
- Block edits aren't sent to the cache (`GiCache.invalidate` has no caller yet): a broken block reaches it when the LOD
  rebuilds the node (new tile structures), and the cells around it keep their old light for their history; a face that
  vanished keeps its cell until it's unseen for 10 s.
- The cache isn't cleared on a world change (lit mode only runs in the overworld; another world's cells at the same
  coordinates fade out over their history, or are evicted after 10 s unseen).
- No bounced block light (torchlight around corners stays vanilla's flood fill). The split by source in gi-design.md's
  "Next" (separate sun, sky and block channels per cell, scaled at read time) would add it and make the time-of-day
  scale exact for the sun-to-sky ratio too.
- Water, entities and the hand get no cache light (they aren't lit terrain).
- The first `lit,gi` frame creates the cache and compiles its kernels on the render thread: a one-time hitch.
- In game, with timings (`BENCH_NOBUILD=1` after the first):
  - `BENCH_FIXTURE=claudeworld-merged bash tools/bench/bench_lod.sh litgion 32768 -PbenchY=150 -PbenchFly=20 -PbenchExtraWait=600 -PbenchHitches=1 -PbenchTrace=1 -Ptaa=true -PmetalExp=lit,nearchunks,rtshadows,sky,gi`
  - the same with label `litgioff` and `-PmetalExp=lit,nearchunks,rtshadows,sky`; `litginosky` with
    `-PmetalExp=lit,nearchunks,rtshadows,gi`.
- The look: `bash tools/bench/fidelity.sh litGiNoon 12 32768 -PmetalExp=lit,nearchunks,rtshadows,sky,gi -Ptaa=true`, with
  `-PfidelityTime=13000` and `18000`, against the same without `gi`; and `-PlitView=8` (green nearly everywhere on
  terrain; red only on thin faces and at the far edge).
- By hand: the log's "gi: cache of 2097152 cells, 72 MB, hit quads from the LOD's buffers (zero copy)", "gi: ... frames"
  and "lit: relit ... GI cache in ..." lines, no "gi: shaders failed"; a house with a window (dark inside but for the
  sunlit patch's bounce; nothing leaks through 1-block walls); a cave mouth; `/time set 12000` from noon (the light drops
  at once and settles within a second or two); `/weather rain` and back; a fast 180-degree turn (cells without samples
  show the sky term for a few frames, no black); breaking a block in a wall; flying fast at the LOD's speed (cells are
  per block face near the camera: watch for blotches where they're new).

## Water that reflects (prototype, `METALMC_EXP=lit,water`, 2026-10-02)

Lit mode left water as it was drawn: vanilla's water color in vanilla's light, no reflections (`LIT_NONE`). With `water`
(and `lit`; the reflections need our sky, `sky`, for the light they reflect) water surfaces reflect the sky and the sun.
Off by default until it's been seen in the game. Without it every shader is the same text as before (its parts are
spliced in only with `litWater`) and every pixel is the same (checked below). `Sources/MetalMCNative/Lit.swift`
(`litPackWater`, `litWaterPixel`, the waves' and the sky map's kernels), the far field's water output in `FarField.swift`,
the relight's call site in `Taa.swift`.

- **The look.** Per water pixel, in scene-linear light (the sky's units, which the relit terrain is in too), in the
  relight, before the aerial perspective: `drawn x (1 - F x open) + sky(R) x F x open + glint`.
  - drawn: the water as it was drawn: vanilla's water in vanilla's light over the floor seen through it (shallow water
    keeps showing its floor).
  - F: Schlick's Fresnel reflectance for water (F0 0.02): 2% looking straight down, 11% at 22 degrees (water near the
    camera from the flight's height), most of the light at grazing angles.
  - sky(R): the sky's light in the reflected direction, without the sun's disk, from a map of the upper hemisphere made
    each frame from `skyLuminance` (`lit_water_sky`, a 256 x 256 paraboloid map: about half a degree a texel at the
    horizon). Rays a wave turns under the horizon take the horizon's light.
  - open: the water's sky light level through vanilla's curve, decoded, as the relight's sky term has it (1 in the open,
    0.57 at 14, 0.21 at 12): water under cover doesn't reflect a sky it can't see, and keeps its drawn color there.
  - glint: the sun's disk through a GGX lobe (Smith's height-correlated visibility, Fresnel at the half vector), with the
    sun's illuminance at the camera's altitude (lit_env's sun term, back in scene units: lit_env leaves its scale in
    env[0].w with water, as it did for the GI cache), times the traced visibility (shadows fall across the glint), as much
    as the sky draws of the disk (none in rain), none once the sun has set. A low sun lays a path of it on the water.
- **Waves.** Eight directional sines, two to an octave (wavelengths 10 down to 1.4 blocks, slopes 0.032 down to 0.02
  radians, 0.053 RMS together, deep water's speeds: angular frequency sqrt(g k)), summed each frame into a mipmapped tile
  of their slopes and the slopes' mean squared length (`lit_water_waves`: 64 blocks, 256 texels a side, whole wave
  numbers across it so it repeats without a seam). Each level holds the waves it can, each whole while its wavelength
  spans 4 of the level's texels and gone at 2, and adds the variance of the rest to the squared length. A water pixel
  samples it once at its footprint (along the view, where it's longest): the mean slope there tilts the normal and the
  slopes' variance goes into the lobe's roughness (LEAN mapping, isotropic), so waves finer than the pixel don't alias:
  near water sparkles, far water has a smooth, broad sun path (the lobe's alpha from 0.05 up to 0.073). The camera's x
  and z modulo the tile (the LOD's camera this frame) keep them in place as it moves; their time is the clock's, wrapped
  every hour. `METALMC_WATERWAVES` scales the slopes (0: flat), `METALMC_WATERROUGH` sets the surface's own alpha under
  them (0.05). Tried and changed:
  - one wave to an octave (5): straight parallel stripes at mid distance, where only the longest are left; two to an
    octave, 70 degrees either side of the wind's direction, cross-hatch;
  - the sum per pixel (8 cosines, faded by the footprint): the same look, 0.12 ms more in the anti-aliasing's resolve;
  - mipmaps averaged down from the top level (`generateMipmaps`): a 2 x 2 box filter leaves the waves near a level's
    limit in it, and the longest waves' crests showed as rays converging on the horizon over mid-distance water (with a
    32- and a 128-block tile alike, so not the tile repeating); every level summed from the waves instead.
- **The G-buffer.** A water texel: face code 0, bit 28 set, its face in bits 24-26; y as terrain's (the depth key, so a
  boat or an entity drawn over water isn't taken for it, and the light levels). Terrain's codes are 1-7 and the clear
  value is 0, so a code of 0 with bit 28 set was free: AO keeps its 5 bits, and every reader that doesn't know water (the
  relight's terrain path, the offline checks) sees "not lit terrain" there, as before. `litPackWater` and `litIsWater` in
  `litShaderHeader`. Gi.swift doesn't read the G-buffer (its requests come from the depth buffer).
- **Writers.**
  - The far field's march: a water column's top (its surface, 10/9 block under its word's top) is flagged; its sides stay
    `LIT_NONE` (terrain seen through the quads' water).
  - The LOD's water quads: not in this change (Lod.swift belongs to another builder; what its water pipeline must write is
    below). Until they write it, water from 192 to 768 blocks doesn't reflect and the far field's does: a seam at 768.
  - Vanilla's water (the near chunks' range): not yet, see "Not done".
- **Where it runs.** `litWaterPixel`, after `litRelightPixel`, in the anti-aliasing's resolve (both lit variants; the
  waves' tile and the sky map are its textures 13 and 14) and in the relight's own pass (textures 6 and 7). Without our
  sky (the daylight curve) water is left as drawn: there's no sky to reflect. Both textures are made in the relight's
  command buffer each frame with our sky (one compute encoder: a dispatch per level of the waves' tile, one for the sky
  map).
- **Debug views** (`METALMC_LITVIEW`): 9 which water reflects (cyan; magenta where something nearer was drawn over it), 10
  the reflections alone (with the aerial perspective over them).

### For the LOD's water (Lod.swift, its builder)

The water pipelines write the G-buffer with water on, and the shared shading flags water fragments:

```
// lodShadeLit (the shader), after `out.gbuf = litPack(albedo, ao, (in.matFace >> 8) & 7, in.pos.z, sky, block);`:
#if LIT_WATER
    // Water (METALMC_EXP=water, Lit.swift): its surface flagged for the relight's reflections (the water pipelines write
    // the G-buffer with water on).
    if ((in.matFace & 255u) == MAT_WATER) out.gbuf = litPackWater((in.matFace >> 8) & 7u, in.pos.z, sky, block);
#endif

// LodRenderer.makePipeline (Swift), both places that set the extra targets' write masks (the mesh path, the vertex path):
if i > 0 && !(litWritesGbuffer(i, f) && (!water || litWater)) { d.colorAttachments[i].writeMask = [] }
if (i > 0 && !(litWritesGbuffer(i, f) && (!water || litWater))) || box { d.colorAttachments[i].writeMask = [] }
```

and the comment over `LodOut` ("The water pipelines don't write the G-buffer") becomes "The water pipelines write it
only with water on (METALMC_EXP=water): the flag for the relight's reflections". `LIT_WATER` and `litPackWater` exist
only with the switch (litShaderHeader), so without it the LOD's text and pipelines are as before. The water quads
already write depth in lit mode, which the flag's depth key needs. Blending stays on the color target only: the
G-buffer (an integer format) is written, the last water fragment through the depth test (the nearest: water writes
depth) wins. Checked in a scratch copy (built, run in litflow; the pictures and numbers below with LOD water come from
it). It costs about 0.06 ms in the main pass (measured, below).

### Verified offline (2026-10-02, no game)

- `swift build -c release`: clean (the two warnings in `RtShadows.apply` were there before). `litflow ... compile` (a
  new mode: every variant compiled with the device's compiler, as the library builds them) passes under
  `lit,rtshadows,sky,water`, `lit,rtshadows,water`, `...,sky,hdr,water`, `...,sky,gi,water` and `...,gi,water`: the
  anti-aliasing's resolve (16 and 32 pixel tiles; plain, sky, lit, lit and sky), the far field (its march for the level
  pass's targets and its kernels) and lit mode's own pass (RGBA8, RGBA16Float, RG11B10Float) with its kernels.
- **Without `water` nothing changed.** The anti-aliasing's and the far field's shader sources under `lit,rtshadows,sky`,
  `lit,rtshadows`, `...,sky,gi`, `...,sky,hdr`, `...,gi`, `rtshadows,sky` and `sky` are byte for byte the old build's
  (`mmc_debug_taa_shader_source`, `mmc_debug_far_shader_source`), and litflow's 17 pictures in `sky` and in `sdr` mode
  (far field on) are bit for bit the old build's (`bench_out/agents/water/exact-old{,-sdr}` against `exact-new2`,
  `exact-new-sdr`).
- **"Off" is the old frame.** The old build through the same water section (no switch: only "off" pictures) draws the
  default view at dusk bit for bit as the new build does with the reflections off (`old-sea` against `committed-sea2`).
- **The reflections touch water only.** `LITFLOW_WATER=1` draws each time of day with them off and on in the same build,
  the "on" frames at the same places in every 64-frame cycle (the shadows' disk samples, the sky's dither): the pixels
  that changed are the flagged water but for 0-933 of 1.93 M, all far land near the horizon that took a new shadow in
  between (the shadows' tile structures still build in the background after the warm-up; marked in
  `*-changed-dry.png`).
- The G-buffer flags 2.6% of the default view (the far field's water) with the committed code, 31.5% with the LOD's
  water flagged too (the scratch copy), 54.4% from low over the sea; debug view 9 shows all of it reflecting (no
  magenta: nothing over the water in these views).
- Pictures (`bench_out/agents/water/`, 1728 x 1117, means of 16 frames, the waves 1/120 s on a frame; `-off` is the
  frame as before, `-on` with the reflections, `-on-taa` the same through the anti-aliasing's resolve, `-reflections`
  debug view 10):
  - `committed-sea2/sky-water-sea-{dusk,noon,sunset}-{off,on,on-taa,reflections}.png`: the default view (8, 150, 8,
    looking west 22 degrees down) with the committed code: the far field's sea band takes the sky and the sun's path;
    the LOD's water nearer than 768 blocks doesn't reflect yet (the seam the LOD's write removes).
  - `lod-sea4/...`: the same view with the LOD's water flagged: the whole sea reflects, the far sea the bright sky low
    down, near water a little of the sky over its floor, the waves cross-hatched at mid distance.
  - `lod-lowsea3/sky-water-lowsea-{dusk,sunset,noon}-*.png`: 7 blocks over the open sea looking west (-300, 70, -400, 4
    degrees down), the sun ahead at dusk (8 degrees up) and sunset (2): the sun's glittering path from the horizon to the
    camera, broken by the waves near it; at noon the sea takes the sky's blue, brighter toward the horizon.
    `lod-lowsea-hdr/hdr-water-lowsea-{sunset,noon}-*.png`: the same with HDR's float target (through the screenshot's
    SDR copy): the path keeps its orange where SDR's clamp per channel whitens it.
  - `lod-pond/sky-water-pond-{noon,low}-*.png`: a lake in the savanna (-1760, 90, -890, looking east 18 degrees down) at
    noon, and with the sun 7 degrees up ahead: its path broken by the trees' shadows (the traced visibility).
  - `lod-river2/sky-water-river-{noon,dusk}-*.png`: a river in a canyon (718, 100, 140, looking south 15 degrees down):
    the river takes the sky where it should take the canyon's walls, the brightest thing in the canyon at dusk. What's
    missing: terrain in the reflection, and how much sky the water sees (the LOD's water always has sky light 15).

### Costs (offline, M3 Pro, 3456 x 2234)

`LITFLOW_WATER=1 litflow ... sky time` on the default view, morning sun, RGBA8 target: medians of 60 command buffers each,
the reflections on and off alternating on the same G-buffer (`mmc_debug_lit_water_time`); "no water" is the same library
without the switch, run back to back (its shaders are the old build's). The waves' tile and the sky map are made in every
"on" relight, so their cost is in the "on" figures.

| | no water | water compiled in, reflections off | reflections on |
|---|---|---|---|
| anti-aliasing resolve, relight and aerial perspective in its load (31.5% of the frame water: the LOD's flagged) | 2.052-2.062 ms | 2.104-2.105 | 2.348-2.350 |
| the relight as a pass of its own (no anti-aliasing), same | 0.658-0.664 | 0.710-0.711 | 0.833 |
| main pass with the G-buffer, same | 3.592 | | 3.654-3.656 |
| anti-aliasing resolve, committed code (2.6% water: the far field's) | 2.052-2.062 | 2.094-2.100 | 2.128-2.136 |
| the relight's own pass, committed code | 0.658-0.664 | 0.673-0.688 | 0.684-0.695 |
| main pass with the G-buffer, committed code | 3.592 | | 3.593-3.594 |

So with a third of the frame water about 0.29 ms in the anti-aliasing's resolve (of which 0.05 is the water code being
compiled into it, paid with no water in view; the reflections' own work 0.24) plus 0.06 in the main pass (the LOD's
water pipelines writing the G-buffer): 0.35 ms a frame. With the committed code alone (the far field's water) about
0.08. Over the open sea from low down (`lowsea`, 54% water) the reflections add about 0.4 ms in the resolve, on against
off (0.41 a step before the last); the terrain they cover costs the relight about as much, so that view's resolve
(1.97 ms) stays under the default view's.

Getting there (the resolve, on against off, the default view with the LOD's water flagged):

| | +ms |
|---|---|
| first: 8 cosines a pixel for the waves, the sky view table looked up per pixel | 0.37 |
| the waves summed into a tile once a frame, one sample a pixel | 0.27 |
| the sky into a map once a frame, one sample a pixel | 0.23 |
| the tile's levels each summed from the waves (no rays; 9 dispatches, serial) | 0.26 |
| those dispatches side by side, the levels where every wave has faded made once | 0.24 |

The first version's parts (an instrumented copy that switched each off at run time): the waves 0.12 ms, the sky lookup
0.08, the glint 0.04, the rest 0.13 (the color decoded and encoded again, six powers; the position; Fresnel). The rest is
the floor of doing this per pixel in linear light; it's about what the relight costs a terrain pixel. In the frame
budget's terms: lit mode ran at 120.5 fps in game with 32 x 32 tiles (8.30 ms, frame-budget.md), so 0.35 ms more with a
third of the screen water puts it past 8.33 ms; something else has to give for 120 Hz with this much water in view.

### Not done, and next

- **The LOD's water** (192-768 blocks): flagged by the code above once its builder adds it. Until then a seam at 768
  blocks between unreflective and reflective water.
- **Vanilla's water** (the near chunks' range): not flagged. Its translucent layer (water, stained glass and ice share
  it; sorted per quad, so the near chunks leave it to vanilla) is vanilla's own pipeline, translated on the Java side,
  and in the lit pass it gets the G-buffer attachment with no writes (Backend.swift, PipelineBox.state). Flagging it
  needs a variant of that pipeline whose fragment shader also writes `litPackWater` where its texel comes from the
  water sprites (the atlas rectangles of block/water_still and block/water_flow, which the Java side knows) and writes
  the G-buffer's own value back elsewhere (read with programmable blending, so glass and ice leave the terrain behind
  them as it was), with the face from its position's derivatives (a top if it faces up) and the sky light from its light
  coordinates; the variant's G-buffer attachment writable. Backend.swift and the Java side, outside this change. Until
  then near water keeps vanilla's look, with a seam where the LOD's begins.
- **Terrain in the reflection** (screen-space reflections, next): a march along the reflected ray through the depth
  buffer, with the sky as the fallback. The canyon river shows what it's for; so do lakes under hills at grazing angles.
- **How much sky the water sees.** The far field's water has its sky light (12 under a canopy); the LOD's always 15. A
  canyon or a cave lake reflects the open sky. The traced rays, or the GI cache's cells, could say how open it is.
- **The water's own color** is still vanilla's light: at dusk and at night it's brighter than the relit land around it
  (the relight's "Not done"). Its body could be relit by the change in its light, the floor seen through it left as it
  is.
- No moon glint at night (the moon is vanilla's; its light the relight's moon term); the glint, for the sun only.
- Water's sides (falls) and water seen from below don't reflect.
- Not run in game.

### What to look at in game

- With timings: `BENCH_FIXTURE=claudeworld-merged bash tools/bench/bench_lod.sh litwater 32768 -PbenchY=150 -PbenchFly=20
  -PbenchExtraWait=600 -PbenchHitches=1 -PbenchTrace=1 -Ptaa=true -PmetalExp=lit,nearchunks,rtshadows,sky,water`, against
  the same without `water` (label `litnowater`); after the LOD's water is flagged, again.
- The look: `bash tools/bench/fidelity.sh litWater 12 32768 -PmetalExp=lit,nearchunks,rtshadows,sky,water -Ptaa=true`
  with `-PfidelityTime=12000` (sunset) and `6000`; `-PlitView=9` (cyan: water that reflects).
- By hand: the sea at sunset from a hill and from a boat (the path; the boat's pixels keep their color); the waves in
  motion (no swimming as the camera moves, no shimmer at mid distance); rain (the sky's gray in the water, no glint);
  night (dark water, no glint); a lake under an overhang (sky light under 15 dims its reflection) and a river in a
  canyon (the sky where the walls should be); the seams at the LOD's start (vanilla's water) and at 768 blocks (until the
  LOD writes the flag); `METALMC_WATERWAVES=0` (a mirror) and `2`.

## Colored block light (prototype, `METALMC_EXP=lit,coloredlight`, 2026-10-05)

Lit mode took block light from vanilla's lightmap: one warm white at vanilla's level. With `coloredlight` (and `lit`;
meant with `nearchunks`, whose terrain is what's relit near the camera) block light has the color of what gives it off:
torches and lanterns a candle-warm orange, fire and campfires a redder orange that flickers, soul fire cyan, redstone
red, lava and magma orange-red, glowstone and redstone lamps warm, sea lanterns, end rods and beacons a cool white, the
three froglights yellow, green and pink, amethyst, crying obsidian and portals a faint purple, glow lichen teal. Off by
default; without it every shader is the same text as before and nothing else runs (checked below).
`Sources/MetalMCNative/ColoredLight.swift` (the volume, its kernels, the relight's part), `ColoredLightOffline.swift`
(region files, the test scene, checks); on the Java side `metalmc.light.ColoredLight` (the blocks),
`metalmc.light.mixin.ClientLevelLightMixin` (block changes), `metalmc.backend.MetalColoredLight` (the bindings) and a call
in `GameRendererLodMixin`; the splices in Lit.swift and Taa.swift are marked "colored block light". In lab mode the
volume's kernels are `colored_light.metal` and the relight's part is in `lit_relight_header.metal` (ShaderLab: an edit
rebuilds the pipelines between two frames; the volume keeps its light).

### Design

- **Eight colors, each spread exactly as vanilla spreads block light.** Every cell of a volume around the camera holds
  eight light levels, one per color ("bucket": warm, fire, soul, red, lava, white, green, purple), 4 bits each in one
  32-bit word. A light-emitting block puts its vanilla level (`getLightEmission`) in its color's bucket (some in a second
  bucket a few levels lower, which tints them: glow lichen green plus soul, pink froglights purple plus red). Each bucket
  then follows vanilla's rule: a cell's level is its own emission, or its brightest neighbor's less max(1, the cell's light
  dampening); opaque cells hold only their own emission. So each color reaches exactly as far as vanilla's light does,
  goes around corners the same way and is stopped by the same walls, and the brightest bucket of a cell is vanilla's own
  level there. Where lights of different colors overlap, their light adds (a torch and a soul torch make white between
  them); two lights of one color don't add, as in vanilla.
- **Why buckets, not RGB.** A flood fill of RGB light by "brightest neighbor less a step" shifts hue with distance (blue
  runs out first) and takes the brightest channel of each source where they meet (a torch and a soul torch make
  magenta); a fill that sums and diffuses (light propagation volumes) loses light in narrow tunnels and doesn't reach as
  far as vanilla's, so caves vanilla lights would go dark. Integer levels per color are exact (no drift, nothing to
  converge numerically), compact (4 bytes a cell) and checkable against vanilla cell for cell.
- **The volume:** 256 x 128 x 256 cells of one block around the camera (128 either side, 64 above and below; it moves a
  whole section at a time when the camera is 12 blocks off its center, 8 vertically), addressed modulo its size, so
  moving only replaces the sections that enter it. 33.5 MB of light, 16.8 MB of block codes (each cell's dampening,
  emission and color class), and the filtered light the relight samples (100 MB, below): 151 MB.
- **The flood fill, on the GPU, only where something changed.** Each frame: sections that entered the volume or changed
  are uploaded (their light reset to their own emission), single block changes applied in place; then the 8 x 8 x 8
  bricks that changed since the last frame, with their 26 neighbors, are listed, and the fill runs over them 4 times
  (`METALMC_CLITER`), in place, a threadgroup a brick, a thread a cell (the eight levels as two sets of four bytes:
  `max`, `subsat`). A brick whose cells changed is stamped, so the next frame lists it and its neighbors again. In a still
  scene the list is empty and nothing runs; a light placed or broken settles over a few frames (light moves at least 4
  blocks a frame: a torch's 14 in 4 frames).
- **Resolve into filtered textures.** The bricks that changed are resolved into an RGBA16Float 3D texture (the linear
  color of every bucket but fire, each bucket's color times vanilla's block light curve at its level, and whether the
  cell is open: not opaque) and an RG16Float one (the fire bucket's light, and the curve at the brightest bucket's level:
  vanilla's equivalent), all premultiplied by open.
- **Sampling.** The relight samples both textures once (trilinear) half a block in front of the pixel's face (in the
  cell the face looks into; plants in their own) and divides by the open share: light is averaged over the open cells
  around the sample only, so solid cells don't darken it and the far side of a 1-block wall never reaches it (along the
  face's normal the sample is at a cell's center; across it, the cells beside the face's own). Within 16 blocks of the
  volume's edge it fades to vanilla's block light.
- **Vanilla's level as the sanity reference.** The G-buffer has vanilla's block light at the pixel (smooth lighting's).
  The volume's vanilla equivalent may not pass the curve at vanilla's level plus one level (`METALMC_CLSLACK`): above
  it the colored light is scaled down to it, so nothing glows where vanilla is dark (a stale cell, a leak, light through
  a slab that vanilla blocks). Where it falls short of the curve at vanilla's level less one level (outside the volume,
  sections not uploaded yet, a light the volume doesn't know), vanilla's own block light makes up the share it misses, so
  nothing is darker than vanilla either. Where vanilla's level is 0 the volume isn't sampled at all (most terrain by day).
- **As bright as the game's own lightmap.** The curve is vanilla's lightmap formula at the default brightness; the
  relight scales the colored light by the game's lightmap at the pixel's level (its luminance, less level 0's) over the
  curve there (clamped to 0.5-2), so the brightness setting, night vision, the darkness effect and vanilla's own torch
  flicker apply to it as to vanilla's block light. Offline, where the lightmap is that formula, the ratio is 1.
- **Fire flickers:** the fire bucket's light times 1 + 0.12 x two octaves of smoothed noise in time (7 and 1.9 Hz;
  `METALMC_CLFLICKER`), applied in the relight (the volume holds it apart), all fires together.
- **Colors** (`clBucketSpecs`): a chroma shown at a luminance; the warm buckets about as bright as vanilla's torchlight
  (whose luminance the curve is), the saturated ones less (they read brighter than their luminance).

  | Bucket | Chroma (linear) | Luminance | Lights |
  |---|---|---|---|
  | warm | 1.00, 0.56, 0.25 | 0.95 | torches, lanterns, glowstone, jack o'lanterns, redstone lamps, copper bulbs; + green: glow berries, ochre froglights; + lava: shroomlights |
  | fire | 1.00, 0.46, 0.15 | 0.95 | fire, campfires, candles, lit furnaces, trial spawners (flickers) |
  | soul | 0.18, 0.76, 1.00 | 0.85 | soul torches, lanterns, fire and campfires, sculk; + white: conduits |
  | red | 1.00, 0.07, 0.03 | 0.55 | redstone torches and ore |
  | lava | 1.00, 0.30, 0.05 | 0.90 | lava, magma |
  | white | 0.78, 0.90, 1.00 | 0.95 | end rods, beacons, light blocks; + soul: sea lanterns |
  | green | 0.36, 1.00, 0.30 | 0.85 | verdant froglights, copper torches and lanterns; + soul: glow lichen, sea pickles |
  | purple | 0.62, 0.28, 1.00 | 0.70 | amethyst, crying obsidian, portals, enchanting tables, respawn anchors; + red: pearlescent froglights |

- **Where the blocks come from.** In the game, `metalmc.light.ColoredLight` sends every overworld chunk the client loads
  (the client thread copies its sections, a worker thread converts them) as vanilla describes each block: its light
  dampening (`getLightDampening`, 15 for `isSolidRender`), its emission (`getLightEmission`: lit furnaces, candles by
  count, sea pickles, respawn anchors all as vanilla has them) and a color class from its name (`mmc_cl_classify`). Every
  block change after that (`ClientLevel.sendBlockUpdated`, after the chunk holds it) is sent if it changes the block's
  code; if a conversion of its chunk is still queued, the chunk is captured again instead. A new level starts over.
  Offline the same codes come from the region files' block states (names and properties: vanilla's emission levels;
  dampening 15 for full opaque blocks, 1 for water and leaves, 0 for the rest).

### Verified offline (2026-10-05, no game)

`tools/litflow.swift` with `LITFLOW_CL=1` (claudeworld-merged, the game's call sequence, the relight in its own pass and in
the anti-aliasing's resolve, our sky; 1728 x 1117 pictures, means of 16 frames; `bench_out/agents/coloredlight/`):

    LITFLOW_CL=1 LITFLOW_VIEW=0.5,227.6,2.5,0,47 .build/litflow .build/release/libMetalMCNative.dylib \
      fixtures/claudeworld-merged <out> sky time

- **The test gallery** (`mod/src/client/resources/metalmc/coloredlight_scene.txt`, vanilla commands, which the game's
  tour runs too): a stone floor floating at y 199 with 18 cells walled 4 high (1-block walls), one light each: torch,
  soul torch, redstone torch, lantern, soul lantern, campfire; glowstone, sea lantern, end rod, the three froglights;
  shroomlight, jack o'lantern, crying obsidian, amethyst, glow lichen, a lava pool; and in front a torch, a soul torch
  and a redstone torch 6 blocks apart, magma, candles. Offline the scene is applied to the light's store and to the LOD
  (its chunks decoded from the region files with the scene on top and handed to the LOD as live chunks), whose level 0
  stands in for vanilla's near terrain.
- **The flood fill is vanilla's, exactly.** The GPU's light, read back, against a flood fill on the CPU over the same
  codes (per color, vanilla's rule, the volume's edges dark): 0 of 8,388,608 cells differ, at each of three views and
  after placing a soul torch and breaking it again; the GPU's codes are the store's in every cell.
- **No change without the switch.** The anti-aliasing's resolve, lit mode's own pass, the far field and the GI cache's
  shader sources under nine `METALMC_EXP` sets without `coloredlight` (none, lit, lit with shadows, the sky, the GI cache,
  water, HDR, sky alone, sky with HDR) are the base build's byte for byte (36 sources). Every variant compiles with it on
  (the resolve's 16 and 32 pixel tiles with and without the sky and the relight; the relight's pass on RGBA8,
  RGBA16Float and RG11B10Float; with water and the GI cache), and the flood fill takes 512 threads a threadgroup.
- **Nothing glows where vanilla is dark:** where vanilla's level is 0 the relight doesn't sample the volume, so those
  pixels are vanilla's. Checked on the frame as drawn: the colored capture against vanilla's (both in the same place of
  the shadows' and the sky's 64-frame cycle), with a second vanilla capture after it as the yardstick (far terrain whose
  shadow structures still build changes a few hundred pixels between any two captures). Where vanilla is dark the
  colored frame differs from vanilla's at 0 pixels from above (the two vanilla frames: 0), at 251 and 106 of 249,357
  low over the gallery (vanilla's own: 393 and 347), and at 141 in the real-terrain view (vanilla's own: 152), where the
  volume touches nothing at all; a glow would show at thousands (the volume has light under 60,474 of those pixels low
  over the gallery). In the debug view that does sample there, the check takes
  the volume's light to the curve at one level (0.3% of full); the volume had light where vanilla's reference didn't
  only around the gallery's candles, amethyst and crying obsidian: the LOD, which stands in for vanilla offline, doesn't
  know them as light sources (in the game vanilla does).
- **The look** (midnight, the gallery from above, `cl-above-midnight-pair.png`: vanilla's block light left, colored
  right; `cl-low-midnight-colored-taa.png` low over the mixing area through the anti-aliasing): every cell its own
  color, stopped by its walls and spilling faintly over them; the mixing area red, cyan-white and orange where the three
  lights overlap. Over the relit terrain with block light the mean luma is vanilla's (52.2 against 52.2 from above,
  53.8 against 54.0 low): the colored light is as bright as vanilla's, its color and shape are what change. The check
  (debug view 12, `-check.png`) applies the colored light to 99.4-100% of the pixels vanilla lights; it scales it down
  to vanilla's level at 1.3-1.7% (the LOD's flat per-face light against the volume's trilinear one, and the sources the
  LOD doesn't know). At noon the colors show only in shade (`cl-above-noon-pair.png`).

### Costs (offline, M3 Pro)

The volume's work alone, in a command buffer of its own per frame (`mmc_debug_cl_bench`; the gallery from above, 6,873
sections in the store), and the relight at 3456 x 2234 with the volume on and off on one G-buffer (the code compiled in
either way; medians of 60, alternating; other builders' work shared the GPU, so fastest-of figures are noise):

| | GPU ms |
|---|---|
| from scratch (joining, a teleport): every section uploaded, light from nothing | 2.7-6.3, 2.9-5.3, 3.4, 2.5-2.8, 0.1-0.15 over the first 5 frames (5,800-8,500 bricks of 16,384), then idle |
| still | 0.007-0.008 a frame (the listing over 16,384 bricks; nothing to fill) |
| moving at the flight's 20 blocks/s (1/6 block a frame), 240 frames | median 0.007, mean 0.015-0.019, slowest 0.92-1.11 (the frame a slab of 128 sections enters) |
| moving at 120 blocks/s | median 0.007, mean 0.07, slowest 1.24-1.27 |
| a torch placed or broken | 0.025-0.042 for 4 frames (27, 36, 60, 81 bricks), then idle |
| the relight's own pass | 1.10 -> 1.19-1.21 (+0.09-0.10) |
| the anti-aliasing resolve with the relight in its load | 2.35-2.38 -> 2.47-2.51 (+0.10-0.14) |

Memory: 151 MB on the GPU (light 33.5, codes 16.8, the two filtered textures 100.6), plus the store's copy of the blocks
around the player on the CPU (8 KB per section that isn't one block throughout: 7,886 sections in the game at render
distance 12, 63 MB).

### In game (2026-10-05, native 3456 x 2234, TAA on)

- **It runs.** `BENCH_FIXTURE=claudeworld-merged bash tools/bench/bench_lod.sh clTourOn2 2048 -PbenchTour=coloredlight
  -PbenchTrace=1 -Ptaa=true -PmetalExp=lit,nearchunks,rtshadows,sky,coloredlight`, and `clTourOff2` without
  `coloredlight` for the pictures to compare: no errors; "coloredlight: volume 256 x 128 x 256, 151 MB, 4 passes a
  frame"; the Java side had sent 7,886 sections when the world loaded and the volume was full and lit within 4 frames
  (2,052 sections uploaded); the gallery's 4,174 block changes (the tour builds it with `fill` and `setblock`), the
  roof's 3,763 and the later swaps reached it through the block-change path, and after the teleports to the two caves
  below (2,304 and 1,957 sections entering the volume) the fill ran in 10 and 26 frames and was idle again.
- **The tour** (`-PbenchTour=coloredlight`, `metalmc.bench.ColoredLightTour`): the gallery at midnight from above and
  low over its mixing area, at noon, sealed in by a roof and outer walls (a cave: no sky light inside), after its three
  mixing-area lights are swapped and a lava pool is poured beside them, and two natural caves of the fixture with lava
  (found by litflow: open around, a line of sight to lava): one at y 20 under the gallery, one at y -51 over a lava lake.
  Pairs, vanilla's block light left and colored right: `bench_out/agents/coloredlight/game/pair-tour-<step>.png`.
  - At night every cell has its own color, stopped by the cell's walls; the mixing area red, cyan and orange, white where
    they overlap. (The fire bucket's flicker doesn't show in a still.) At noon the colors show only in shade.
  - Sealed in, the room is dark but for the colored pools and their tint on the ceiling; vanilla's version is one warm
    white.
  - The swaps relight within a few frames, and the poured lava floods the floor and the ceiling around it with its
    orange-red, with the soul torch's cyan on the wall beside it (`09-cl-cave-edit-side`).
  - The lava lake at y -51 (`11-cl-deep-lava`): the deepslate walls and ceiling around it orange-red where vanilla's are
    a warm gray-white; the most convincing picture of the set. The cave at y 20 (`10-cl-lava-cave`): glow lichen's teal
    on the wall in front, the lava's orange-red low on the left.
- **Cost** (per-pass times, `-PbenchTrace=1`, `tools/bench/passes.py`; 3 traced frames per view, the same views without
  the volume in `clTourOff2`):

  | View | the volume's pass | the resolve (+ relight), without -> with |
  |---|---|---|
  | night, low over the gallery (about half the pixels block-lit) | 0.007-0.009 | 2.12 -> 2.68 (+0.56) |
  | the sealed room (nearly every pixel block-lit) | 0.009-0.016 | 2.55 -> 3.17 (+0.62) |
  | the lava lake at y -51 (every pixel) | 0.007-0.016 | 2.44 -> 3.10 (+0.66) |

  The volume's pass is the listing alone (the scenes are still); the resolve pays for the two 3D samples on every
  block-lit pixel, the flicker's noise and three evaluations of the curve (before the calibration, one curve fewer, the
  sealed room cost +0.53). The G-buffer passes are unchanged (0.21 and 0.69 ms both ways).
  - The real-terrain flight (`bench_lod.sh clFly 32768 -PbenchY=150 -PbenchFly=20 -PbenchExtraWait=600 -PbenchHitches=1
    -PbenchTrace=1 -Ptaa=true -PmetalExp=lit,nearchunks,rtshadows,sky,coloredlight`, by day): 129.2 fps, p99 9.85 ms,
    1,481 frames over 8.33 ms; the volume's pass 0.007 ms (median of 21 traced frames) while it follows the flight
    (1,400-1,540 sections uploaded every 1,200 frames as they stream in and the volume moves); the resolve 1.99 ms. Tonight's
    lit flight without it (`v_lit`, the same settings, the main checkout's queue): 126.0 fps, p99 10.17, the resolve
    1.89 ms. So by day about +0.1 ms in the resolve, the rest within run-to-run noise (not an A/B in one checkout).
- **Lab mode's natural scenes** (`tools/bench/lab.sh`, the full look `lit,nearchunks,rtshadows,sky,gi,water` with and
  without `coloredlight`, scenes of `tools/bench/scenes.json`; pairs, vanilla's block light left:
  `bench_out/agents/coloredlight/lab/pair-<scene>.png`): no errors with the GI cache and water on. (The first session
  with colored light took its switches from the settings, below; the rest from `METALMC_EXP`.)
  - The plains village at midnight (`night_torches`, its 64 wall torches): the lit walls, paths and lamp posts a warmer
    orange; mean luma 21.5 -> 21.6, mean chroma 16.3 -> 17.6 (its light was warm already, so the change is mild).
  - The mineshaft corridor lit by its own wall torches (`mineshaft`): warmer, and brighter: mean luma 68.0 -> 75.4 and
    70.5 -> 74.8 in two pairs of sessions (vanilla's own frame differed by 4% between sessions). Its walls are at levels
    11-14, where the volume's sample half a block in front of a face can sit up to the one level of slack over vanilla's
    smooth light. With the slack at 0 (`-PclSlack=0`, which pins the brightness to vanilla's) the mineshaft is 73.1
    against vanilla's 70.5 and the torch cave's two pictures 29.8 and 34.0 (slack 1: 30.2 and 34.3; vanilla's 30.7 and
    32.6): the slack is about half of the mineshaft's difference, the rest within what vanilla's own frames vary, and
    the pictures look the same; the default stays at 1 (light adds up where colors overlap).
  - The torch cave (`torch_cave`: ten torches and two lanterns its setup places on a cavern floor). The scene's own
    camera has a clear line to two of the twelve (checked against the fixture's blocks through the colored light's
    store) and its pictures are a dark rock face with or without colored light, so these are from a spot in the cavern
    that sees all twelve, `tp 110.5 21.98 -58.5 286 40`: as placed (`torch_cave_warm`); with a soul lantern, a soul
    torch, a redstone torch and a campfire in place of four of them (`torch_cave_mixed`: the soul lights whiten the
    torches' orange where they overlap, the redstone torch a small red pool); and with a sea lantern and pearlescent and
    verdant froglights in place of three more
    (`torch_cave_showcase`: purple, green and cool white pools meeting the torches' orange across the cavern floor,
    where vanilla's is one warm white; the best picture of the set). Mean luma 30.7 -> 30.2, 30.0 -> 30.6, 32.6 -> 34.3.
    The world's own ticking (fluids settling, plants, leaves) sent the volume 15-140 block changes a second there; the
    fill ran in 16-41% of the frames, over 50-60 bricks.
  - A trip to the Nether and back (`execute in minecraft:the_nether run tp`, then the overworld again: a new level each
    way): the store started over (10,133 sections, then 5,885 sent again as the overworld's chunks came back) and the
    village at midnight was colored as before (`pair-night_torches-before-after-nether.png`: before left, after right,
    the latter at slack 0).
- **The settings' route.** With no `METALMC_EXP` in the environment and `lighting`, `sky`, `bounceLight`,
  `waterReflections` and `experiments=rtshadows,coloredlight` in `config/metalmc.properties`, the game turned on
  "lit,nearchunks,sky,gi,water,rtshadows,coloredlight" and the colored light's Java side sent 8,560 sections. That
  needed a fix outside this feature: the settings reach the native side by `setenv`, which Java's `System.getenv` (a
  copy taken at startup) never sees, so `MetalLit.ENABLED` and `MetalNearChunks.ENABLED` had stayed false (lit mode's
  relight and the near chunks off on the Java side while the native side had them on); their `experiment()` now falls
  back to `MetalMCConfig.nativeExperiments()`, as `MetalColoredLight`'s does.

### Shadows from block lights, and block light bounced through the GI cache

Both part of `coloredlight`: the shadows with `rtshadows`, off by default (`-PclShadow=<strength>`: 0.75 to try them),
the bounce with `gi`, on (`-PclBounce=<gain>`, 1; 0 none). Without `coloredlight` the cache's kernels, the relight and the resolve are
main's text (the 36-source check).

- **Shadows** (`ColoredLightShadows.swift`). The volume spreads light as vanilla does, around a pillar and into the space
  behind it, so a torch casts no shadow. After the sun's shadow rays and before the relight, one ray per 4 x 4 pixels (a
  different pixel of the block each frame, which the anti-aliasing accumulates, as the sun's) goes from the surface
  toward the light its light comes from, through the sun shadows' acceleration structures (`RtShadows.takeStructure`:
  the LOD's blocks, near terrain included; no entities, and partial blocks as the LOD has them). The light is found in
  the volume, with no list of lights: in the cell the face looks into a bucket is picked by its share of the cell's
  light (a different pick each frame where colors mix), then that bucket's light is climbed, each step to the brightest
  of the six neighbors, until none is brighter. Vanilla's rule makes light fall at least a level a cell away from a
  light, so that cell is the light (at most 15 steps, 6 reads each). The ray is aimed at a point in the middle of the
  light's cell that moves every frame (soft edges) and stops where it enters that cell (an opaque light, glowstone, is
  in the structure); a face turned away from the light is in shadow. The relight takes the colored light (not vanilla's
  make-up) down by the strength where the ray was blocked. Light that reached a pixel around a corner, which vanilla's
  rule and the volume give at nearly full strength, is shadowed too: only bounced light (below) reaches there.
- **Bounce** (`ColoredLightBounce.swift`, splices in `Gi.swift`). The cache's cells hold the sky's and the sun's light
  per unit of the frame's daylight, which block light doesn't follow, so block light has a channel of its own: a half4
  per slot (16 MB at 2 M slots), rgb in absolute light and the cell's fingerprint in a, so a slot another cell took
  over reads as empty and no initialization path changes. The cache's update fills it where its bounce rays hit: the
  volume's colored light half a block in front of the hit (as the relight samples it) plus the hit cell's own bounced
  block light (so it bounces on), times the hit's color, averaged with the same weights as the sky light. The request
  resolves it beside the sky light (the same 2 x 2 cells, bilinear) into a half-resolution RG11B10Float texture, bound
  in the relight's `giStandIn` slot that had been kept free, and the relight adds it to the block light where vanilla's
  level isn't 0 (so caves vanilla leaves dark stay dark). The cache runs before the volume's update each frame, so it
  reads the volume as the last frame left it.

In game (lab sessions with the full look, `lit,nearchunks,rtshadows,sky,gi,water,coloredlight`, at the torch cave as
placed and with the showcase swaps, the mineshaft and the village at midnight; pairs in
`bench_out/agents/coloredlight/lab2/`: `shadows-<spot>.png` without | with the shadows (0.75, no bounce),
`both-<spot>.png` neither | the bounce alone; the tour's with both in `gameG/`):

- **The first shadows were noisy.** Rays from a wall toward a light on that wall run along it, and from 0.03 block off
  the surface (the sun's offset) they hit it again: salt-and-pepper shadow on the walls beside the lights, and the
  scenes 14-17% darker in mean luma (torch cave 34.3 -> 29.6, mineshaft 74.7 -> 61.8, village 21.6 -> 19.5). Now the
  rays start 0.1 block off, aim within 0.15 of the light's middle, and the relight samples the traced texture
  bilinearly. With those (`shadows-<spot>.png`): the rock steps of the torch cave are dark on their sides away from
  the lights, and the lava lake's foreground rocks cast real shadows across the cave (tour step `11-cl-deep-lava` in
  `gameG/`: without | with both), the look this is for; the open gallery barely changes (every floor cell sees its
  torch). Mean luma -13% in the torch cave (30.4 -> 26.4 as placed, 34.3 -> 29.4 with the showcase lights), -16% in the
  mineshaft (74.7 -> 62.4), -8% in the village (21.6 -> 19.8): light that went around corners now gets there only as
  bounce. Still grainy on walls right next to their light (the mineshaft's beam and the wall beside it), which the
  anti-aliasing doesn't settle: so they're off by default (`-PclShadow=0.75` to try them).
- **The bounce shows no effect yet** (`both-<spot>.png`: neither | the bounce alone, the shadows being off by default by
  then): mean luma 30.4 -> 30.3 and 34.3 -> 34.5 in the torch cave, 74.7 -> 73.6 in the mineshaft, 21.6 -> 21.6 in the
  village, and the pictures look the same. It runs without errors (the kernels compile with it, the cache logs as
  before), but whether its channel reaches the image (a plumbing fault: the volume's inputs at the cache's update,
  the block channel's resolve, its binding in the `giStandIn` slot) or is just weak (a floor lit by a torch bounces a few
  percent of its light back) isn't established. Next: a debug view of the channel (`giStandIn` alone), then its
  plumbing or its gain (`-PclBounce`).
- **Costs** (traced coloredlight tour with the GI cache, `-PbenchTrace=1`, 3 frames each low over the gallery at night,
  in the sealed room and over the lava lake; `clTourG0` without the two, `-PclShadow=0 -PclBounce=0`, and `clTourG1`
  with both, shadows at 0.75): the shadows' own pass 0.80-0.89 ms where most pixels have block light (0.35 at the lava
  lake); the cache's request and resolve 0.89 -> 1.13 ms (the block channel's resolve with data in it; G0 already has
  the channel's storage and resolve, at gain 0), its update unchanged (0.35 -> 0.31); the anti-aliasing resolve 3.53 ->
  3.55 median (in the sealed room 3.65-3.85 -> 3.92-4.50: the bilinear shadow sample on every block-lit pixel). The
  sun's shadow pass read 0.22 -> 0.74 ms in the same frames: the block lights' pass runs right after it with nothing
  to wait for, so the GPU overlaps them and the sun's timestamps take in part of it. Memory: the channel 16 MB (2 M
  slots x 8 bytes), the half-resolution block light 11 MB, the shadows' quarter-resolution texture 0.5 MB.

### Not done, and next

- **Bounce through the GI cache:** built (above) but with no visible effect in the lab scenes yet: find out whether it
  reaches the image (a debug view of its channel) before tuning it. Then: it doesn't follow the fire's flicker, a
  cell's block channel changes only as fast as the cache updates it, and far cells (past the volume) get none.
- **Shadows from lights:** built (above), off by default for their grain on walls next to their lights: a few frames of
  history of their own (the sun's visibility gets by on the anti-aliasing's), or two rays a sample. Entities, fences
  and slabs cast none (not in the structures, or as the LOD has them); one light per ray (the anti-aliasing mixes the
  picks where colors overlap). With them on, scenes are 8-16% darker: light that went around corners comes only as
  bounce, so they want the bounce working.
- **One sample in the resolve** (the +0.6 ms): not started.
- **Partial blocks** (slabs, stairs, walls) let light through in the volume where vanilla stops it at their full faces;
  vanilla's level caps what leaks (the check), so it shows as a tint, not a glow.
- **The volume's reach:** colors within about 100 blocks of the camera horizontally and 40 vertically (its size less the
  recentering margin and the fade); past that vanilla's warm light. Flying high over terrain, the ground is often below
  it. A second level of 2-block cells (512 blocks across) would carry colors to villages seen from afar, or the LOD's
  light lists could carry a color class to its levels 0-4.
- **Faster:** the relight's part costs 0.6-0.66 ms in the resolve with the whole screen block-lit; one RGBA16Float sample
  would do if the fire bucket's light and vanilla's equivalent shared a texel with the color (e.g. RGB9E5 color, the open
  share and vanilla's equivalent in a second half of one RGBA32 texel). From scratch (joining, a teleport) the fill runs
  3-6 ms a frame for 4 frames: several passes per memory pass (a brick with its halo in threadgroup memory) and spreading
  the uploads would take that down.
- **Not colored:** entities, the hand, particles, water and translucent blocks keep vanilla's light (they aren't relit);
  a face at block light 15 next to a light that isn't opaque (the floor under a lantern, around lava) is a light source to
  the relight, full bright as before.
- **Flicker** is one clock for all fires; torches don't flicker.
- **Colors** are a first pass on the test gallery (`clBucketSpecs`, `ClClass.buckets`); `-PclGain`, `-PclFlicker`,
  `-PclSlack` to tune (the game logs them with the volume's size).
- **Brightness against vanilla's:** the same within a few percent of mean luma in the gallery, the village and the torch
  cave, 6-11% brighter in the mineshaft (half of it the slack, above). `-PclSlack=0` pins it to vanilla's at the cost of
  light adding up where colors overlap; in the lab the two look the same.
- In the game: the gallery, two natural lava caves, the village at night, the mineshaft and the torch cave; not yet:
  the Nether (lit mode is overworld only), many lights flickering in motion, a long flight underground, multiplayer.

## Post-processing (prototype, `METALMC_EXP=post`, 2026-10-05)

`Sources/MetalMCNative/Post.swift`, `metalmc.backend.MetalPost` and a call in `GameRendererLodMixin` (after the
anti-aliasing, before the hand); hooks marked "Post hook" in Sky.swift and Hdr.swift. Off by default; meant for
`-PmetalExp=lit,nearchunks,rtshadows,sky,post` (with `gi`, `water`, `hdr` as wanted). It needs our sky to have drawn the
frame (`Sky.frameActive`): elsewhere (the Nether, the End, under water) the frame is vanilla's and post doesn't run.

### The frame in scene-linear light

Until now the level became display light as it was drawn: the sky pass and the aerial perspective step (in the
anti-aliasing's resolve) applied the tone curve, and the history held display light. Bloom and eye adaptation need the
light before that, so with `post`:

- **The main target is float even on an SDR display** (`floatMainTarget` in Hdr.swift: RG11B10Float as with HDR, or
  RGBA16Float with `METALMC_HDRFORMAT=rgba16f`; `mmc_hdr_enabled` reports it, so MainTargetMixin and the rest of the
  Java side treat it like HDR's). The layer and the present stay 8-bit SDR unless `hdr` is on. RG11B10Float is 32 bits
  like RGBA8: the bandwidth of every pass is unchanged.
- **The sky pass and the aerial perspective step leave the tone curve out** (Sky.swift: the knee pushed to 65504, so
  `skyToneMap` is the identity): the frame holds scene-linear light, sRGB-encoded (extended past 1: the sun's disk at
  about 35,000 encodes to about 82). The relight doesn't clamp on a float target. The anti-aliasing resolves this into its
  float history as before (its neighborhood clip in YCoCg of the encoded values, which compresses highlights the way a
  tone-mapped resolve would; nothing in Taa.swift changed).
- **One tone curve.** Post's composite is the only place light becomes display light; with `hdr` it maps into the
  display's headroom instead of compressing into SDR (below). Nothing is mapped twice.

### The chain (after the anti-aliasing's resolve, in the same command buffer)

Post reads the anti-aliasing's new history (without anti-aliasing, the frame itself), then takes the anti-aliasing's
waiting copy into the frame and leaves its own in its place: the composite's display light, with the same contrast-
adaptive sharpening (on the values over the display's largest, so it's the same in HDR) and dither (relative on the packed
frame), folded into the hand's pass like the copy it replaces (pass folding, Backend.swift).

1. **Bloom** (`post_bloom_first`, `post_bloom_down`, `post_bloom_up`): seven levels below the frame, from half
   resolution down to 1/128 (27 x 17 at the panel's resolution). Down: Jimenez's 13-tap filter (five overlapping 2 x 2
   boxes of bilinear taps). The first level reads the frame once per 16 x 16 threadgroup into threadgroup memory in
   linear light (a bilinear tap of the sRGB-encoded frame would average wrongly) and weighs each box down by how far past
   a firefly limit it is (256 times the exposure's mid-gray): a softened Karis average, so a glint's few pixels can't make
   the bloom flicker while a large bright area (the sun's disk, lava) keeps its weight. Light sources weigh 6 times their
   light (lit terrain whose G-buffer texel has block light 14 or more and whose depth key matches: torches, lanterns,
   glowstone, lava). The same pass puts lit terrain's albedo luminance in alpha (1 for anything else) for the light
   meter. Up: each level's own light times its share plus the level below through a 3 x 3 tent; the shares
   are equal (`POST_BLOOM_SHAPE` 1) and sum to 1, so the bloom is a normalized blur with a long tail (a sum of
   doubling-width Gaussians of equal energy falls off like 1/r^2). Composite: `mix(scene, bloom, 0.05)`: energy
   conserving, what the glow gets the source loses. Only light much brighter than its surroundings shows it: the sun
   and the sky around it, glints on water, torches at night and in caves.
2. **Eye adaptation** (`post_histogram`, `post_exposure`): a 128-bin histogram of log2 light (2^-14 to 2^10, scene
   units) over the bloom's second level (a quarter of the resolution), each texel weighing 1 at the screen's edges up
   to 4 a little below its middle (leaning away from the sky). In lit mode it meters halfway between luminance and
   light: a texel's luminance over the square root of its mean albedo luminance (alpha; sky and water count as 1), so a
   dark forest canopy reads much closer to the plains under the same sun, while a sunset's bright sky still counts
   (In game, below, for luminance alone and light alone). The weighted mean of the log2 values between the 30th and
   85th percentiles (a black corner, the sun and the light sources don't count); against the reference (noon outdoors,
   the metered value that gets 0 stops) the exposure makes up 20% of a darker scene's difference within 2.5 stops of it
   and 70% past that, and 30% of a brighter one's, within -2 and +2 stops: daylight's own swings (dusk, shade, a
   forest), which the sky's adaptation to the sun's height already covers, barely move it; night, caves and interiors
   open up (partial adaptation: readable, still darker than day). Followed on the GPU with no readback: toward a darker exposure
   (into the light) with a time constant of 0.45 s, toward a brighter one (into the dark) with 2.2 s, so stepping out of
   a cave overexposes and blooms for a moment, then settles, and walking into one opens up over a couple of seconds.
   A camera jump (16 blocks in a frame: a teleport, a respawn, a lab `scene`) snaps it. In stops on top of the sky's own
   adaptation to the sun's height (Sky.prepare, up to 3.5 stops at night), which stays, so the units above (sun 12,
   vanilla's white 1) are unchanged and post's exposure is 0 at the reference.
3. **Light shafts** (`post_shaft_mask`, `post_shaft_blur`, `post_shaft_color`), at a quarter of the resolution: the
   share of each texel's 4 x 4 pixels that are sky (depth 0; vanilla's clouds write depth, so they cast shafts too),
   blurred radially toward the sun's position on the screen (after Mitchell, GPU Gems 3; two passes of 12 taps, the
   second spanning one tap of the first: the weighted mean of 144 taps along the line), times the sky's own light in the
   texel's direction (`skyLuminance`: its glow around the sun; below the horizon, the horizon's). Over terrain and
   clouds that adds the air in front lit where the sun gets through (22% of the sky's light, times how much of the way to
   the sun is open, times how much air there is in front: `1 - exp(-distance / 192 blocks)` at the texel's mean depth, so
   a tree nearby gets little and a ridge a few hundred blocks off nearly all); over the sky it darkens it where something
   between it and the sun shadows the air (30% of the sky's light, times how much is blocked): crepuscular rays both
   ways, from the sky's own colors and brightness. Faded out as the sun leaves the screen (fully gone a screen's width
   past its edge), below the horizon, behind the camera, in rain.
4. **The tone curve** (`post_composite`, `METALMC_TONEMAP`): AgX by default (Sobotka's, as in Blender 4; the polynomial
   fit and Rec.709 matrices of Wrensch's minimal version), with a look (saturation 1.3 on its sigmoid's output, power
   1: its "punchy" look's saturation without its contrast, which crushed dark scenes): bright saturated light runs to white without the hue skews a
   per-channel curve on the original primaries gives (a sunset's orange stays orange as it brightens). `aces` is
   Hill's fit of the RRT and ODT (more contrast, more saturated, exposure 1.6 to keep the mid-tones), `gt` Uchimura's,
   `legacy` the sky's shoulder (the look before post: with `legacy` and the effects off, post draws the frame as before).
   **HDR output** (`hdr`): the same curve below a knee, and above it (luminance 0.8, exposed) the highlights carried on
   toward the display's headroom, `d * (1 + (H - 1) * (1 - exp(-t^2)))` with `t` the luminance past the knee over 4:
   zero slope where it starts, the curve's hue kept, the identity at H = 1. So SDR white stays where SDR puts it and the
   sun, its glow and glints go past it to the panel's peak. The screenshot copy (Hdr.swift) then rolls that into SDR as
   before; on an SDR display post's frame is SDR already and the copy only clamps it.

Every constant of the look is a `#define` at the top of `post.metal` (lab mode: edit and save, the chain recompiles in
about a second). Environment switches set their defaults: `METALMC_TONEMAP` (agx, aces, gt, legacy), `METALMC_BLOOM`
(0.05), `METALMC_SHAFTS` (0.22), `METALMC_POSTEV` (exposure compensation, stops), `METALMC_ADAPT=0` (no eye
adaptation), `METALMC_POSTVIEW` (1 the bloom alone, 2 the shafts alone, 3 the exposure meter in false color, 4 no tone
curve). The log says every 1200 frames: "post: N frames; exposure +x stops (target, metered log2 light)".

### In game (2026-10-05, lab mode: 3456 x 2234 fullscreen, TAA, vsync, LOD 8192, `lit,nearchunks,rtshadows,sky,gi,water,post`)

- **It runs.** "post: 3456x2234, bloom 7 levels down to 27x18, light shafts at 864x559, tone curve agx", reading the
  anti-aliasing's history; two sessions, the nine scenes of the lab tour each time, no errors; post.metal edited and
  reloaded live (0.4-0.8 s to compile). Screenshots go through the SDR copy (identity on post's SDR frame).
- **Before and after**, in the main checkout's `bench_out/agents/post/`: `before/<scene>.png` is lab3 (the same code and
  switches without post, commit cbb4db8); `final-defaults/` the committed defaults (session 3); `session1-final/` post
  metering luminance (AgX power 1.15); `session2-final/` metering light (power 1.0); `compare-<scene>.png` in each stacks
  before over after at 1100 px; `session1-first/` the first pass before any tuning. Mean 8-bit luma (the share of pixels
  under 8):

  | Scene | Before | Session 1 (luminance meter) | Session 2 (light meter) | Final defaults (session 3) |
  |---|---|---|---|---|
  | noon_overview | 114.2 (0.0%) | 111.9 (0.0%) | 112.8 (0.0%) | (the LOD hadn't loaded: no comparison) |
  | sunset_water | 115.1 (0.5%) | 119.2 (0.8%) | 141.3 (0.0%) | 121.5 (0.3%) |
  | torch_cave | 31.6 (4.6%) | 48.8 (18.4%) | 57.1 (4.3%) | 37.4 (the scene's camera moved in main) |
  | night_torches | 21.7 (9.9%) | 37.2 (8.9%) | 45.4 (3.4%) | 35.3 (8.4%) |
  | forest | 78.5 (6.6%) | 93.1 (9.1%) | 78.2 (14.8%) | 79.5 (14.1%) |
  | mineshaft | 67.6 | 88.1 | 86.2 | 80.0 |
  | rain | 72.8 | 88.7 | 94.2 | 83.1 |
  | water_closeup | 101.9 | 111.5 | 147.3 | 115.8 |
  | mountain_view | 103.7 (2.3%) | 103.0 (13.0%) | 103.9 (9.0%) | 99.9 (12.5%) |

  - Sunset: crepuscular rays fan out from the sun through the gaps in the clouds and darken the sky in their shadows, the
    sun blooms warm over the water with its reflection, the far shore sits in lit haze. The first pass (shafts 0.35) had
    them strong enough to wash the clouds and the far silhouettes; 0.22 keeps them.
  - Noon: as bright as before, the lava pool on the left glows, the sky a softer blue (AgX takes bright saturated light
    toward white), the clouds a light gray rather than white.
  - Caves and night: readable. The cave's walls come up from near black with their detail, the lantern and torches glow;
    the village at midnight shows its fields, river and houses, torches and windows glow, the sky stays a night sky.
    AgX's look power of 1.15 crushed a quarter to a third of the dark frames under 8; 1.0 keeps them (session 2).
  - The meter, three ways. Luminance (session 1): under the same sun the forest's dark canopy read two stops darker than
    the plains (log2 -4.9 against -2.9), so the forest got +1.4 stops and its sky washed out. Light, luminance over albedo
    (session 2): noon read -0.95 against its reference of -1.0, the forest matched the plains, caves and night came up
    without crushing; but a sunset's land is lit dimly under a bright sky, so sunset_water got +1 stop (the sky washed
    pale, mean luma 141) and so did dusk offline. The default is halfway (luminance over the square root of albedo,
    `POST_METER_LIGHT` 0.5), with the reference at the midpoint (-1.9). Offline that still read the default view's dusk and
    sunset 2.4-2.7 stops under noon, as the light meter did: at 60% partial adaptation they got +1.4 and +1.6 stops
    (mean luma 158 and 148 against 87 and 82). Dusk is that much darker (a top under a sun 8 degrees up gets 14% of
    noon's light) and the sky's own adaptation already gives some back, so the dark side now has two slopes: 20% within
    2.5 stops of the reference, 70% past it (night, caves). Session 3 ran those defaults from the built-in source: the
    sunset keeps its rays, its blue sky and the sun's warm bloom at about its old brightness (121.5), the forest and the
    mountain are where they were, night and the mineshaft come up, the torch-lit cave (main moved its camera to see all
    twelve lights) is bright where the torches are and keeps its black rock in front.
  - ACES (`METALMC_TONEMAP=aces`) tried live at sunset, noon and forest: more contrast, but it takes the sky around the
    setting sun to flat white where AgX keeps its gradient and color. AgX stays the default.

### Costs

Speed wasn't the goal tonight; these are the measurements to start from.

- **Offline, each stage alone** (`LITFLOW_POST=1 litflow ... sky time`, 3456 x 2234, the default view at sunset with the
  sun on screen, median of 40): bloom 1.46 ms, exposure 0.04, light shafts 0.18, composite 0.65, the copy into the frame
  0.40 (the anti-aliasing's own copy cost about that; post's replaces it). Whole frames (sky, main pass, shadows, the
  relight and aerial perspective in the resolve) with and without post, alternating: 8.50 against 6.14 ms, +2.36.
- **In game, traced** (`trace 5` in the lab, `passes.py` on the game log; compute passes' totals overlap their
  neighbors, so they don't add up): session 3 at sunset_water, the final defaults: bloom 1.38 ms, exposure 0.04, shafts
  0.20, composite 0.66, the copy 0.39 of fragment work (0.61 in session 1, before the composite's output went from
  RGBA16Float to RGB10A2; session 1's shafts read 1.42 there, overlapping the bloom). In session 1 sunset_water held
  119 fps at 120 Hz with the whole look and post.
- Where the bloom's 1.46 ms goes: its first pass reads the frame (4 bytes a pixel), the G-buffer (8) and the depth (4) at
  every pixel, 124 MB at the panel's resolution; the rest of the chain is small. Reading the G-buffer and depth once per
  2 x 2 block would take most of it away (below).
- Eye adaptation with the final defaults, offline (litflow's default view; `post-default-<time>-<label>.png` and
  `bench_out/agents/post/offline/` in the main checkout): noon +0.05 stops (mean luma 114.5 against 110.4 before),
  dusk +0.48 (132.2 against 86.6; the rest is the sun's bloom and shafts in a view that looks toward it), sunset +0.63
  (122.1 against 82.2), midnight +1.35 (39.5 against 22.8). In time, at 120 Hz steps: noon -> midnight at once, +0.57
  stops after 1 s, +0.89 after 2 s, +1.36 after 6 s; midnight -> noon, +1.34 -> +0.24 after 1 s, +0.12 after 2 s.
- The tone curves (`mmc_debug_post_curve`, litflow): at headroom 2, 4 and 8 every curve is the SDR one below the knee to
  the 4th decimal and reaches the headroom (AgX: 4 -> 0.84 in SDR, 1.24 at headroom 2, 2.04 at 4); AgX's outset matrix
  takes saturated highlights a little past 1 (the check flagged it), now clamped to the headroom.
- The final defaults' exposure in game (session 3's log, one line every 1200 frames, matched to the scenes by time):
  rain +0.30 stops (metered log2 -3.42), forest +0.70 (-4.69), the torch-lit cave +0.20 (-2.88: with all twelve
  lights in view it's well lit), mineshaft +0.24, sunset_water +0.24 (-3.09), water_closeup +0.16, mountain_view
  +0.24, midnight in the village +1.35 to +1.54 (-5.6 to -5.9). Daylight stays within a stop of noon; night opens up
  by about a stop and a half.

### Not done, and next

- **Volumetric light from the traced shadows.** The shafts are screen-space: they need the sun on or near the screen, and
  they can't show beams in the air when the sun is behind the camera (under a canopy, looking away from the sun). The
  better version marches each quarter-resolution pixel's view ray through a thin fog (8 steps, jittered per frame and
  accumulated like the shadows' rays), tests each step's visibility toward the sun with one ray into RtShadows'
  instance structure (it has `tlas`, `tlasAccels` and `origin`; an accessor and `useResources` are all it needs), and
  adds `sigma x phase(theta) x sun x visibility x transmittance` in the composite. At the shadows' measured rate (about
  0.3 ms per half-million rays) that's 1-2 ms at 4-8 rays per quarter-resolution pixel; temporal reuse would bring it
  down.
- Light sources only weigh more in the bloom; the image itself keeps them at lit mode's full bright (1). A torch flame
  that is 4-8 times brighter in the image too (and so goes white-hot through the curve) needs lit mode's relight to give
  emitters more than 1 (Lit.swift; with colored block light, its volume's emitters).
- Clouds and water keep vanilla's light: under the filmic curve vanilla's white clouds come out a light gray.
- Cheaper: the bloom's first pass reads the G-buffer and the depth at every pixel (93 MB at the panel's resolution, for
  the light sources and the meter's albedo); one read per 2 x 2 block would do for both and should take the bloom from
  1.46 ms to under 1. The composite and the copy could be one pass if the sharpening read its neighbors' tone-mapped
  values from threadgroup memory (a compute pass can't write the frame today: it has no shader-write usage).
- Lit mode's G-buffer layout is read in `post_bloom_first` (face code in x's top bits, block light in y's top byte, the
  depth key; colored block light left it as it was): a change to it has to follow there.
- No Purkinje shift (night desaturating toward blue) in the eye adaptation; lit mode's moonlight is already bluer.
- HDR output: checked offline (the curves at headroom 2-8, above), not yet seen on the panel.
- Not seen in game with the final defaults: noon_overview (the first scene of session 3 caught the LOD still loading);
  sessions 1 and 2 put noon within 2% of before with each meter, and the defaults sit between them.
- With the external pipeline (`METALMC_EXTPIPE`): not tried together.
