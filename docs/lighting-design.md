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
  it is.
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

### Next: the GI cache as the sky term (gi-design.md, "Integration" steps 1-3, not wired)

1. RtShadows' tile structures with the per-triangle material and face (`giTileGeometry` / `giBuildBlas`, or one
   geometry per quad range, zero copy) when `lit,gi` is on.
2. In `RtShadows.trace`, after the instance structure is built: `GiCache.shared.encodeFrame(cb, depth:, out:, code:,
   invViewProj:, cam:, origin:, accel: tlas, accels: tlasAccels, sunDir:, sunUp:)` into half-resolution irradiance and
   code textures, kept for the relight like the visibility; `giSky` from the atmosphere (its sky view table) so the
   cache's light is in the relight's units.
3. In `litRelightPixel`: `giUpsample(irr, code, q, face, rel)` where it has data (`w == 1`) replaces the sky term
   (`env[1 + face] x sky light curve`), times AO; the G-buffer's face and the relight's `rel` are what it needs.

## HDR's main target packed (2026-10-01)

HDR's float main target cost about a millisecond over the sky alone, mostly bandwidth: every pass that loads or stores
it moves 64 bits a pixel. It's now RG11B10Float by default (`METALMC_HDRFORMAT=rgba16f` for RGBA16Float), and so is the
anti-aliasing's history; the drawable stays RGBA16Float for EDR. The real-terrain flight with sky and HDR (far field
on): 124.7 → 136.3 fps, frames over 8.33 ms 2,166 → 964. Its 6-bit mantissas (5 in blue) banded the sky's glow around
the sun in rings with an absolute dither (`m9Tp`); the sky pass and the anti-aliasing now dither relative to the value
(1/64 of it, 1/32 in blue), and the sunset is as smooth as with RGBA16Float (`m10Tp`). No alpha: the render tour's
GUI, inventory, pause blur, particles, the Nether and the End look as before.
