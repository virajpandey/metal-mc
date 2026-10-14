# Lab mode: look iteration without relaunching

A game run costs about 4 minutes for a minute of useful time (gradle and the JVM, the resource reload, the world, the
LOD's warm-up). Lab mode keeps one game up and makes the look's loop a matter of seconds: our Metal shaders come from
files and are recompiled when they're saved, a control file moves the camera between saved scenes and takes
screenshots, and every run lands in a ledger.

## A session

```bash
bash tools/bench/lab.sh look1 &          # in the background (a Claude session: run_in_background)
tail -f mod/run/metalmc-control.log      # "ready: ..." once the world has loaded
echo "scene sunset_water" >> mod/run/metalmc-control.txt
echo "shot sunset-a"      >> mod/run/metalmc-control.txt   # -> mod/run/screenshots/sunset-a.png
# edit mod/run/shaders/lit_relight_header.metal (the relight) and save; about a second later shaderlab.log says
#   taa.metal compiled in 495 ms / lit.metal compiled in 439 ms / reloading taa, lit next frame / reloaded taa, lit
echo "wait 60"            >> mod/run/metalmc-control.txt   # 3 s: the anti-aliasing's history and the GI cache settle
echo "shot sunset-b"      >> mod/run/metalmc-control.txt
echo "quit"               >> mod/run/metalmc-control.txt
```

`tools/bench/labctl.sh` does the appending and waits for the results: `bash tools/bench/labctl.sh "scene forest" "shot
forest-a"` prints the two result lines once the shot is written (`-n 19 "tour x"` for a tour's 19 results).

Measured (2026-10-05, M3 Pro): the world is ready about 50 s after the session gets the GPU lock (the first LOD build
from a cold cache takes a minute or two more); a save to rebuilt pipelines is about a second (lit and taa compile in
about 0.5 s each); `tour` visits all nine scenes and writes nine screenshots in 74 s once the LOD is built (ten since cave_mouth joined).

`lab.sh [label] [gradle args]` is `bench_lod.sh` with `-PbenchLab=1 -PshaderDir=mod/run/shaders -Ptaa=true` and the full
look (`LAB_EXP`, default `lit,nearchunks,rtshadows,sky,gi,water`), on `claudeworld-merged` (`LAB_FIXTURE`), LOD to 8192
(`LAB_FAR`), at 120 Hz with vsync (`LAB_VSYNC=false` for uncapped), for at most 90 minutes (`LAB_MINUTES`). It builds the
Swift library under the GPU lock first, and like every game run it holds the GPU lock while the game is up: other
builders' GPU tests wait, so `quit` when done. `LAB_SCRIPT=file` queues a file of commands at the start. The game log is
`bench_out/run_<label>.log`.

## The control file

`mod/run/metalmc-control.txt`: append lines (`echo ... >>`, one line per write). Every tick the game takes the file's
lines into its queue (it renames the file away whole and reads it a tick later, so a line appended at the same moment
is never lost; rewriting the file to drop each line as it ran lost 3 of 17 lines appended in a loop), so the file
empties and `mod/run/metalmc-control.pending` lists what hasn't started. It runs one command at a time and logs each
result to `mod/run/metalmc-control.log` and the game log (`[metalmc-lab] 03:42:12.932 #14 shot lab1-noon_overview -> ok:
/.../lab1-noon_overview.png`). Lines starting with `#` are skipped. lab.sh empties the file at the start (stale
commands would run at once).

| Command | |
|---|---|
| `scene NAME` | go to a scene of `tools/bench/scenes.json`: pose, time of day, weather, its setup commands; done once its `settle` ticks (100) have passed, the LOD is built and its quad count has held for 3 s (after a jump of more than 256 blocks, only once it has changed: the LOD updates every 2 s), and the terrain around has compiled (at most 60 s) |
| `shot LABEL` | screenshot of the main target (the panel's resolution, no HUD) into `mod/run/screenshots/LABEL.png`; done when the file is written |
| `mask LABEL` | the class mask of the same view into `mod/run/screenshots/LABEL-mask.png` (lit mode; see "The class mask" below); done when the file is written, and the result lists the classes' shares |
| `burst LABEL [N] [SAVE]` | the flicker meter (below): the next N presented frames (32) tracked per pixel into `LABEL-flicker.png`, `-motion.png`, `-flicker.bin`, the first SAVE (0, at most 16) kept as `LABEL-fNN.png`; the result has the frame's numbers |
| `pan DYAW [DX DZ]` | turn (degrees) and move (blocks) the held pose every tick, smoothly (the frames interpolate between ticks); `pan 0`, `scene` or `free` stop it |
| `reload` | recompile every shader file now, changed or not, and rebuild their pipelines (normally the watcher does it on save) |
| `wait TICKS` | 20 ticks a second |
| `tour [PREFIX]` | `scene` + `shot PREFIX-NAME` for every scene (PREFIX: tour) |
| `trace [N]` | per-pass GPU timestamps of the next N frames (3) into the game log: `python3 tools/bench/passes.py bench_out/run_LABEL.log` |
| `quit` | stop the game |
| `time T`, `weather clear\|rain\|thunder`, `tp X Y Z [YAW PITCH]`, `free`, `cmd COMMAND`, `hud on\|off`, `fov D`, `status`, `scenes`, `echo TEXT` | `free` lets go of the held pose; `cmd` runs any server command; `status` prints the pose, time, rain, fps and the LOD's state |

Time and weather are frozen (the gamerules), mobs don't spawn, and the HUD is hidden. The pose is held every tick
until `free`, so `tp` and `scene` stay put. After a `tp`, `wait` before a `shot` (60-100 ticks): the screenshot is the
frame on screen, and the move and the chunks take a few ticks.

## Shader files

With `-PshaderDir=<dir>` (env `METALMC_SHADERDIR`), every library of ours compiled at runtime comes from
`<dir>/<name>.metal` (`Sources/MetalMCNative/ShaderLab.swift`). The game writes each file from its built-in source the
first time it compiles that library, so the directory fills with what the session's switches use. The shared headers
are files of their own, pulled in with `#include "name.metal"`, so one edit reaches every library that has it:

| File | From | Draws |
|---|---|---|
| `lit.metal` | `Lit.swift` litShaderSource | lit mode's relight pass, the sun and sky light (lit_env), the water's waves and sky map |
| `taa.metal` (two libraries) | `Taa.swift` | the anti-aliasing's resolve, which also relights (lit) and applies the aerial perspective (sky) as it loads |
| `sky.metal` | `Sky.swift` | the atmosphere's tables, the sky, the aerial perspective |
| `gi.metal` | `Gi.swift` | the GI cache's kernels (swapped in place: the cache keeps its light) |
| `rt_shadows.metal` | `RtShadows.swift` | the traced sun shadows |
| `lod.metal`, `far_field.metal`, `near_chunks.metal` | `Lod.swift`, `FarField.swift`, `NearChunks.swift` | the LOD's quads, the far field's march, our near terrain (with their G-buffer writes) |
| `hdr.metal`, `util.metal` | `Hdr.swift`, `Backend.swift` | the HDR present; clears and the present blit |
| `lab_mask.metal` | `LabMask.swift` | the `mask` command's classes (their rules are `#define`s at its top) |
| `sky_header.metal`, `lit_header.metal`, `lit_relight_header.metal`, `gi_upsample_header.metal` | skyShaderHeader, litShaderHeader, litRelightHeader, giUpsampleHeader | shared: the atmosphere's functions; the G-buffer's packing; **the relight (litRelightPixel) and the water's shading**, in both lit.metal and taa.metal; the GI upsample |

A watcher polls the files twice a second. A library whose files changed is recompiled off the render thread; if it
compiles, and still has every function the old one had, its pipelines are dropped between two frames and rebuilt from it
on their next use (the LOD's and the far field's compile in the background, so they're missing for a few frames; the
sky's tables and the anti-aliasing's history start over). A compile error goes to `<dir>/shaderlab.log` and the game log
once, with the libraries it broke and each error's file and line (`taa.metal, lit.metal failed to compile; the old
pipelines keep running: /.../lit_relight_header.metal:157:5: error: invalid use of 'this' ...`), and the old pipelines
keep running. The next save that compiles is picked up the same way.

`<dir>/.orig/` holds the built-in source each file was written from. `diff -u .orig/lit.metal lit.metal` is your edit,
to port back into the Swift string it came from (mind `\(...)` interpolations: the file holds what they expanded to
with this session's switches). A file you haven't edited follows the built-in source when that changes (someone's merge).
An edited one is kept, with a warning and the new built-in source in `.orig/<file>.new` to merge; delete the file to
start over from the built-in source at the next start (while the game runs, a deleted file leaves the shaders as they
are).

A new library joins by compiling through `ShaderLab.library("name", source) { _ in ... }` instead of
`device.makeLibrary(source:options:)`; the closure (lab mode only, render thread, between frames) forgets the library
and every pipeline cached from it, so the lazy setup that made them runs again (`Lit.ensureLibrary` is the pattern:
`library = nil`, the per-format caches emptied, `failed = false`). Pipelines made in the same lazy block as the library
need nothing more; a cache filled elsewhere has to be emptied in the closure, or it keeps the old shaders.

Not from files: vanilla's own shaders (translated from SPIR-V per pipeline, `mmc_pipeline_create`), the byte-copy kernel,
and the offline tests' shaders (FarFieldDebug, `mmc_debug_gi_reload`, RtProbe, the near chunks' reference renderer,
Sky's timing fill). What needs a restart: `METALMC_EXP` switches and the other `METALMC_*` settings (read once), and
Swift code.

## Scenes

`tools/bench/scenes.json`, read again at every `scene` (edits apply at once): `name`, `pos` (the feet; the eye is 1.62
higher), `yaw` and `pitch` (Minecraft's: yaw 0 faces south, 90 west, 180 north, 270 east; pitch > 0 looks down), `time`
(6000 noon, 12500 sunset, 18000 midnight), `weather`, `setup` (server commands on arrival: the fixture is a fresh copy
every session), `settle` (ticks), `note`. To add one: `tp X Y Z YAW PITCH`, `wait 100` (the chunks load), `shot`, look,
adjust, and copy the pose in (`status` prints it). Keep cameras out of y 190-198: the clouds are a slab there.

The ten on `claudeworld-merged`: noon_overview, night_torches (the plains village's own torches at midnight), rain (the
same village), forest, torch_cave (a cavern, torches placed by its setup), mineshaft (a corridor lit by its own wall
torches), sunset_water, water_closeup, cave_mouth (out of a shallow cave at the morning sun: the volumetric light's beams,
2026-10-10), mountain_view. The notes in the file say where each is and how it was found. The SEUS twins (census 1) cover
all but cave_mouth.

## The class mask

`mask LABEL` (`Sources/MetalMCNative/LabMask.swift`, lit mode only) writes `LABEL-mask.png`: the frame's size and view,
every pixel one flat color for what it shows, from lit mode's G-buffer, the depth and the traced sun visibility of the
next 16 frames (the visibility's samples rotate over each 4 x 4 block's pixels; their mean says sunlit or not). The camera
holds still in lab mode, so a `shot` and a `mask` with the same label line up pixel for pixel (the G-buffer's sub-pixel
jitter aside). It takes about half a second. The classes, first rule that holds:

| Class | Color (RGB) | Rule |
|---|---|---|
| sky | 60 120 220 | nothing drawn (depth 0), or vanilla's cloud slab (the 4 blocks from the cloud height) |
| water | 0 200 200 | the G-buffer's water flag (the LOD's and the far field's water), or a surface 8/9 of the way up a block (a source block's top) where no lit terrain shows: vanilla's water |
| unlit | 220 30 90 | anything else that isn't lit terrain: entities, the hand, particles, glass |
| far | 150 150 150 | lit terrain past 512 blocks |
| blocklit | 255 120 20 | lit terrain with block light 8 or more where the sky light is under 10 or the sun is down; light sources (block light 15) |
| vegetation | 40 170 40 | with `leaflight` the G-buffer's foliage class (`litFoliageClass`: leaves and plants; grass blocks' tops are ground and go to sunlit or shade); without it an albedo-hue rule: faces that aren't axis-aligned (cross plants: grass, flowers, ferns, crops) and green albedo (green the largest channel, above blue by 10%, saturation over 0.2: leaves, grass blocks' tops, vines) |
| sunlit | 255 215 60 | lit terrain whose sun visibility is over 0.5 and whose face is turned to the sun, by day |
| shade | 120 70 180 | the rest of lit terrain |

The rules' constants are `#define`s at the top of `lab_mask.metal` (editable live like any shader file).

## Twins

SEUS PTGI renders inside our engine with our camera and geometry (the external pipeline), so every pixel of our frame
has a twin in its frame. Three tools make "does it look like SEUS" a number (docs/lighting-design.md, "The grade"):

- `tools/twindiff.swift` (`swiftc -O tools/twindiff.swift -o bench_out/twindiff`): `twindiff ours.png seus.png mask.png
  [--json out.json] [--board out.jpg] [--name NAME] [--weights sky=0.25,...]`. Per class and frame: the share, OKLab
  mean L, a, b and chroma, L at the 10th, 50th and 90th percentiles, local contrast; the deltas and a score (lower is
  better; the formula is in lighting-design.md); the census's luma numbers (ffmpeg's YAVG, YLOW and SATAVG of the bottom
  55% and the top 20%); a board: ours | SEUS over the mask | the lightness difference (red: ours brighter, blue: darker).
  About a second a scene.
- `tools/bench/twins.sh <label> [LAB_EXP]`: one command for a census. A lab session (the Claude edition's switches by
  default, `...,post,leaflight,wave,clouds,godrays`; a throwaway scene first, as a session's first scene can catch the LOD
  mid-build; 300 ticks after each scene to converge), `scene`, `shot` and `mask` for all nine scenes, each scored
  against `~/Projects/metal-mc-private/twins/census1/seus1-<scene>.png`, the boards and
  `summary.md` (tools/bench/twinsum.py) in the main checkout's `bench_out/twins/<label>/` (TWINS_OUT). TWINS_NOGAME=1
  scores the shots already there again. Boards hold SEUS's pixels: they stay in bench_out, out of the repository.
- `tools/bench/autotune.py <label> --knobs ... --scenes ...`: a lab session that searches the look's knobs (the
  `#define`s of post.metal and lit_relight_header.metal) against the twins; see its header and lighting-design.md.

## The run ledger

Every `bench_lod.sh` run (so every `fidelity.sh` and `lab.sh` run too) appends one JSON line to `bench_out/ledger.jsonl`
in the main checkout, whichever worktree ran it (`tools/bench/ledger.py`; it is only ever appended to). A line has the
time, kind (fly, bench, tour, fidelity, lab), label, commit, branch, `dirty` (uncommitted changes in the build), the
worktree, the args and `METALMC_EXP`, fixture, LOD reach, render distance, vsync, every number of the `METALMC_BENCH`
summary, the screenshot count, and validity flags: `screen_locked` (run_mc.sh notes the lock screen every 10 s),
`fps_pinned_120` (uncapped but averaging 120.0: paced by something), `lod_still_building` (lodready.py), `windowed` (not
the panel's 3456 x 2234 fullscreen), `dialogs`; `valid` is all clear for a timed run, and fullscreen for a screenshot
run. `python3 tools/bench/ledger.py --show 20` lists the last runs.

## The flicker meter

`burst LABEL [N] [SAVE]` (`Sources/MetalMCNative/Flicker.swift`, 2026-10-13) sees what a screenshot can't: the next N frames
the present blit sends out, every pixel's luma (0-255, the stored values) tracked through them.

- **d1**: mean |l(t) - l(t-1)|, any change.
- **d2**: mean |l(t) - 2 l(t-1) + l(t-2)|, change that reverses. Smooth motion has almost none; a pixel alternating
  between two values has twice its d1. With the camera held still (lab mode), d2 is the flicker.
- The summary: mean d1 and d2, d2's p99, and the share of pixels with d2 over 2 and over 6 levels. **The HDR frame's
  dither (RG11B10, moving every frame) puts the floor at d2 about 2-2.5**, so read d2 > 6 as the visible part.
- `tools/flickstat.swift` (`swiftc -O tools/flickstat.swift -o bench_out/flickstat`): `flickstat LABEL-flicker.bin
  [LABEL-mask.png] [--json out.json] [--hot N]` splits it by the class mask (`mask LABEL` on the same view) and lists
  the hottest 64 x 64 tiles, for cropping the kept frames.
- Under `pan`, d2 isn't flicker any more: fine texture sliding across pixels reverses too. Judge motion by the kept
  frames (ghosting, smearing).

What it found on the Claude edition (main 34ac2e6), d2 > 6: water_closeup 30.4%, sunset_water 23.8%, forest 13.2%,
noon_overview 7.7%. An ablation (one session per switch off) put it on two causes, both the same pattern, a 1-in-16
stochastic sample per 4 x 4 block read back nearest with no history (the anti-aliasing can't average a block that flips
whole: its neighborhood clip flips with it):

- the traced sun visibility (rtshadows off: forest 13.2% -> 2.5%): every shadow edge flipped 4 x 4 blocks;
- the water's reflection rays (at grazing angles a hit on the far shore or the sky by a wave's slope; the checkerboard's
  copy of the texel beside made horizontal bars).

Fixed (branch flicker): the visibility's history (`VisFilter.swift`) and a depth-aware upsample (`litVisAt`), the
reflection's history and a bilinear upsample (Water.swift), the resolve on tone-weighted colors (`TAA_TONEMAP`). After:
water_closeup 2.7%, sunset_water 1.9%, forest 2.8%, noon_overview 2.7%: the dither's floor.
