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

Measured (2026-10-05, M3 Pro): the world is ready about 50 s after the session gets the GPU lock (the first LOD build
from a cold cache takes a minute or two more); a save to rebuilt pipelines is about a second (lit and taa compile in
about 0.5 s each); `tour` visits all nine scenes and writes nine screenshots in 74 s once the LOD is built.

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
| `reload` | recompile every shader file now, changed or not, and rebuild their pipelines (normally the watcher does it on save) |
| `wait TICKS` | 20 ticks a second |
| `tour [PREFIX]` | `scene` + `shot PREFIX-NAME` for every scene (PREFIX: tour) |
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
start over from the built-in source.

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

The nine on `claudeworld-merged`: noon_overview, night_torches (the plains village's own torches at midnight), rain (the
same village), forest, torch_cave (a cavern, torches placed by its setup), mineshaft (a corridor lit by its own wall
torches), sunset_water, water_closeup, mountain_view. The notes in the file say where each is and how it was found.

## The run ledger

Every `bench_lod.sh` run (so every `fidelity.sh` and `lab.sh` run too) appends one JSON line to `bench_out/ledger.jsonl`
in the main checkout, whichever worktree ran it (`tools/bench/ledger.py`; it is only ever appended to). A line has the
time, kind (fly, bench, tour, fidelity, lab), label, commit, branch, `dirty` (uncommitted changes in the build), the
worktree, the args and `METALMC_EXP`, fixture, LOD reach, render distance, vsync, every number of the `METALMC_BENCH`
summary, the screenshot count, and validity flags: `screen_locked` (run_mc.sh notes the lock screen every 10 s),
`fps_pinned_120` (uncapped but averaging 120.0: paced by something), `lod_still_building` (lodready.py), `windowed` (not
the panel's 3456 x 2234 fullscreen), `dialogs`; `valid` is all clear for a timed run, and fullscreen for a screenshot
run. `python3 tools/bench/ledger.py --show 20` lists the last runs.
