#!/bin/bash
# usage: tools/bench/lab.sh [label] [extra gradle args...]
# Lab mode (docs/lab-mode.md): one long-lived game for look work. Like a bench run (fullscreen, a fresh copy of the
# fixture, the GPU lock held while it runs, a ledger line at the end), but it stays up until `quit` (or LAB_MINUTES):
#   - it runs the commands appended to mod/run/metalmc-control.txt (scene, shot, reload, wait, tour, quit, ...; results
#     in mod/run/metalmc-control.log), with the scenes of tools/bench/scenes.json;
#   - it takes our Metal shaders from $LAB_SHADERS (mod/run/shaders), recompiling a file within a second of a save.
# Run it in the background and drive it by appending lines:
#   bash tools/bench/lab.sh look1 &
#   echo "scene sunset_water" >> mod/run/metalmc-control.txt
#   echo "shot sunset-a" >> mod/run/metalmc-control.txt      -> mod/run/screenshots/sunset-a.png
# Env: LAB_EXP (lit,nearchunks,rtshadows,sky,gi,water), LAB_FAR (LOD reach, 8192), LAB_FIXTURE (claudeworld-merged),
#      LAB_MINUTES (90), LAB_SHADERS, LAB_SCRIPT (a file of commands to queue at the start), LAB_VSYNC (true: 120 Hz).
LABEL=${1:-lab}; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 1
SHADERS=${LAB_SHADERS:-$ROOT/mod/run/shaders}
CTL=mod/run/metalmc-control.txt
mkdir -p mod/run
# Commands left from an earlier session would run at once: start from an empty file, or LAB_SCRIPT's commands.
if [ -n "$LAB_SCRIPT" ]; then cp "$LAB_SCRIPT" "$CTL"; else : > "$CTL"; fi
echo "== $(date '+%F %T') lab session $LABEL: shaders in $SHADERS, log bench_out/run_$LABEL.log" >> mod/run/metalmc-control.log
# The Swift library, built under the GPU lock (gradle's own build of it then has nothing left to do).
bash tools/bench/gpuwait.sh swift build -c release --product MetalMCNative 2>&1 | grep -E 'error|Build complete' | tail -3
LEDGER_KIND=lab BENCH_NOBUILD=1 BENCH_VSYNC=${LAB_VSYNC:-true} BENCH_FIXTURE=${LAB_FIXTURE:-claudeworld-merged} \
  BENCH_TIMEOUT=$(( ${LAB_MINUTES:-90} * 60 )) bash tools/bench/bench_lod.sh "$LABEL" "${LAB_FAR:-8192}" -PbenchLab=1 \
  -PshaderDir="$SHADERS" -Ptaa=true -PmetalExp="${LAB_EXP:-lit,nearchunks,rtshadows,sky,gi,water}" "$@"
# Uncapped again for the bench runs that follow (bench_lod.sh leaves vsync as it set it).
sed -i '' "s/^enableVsync:.*/enableVsync:false/" mod/run/options.txt
