#!/bin/bash
# usage: verify_all.sh <prefix> [extra gradle args for every run...]
# The full in-game check after a batch of changes, one game at a time:
#   1. fidelity tour (4 km world, fog off) with quads and with the far field: near band vs vanilla (nA/nB), far band vs
#      the level-0 answer key (nL0c, level 0 out to 2 km, rendered 2026-10-01 at the panel's resolution);
#   2. the horizon tour (LOD 262144, generated terrain, far field on);
#   3. flights on Viraj's world: uncapped (fps, p99, frames over 8.33 ms) and at 120 Hz with vsync (dropped frames),
#      quads vs far field.
# Screenshots: mod/run/screenshots/<prefix>{Q,F,H}-tour-*.png. Logs: bench_out/run_<label>.log.
P=$1; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 1
FS=$ROOT/bench_out/fidscore
SHOTS=$ROOT/mod/run/screenshots
# A system dialog over the fullscreen game paces every run to 120 fps (see docs/lod-design.md): refuse to start.
if "$ROOT/bench_out/winlist" | grep -q UserNotificationCenter; then echo "a system dialog is on screen; not starting"; exit 1; fi

fid() { BENCH_NOBUILD=1 BENCH_TIMEOUT=${T:-900} bash tools/bench/fidelity.sh "$@"; }
fid ${P}Q 12 8192 -PlodGenerate=0 -PfarField=0 "$@"
fid ${P}F 12 8192 -PlodGenerate=0 -PfarField=1 "$@"
T=2400 fid ${P}H 12 262144 -PfarField=1 -PbenchExtraWait=6000 "$@"

fly() { BENCH_NOBUILD=1 BENCH_FIXTURE=claudeworld-merged BENCH_TIMEOUT=900 bash tools/bench/bench_lod.sh "$1" 32768 -PbenchY=150 \
          -PbenchFly=20 -PbenchExtraWait=600 -PbenchHitches=1 -PbenchTrace=1 -Ptaa=true "${@:2}" > /dev/null 2>&1; }
fly ${P}fq -PfarField=0 "$@"
fly ${P}ff -PfarField=1 "$@"
BENCH_VSYNC=true fly ${P}vq -PfarField=0 "$@"
BENCH_VSYNC=true fly ${P}vf -PfarField=1 "$@"
sed -i '' "s/^enableVsync:.*/enableVsync:false/" mod/run/options.txt

echo "== near band vs vanilla (192-512 blocks)"
for L in ${P}Q ${P}F; do (cd "$SHOTS" && "$FS" . nA nB $L | tail -1); done
echo "== far band vs the level-0 answer key (512 blocks-2 km)"
for L in ${P}Q ${P}F; do (cd "$SHOTS" && "$FS" . nL0c nA $L | tail -1); done
echo "== flights"
for L in ${P}fq ${P}ff ${P}vq ${P}vf; do
  echo "$L: $(grep -h 'METALMC_BENCH label' bench_out/run_$L.log | grep -o 'fps_mean=[^ ]*\|ms_p99=[^ ]*\|frames_over_8ms=[^ ]*\|frames_over_16ms=[^ ]*\|per_frame_lod_kquads=[^ ]*' | tr '\n' ' ')"
done
