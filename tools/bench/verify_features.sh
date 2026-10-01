#!/bin/bash
# usage: verify_features.sh <prefix>
# In-game checks of the switchable features, each against the same run without it (verify_all.sh's <prefix>Q/<prefix>fq
# are the baselines): smart LOD (fidelity + flight), near chunks (fidelity + flight), the sky and HDR (the render tour's
# day/sunset/night/rain views, and the LOD tour to the horizon with the far field), one game at a time.
P=$1
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 1
FS=$ROOT/bench_out/fidscore
SHOTS=$ROOT/mod/run/screenshots
# A system dialog over the fullscreen game paces every run to 120 fps: screenshots stay valid, timings don't. SHOTS_ONLY=1 runs
# just the screenshot checks (and is allowed with a dialog on screen).
if [ -z "$SHOTS_ONLY" ] && "$ROOT/bench_out/winlist" | grep -q UserNotificationCenter; then echo "a system dialog is on screen; not starting"; exit 1; fi

fid() { BENCH_NOBUILD=1 BENCH_TIMEOUT=${T:-900} bash tools/bench/fidelity.sh "$@"; }
fly() { [ -n "$SHOTS_ONLY" ] && return; BENCH_NOBUILD=1 BENCH_FIXTURE=claudeworld-merged BENCH_TIMEOUT=900 bash tools/bench/bench_lod.sh "$1" 32768 -PbenchY=150 \
          -PbenchFly=20 -PbenchExtraWait=600 -PbenchHitches=1 -PbenchTrace=1 -Ptaa=true "${@:2}" > /dev/null 2>&1; }
tour() { BENCH_NOBUILD=1 BENCH_FIXTURE=claudeworld-merged BENCH_TIMEOUT=${T:-900} bash tools/bench/bench_lod.sh "$1" "$2" -Ptaa=true "${@:3}" > /dev/null 2>&1; }

fid ${P}S 12 8192 -PlodGenerate=0 -PmetalExp=smartlod
fly ${P}fs -PmetalExp=smartlod
fid ${P}N 12 8192 -PlodGenerate=0 -PmetalExp=nearchunks
fly ${P}fn -PmetalExp=nearchunks
tour ${P}T0 32768 -PbenchTour=1
tour ${P}T1 32768 -PbenchTour=1 -PmetalExp=sky
tour ${P}T2 32768 -PbenchTour=1 -PmetalExp=sky,hdr
T=2400 tour ${P}L1 262144 -PbenchTour=lod -PfarField=1 -PmetalExp=sky -PbenchExtraWait=6000
fly ${P}fk -PmetalExp=sky
fly ${P}fh -PmetalExp=sky,hdr

echo "== near band vs vanilla"
for L in ${P}S ${P}N; do (cd "$SHOTS" && "$FS" . nA nB $L | tail -1); done
echo "== far band vs the level-0 answer key"
for L in ${P}S ${P}N; do (cd "$SHOTS" && "$FS" . nL0 nA $L | tail -1); done
echo "== flights"
for L in ${P}fs ${P}fn ${P}fk ${P}fh; do
  echo "$L: $(grep -h 'METALMC_BENCH label' bench_out/run_$L.log | grep -o 'fps_mean=[^ ]*\|ms_p99=[^ ]*\|frames_over_8ms=[^ ]*\|per_frame_lod_kquads=[^ ]*' | tr '\n' ' ')"
done
echo "== errors"
for L in ${P}S ${P}fs ${P}N ${P}fn ${P}T0 ${P}T1 ${P}T2 ${P}L1 ${P}fk ${P}fh; do
  echo "$L: $(grep -hc 'Exception\|GPU error\|failed' bench_out/run_$L.log) exception/failure lines; screenshots $(ls "$SHOTS" | grep -c "^$L-")"
done
