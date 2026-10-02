#!/bin/bash
# usage: bench_lod.sh <label> <lodFar or 0> [extra gradle args...]
# Fullscreen, render distance $BENCH_RD (12), noon, on fixtures/$BENCH_FIXTURE (claudeworld-huge); prints the bench
# summary. Log: bench_out/run_<label>.log. Note gradle's runClient rebuilds the Swift library from the working tree
# on every run (BENCH_NOBUILD only skips the explicit build here), so don't edit Swift mid-batch.
LABEL=$1; FAR=$2; shift 2
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/bench_out
cd "$ROOT" || exit 1
[ -n "$BENCH_NOBUILD" ] || swift build -c release --product MetalMCNative 2>&1 | grep -E 'error|warning: unre|Build complete' | tail -5
OPT=mod/run/options.txt
RD=${BENCH_RD:-12}
# Never write a non-numeric renderDistance: Minecraft stops reading options at a bad line and resets the rest.
# BENCH_VSYNC=true: a real 120 Hz run (count frames_over_16ms as dropped frames); default off (uncapped).
sed -i '' "s/^fullscreen:.*/fullscreen:true/; s/^renderDistance:.*/renderDistance:$RD/; s/^pauseOnLostFocus:.*/pauseOnLostFocus:false/; s/^enableVsync:.*/enableVsync:${BENCH_VSYNC:-false}/; s/^startedCleanly:.*/startedCleanly:true/" $OPT
# Fullscreen at the panel's own pixels (1728 x 1117 points at 2x: 3456 x 2234) whatever the desktop's scaling: the
# exclusive fullscreen takes its mode from fullscreenResolution, and without one it uses the desktop's (More Space:
# 4112 x 2658 rendered and scaled down by the system). A crash during startup makes the next start windowed and drops
# the mode (startedCleanly), so both are set before every run.
MODE='{"width":1728,"height":1117,"red_bits":8,"green_bits":8,"blue_bits":8,"refresh_rate":120.0}'
if grep -q '^fullscreenResolution:' $OPT; then sed -i '' "s/^fullscreenResolution:.*/fullscreenResolution:$MODE/" $OPT; else echo "fullscreenResolution:$MODE" >> $OPT; fi
if [ "$FAR" = "0" ]; then LODARGS="-Plod=0"; else LODARGS="-Plod=1 -PlodFar=$FAR"; fi
# The display is held on only while someone is at the laptop (see runq.sh); the company's idle lock is left alone.
KEEP=-i; [ "$BENCH_ATTENDED" = 1 ] && KEEP=-di
FIXTURE=${BENCH_FIXTURE:-claudeworld-huge} caffeinate $KEEP bash "$ROOT/tools/bench/run_mc.sh" "$OUT/run_$LABEL.log" ${BENCH_TIMEOUT:-300} \
  -PmetalBackend=metal -PbenchNoon=1 -PbenchLabel=$LABEL $LODARGS "$@"
grep -h 'METALMC_BENCH' "$OUT/run_$LABEL.log" | grep -o 'fps_mean=[^ ]*\|ms_p95=[^ ]*\|ms_p99=[^ ]*\|frames_over_[^ ]*\|metal_gpu_ms_mean=[^ ]*\|per_frame_lod_kquads=[^ ]*' | tr '\n' ' '
echo
grep -h 'exit\|Exception\|error:' "$OUT/run_$LABEL.log" | grep -v 'Realms\|SignedJWT' | head -5
python3 "$ROOT/tools/bench/prune_worlds.py" 2
