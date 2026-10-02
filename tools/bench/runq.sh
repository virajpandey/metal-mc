#!/bin/bash
# usage: tools/bench/runq.sh  (in the background; start it again when it stops, for the rest)
# Runs jobs from bench_out/queue.txt (one shell command per line, using the functions below) in order, taking each off
# the queue as it starts, until the queue is empty or $RUNQ_MINUTES (70) minutes have passed. Start it as a background
# task with the longest timeout (2 hours; the default is 30 minutes): 70 minutes plus the longest job (a 40-minute
# horizon tour) fits, so a job is never cut in the middle. Start it again for the rest. Output: bench_out/queue.log.
cd "$(dirname "$0")/../.." || exit 1
# Keeps the Mac and its display awake while jobs run: the lock screen over the game paces it at 120 Hz.
caffeinate -dims -w $$ &
Q=bench_out/queue.txt
L=bench_out/queue.log
FS=$PWD/bench_out/fidscore
SHOTS=$PWD/mod/run/screenshots
# fly <label> [gradle args]: a traced flight on Viraj's world (native, TAA on); summary and per-pass medians.
fly() { BENCH_NOBUILD=1 BENCH_FIXTURE=claudeworld-merged BENCH_TIMEOUT=900 bash tools/bench/bench_lod.sh "$1" 32768 -PbenchY=150 \
          -PbenchFly=20 -PbenchExtraWait=600 -PbenchHitches=1 -PbenchTrace=1 -Ptaa=true "${@:2}" > /dev/null 2>&1
        echo "$1: $(grep -h 'METALMC_BENCH label' bench_out/run_$1.log | grep -o 'fps_mean=[^ ]*\|ms_p99=[^ ]*\|frames_over_8ms=[^ ]*\|per_frame_lod_kquads=[^ ]*' | tr '\n' ' ') $(python3 tools/bench/lodready.py bench_out/run_$1.log)"
        python3 tools/bench/passes.py bench_out/run_$1.log; }
# vfly <label> [gradle args]: the same at 120 Hz with vsync (frames_over_16ms = dropped frames).
vfly() { BENCH_VSYNC=true fly "$@"; sed -i '' "s/^enableVsync:.*/enableVsync:false/" mod/run/options.txt; }
# fid <label> [gradle args]: the fidelity tour (4 km world, fog off, far field on) at render distance 12, LOD 8192.
fid() { BENCH_NOBUILD=1 BENCH_TIMEOUT=900 bash tools/bench/fidelity.sh "$1" 12 8192 -PlodGenerate=0 "${@:2}" > /dev/null 2>&1
        echo "$1: $(ls "$SHOTS"/$1-tour-*.png 2>/dev/null | wc -l | tr -d ' ') screenshots, $(sips -g pixelWidth "$SHOTS/$1-tour-00-ground-north.png" 2>/dev/null | tail -1 | awk '{print $2}') wide"; }
# near <labels...> / far <labels...>: fidelity scores, near band vs vanilla (192-512), far band vs the level-0 answer key.
near() { for x in "$@"; do (cd "$SHOTS" && "$FS" . nA nB "$x" | tail -1); done; }
far() { for x in "$@"; do (cd "$SHOTS" && "$FS" . nL0c nA "$x" | tail -1); done; }
build() { swift build -c release --product MetalMCNative 2>&1 | grep -E 'error|Build complete' | tail -3; }
# The lock screen (the company's idle lock) over the fullscreen game paces it at 120 Hz: timed jobs wait for an unlock
# (the runner stops and leaves them queued); fidelity tours, which read the render target, run anyway.
locked() { bench_out/winlist 2>/dev/null | awk '$1 == "layer" && $2 >= 1900 && $4 > 0 && $5 == "loginwindow" { found = 1 } END { exit !found }'; }
start=$(date +%s)
while [ -s "$Q" ] && [ $(( $(date +%s) - start )) -lt $(( ${RUNQ_MINUTES:-70} * 60 )) ]; do
  job=$(head -1 "$Q")
  case "$job" in fly\ *|vfly\ *)
    if locked; then echo "== $(date +%T) the screen is locked: timed jobs wait (runner stopped)" >> "$L"; break; fi ;;
  esac
  tail -n +2 "$Q" > "$Q.tmp" && mv "$Q.tmp" "$Q"
  echo "== $(date +%T) $job" >> "$L"
  eval "$job" >> "$L" 2>&1
done
echo "== $(date +%T) runner stopped, $(grep -c . "$Q" 2>/dev/null || echo 0) jobs left" >> "$L"
