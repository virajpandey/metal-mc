#!/bin/bash
# Generated far terrain from level 2 (default) vs level 3, on the small world where most of the view is generated.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/bench_out
run() {
  L=$1; shift
  BENCH_FIXTURE=claudeworld BENCH_NOBUILD=1 BENCH_TIMEOUT=700 bash "$ROOT/tools/bench/bench_lod.sh" $L 32768 -PbenchY=150 -PbenchFly=20 -PbenchExtraWait=1200 -PbenchHitches=1 "$@" > "$OUT/$L.out" 2>&1
  echo "$L $(head -1 "$OUT/$L.out")"
}
run fm3 -PfarMin=3
run fm2
BENCH_VSYNC=true run vfm2
sed -i '' "s/^enableVsync:.*/enableVsync:false/" "$ROOT/mod/run/options.txt"
