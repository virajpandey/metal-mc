#!/bin/bash
# 120 Hz (vsync) flight A/B: dropped frames (frames_over_16ms) for TAA and split factor 3, then an uncapped split-3
# rerun and a midnight fidelity pair with and without far light. Needs the screen unlocked.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/bench_out
run() {
  L=$1; shift
  BENCH_NOBUILD=1 BENCH_TIMEOUT=600 bash "$ROOT/tools/bench/bench_lod.sh" $L 32768 -PbenchY=150 -PbenchFly=20 -PbenchExtraWait=400 -PbenchHitches=1 "$@" > "$OUT/$L.out" 2>&1
  lock=$(ioreg -n Root -d1 | grep -o '"CGSSessionScreenIsLocked"=[A-Za-z]*' || echo unlocked)
  echo "$L [$lock] $(head -1 "$OUT/$L.out")"
}
export BENCH_VSYNC=true
run vBase
run vTaa -Ptaa=true
run vS3 -PlodSplit=3
run vS3Taa -PlodSplit=3 -Ptaa=true
export BENCH_VSYNC=false
run nS3b -PlodSplit=3
run nBase3
sed -i '' "s/^enableVsync:.*/enableVsync:false/" "$ROOT/mod/run/options.txt"
BENCH_NOBUILD=1 BENCH_TIMEOUT=900 bash "$ROOT/tools/bench/fidelity.sh" midLight 12 8192 -PlodGenerate=0 -PfidelityTime=18000
BENCH_NOBUILD=1 BENCH_TIMEOUT=900 bash "$ROOT/tools/bench/fidelity.sh" midNoLight 12 8192 -PlodGenerate=0 -PfidelityTime=18000 -PmetalExp=nofarlight
