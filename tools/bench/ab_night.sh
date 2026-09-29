#!/bin/bash
# Native fullscreen A/B on the flight bench (8 km world, y 150, 20 blocks/s): TAA and the LOD split factor.
# Alternates runs so drift shows up. Needs the screen unlocked, or runs are windowed and display-paced.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/bench_out
run() {
  L=$1; shift
  BENCH_NOBUILD=1 BENCH_TIMEOUT=600 bash "$ROOT/tools/bench/bench_lod.sh" $L 32768 -PbenchY=150 -PbenchFly=20 -PbenchExtraWait=400 -PbenchHitches=1 "$@" > "$OUT/$L.out" 2>&1
  lock=$(ioreg -n Root -d1 | grep -o '"CGSSessionScreenIsLocked"=[A-Za-z]*' || echo unlocked)
  echo "$L [$lock] $(head -1 "$OUT/$L.out")"
  grep -h "TAA: resolve" "$OUT/run_$L.log" | head -1 | cut -c 17-80
}
run nBase1
run nTaa1 -Ptaa=true
run nS3 -PlodSplit=3
run nS3Taa -PlodSplit=3 -Ptaa=true
run nBase2
run nTaa2 -Ptaa=true
