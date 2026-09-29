#!/bin/bash
# The TAA tour (rain, smoke and mobs, third person, fast turn) with TAA off and on, on the 4 km world.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
for a in off on; do
  L=taatour_$a
  BENCH_NOBUILD=1 BENCH_TIMEOUT=400 BENCH_FIXTURE=claudeworld-big bash "$ROOT/tools/bench/bench_lod.sh" $L 32768 -PbenchTour=taa -PlodGenerate=0 $([ $a = on ] && echo -Ptaa=true) > /dev/null 2>&1
  echo "$L: $(ls "$ROOT/mod/run/screenshots" | grep -c "^$L-tour-") screenshots"
done
