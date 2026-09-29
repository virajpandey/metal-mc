#!/bin/bash
# TAA on the walk tour (ground, view bobbing, vanilla terrain in motion) and the zoom tour (field of view change),
# with and without TAA, on the 4 km world. Screenshots: mod/run/screenshots/<label>-tour-*.png.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
for t in walk zoom; do
  for a in off on; do
    L=taa${a}_$t
    BENCH_NOBUILD=1 BENCH_TIMEOUT=400 BENCH_FIXTURE=claudeworld-big bash "$ROOT/tools/bench/bench_lod.sh" $L 32768 -PbenchTour=$t -PlodGenerate=0 $([ $a = on ] && echo -Ptaa=true) > /dev/null 2>&1
    echo "$L: $(ls "$ROOT/mod/run/screenshots" | grep -c "^$L-tour-") screenshots"
  done
done
