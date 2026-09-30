#!/bin/bash
# usage: profile_split.sh <prefix> [fixture, default claudeworld-merged]
# Three uncapped native flights with per-pass GPU tracing (-PbenchTrace=1): everything on (<prefix>A), LOD off
# (<prefix>B) and TAA off (<prefix>C). The differences split the frame into vanilla, LOD and TAA costs.
P=$1; FIX=${2:-claudeworld-merged}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
run() { BENCH_FIXTURE=$FIX BENCH_TIMEOUT=900 bash "$ROOT/tools/bench/bench_lod.sh" "$1" "$2" -PbenchY=150 -PbenchFly=20 \
          -PbenchExtraWait=600 -PbenchHitches=1 -PbenchTrace=1 "${@:3}" > /dev/null 2>&1; }
run ${P}A 32768 -Ptaa=true
run ${P}B 0 -Ptaa=true
run ${P}C 32768 -Ptaa=false
for L in ${P}A ${P}B ${P}C; do
  echo "== $L: $(grep -h 'METALMC_BENCH label' "$ROOT/bench_out/run_$L.log" | grep -o 'fps_mean=[^ ]*\|ms_p99=[^ ]*\|metal_gpu_ms_mean=[^ ]*\|per_frame_lod_kquads=[^ ]*' | tr '\n' ' ')"
  # The last traced submit's per-pass lines.
  awk '/profile submit/{buf=$0; n=1; next} n&&/^  total/{buf=buf"\n"$0; next} n{last=buf; n=0} END{if(n)last=buf; print last}' "$ROOT/bench_out/run_$L.log" | head -40
done
