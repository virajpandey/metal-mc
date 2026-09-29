#!/bin/bash
# Split factor 4 vs the default 3: native fidelity (far band vs nR2) and a 120 Hz vsync flight.
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
bash "$ROOT/tools/bench/fidcheck.sh" nC31s4 -PlodSplit=4 2>&1 | tail -4
BENCH_VSYNC=true BENCH_NOBUILD=1 BENCH_TIMEOUT=600 bash "$ROOT/tools/bench/bench_lod.sh" vS4 32768 -PbenchY=150 -PbenchFly=20 -PbenchExtraWait=400 -PbenchHitches=1 -PlodSplit=4 2>&1 | head -1
BENCH_NOBUILD=1 BENCH_TIMEOUT=600 bash "$ROOT/tools/bench/bench_lod.sh" nS4 32768 -PbenchY=150 -PbenchFly=20 -PbenchExtraWait=400 -PbenchHitches=1 -PlodSplit=4 2>&1 | head -1
sed -i '' "s/^enableVsync:.*/enableVsync:false/" "$ROOT/mod/run/options.txt"
