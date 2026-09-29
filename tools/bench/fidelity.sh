#!/bin/bash
# usage: fidelity.sh <label> <renderDistance> <lodFar|0> [extra gradle args...]
# The fidelity tour (fog off, no clouds, no HUD, no mobs) on the 4 km world; screenshots land in
# mod/run/screenshots/<label>-tour-NN-<step>.png. Score them with fidcheck.sh or bench_out/fidscore.
LABEL=$1; RD=$2; FAR=$3; shift 3
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OPT=$ROOT/mod/run/options.txt
sed -i '' 's/^renderClouds:.*/renderClouds:"false"/' $OPT
BENCH_RD=$RD BENCH_FIXTURE=${FID_FIXTURE:-claudeworld-big} bash "$ROOT/tools/bench/bench_lod.sh" "$LABEL" "$FAR" -PbenchTour=fidelity -Pfidelity=1 "$@" > /dev/null 2>&1
sed -i '' 's/^renderClouds:.*/renderClouds:"true"/; s/^renderDistance:.*/renderDistance:12/' $OPT
echo "$LABEL: $(ls "$ROOT/mod/run/screenshots" | grep -c "^$LABEL-tour-") screenshots"
grep -h 'Exception\|GPU error' "$ROOT/bench_out/run_$LABEL.log" | grep -v 'Realms\|SignedJWT' | head -3
