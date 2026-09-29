#!/bin/bash
# usage: run_mc.sh <logfile> <timeout_s> [gradle args...]
# Runs the Fabric client on a fresh copy of fixtures/$FIXTURE (default claudeworld), always loaded as
# saves/claudeworld, and kills it after timeout_s (macOS has no `timeout`). The previous copy is moved to
# bench_out/old_worlds (prune_worlds.py keeps the newest few). Fixtures are copies: never point this at a real save.
LOG=$1; T=$2; shift 2
FIX=${FIXTURE:-claudeworld}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/bench_out
mkdir -p "$OUT/old_worlds"
cd "$ROOT/mod" || exit 1
[ -d run/saves/claudeworld ] && mv run/saves/claudeworld "$OUT/old_worlds/claudeworld-$(date +%s)"
cp -Rp "../fixtures/$FIX" run/saves/claudeworld   # -p keeps mtimes, so the LOD's region cache still matches
# Let Spotlight and the security scanners finish with the fresh copy before timing anything.
sleep ${SETTLE:-30}
( sleep "$T"; pkill -f KnotClient ) & WD=$!
env JAVA_HOME=/opt/homebrew/opt/openjdk@25 ./gradlew runClient "$@" > "$LOG" 2>&1
echo "exit $?" >> "$LOG"
kill $WD 2>/dev/null
