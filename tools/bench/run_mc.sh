#!/bin/bash
# usage: run_mc.sh <logfile> <timeout_s> [gradle args...]
# Runs the Fabric client on a fresh copy of fixtures/$FIXTURE (default claudeworld), always loaded as
# saves/claudeworld, and kills it after timeout_s (macOS has no `timeout`). The previous copy is moved to
# bench_out/old_worlds (prune_worlds.py keeps the newest few). Fixtures are copies: never point this at a real save.
#
# A system dialog (a permission prompt, a crash notice, a notification banner) over the fullscreen game paces it to the
# display's 120 Hz, which makes a timed run's numbers worthless but leaves screenshots as they were. So a timed run (a
# flight, -PbenchFly) waits for a clear screen before it starts and is run again (up to 3 tries) if one showed up while
# it ran; any run notes the dialogs it saw in the log (METALMC_DIALOG lines).
LOG=$1; T=$2; shift 2
FIX=${FIXTURE:-claudeworld}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$ROOT/bench_out
mkdir -p "$OUT/old_worlds"
cd "$ROOT/mod" || exit 1
dialog() { "$OUT/winlist" | grep -E "UserNotificationCenter|SecurityAgent|CoreServicesUIAgent|Agents Overlay|^layer [1-9][0-9]* alpha [^ ]+ NotificationCenter " | head -1; }
TIMED=; case " $* " in *" -PbenchFly="*) TIMED=1 ;; esac
for TRY in 1 2 3; do
  if [ -n "$TIMED" ] && [ -n "$(dialog)" ]; then
    echo "$(date +%T) waiting for a clear screen: $(dialog)" >> "$OUT/dialog_waits.log"
    while [ -n "$(dialog)" ]; do sleep 20; done
  fi
  [ -d run/saves/claudeworld ] && mv run/saves/claudeworld "$OUT/old_worlds/claudeworld-$(date +%s)"
  cp -Rp "../fixtures/$FIX" run/saves/claudeworld   # -p keeps mtimes, so the LOD's region cache still matches
  # Read the fresh copy once now, so the machine's security scanners check its new files here rather than while the
  # LOD reads them (on 2026-10-01 they made the LOD's first build 2.5x slower, past the bench's wait for it).
  find run/saves/claudeworld -type f -exec cat {} + > /dev/null
  # Let Spotlight and the security scanners finish with the fresh copy before timing anything.
  sleep ${SETTLE:-30}
  : > "$LOG.dialogs"
  ( sleep "$T"; pkill -f KnotClient ) & WD=$!
  ( while sleep 10; do d=$(dialog); [ -n "$d" ] && echo "METALMC_DIALOG $(date +%T) $d" >> "$LOG.dialogs"; done ) & DW=$!
  env JAVA_HOME=/opt/homebrew/opt/openjdk@25 ./gradlew runClient "$@" > "$LOG" 2>&1
  echo "exit $?" >> "$LOG"
  kill $WD $DW 2>/dev/null
  cat "$LOG.dialogs" >> "$LOG"
  if [ -z "$TIMED" ] || [ ! -s "$LOG.dialogs" ]; then break; fi
  echo "$(date +%T) $(basename "$LOG"): a dialog showed up during the run (try $TRY): $(head -1 "$LOG.dialogs")" >> "$OUT/dialog_waits.log"
done
: > "$LOG.dialogs"
