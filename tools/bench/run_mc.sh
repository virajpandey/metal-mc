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
dialog() { "$WINLIST" | grep -E "UserNotificationCenter|SecurityAgent|CoreServicesUIAgent|Agents Overlay|^layer [1-9][0-9]* alpha [^ ]+ NotificationCenter " | head -1; }
# The lock screen (loginwindow's shield) over the game paces it at 120 Hz: noted as METALMC_LOCKED lines for the ledger.
locked() { "$WINLIST" 2>/dev/null | awk '$1 == "layer" && $2 >= 1900 && $4 > 0 && $5 == "loginwindow" { found = 1 } END { exit !found }'; }
# With the lid closed the built-in display is off: the game can't take it fullscreen and runs in an 854 x 480 window,
# which made four calibration runs worthless (2026-10-01). Wait for the lid, and run again if fullscreen failed anyway.
lid() { ioreg -r -k AppleClamshellState -d 4 | grep -q '"AppleClamshellState" = Yes'; }
# The GPU lock (tools/bench/gpulock.inc), held for the whole run: offline GPU tests (tools/bench/gpuwait.sh) wait for it,
# and the run waits for them. It lives in the main checkout's bench_out, whichever worktree runs this.
source "$ROOT/tools/bench/gpulock.inc"
MAIN=$(git -C "$ROOT" worktree list --porcelain 2>/dev/null | awk 'NR==1{print $2}')
LOCK=${MAIN:-$ROOT}/bench_out/gpu.lockdir
# The window lister (bench_out/winlist, built from tools/bench/winlist.swift): a worktree uses the main checkout's.
WINLIST=$OUT/winlist; [ -x "$WINLIST" ] || WINLIST=${MAIN:-$ROOT}/bench_out/winlist
trap 'gpu_unlock "$LOCK"' EXIT
TIMED=; case " $* " in *" -PbenchFly="*) TIMED=1 ;; esac
for TRY in 1 2 3; do
  if lid; then
    echo "$(date +%T) waiting for the lid to open" >> "$OUT/dialog_waits.log"
    while lid; do sleep 20; done
  fi
  if [ -n "$TIMED" ] && [ -n "$(dialog)" ]; then
    echo "$(date +%T) waiting for a clear screen: $(dialog)" >> "$OUT/dialog_waits.log"
    while [ -n "$(dialog)" ]; do sleep 20; done
  fi
  gpu_lock "$LOCK" "game run $(basename "$LOG")" 2>> "$OUT/dialog_waits.log"
  [ -d run/saves/claudeworld ] && mv run/saves/claudeworld "$OUT/old_worlds/claudeworld-$(date +%s)"
  mkdir -p run/saves   # a fresh worktree has no run directory yet
  cp -Rp "../fixtures/$FIX" run/saves/claudeworld   # -p keeps mtimes, so the LOD's region cache still matches
  # Let Spotlight and the security scanners finish with the fresh copy before timing anything.
  sleep ${SETTLE:-30}
  : > "$LOG.dialogs"; : > "$LOG.locked"
  # The game is forked by gradle's daemon and inherits its priority: a daemon started at background priority (by a build
  # under taskpolicy -b) runs the game on the efficiency cores, 3-5x slower (2026-10-01). Stop any such daemon first.
  if ps -Ao pri,command | awk '/GradleDaemon/ && !/awk/ && $1 < 20 { found = 1 } END { exit !found }'; then
    echo "$(date +%T) stopping a background-priority gradle daemon" >> "$OUT/dialog_waits.log"
    env JAVA_HOME=/opt/homebrew/opt/openjdk@25 ./gradlew --stop > /dev/null 2>&1
  fi
  ( sleep "$T"; pkill -f KnotClient ) & WD=$!
  ( while sleep 10; do d=$(dialog); [ -n "$d" ] && echo "METALMC_DIALOG $(date +%T) $d" >> "$LOG.dialogs"
                       locked && echo "METALMC_LOCKED $(date +%T)" >> "$LOG.locked"; done ) & DW=$!
  env JAVA_HOME=/opt/homebrew/opt/openjdk@25 ./gradlew runClient "$@" > "$LOG" 2>&1
  echo "exit $?" >> "$LOG"
  kill $WD $DW 2>/dev/null
  cat "$LOG.dialogs" "$LOG.locked" >> "$LOG"
  if grep -q "Couldn't enter fullscreen" "$LOG" && [ $TRY -lt 3 ]; then
    echo "$(date +%T) $(basename "$LOG"): the game couldn't go fullscreen (try $TRY), running it again" >> "$OUT/dialog_waits.log"
    sleep 30; continue
  fi
  if [ -z "$TIMED" ] || [ ! -s "$LOG.dialogs" ]; then break; fi
  echo "$(date +%T) $(basename "$LOG"): a dialog showed up during the run (try $TRY): $(head -1 "$LOG.dialogs")" >> "$OUT/dialog_waits.log"
done
: > "$LOG.dialogs"; : > "$LOG.locked"
