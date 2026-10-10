#!/bin/bash
# usage: tools/bench/twins.sh <label> [LAB_EXP]
# The twin census in one command (docs/lab-mode.md, "Twins"): a lab session with the look's switches (LAB_EXP; default
# lit,nearchunks,rtshadows,sky,gi,water,coloredlight,post), every scene shot with its class mask (`shot` + `mask`), each
# scored against its twin by SEUS (tools/twindiff.swift against TWINS_SEUS/<TWINS_PREFIX>-<scene>.png), and the boards
# (ours | SEUS over mask | the lightness difference) and a summary table (tools/bench/twinsum.py) into TWINS_OUT.
#   bash tools/bench/twins.sh census2                     -> bench_out/twins/census2/ of the main checkout
#   TWINS_NOGAME=1 bash tools/bench/twins.sh census2      score the shots already there again (a new scorer, new twins)
# Env: TWINS_OUT (<main checkout>/bench_out/twins/<label>), TWINS_SEUS (~/Projects/metal-mc-private/twins/census1),
#      TWINS_PREFIX (seus1), TWINS_SCENES (all nine, in scenes.json's order), TWINS_SETTLE (ticks after each scene, 300),
#      TWINS_TRACE=1 (per-pass GPU times at the last scene into passes.txt), TWINS_NOGAME, LAB_MINUTES (20), and lab.sh's
#      (LAB_FAR, LAB_FIXTURE, ...). Boards hold SEUS's pixels: TWINS_OUT stays out of the repository (bench_out is).
LABEL=${1:?usage: twins.sh <label> [LAB_EXP]}
EXP=${2:-${LAB_EXP:-lit,nearchunks,rtshadows,sky,gi,water,coloredlight,post}}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT" || exit 1
MAIN=$(git worktree list --porcelain 2>/dev/null | awk 'NR==1{print $2}')
MAIN=${MAIN:-$ROOT}
SEUS=${TWINS_SEUS:-$HOME/Projects/metal-mc-private/twins/census1}
PREFIX=${TWINS_PREFIX:-seus1}
OUT=${TWINS_OUT:-$MAIN/bench_out/twins/$LABEL}
SCENES=${TWINS_SCENES:-noon_overview night_torches rain forest torch_cave mineshaft sunset_water water_closeup mountain_view}
mkdir -p "$OUT" "$ROOT/bench_out"

# The scorer, built when its source is newer.
TD=$ROOT/bench_out/twindiff
if [ ! -x "$TD" ] || [ tools/twindiff.swift -nt "$TD" ]; then
  swiftc -O tools/twindiff.swift -o "$TD" || exit 1
fi

if [ -z "$TWINS_NOGAME" ]; then
  # A throwaway scene first: a session's first scene caught the LOD mid-build (census 1's noon_overview). The real
  # visit to the same scene then finds it built. After each `scene`, TWINS_SETTLE ticks (300): `scene` waits for the LOD
  # and the terrain, not for the GI cache, the shadows' structures and the anti-aliasing to converge (after a long jump
  # that takes about 400 ticks: mountain_view scored 22.9 at once, 19.4 converged).
  SCRIPT=$OUT/session.txt
  {
    echo "scene noon_overview"
    echo "wait 400"
    for s in $SCENES; do
      echo "scene $s"
      echo "wait ${TWINS_SETTLE:-300}"
      echo "shot twins-$LABEL-$s"
      echo "mask twins-$LABEL-$s"
    done
    # TWINS_TRACE=1: per-pass GPU times of 5 frames at the last scene (tools/bench/passes.py on the game log).
    if [ -n "$TWINS_TRACE" ]; then echo "trace 5"; echo "wait 40"; fi
    echo "quit"
  } > "$SCRIPT"
  echo "== $(date '+%T') twins $LABEL: a lab session with $EXP"
  LAB_EXP=$EXP LAB_MINUTES=${LAB_MINUTES:-20} LAB_SCRIPT=$SCRIPT bash tools/bench/lab.sh "twins-$LABEL"
  for s in $SCENES; do
    mv -f "mod/run/screenshots/twins-$LABEL-$s.png" "$OUT/ours-$s.png" 2>/dev/null
    mv -f "mod/run/screenshots/twins-$LABEL-$s-mask.png" "$OUT/mask-$s.png" 2>/dev/null
  done
  cp "bench_out/run_twins-$LABEL.log" "$OUT/game.log" 2>/dev/null
  # Commands that failed, and scenes that didn't settle.
  grep -h '\[metalmc-lab\]' "bench_out/run_twins-$LABEL.log" | grep -E ' -> error|not settled' | tail -20
  if [ -n "$TWINS_TRACE" ]; then python3 tools/bench/passes.py "bench_out/run_twins-$LABEL.log" > "$OUT/passes.txt" 2>&1; fi
fi

echo "== $(date '+%T') scoring against $SEUS/$PREFIX-*.png"
for s in $SCENES; do
  if [ -f "$OUT/ours-$s.png" ] && [ -f "$OUT/mask-$s.png" ] && [ -f "$SEUS/$PREFIX-$s.png" ]; then
    "$TD" "$OUT/ours-$s.png" "$SEUS/$PREFIX-$s.png" "$OUT/mask-$s.png" --name "$s" --json "$OUT/score-$s.json" \
      --board "$OUT/board-$s.jpg" --quiet &
  else
    echo "$s: no ours, mask or twin ($OUT/ours-$s.png, $OUT/mask-$s.png, $SEUS/$PREFIX-$s.png)"
  fi
done
wait
python3 tools/bench/twinsum.py "$OUT" > "$OUT/summary.md"
cat "$OUT/summary.md"
