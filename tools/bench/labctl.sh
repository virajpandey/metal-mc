#!/bin/bash
# usage: tools/bench/labctl.sh [-t TIMEOUT_S] [-n RESULTS] "command" ...
# Drives a running lab session (tools/bench/lab.sh, docs/lab-mode.md): appends the commands to mod/run/metalmc-control.txt
# and waits until the control log has a result for each (or RESULTS of them: `tour` makes 2 per scene and 1 of its own),
# then prints those results. TIMEOUT_S: 600.
#   bash tools/bench/labctl.sh "scene sunset_water" "shot sunset-a"
#   bash tools/bench/labctl.sh -n 19 "tour look2"
R=$(cd "$(dirname "$0")/../../mod/run" && pwd) || exit 1
T=600; N=
while true; do
  case "$1" in
    -t) T=$2; shift 2 ;;
    -n) N=$2; shift 2 ;;
    *) break ;;
  esac
done
N=${N:-$#}
before=$(grep -c ' -> ' "$R/metalmc-control.log" 2>/dev/null)
for c in "$@"; do echo "$c" >> "$R/metalmc-control.txt"; done
end=$(( $(date +%s) + T ))
until [ $(( $(grep -c ' -> ' "$R/metalmc-control.log" 2>/dev/null) - before )) -ge "$N" ] || [ "$(date +%s)" -ge "$end" ]; do
  sleep 0.5
done
grep ' -> ' "$R/metalmc-control.log" | tail -n "$N"
