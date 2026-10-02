#!/bin/bash
# usage: gpuwait.sh <command args...>
# Runs the command holding the GPU lock, which every game run takes too (run_mc.sh): offline GPU tests and timed game
# runs spoil each other's timings, so they take turns. Waits for the lock first. The lock is a directory (mkdir is
# atomic) in the main checkout's bench_out, whichever worktree this runs from; a lock whose holder died is taken over.
# Builds don't need it if they run at background priority (`taskpolicy -b swift build ...`; never gradle under -b).
#   bash tools/bench/gpuwait.sh python3 tools/fartest.py .build/release render ...
MAIN=$(git -C "$(dirname "$0")" worktree list --porcelain 2>/dev/null | awk 'NR==1{print $2}')
LOCK=${MAIN:-$(cd "$(dirname "$0")/../.." && pwd)}/bench_out/gpu.lockdir
source "$(dirname "$0")/gpulock.inc"
gpu_lock "$LOCK" "$*"
trap 'gpu_unlock "$LOCK"' EXIT
"$@"
