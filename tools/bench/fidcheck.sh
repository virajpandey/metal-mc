#!/bin/bash
# usage: fidcheck.sh <label> [extra gradle args...]
# One LOD candidate against the native-resolution references (nA = vanilla RD 32, nB = RD 12, nR2 = LOD reference):
# near = the 192-512 block band against vanilla, far = beyond 512 against nR2. Needs the screen unlocked (native size).
L=$1; shift
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
FS=$ROOT/bench_out/fidscore
[ -x "$FS" ] || swiftc -O "$ROOT/tools/fidscore.swift" -o "$FS"
BENCH_NOBUILD=1 BENCH_TIMEOUT=900 bash "$ROOT/tools/bench/fidelity.sh" $L 12 8192 -PlodGenerate=0 "$@"
cd "$ROOT/mod/run/screenshots" || exit 1
echo "== $L near"; "$FS" . nA nB $L | tail -1
echo "== $L far"; "$FS" . nR2 nA $L | tail -1
