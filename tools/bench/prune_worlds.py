"""Keeps the newest N world copies in bench_out/old_worlds (default 2) and deletes the rest."""
import os, shutil, sys
d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "bench_out", "old_worlds")
keep = int(sys.argv[1]) if len(sys.argv) > 1 else 2
if os.path.isdir(d):
    worlds = sorted((os.path.join(d, w) for w in os.listdir(d)), key=os.path.getmtime, reverse=True)
    for w in worlds[keep:]:
        shutil.rmtree(w, ignore_errors=True)
    print(f"kept {min(keep, len(worlds))}, removed {max(0, len(worlds) - keep)}")
