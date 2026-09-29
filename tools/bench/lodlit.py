"""Builds a world's overworld LOD offline through the native library and reports quads and lit quads per level.

usage: python3 tools/bench/lodlit.py <world dir> [center x] [center z] [--live]
  --live: feed the region files through the live-chunk path (mmc_debug_ingest_region) with no region directory,
          as on a multiplayer server, instead of reading them as region files.
Set METALMC_EXP=nofarlight to compare against no light past level 0. Needs .build/release/libMetalMCNative.dylib.
"""
import ctypes
import os
import sys
import tempfile
import time
from ctypes import POINTER, c_char_p, c_int32, c_int64

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
lib = ctypes.CDLL(os.path.join(root, ".build/release/libMetalMCNative.dylib"))
lib.mmc_lod_open3.argtypes = [c_char_p] * 4 + [c_int32] * 3
lib.mmc_lod_open3.restype = c_int64
lib.mmc_lod_status.argtypes = [POINTER(c_int64)]
lib.mmc_debug_lod_lit_quads.argtypes = [POINTER(c_int64)]
lib.mmc_debug_ingest_region.argtypes = [c_char_p]
lib.mmc_debug_ingest_region.restype = c_int32
lib.mmc_lod_set_detail.argtypes = [c_int32]

args = [a for a in sys.argv[1:] if not a.startswith("--")]
live = "--live" in sys.argv
world = args[0]
cx = int(args[1]) if len(args) > 1 else 8
cz = int(args[2]) if len(args) > 2 else 8
lib.mmc_lod_set_detail(768)
t0 = time.time()
if live:
    lib.mmc_lod_open3(b"", tempfile.mkdtemp(prefix="lodstore-").encode(), b"", b"minecraft:overworld", 32768, cx, cz)
    rd = os.path.join(world, "dimensions/minecraft/overworld/region")
    near = sorted((f for f in os.listdir(rd) if f.endswith(".mca")),
                  key=lambda f: sum(abs(int(v) * 512 + 256 - c) for v, c in zip(f.split(".")[1:3], (cx, cz))))[:16]
    chunks = sum(lib.mmc_debug_ingest_region(os.path.join(rd, f).encode()) for f in near)
    print(f"ingested {chunks} live chunks from {len(near)} regions")
else:
    lib.mmc_lod_open3(world.encode(), b"", b"", b"minecraft:overworld", 32768, cx, cz)
out = (c_int64 * 4)()
last, stable = -1, 0
while stable < 20 and time.time() - t0 < 600:
    lib.mmc_lod_status(out)
    stable = stable + 1 if out[0] == 2 and out[2] == last and (live or out[3] == 1) else 0
    last = out[2]
    time.sleep(0.5)
lit = (c_int64 * 32)()
lib.mmc_debug_lod_lit_quads(lit)
print(f"built in {time.time() - t0 - 10:.0f} s: {out[1]} nodes, {out[2]} quads")
for level in range(16):
    q, l = lit[2 * level], lit[2 * level + 1]
    if q:
        print(f"L{level}: {q} quads, {l} lit ({100 * l / q:.2f}%)")
