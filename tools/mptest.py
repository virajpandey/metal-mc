"""Multiplayer-path test without a server.
1. Build the LOD from a world's region files (single-player path).
2. Build it again with no region files, feeding every chunk through the live-ingest path, saved to a store.
3. Reopen from the saved store only.
All three should give the same node and quad counts.
usage: mptest.py <world dir> <far>"""
import ctypes
import os
import sys
import tempfile
import time

lib = ctypes.CDLL("/Users/rachnap/Projects/metal-mc/.build/release/libMetalMCNative.dylib")
lib.mmc_lod_open.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
lib.mmc_lod_open.restype = ctypes.c_int32
lib.mmc_lod_open2.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
lib.mmc_lod_open2.restype = ctypes.c_int32
lib.mmc_lod_status.argtypes = [ctypes.POINTER(ctypes.c_int64)]
lib.mmc_debug_ingest_region.argtypes = [ctypes.c_char_p]
lib.mmc_debug_ingest_region.restype = ctypes.c_int32

world, far = sys.argv[1], int(sys.argv[2])
region_dir = os.path.join(world, "dimensions/minecraft/overworld/region")
store = tempfile.mkdtemp(prefix="lodstore-")


def wait_stable(label, t0):
    out = (ctypes.c_int64 * 3)()
    last, stable = -1, 0
    while stable < 32:   # quad count unchanged for 8 s
        lib.mmc_lod_status(out)
        stable = stable + 1 if out[0] == 2 and out[2] == last else 0
        last = out[2]
        time.sleep(0.25)
    print(f"{label}: nodes={out[1]} quads={out[2]} ({time.time() - t0 - 8:.1f} s)", flush=True)
    return out[1], out[2]


t0 = time.time()
lib.mmc_lod_open(world.encode(), far, 0, 0)
a = wait_stable("region files", t0)
lib.mmc_lod_close()

t0 = time.time()
lib.mmc_lod_open2(b"", store.encode(), far, 0, 0)
n = 0
for f in sorted(os.listdir(region_dir)):
    if f.endswith(".mca"):
        n += lib.mmc_debug_ingest_region(os.path.join(region_dir, f).encode())
print(f"ingested {n} chunks in {time.time() - t0:.1f} s", flush=True)
b = wait_stable("live ingest", t0)
lib.mmc_lod_close()
time.sleep(3)   # let the update thread finish its last pass and save
files = [f for f in os.listdir(store) if f.endswith(".lod")]
size = sum(os.path.getsize(os.path.join(store, f)) for f in files)
print(f"store: {len(files)} region files, {size / 1e6:.1f} MB", flush=True)

t0 = time.time()
lib.mmc_lod_open2(b"", store.encode(), far, 0, 0)
c = wait_stable("reopened store", t0)
lib.mmc_lod_close()
print("MATCH" if a == b == c else f"MISMATCH {a} {b} {c}")
