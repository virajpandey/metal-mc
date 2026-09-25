"""Builds LOD for a world save through libMetalMCNative, outside the game, and prints the result.
usage: python3 lodtest.py <world dir> [far] [centerX] [centerZ]"""
import ctypes
import sys
import time

lib = ctypes.CDLL("/Users/rachnap/Projects/metal-mc/.build/release/libMetalMCNative.dylib")
lib.mmc_lod_open.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
lib.mmc_lod_open.restype = ctypes.c_int32
lib.mmc_lod_status.argtypes = [ctypes.POINTER(ctypes.c_int64)]

world = sys.argv[1]
far = int(sys.argv[2]) if len(sys.argv) > 2 else 2048
cx = int(sys.argv[3]) if len(sys.argv) > 3 else 0
cz = int(sys.argv[4]) if len(sys.argv) > 4 else 0
t0 = time.time()
print("open:", lib.mmc_lod_open(world.encode(), far, cx, cz), flush=True)
out = (ctypes.c_int64 * 3)()
last, stable = -1, 0
while stable < 32:   # streaming never "finishes": wait until the quad count holds for 8 s
    lib.mmc_lod_status(out)
    stable = stable + 1 if out[0] == 2 and out[2] == last else 0
    last = out[2]
    time.sleep(0.25)
print(f"state={out[0]} nodes={out[1]} quads={out[2]} ({out[2] * 8 / 1e6:.1f} MB) in {time.time() - t0:.1f} s", flush=True)
