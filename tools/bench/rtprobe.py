"""Hardware ray-tracing probe: builds a world's LOD offline, then traces a primary + sun shadow ray per pixel against
the chosen nodes' geometry (mmc_debug_rt_probe) and prints acceleration-structure build and trace times.

usage: python3 tools/bench/rtprobe.py <world dir> [dylib] [x y z yaw pitch]
  default camera: (8, 150, 8) looking north (yaw 180), 10 degrees down. Region data only (no generated far terrain).
"""
import ctypes
import os
import sys
import time
from ctypes import POINTER, c_char_p, c_double, c_float, c_int32, c_int64

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
args = sys.argv[1:]
world = args[0]
dylib = args[1] if len(args) > 1 else os.path.join(root, ".build/release/libMetalMCNative.dylib")
cam = [float(v) for v in args[2:7]] if len(args) >= 7 else [8.5, 150.0, 8.5, 180.0, 10.0]
lib = ctypes.CDLL(dylib)
lib.mmc_lod_open3.argtypes = [c_char_p] * 4 + [c_int32] * 3
lib.mmc_lod_open3.restype = c_int64
lib.mmc_lod_status.argtypes = [POINTER(c_int64)]
lib.mmc_lod_set_detail.argtypes = [c_int32]
lib.mmc_debug_rt_probe.argtypes = [c_double, c_double, c_double, c_float, c_float, c_int32, c_int32, POINTER(c_double)]
lib.mmc_debug_rt_probe.restype = c_int32

lib.mmc_lod_set_detail(768)
t0 = time.time()
lib.mmc_lod_open3(world.encode(), b"", b"", b"minecraft:overworld", 32768, int(cam[0]), int(cam[2]))
out = (c_int64 * 4)()
last, stable = -1, 0
while stable < 20 and time.time() - t0 < 600:
    lib.mmc_lod_status(out)
    stable = stable + 1 if out[0] == 2 and out[2] == last and out[3] == 1 else 0
    last = out[2]
    time.sleep(0.5)
print(f"LOD built: {out[1]} nodes, {out[2]} quads")
for w, h in [(1728, 1117), (3456, 2234)]:
    r = (c_double * 8)()
    ok = lib.mmc_debug_rt_probe(cam[0], cam[1], cam[2], cam[3], cam[4], w, h, r)
    if not ok:
        print("probe failed")
        break
    print(f"{w}x{h}: {int(r[0])} triangles in {int(r[1])} nodes, BLAS build {r[2]:.1f} ms total, "
          f"TLAS {r[3]:.2f} ms, trace (primary + shadow ray per pixel) {r[4]:.2f} ms, "
          f"hits {100 * r[5]:.1f}%, shadowed {100 * r[6]:.1f}% of hits, BLAS memory {r[7] / 1e6:.0f} MB")
