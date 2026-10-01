"""Offline test of the world-space irradiance cache (Sources/MetalMCNative/Gi.swift, docs/gi-design.md) through
libMetalMCNative, outside the game: builds a scene's acceleration structures, runs the cache for some frames from a
fixed camera, prints GPU times and cache statistics, and writes debug views as PNGs.

usage:
  python3 gicache.py selftest                       key round trips and ray/winding checks on the synthetic scene
  python3 gicache.py synth <outdir> [w h frames]    the synthetic scene's views (house, glowstone room, tunnel, outside)
  python3 gicache.py litsynth <outdir> [w h frames] lit mode's relight without and with the cache (debug views 6 and 7)
                                                    on the synthetic scene, in lit mode's light (its daylight curve, as
                                                    in the game without the sky): the house's room at mid-morning and
                                                    at dusk, and outside
  python3 gicache.py edit <outdir>                  opens the house's east wall after convergence: invalidation
  python3 gicache.py region <r.X.Z.mca> <outdir> [x y z yaw pitch]   real terrain at the panel's resolution (3456 x 2234)

The structures are built as RtShadows builds them with METALMC_EXP=lit,gi (zero copy: one geometry per quad range, a
hit's quad read from the node's buffer); GICACHE_PRIMDATA=1 builds them with per-triangle data instead (the other route,
for comparison). METALMC_DYLIB overrides the library path (default: .build/release under the repository). METALMC_GICAP,
_GIBUDGET, _GISPP and _GIRANGE set the cache's size, budget, samples and level-0 range as in the game."""
import ctypes
import os
import struct
import sys
import time
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib = ctypes.CDLL(os.environ.get("METALMC_DYLIB", os.path.join(REPO, ".build", "release", "libMetalMCNative.dylib")))
lib.mmc_debug_gi_scene.argtypes = [ctypes.c_int32, ctypes.c_char_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_double)]
lib.mmc_debug_gi_scene.restype = ctypes.c_int32
lib.mmc_debug_gi_run.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_float, ctypes.c_float, ctypes.c_int32,
                                 ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_float,
                                 ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_double)]
lib.mmc_debug_gi_run.restype = ctypes.c_int32
lib.mmc_debug_gi_set_block.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
lib.mmc_debug_gi_set_block.restype = ctypes.c_int32
lib.mmc_debug_gi_selftest.argtypes = [ctypes.POINTER(ctypes.c_double)]
lib.mmc_debug_gi_selftest.restype = ctypes.c_int32
lib.mmc_debug_gi_lit_light.argtypes = [ctypes.c_int32]
lib.mmc_debug_gi_lit_light.restype = None

# The structures' route for the cache (mmc_debug_gi_scene flags bit 1: per-triangle data instead of zero copy).
ROUTE = 2 if os.environ.get("GICACHE_PRIMDATA") == "1" else 0

MODES = {0: "lit", 1: "irradiance", 2: "cells", 3: "samples", 4: "nocache", 5: "bounce", 6: "litmode", 7: "litmode-gi"}
STATS = ["primary ms", "request ms", "schedule ms", "update ms", "resolve ms", "request max", "schedule max", "update max",
         "resolve max", "live cells", "converged", "created", "bucket full", "evicted", "no surface", "rays last frame",
         "updated last frame", "bounce hits cached", "bounce hits uncached", "flicker", "pixels with data", "debug view ms",
         "created by bounces", "first-point samples", "upsample ms", "upsample marginal ms", "px surface", "px looked up",
         "px no data", "upsample marginal no lookups ms"]


def write_png(path, w, h, rgba):
    raw = b"".join(b"\x00" + rgba[y * w * 4:(y + 1) * w * 4] for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def downscale(rgba, w, h, k):
    """Box-filters an RGBA8 image by k along each axis (for viewing native-resolution runs)."""
    if k == 1:
        return rgba, w, h
    ow, oh = w // k, h // k
    out = bytearray(ow * oh * 4)
    for y in range(oh):
        for x in range(ow):
            acc = [0, 0, 0, 0]
            for dy in range(k):
                row = ((y * k + dy) * w + x * k) * 4
                for dx in range(k):
                    for c in range(4):
                        acc[c] += rgba[row + dx * 4 + c]
            for c in range(4):
                out[(y * ow + x) * 4 + c] = acc[c] // (k * k)
    return bytes(out), ow, oh


def scene(kind, path=None, flags=None):
    flags = ROUTE if flags is None else flags
    out = (ctypes.c_double * 8)()
    t0 = time.time()
    ok = lib.mmc_debug_gi_scene(kind, path.encode() if path else None, flags, out)
    if not ok:
        sys.exit("scene failed")
    print(f"scene {kind} {path or 'synthetic'}: {out[0] / 1e6:.2f} M triangles, {out[1] / 2**20:.1f} MB of structures, "
          f"{out[3]:.0f} nodes, built in {out[2] / 1000:.1f} s ({time.time() - t0:.1f} s wall)", flush=True)
    return out[0], out[1]


def run(cam, yaw, pitch, w, h, frames, sun, mode, png=None, scale=1, quiet=False, exposure=1.0):
    buf = (ctypes.c_uint8 * (w * h * 4))()
    st = (ctypes.c_double * 32)()
    if not lib.mmc_debug_gi_run(cam[0], cam[1], cam[2], yaw, pitch, w, h, frames, sun, mode, exposure, buf, st):
        sys.exit("run failed")
    if png:
        img, ow, oh = downscale(bytes(buf), w, h, scale)
        write_png(png, ow, oh, img)
    if not quiet:
        s = {STATS[i]: st[i] for i in range(len(STATS))}
        print(f"  {frames} frames at {w}x{h}, sun {sun}, view {MODES[mode]}{' -> ' + png if png else ''}")
        print(f"    GPU ms: request {s['request ms']:.3f} (max {s['request max']:.3f}), schedule {s['schedule ms']:.3f}, "
              f"update {s['update ms']:.3f} (max {s['update max']:.3f}), resolve {s['resolve ms']:.3f} (max {s['resolve max']:.3f}); "
              f"total {s['request ms'] + s['schedule ms'] + s['update ms'] + s['resolve ms']:.3f}; lighting pass share (upsample) {s['upsample ms']:.3f}, of which the cache's {s['upsample marginal ms']:.3f} "
              f"({s['upsample marginal no lookups ms']:.3f} without lookups)")
        print(f"    cells: live {s['live cells']:.0f}, converged {s['converged']:.0f}, created by the screen {s['created']:.0f} "
              f"and by bounces {s['created by bounces']:.0f}, bucket full {s['bucket full']:.0f}, evicted {s['evicted']:.0f}, "
              f"no surface {s['no surface']:.0f}, first-point samples {s['first-point samples']:.0f}; last frame updated "
              f"{s['updated last frame']:.0f} cells with {s['rays last frame']:.0f} rays; bounce hits cached/uncached "
              f"{s['bounce hits cached']:.0f}/{s['bounce hits uncached']:.0f}; flicker {s['flicker'] * 100:.3f}%; "
              f"samples with data {s['pixels with data'] * 100:.1f}%; upsample: {s['px looked up'] / max(s['px surface'], 1) * 100:.1f}% "
              f"of surface pixels looked cells up, {s['px no data'] / max(s['px surface'], 1) * 100:.1f}% had cells without samples", flush=True)
    return st


def selftest():
    scene(0)
    out = (ctypes.c_double * 64)()
    if not lib.mmc_debug_gi_selftest(out):
        sys.exit("selftest failed to run")
    print(f"keys: {out[0]:.0f} checked, {out[1]:.0f} wrong; rays: {out[2]:.0f} checked, {out[3]:.0f} wrong")
    names = ["ground from above", "ceiling from inside", "ground's top from inside the stone", "tunnel, west"]
    for i, n in enumerate(names):
        b = 4 + i * 7
        print(f"  {n}: no cull hit {out[b]:.0f} face {out[b + 1]:.0f} front {out[b + 2]:.0f} at {out[b + 3]:.2f}; "
              f"back culled hit {out[b + 4]:.0f} face {out[b + 5]:.0f} at {out[b + 6]:.2f}")
    return out[1] == 0 and out[3] == 0


VIEWS = {
    # name: camera, yaw, pitch, sun angle (degrees; -60: mid-morning, the sun in the east), exposure (interiors are dark)
    "house": ((114.5, 69.0, 101.5), 60.0, 22.0, -60.0, 3.0),
    "glow": ((69.5, 66.5, 69.5), 135.0, 8.0, -60.0, 1.5),
    "tunnel": ((203.5, 65.6, 182.5), 90.0, 4.0, -60.0, 6.0),
    "outside": ((150.0, 85.0, 80.0), 56.3, 18.6, -60.0, 1.0),
}


def synth(outdir, w=1152, h=745, frames=96):
    os.makedirs(outdir, exist_ok=True)
    scene(0)
    for name, (cam, yaw, pitch, sun, ev) in VIEWS.items():
        lib.mmc_debug_gi_reset()
        print(f"{name}:")
        run(cam, yaw, pitch, w, h, frames, sun, 0, os.path.join(outdir, f"{name}-lit.png"), exposure=ev)
        for mode in (4, 1, 2, 3, 5):
            run(cam, yaw, pitch, w, h, 1, sun, mode, os.path.join(outdir, f"{name}-{MODES[mode]}.png"), quiet=True, exposure=ev)


def mean_luma(rgba):
    """Mean 8-bit luma of an RGBA8 image."""
    n = len(rgba) // 4
    r, g, b = sum(rgba[0::4]), sum(rgba[1::4]), sum(rgba[2::4])
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / max(n, 1)


LIT_VIEWS = {
    # name: camera, yaw, pitch, sun angle (vanilla's, degrees), exposure (the same for both pictures of a view)
    "room": ((114.5, 69.0, 101.5), 60.0, 22.0, -60.0, 4.0),
    "room-dusk": ((114.5, 69.0, 101.5), 60.0, 22.0, 82.0, 4.0),
    "outside": ((150.0, 85.0, 80.0), 56.3, 18.6, -60.0, 1.0),
}


def litsynth(outdir, w=1152, h=745, frames=128):
    """Lit mode's relight on the synthetic scene without the cache (its sky term: the open sky's light on the face times
    the sky light level's curve) and with the cache's light in its place (debug views 6 and 7), in lit mode's light."""
    os.makedirs(outdir, exist_ok=True)
    scene(0)
    lib.mmc_debug_gi_lit_light(1)
    for name, (cam, yaw, pitch, sun, ev) in LIT_VIEWS.items():
        lib.mmc_debug_gi_reset()
        print(f"{name}:")
        st = run(cam, yaw, pitch, w, h, frames, sun, 7, os.path.join(outdir, f"{name}-lit-gi.png"), exposure=ev)
        buf = (ctypes.c_uint8 * (w * h * 4))()
        stats = (ctypes.c_double * 32)()
        lumas = {}
        for mode in (6, 7):
            lib.mmc_debug_gi_run(cam[0], cam[1], cam[2], yaw, pitch, w, h, 1, sun, mode, ev, buf, stats)
            lumas[mode] = mean_luma(bytes(buf))
            img = bytes(buf)
            write_png(os.path.join(outdir, f"{name}-{'lit' if mode == 6 else 'lit-gi'}.png"), w, h, img)
        print(f"  mean luma (exposure {ev}): lit {lumas[6]:.1f}, lit+gi {lumas[7]:.1f}; frame-to-frame change of the cache's "
              f"light {st[19] * 100:.2f}%", flush=True)
    lib.mmc_debug_gi_lit_light(0)


def edit(outdir, w=1152, h=745):
    os.makedirs(outdir, exist_ok=True)
    scene(0)
    cam, yaw, pitch, sun, ev = VIEWS["house"]
    lib.mmc_debug_gi_reset()
    run(cam, yaw, pitch, w, h, 96, sun, 0, os.path.join(outdir, "edit-0-before.png"), exposure=ev)
    # Open a 4 x 4 hole in the east wall below the window (x 117, z 105-108, y 64-65), and one in the roof.
    t0 = time.time()
    for z in range(105, 109):
        for y in (64, 65):
            lib.mmc_debug_gi_set_block(117, y, z, 0)
    for x in range(104, 110):
        for z in range(102, 106):
            lib.mmc_debug_gi_set_block(x, 72, z, 0)
    print(f"  edits (re-mesh and structures, test only) {time.time() - t0:.1f} s")
    done = 0
    for n, frames in ((1, 1), (2, 8), (3, 32), (4, 96)):
        run(cam, yaw, pitch, w, h, frames - done, sun, 0, os.path.join(outdir, f"edit-{n}-after-{frames}.png"), exposure=ev)
        done = frames


def region(path, outdir, cam=None, yaw=0.0, pitch=20.0):
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(path).split(".")
    rx, rz = int(name[1]), int(name[2])
    tri0, bytes0 = scene(1, path, 1)
    tri2, bytes2 = scene(1, path, 2)
    tri1, bytes1 = scene(1, path, ROUTE)
    print(f"structure memory: {bytes0 / tri0:.1f} bytes per triangle without what bounce rays need (RtShadows without the cache), "
          f"{bytes2 / tri2:.1f} with per-triangle data ({(bytes2 - bytes0) / 2**20:+.1f} MB), "
          f"{bytes1 / tri1:.1f} {'zero copy' if ROUTE == 0 else 'per-triangle data'} ({(bytes1 - bytes0) / 2**20:+.1f} MB, the route the runs use)")
    if cam is None:
        cam = (rx * 512 + 256.5, 140.0, rz * 512 + 40.5)
    w, h = 3456, 2234
    lib.mmc_debug_gi_reset()
    print("real terrain, panel resolution:")
    run(cam, yaw, pitch, w, h, 4, -45.0, 0, os.path.join(outdir, "region-lit-4.png"), scale=3)
    run(cam, yaw, pitch, w, h, 124, -45.0, 0, os.path.join(outdir, "region-lit-128.png"), scale=3)
    for mode in (4, 1, 2, 3):
        run(cam, yaw, pitch, w, h, 1, -45.0, mode, os.path.join(outdir, f"region-{MODES[mode]}.png"), scale=3, quiet=True)
    # Turning around: the cells behind the camera are new (and some came from bounce rays).
    print("turned 180 degrees:")
    run(cam, yaw + 180, pitch, w, h, 1, -45.0, 0, os.path.join(outdir, "region-turn-1.png"), scale=3)
    run(cam, yaw + 180, pitch, w, h, 31, -45.0, 0, os.path.join(outdir, "region-turn-32.png"), scale=3)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        sys.exit(0 if selftest() else 1)
    elif cmd == "synth":
        a = [int(v) for v in sys.argv[3:6]]
        synth(sys.argv[2], *a)
    elif cmd == "litsynth":
        a = [int(v) for v in sys.argv[3:6]]
        litsynth(sys.argv[2], *a)
    elif cmd == "edit":
        edit(sys.argv[2])
    elif cmd == "region":
        extra = [float(v) for v in sys.argv[4:9]]
        if len(extra) == 5:
            region(sys.argv[2], sys.argv[3], tuple(extra[:3]), extra[3], extra[4])
        else:
            region(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
