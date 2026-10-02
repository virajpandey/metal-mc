"""Offline test of the world-space irradiance cache (Sources/MetalMCNative/Gi.swift, docs/gi-design.md) through
libMetalMCNative, outside the game: builds a scene's acceleration structures, runs the cache for some frames from a
fixed camera, prints GPU times and cache statistics, and writes debug views as PNGs.

usage:
  python3 gicache.py selftest                       key round trips and ray/winding checks on the synthetic scene, and
                                                    the resolve's RG11B10 packing against the texture unit's
  python3 gicache.py synth <outdir> [w h frames]    the synthetic scene's views (house, glowstone room, tunnel, outside)
  python3 gicache.py litsynth <outdir> [w h frames] lit mode's relight without and with the cache (debug views 6 and 7)
                                                    on the synthetic scene, in lit mode's light (its daylight curve, as
                                                    in the game without the sky): the house's room at mid-morning and
                                                    at dusk, and outside
  python3 gicache.py edit <outdir>                  opens the house's east wall after convergence: invalidation
  python3 gicache.py region <r.X.Z.mca> <outdir> [x y z yaw pitch]   real terrain at the panel's resolution (3456 x 2234)
  python3 gicache.py bench <r.X.Z.mca> <outdir> [tag=<prefix>] [load=<table.bin>] [frames=160] [timed=96] [flush=96]
                    [fly=<blocks a frame>] [view=x,y,z,yaw,pitch]
                                                    the game's frame for the cache on real terrain at the panel's
                                                    resolution in lit mode's light (mmc_debug_gi_profile): each frame
                                                    after `flush` MB of other traffic (cold caches, as in a frame) and
                                                    the game's shadow rays, then the cache's encoders with the game
                                                    profile's timestamps and the anti-aliasing resolve's load loop with
                                                    the cache's share (a proxy); once with the encoders one after
                                                    another (each one's own time), once alternating frames with and
                                                    without the cache (what it costs the frame), and with fly= while
                                                    flying. Saves the converged table (<tag>table.bin; load= starts
                                                    from one instead) and writes what the relight takes from the cache
                                                    as floats (<tag>up9.f32, for cmp) and the irradiance and lit mode
                                                    views
  python3 gicache.py cmp <a.f32> <b.f32>            compares two bench pictures (to the bit, and the light's statistics)
  python3 gicache.py src <out.metal>                the upsample header and kernels as built (the start of a variant)
  python3 gicache.py exp <r.X.Z.mca> <outdir> <variant.metal ...> [rounds=3] [timed=64] [conv=256] [convfor=<names>]
                    [fly=<blocks a frame>]
                                                    kernel variants (src's text, edited) against the built-in kernels
                                                    in one process: their times on one converged table, alternating;
                                                    what the relight takes on that table (to the bit); the light
                                                    converged from empty against two built-in runs
  python3 gicache.py lag <r.X.Z.mca> <outdir> [variant.metal ...]
                                                    how fast the light follows a 20 degree jump of the sun (the
                                                    built-in kernels, or variants): against the light settled there

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
lib.mmc_debug_gi_reset.argtypes = []
lib.mmc_debug_gi_reset.restype = None


def bind(name, argtypes, restype):
    """Binds a debug entry point if this build has it (an older build, METALMC_DYLIB, lacks the newer ones)."""
    if hasattr(lib, name):
        f = getattr(lib, name)
        f.argtypes, f.restype = argtypes, restype


bind("mmc_debug_gi_profile", [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_float, ctypes.c_float, ctypes.c_int32,
                              ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_int32, ctypes.c_float,
                              ctypes.POINTER(ctypes.c_double), ctypes.c_char_p, ctypes.c_int32], ctypes.c_int32)
bind("mmc_debug_gi_view", [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_float, ctypes.c_float, ctypes.c_int32,
                           ctypes.c_int32, ctypes.c_float, ctypes.c_int32, ctypes.c_float, ctypes.c_int32,
                           ctypes.POINTER(ctypes.c_uint8), ctypes.c_void_p], ctypes.c_int32)
bind("mmc_debug_gi_save", [ctypes.c_char_p], ctypes.c_int32)
bind("mmc_debug_gi_load", [ctypes.c_char_p], ctypes.c_int32)
bind("mmc_debug_gi_diff", [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)], ctypes.c_int32)
bind("mmc_debug_gi_reload", [ctypes.c_char_p], ctypes.c_int32)
bind("mmc_debug_gi_shader_source", [ctypes.c_char_p, ctypes.c_int32], ctypes.c_int32)

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
    print(f"RG11B10 packing (the resolve's) against the texture unit's: {out[40]:.0f} values, {out[41]:.0f} differ"
          + (f" (first: {out[42]!r} gives {int(out[43]):#010x} there, {int(out[44]):#010x} here)" if out[41] else ""))
    return out[1] == 0 and out[3] == 0 and out[40] > 0 and out[41] == 0


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


# mmc_debug_gi_profile's modes: the game's shadow rays first; encoders one after another (each time its own); every
# other frame without the cache (the frame's time with it less without it: what it costs the frame).
SHADOW, SERIAL, AB = 1, 2, 4


def profile(cam, yaw, pitch, w, h, frames, sun, flush, quiet=False, mode=SHADOW | SERIAL, fly=0.0):
    """The game's frame for the cache (mmc_debug_gi_profile): per-encoder GPU ms, medians and fastest. fly: blocks the
    camera moves along its view each frame."""
    out = (ctypes.c_double * 40)()
    labels = ctypes.create_string_buffer(2048)
    n = lib.mmc_debug_gi_profile(cam[0], cam[1], cam[2], yaw, pitch, w, h, frames, sun, flush, mode, fly, out, labels, len(labels))
    if n <= 0:
        sys.exit("profile failed")
    names = labels.value.decode().split("|")
    r = ({name: (out[2 * i], out[2 * i + 1]) for i, name in enumerate(names)}
         | {"cache": (out[24], out[25]), "frame": (out[26], out[27]), "frame without the cache": (out[28], out[29]),
            "rays": (out[30], out[30]), "updated": (out[31], out[31])})
    if not quiet:
        how = (("encoders one after another" if mode & SERIAL else "encoders overlapping as in a frame")
               + (", every other frame without the cache" if mode & AB else "") + (f", flying {fly} blocks a frame" if fly else ""))
        print(f"  {frames} frames at {w}x{h}, {flush} MB flushed before each, {how} (medians over the last 3/4, fastest):")
        for i, name in enumerate(names):
            print(f"    {out[2 * i]:6.3f} ({out[2 * i + 1]:6.3f})  {name}")
        print(f"    {out[24]:6.3f} ({out[25]:6.3f})  the cache's encoders, first start to last end")
        print(f"    {out[26]:6.3f} ({out[27]:6.3f})  the frame" + (f"; without the cache {out[28]:.3f} ({out[29]:.3f}): +{out[26] - out[28]:.3f}" if mode & AB else ""))
        print(f"    rays a frame {out[30]:.0f}, cells updated in the last frame {out[31]:.0f}, live cells {out[32]:.0f}", flush=True)
    return r


def view(cam, yaw, pitch, w, h, sun, mode, exposure, png=None, scale=1, floats_path=None):
    """Debug view `mode` of the cache as it stands (mmc_debug_gi_view: no frame run)."""
    ow, oh = w // scale, h // scale
    buf = (ctypes.c_uint8 * (ow * oh * 4))()
    fl = (ctypes.c_float * (w * h * 4))() if floats_path else None
    if not lib.mmc_debug_gi_view(cam[0], cam[1], cam[2], yaw, pitch, w, h, sun, mode, exposure, scale, buf,
                                 ctypes.cast(fl, ctypes.c_void_p) if fl is not None else None):
        sys.exit("view failed")
    if png:
        write_png(png, ow, oh, bytes(buf))
    if floats_path:
        with open(floats_path, "wb") as f:
            f.write(bytes(fl))


def cmp(a, b):
    """Two debug-view-9 float pictures (mmc_debug_gi_diff)."""
    out = (ctypes.c_double * 16)()
    if not lib.mmc_debug_gi_diff(a.encode(), b.encode(), out):
        sys.exit("diff failed (sizes?)")
    n = out[0]
    print(f"  {a} vs {b}: {out[1] / n * 100:.3f}% of pixels the same to the bit; light in both {out[2] / n * 100:.2f}%, "
          f"in one only {out[3] / n * 100:.4f}% ({out[3]:.0f} px), different no-data codes {out[4]:.0f} px; mean luminance "
          f"{out[5]:.5f} vs {out[6]:.5f} ({(out[6] / max(out[5], 1e-9) - 1) * 100:+.3f}%), mean |difference| {out[7] * 100:.3f}%, "
          f"largest {out[8] * 100:.2f}%, {out[9]:.0f} px over 1/32", flush=True)
    return out


def bench(path, outdir, cam=None, yaw=0.0, pitch=20.0, tag="", load=None, frames=160, timed=96, flush=96, fly=0.0):
    """The game's frame for the cache on real terrain at the panel's resolution, in lit mode's light: converge (or load
    a saved table), profile it per encoder, save the table, and write debug view 9 as floats (for cmp) and the
    irradiance and lit mode views as PNGs. The views run no frame, so two builds give the same pictures on the same
    table to the bit if they resolve and upsample alike."""
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(path).split(".")
    rx, rz = int(name[1]), int(name[2])
    scene(1, path, ROUTE)
    if cam is None:
        cam = (rx * 512 + 256.5, 140.0, rz * 512 + 40.5)
    w, h, sun = 3456, 2234, -45.0
    lib.mmc_debug_gi_lit_light(1)
    lib.mmc_debug_gi_reset()
    pre = os.path.join(outdir, tag)
    if load:
        if not lib.mmc_debug_gi_load(load.encode()):
            sys.exit(f"couldn't load {load}")
        print(f"loaded {load}")
    else:
        print(f"converging ({frames} frames):")
        profile(cam, yaw, pitch, w, h, frames, sun, flush)
        if not lib.mmc_debug_gi_save((pre + "table.bin").encode()):
            sys.exit("save failed")
    # The views first (they run no frame), then the timed frames.
    view(cam, yaw, pitch, w, h, sun, 9, 1.0, floats_path=pre + "up9.f32")
    view(cam, yaw, pitch, w, h, sun, 1, 1.0, pre + "irradiance.png", scale=2)
    view(cam, yaw, pitch, w, h, sun, 7, 1.0, pre + "litmode-gi.png", scale=2)
    print("timed:")
    r = profile(cam, yaw, pitch, w, h, timed, sun, flush, mode=SHADOW | SERIAL)
    if load:
        lib.mmc_debug_gi_load(load.encode())
    else:
        lib.mmc_debug_gi_load((pre + "table.bin").encode())
    r = r | {"ab": profile(cam, yaw, pitch, w, h, 2 * timed, sun, flush, mode=SHADOW | AB)}
    if fly:
        # Flying from the converged table: cells come into view and change level every frame, as in the game's benchmark.
        print(f"flying {fly} blocks a frame:")
        lib.mmc_debug_gi_load((load or pre + "table.bin").encode())
        r["fly"] = profile(cam, yaw, pitch, w, h, timed, sun, flush, mode=SHADOW | SERIAL, fly=fly)
        lib.mmc_debug_gi_load((load or pre + "table.bin").encode())
        r["flyab"] = profile(cam, yaw, pitch, w, h, 2 * timed, sun, flush, mode=SHADOW | AB, fly=fly)
    return r


def exp(path, outdir, variants, cam=None, yaw=0.0, pitch=20.0, frames=160, timed=64, flush=96, rounds=3, conv=256, convfor=None, fly=0.0):
    """Kernel variants (files of the upsample header and kernels, mmc_debug_gi_reload) against the built-in ones in one
    process, on one scene: per-encoder times on the converged table, alternating; what the relight takes on the same
    table (to the bit); and the light after converging from empty (two built-in runs give the noise)."""
    os.makedirs(outdir, exist_ok=True)
    name = os.path.basename(path).split(".")
    rx, rz = int(name[1]), int(name[2])
    scene(1, path, ROUTE)
    if cam is None:
        cam = (rx * 512 + 256.5, 140.0, rz * 512 + 40.5)
    w, h, sun = 3456, 2234, -45.0
    lib.mmc_debug_gi_lit_light(1)
    names = ["builtin"] + variants

    def load(v):
        if not lib.mmc_debug_gi_reload(None if v == "builtin" else v.encode()):
            sys.exit(f"reload {v} failed")

    lib.mmc_debug_gi_reset()
    print(f"converging ({frames} frames, built-in kernels)", flush=True)
    profile(cam, yaw, pitch, w, h, frames, sun, flush, quiet=True)
    table = os.path.join(outdir, "table.bin")
    lib.mmc_debug_gi_save(table.encode())
    times = {v: [] for v in names}
    costs = {v: [] for v in names}
    for r in range(rounds):
        for v in names:
            load(v)
            lib.mmc_debug_gi_load(table.encode())
            times[v].append(profile(cam, yaw, pitch, w, h, timed, sun, flush, quiet=True, mode=SHADOW | SERIAL, fly=fly))
            lib.mmc_debug_gi_load(table.encode())
            ab = profile(cam, yaw, pitch, w, h, 2 * timed, sun, flush, quiet=True, mode=SHADOW | AB, fly=fly)
            costs[v].append((ab["frame"][0] - ab["frame without the cache"][0], ab["frame"][1] - ab["frame without the cache"][1]))
    print(f"per encoder, ms (median of {timed} frames, flush {flush} MB, encoders one after another; each round from the same table),")
    print("then what the cache costs the frame (frames with it less without it, alternating, encoders overlapping as in a frame):")
    for v in names:
        keys = [k for k in times[v][0] if k not in ("flush", "frame without the cache")]
        print(f"  {os.path.basename(v)}:")
        for k in keys:
            vals = [t[k][0] for t in times[v]]
            print(f"    {' '.join(f'{x:6.3f}' for x in vals)}   {k}")
        print(f"    {' '.join(f'{x[0]:6.3f}' for x in costs[v])}   the cache's cost in the frame (median; fastest {' '.join(f'{x[1]:.3f}' for x in costs[v])})")
    # The relight's input on the same table, each variant's resolve and upsample.
    print("what the relight takes, on the converged table:")
    base = None
    for v in names:
        load(v)
        lib.mmc_debug_gi_load(table.encode())
        f = os.path.join(outdir, os.path.basename(v) + ".table.f32")
        view(cam, yaw, pitch, w, h, sun, 9, 1.0, floats_path=f)
        if base is None:
            base = f
        else:
            cmp(base, f)
    # Converged from empty: two built-in runs (the noise), then each variant.
    print(f"converged from empty ({conv} frames):")
    runs = []
    for v in ["builtin"] + [v for v in names if convfor is None or v == "builtin" or os.path.basename(v) in convfor]:
        load(v)
        lib.mmc_debug_gi_reset()
        profile(cam, yaw, pitch, w, h, conv, sun, flush, quiet=True)
        f = os.path.join(outdir, os.path.basename(v) + f".conv{len(runs)}.f32")
        view(cam, yaw, pitch, w, h, sun, 9, 1.0, floats_path=f)
        runs.append(f)
    for f in runs[1:]:
        cmp(runs[0], f)
    load("builtin")


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
    elif cmd == "bench":
        opts = dict(a.split("=", 1) for a in sys.argv[4:] if "=" in a)
        view_at = [float(v) for v in opts["view"].split(",")] if "view" in opts else None
        bench(sys.argv[2], sys.argv[3], tuple(view_at[:3]) if view_at else None, view_at[3] if view_at else 0.0,
              view_at[4] if view_at else 20.0, tag=opts.get("tag", ""), load=opts.get("load"), frames=int(opts.get("frames", 160)),
              timed=int(opts.get("timed", 96)), flush=int(opts.get("flush", 96)), fly=float(opts.get("fly", 0)))
    elif cmd == "lag":
        # How fast the light follows a jump of the sun (converged at -45 degrees, then at -25): what the relight takes
        # after 32, 128 and 384 frames against the built-in kernels converged at -25 from empty.
        path, outdir, variants = sys.argv[2], sys.argv[3], [a for a in sys.argv[4:]]
        os.makedirs(outdir, exist_ok=True)
        name = os.path.basename(path).split(".")
        rx, rz = int(name[1]), int(name[2])
        scene(1, path, ROUTE)
        cam, w, h = (rx * 512 + 256.5, 140.0, rz * 512 + 40.5), 3456, 2234
        lib.mmc_debug_gi_lit_light(1)
        ref = os.path.join(outdir, "ref.f32")
        lib.mmc_debug_gi_reload(None)
        lib.mmc_debug_gi_reset()
        profile(cam, 0.0, 20.0, w, h, 512, -25.0, 96, quiet=True)
        view(cam, 0.0, 20.0, w, h, -25.0, 9, 1.0, floats_path=ref)
        for v in ["builtin"] + variants:
            if not lib.mmc_debug_gi_reload(None if v == "builtin" else v.encode()):
                sys.exit(f"reload {v} failed")
            lib.mmc_debug_gi_reset()
            r = profile(cam, 0.0, 20.0, w, h, 320, -45.0, 96, quiet=True)
            print(f"{os.path.basename(v)}: converged at -45, {r['rays'][0]:.0f} rays and {r['updated'][0]:.0f} cells a frame; then at -25:", flush=True)
            done = 0
            for n in (32, 128, 384):
                profile(cam, 0.0, 20.0, w, h, n - done, -25.0, 96, quiet=True)
                done = n
                f = os.path.join(outdir, f"{os.path.basename(v)}.{n}.f32")
                view(cam, 0.0, 20.0, w, h, -25.0, 9, 1.0, floats_path=f)
                cmp(ref, f)
        lib.mmc_debug_gi_reload(None)
    elif cmd == "cmp":
        cmp(sys.argv[2], sys.argv[3])
    elif cmd == "src":
        buf = ctypes.create_string_buffer(1 << 20)
        n = lib.mmc_debug_gi_shader_source(buf, len(buf))
        if n >= len(buf):
            sys.exit(f"the source is {n} bytes, over the buffer's")
        with open(sys.argv[2], "w") as f:
            f.write(buf.value.decode())
        print(f"{n} bytes")
    elif cmd == "exp":
        args = [a for a in sys.argv[4:] if "=" not in a]
        opts = dict(a.split("=", 1) for a in sys.argv[4:] if "=" in a)
        exp(sys.argv[2], sys.argv[3], args, timed=int(opts.get("timed", 64)), rounds=int(opts.get("rounds", 3)),
            convfor=opts["convfor"].split(",") if "convfor" in opts else None, fly=float(opts.get("fly", 0)),
            flush=int(opts.get("flush", 96)), conv=int(opts.get("conv", 256)))
    else:
        sys.exit(__doc__)
