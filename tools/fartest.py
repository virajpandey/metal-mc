"""Offline checks of the far field's columns through libMetalMCNative (FarFieldDebug.swift), outside the game.
usage: python3 fartest.py <lib dir> shader <out.metal>            the march's shader source, for an offline compile check
       python3 fartest.py <lib dir> compile <file.metal>...        compiles variants of it (with and without FF_STATS)
       python3 fartest.py <lib dir> region <r.X.Z.mca> <level>     one region's columns: cells vs the voxels' tops
       python3 fartest.py <lib dir> stats <region dir>             cell statistics against level-1 truth from sample cameras
       python3 fartest.py <lib dir> seam <region dir> <far cache dir> <level 2-6>...   real columns vs the generator's
       python3 fartest.py <lib dir> render <world dir> <far cache dir> <out prefix> <far> <x> <y> <z> <yaw> <pitch> [...]
       python3 fartest.py <lib dir> serve <world dir> <far cache dir> <far> <x> <z> <command dir>
The far field must be on (METALMC_FARFIELD=<level>, set here to 1 unless given); METALMC_EXP picks the A/B switches.
render: FARTEST_CACHE=<dir> keeps the regions' quadrant cache there; FARTEST_PROBE=level,cx0,cz0,w,h prints cells' words;
FARTEST_FOV (vertical, degrees, default 70) and FARTEST_SIZE (WxH, default 1728x1117) set the camera; FARTEST_SHELL
(blocks, default 32) where rays start; FARTEST_REPEAT=n logs the march's median GPU time over n draws; FARTEST_NEAR
(blocks) stands in for the game's nearer geometry: a first draw from there fills the depth the timed ones test against.
serve: loads the world once, then runs the JSON commands it finds in <command dir> (*.cmd, in name order; each renamed
*.done when run, its results in *.out): {"prefix": out prefix, "views": [[x, y, z, yaw, pitch], ...], "variants":
[[name, shader file or "" for the built-in, stats 0/1], ...], "repeat": n, "rounds": n, "size": "WxH", "shell": blocks,
"fov": degrees, "near": blocks (FARTEST_NEAR: nearer geometry stood in for, 0 none), "env": {other FARTEST_ settings
for this command}}. Each view is drawn with each
variant (shader files: `shader`'s output, edited), rounds times
interleaved; the .out has each one's median GPU time and how its image differs from the first variant's. The GPU lock
(tools/bench/gpulock.inc, as gpuwait.sh takes it) is held while the world loads and while commands run (from one's start
until none is waiting), not between.
<far cache dir>: <game dir>/metalmc/lod/far/<world>-<seed hash> (read only)."""
import ctypes
import glob
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("METALMC_FARFIELD", "1")
os.environ.setdefault("METALMC_LOD0", "0")   # no level 0 (not drawn by the far field; slow to build)
lib = ctypes.CDLL(os.path.join(sys.argv[1], "libMetalMCNative.dylib"))
cmd = sys.argv[2]
TOOLS = os.path.dirname(os.path.abspath(__file__))


def gpu_lock(what):
    """Takes the GPU lock the way gpuwait.sh does (a bash holding it until told to let go); returns its handle."""
    main = subprocess.run(["git", "-C", TOOLS, "worktree", "list", "--porcelain"], capture_output=True, text=True).stdout.split("\n")[0].split(" ", 1)[1]
    lock = os.path.join(main, "bench_out", "gpu.lockdir")
    inc = os.path.join(TOOLS, "bench", "gpulock.inc")
    p = subprocess.Popen(["bash", "-c", f'source "{inc}"; gpu_lock "{lock}" "$1"; echo locked; read _; gpu_unlock "{lock}"', "lock", what],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    if p.stdout.readline().strip() != "locked":
        raise RuntimeError("couldn't take the GPU lock")
    return p


def gpu_unlock(p):
    p.stdin.write("\n")
    p.stdin.flush()
    p.wait()


def open_world(world, far_dir, far, x, z):
    lib.mmc_lod_open3.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
    lib.mmc_lod_open3.restype = ctypes.c_int64
    lib.mmc_lod_far_cache.argtypes = [ctypes.c_int64, ctypes.c_char_p]
    lib.mmc_lod_far_wanted.argtypes = [ctypes.c_int64, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32]
    lib.mmc_debug_far_render.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_float, ctypes.c_float, ctypes.c_float,
                                         ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_char_p]
    # FARTEST_CACHE: a directory for the regions' quadrant cache (default none: every region is read from its file).
    wid = lib.mmc_lod_open3(world.encode(), b"", os.environ.get("FARTEST_CACHE", "").encode(), b"minecraft:overworld", far, int(x), int(z))
    lib.mmc_lod_far_cache(wid, far_dir.encode())
    buf = (ctypes.c_int32 * 3 * 512)()
    status = (ctypes.c_int64 * 4)()
    t0, last, stable = time.time(), -1, 0
    while stable < 60:   # cached generated nodes loaded, the first pass done, and no node added for 15 s (they install in batches)
        lib.mmc_lod_far_wanted(wid, ctypes.cast(buf, ctypes.POINTER(ctypes.c_int32)), 512)
        lib.mmc_lod_status(status)
        stable = stable + 1 if status[3] == 1 and status[1] == last else 0
        last = status[1]
        time.sleep(0.25)
    print(f"world: {status[1]} nodes after {time.time() - t0:.0f} s", flush=True)


def render(x, y, z, yaw, pitch, fov, size, shell, path):
    for _ in range(100):
        r = lib.mmc_debug_far_render(x, y, z, yaw, pitch, fov, size[0], size[1], shell, path.encode())
        if r != 0:
            return r
        time.sleep(0.1)
    return 0


if cmd == "shader":
    buf = ctypes.create_string_buffer(1 << 20)
    n = lib.mmc_debug_far_shader_source(buf, len(buf))
    open(sys.argv[3], "w").write(buf.value.decode())
    print(f"{n} bytes")

elif cmd == "compile":   # <file.metal>...: each compiles (the device's compiler), with and without FF_STATS ("" the built-in)
    lib.mmc_debug_far_set_shader.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    for path in sys.argv[3:]:
        print(f"{path or 'built-in'}: {' '.join('ok' if lib.mmc_debug_far_set_shader(path.encode(), st) == 1 else 'FAILED' for st in (0, 1))}", flush=True)

elif cmd == "region":
    level = int(sys.argv[4])
    side = 512 >> level
    out = {}
    for mode in (0, 1, 2):
        words = (ctypes.c_uint32 * (2 * 256 * 256))()
        lib.mmc_debug_far_region(sys.argv[3].encode(), level, mode, words)
        out[mode] = words
    if level >= 2:
        print(f"through the level-2 quadrant: {sum(out[0][i] != out[2][i] for i in range(2 * 256 * 256))} words differ")
    diffs, canopy, cols = [], 0, 0
    for z in range(side):
        for x in range(side):
            i = z * 256 + x
            a0, a1, b0 = out[0][2 * i], out[0][2 * i + 1], out[1][2 * i]
            if a0 == 0:
                continue
            cols += 1
            top = lambda w: (w & 511) + ((w >> 9) & 127)
            diffs.append(top(b0) - max(top(a0), a1 & 511))
            canopy += a1 != 0
    diffs.sort()
    print(f"level {level}: {cols} columns, canopy {canopy / max(cols, 1):.1%}, voxel top - cell top: mean {sum(diffs) / max(len(diffs), 1):.2f}"
          f" median {diffs[len(diffs) // 2] if diffs else 0} p95 {diffs[int(len(diffs) * 0.95)] if diffs else 0} max {diffs[-1] if diffs else 0}")

elif cmd == "stats":
    out = (ctypes.c_double * (5 * 48))()
    t0 = time.time()
    lib.mmc_debug_far_eval(sys.argv[3].encode(), out)
    names = ["voxel tops", "cells, canopy dithered", "cells, canopy >= half", "cells, canopy any", "max, canopy any", "max, canopy >= half"]
    print(f"({time.time() - t0:.0f} s) per level: |silhouette| px, signed px, depth err, >5% off, one-sided px/ray, |top - truth| blocks, signed, canopy share")
    for l in range(2, 7):
        for v in range(6):
            o = (l - 2) * 48 + v * 8
            print(f"L{l} {names[v]:24s} {out[o]:6.2f} {out[o + 1]:+6.2f} {out[o + 2]:6.3f} {out[o + 3]:6.1%} {out[o + 4]:6.2f} {out[o + 5]:6.2f} {out[o + 6]:+6.2f} {out[o + 7]:6.1%}")

elif cmd == "seam":
    for level in sys.argv[5:]:
        out = (ctypes.c_double * 14)()
        lib.mmc_debug_far_seam(sys.argv[3].encode(), sys.argv[4].encode(), int(level), out)
        print(f"L{level}: {int(out[0])} columns; real - generated top: voxel tops mean {out[1]:+.2f} |.| {out[2]:.2f} p95 {out[3]:.1f};"
              f" cells mean {out[4]:+.2f} |.| {out[5]:.2f} p95 {out[6]:.1f}; ground alone mean {out[7]:+.2f} |.| {out[8]:.2f} p95 {out[9]:.1f}")
        print(f"     seam: {int(out[10])} boundary cells, |step| voxel tops {out[11]:.2f}, cells {out[12]:.2f}, generated terrain's own {out[13]:.2f}")

elif cmd == "render":
    world, far_dir, prefix, far = sys.argv[3], sys.argv[4], sys.argv[5], int(sys.argv[6])
    views = [tuple(float(v) for v in sys.argv[i:i + 5]) for i in range(7, len(sys.argv), 5)]
    x, _, z, _, _ = views[0]
    open_world(world, far_dir, far, x, z)
    if os.environ.get("FARTEST_PROBE"):   # level,cx0,cz0,w,h: print those cells' words (top, depth, canopy top/underside, mats)
        level, cx0, cz0, w, h = (int(v) for v in os.environ["FARTEST_PROBE"].split(","))
        words = (ctypes.c_uint32 * (2 * w * h))()
        lib.mmc_debug_far_cells(level, cx0, cz0, w, h, words)
        for j in range(h):
            print(f"z {cz0 + j}: " + " ".join(f"{words[2 * (j * w + i)] & 511}+{(words[2 * (j * w + i)] >> 9) & 127}/{(words[2 * (j * w + i)] >> 16) & 255}"
                                            f"|{words[2 * (j * w + i) + 1] & 511}-{(words[2 * (j * w + i) + 1] >> 9) & 511}" for i in range(w)))
    fov = float(os.environ.get("FARTEST_FOV", "70"))   # vertical field of view, degrees (a narrow one zooms in)
    size = tuple(int(v) for v in os.environ.get("FARTEST_SIZE", "1728x1117").split("x"))
    shell = float(os.environ.get("FARTEST_SHELL", "32"))   # where rays start (blocks): the game's is its nearest far tile
    for k, (x, y, z, yaw, pitch) in enumerate(views):
        path = f"{prefix}{k}.png"
        r = render(x, y, z, yaw, pitch, fov, size, shell, path)
        print(f"{path}: {r}", flush=True)

elif cmd == "serve":
    world, far_dir, far, x, z, cmd_dir = sys.argv[3], sys.argv[4], int(sys.argv[5]), float(sys.argv[6]), float(sys.argv[7]), sys.argv[8]
    lib.mmc_debug_far_set_shader.argtypes = [ctypes.c_char_p, ctypes.c_int32]
    lib.mmc_debug_far_last_ms.restype = ctypes.c_double
    lib.mmc_debug_far_compare.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_double)]
    lk = gpu_lock("fartest serve: world load")
    try:
        open_world(world, far_dir, far, x, z)
    finally:
        gpu_unlock(lk)
    print(f"serving {cmd_dir}", flush=True)
    env_set = []
    lk = None   # held from a command's start until no command is waiting
    while True:
        cmds = sorted(glob.glob(os.path.join(cmd_dir, "*.cmd")))
        if not cmds:
            if lk:
                gpu_unlock(lk)
                lk = None
            time.sleep(0.5)
            continue
        cpath = cmds[0]
        base = cpath[:-4]
        try:
            c = json.load(open(cpath))
        except Exception as e:   # half written: try again
            time.sleep(0.5)
            try:
                c = json.load(open(cpath))
            except Exception:
                os.rename(cpath, base + ".bad")
                print(f"{cpath}: bad command ({e})", flush=True)
                continue
        if c.get("quit"):
            os.rename(cpath, base + ".done")
            if lk:
                gpu_unlock(lk)
            break
        size = tuple(int(v) for v in c.get("size", "3456x2234").split("x"))
        os.environ["FARTEST_REPEAT"] = str(c.get("repeat", 12))
        os.environ["FARTEST_NEAR"] = str(c.get("near", 0))   # the nearer geometry stood in for (mmc_debug_far_render)
        for k in env_set:   # the previous command's
            os.environ.pop(k, None)
        env_set = list(c.get("env", {}))
        for k, v in c.get("env", {}).items():   # other settings of mmc_debug_far_render for this command
            os.environ[k] = str(v)
        shell, fov, rounds = float(c.get("shell", 512)), float(c.get("fov", 70)), int(c.get("rounds", 1))
        views, variants, prefix = c["views"], c["variants"], c["prefix"]
        times = {}
        lines = []
        print(f"=== {os.path.basename(cpath)}", flush=True)
        if not lk:
            lk = gpu_lock(f"fartest serve: {os.path.basename(cpath)}")
        try:
            for rd in range(rounds):
                for k, (vx, vy, vz, yaw, pitch) in enumerate(views):
                    for v in variants:
                        name, src, stats = v[0], v[1], (v[2] if len(v) > 2 else 0)
                        if lib.mmc_debug_far_set_shader(src.encode(), stats) != 1:
                            lines.append(f"{name}: shader failed")
                            continue
                        path = f"{prefix}{k}-{name}.png"
                        print(f"--- view {k} variant {name} round {rd}", file=sys.stderr, flush=True)
                        r = render(vx, vy, vz, yaw, pitch, fov, size, shell, path)
                        if r != 1:
                            lines.append(f"view {k} {name}: render failed ({r})")
                            continue
                        times.setdefault((k, name), []).append(lib.mmc_debug_far_last_ms())
        finally:
            lib.mmc_debug_far_set_shader(b"", 0)
        for k in range(len(views)):
            ref = variants[0][0]
            for v in variants:
                t = sorted(times.get((k, v[0]), [0]))
                diff = ""
                if v[0] != ref:
                    o = (ctypes.c_double * 8)()
                    a, b = f"{prefix}{k}-{ref}.png", f"{prefix}{k}-{v[0]}.png"
                    if lib.mmc_debug_far_compare(a.encode(), b.encode(), f"{prefix}{k}-{v[0]}-diff.png".encode(), o) == 1:
                        diff = f"  vs {ref}: {int(o[0])} px differ, {int(o[1])} by >2, {int(o[2])} by >16, max {int(o[3])}" + \
                               (f", box x {int(o[4])}-{int(o[6])} y {int(o[5])}-{int(o[7])}" if o[0] > 0 else "")
                lines.append(f"view {k} {v[0]:>14s}: {t[len(t) // 2]:.3f} ms (rounds {' '.join(f'{x:.3f}' for x in times.get((k, v[0]), []))}){diff}")
        out = "\n".join(lines)
        print(out, flush=True)
        open(base + ".out", "w").write(out + "\n")
        os.rename(cpath, base + ".done")
