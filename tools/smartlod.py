"""Smart LOD selection (METALMC_EXP=smartlod) checked offline: builds the LOD of a save through libMetalMCNative, then
compares the distance rule with smart selection at several camera positions, heights and directions: quads the view
would draw, the projected error left on screen, where the quads went (rough or flat terrain), that every area is
drawn exactly once, and how often tiles flip along a camera path.

usage: python3 tools/smartlod.py [world dir] [far]
MMC_LIB overrides the library (default: this checkout's .build-wt, then .build). Never starts the game."""
import ctypes
import math
import os
import sys
import time

os.environ["METALMC_EXP"] = ",".join(filter(None, [os.environ.get("METALMC_EXP", ""), "smartlod"]))
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lib_path = os.environ.get("MMC_LIB") or next(p for p in [f"{root}/.build-wt/release/libMetalMCNative.dylib",
                                                           f"{root}/.build/release/libMetalMCNative.dylib"] if os.path.exists(p))
lib = ctypes.CDLL(lib_path)
I64P = ctypes.POINTER(ctypes.c_int64)
lib.mmc_lod_open.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
lib.mmc_lod_open.restype = ctypes.c_int32
lib.mmc_lod_status.argtypes = [I64P]
lib.mmc_lod_center.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_int32]
lib.mmc_debug_lod_tile_errors.argtypes = [I64P, ctypes.c_int32]
lib.mmc_debug_lod_tile_errors.restype = ctypes.c_int32
lib.mmc_debug_lod_nodes.argtypes = [I64P, ctypes.c_int32]
lib.mmc_debug_lod_nodes.restype = ctypes.c_int32
lib.mmc_debug_lod_select_smart.argtypes = [ctypes.c_double] * 6 + [ctypes.c_int32, ctypes.c_double, ctypes.c_int64, ctypes.c_double,
                                                                    I64P, ctypes.c_int32, I64P]
lib.mmc_debug_lod_select_smart.restype = ctypes.c_int32
lib.mmc_debug_lod_tile_error_cost.argtypes = [ctypes.c_char_p, ctypes.c_int32, I64P]
lib.mmc_debug_lod_quads_in.argtypes = [ctypes.c_int32] * 4 + [I64P, ctypes.c_int32]
lib.mmc_debug_lod_quads_in.restype = ctypes.c_int32
lib.mmc_debug_lod_smart_graph_us.argtypes = [ctypes.c_int32]
lib.mmc_debug_lod_smart_graph_us.restype = ctypes.c_double

world = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fixtures/claudeworld-merged")
far = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
region_dir = f"{world}/dimensions/minecraft/overworld/region"
SPLIT = 3.0
PPR = 1117.0 / math.tan(math.radians(35))   # panel pixels per radian at the center of a 70-degree view
FLAT, ROUGH = 2.0, 6.0   # level-2 cell RMS error, blocks: under half a level-2 voxel is flat, over 1.5 voxels rough


def status():
    out = (ctypes.c_int64 * 4)()
    lib.mmc_lod_status(out)
    return list(out)


def wait_stable(label, hold=40):
    """Streaming never finishes: wait for the first full build, then until the node and quad counts hold for `hold` x
    0.25 s."""
    t0 = time.time()
    last, stable = None, 0
    while stable < hold:
        s = status()
        cur = (s[1], s[2])
        stable = stable + 1 if s[0] == 2 and s[3] == 1 and cur == last else 0
        last = cur
        time.sleep(0.25)
    print(f"[{label}] nodes={last[0]} quads={last[1]} ({time.time() - t0:.0f} s)", flush=True)


def tile_errors():
    buf = (ctypes.c_int64 * (35 * 20000))()
    n = lib.mmc_debug_lod_tile_errors(buf, 20000)
    nodes = {}
    for i in range(n):
        o = buf[35 * i:35 * i + 35]
        nodes[(o[0], o[1], o[2])] = [(o[3 + 2 * t] / 1000, o[4 + 2 * t] / 1000) if o[3] >= 0 else None for t in range(16)]
    return nodes


timings = {"distance": [], "smart": [], "graph": []}   # microseconds per selection, per graph rebuild


def select(cam, yaw, pitch, mode, err, budget, fov=70.0):
    cap = 60000
    buf = (ctypes.c_int64 * (9 * cap))()
    stats = (ctypes.c_int64 * 7)()
    n = lib.mmc_debug_lod_select_smart(cam[0], cam[1], cam[2], yaw, pitch, fov, mode, err, budget, SPLIT, buf, cap, stats)
    timings["distance" if mode == 0 else "smart"].append(stats[5])
    if stats[6]:
        timings["graph"].append(stats[6])
    return [tuple(buf[9 * i:9 * i + 9]) for i in range(n)], list(stats)


def pct(xs, p):
    if not xs:
        return 0
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


def ground(x, z):
    """Top of the terrain near (x, z): the highest +Y face of the finest level there."""
    cap = 20000
    buf = (ctypes.c_int64 * (8 * cap))()
    n = lib.mmc_debug_lod_quads_in(int(x) - 4, int(z) - 4, int(x) + 4, int(z) + 4, buf, cap)
    best = {}
    for i in range(n):
        lvl, _, y, _, face = buf[8 * i:8 * i + 5]
        if face == 2:
            best[lvl] = max(best.get(lvl, -64), y)
    return best[min(best)] if best else 64


lib.mmc_debug_lod_tile_errors_synthetic.argtypes = [ctypes.c_int32] * 4 + [I64P]
lib.mmc_debug_lod_smart_background_check.restype = ctypes.c_int32

# 0. The error metric on made-up grids, against values worked out by hand (level 1: 2-block voxels, level 2 above).
print("Tile errors of made-up level-1 grids (RMS, max in blocks; tiles listed per value):")
for name, args, expect in [("flat, top voxel 31 (odd)", (0, 31, 0, 0), {(0.0, 0.0): 16}),
                           ("flat, top voxel 30 (even)", (0, 30, 0, 0), {(2.0, 2.0): 16}),
                           ("20-voxel tower at (5, 5)", (1, 31, 0, 0), {(1.25, 40.0): 1, (0.0, 0.0): 15}),
                           ("cliff 31 -> 51 at x 128", (2, 31, 51, 128), {(0.0, 0.0): 16}),
                           ("cliff 31 -> 51 at x 129", (2, 31, 51, 129), {(7.071, 40.0): 4, (0.0, 0.0): 12})]:
    out = (ctypes.c_int64 * 32)()
    lib.mmc_debug_lod_tile_errors_synthetic(*args, out)
    got = {}
    for t in range(16):
        got[(out[2 * t] / 1000, out[2 * t + 1] / 1000)] = got.get((out[2 * t] / 1000, out[2 * t + 1] / 1000), 0) + 1
    print(f"  {name:26s} {'ok' if got == expect else 'MISMATCH'} {got}")

print(f"library {lib_path}\nopen {world} far {far}: {lib.mmc_lod_open(world.encode(), far, 0, 0)}", flush=True)
wait_stable("first build")
print(f"Background graph builds (0 = old graph kept until the new one landed, splits carried): "
      f"{lib.mmc_debug_lod_smart_background_check()}")

# 1. Error distributions per level (tile RMS / max, blocks).
errs = tile_errors()
print("\nTile errors (the parent's surface against this level's, per 64 x 64-voxel tile, tiles with data), blocks:")
for lvl in sorted({k[0] for k in errs}):
    rms = [e[0] for k, v in errs.items() if k[0] == lvl for e in v if e is not None and e[1] > 0]
    mx = [e[1] for k, v in errs.items() if k[0] == lvl for e in v if e is not None and e[1] > 0]
    if not rms:
        print(f"  level {lvl}: no error data")
        continue
    print(f"  level {lvl}: {len(rms):4d} tiles  RMS p10 {pct(rms, 10):5.2f} p50 {pct(rms, 50):5.2f} p90 {pct(rms, 90):5.2f}"
          f" max {max(rms):5.1f}   max-error p50 {pct(mx, 50):5.1f} p90 {pct(mx, 90):5.1f} max {max(mx):4.0f}")

# Would level 2's own roughness say where level 1 is worth building farther out? For each level-2 tile whose four
# level-1 tiles exist: its RMS error (level 3 against level 2) against theirs (level 2 against level 1).
pairs = []
for (lvl, nx, nz), v in errs.items():
    if lvl != 2:
        continue
    for t, e in enumerate(v):
        if e is None or e[1] <= 0:
            continue
        # Level-2 tile (tx, tz) covers level-1 node (2nx + tx // 2, 2nz + tz // 2), tiles 2 (tx % 2) + i, 2 (tz % 2) + j.
        tx, tz = t % 4, t // 4
        child = errs.get((1, 2 * nx + tx // 2, 2 * nz + tz // 2))
        if not child:
            continue
        sub = [child[(2 * (tz % 2) + j) * 4 + 2 * (tx % 2) + i] for j in range(2) for i in range(2)]
        if any(c is None or c[1] <= 0 for c in sub):
            continue
        pairs.append((e[0], math.sqrt(sum(c[0] ** 2 for c in sub) / 4)))
if len(pairs) > 2:
    mx_, my_ = sum(p[0] for p in pairs) / len(pairs), sum(p[1] for p in pairs) / len(pairs)
    cov = sum((a - mx_) * (b - my_) for a, b in pairs)
    r = cov / math.sqrt(sum((a - mx_) ** 2 for a, _ in pairs) * sum((b - my_) ** 2 for _, b in pairs))
    top = sorted(pairs, key=lambda p: -p[0])[:len(pairs) // 3]
    print(f"\nLevel-2 roughness as a predictor of level-1 error: {len(pairs)} tiles, Pearson r {r:.2f}; the roughest third by"
          f" level 2 holds {100 * sum(p[1] ** 2 for p in top) / sum(p[1] ** 2 for p in pairs):.0f}% of the level-1 squared error")

# 2. The cost of measuring the errors, against filling and meshing the same node.
print("\nError pass cost (one region; fill + mesh vs lodTileErrors):")
for name in ["r.0.0.mca", "r.-3.2.mca", "r.2.-4.mca"]:
    for lvl in (1, 2):
        out = (ctypes.c_int64 * 34)()
        lib.mmc_debug_lod_tile_error_cost(f"{region_dir}/{name}".encode(), lvl, out)
        if out[0]:
            print(f"  {name} level {lvl}: fill + mesh {out[0] / 1000:6.1f} ms, errors {out[1] / 1000:5.2f} ms ({100 * out[1] / out[0]:.1f}%)")

# 3. Roughness per 256-block cell: the level-2 tiles' RMS error (level 3 against level 2), which covers the whole save.
rough = {}
for (lvl, nx, nz), v in errs.items():
    if lvl != 2:
        continue
    for t, e in enumerate(v):
        if e is not None and e[1] > 0:
            rough[(nx * 4 + t % 4, nz * 4 + t // 4)] = e[0]
cells = list(rough.values())
print(f"\nLevel-2 cells (256 blocks) with data: {len(cells)}; flat (RMS < {FLAT}) {sum(v < FLAT for v in cells)},"
      f" rough (>= {ROUGH}) {sum(v >= ROUGH for v in cells)}")
xs = sorted({c[0] for c in rough}); zs = sorted({c[1] for c in rough})
print("  map (x right, z down; . no data, digit = RMS / 2 blocks, # >= 18):")
for cz in range(zs[0], zs[-1] + 1):
    row = "".join("." if (cx, cz) not in rough else ("#" if rough[(cx, cz)] >= 18 else str(int(rough[(cx, cz)] / 2)))
                  for cx in range(xs[0], xs[-1] + 1))
    print(f"  z {cz * 256:6d} {row}")
print(f"  x from {xs[0] * 256} in steps of 256")


def evaluate(tiles, cam):
    """Quads in view, projected error left on screen (against the finest level that exists there), and where the quads
    went. Measured on a fixed grid of 64-block cells, so selections compare: each chosen tile's facing quads and error
    are spread over its cells, and a cell's screen weight is its area over the squared distance from the camera to its
    middle at y 64 (16 blocks as a floor). Classes (flat, mid, rough) are by the level-2 cell's RMS error; the level-0
    ring (the same for both rules) is left out of them."""
    cost = sum(t[4] for t in tiles)
    errs_px, weights = [], []
    cls = {k: [0.0, 0.0, 0.0] for k in ("flat", "mid", "rough")}   # quads, screen weight, error x weight
    per_level = {}
    for (lvl, nx, nz, t, facing, q, e, dh, d3) in tiles:
        per_level[lvl] = per_level.get(lvl, 0) + facing
        if facing == 0:
            continue
        span = 1 << lvl
        x0, z0 = (nx * 4 + t % 4) * span * 64, (nz * 4 + t // 4) * span * 64
        per_cell = facing / (span * span)
        for j in range(span):
            for i in range(span):
                mx, mz = x0 + 64 * i + 32, z0 + 64 * j + 32
                d = math.sqrt((mx - cam[0]) ** 2 + (mz - cam[2]) ** 2 + (64 - cam[1]) ** 2)
                w = 4096 / max(d, 16) ** 2
                err_px = (max(e, 0) / 1000) / max(d, 1) * PPR
                errs_px.append(err_px); weights.append(w)
                r = rough.get(((x0 + 64 * i) // 256, (z0 + 64 * j) // 256))
                if lvl >= 1 and r is not None:
                    k = "flat" if r < FLAT else ("rough" if r >= ROUGH else "mid")
                    cls[k][0] += per_cell; cls[k][1] += w; cls[k][2] += err_px * w
    wsum = sum(weights) or 1
    acc, p90 = 0.0, 0.0
    for a, b in sorted(zip(errs_px, weights)):
        acc += b
        if acc >= 0.9 * wsum:
            p90 = a
            break
    return {"cost": cost, "err_mean": sum(a * b for a, b in zip(errs_px, weights)) / wsum, "err_p90": p90,
            "err_max": max(errs_px, default=0), "classes": cls, "levels": per_level}


def coverage(tiles):
    """64-block cells drawn by more than one chosen tile, and the set of cells drawn."""
    seen = {}
    for (lvl, nx, nz, t, *_rest) in tiles:
        span = 1 << lvl
        x0, z0 = nx * 4 * span + (t % 4) * span, nz * 4 * span + (t // 4) * span
        for j in range(span):
            for i in range(span):
                seen[(x0 + i, z0 + j)] = seen.get((x0 + i, z0 + j), 0) + 1
    return sum(1 for c in seen.values() if c > 1), set(seen)


def fmt_classes(c):
    q = sum(v[0] for v in c.values()) or 1
    return "  ".join(f"{k} {100 * v[0] / q:4.1f}% of L1+ quads, {v[0] / max(v[1], 1e-9) / 1000:5.1f} K per unit screen, err {v[2] / v[1] if v[1] else 0:4.2f} px"
                     for k, v in c.items())


def fmt(name, r, st, base=None):
    lv = " ".join(f"L{k} {v // 1000}K" for k, v in sorted(r["levels"].items()) if v)
    d = f" ({100 * (r['cost'] / base - 1):+5.1f}%)" if base else "         "
    extra = f"  [splits: {st[1]} forced, {st[2]} error, {st[4]} fallback, {st[3]} over budget; {st[5]} us]" if st else ""
    return (f"  {name:28s} {r['cost'] / 1000:6.0f} K quads{d}  error px mean {r['err_mean']:4.2f} p90 {r['err_p90']:4.2f} max {r['err_max']:5.1f}{extra}\n"
            f"  {'':28s} {lv}\n  {'':28s} {fmt_classes(r['classes'])}")


def compare(label, views, modes, verbose=True):
    print(f"\n=== {label} ===")
    totals = {}
    for vlabel, cam, yaw, pitch in views:
        uni, _ = select(cam, yaw, pitch, 0, 0, 0)
        ru = evaluate(uni, cam)
        overlap_u, cells_u = coverage(uni)
        if verbose:
            print(f"\n{vlabel} at ({cam[0]:.0f}, {cam[1]:.0f}, {cam[2]:.0f}) yaw {yaw} pitch {pitch}:")
            print(fmt("distance rule", ru, None))
        rows = [("distance rule", ru)]
        for name, err, budget in modes:
            b = budget if budget > 1 else int(ru["cost"] * budget)
            sm, st = select(cam, yaw, pitch, 1, err, b)
            r = evaluate(sm, cam)
            overlap, cells = coverage(sm)
            if overlap or cells != cells_u:
                print(f"  !! {vlabel}, {name}: {overlap} cells drawn twice, {len(cells_u - cells)} cells not drawn, {len(cells - cells_u)} extra")
            if verbose:
                print(fmt(name, r, st, ru["cost"]))
            rows.append((name, r))
        for name, r in rows:
            t = totals.setdefault(name, {"cost": 0, "mean": 0.0, "p90": 0.0, "n": 0, "max": 0.0,
                                         "classes": {k: [0.0, 0.0, 0.0] for k in ("flat", "mid", "rough")}})
            t["cost"] += r["cost"]; t["mean"] += r["err_mean"]; t["p90"] += r["err_p90"]; t["n"] += 1
            t["max"] = max(t["max"], r["err_max"])
            for k, v in r["classes"].items():
                for j in range(3):
                    t["classes"][k][j] += v[j]
    print(f"\n{label}, totals over {len(views)} views:")
    base = totals["distance rule"]["cost"]
    for name, t in totals.items():
        print(f"  {name:28s} {t['cost'] / 1000:8.0f} K quads ({100 * (t['cost'] / base - 1):+5.1f}%), mean error {t['mean'] / t['n']:.3f} px,"
              f" mean p90 {t['p90'] / t['n']:.2f} px, worst {t['max']:.1f} px")
        print(f"  {'':28s} {fmt_classes(t['classes'])}")
    return totals


def recenter(x, z, label):
    lib.mmc_lod_center(x, z, 0)
    time.sleep(3)
    wait_stable(label)


MODES = [("smart 0.25 px, same budget", 0.25, 1.0), ("smart 0.25 px, 75% budget", 0.25, 0.75),
         ("smart 2 px, 3 M (defaults)", 2.0, 3_000_000)]
SWEEP = [(f"smart {e} px, no budget", e, 100_000_000) for e in (1.0, 1.5, 2.0, 3.0, 4.0, 6.0)]

# 4a. The world's middle: level 1 reaches past most of the save in every direction.
g0 = ground(0, 0)
middle = [(f"ground yaw {y}", (0.5, g0 + 2.0, 0.5), y, 0) for y in (0, 90, 180, 270)]
middle += [(f"y150 yaw {y}", (0.5, 150.0, 0.5), y, 20) for y in (45, 135, 225, 315)]
middle += [(f"y260 yaw {y}", (0.5, 260.0, 0.5), y, 30) for y in (0, 180)]
compare("middle: camera at (0, 0)", middle, MODES)
compare("middle, threshold sweep", middle, SWEEP, verbose=False)
print(f"\nGraph build (per mesh-set change): {lib.mmc_debug_lod_smart_graph_us(50):.0f} us for {status()[1]} nodes")

# 4b. A corner, looking across the save (5-7 km of terrain ahead): levels 2-4 carry most of the view.
cx, cz = -1900, -1900   # 150 blocks inside the save's corner (its data starts at -2048)
recenter(cx, cz, "recentered on the corner")
gc = ground(cx, cz)
corner = [(f"ground yaw {y}", (cx + 0.5, gc + 2.0, cz + 0.5), y, 0) for y in (300, 315, 330)]
corner += [(f"y150 yaw {y}", (cx + 0.5, 150.0, cz + 0.5), y, 15) for y in (300, 315, 330)]
corner += [(f"y260 yaw {y}", (cx + 0.5, 260.0, cz + 0.5), y, 25) for y in (315,)]
compare(f"corner: camera at ({cx}, {cz}) looking across the save", corner, MODES)
compare("corner, threshold sweep", corner, SWEEP, verbose=False)
print(f"\nGraph build (per mesh-set change): {lib.mmc_debug_lod_smart_graph_us(50):.0f} us for {status()[1]} nodes")

# 5. Hysteresis. A straight flight at 20 blocks/s sampled every 0.1 s, and a camera hovering in place while the view
# sways 3 degrees (every frame), counting tiles that change between samples with and without last frame's splits.
print("\n=== Tiles changing between frames ===")
for name, err, budget in [("defaults", 2.0, 3_000_000), ("tight budget", 0.25, 1_200_000)]:
    for path in ("flight", "hover"):
        for mode, mlabel in ((1, "no hysteresis"), (2, "hysteresis")):
            prev, flips, costs = None, [], []
            select((cx + 0.5, 150.0, cz + 0.5), 315, 15, 1, err, budget)   # reset the state
            for step in range(120):
                if path == "flight":
                    cam, yaw = (cx + 0.5 + 1.41 * step, 150.0, cz + 0.5 + 1.41 * step), 315 + 10 * math.sin(step / 15)
                else:
                    cam, yaw = (cx + 0.5, 150.0, cz + 0.5), 315 + 3 * math.sin(step / 4)
                tiles, st = select(cam, yaw, 15, mode, err, budget)
                cur = {t[:4] for t in tiles}
                if prev is not None:
                    flips.append(len(cur ^ prev))
                prev = cur
                costs.append(st[0])
            print(f"  {name:13s} {path:7s} {mlabel:14s} tiles changed per sample: mean {sum(flips) / len(flips):5.1f}, max {max(flips):3d};"
                  f" quads {min(costs) / 1000:.0f}-{max(costs) / 1000:.0f} K")

# 6. What finer levels farther out would cost: nodes as built now.
buf = (ctypes.c_int64 * (4 * 20000))()
n = lib.mmc_debug_lod_nodes(buf, 20000)
print()
for lvl in (0, 1, 2):
    q = [buf[4 * i + 3] for i in range(n) if buf[4 * i] == lvl]
    if q:
        print(f"Level {lvl}: {len(q)} nodes, quads per node mean {sum(q) / len(q) / 1000:.0f} K (max {max(q) / 1000:.0f} K),"
              f" about {sum(q) / len(q) * 12 / 1e6:.1f} MB of GPU buffers each (8-byte quads, 4-byte AO offsets, AO rims aside)")

print("\nSelection time (this process, M3 Pro), microseconds:")
for k, v in timings.items():
    if v:
        print(f"  {k:9s} median {pct(v, 50)}  p90 {pct(v, 90)}  max {max(v)}  ({len(v)} calls)")
