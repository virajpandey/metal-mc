#!/usr/bin/env python3
"""Autotune the look's knobs against the SEUS twins in one lab session (docs/lighting-design.md, "The grade").

usage: autotune.py <label> [--knobs NAME,...] [--start FILE.json] [--scenes NAME,...] [--minutes 38] [--passes N]
                   [--exp FLAGS] [--out DIR] [--attach]
       autotune.py <label> --compare A.json B.json ... [--scenes ...]     shots and boards of fixed settings, same session
       autotune.py --apply FILE.json                                      write settings into mod/run/shaders now

The knobs are #defines in the hot-reloaded shader files (docs/lab-mode.md): POST_* in post.metal, LIT_* in
lit_relight_header.metal. Each pass of the search evaluates the current settings (the center) and, for every knob, the
center with that knob a step down and a step up, on every scene; scene by scene, so the camera moves once per scene and
pass:
  1. rewrite the #defines (the relight's header first, then post.metal, whose reload snaps the eye adaptation to the
     new light), wait for "reloaded" in shaderlab.log, then `wait` (the anti-aliasing's history settles; 40 ticks after
     a change to the light, 6 after one to post alone);
  2. `shot`, scored against the scene's SEUS twin with its class mask (`mask`, once per scene: the classes don't depend
     on the knobs) by tools/twindiff.swift in the background.
The objective is the sum over the scenes: twindiff's total for the day scenes, tools/bench/twinsum.py's night score
(terrain luma 35-45, warm torch pools, no crushed shadows) for night and caves. After a pass each knob moves to the
minimum of the parabola through its three points (within one step; toward the better probe if it isn't convex), all
at once, damped; a knob whose probes were both worse has its step shrunk. If the new center comes out worse than the
best so far, the search goes back to the best and halves every step.

Everything lands in --out (bench_out/agents/grade/autotune-<label> of the main checkout): evals.jsonl (every shot's
settings and scores), progress.log, best.json, and the best settings' shots and boards.
"""
import concurrent.futures as cf
import json
import math
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.dont_write_bytecode = True   # no __pycache__ in tools/bench
import twinsum  # noqa: E402

RUN = os.path.join(ROOT, "mod", "run")
SHADERS = os.environ.get("LAB_SHADERS", os.path.join(RUN, "shaders"))
CTL = os.path.join(RUN, "metalmc-control.txt")
CTL_LOG = os.path.join(RUN, "metalmc-control.log")
LAB_LOG = os.path.join(SHADERS, "shaderlab.log")
SEUS = os.environ.get("TWINS_SEUS", os.path.expanduser("~/Projects/metal-mc-private/twins/census1"))
PREFIX = os.environ.get("TWINS_PREFIX", "seus1")
TWINDIFF = os.path.join(ROOT, "bench_out", "twindiff")
FULL_LOOK = "lit,nearchunks,rtshadows,sky,gi,water,coloredlight,post"

# name: (file, default, low, high, step, log-scale step). The defaults are the built-in sources'.
KNOBS = {
    "POST_EV": ("post.metal", 0.0, -1.5, 2.5, 0.3, False),
    "POST_GRADE_EV": ("post.metal", 0.0, -1.5, 2.5, 0.3, False),
    "POST_DUSK": ("post.metal", 1.0, 0.0, 1.0, 0.25, False),
    "POST_CAVE": ("post.metal", 1.0, 0.0, 1.0, 0.25, False),
    "POST_ADAPT_REF": ("post.metal", -1.9, -3.5, -0.5, 0.35, False),
    "POST_ADAPT_DARK_NEAR": ("post.metal", 0.2, 0.0, 0.7, 0.1, False),
    "POST_ADAPT_DARK": ("post.metal", 0.7, 0.1, 1.0, 0.15, False),
    "POST_LOCAL": ("post.metal", 0.0, 0.0, 0.8, 0.15, False),
    "POST_LOCAL_DARK": ("post.metal", 0.0, 0.0, 0.8, 0.15, False),
    "POST_AGX_SAT": ("post.metal", 1.3, 0.9, 2.2, 0.15, False),
    "POST_AGX_POWER": ("post.metal", 1.0, 0.7, 1.4, 0.08, False),
    "POST_PURKINJE": ("post.metal", 0.0, 0.0, 1.0, 0.25, False),
    "LIT_SUN_GAIN": ("lit_relight_header.metal", 1.0, 0.4, 2.5, 0.15, True),
    "LIT_SKY_GAIN": ("lit_relight_header.metal", 1.0, 0.5, 5.0, 0.35, True),
    "LIT_SHADE_WARM": ("lit_relight_header.metal", 0.0, -0.4, 0.4, 0.08, False),
    "LIT_BLOCK_GAIN": ("lit_relight_header.metal", 1.0, 0.1, 2.0, 0.35, True),
    "LIT_BLOCK_WARM": ("lit_relight_header.metal", 0.0, -0.2, 0.7, 0.12, False),
    "LIT_BLOCK_SAT": ("lit_relight_header.metal", 1.0, 0.5, 3.0, 0.25, True),
    "LIT_BLOCK_FALLOFF": ("lit_relight_header.metal", 0.0, 0.0, 6.0, 0.8, False),
}
# The day's knobs. The exposure is POST_GRADE_EV (stops, on top of the eye adaptation, fading at dusk and under
# cover), not POST_ADAPT_REF: within the adaptation's knee a change of the reference moves the exposure by only 20-30% of
# it (lab, 2026-10-10). (The first sessions tuned POST_EV, which then faded the same way.)
DAY_KNOBS = ["POST_GRADE_EV", "POST_LOCAL_DARK", "POST_DUSK", "POST_AGX_SAT", "LIT_SKY_GAIN", "LIT_SUN_GAIN", "LIT_SHADE_WARM"]
NIGHT_KNOBS = ["POST_ADAPT_DARK", "LIT_BLOCK_GAIN", "LIT_BLOCK_FALLOFF", "LIT_BLOCK_WARM", "LIT_BLOCK_SAT"]
DAY_SCENES = ["noon_overview", "forest", "sunset_water", "water_closeup", "mountain_view"]
NIGHT_SCENES = ["night_torches", "torch_cave", "mineshaft"]


def main_checkout():
    try:
        out = subprocess.run(["git", "-C", ROOT, "worktree", "list", "--porcelain"], capture_output=True, text=True).stdout
        return out.splitlines()[0].split(" ", 1)[1]
    except Exception:
        return ROOT


class Session:
    """A lab session driven through the control file (docs/lab-mode.md)."""

    def __init__(self, log):
        self.log = log

    def results(self):
        try:
            with open(CTL_LOG, errors="replace") as f:
                return [l.rstrip("\n") for l in f if " -> " in l]
        except FileNotFoundError:
            return []

    def run(self, *cmds, n=None, timeout=600):
        """Appends the commands and waits for their results (n of them; default one each)."""
        before = len(self.results())
        with open(CTL, "a") as f:
            for c in cmds:
                f.write(c + "\n")
        n = len(cmds) if n is None else n
        end = time.time() + timeout
        while time.time() < end:
            r = self.results()
            if len(r) - before >= n:
                return r[before:before + n]
            time.sleep(0.2)
        raise TimeoutError(f"no result for {cmds} after {timeout} s")

    def wait_ready(self, since, timeout):
        end = time.time() + timeout
        while time.time() < end:
            try:
                with open(CTL_LOG, errors="replace") as f:
                    f.seek(since)
                    text = f.read()
                if " ready: " in text:
                    return
            except FileNotFoundError:
                pass
            time.sleep(1)
        raise TimeoutError("the lab session didn't get ready")


def read_defines(path):
    with open(path) as f:
        return f.read()


def write_atomic(path, text):
    tmp = path + ".autotune.tmp"
    with open(tmp, "w") as f:
        f.write(text)
    os.replace(tmp, path)


def fmt(v):
    return f"{v:.4f}"


class Knobs:
    """The #defines in the shader files: set them and wait for the reload."""

    def __init__(self, log):
        self.log = log
        self.current = {}
        self.bump = 0

    def apply(self, values, wait=True, timeout=60):
        """Writes the knobs that differ from the files' (the relight's header first, then post.metal) and waits for the
        reloads. Returns True if the light changed (lit_relight_header.metal was rewritten)."""
        changed = {k: v for k, v in values.items() if self.current.get(k) is None or abs(self.current[k] - v) > 1e-9}
        if not changed:
            return False
        mark = os.path.getsize(LAB_LOG) if os.path.exists(LAB_LOG) else 0
        lit = False
        for fname in ["lit_relight_header.metal", "post.metal"]:
            names = [k for k in changed if KNOBS[k][0] == fname]
            path = os.path.join(SHADERS, fname)
            old = read_defines(path)
            text = old
            for k in names:
                v = changed[k]
                text, n = re.subn(rf"^#define {k} .*$", f"#define {k} {'(' + fmt(v) + ')' if v < 0 else fmt(v)}", text, flags=re.M)
                if n != 1:
                    raise RuntimeError(f"{fname} has {n} '#define {k}' lines")
            if fname == "post.metal":
                # post.metal always reloads (a comment that changes every time): its reload snaps the eye adaptation to the
                # new light.
                text = re.sub(r"\n// autotune step \d+\n?$", "\n", text.rstrip("\n") + "\n")
                text = text.rstrip("\n") + f"\n// autotune step {time.time_ns()}\n"
            elif text == old:
                continue
            else:
                lit = True
            write_atomic(path, text)
        self.current.update(changed)
        if not wait:
            return lit
        want = {"post"} | ({"lit", "taa"} if lit else set())
        end = time.time() + timeout
        while time.time() < end:
            with open(LAB_LOG, errors="replace") as f:
                f.seek(mark)
                text = f.read()
            if "failed to compile" in text:
                raise RuntimeError("a shader failed to compile:\n" + text[-2000:])
            got = set()
            for line in text.splitlines():
                m = re.search(r"reloaded (.*?):", line)
                if m:
                    got |= {s.strip() for s in m.group(1).split(",")}
            if want <= got:
                return lit
            time.sleep(0.1)
        raise TimeoutError(f"no reload of {want} after {timeout} s")


def clamp(k, v):
    _, _, lo, hi, _, _ = KNOBS[k]
    return min(max(v, lo), hi)


def probe(k, center, step, sign):
    """The knob's value a step down (sign -1) or up from center: multiplicative for the log-scale knobs."""
    log = KNOBS[k][5]
    return clamp(k, center * math.exp(sign * step) if log else center + sign * step)


def next_center(center, cands, obj, knobs, steps):
    """The search's next center from a pass's objectives (obj: candidate name -> sum over the scenes; cands: name ->
    settings). Per knob, the minimum of the parabola through its points (in log space for the log-scale knobs), within a
    step of the center, damped by 0.8; with one probe (the other clamped away at a bound), toward it if it's better. A
    knob whose probes were all worse moves at most half a step and has its step shrunk (steps is updated)."""
    f0 = obj["center"]
    new = dict(center)
    for k in knobs:
        log_k = KNOBS[k][5]

        def u(v):
            return math.log(v) if log_k else v
        pts = [(u(center[k]), f0)]
        for tag in "-+":
            if f"{k}{tag}" in obj:
                pts.append((u(cands[f"{k}{tag}"][k]), obj[f"{k}{tag}"]))
        x0 = u(center[k])
        better = [pt for pt in pts[1:] if pt[1] < f0]
        if len(pts) == 3:
            (a, fa), (b, fb), (c, fc) = sorted(pts)
            den = (b - a) * (fb - fc) - (b - c) * (fb - fa)
            convex = (fc - fb) / (c - b) - (fb - fa) / (b - a) > 0
            if convex and abs(den) > 1e-12:
                target = b - 0.5 * ((b - a) ** 2 * (fb - fc) - (b - c) ** 2 * (fb - fa)) / den
            else:
                target = min(pts, key=lambda q: q[1])[0]
        elif better:
            target = better[0][0]
        else:
            target = x0
        span = steps[k] if better else 0.5 * steps[k]
        if not better:
            steps[k] *= 0.6
        elif abs(target - x0) > span:
            # The minimum looks farther than a step: move a step and take longer ones from here.
            steps[k] *= 1.5
        target = x0 + 0.8 * max(-span, min(span, target - x0))
        new[k] = clamp(k, math.exp(target) if log_k else target)
    return new


#: Class weights in the tuning (tools/twindiff.swift's --weights): the sky (SEUS's clouds are volumetric, ours aren't yet)
#: and water (the water helper is rebuilding it tonight) count a quarter, like the unlit class.
WEIGHTS = os.environ.get("AUTOTUNE_WEIGHTS", "sky=0.25,water=0.25,unlit=0.25")


def score_shot(ours, scene, mask, out_json):
    twin = os.path.join(SEUS, f"{PREFIX}-{scene}.png")
    subprocess.run([TWINDIFF, ours, twin, mask, "--name", scene, "--json", out_json, "--quiet", "--weights", WEIGHTS], check=True)
    with open(out_json) as f:
        j = json.load(f)
    if scene in twinsum.NIGHT:
        return twinsum.night_score(j)["points"], j
    return j["total"], j


def build_twindiff():
    src = os.path.join(ROOT, "tools", "twindiff.swift")
    if not os.path.exists(TWINDIFF) or os.path.getmtime(src) > os.path.getmtime(TWINDIFF):
        os.makedirs(os.path.dirname(TWINDIFF), exist_ok=True)
        subprocess.run(["swiftc", "-O", src, "-o", TWINDIFF], check=True)


def start_session(label, exp, minutes, log):
    """Starts lab.sh in the background (from the built-in shaders: the files are deleted first) and waits for ready."""
    if os.path.isdir(SHADERS):
        for name in os.listdir(SHADERS):
            if name.endswith(".metal") or name.endswith(".autotune.tmp"):
                os.remove(os.path.join(SHADERS, name))
    since = os.path.getsize(CTL_LOG) if os.path.exists(CTL_LOG) else 0
    env = dict(os.environ, LAB_EXP=exp, LAB_MINUTES=str(minutes))
    out = open(os.path.join(log.dir, "lab.out"), "a")
    proc = subprocess.Popen(["bash", os.path.join(ROOT, "tools", "bench", "lab.sh"), f"autotune-{label}"], cwd=ROOT, env=env,
                            stdout=out, stderr=subprocess.STDOUT)
    log(f"lab session autotune-{label} started (pid {proc.pid}), {exp}, {minutes} min; waiting for the GPU lock and the world")
    Session(log).wait_ready(since, timeout=4 * 3600)
    return proc


class Log:
    def __init__(self, d):
        self.dir = d
        os.makedirs(d, exist_ok=True)
        self.path = os.path.join(d, "progress.log")

    def __call__(self, s):
        line = time.strftime("%H:%M:%S ") + s
        print(line, flush=True)
        with open(self.path, "a") as f:
            f.write(line + "\n")


def wait_shader_files(timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        if all(os.path.exists(os.path.join(SHADERS, f)) for f in ("post.metal", "lit_relight_header.metal")):
            return
        time.sleep(1)
    raise TimeoutError("the shader files weren't written")


def tune(args, log):
    knobs = args.knobs
    scenes = args.scenes
    start = {k: KNOBS[k][1] for k in KNOBS}
    if args.start:
        with open(args.start) as f:
            start.update(json.load(f).get("knobs", {}))
    steps = {k: KNOBS[k][4] for k in knobs}
    t0 = time.time()
    deadline = t0 + args.minutes * 60
    sess = Session(log)
    kn = Knobs(log)
    pool = cf.ThreadPoolExecutor(max_workers=4)
    masks = {}
    evals = open(os.path.join(log.dir, "evals.jsonl"), "a")
    best = None   # (objective, settings, pass)
    center = dict(start)
    history = []
    shot_dir = os.path.join(log.dir, "shots")
    os.makedirs(shot_dir, exist_ok=True)
    # A throwaway scene first (a session's first scene caught the LOD mid-build).
    log("throwaway scene: " + "; ".join(sess.run(f"scene {scenes[0]}", "wait 300", timeout=900)))
    for p in range(args.passes):
        if time.time() > deadline - 60:
            log("out of time")
            break
        # The candidates: the center, then each knob a step down and up (post knobs first: cheaper reloads).
        cands = [("center", dict(center))]
        order = sorted(knobs, key=lambda k: KNOBS[k][0] != "post.metal")
        for k in order:
            for sign, tag in ((-1, "-"), (1, "+")):
                v = probe(k, center[k], steps[k], sign)
                if abs(v - center[k]) > 1e-9:
                    c = dict(center)
                    c[k] = v
                    cands.append((f"{k}{tag}", c))
        log(f"pass {p}: {len(cands)} candidates x {len(scenes)} scenes; center " +
            ", ".join(f"{k} {fmt(center[k])}" for k in knobs) + "; steps " + ", ".join(f"{k} {steps[k]:.3f}" for k in knobs))
        futures = {}
        for scene in scenes:
            if time.time() > deadline - 30:
                break
            # `scene` settles the LOD and the terrain, not the GI cache, the shadows' structures and the anti-aliasing's
            # history: after a long jump a shot needs about 400 ticks to converge (mountain_view after water_closeup,
            # lab 2026-10-10: 22.9 points at once, 20.1 after 100 ticks, 19.4 after 400 and 800).
            r = sess.run(f"scene {scene}", f"wait {args.settle}", timeout=900)
            if scene not in masks:
                mlabel = f"at-{args.label}-{scene}"
                mr = sess.run(f"mask {mlabel}", timeout=120)
                src = os.path.join(RUN, "screenshots", f"{mlabel}-mask.png")
                dst = os.path.join(log.dir, f"mask-{scene}.png")
                if os.path.exists(src):
                    os.replace(src, dst)
                    masks[scene] = dst
                log(f"  {scene}: {r[0].split(' -> ', 1)[1][:120]}; mask {mr[0].split(' -> ', 1)[1][:160]}")
            if scene not in masks:
                log(f"  {scene}: no mask, skipped")
                continue
            for name, c in cands:
                lit = kn.apply(c)
                sess.run(f"wait {40 if lit else 6}", timeout=60)
                label = f"at-{args.label}-p{p}-{scene}-{name}".replace("+", "p").replace("-", "_")
                sr = sess.run(f"shot {label}", timeout=120)
                src = os.path.join(RUN, "screenshots", f"{label}.png")
                if not os.path.exists(src):
                    log(f"  {scene} {name}: no shot ({sr})")
                    continue
                dst = os.path.join(shot_dir, f"{label}.png")
                os.replace(src, dst)
                futures[(scene, name)] = (pool.submit(score_shot, dst, scene, masks[scene], dst[:-4] + ".json"), dst)
        # Scores.
        table = {}
        for (scene, name), (fut, path) in futures.items():
            try:
                s, j = fut.result()
            except Exception as e:  # noqa: BLE001
                log(f"  {scene} {name}: scoring failed: {e}")
                continue
            table.setdefault(name, {})[scene] = s
            c = dict(cands)[name]
            evals.write(json.dumps({"pass": p, "scene": scene, "cand": name, "knobs": {k: c[k] for k in knobs}, "score": s,
                                    "total": j["total"], "regions": j["regions"]}) + "\n")
            evals.flush()
            # Keep the center's shots (boards later); the probes' go.
            if name != "center":
                os.remove(path)
        done = [s for s in scenes if all(s in table.get(n, {}) for n, _ in cands)]
        if len(done) < len(scenes):
            log(f"  pass {p} incomplete ({len(done)} of {len(scenes)} scenes scored): it doesn't count")
            break
        obj = {n: sum(table[n][s] for s in done) for n, _ in cands if n in table}
        f0 = obj.get("center")
        log(f"  pass {p} over {len(done)} scenes: center {f0:.2f}; " +
            ", ".join(f"{n} {v:.2f}" for n, v in sorted(obj.items(), key=lambda x: x[1])[:6]))
        history.append({"pass": p, "center": dict(center), "objective": f0, "scenes": done,
                         "per_scene": {s: table["center"][s] for s in done}})
        if best is None or f0 < best[0]:
            best = (f0, dict(center), p)
        elif f0 > best[0] + 0.05 * abs(best[0]):
            log(f"  the center ({f0:.2f}) is worse than the best ({best[0]:.2f}, pass {best[2]}): back to it, steps halved")
            center = dict(best[1])
            steps = {k: v * 0.5 for k, v in steps.items()}
            continue
        new = next_center(center, dict(cands), obj, knobs, steps)
        log("  next center: " + ", ".join(f"{k} {fmt(center[k])} -> {fmt(new[k])}" for k in knobs if abs(new[k] - center[k]) > 1e-6))
        center = new
        with open(os.path.join(log.dir, "best.json"), "w") as f:
            json.dump({"objective": best[0], "pass": best[2], "knobs": best[1], "scenes": scenes, "history": history}, f, indent=1)
    pool.shutdown(wait=True)
    if best:
        with open(os.path.join(log.dir, "best.json"), "w") as f:
            json.dump({"objective": best[0], "pass": best[2], "knobs": best[1], "scenes": scenes, "history": history}, f, indent=1)
        log(f"best: {best[0]:.2f} at pass {best[2]}: " + ", ".join(f"{k} {fmt(best[1][k])}" for k in knobs))
    return best


def compare(args, log):
    """Shots and boards of fixed settings (JSON files with "knobs"), each scene visited once."""
    sess = Session(log)
    kn = Knobs(log)
    configs = []
    for path in args.compare:
        with open(path) as f:
            d = json.load(f)
        c = {k: KNOBS[k][1] for k in KNOBS}
        c.update(d.get("knobs", {}))
        configs.append((os.path.splitext(os.path.basename(path))[0], c))
    log("throwaway scene: " + "; ".join(sess.run(f"scene {args.scenes[0]}", "wait 300", timeout=900)))
    pool = cf.ThreadPoolExecutor(max_workers=4)
    futs = []
    for scene in args.scenes:
        sess.run(f"scene {scene}", f"wait {args.settle}", timeout=900)
        mlabel = f"cmp-{args.label}-{scene}"
        sess.run(f"mask {mlabel}", timeout=120)
        mask = os.path.join(log.dir, f"mask-{scene}.png")
        os.replace(os.path.join(RUN, "screenshots", f"{mlabel}-mask.png"), mask)
        for name, c in configs:
            lit = kn.apply(c)
            sess.run(f"wait {40 if lit else 6}", timeout=60)
            label = f"cmp-{args.label}-{scene}-{name}"
            sess.run(f"shot {label}", timeout=120)
            dst = os.path.join(log.dir, f"{name}-{scene}.png")
            os.replace(os.path.join(RUN, "screenshots", f"{label}.png"), dst)
            twin = os.path.join(SEUS, f"{PREFIX}-{scene}.png")
            futs.append(pool.submit(subprocess.run, [TWINDIFF, dst, twin, mask, "--name", f"{scene} {name}", "--json",
                                                     dst[:-4] + ".json", "--board", os.path.join(log.dir, f"board-{name}-{scene}.jpg"),
                                                     "--quiet"], check=True))
    if args.trace:
        # Per-pass GPU times of 5 frames at the last scene with the last settings (tools/bench/passes.py on the game log).
        sess.run("trace 5", "wait 40", timeout=120)
        log("traced 5 frames at " + args.scenes[-1] + " (tools/bench/passes.py on the session's game log)")
    for f in futs:
        f.result()
    for name, _ in configs:
        tot = 0.0
        rows = []
        for scene in args.scenes:
            p = os.path.join(log.dir, f"{name}-{scene}.json")
            if not os.path.exists(p):
                continue
            with open(p) as fh:
                j = json.load(fh)
            s = twinsum.night_score(j)["points"] if scene in twinsum.NIGHT else j["total"]
            tot += s
            rows.append(f"{scene} {s:.2f} (terrain Y {j['regions']['terrain_y_ours']:.1f})")
        log(f"{name}: {tot:.2f}: " + "; ".join(rows))


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("label", nargs="?")
    ap.add_argument("--knobs", default=",".join(DAY_KNOBS))
    ap.add_argument("--scenes", default=",".join(DAY_SCENES))
    ap.add_argument("--start")
    ap.add_argument("--minutes", type=float, default=38)
    ap.add_argument("--settle", type=int, default=300, help="ticks to wait after each scene move (default 300)")
    ap.add_argument("--lab-minutes", type=int, help="the lab session's limit (default --minutes + 6)")
    ap.add_argument("--passes", type=int, default=50)
    ap.add_argument("--exp", default=FULL_LOOK)
    ap.add_argument("--out")
    ap.add_argument("--attach", action="store_true", help="drive a lab session that's already running")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--trace", action="store_true", help="compare: trace 5 frames' passes at the end")
    ap.add_argument("--no-quit", action="store_true", help="leave the session running at the end")
    ap.add_argument("--apply")
    a = ap.parse_args()
    if a.apply:
        with open(a.apply) as f:
            Knobs(print).apply(json.load(f)["knobs"], wait=False)
        return
    if not a.label:
        ap.error("a label")
    a.knobs = [k for k in a.knobs.split(",") if k]
    for k in a.knobs:
        if k not in KNOBS:
            ap.error(f"unknown knob {k} ({', '.join(KNOBS)})")
    a.scenes = [s for s in a.scenes.split(",") if s]
    out = a.out or os.path.join(main_checkout(), "bench_out", "agents", "grade", f"autotune-{a.label}")
    log = Log(out)
    build_twindiff()
    proc = None
    if not a.attach:
        proc = start_session(a.label, a.exp, a.lab_minutes or int(a.minutes) + 6, log)
    wait_shader_files()
    try:
        if a.compare:
            compare(a, log)
        else:
            tune(a, log)
    finally:
        if proc and not a.no_quit:
            try:
                Session(log).run("quit", timeout=60)
            except Exception:  # noqa: BLE001
                pass
            proc.wait(timeout=600)
            log("session over")


if __name__ == "__main__":
    main()
