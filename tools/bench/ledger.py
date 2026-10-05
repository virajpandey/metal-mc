"""Appends one JSON line per game run to the run ledger: bench_out/ledger.jsonl in the main checkout (whichever worktree
ran it), opened for appending only, never truncated.

usage (bench_lod.sh calls it after every run, fidelity.sh and lab.sh through it):
  python3 tools/bench/ledger.py <label> <run log> <fixture> <start, epoch seconds> <lodFar or 0> [gradle args...]
Environment: LEDGER_KIND (bench, or fidelity / lab from those scripts), BENCH_RD, BENCH_VSYNC, BENCH_TIMEOUT.
  python3 tools/bench/ledger.py --show [N]     the last N runs (20), one line each

A line: time, kind, label, git commit/branch/dirty and worktree, the args and the switches they set, fixture, render
distance, vsync, the summary numbers (every key=value of the METALMC_BENCH line), screenshots, and validity flags:
  screen_locked      the lock screen was up during the run (METALMC_LOCKED lines from run_mc.sh's watcher)
  fps_pinned_120     uncapped (no vsync) and fps_mean within 0.2 of 120.0: something paced it at the display's rate
  lod_still_building timing started before the LOD's first build finished (tools/bench/lodready.py)
  windowed           not fullscreen (the game couldn't take the display, or the window isn't the panel's 3456 x 2234)
  dialogs            system dialogs seen over the game (METALMC_DIALOG lines)
  valid              a timed run: none of the above; a run for screenshots (tour, lab): not windowed
"""
import json, os, re, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))


def git(*args):
    try:
        return subprocess.run(["git", "-C", ROOT] + list(args), capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def ledger_path():
    main = ""
    for line in git("worktree", "list", "--porcelain").splitlines():
        if line.startswith("worktree "):
            main = line[len("worktree "):]
            break
    return os.path.join(main or ROOT, "bench_out", "ledger.jsonl")


def number(v):
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v.strip('"')


def summary(log):
    """Every key=value of the run's METALMC_BENCH line (quoted values may hold spaces)."""
    for line in log.splitlines():
        if "METALMC_BENCH label=" in line:
            body = line[line.index("METALMC_BENCH") + len("METALMC_BENCH"):]
            return {k: number(v) for k, v in re.findall(r'(\w+)=("[^"]*"|\S+)', body)}
    return {}


def switches(args):
    """The -Pname=value gradle args as a dict."""
    out = {}
    for a in args:
        m = re.match(r"-P(\w+)(?:=(.*))?$", a)
        if m:
            out[m.group(1)] = number(m.group(2)) if m.group(2) is not None else True
    return out


def record(label, log_path, fixture, start, far, args):
    try:
        log = open(log_path, errors="replace").read()
    except OSError:
        log = ""
    s = summary(log)
    sw = switches(args)
    kind = os.environ.get("LEDGER_KIND") or ("fly" if "benchFly" in sw else "tour" if "benchTour" in sw else "bench")
    vsync = os.environ.get("BENCH_VSYNC", "false") == "true"
    shots_dir = os.path.join(ROOT, "mod", "run", "screenshots")
    try:
        shots = sum(1 for f in os.listdir(shots_dir) if f.startswith(label + "-") and os.path.getmtime(os.path.join(shots_dir, f)) >= start)
    except OSError:
        shots = 0
    lodready = ""
    if "benchHitches" in sw:
        try:
            lodready = subprocess.run([sys.executable, os.path.join(HERE, "lodready.py"), log_path], capture_output=True,
                                      text=True, timeout=30).stdout.strip()
        except Exception:
            pass
    fps = s.get("fps_mean")
    loaded = re.search(r"world loaded.*fullscreen=(\w+) window=(\d+)x(\d+)", log)
    full = s.get("fullscreen", loaded.group(1) if loaded else None)
    window = s.get("window", loaded.group(2) + "x" + loaded.group(3) if loaded else None)
    flags = {
        "screen_locked": "METALMC_LOCKED" in log,
        "fps_pinned_120": (not vsync) and isinstance(fps, (int, float)) and abs(fps - 120.0) <= 0.2,
        "lod_still_building": "LOD STILL BUILDING" in lodready,
        "windowed": "Couldn't enter fullscreen" in log or str(full).lower() == "false"
                    or (window is not None and str(window) != "3456x2234"),
        "dialogs": len(re.findall(r"^METALMC_DIALOG", log, re.M)),
    }
    exit_m = re.findall(r"^exit (\d+)", log, re.M)
    entry = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(start)) if start else None,
        "seconds": round(time.time() - start) if start else None,
        "kind": kind,
        "label": label,
        "commit": git("rev-parse", "--short", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "worktree": ROOT,
        "fixture": fixture,
        "lod_far": number(far),
        "render_distance": number(os.environ.get("BENCH_RD", "12")),
        "vsync": vsync,
        "exp": sw.get("metalExp", ""),
        "args": args,
        "summary": s,
        "lod_ready": lodready or None,
        "screenshots": shots,
        "window": window,
        "exit": int(exit_m[-1]) if exit_m else None,
        "errors": len(re.findall(r"Exception|GPU error", log)),
        "log": log_path,
    }
    entry.update(flags)
    # A timed run (one with a summary) needs all of them clear; screenshots (tours, lab) only need the panel's resolution:
    # they're read from the render target, which a lock screen or a dialog over the game doesn't touch.
    timed = bool(s)
    bad = [k for k in (("screen_locked", "fps_pinned_120", "lod_still_building", "windowed", "dialogs") if timed else ("windowed",))
           if flags[k]]
    entry["valid"] = not bad
    path = ledger_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:   # append only: the ledger is never rewritten
        f.write(json.dumps(entry) + "\n")
    print("ledger: %s %s%s -> %s" % (kind, label, " (NOT VALID: " + ", ".join(bad) + ")" if bad else "", path))


def show(n):
    path = ledger_path()
    try:
        lines = open(path).read().splitlines()[-n:]
    except OSError:
        print("no ledger at " + path)
        return
    for line in lines:
        try:
            e = json.loads(line)
        except ValueError:
            continue
        s = e.get("summary", {})
        nums = " ".join("%s=%s" % (k, s[k]) for k in ("fps_mean", "ms_p99", "frames_over_8ms") if k in s)
        bad = [k for k in ("screen_locked", "fps_pinned_120", "lod_still_building", "windowed") if e.get(k)]
        print("%s %-8s %-24s %s%s %s %s %s" % (e.get("time", "")[:19], e.get("kind", ""), e.get("label", ""), e.get("commit", ""),
              "+" if e.get("dirty") else "", e.get("branch", ""), nums, "VALID" if e.get("valid") else "not valid: " + ",".join(bad)))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--show":
        show(int(sys.argv[2]) if len(sys.argv) > 2 else 20)
    elif len(sys.argv) < 6:
        print(__doc__)
        sys.exit(1)
    else:
        record(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4] or 0), sys.argv[5], sys.argv[6:])
