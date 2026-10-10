#!/usr/bin/env python3
"""The summary of a twin census (tools/bench/twins.sh; docs/lab-mode.md, "Twins").

usage: twinsum.py <twins dir> [--before <twins dir>] [--json]

Reads the score-<scene>.json files tools/twindiff.swift wrote and prints a markdown summary: each scene's score (lower is
better) and the classes that carry it, the day total (the scenes tuned against SEUS), and night and caves against the
Claude edition's own target (terrain luma 35-45 with warm torchlight, not SEUS's near-black). --before adds the change
from an earlier census. --json prints the numbers instead (the autotune reads them through this module).
"""
import glob
import json
import math
import os
import sys

CLASSES = ["sky", "water", "vegetation", "sunlit", "shade", "blocklit", "far", "unlit"]
# Tuned against SEUS: the day scenes with valid twins.
DAY = ["noon_overview", "forest", "mountain_view", "sunset_water", "water_closeup"]
# Twins that can't be compared (census 1): SEUS drew no rain.
INVALID = {"rain": "SEUS drew no rain"}
# Ours to style: night and caves, against a target between ours and SEUS's.
NIGHT = ["night_torches", "torch_cave", "mineshaft"]
NIGHT_LUMA = (35.0, 45.0)


def load(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "score-*.json"))):
        with open(f) as fh:
            j = json.load(fh)
        out[j.get("name") or os.path.basename(f)[len("score-"):-len(".json")]] = j
    return out


def cls(j, name):
    for c in j["classes"]:
        if c["class"] == name:
            return c
    return None


def night_score(j):
    """Night and caves (points, lower is better): the terrain's mean luma (bottom 55%, 8-bit) outside 35-45, half a
    point per level; the torchlight's color against SEUS's warm pools (the block-lit class's OKLab a, b and chroma, not
    its lightness), 100 x share x distance; and crushed shadows (the shade and block-lit classes' 10th percentile of OKLab
    L under 0.12), 100 x share x shortfall."""
    y = j["regions"]["terrain_y_ours"]
    lo, hi = NIGHT_LUMA
    luma = 0.5 * max(0.0, lo - y, y - hi)
    warm = 0.0
    c = cls(j, "blocklit")
    if c and c.get("delta"):
        d = c["delta"]
        warm = 100 * c["share"] * math.sqrt(d["a"] ** 2 + d["b"] ** 2 + d["C"] ** 2)
    dark = 0.0
    for name in ("shade", "blocklit"):
        c = cls(j, name)
        if c and c["share"] > 0:
            dark += 100 * c["share"] * max(0.0, 0.12 - c["ours"]["p10"])
    return {"points": luma + warm + dark, "luma": luma, "warm": warm, "dark": dark, "terrain_y": y}


def summarize(d):
    scores = load(d)
    day = sum(scores[s]["total"] for s in DAY if s in scores)
    night = {s: night_score(scores[s]) for s in NIGHT if s in scores}
    return {"dir": d, "scenes": scores, "day": day, "day_scenes": [s for s in DAY if s in scores],
            "night": night, "night_total": sum(v["points"] for v in night.values())}


def main():
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        sys.exit(2)
    before = None
    if "--before" in args:
        i = args.index("--before")
        before = summarize(args[i + 1])
        del args[i:i + 2]
    as_json = "--json" in args
    args = [a for a in args if a != "--json"]
    s = summarize(args[0])
    if as_json:
        print(json.dumps({k: v for k, v in s.items() if k != "scenes"} | {
            "totals": {k: v["total"] for k, v in s["scenes"].items()}}, indent=1, sort_keys=True))
        return
    sc = s["scenes"]
    print(f"# Twin census: {os.path.basename(os.path.normpath(args[0]))}\n")
    print("Scores from tools/twindiff.swift (lower is better; 100 x the share-weighted class distances, "
          "docs/lighting-design.md \"The grade\"). Luma as ffmpeg's signalstats has it (YAVG / YLOW / SATAVG: BT.601, "
          "limited range), terrain = bottom 55%, sky = top 20%.\n")
    head = "| scene | score |" + (" before | change |" if before else "") + \
           " terrain Y / p10 / sat, ours | SEUS | sky Y ours / SEUS | biggest classes (points) |"
    print(head)
    print("|" + "---|" * (head.count("|") - 1))
    order = DAY + [k for k in INVALID] + NIGHT
    order += [k for k in sc if k not in order]
    for name in order:
        if name not in sc:
            continue
        j = sc[name]
        tops = sorted((c for c in j["classes"] if c.get("contribution") is not None), key=lambda c: -c["contribution"])[:3]
        top = ", ".join(f"{c['class']} {c['contribution']:.1f}" for c in tops)
        r = j["regions"]
        note = f" ({INVALID[name]})" if name in INVALID else ""
        row = f"| {name}{note} | {j['total']:.2f} |"
        if before:
            b = before["scenes"].get(name)
            row += f" {b['total']:.2f} | {j['total'] - b['total']:+.2f} |" if b else " | |"
        row += (f" {r['terrain_y_ours']:.1f} / {r.get('terrain_p10_ours', 0):.0f} / {r.get('terrain_sat_ours', 0):.1f} |"
                f" {r['terrain_y_twin']:.1f} / {r.get('terrain_p10_twin', 0):.0f} / {r.get('terrain_sat_twin', 0):.1f} |"
                f" {r['sky_y_ours']:.1f} / {r['sky_y_twin']:.1f} | {top} |")
        print(row)
    print()
    line = f"**Day total** ({', '.join(s['day_scenes'])}): **{s['day']:.2f}**"
    if before:
        line += f" (before {before['day']:.2f}, {s['day'] - before['day']:+.2f})"
    print(line + "\n")
    if s["night"]:
        print(f"**Night and caves** against the Claude edition's target (terrain luma {NIGHT_LUMA[0]:.0f}-{NIGHT_LUMA[1]:.0f}, "
              "warm torch pools, no crushed shadows): points = luma + warmth + dark, lower is better.\n")
        print("| scene | terrain Y | luma | warmth | dark | points |" + (" before |" if before else ""))
        print("|---|---|---|---|---|---|" + ("---|" if before else ""))
        for name, v in s["night"].items():
            row = f"| {name} | {v['terrain_y']:.1f} | {v['luma']:.2f} | {v['warm']:.2f} | {v['dark']:.2f} | {v['points']:.2f} |"
            if before and name in before["night"]:
                row += f" {before['night'][name]['points']:.2f} |"
            print(row)
        print()
    # Points by class and scene: where the gap is.
    names = [n for n in order if n in sc]
    print("Points by class (rows) and scene (columns):\n")
    print("| class | " + " | ".join(names) + " | day |")
    print("|---|" + "---|" * (len(names) + 1))
    for c in CLASSES:
        vals = []
        dsum = 0.0
        for n in names:
            x = cls(sc[n], c)
            v = x.get("contribution") if x else None
            vals.append(f"{v:.1f}" if v is not None else "")
            if v is not None and n in DAY:
                dsum += v
        print(f"| {c} | " + " | ".join(vals) + f" | {dsum:.1f} |")
    print()
    # Mean lightness and chroma of the classes that matter most, ours against SEUS, day scenes.
    print("Class means, ours / SEUS (OKLab L and chroma; day scenes):\n")
    print("| scene | " + " | ".join(f"{c} L" for c in ("vegetation", "sunlit", "shade", "far", "water", "sky")) + " | vegetation C | sunlit C |")
    print("|---|" + "---|" * 8)
    for n in DAY:
        if n not in sc:
            continue
        cells = []
        for c in ("vegetation", "sunlit", "shade", "far", "water", "sky"):
            x = cls(sc[n], c)
            cells.append(f"{x['ours']['L']:.2f} / {x['twin']['L']:.2f}" if x and x["share"] > 0.002 else "")
        for c in ("vegetation", "sunlit"):
            x = cls(sc[n], c)
            cells.append(f"{x['ours']['C']:.3f} / {x['twin']['C']:.3f}" if x and x["share"] > 0.002 else "")
        print(f"| {n} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
