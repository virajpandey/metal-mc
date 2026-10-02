#!/usr/bin/env python3
"""usage: passes.py <run log> [<run log> ...]

Per-pass GPU times from a traced flight (-PbenchTrace=1): every "profile submit" block the backend logged, each pass
keyed by its label and how many passes with that label came before it in the submit ("#2": the second), then the median
over the traced submits, in frame order. Passes overlap on the GPU (one pass's vertex work runs under the previous one's
fragment work), so the passes can add up to more than the command buffer's span. Render passes show their vertex and
fragment stages too; -1 means the GPU gave no timestamp (an empty stage)."""
import re
import statistics
import sys

LINE = re.compile(r'^\s+total\s+(-?[\d.]+) ms(?:\s+vertex\s+(-?[\d.]+)\s+fragment\s+(-?[\d.]+))?\s+(.*?)\s*$')
HEAD = re.compile(r'profile submit (\d+): cb gpu ([\d.]+) ms')


def submits(path):
    out, cur = [], None
    for line in open(path, errors='ignore'):
        m = HEAD.search(line)
        if m:
            cur = {'gpu': float(m.group(2)), 'passes': [], 'seen': {}}
            out.append(cur)
            continue
        m = LINE.match(line)
        if m and cur is not None:
            label = m.group(4)
            k = cur['seen'].get(label, 0) + 1
            cur['seen'][label] = k
            stages = (float(m.group(2)), float(m.group(3))) if m.group(2) is not None else None
            cur['passes'].append((label if k == 1 else f'{label} #{k}', float(m.group(1)), stages))
        elif cur is not None and not line.startswith('  '):
            cur = None
    return out


def med(v):
    v = [x for x in v if x >= 0]
    return statistics.median(v) if v else -1.0


for path in sys.argv[1:]:
    subs = [s for s in submits(path) if s['passes']]
    if not subs:
        print(f'{path}: no traced submits')
        continue
    order, total, vert, frag = [], {}, {}, {}
    for s in subs:
        for key, t, st in s['passes']:
            if key not in total:
                order.append(key)
            total.setdefault(key, []).append(t)
            if st:
                vert.setdefault(key, []).append(st[0])
                frag.setdefault(key, []).append(st[1])
    print(f'{path}: {len(subs)} traced submits, command buffer median {statistics.median(s["gpu"] for s in subs):.3f} ms')
    for key in order:
        stages = f'  vertex {med(vert[key]):6.3f}  fragment {med(frag[key]):6.3f}' if key in vert else ' ' * 32
        print(f'  {med(total[key]):7.3f} ms{stages}  (in {len(total[key]):2d})  {key}')
