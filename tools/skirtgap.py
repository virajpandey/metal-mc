"""Underwater node-edge skirts against the step they have to close, on the merged test world's ocean regions.
For a pair of adjacent nodes A | B (the same level, or A one level coarser than B), takes A's skirt quads on the shared
edge that are under water (depth code > 0, or water), and per column along the edge compares what a camera above the sea
can see of them (the part above B's floor) with the step between the two floors (A's floor top - B's, when positive).
Where a flooded cave under a ledge meets the border, the step stays open: the cave's mouth, as inside a node.
usage: python3 tools/skirtgap.py <libMetalMCNative.dylib>   (METALMC_EXP=oldskirts for the old skirts; VERBOSE=1 lists
the columns whose step is left open)"""
import ctypes, os, sys
lib = ctypes.CDLL(sys.argv[1])
lib.mmc_debug_region_node_mesh.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
                                           ctypes.POINTER(ctypes.c_uint32), ctypes.c_int32, ctypes.POINTER(ctypes.c_int16), ctypes.c_void_p]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Fixtures aren't in git worktrees: fall back to the main checkout's.
FIX = next(p for p in [os.path.join(ROOT, "fixtures"), "/Users/rachnap/Projects/metal-mc/fixtures"] if os.path.isdir(p))
REG = os.path.join(FIX, "claudeworld-merged/dimensions/minecraft/overworld/region/r.%d.%d.mca")
CAP = 3_000_000
VERBOSE = os.environ.get("VERBOSE") is not None
NONE = -32768
WATER_MATS = set(range(128, 160)) | {5}

def node(rx, rz, level, qx=0, qz=0):
    """A node built from a region as LodWorld builds it: level 0 is quarter (qx, qz), level 1 the region."""
    q = (ctypes.c_uint32 * (2 * CAP))()
    c = (ctypes.c_int16 * (2 * 256 * 256))()
    n = lib.mmc_debug_region_node_mesh((REG % (rx, rz)).encode(), level, qx, qz, q, CAP, c, None)
    if n < 0: return None
    quads = [(q[2 * i], q[2 * i + 1]) for i in range(min(n, CAP))]
    return dict(level=level, quads=quads, floor=[c[2 * i] for i in range(65536)], water=[c[2 * i + 1] for i in range(65536)])

def edge_cover(nd, side):
    """Underwater skirt coverage on one edge ('+x', '-x', '+z', '-z'): {j: [(lo, hi)]} in world blocks, j the voxel index
    along the edge; and the number of quads."""
    face = {'+x': 0, '-x': 1, '+z': 4, '-z': 5}[side]
    s = 1 << nd['level']
    cov, nq = {}, 0
    for w0, w1 in nd['quads']:
        f = (w0 >> 25) & 7
        if f != face: continue
        x, z, y = w0 & 255, (w0 >> 8) & 255, (w0 >> 16) & 511
        if side in ('+x', '-x') and x != (255 if side == '+x' else 0): continue
        if side in ('+z', '-z') and z != (255 if side == '+z' else 0): continue
        if w0 >> 28 == 0 and (w1 & 255) not in WATER_MATS: continue
        qw, qh = ((w1 >> 8) & 255) + 1, ((w1 >> 16) & 255) + 1
        if f < 2: y0, y1, j0, j1 = y, y + qw, z, z + qh    # X faces: w along y, h along z
        else: y0, y1, j0, j1 = y, y + qh, x, x + qw        # Z faces: w along x, h along y
        nq += 1
        for j in range(j0, j1): cov.setdefault(j, []).append((y0 * s - 64, y1 * s - 64))
    return cov, nq

def column(side, j):
    """Index of the edge column at voxel j along `side`."""
    return {'+x': j * 256 + 255, '-x': j * 256, '+z': 255 * 256 + j, '-z': j}[side]

def compare(name, a, aside, b, bside, boffset=0):
    """A's skirt on `aside` against B's edge columns on `bside`. If B is one level finer, A's voxel j meets B's columns
    2j - boffset and 2j + 1 - boffset (boffset 256 for the second of B's two quarters along the edge)."""
    cov, nq = edge_cover(a, aside)
    ratio = 1 << (a['level'] - b['level'])
    cols = seen = step = over = under = 0
    for j in range(256):
        fa = a['floor'][column(aside, j)]
        if a['water'][column(aside, j)] == NONE or fa == NONE: continue
        for k in range(ratio):
            jb = j * ratio + k - boffset
            if jb < 0 or jb >= 256: continue
            fb = b['floor'][column(bside, jb)]
            if fb == NONE: continue
            cols += 1
            ivs = cov.get(j, [])
            v = sum(max(0, hi - max(lo, fb)) for lo, hi in ivs)   # the part above B's floor
            g = max(0, fa - fb)
            if g > v and VERBOSE:
                print(f"   step left open at {j}/{k}: A floor {fa}, B floor {fb}, skirt {sorted(ivs)}")
            seen += v; step += g
            over += max(0, v - g); under += max(0, g - v)
    print(f"{name}: {nq} underwater skirt quads over {cols} columns; seen above the neighbor's floor {seen} block-columns, "
          f"step {step} (beyond the step {over}, step left open {under})")

if __name__ == "__main__":
    # Level 0: quarters of ocean regions, x- and z-adjacent, both ways.
    q00, q10, q01 = node(-2, -1, 0, 0, 0), node(-2, -1, 0, 1, 0), node(-2, -1, 0, 0, 1)
    compare("L0 r.-2.-1 q(0,0) +x | q(1,0)", q00, '+x', q10, '-x')
    compare("L0 r.-2.-1 q(1,0) -x | q(0,0)", q10, '-x', q00, '+x')
    compare("L0 r.-2.-1 q(0,0) +z | q(0,1)", q00, '+z', q01, '-z')
    compare("L0 r.-2.-1 q(0,1) -z | q(0,0)", q01, '-z', q00, '+z')
    r30, r31 = node(-3, 0, 0, 1, 0), node(-3, 0, 0, 1, 1)
    compare("L0 r.-3.0 q(1,0) +z | q(1,1)", r30, '+z', r31, '-z')
    compare("L0 r.-3.0 q(1,1) -z | q(1,0)", r31, '-z', r30, '+z')
    # Level 1: adjacent ocean regions.
    a, b = node(-2, -2, 1), node(-2, -1, 1)
    compare("L1 r.-2.-2 +z | r.-2.-1", a, '+z', b, '-z')
    compare("L1 r.-2.-1 -z | r.-2.-2", b, '-z', a, '+z')
    c, d = node(-3, 0, 1), node(-2, 0, 1)
    compare("L1 r.-3.0 +x | r.-2.0", c, '+x', d, '-x')
    compare("L1 r.-2.0 -x | r.-3.0", d, '-x', c, '+x')
    # Level 1 next to level 0: r.-2.-2 at level 1 over the two level-0 quarters of r.-2.-1 along its north edge.
    compare("L1 r.-2.-2 +z | L0 r.-2.-1 q(0,0)", a, '+z', q00, '-z', 0)
    compare("L1 r.-2.-2 +z | L0 r.-2.-1 q(1,0)", a, '+z', q10, '-z', 256)
