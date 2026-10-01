"""Offline checks for near chunks (Sources/MetalMCNative/NearChunks.swift, NearChunksCheck.swift) through libMetalMCNative.
Nothing here starts Minecraft.

usage: python3 tools/neartest.py [world dir] [out dir]
  world dir: a save whose overworld regions feed the real-terrain checks (default: fixtures/claudeworld-merged)
  out dir:   where the render comparison PNGs go (default: .build/neartest)
  NEARTEST_LIB=<path> picks the dylib (default: .build/release/libMetalMCNative.dylib next to this script's repo).

Checks:
  1. the arena's slab allocator against its invariants (random allocations and frees);
  2. the codec round trip on vanilla-like vertices synthesized from real chunks (cube faces, ambient occlusion, tints,
     smooth light, UV rotations, the grass overlay) and on hand-made non-cube shapes (slabs, torches, offset cross
     plants, lava with 0.001 insets and flow-rotated UVs, a rotated element, tinted partial faces): position and UV error,
     colors and light exact, bytes per face against vanilla's 112, encode time;
  3. the shader's decoder (compute) against the CPU decoder, value for value;
  4. layers the codec must refuse (translucent alpha, light over 255, NaN, positions out of range);
  5. one real section and the shapes drawn through the near-chunk pipeline and through a transcription of vanilla's
     terrain vertex shader reading the original vertices: pixel differences, with PNGs to look at; then again through
     the game's own draw path (the arena, mmc_pass_begin, mmc_near_draw, mmc_submit), which also checks that a freed
     arena range isn't reused before the GPU finishes the submit that drew it.
With METALMC_EXP=lit the game's draw path runs in lit mode (Lit.swift): its pass gets the G-buffer, and each arena render
also checks what the near chunks wrote there (marked pixels, faces, the depth key, and the relight's overlay test on the
image as drawn). Check 5's vanilla comparison then allows small differences: lit mode samples the lightmap per pixel.
"""
import ctypes
import glob
import math
import os
import random
import struct
import sys
import time
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIB = os.environ.get("NEARTEST_LIB", os.path.join(REPO, ".build/release/libMetalMCNative.dylib"))
WORLD = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "fixtures/claudeworld-merged")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(REPO, ".build/neartest")

lib = ctypes.CDLL(LIB)
P = ctypes.c_void_p
lib.mmc_near_debug_synth_chunk.argtypes = [ctypes.c_char_p, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, P, ctypes.c_int32, P]
lib.mmc_near_debug_synth_chunk.restype = ctypes.c_int32
lib.mmc_near_debug_encode.argtypes = [P, ctypes.c_int32, P, ctypes.c_int32, P]
lib.mmc_near_debug_encode.restype = ctypes.c_int32
lib.mmc_near_debug_decode_cpu.argtypes = [P, ctypes.c_int32, P]
lib.mmc_near_debug_decode_gpu.argtypes = [P, ctypes.c_int32, ctypes.c_int32, P]
lib.mmc_near_debug_decode_gpu.restype = ctypes.c_int32
lib.mmc_near_debug_arena_check.argtypes = [ctypes.c_int32, ctypes.c_int32, ctypes.c_uint64, P]
lib.mmc_near_debug_arena_check.restype = ctypes.c_int32
lib.mmc_near_debug_render.argtypes = [P, ctypes.c_int32, ctypes.c_int32, P, P, ctypes.c_float, ctypes.c_float, P,
                                      ctypes.c_int32, ctypes.c_int32, P, P, P]
lib.mmc_near_debug_render.restype = ctypes.c_int32
lib.mmc_near_debug_render_arena.argtypes = [P, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, P, P, ctypes.c_float, ctypes.c_float, P,
                                            ctypes.c_int32, ctypes.c_int32, P, P, P]
lib.mmc_near_debug_render_arena.restype = ctypes.c_int32
# Lit mode (METALMC_EXP=lit): the arena render also checks the G-buffer the near chunks write (mmc_near_debug_lit_stats).
LIT = "lit" in os.environ.get("METALMC_EXP", "").split(",")
if LIT:
    lib.mmc_near_debug_lit_stats.argtypes = [P]

failures = []


def check(ok, what):
    if not ok:
        failures.append(what)
        print("  FAIL:", what)


# ---------------------------------------------------------------- vertices

def vertex(p, rgb, uv, light, alpha=255):
    return struct.pack("<3f4B2f2h", p[0], p[1], p[2], rgb[0], rgb[1], rgb[2], alpha, uv[0], uv[1], light[0], light[1])


def unpack_vertices(buf, n):
    out = []
    for i in range(n):
        x, y, z, r, g, b, a, u, v, bl, sk = struct.unpack_from("<3f4B2f2h", buf, 28 * i)
        out.append((x, y, z, u, v, r, g, b, bl, sk))
    return out


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def tinted(tint, g):
    return tuple(c * g // 255 for c in tint)


GRASS = (0x91, 0xBD, 0x59)
ATLAS = 1024.0


def sprite_uv(col, row, fu, fv, size=16):
    """Vanilla's sprite.getU/getV in float: u0 + (u1 - u0) * f."""
    u0, u1 = f32(col * 16 / ATLAS), f32((col * 16 + size) / ATLAS)
    v0, v1 = f32(row * 16 / ATLAS), f32((row * 16 + size) / ATLAS)
    return (f32(u0 + f32(u1 - u0) * f32(fu)), f32(v0 + f32(v1 - v0) * f32(fv)))


def shapes():
    """Hand-made non-cube quads in vanilla's vertex order, one section's worth."""
    rnd = random.Random(7)
    q = []

    def add(pts, rgbs, uvs, lights):
        q.append(b"".join(vertex(pts[k], rgbs[k], uvs[k], lights[k]) for k in range(4)))

    def light4():
        return [(rnd.choice([0, 12, 60, 240]), rnd.randrange(0, 61) * 4) for _ in range(4)]

    def grays(base):
        return [max(0, min(255, base - rnd.randrange(0, 60))) for _ in range(4)]

    # Slab (bottom half) at (2, 3, 4): top at y + 0.5 with weighted ambient occlusion, sides with the lower half of the sprite.
    x, y, z = 2, 3, 4
    g = grays(255)
    add([(x, y + .5, z), (x, y + .5, z + 1), (x + 1, y + .5, z + 1), (x + 1, y + .5, z)], [(v, v, v) for v in g],
        [sprite_uv(3, 1, 0, 0), sprite_uv(3, 1, 0, 1), sprite_uv(3, 1, 1, 1), sprite_uv(3, 1, 1, 0)], light4())
    g = grays(204)
    add([(x + 1, y + .5, z), (x + 1, y, z), (x, y, z), (x, y + .5, z)], [(v, v, v) for v in g],
        [sprite_uv(3, 1, 0, .5), sprite_uv(3, 1, 0, 1), sprite_uv(3, 1, 1, 1), sprite_uv(3, 1, 1, .5)], light4())
    # Stairs step at (3, 3, 4): upper half box z 0.5..1.
    add([(3, 4, 4.5), (3, 4, 5), (4, 4, 5), (4, 4, 4.5)], [(v, v, v) for v in grays(255)],
        [sprite_uv(3, 1, 0, .5), sprite_uv(3, 1, 0, 1), sprite_uv(3, 1, 1, 1), sprite_uv(3, 1, 1, .5)], light4())
    # Torch at (5, 2, 5): 7/16..9/16 wide, 10/16 tall; flat light at full block light (emissive).
    lo, hi, top = 7 / 16, 9 / 16, 10 / 16
    for (a, b) in [((5 + lo, 2 + top, 5 + lo), (5 + hi, 2, 5 + lo)), ((5 + hi, 2 + top, 5 + hi), (5 + lo, 2, 5 + hi))]:
        add([(a[0], a[1], a[2]), (a[0], b[1], a[2]), (b[0], b[1], b[2]), (b[0], a[1], b[2])], [(204, 204, 204)] * 4,
            [sprite_uv(5, 0, 7 / 16, 6 / 16), sprite_uv(5, 0, 7 / 16, 1), sprite_uv(5, 0, 9 / 16, 1), sprite_uv(5, 0, 9 / 16, 6 / 16)],
            [(240, 240)] * 4)
    add([(5 + lo, 2 + top, 5 + lo), (5 + lo, 2 + top, 5 + hi), (5 + hi, 2 + top, 5 + hi), (5 + hi, 2 + top, 5 + lo)], [(255, 255, 255)] * 4,
        [sprite_uv(5, 0, 7 / 16, 6 / 16), sprite_uv(5, 0, 7 / 16, 8 / 16), sprite_uv(5, 0, 9 / 16, 8 / 16), sprite_uv(5, 0, 9 / 16, 6 / 16)],
        [(240, 240)] * 4)
    # Short grass (cross model) at a few blocks with vanilla's random XZ(+Y) offset; each plane double-sided.
    for (bx, by, bz, seed) in [(6, 4, 7, 11), (7, 4, 7, 12345), (8, 4, 2, 999)]:
        h = seed * 3129871 ^ bz * 116129781 ^ by
        h = (h * h * 42317861 + h * 11) & 0xFFFFFFFFFFFF
        dx = f32(((h >> 16 & 15) / 15.0 - 0.5) * 0.5)
        dz = f32(((h >> 24 & 15) / 15.0 - 0.5) * 0.5)
        dy = f32(((h >> 20 & 15) / 15.0 - 1.0) * 0.2)
        a, b = f32(0.5 - 0.45 * math.sqrt(0.5)), f32(0.5 + 0.45 * math.sqrt(0.5))
        tint = tinted(GRASS, 229)
        for (p0, p1) in [((a, a), (b, b)), ((a, b), (b, a))]:
            pts = [(bx + dx + p0[0], by + dy + 1, bz + dz + p0[1]), (bx + dx + p0[0], by + dy, bz + dz + p0[1]),
                   (bx + dx + p1[0], by + dy, bz + dz + p1[1]), (bx + dx + p1[0], by + dy + 1, bz + dz + p1[1])]
            uvs = [sprite_uv(9, 2, 0, 0), sprite_uv(9, 2, 0, 1), sprite_uv(9, 2, 1, 1), sprite_uv(9, 2, 1, 0)]
            lt = light4()
            add(pts, [tint] * 4, uvs, lt)
            add(pts[::-1], [tint] * 4, uvs[::-1], lt[::-1])
    # Lava at (9, 1, 9): sloped top with 0.001 insets and flow-rotated UVs in a 32 x 32 sprite; sides inset 0.001.
    h = [f32(f32(8 / 9) - f32(0.001)), f32(f32(0.7777778) - f32(0.001)), f32(f32(0.6666667) - f32(0.001)), f32(f32(0.8333333) - f32(0.001))]
    ang = math.atan2(0.3, -0.8) - math.pi / 2
    s, c = f32(math.sin(ang) * 0.25), f32(math.cos(ang) * 0.25)
    uvs = [sprite_uv(12, 4, 0.5 + (-c - s), 0.5 + (-c + s), 32), sprite_uv(12, 4, 0.5 + (-c + s), 0.5 + (c + s), 32),
           sprite_uv(12, 4, 0.5 + (c + s), 0.5 + (c - s), 32), sprite_uv(12, 4, 0.5 + (c - s), 0.5 + (-c - s), 32)]
    add([(9, 1 + h[0], 9), (9, 1 + h[1], 10), (10, 1 + h[2], 10), (10, 1 + h[3], 9)], [(255, 255, 255)] * 4, uvs, [(240, 240)] * 4)
    x0 = f32(10 + 1 - 0.001)
    add([(x0, 1 + h[3], 9), (x0, 1, 9), (x0, 1, 10), (x0, 1 + h[2], 10)], [(153, 153, 153)] * 4,
        [sprite_uv(12, 4, 0, 0, 32), sprite_uv(12, 4, 0, .5, 32), sprite_uv(12, 4, .5, .5, 32), sprite_uv(12, 4, .5, 0, 32)],
        [(240, 240)] * 4)
    # A plane rotated 22.5 degrees about y (a model element with rotation), centered on (12.5, 5, 12.5).
    ca, sa = math.cos(math.radians(22.5)), math.sin(math.radians(22.5))
    pts = []
    for (px, py) in [(-0.5, 1), (-0.5, 0), (0.5, 0), (0.5, 1)]:
        pts.append((f32(12.5 + px * ca), f32(5 + py), f32(12.5 + px * sa)))
    add(pts, [(v, v, v) for v in grays(230)], [sprite_uv(2, 7, 0, 0), sprite_uv(2, 7, 0, 1), sprite_uv(2, 7, 1, 1), sprite_uv(2, 7, 1, 0)], light4())
    # Tinted partial faces with weighted ambient occlusion (a carpet-like top at 1/16): any gray level can occur.
    for (bx, bz) in [(13, 2), (14, 2), (13, 3)]:
        g = [rnd.randrange(120, 256) for _ in range(4)]
        add([(bx, 6 + 1 / 16, bz), (bx, 6 + 1 / 16, bz + 1), (bx + 1, 6 + 1 / 16, bz + 1), (bx + 1, 6 + 1 / 16, bz)],
            [tinted(GRASS, v) for v in g],
            [sprite_uv(1, 1, 0, 0), sprite_uv(1, 1, 0, 1), sprite_uv(1, 1, 1, 1), sprite_uv(1, 1, 1, 0)], light4())
    # Models reaching outside their block: a plane at x = -1 and one up to y = 17 (the grid's edges).
    add([(-1, 2, 3), (-1, 1, 3), (-1, 1, 4), (-1, 2, 4)], [(153, 153, 153)] * 4,
        [sprite_uv(4, 4, 0, 0), sprite_uv(4, 4, 0, 1), sprite_uv(4, 4, 1, 1), sprite_uv(4, 4, 1, 0)], light4())
    add([(15, 17, 15), (15, 17, 16), (16, 17, 16), (16, 17, 15)], [(255, 255, 255)] * 4,
        [sprite_uv(4, 4, 0, 0), sprite_uv(4, 4, 0, 1), sprite_uv(4, 4, 1, 1), sprite_uv(4, 4, 1, 0)], light4())
    # A sprite mapped with a 90-degree rotation (the other UV layout) and a skewed mapping (not a rectangle: generic).
    add([(1, 9, 1), (1, 9, 2), (2, 9, 2), (2, 9, 1)], [(255, 255, 255)] * 4,
        [sprite_uv(6, 6, 1, 0), sprite_uv(6, 6, 0, 0), sprite_uv(6, 6, 0, 1), sprite_uv(6, 6, 1, 1)], light4())
    add([(1, 9, 3), (1, 9, 4), (2, 9, 4), (2, 9, 3)], [(255, 255, 255)] * 4,
        [sprite_uv(6, 6, 0, 0), sprite_uv(6, 6, .25, 1), sprite_uv(6, 6, 1, 1), sprite_uv(6, 6, .75, 0)], light4())
    return b"".join(q)


# ---------------------------------------------------------------- codec round trip

class Totals:
    def __init__(self):
        self.quads = self.aligned = self.generic = self.tinted = self.misses = self.words = self.ns = 0
        self.pos_aligned = self.pos_generic = self.uv = 0.0
        self.color_bad = self.light_bad = self.gpu_bad = 0


def roundtrip(data, tot, label):
    """data: bytes of BLOCK vertices."""
    nverts = len(data) // 28
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    words = (ctypes.c_uint32 * (nverts * 8 + 64))()
    info = (ctypes.c_int64 * 6)()
    n = lib.mmc_near_debug_encode(buf, nverts, words, len(words), info)
    if n < 0:
        check(False, f"{label}: layer rejected ({n})")
        return None
    quads = nverts // 4
    cpu = (ctypes.c_float * (nverts * 10))()
    gpu = (ctypes.c_float * (nverts * 10))()
    lib.mmc_near_debug_decode_cpu(words, quads, cpu)
    check(lib.mmc_near_debug_decode_gpu(words, n, quads, gpu) == 1, f"{label}: GPU decode ran")
    orig = unpack_vertices(data, nverts)
    for i in range(nverts):
        o, d = orig[i], cpu[10 * i:10 * i + 10]
        is_generic = words[(i // 4) * 8] & 1
        perr = max(abs(o[j] - d[j]) for j in range(3))
        if is_generic:
            tot.pos_generic = max(tot.pos_generic, perr)
        else:
            tot.pos_aligned = max(tot.pos_aligned, perr)
        tot.uv = max(tot.uv, abs(o[3] - d[3]), abs(o[4] - d[4]))
        if (o[5], o[6], o[7]) != (d[5], d[6], d[7]):
            tot.color_bad += 1
        if (o[8], o[9]) != (d[8], d[9]):
            tot.light_bad += 1
    for i in range(nverts * 10):
        if cpu[i] != gpu[i]:
            tot.gpu_bad += 1
    tot.quads += info[0]
    tot.aligned += info[1]
    tot.generic += info[2]
    tot.tinted += info[3]
    tot.misses += info[4]
    tot.words += n
    tot.ns += info[5]
    return words, n


def report(tot, label):
    print(f"  {label}: {tot.quads} quads, {100 * tot.aligned / max(1, tot.quads):.2f}% aligned, {tot.generic} generic "
          f"({tot.misses} of them aligned shapes whose tint didn't split), {tot.tinted} tinted aligned; "
          f"{4 * tot.words / max(1, tot.quads):.2f} B/face vs 112 ({112 * tot.quads / max(1, 4 * tot.words):.2f}x smaller); "
          f"encode {tot.ns / max(1, tot.quads):.0f} ns/quad")
    print(f"    max position error: aligned {tot.pos_aligned:.3g} blocks, generic {tot.pos_generic:.3g} blocks; "
          f"max UV error {tot.uv:.3g} ({tot.uv * 1024:.3g} texels of a 1024 atlas); "
          f"colors wrong {tot.color_bad}, light wrong {tot.light_bad}, GPU/CPU decode mismatches {tot.gpu_bad}")
    check(tot.pos_aligned == 0, f"{label}: aligned positions exact")
    check(tot.pos_generic <= 0.5 / 8192 + 1e-6, f"{label}: generic positions within 1/16384 block")
    check(tot.uv <= 0.5 / 65536 + 1e-7, f"{label}: UVs within 1/131072")
    check(tot.color_bad == 0 and tot.light_bad == 0, f"{label}: colors and light exact")
    check(tot.gpu_bad == 0, f"{label}: shader decode equals CPU decode")


# ---------------------------------------------------------------- PNG

def write_png(path, w, h, rgba):
    rows = b"".join(b"\x00" + bytes(rgba[4 * w * y:4 * w * (y + 1)]) for y in range(h))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def decoded(data):
    """The vertices as the codec returns them (positions and UVs rounded), in vanilla's 28-byte format."""
    nverts = len(data) // 28
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    words = (ctypes.c_uint32 * (nverts * 8 + 64))()
    info = (ctypes.c_int64 * 6)()
    n = lib.mmc_near_debug_encode(buf, nverts, words, len(words), info)
    assert n > 0
    cpu = (ctypes.c_float * (nverts * 10))()
    lib.mmc_near_debug_decode_cpu(words, nverts // 4, cpu)
    return b"".join(vertex(cpu[10 * i:10 * i + 3], [int(v) for v in cpu[10 * i + 5:10 * i + 8]], cpu[10 * i + 3:10 * i + 5],
                           [int(v) for v in cpu[10 * i + 8:10 * i + 10]]) for i in range(nverts))


def render_once(data, cutout, origin, cam, yaw, pitch, fog, vis, rgss, w, h):
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    ref = (ctypes.c_uint8 * (w * h * 4))()
    near = (ctypes.c_uint8 * (w * h * 4))()
    info = (ctypes.c_int64 * 3)()
    o = (ctypes.c_int32 * 3)(*origin)
    c = (ctypes.c_double * 3)(*cam)
    p = (ctypes.c_float * 4)(fog[0], fog[1], vis, rgss)
    r = lib.mmc_near_debug_render(buf, len(data) // 28, cutout, o, c, yaw, pitch, p, w, h, ref, near, info)
    return r, ref, near, info


def render(data, cutout, origin, cam, yaw, pitch, label, fog=(24.0, 64.0), vis=1.0, rgss=0, w=640, h=400):
    """Vanilla's vertex shader on the original vertices vs the near-chunk pipeline, and vanilla's on the rounded
    vertices vs the pipeline (the pipeline's own equivalence: must be identical)."""
    r, ref, near, info = render_once(data, cutout, origin, cam, yaw, pitch, fog, vis, rgss, w, h)
    check(r == 1, f"{label}: render ran ({r})")
    if r != 1:
        return
    r2, ref2, near2, info2 = render_once(decoded(data), cutout, origin, cam, yaw, pitch, fog, vis, rgss, w, h)
    check(r2 == 1 and info2[0] == 0, f"{label}: pipeline identical to vanilla's shader on the rounded vertices ({info2[0]} pixels differ)")
    check(bytes(near) == bytes(near2), f"{label}: re-encoding the rounded vertices gives the same image")
    # Flip rows for viewing: texture row 0 is the bottom of vanilla's clip space.
    def flip(img):
        return b"".join(bytes(img[4 * w * y:4 * w * (y + 1)]) for y in range(h - 1, -1, -1))
    def opaque(img):
        b = bytearray(img)
        b[3::4] = b"\xff" * (w * h)
        return bytes(b)
    diff = bytearray(w * h * 4)
    for i in range(w * h):
        d = max(abs(ref[4 * i + k] - near[4 * i + k]) for k in range(4))
        diff[4 * i:4 * i + 4] = bytes((min(255, 64 + d), 0, 0, 255)) if d else bytes((24, 24, 24, 255))
    os.makedirs(OUT, exist_ok=True)
    base = os.path.join(OUT, label)
    write_png(base + "-vanilla.png", w, h, opaque(flip(ref)))
    write_png(base + "-near.png", w, h, opaque(flip(near)))
    write_png(base + "-diff.png", w, h, flip(bytes(diff)))
    print(f"  {label}: {info[2]} pixels covered; against the original vertices {info[0]} differ (largest channel "
          f"difference {info[1]}); against the rounded vertices {info2[0]} differ; {base}-*.png")
    return info[0], info[1], info[2]


def render_arena(data, cutout, origin, cam, yaw, pitch, label, fog=(24.0, 64.0), vis=1.0, rgss=0, w=640, h=400):
    """The near image through the game's own path (arena entry after a filler, mmc_pass_begin, mmc_near_draw with two
    records and section index 1, mmc_submit) against vanilla's vertex shader on the same vertices."""
    buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    ref = (ctypes.c_uint8 * (w * h * 4))()
    near = (ctypes.c_uint8 * (w * h * 4))()
    info = (ctypes.c_int64 * 6)()
    o = (ctypes.c_int32 * 3)(*origin)
    c = (ctypes.c_double * 3)(*cam)
    p = (ctypes.c_float * 4)(fog[0], fog[1], vis, rgss)
    quads = len(data) // 112
    r = lib.mmc_near_debug_render_arena(buf, len(data) // 28, cutout, quads // 3, o, c, yaw, pitch, p, w, h, ref, near, info)
    check(r == 1, f"{label}: in-game path ran ({r})")
    if r != 1:
        return
    if LIT:
        # Lit mode (METALMC_EXP=lit): the near shader samples the lightmap per pixel instead of per vertex, so its color is
        # close to vanilla's, not identical (measured: up to 10 levels on real terrain, where smooth lighting varies across
        # a face, 32 on the hand-made shapes); and it writes the G-buffer, which the relight's overlay test must find plain
        # (albedo, face, AO and light levels give back the forward color). Not under a section fade (vis < 1), whose fog
        # blend the G-buffer doesn't hold.
        check(info[1] <= 40, f"{label}: in-game path in lit mode within 40 levels of vanilla's shader ({info[0]} pixels differ, "
              f"largest difference {info[1]})")
        s = (ctypes.c_int64 * 16)()
        lib.mmc_near_debug_lit_stats(s)
        marked = s[0]
        check(s[14] == 1 and marked > 0, f"{label}: lit mode: G-buffer written ({marked} pixels) and relit")
        check(s[8] == marked, f"{label}: lit mode: every marked pixel's depth key matches the depth buffer ({s[8]} of {marked})")
        if vis >= 1.0:
            check(s[9] >= 0.98 * marked, f"{label}: lit mode: the overlay test finds plain terrain ({s[9]} of {marked} plain, {s[10]} flagged)")
        print(f"  {label}: lit G-buffer {marked} px, faces +X {s[1]} -X {s[2]} +Y {s[3]} -Y {s[4]} +Z {s[5]} -Z {s[6]} other {s[7]}; "
              f"key matches {s[8]}; overlay test plain {s[9]}, flagged {s[10]}; largest sky {s[11] / 16:.2f}, block {s[12] / 16:.2f}; "
              f"mean AO {s[13] / 1000:.3f}; forward vs vanilla: {info[0]} pixels differ, largest {info[1]}")
    else:
        # The game's pipeline comes from its own library build, so a rare pixel may round differently (one level at most).
        check(info[1] <= 1 and info[0] <= max(1, info[2] // 1000), f"{label}: in-game path matches vanilla's shader "
              f"({info[0]} pixels differ, largest difference {info[1]})")
    check(info[3] == (2 if quads // 3 > 0 else 1), f"{label}: both records drawn ({info[3]})")
    check(info[4] == 1, f"{label}: a freed range isn't reused before its submit completes")
    check(info[5] == 1, f"{label}: a freed range is reused after its submit completes")
    print(f"  {label}: {info[2]} pixels covered, {info[0]} differ (largest {info[1]}); {info[3]} records drawn; "
          f"freed range held until the GPU finished: {'yes' if info[4] else 'NO'}, reused after: {'yes' if info[5] else 'NO'}")


# ---------------------------------------------------------------- main

def main():
    print(f"lib {LIB}")
    print("1. arena allocator")
    out = (ctypes.c_int64 * 3)()
    for seed in (1, 6, 10):
        r = lib.mmc_near_debug_arena_check(1 << 18, 200000, seed, out)
        check(r == 0, f"arena invariants (seed {seed}, failing step {r})")
        print(f"  seed {seed}: {'ok' if r == 0 else 'FAILED at step ' + str(r)}; {out[0]} allocations, {out[1]} refused when full, "
              f"largest free range at the end {out[2]} slots")

    print("2-3. codec round trip on real terrain")
    regions = sorted(glob.glob(os.path.join(WORLD, "dimensions/minecraft/overworld/region/*.mca")))
    if not regions:
        regions = sorted(glob.glob(os.path.join(WORLD, "region/*.mca")))
    rnd = random.Random(5)
    cap = 1 << 21
    vbuf = (ctypes.c_uint8 * (cap * 28))()
    info = (ctypes.c_int64 * 5)()
    real = Totals()
    plants = 0
    chunks = 0
    surface = None
    for path in rnd.sample(regions, min(12, len(regions))):
        for idx in rnd.sample(range(1024), 6):
            n = lib.mmc_near_debug_synth_chunk(path.encode(), idx, 0, 23, vbuf, cap, info)
            if n <= 0:
                continue
            chunks += 1
            plants += info[4]
            solid = info[1] * 4
            base = ctypes.addressof(vbuf)
            if solid > 0:
                roundtrip(ctypes.string_at(base, solid * 28), real, f"{os.path.basename(path)}#{idx} solid")
            if n > solid:
                roundtrip(ctypes.string_at(base + solid * 28, (n - solid) * 28), real, f"{os.path.basename(path)}#{idx} cutout")
            if surface is None and info[2] > 200:
                surface = (path, idx)
    print(f"  {chunks} chunks from {min(12, len(regions))} regions of {WORLD} ({plants} of the quads are plants' cross-model faces)")
    report(real, "real terrain (solid + cutout)")

    print("   hand-made shapes")
    sb = shapes()
    shp = Totals()
    roundtrip(sb, shp, "shapes")
    report(shp, "shapes")

    print("4. layers the codec must refuse")
    good = vertex((0, 0, 0), (255, 255, 255), (0, 0), (0, 240))
    cases = {
        "alpha 128": vertex((0, 0, 0), (255, 255, 255), (0, 0), (0, 240), alpha=128),
        "light 300": vertex((0, 0, 0), (255, 255, 255), (0, 0), (300, 240)),
        "NaN position": vertex((float("nan"), 0, 0), (255, 255, 255), (0, 0), (0, 240)),
        "position 40": vertex((40.5, 0.3, 0), (255, 255, 255), (0, 0), (0, 240)),
        "UV 1.5": vertex((0, 0, 0), (255, 255, 255), (1.5, 0), (0, 240)),
    }
    words = (ctypes.c_uint32 * 64)()
    inf = (ctypes.c_int64 * 6)()
    for name, bad in cases.items():
        b = good * 3 + bad
        r = lib.mmc_near_debug_encode((ctypes.c_uint8 * len(b)).from_buffer_copy(b), 4, words, 64, inf)
        check(r == -1, f"refuses {name}")
        print(f"  {name}: {'refused' if r == -1 else 'ACCEPTED (' + str(r) + ')'}")
    b = good * 6
    r = lib.mmc_near_debug_encode((ctypes.c_uint8 * len(b)).from_buffer_copy(b), 6, words, 64, inf)
    check(r == -1, "refuses a vertex count that isn't a multiple of 4")

    print("5. rendering: vanilla's vertex shader on the original vertices vs the near-chunk pipeline")
    if surface:
        path, idx = surface
        # The chunk's highest section with leaves or grass, seen from above and to the north.
        for sy in range(23, -1, -1):
            n = lib.mmc_near_debug_synth_chunk(path.encode(), idx, sy, sy, vbuf, cap, info)
            if n > 0 and info[1] + info[2] > 100:
                break
        solid = info[1] * 4
        origin = (0, -64 + 16 * sy, 0)
        cam = (8.37, origin[1] + 22.61, -9.13)
        print(f"  {os.path.basename(path)} chunk {idx}, section {sy} (y {origin[1]}): {info[1]} solid, {info[2]} cutout quads")
        base = ctypes.addressof(vbuf)
        sol = ctypes.string_at(base, solid * 28)
        cut = ctypes.string_at(base + solid * 28, (n - solid) * 28)
        real_diff = [render(sol, 0, origin, cam, 0.0, 50.0, "real-solid"),
                     render(cut, 1, origin, cam, 0.0, 50.0, "real-cutout", vis=0.55),
                     render(sol, 0, origin, (8.37, origin[1] + 6.2, -3.5), 20.0, 18.0, "real-solid-close-rgss", rgss=1, fog=(4.0, 20.0))]
        check(all(d is not None and d[0] == 0 for d in real_diff), "real terrain renders identically to vanilla's shader")
        print("   the same through the game's draw path (arena, mmc_pass_begin, mmc_near_draw, mmc_submit)")
        render_arena(sol, 0, origin, cam, 0.0, 50.0, "arena-real-solid")
        render_arena(cut, 1, origin, cam, 0.0, 50.0, "arena-real-cutout", vis=0.55)
    render(sb, 1, (0, 64, 0), (8.0, 64 + 14.0, -7.0), 0.0, 45.0, "shapes-overview")
    render(sb, 1, (0, 64, 0), (4.5, 64 + 5.3, 3.0), -29.0, 8.8, "shapes-plants-close")
    render(sb, 0, (0, 64, 0), (8.5, 64 + 3.5, 6.5), -26.6, 27.0, "shapes-lava-close")
    render_arena(decoded(sb), 1, (0, 64, 0), (8.0, 64 + 14.0, -7.0), 0.0, 45.0, "arena-shapes-overview")
    render_arena(decoded(sb), 0, (-48, 64, 32), (-48 + 8.5, 64 + 3.5, 32 + 6.5), -26.6, 27.0, "arena-shapes-lava-close")

    print()
    print("FAILED: " + "; ".join(failures) if failures else "all checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
