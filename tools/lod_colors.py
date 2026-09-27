"""Averages Minecraft block textures into LOD material colors and writes LodColors.swift.
Pure Python (no PIL): minimal PNG decoder for 8-bit RGB/RGBA/gray and 1/2/4/8-bit palette images."""
import glob
import os
import struct
import zipfile
import zlib

jar = glob.glob(os.path.expanduser("~/.gradle/caches/fabric-loom/minecraftMaven/net/minecraft/minecraft-clientonly-deobf/26.3/*.jar"))[0]
z = zipfile.ZipFile(jar)

def png(name):
    data = z.read(f"assets/minecraft/textures/block/{name}.png")
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos, idat, plte, trns = 8, b"", None, None
    while pos < len(data):
        n, typ = struct.unpack(">I4s", data[pos:pos + 8])
        body = data[pos + 8:pos + 8 + n]
        if typ == b"IHDR":
            w, h, depth, ctype, _, _, interlace = struct.unpack(">IIBBBBB", body)
        elif typ == b"PLTE":
            plte = body
        elif typ == b"tRNS":
            trns = body
        elif typ == b"IDAT":
            idat += body
        pos += 12 + n
    assert interlace == 0, name
    raw = zlib.decompress(idat)
    chans = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[ctype]
    bpp_bits = chans * depth
    stride = (w * bpp_bits + 7) // 8
    bpp = max(1, bpp_bits // 8)
    rows, prev, i = [], bytearray(stride), 0
    for _ in range(h):
        f = raw[i]; line = bytearray(raw[i + 1:i + 1 + stride]); i += 1 + stride
        for x in range(stride):
            a = line[x - bpp] if x >= bpp else 0
            b = prev[x]
            c = prev[x - bpp] if x >= bpp else 0
            if f == 1: line[x] = (line[x] + a) & 255
            elif f == 2: line[x] = (line[x] + b) & 255
            elif f == 3: line[x] = (line[x] + (a + b) // 2) & 255
            elif f == 4:
                p = a + b - c; pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                line[x] = (line[x] + (a if pa <= pb and pa <= pc else (b if pb <= pc else c))) & 255
        rows.append(line); prev = line
    px = []
    for line in rows:
        for x in range(w):
            if ctype == 3:
                bit = x * depth
                v = (line[bit // 8] >> (8 - depth - bit % 8)) & ((1 << depth) - 1)
                r, g, b = plte[3 * v:3 * v + 3]
                a = trns[v] if trns and v < len(trns) else 255
            elif ctype == 6: r, g, b, a = line[4 * x:4 * x + 4]
            elif ctype == 2:
                r, g, b = line[3 * x:3 * x + 3]
                # tRNS for truecolor: one fully transparent RGB value (16-bit samples, 8-bit images use the low byte).
                a = 0 if trns and len(trns) >= 6 and (r, g, b) == (trns[1], trns[3], trns[5]) else 255
            elif ctype == 4: r = g = b = line[2 * x]; a = line[2 * x + 1]
            else:
                if depth == 8:
                    r = g = b = line[x]
                else:
                    bit = x * depth
                    v = (line[bit // 8] >> (8 - depth - bit % 8)) & ((1 << depth) - 1)
                    r = g = b = v * 255 // ((1 << depth) - 1)
                    # tRNS for grayscale is the raw sample value.
                    a = 0 if trns and len(trns) >= 2 and v == ((trns[0] << 8) | trns[1]) else 255
                    px.append((r, g, b, a))
                    continue
                a = 0 if trns and len(trns) >= 2 and r == trns[1] else 255
            px.append((r, g, b, a))
    return w, h, px

def average(name, tint=None):
    w, h, px = png(name)
    opaque = [p for p in px if p[3] >= 128]
    n = len(opaque)
    r, g, b = (sum(p[k] for p in opaque) / n / 255 for k in range(3))
    if tint is not None:
        r, g, b = r * ((tint >> 16) & 255) / 255, g * ((tint >> 8) & 255) / 255, b * (tint & 255) / 255
    return (r, g, b)

def mix(*cs):
    return tuple(sum(c[k] for c in cs) / len(cs) for k in range(3))

GRASS, FOLIAGE, WATER = 0x91BD59, 0x77AB2F, 0x3F76E4   # plains defaults
colors = [
    ("air", None), ("stone", average("stone")), ("dirt", average("dirt")),
    ("grass", average("grass_block_top", GRASS)), ("sand", average("sand")),
    ("water", average("water_still", WATER)), ("unknown", average("stone")),
    ("deepslate", average("deepslate")), ("gravel", average("gravel")), ("log", average("oak_log")),
    ("planks", average("oak_planks")), ("leaves", average("oak_leaves", FOLIAGE)),
    ("cherryLeaves", average("cherry_leaves")), ("snow", average("snow")), ("ice", average("ice")),
    ("clay", average("clay")), ("terracotta", average("terracotta")), ("lava", average("lava_still")),
    ("cobblestone", average("cobblestone")), ("bricks", average("bricks")), ("path", average("dirt_path_top")),
    ("farmland", average("farmland")), ("hay", average("hay_block_top")), ("wool", average("white_wool")),
    ("moss", average("moss_block")), ("cherryWood", average("cherry_log")),
    ("lightStone", mix(average("diorite"), average("andesite"), average("calcite"))),
    ("granite", average("granite")), ("sandstone", average("sandstone_top")), ("mud", average("mud")),
    ("amethyst", average("amethyst_block")), ("pumpkin", average("pumpkin_side")),
]
def luma(name):
    """Mean Rec. 709 luma of the texture's opaque texels (untinted, gamma space like the atlas)."""
    w, h, px = png(name)
    opaque = [p for p in px if p[3] >= 128]
    return sum(0.2126 * p[0] + 0.7152 * p[1] + 0.0722 * p[2] for p in opaque) / len(opaque) / 255

# Top and side textures per material, for LOD texture detail (same order as colors). Grass sides use dirt:
# a 2-block LOD voxel would otherwise show a grass edge on every block of a cliff.
sprites = [
    ("air", None, None), ("stone", "stone", "stone"), ("dirt", "dirt", "dirt"),
    ("grass", "grass_block_top", "dirt"), ("sand", "sand", "sand"), ("water", "water_still", "water_still"),
    ("unknown", "stone", "stone"), ("deepslate", "deepslate_top", "deepslate"), ("gravel", "gravel", "gravel"),
    ("log", "oak_log_top", "oak_log"), ("planks", "oak_planks", "oak_planks"), ("leaves", "oak_leaves", "oak_leaves"),
    ("cherryLeaves", "cherry_leaves", "cherry_leaves"), ("snow", "snow", "snow"), ("ice", "ice", "ice"),
    ("clay", "clay", "clay"), ("terracotta", "terracotta", "terracotta"), ("lava", "lava_still", "lava_still"),
    ("cobblestone", "cobblestone", "cobblestone"), ("bricks", "bricks", "bricks"), ("path", "dirt_path_top", "dirt_path_side"),
    ("farmland", "farmland", "dirt"), ("hay", "hay_block_top", "hay_block_side"), ("wool", "white_wool", "white_wool"),
    ("moss", "moss_block", "moss_block"), ("cherryWood", "cherry_log_top", "cherry_log"),
    ("lightStone", "calcite", "calcite"), ("granite", "granite", "granite"), ("sandstone", "sandstone_top", "sandstone"),
    ("mud", "mud", "mud"), ("amethyst", "amethyst_block", "amethyst_block"), ("pumpkin", "pumpkin_top", "pumpkin_side"),
]
assert [n for n, _, _ in sprites] == [n for n, _ in colors]
sprite_lines = []
for name, top, side in sprites:
    if top is None:
        sprite_lines.append(f'    LodSprite(top: "", side: "", topLuma: 1, sideLuma: 1),   // {name}')
    else:
        sprite_lines.append(f'    LodSprite(top: "{top}", side: "{side}", topLuma: {luma(top):.3f}, sideLuma: {luma(side):.3f}),   // {name}')
# Past the materials: the grass side overlay (top) and base (side), for full-resolution grass sides, which
# show vanilla's fringe (lodGrassSideSprite).
sprite_lines.append(f'    LodSprite(top: "grass_block_side_overlay", side: "grass_block_side", topLuma: {luma("grass_block_side_overlay"):.3f}, sideLuma: {luma("grass_block_side"):.3f}),   // grass side (not a material)')

# Grass block sides are dirt with a biome-tinted grass fringe on top (grass_block_side_overlay). The side
# color for tint T is (1 - f) * base + f * gray * T, with f the overlay's share of the texture.
def grass_side_parts():
    w, h, base = png("grass_block_side")
    _, _, over = png("grass_block_side_overlay")
    fringe = [i for i in range(len(over)) if over[i][3] >= 128]
    rest = [i for i in range(len(base)) if over[i][3] < 128 and base[i][3] >= 128]
    f = len(fringe) / len(base)
    b = tuple(sum(base[i][k] for i in rest) / len(rest) / 255 for k in range(3))
    g = sum((over[i][0] + over[i][1] + over[i][2]) / 3 for i in fringe) / len(fringe) / 255
    return b, f, g

GS_BASE, GS_FRINGE, GS_GRAY = grass_side_parts()

def grass_side(tint):
    t = ((tint >> 16) & 255) / 255, ((tint >> 8) & 255) / 255, (tint & 255) / 255
    return tuple((1 - GS_FRINGE) * GS_BASE[k] + GS_FRINGE * GS_GRAY * t[k] for k in range(3))

# Top, side and bottom colors per material (same order as colors). Most materials look the same from every
# side; these don't. Vanilla hillsides are mostly the brown of grass block sides, not grass green.
faces = {
    "grass": (average("grass_block_top", GRASS), grass_side(GRASS), average("dirt")),
    "deepslate": (average("deepslate_top"), average("deepslate"), average("deepslate_top")),
    "log": (average("oak_log_top"), average("oak_log"), average("oak_log_top")),
    "path": (average("dirt_path_top"), average("dirt_path_side"), average("dirt")),
    "farmland": (average("farmland"), average("dirt"), average("dirt")),
    "hay": (average("hay_block_top"), average("hay_block_side"), average("hay_block_top")),
    "cherryWood": (average("cherry_log_top"), average("cherry_log"), average("cherry_log_top")),
    "sandstone": (average("sandstone_top"), average("sandstone"), average("sandstone_bottom")),
    "pumpkin": (average("pumpkin_top"), average("pumpkin_side"), average("pumpkin_top")),
}

lines = []
for name, c in colors:
    if c is None:
        lines.append(f"    SIMD4(0, 0, 0, 0), SIMD4(0, 0, 0, 0), SIMD4(0, 0, 0, 0),   // {name}")
    else:
        top, side, bottom = faces.get(name, (c, c, c))
        lines.append("    " + ", ".join(f"SIMD4({v[0]:.3f}, {v[1]:.3f}, {v[2]:.3f}, 1)" for v in (top, side, bottom)) + f",   // {name}")
out = """import simd

// Generated by lod_colors.py from Minecraft 26.3's block textures: the mean color of each material's
// representative textures' opaque texels, with the plains grass/foliage/water tints. Three entries per
// material (top, side, bottom), indexed by MetalMCCore.Mat raw value * 3. Keeps far LOD terrain close to
// the average color of vanilla's textured terrain at the seam.
let lodMaterialFaceColors: [SIMD4<Float>] = [
""" + "\n".join(lines) + "\n]\n\n" + f"""/// Grass block sides: the untinted dirt part's mean color, the tinted fringe's share of the texture, and the
/// fringe overlay's mean gray. Side color for tint T = (1 - fringe) * base + fringe * gray * T.
let lodGrassSideBase = SIMD3<Float>({GS_BASE[0]:.3f}, {GS_BASE[1]:.3f}, {GS_BASE[2]:.3f})
let lodGrassSideFringe: Float = {GS_FRINGE:.3f}
let lodGrassOverlayGray: Float = {GS_GRAY:.3f}
""" + """
/// Block textures used for LOD texture detail (names under block/ in the block atlas), with the mean luma
/// of each texture's opaque texels. The LOD multiplies its flat color by texel luma / mean luma.
struct LodSprite {
    let top: String
    let side: String
    let topLuma: Float
    let sideLuma: Float
}

let lodMaterialSprites: [LodSprite] = [
""" + "\n".join(sprite_lines) + "\n]\n"
open(os.path.expanduser("~/Projects/metal-mc/Sources/MetalMCNative/LodColors.swift"), "w").write(out)
print(out)
