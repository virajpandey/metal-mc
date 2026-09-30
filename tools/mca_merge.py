"""Merge pregenerated chunks into a Minecraft save without touching the chunks it already has.

usage: python3 tools/mca_merge.py <base save> <donor save> <output save> [dimension, default overworld]

For every chunk position in either save's region files: the base save's chunk stays byte for byte if it's fully
generated (status minecraft:full); otherwise the donor's chunk is used if the donor has one (preferring a full one).
The same decision is applied to the chunk's entity and point-of-interest data (entities/, poi/), so animals and
villages come with the terrain they belong to. level.dat, player data and every other dimension come from the base.
The output save is written fresh (the base and donor are only read). Oversized chunks stored in .mcc files are
carried along. Both saves must be the same world (seed) and game version.
"""
import os
import shutil
import struct
import sys
import zlib

SECTOR = 4096


def read_region(path):
    """{index: (compression byte, payload bytes, timestamp)} for a region file; {} if missing."""
    if not os.path.exists(path):
        return {}
    data = open(path, "rb").read()
    if len(data) < 2 * SECTOR:
        return {}
    chunks = {}
    for i in range(1024):
        loc = struct.unpack(">I", data[4 * i:4 * i + 4])[0]
        if not loc:
            continue
        off = (loc >> 8) * SECTOR
        if off + 5 > len(data):
            continue
        length, ctype = struct.unpack(">IB", data[off:off + 5])
        payload = data[off + 5:off + 4 + length]
        ts = struct.unpack(">I", data[SECTOR + 4 * i:SECTOR + 4 * i + 4])[0]
        chunks[i] = (ctype, payload, ts)
    return chunks


def write_region(path, chunks):
    if not chunks:
        return
    header = bytearray(2 * SECTOR)
    body = bytearray()
    sector = 2
    for i in sorted(chunks):
        ctype, payload, ts = chunks[i]
        blob = struct.pack(">IB", len(payload) + 1, ctype) + payload
        pad = (-len(blob)) % SECTOR
        blob += b"\0" * pad
        count = len(blob) // SECTOR
        if count > 255:
            raise ValueError(f"{path}: chunk {i} too large for inline storage ({count} sectors)")
        struct.pack_into(">I", header, 4 * i, (sector << 8) | count)
        struct.pack_into(">I", header, SECTOR + 4 * i, ts)
        body += blob
        sector += count
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(header)
        f.write(body)
    os.replace(tmp, path)


def status(entry, mcc_path):
    ctype, payload, _ = entry
    if ctype & 0x80:  # stored externally
        if not os.path.exists(mcc_path):
            return "?"
        payload = open(mcc_path, "rb").read()
    try:
        c = ctype & 0x7F
        nbt = zlib.decompress(payload) if c == 2 else (__import__("gzip").decompress(payload) if c == 1 else payload)
    except Exception:
        return "?"
    k = nbt.find(b"\x08\x00\x06Status")
    if k < 0:
        return "?"
    n = struct.unpack(">H", nbt[k + 9:k + 11])[0]
    return nbt[k + 11:k + 11 + n].decode(errors="replace")


def dim_dir(save, dim):
    return os.path.join(save, "dimensions", "minecraft", dim)


def main():
    base, donor, out = sys.argv[1:4]
    dim = sys.argv[4] if len(sys.argv) > 4 else "overworld"
    if os.path.exists(out):
        sys.exit(f"output exists: {out}")
    shutil.copytree(base, out, ignore=shutil.ignore_patterns("session.lock"))
    kept = filled = upgraded = 0
    for sub in ("region",):
        bdir, ddir, odir = (os.path.join(dim_dir(s, dim), sub) for s in (base, donor, out))
        names = sorted(set(f for d in (bdir, ddir) if os.path.isdir(d) for f in os.listdir(d) if f.endswith(".mca")))
        os.makedirs(odir, exist_ok=True)
        for name in names:
            rx, rz = map(int, name.split(".")[1:3])
            b, d = read_region(os.path.join(bdir, name)), read_region(os.path.join(ddir, name))
            merged, source = {}, {}
            for i in set(b) | set(d):
                cx, cz = rx * 32 + i % 32, rz * 32 + i // 32
                mcc = f"c.{cx}.{cz}.mcc"
                bs = status(b[i], os.path.join(bdir, mcc)) if i in b else None
                ds = status(d[i], os.path.join(ddir, mcc)) if i in d else None
                if bs == "minecraft:full" or (i in b and i not in d) or (i in b and ds != "minecraft:full" and bs is not None):
                    merged[i], source[i] = b[i], "base"
                    kept += 1
                else:
                    merged[i], source[i] = d[i], "donor"
                    if i in b:
                        upgraded += 1
                    else:
                        filled += 1
                    if d[i][0] & 0x80:
                        shutil.copy2(os.path.join(ddir, mcc), os.path.join(odir, mcc))
            write_region(os.path.join(odir, name), merged)
            # Entity and point-of-interest data follow the terrain's source, chunk by chunk.
            for side in ("entities", "poi"):
                sb, sd, so = (os.path.join(dim_dir(s, dim), side) for s in (base, donor, out))
                eb, ed = read_region(os.path.join(sb, name)), read_region(os.path.join(sd, name))
                em = {}
                for i, src in source.items():
                    pick = eb if src == "base" else ed
                    if i in pick:
                        em[i] = pick[i]
                if em:
                    os.makedirs(so, exist_ok=True)
                    write_region(os.path.join(so, name), em)
                elif os.path.exists(os.path.join(so, name)):
                    os.remove(os.path.join(so, name))
    print(f"{dim}: kept {kept} base chunks, filled {filled} new chunks, replaced {upgraded} unfinished chunks")


if __name__ == "__main__":
    main()
