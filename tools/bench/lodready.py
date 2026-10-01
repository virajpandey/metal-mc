"""Whether a flight's timing started after the LOD's first build: compares the build's completion (its first
"LOD: update ... in N s" line) with the first long frame logged during timing (METALMC_HITCH, -PbenchHitches=1).
usage: python3 lodready.py <run log>  -> "lod ready" or "LOD STILL BUILDING (timing started N s early)" """
import re, sys

def secs(h, m, s):
    return int(h) * 3600 + int(m) * 60 + float(s)

built = hitch = None
for line in open(sys.argv[1], errors="replace"):
    if built is None:
        m = re.search(r"\[metalmc-native\] (\d\d):(\d\d):(\d\d\.\d+) LOD: update .* in [\d.]+ s", line)
        if m: built = secs(*m.groups())
    if hitch is None:
        m = re.search(r"METALMC_HITCH (\d\d):(\d\d):(\d\d\.\d+)", line)
        if m: hitch = secs(*m.groups())
if built is None or hitch is None:
    print("lod ready: unknown")
elif hitch < built:
    print(f"LOD STILL BUILDING (timing started {built - hitch:.0f} s early)")
else:
    print("lod ready")
