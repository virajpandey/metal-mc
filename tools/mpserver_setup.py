"""Prepares mod/run-server for a local offline-mode test server with a copy of a fixture world, then
start it with `./gradlew runServer` from mod/ and join with `-PmpServer=127.0.0.1:25565`.
usage: mpserver_setup.py <fixture name>"""
import hashlib
import json
import os
import shutil
import sys
import time
import uuid

fixture = sys.argv[1]
root = "/Users/rachnap/Projects/metal-mc"
run = os.path.join(root, "mod/run-server")
os.makedirs(run, exist_ok=True)
world = os.path.join(run, "world")
if os.path.exists(world):
    shutil.move(world, os.path.join(run, f"world-old-{int(time.time())}"))
shutil.copytree(os.path.join(root, "fixtures", fixture), world)
lock = os.path.join(world, "session.lock")
if os.path.exists(lock):
    os.remove(lock)

with open(os.path.join(run, "eula.txt"), "w") as f:
    f.write("eula=true\n")
props = {
    "online-mode": "false", "level-name": "world", "view-distance": "12", "simulation-distance": "8",
    "gamemode": "creative", "spawn-protection": "0", "server-port": "25565", "server-ip": "127.0.0.1",
    "motd": "MetalMC LOD test", "sync-chunk-writes": "false", "allow-flight": "true",
}
with open(os.path.join(run, "server.properties"), "w") as f:
    for k, v in props.items():
        f.write(f"{k}={v}\n")
# Offline-mode UUID, as Java's UUID.nameUUIDFromBytes("OfflinePlayer:" + name).
name = "MetalMCBench"
h = bytearray(hashlib.md5(("OfflinePlayer:" + name).encode()).digest())
h[6] = (h[6] & 0x0F) | 0x30
h[8] = (h[8] & 0x3F) | 0x80
with open(os.path.join(run, "ops.json"), "w") as f:
    json.dump([{"uuid": str(uuid.UUID(bytes=bytes(h))), "name": name, "level": 4, "bypassesPlayerLimit": False}], f)
print("server ready in", run)
