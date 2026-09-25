# MetalMC Bench (Fabric mod, Minecraft 26.3)

This mod runs an automated in-game benchmark, used to measure vanilla Minecraft's baseline on OpenGL and Vulkan (MoltenVK) before comparing the Metal backend.

When launched with `-Dmetalmc.bench=1` (set by `./gradlew runClient`), it:

1. opens `run/saves/claudeworld` through quick play,
2. holds the camera at the start pose for 20 s so chunks can build,
3. flies a fixed 60 s orbit (radius 140, y = 150, looking at the center) while timing every rendered frame,
4. writes `run/metalmc-bench/<label>-<time>.csv` and `.txt`, prints a `METALMC_BENCH` summary line, and quits.

The graphics backend is chosen by `preferredGraphicsBackend` in `run/options.txt` (`"opengl"` or `"vulkan"`). The summary records the backend that actually started, so a silent fallback can't mislabel a run.

```bash
env JAVA_HOME=/opt/homebrew/opt/openjdk@25 ./gradlew runClient -PbenchLabel=opengl
```

Built from the official Fabric example mod template (CC0).
