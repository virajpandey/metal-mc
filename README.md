# MetalMC

An experimental Metal terrain renderer for Minecraft-style worlds on Apple Silicon. The goal is to cover the near-terrain work of Sodium and Nvidium and the far-terrain LOD work of Voxy in one engine.

## Status: milestone 0

This milestone has a procedural voxel world, parallel meshing with face culling, and one Metal pipeline (opaque pass, then translucent water, with distance fog). Culling is CPU frustum plus fog-distance culling. There's also a headless benchmark harness.

## Build

```bash
swift build -c release
```

## Run

Interactive: WASD to move, E or Space to go up, Q to go down, arrow keys to look.

```bash
.build/release/MetalMCViewer --chunks 32
```

Benchmark (offscreen). It writes `frames.csv`, PNG captures, and `summary.txt`, and prints one `BENCH` line.

```bash
.build/release/MetalMCViewer --bench --chunks 32 --frames 600 --out bench_out
```

## Telemetry

| Field | Meaning |
|---|---|
| `cpu_ms` | CPU time to cull and encode a frame |
| `gpu_ms` | `gpuEndTime - gpuStartTime` of the frame's command buffer |
| `culled_pct` | Share of non-empty sections skipped by frustum or fog culling |
| `hole_pct` | Share of lower-half pixels still matching the sky color. The camera always looks down at terrain, so a nonzero value means missing geometry |

## Roadmap

1. Load real Minecraft worlds from Anvil `.mca` region files.
2. GPU-driven culling with Metal object and mesh shaders.
3. Far-terrain LOD, Voxy-style.
4. Shading: sun shadows, sky, and fog.
5. In-game integration as a Fabric mod.

Backface culling stays off until an image test checks face winding.

## Target

Minecraft Java 26.3, the current release. We'll port when 26.4 ships. In 26.x, overworld region files are at `dimensions/minecraft/overworld/region/`. Test worlds go in `fixtures/`, which git ignores.

## Clean-room policy

Sodium (PolyForm Shield) and Voxy (All Rights Reserved) are studied for techniques only, and none of their code is used. Code adapted from LGPL-3.0 projects, such as Distant Horizons, is credited in the file where it's used.

## License

LGPL-3.0-only. See `COPYING.LESSER` and `COPYING`.
