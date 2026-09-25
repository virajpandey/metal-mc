# MetalMC

An experimental Metal terrain renderer for Minecraft-style worlds on Apple Silicon. The goal is to cover the near-terrain work of Sodium and Nvidium and the far-terrain LOD work of Voxy in one engine.

## Status: milestone 2 (near engine), mostly done

- **Milestone 1:** loads Minecraft 26.x worlds from Anvil region files. That includes 26.x's palette changes: plain-string entries, `{"": name}` wrappers, and `{id, properties}`.
- **Milestone 2:**
  - Quads are 4 bytes each, expanded by the vertex shader.
  - Greedy meshing, face-direction buckets, backface culling, and reverse-Z depth.
  - Golden-hash and fuzzy image-diff checks.

Results on `claudeworld` (841 chunks, M3 Pro, 1280×720, 600-frame camera path):

| Step | Quads | GPU memory | GPU ms (mean) |
|---|---|---|---|
| M1: 16-byte vertices + indices | 2.48M | 208.4 MB | 2.42 |
| + backface culling, reverse-Z | 2.48M | 208.4 MB | 1.82 |
| + 4-byte pulled quads | 2.48M | 9.6 MB | 2.61 |
| + face buckets | 2.48M | 9.6 MB | 1.81 |
| + greedy meshing | 0.95M | 3.7 MB | 1.07 |

Procedural 64×64 chunks: 534K quads, 2.2 MB, GPU 0.37 ms mean / 0.97 ms p99.

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

Load a real world save, then check it against golden hashes or a reference run:

```bash
.build/release/MetalMCViewer --bench --world fixtures/claudeworld --golden goldens/claudeworld.txt --compare bench_out/ref_m2d
```

A/B switches: `--no-cull`, `--cw`, `--standard-z`, `--no-buckets`, `--no-greedy`.

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
