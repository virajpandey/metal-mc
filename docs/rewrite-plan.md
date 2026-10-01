# Rewrite plan: one GPU-driven terrain system from the player's feet to the horizon (draft, for Viraj's go)

## Where a frame goes today

Viraj's world (4.1 km of real terrain), native 3456×2234, flying at y 150, uncapped, TAA on (`tools/bench/profile_split.sh`):

| Part | Cost | Notes |
|---|---|---|
| LOD | ~2.8 ms of the frame; 3.6 ms of main-pass GPU | 2.4 ms is the vertex stage for 1.83 M quads, and 85-90% of those quads own no pixel |
| TAA | ~1.1 ms | |
| Vanilla terrain at render distance 12 | ~0.6 ms | |
| Everything else | ~1.8 ms | entities, sky, clouds, hand and HUD, presenting |

The LOD's quads are the big lever: at levels 1 and up, 95% of their vertex work draws nothing. Vanilla's near chunks cost little in GPU time. They matter as the foundation the lighting needs (one terrain format, no vanilla/LOD seam) more than as a cost.

## Order

1. **Far field (started 2026-09-30, prototype working, `METALMC_FARFIELD=1`).** A per-pixel height-field ray march replaces the LOD's quads from level 1 out (docs/far-field-design.md). Next:
   - shading parity (AO at column feet, water surface drop, fades);
   - then rings past the LOD from the world generator, at full vertical precision, to reach the horizon;
   - then level 0's hidden quads: move the quad/march boundary inward, measured against vanilla.
2. **Near chunks in our format (swing 1).**
   - Vanilla's section compiler keeps meshing: it knows every block model, fluid and tint.
   - Its 28-byte vertices get repacked into a compact form (about 8-12 bytes) in one GPU arena.
   - One GPU-built draw list covers every section.
   - Near chunks become level "−1" of the same system: one seam fewer, and one format for the lighting to extend.
   - First step built and checked offline (2026-09-30, `METALMC_EXP=nearchunks`, docs/near-chunks-design.md): solid and cutout layers repacked on the mesh workers into 32-byte quad records (3.46× smaller than vanilla's on real terrain), drawn from one arena with vanilla's own uniforms, pixel-identical to vanilla's shader in the offline render checks. The draw list is still built on the CPU; in-game checks pending.
3. **Visibility buffer + deferred lighting in tile memory (swing 2).** This is the base the lighting is built on. Each pixel is shaded once, and the lighting buffers live in the GPU's on-chip tile memory instead of RAM.
   - First step built and checked offline (2026-10-01, `METALMC_EXP=lit`, docs/lighting-design.md, "Lit mode"): the terrain shaders write an 8-byte G-buffer in the main pass (stored, not yet in tile memory), and the terrain is relit after the level with the sun, sky, moon and block light, in the anti-aliasing's resolve. In-game checks pending.
4. **Lighting milestone 1:**
   - sky and atmosphere;
   - an HDR/EDR tone curve for the XDR panel;
   - ray-traced sun shadows (a prototype exists, off by default) out to the horizon, using the height field past the ray-traced range.

   Scored against SEUS reference stills once there's a way to render them: SEUS crashes or hangs Apple's OpenGL shader compiler, so that needs our own GLSL→Metal loader or a PC.

Each rewrite is built beside the old path with an on/off switch. It becomes the default only if it scores at least as well against vanilla (the fidelity tour) and holds a locked 120 Hz (the flight bench).

## Who builds what (one parallel layer, split by files)

| Builder | Owns | Checks its own work with |
|---|---|---|
| A | `FarField.swift`, far-field docs | the fidelity tour against the quad LOD and vanilla, flight bench GPU times, screenshots |
| B | near-chunk repacking and draw list: new `NearChunks.swift` plus the section mixins | vanilla-vs-new screenshot diffs on the fidelity tour, section counts, flight bench |
| C | sky, atmosphere and HDR output: new `Sky.swift`, `Hdr.swift` plus the sky mixins | reference photos and screenshots at dawn, noon and dusk, the EDR headroom check |

- **Reviewer:** GPT-6 Astra via codex, read-only, running at the same time as the builders and reviewing their diffs. There is no separate review layer after the build.
- **Integration:** I run the full bench suite (fidelity tour, flight bench, vsync drop count) myself before anything becomes a default.
- **Why the split:** the three touch different files, so they can't collide. B and C both touch Java mixins, but different ones.
