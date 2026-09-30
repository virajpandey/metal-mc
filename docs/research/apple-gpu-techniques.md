# GPU rendering research for MetalMC

Research current to **September 30, 2026**. No files changed; no builds or benchmarks run.

**The strongest architecture is bounded terrain LOD, GPU-driven visibility, tile-resident direct lighting, amortized probe GI, and selective ray tracing.** The sources establish the necessary mechanisms, but do not establish that their combination achieves native-resolution, SEUS-quality rendering at a locked 120 Hz on M3 Pro.

“Infinite distance” needs to mean progressively coarser, streamed terrain within fixed memory and processing budgets. Voxy demonstrates this general approach; it does not retain arbitrary distances at full block detail. [Voxy](https://modrinth.com/mod/voxy)

The display’s native resolution is **3456×2234**, with adaptive refresh up to **120 Hz**. That gives **8.333 ms per frame** and **7.721 million output pixels**; native rendering at 120 Hz processes approximately **926 million pixel positions/second**, before overdraw or additional passes. These are calculations from Apple’s specifications. [MacBook Pro specifications](https://support.apple.com/en-us/117737)

Throughout this report, **published measurements** retain their original hardware and workload; **project recommendations** are engineering judgments, not measured M3 Pro results.

## 1. Apple TBDR: keeping deferred lighting intermediates on chip

- Apple’s tile-based deferred renderer can retain render attachments in tile memory while executing multiple drawing and lighting phases within one render pass. This is distinct from an engine merely dividing its lighting work into screen-space tiles. [Apple: Modern Rendering with Metal](https://developer.apple.com/videos/play/wwdc2019/601/)

- A texture with `.storageMode = .memoryless` exists only during its render pass. It cannot preserve contents through ordinary `.load` or `.store` operations, or become a texture sampled by a subsequent compute pass. [Memoryless resource rules](https://developer.apple.com/documentation/metal/mtlresourceoptions/storagemodememoryless)

- **Programmable blending/framebuffer fetch** lets a fragment shader read the existing color-attachment value at its pixel. Deferred lighting can therefore consume G-buffer values without first writing and rereading G-buffer textures in device memory. [Apple: optimized Metal rendering](https://developer.apple.com/videos/play/wwdc2019/606/)

- **Implicit imageblocks** derive their layout from render attachments. **Explicit imageblocks** use shader-defined structures and support temporary data such as arrays of transparent fragments. Both forms can coexist. [Apple: Imageblocks](https://developer.apple.com/videos/play/tech-talks/603/)

- Fragment shaders access their own imageblock pixel; kernel-based tile shaders can access the tile’s imageblock. Explicit structures support packed normalized and specialized formats rather than requiring every field to occupy a full floating-point vector. [Apple: Imageblocks](https://developer.apple.com/videos/play/tech-talks/603/)

A practical single-pass deferred sequence is:

1. Attach transient G-buffer targets and a persistent HDR color target; clear or initialize their contents.
2. Rasterize opaque and alpha-tested terrain into the G-buffer.
3. Optionally dispatch a tile shader to derive depth bounds and construct a tile-local light list.
4. Shade through framebuffer-fetch fragment shaders or a tile kernel; accumulate HDR lighting locally.
5. Store final color and only the auxiliary buffers needed by later effects. Apple demonstrates this render-pass organization and tile-local light lists. [Deferred organization](https://developer.apple.com/videos/play/wwdc2019/601/), [tile shading](https://developer.apple.com/videos/play/tech-talks/604/)

**Relevant Apple9 limits:** these are API limits from Apple’s current table, not a statement that every combination reaches maximum occupancy. [Metal feature tables, pp. 7–10](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

| Resource or limit | Apple9 |
|---|---:|
| Explicit imageblock allocation | 32 KB |
| Implicit imageblock allocation | 256 KB |
| Explicit imageblock size per pixel/sample | 64 B |
| Implicit imageblock size per pixel/sample | 128 B |
| Threadgroup memory allocation | 32 KB |
| Color attachments / raster order groups | 8 / 8 |
| Maximum tile, no MSAA / 4× MSAA | 32×32 / 32×16 |

- Imageblock and threadgroup allocations share the applicable imageblock budget. **The 256 KB implicit limit does not grant a 256 KB custom explicit imageblock.** [Metal feature tables](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf)

- Size explicit storage using pipeline `imageblockSampleLength` and `imageblockMemoryLength(forDimensions:)`, then configure the render pass. Excessive tile dimensions or layer counts can make render-pass creation fail. [Apple’s imageblock sample](https://developer.apple.com/documentation/metal/implementing-order-independent-transparency-with-image-blocks)

- `dispatchThreadsPerTile` runs within the render encoder. Draws before a tile dispatch complete their tile-memory accesses before it executes; subsequent draws see its results. This synchronization does not imply arbitrary ordering between fragment invocations from separate draws. [Tile-shading ordering](https://developer.apple.com/videos/play/tech-talks/604/)

- **Raster order groups** order conflicting accesses from fragments covering the same pixel. Separate groups can let lighting calculations proceed concurrently while ordering only the final accumulation; one unnecessarily broad group can serialize substantial shading work. [Apple: Raster Order Groups](https://developer.apple.com/videos/play/tech-talks/605/)

- Framebuffer fetch does not expose the depth attachment through the color-input mechanism. Apple’s deferred example therefore includes a linear-depth color attachment; alternatively, choose another supported reconstruction/storage arrangement. [Apple’s deferred implementation](https://developer.apple.com/videos/play/wwdc2019/601/)

- **Suggested starting layout:** packed albedo/material, packed normal/roughness, and linear depth, with `rgba16Float` HDR accumulation. Treat the exact packing as an experiment; explicit packed types and attachment formats have different layout constraints. [Imageblock formats](https://developer.apple.com/videos/play/tech-talks/603/)

- **Published savings:** Apple’s WWDC19 example reports eliminating approximately **85 MB of bandwidth and footprint** by correcting transient MSAA storage. It also shows Afterpulse keeping four G-buffer attachments transient, storing only color and depth. The 85 MB figure is not an M3 deferred-lighting benchmark. [WWDC19 measurement](https://developer.apple.com/videos/play/wwdc2019/606/)

- **Calculated opportunity:** at this display resolution, avoiding one write and one read of a hypothetical **16 B/pixel** G-buffer removes **247 MB/frame**, or **29.65 GB/s at 120 Hz**, before compression/cache effects. This is traffic arithmetic, not measured DRAM savings.

- **Critical boundary:** SSR, HZB construction, temporal reprojection, and spatial denoising need data beyond the current tile or pass. Export their required depth, normals, motion, or IDs selectively; a wholly memoryless G-buffer cannot serve those later consumers. [Memoryless lifetime](https://developer.apple.com/documentation/metal/mtlstoragemode/memoryless), [tile scope](https://developer.apple.com/videos/play/tech-talks/604/)

## 2. Visibility buffers on Metal

- A visibility buffer stores the identity of the visible primitive, then reconstructs material attributes during shading. The original paper targets the bandwidth and shading overhead of conventional G-buffers. [Burns and Hunt, 2013](https://jcgt.org/published/0002/02/04/paper.pdf)

- The Forge’s documented implementation uses **32-bit primitive identification plus 32-bit depth**. Its historical encoding reserves one alpha-mask bit, eight draw-ID bits, and 23 triangle-ID bits; this is an implementation choice, not a universal visibility-buffer format. [Engel: Triangle Visibility Buffer](https://diaryofagraphicsprogrammer.blogspot.com/2018/03/triangle-visibility-buffer.html)

- On Metal, rasterize an integer primitive reference using `primitive_id` or an explicitly supplied identifier, then fetch the referenced geometry/material in a shading pass. The Forge provides a real Metal implementation, including indexed geometry and GPU-driven rendering. [The Forge](https://github.com/ConfettiFX/The-Forge)

- **Terrain-specific proposal:** store a compact quad reference plus enough information to identify the constituent triangle. Flat block faces can reconstruct normals and UV orientation cheaply; arbitrary models still need their geometry and interpolation data. This adapts the primitive-reference approach. [Visibility-buffer reconstruction](https://jcgt.org/published/0002/02/04/paper.pdf)

- A quad ID alone is insufficient for every Minecraft model: nonplanar quads, distinct corner attributes, and triangulation choices affect interpolation. Preserve the original triangle interpretation or restrict the compact path to compatible terrain faces. [Attribute reconstruction method](https://diaryofagraphicsprogrammer.blogspot.com/2018/03/triangle-visibility-buffer.html)

- Shading pays for geometry fetches, material indirection, barycentric reconstruction, and texture gradients. Engel computes analytic derivatives from triangle geometry; blindly differentiating reconstructed attributes across unrelated neighboring primitives is unsuitable. [Analytic gradients](https://diaryofagraphicsprogrammer.blogspot.com/2018/03/triangle-visibility-buffer.html)

- **Alpha-cutout leaves must perform their alpha test during visibility rasterization**, before committing the winning ID/depth. Deferring that test until shading would leave missing geometry behind rejected leaves. The Forge separates opaque and alpha-masked visibility paths. [Alpha-masked visibility](https://diaryofagraphicsprogrammer.blogspot.com/2018/03/triangle-visibility-buffer.html)

- A one-layer visibility buffer cannot describe multiple translucent surfaces. Use a separate forward/transparency path or retain multiple visibility layers. Apple’s imageblock OIT sample stores four layers and explicitly drops excess fragments, making its bounded approximation clear. [Apple OIT sample](https://developer.apple.com/documentation/metal/implementing-order-independent-transparency-with-image-blocks)

- **Fit with TBDR:** an ID imageblock followed by tile shading could retain visibility locally. A compute shading pass instead requires exporting visibility/depth, gaining scheduling freedom at the cost of pass boundaries and memory traffic. This is an architectural inference from Metal’s tile lifetime. [Tile shading](https://developer.apple.com/videos/play/tech-talks/604/)

- **The comparison that matters:** visibility-buffer shading versus tile-resident deferred shading. Eliminating a device-memory G-buffer is a stronger advantage than replacing an already transient G-buffer; the latter comparison depends on reconstruction, material coherence, and auxiliary-buffer costs. [Original visibility-buffer motivation](https://jcgt.org/published/0002/02/04/paper.pdf), [Apple deferred approach](https://developer.apple.com/videos/play/wwdc2019/601/)

- The Forge also publishes a compute-driven Visibility Buffer 2.0 path, including macOS/iOS support. Its existence establishes feasibility, not that software rasterization beats Apple’s hardware rasterizer for Minecraft terrain. [The Forge’s implementation history](https://github.com/ConfettiFX/The-Forge)

## 3. M3-generation features: caching, ray tracing, and mesh shading

**Dynamic Caching and execution**

- M3 belongs to **Apple GPU family 9**. Its shader core dynamically allocates register storage over a shader’s execution and shares flexible on-chip storage among several memory uses. This reduces penalties from provisioning every invocation for peak register demand. [Apple’s M3 architecture talk](https://developer.apple.com/videos/play/tech-talks/111375/)

- Dynamic Caching does not remove occupancy constraints: tile, threadgroup, stack, register, and buffer working sets still compete. Keep intermediate structures small and measure the complete shader workload. [Apple’s M3 architecture talk](https://developer.apple.com/videos/play/tech-talks/111375/)

**Ray tracing**

- Metal ray intersection can execute from compute and fragment functions on Apple silicon. An `intersector` call may invoke custom intersection functions; `intersection_query` instead exposes candidate processing to the calling shader. [Metal ray-tracing guide](https://developer.apple.com/videos/play/wwdc2023/10128/)

- M3 adds fixed-function traversal and intersection-function reordering. Apple recommends **`intersector` over `intersection_query` where possible**: queries increase RT scratch traffic and disable the reorder stage. Neither interface makes material shading free. [M3 ray-tracing architecture](https://developer.apple.com/videos/play/tech-talks/111375/)

- Alpha-tested foliage needs intersection-time alpha rejection. Small per-primitive data can reduce lookup chains; Apple reports **10–16% improvement in one test application**, not a general foliage or M3 speedup. [Metal ray-tracing performance](https://developer.apple.com/videos/play/wwdc2022/10105/)

- Metal calls bottom-level geometry structures **primitive acceleration structures**, and top-level structures **instance acceleration structures**. Query allocation and build/refit scratch sizes from the device rather than estimating them from triangle count. [Acceleration-structure setup](https://developer.apple.com/videos/play/wwdc2023/10128/)

- `.refit` must be selected for a modifiable structure. Apple warns that this more conservative representation can reduce intersection performance; refitting is most appropriate for small geometry changes. [Refit usage flag](https://developer.apple.com/documentation/metal/mtlaccelerationstructureusage/refit)

- **Terrain implication:** rebuild affected geometry structures after topology-changing block edits; preserve unchanged structures. Camera movement alone need not rebuild an unchanged world-space acceleration structure. This follows from structures representing scene geometry and instance transforms. [Acceleration-structure model](https://developer.apple.com/documentation/metal/mtlaccelerationstructure)

- Apple’s documented standard capacities include **2²⁴ instances**, **2²⁴ geometries**, and **2²⁸ primitives** in their respective structures. `extendedLimits` raises these to 2³⁰ but may affect performance; check API availability before using it. Capacity is not a practical per-frame target. [Acceleration-structure limits](https://developer.apple.com/documentation/metal/mtlaccelerationstructureusage/extendedlimits)

- Apple’s 2022 discussion considers rebuilding a TLAS containing a few thousand objects reasonable. For larger scenes, its 2023 guidance separates static and dynamic substructures, optionally through another instancing level, trading traversal overhead against reduced rebuilding. [2022 guidance](https://developer.apple.com/videos/play/wwdc2022/10105/), [2023 guidance](https://developer.apple.com/videos/play/wwdc2023/10128/)

- **Compaction:** build, obtain compacted size on the GPU, allocate the smaller destination, then copy-and-compact. Retire the original only after completion. Schedule this during streaming rather than inserting a blocking readback into the frame loop. [Compaction sequence](https://developer.apple.com/videos/play/wwdc2023/10128/)

- Batch independent builds in one acceleration-structure encoder with nonoverlapping scratch storage. Apple’s 2022 software improvements reported up to **2.3× faster builds**, **38% faster refits**, and **2.8× faster parallel builds**; these predate M3 and are not M3 hardware ratios. [Published build improvements](https://developer.apple.com/videos/play/wwdc2022/10105/)

- I found **no transferable M3 Pro milliseconds-per-TLAS-rebuild curve** specifying instance count, geometry, flags, and concurrent workload. Budget AS maintenance separately from ray traversal and measure both.

**Object and mesh stages**

- Object shaders can cull/select LOD and dispatch mesh threadgroups; mesh shaders emit vertices and primitives directly to rasterization. This can avoid an intermediate compute-generated geometry or draw-command buffer. [Metal mesh-shader model](https://developer.apple.com/videos/play/wwdc2022/10162/)

- Metal’s mesh output supports up to **256 vertices**, **512 primitives**, and **16 KB total output**; object payload is separately limited. These maxima are not recommended meshlet sizes. [Mesh-shader limits](https://developer.apple.com/videos/play/wwdc2022/10162/)

- M3 improves object/mesh scheduling to keep intermediate data on chip and supports mesh draws in ICBs. Apple also recommends tightly sizing output structures and omitting rejected primitives instead of emitting them for later culling. [M3 mesh implementation](https://developer.apple.com/videos/play/tech-talks/111375/)

- **Likely wins over vertex pulling:** substantial meshlet rejection, compressed/procedural expansion, and shared per-meshlet calculations. **Possible losses:** tiny batches, little rejection, oversized payloads, and cheap geometry already handled efficiently by indexed/pulled vertices. These are workload-based predictions, not published M3 Pro crossover measurements. [Mesh pipeline rationale](https://developer.apple.com/videos/play/wwdc2022/10162/)

## 4. GPU-driven submission and hierarchical visibility

- Metal compute kernels can generate ICB commands for visible objects, avoiding CPU readback of visibility followed by CPU draw encoding. Apple provides a complete sample. [GPU-generated ICBs](https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-gpu)

- Reset reused commands, encode visible work, and execute the resulting range. Apple’s sample additionally optimizes the ICB to remove empty commands and redundant state; optimization itself is work and should be justified by profiling. [ICB lifecycle](https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-gpu)

- An ICB can also retain commands across frames. Rebuilding every command is unnecessary when stable command data and changing visibility can be represented more cheaply. [Persistent ICBs](https://developer.apple.com/documentation/metal/creating-an-indirect-command-buffer)

- **Metal 3 bindless:** tier-2 argument buffers can contain GPU buffer addresses and resource IDs written directly by the application. Unbounded arrays support scene/material tables without rebinding every resource for every draw. [Go bindless with Metal 3](https://developer.apple.com/videos/play/wwdc2022/10101/)

- **Heaps are allocation/residency tools, not the binding mechanism.** Argument buffers describe references; `useResource`/`useHeap` declare indirectly accessed resources. Heap allocation can reduce per-resource CPU work. [Argument buffers and heaps](https://developer.apple.com/videos/play/wwdc2022/10101/)

- Heap resources require deliberate synchronization. Apple warns both about missing residency and about conservative dependencies from tracked heaps; untracked heaps require explicit fences/events at actual hazards. Swift object lifetime alone does not establish GPU completion. [Residency and hazards](https://developer.apple.com/videos/play/wwdc2022/10101/)

- **Two-phase HZB algorithm:** render previously visible geometry at the current camera pose; construct a hierarchical depth buffer; test remaining candidates against it; render newly visible survivors. Nanite documents this conservative approach. [Karis et al.: Nanite](https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf)

- Another implementation tests against previous-frame HZB first and retests rejected candidates against fresh depth. The essential requirement is recovering candidates rejected by stale visibility rather than accepting temporal false occlusion. [Two-phase implementation discussion](https://hannosprogrammingblog.blogspot.com/2017/11/two-phase-occlusion-culling-part-1.html)

- **Metal mapping:** compute cull/ICB → depth render → compute depth pyramid → compute retest/ICB → second render. Apple demonstrates GPU-generated occluder and scene passes without a CPU round trip. [Modern Rendering with Metal](https://developer.apple.com/videos/play/wwdc2019/601/)

- **Reduction rule:** retain the farthest covered depth in each HZB region: maximum for conventional increasing depth, minimum for reversed depth. Test conservative projected bounds against every overlapped region at the chosen level. This follows from the requirement never to classify potentially visible geometry as fully occluded. [Conservative HZB basis](https://advances.realtimerendering.com/s2021/Karis_Nanite_SIGGRAPH_Advances_2021_final.pdf)

- **TBDR tradeoff:** current-frame HZB creates a genuine cross-tile dependency and normally requires storing depth and ending the relevant render pass. Compare those costs against geometry saved; a tile shader cannot construct a globally synchronized full-screen pyramid from isolated tile contents. [Tile execution scope](https://developer.apple.com/videos/play/tech-talks/604/)

- **Recommended hierarchy:** region/section culling first, meshlet culling second, cheap directional/per-quad rejection last. Per-triangle tests fetch and transform geometry themselves, so fine-grained culling needs enough rejection to repay its cost. [Triangle-filtering costs](https://diaryofagraphicsprogrammer.blogspot.com/2018/03/triangle-visibility-buffer.html)

## 5. GI and denoising under an 8.33 ms frame budget

These measurements have different scopes and **must not be added together as a predicted frame time**.

| Technique | Published measurement | Hardware, resolution, and scope |
|---|---|---|
| DDGI prototype | 1–2 ms/frame | RTX 2080 Ti, 1920×1080 at 60 Hz; approximately 100 ms indirect-light response latency. [Author’s report](https://morgan3d.github.io/articles/2019-04-01-ddgi/overview.html) |
| ReSTIR GI example | 8.9 ms | RTX 3090, 1080p, one sample/pixel, two-bounce example, without denoising. [Paper, Fig. 1](https://d1qx31qr3h6wln.cloudfront.net/publications/ReSTIR%20GI.pdf) |
| Original SVGF | About 10 ms ±15% | 1920×1080 reconstruction; 2017 implementation, not current Metal timing. [Paper and abstract](https://research.nvidia.com/publication/2017-07_spatiotemporal-variance-guided-filtering-real-time-reconstruction-path-traced) |
| NRD REBLUR / RELAX | 2.55 / 3.25 ms | RTX 4080, native 1440p, combined diffuse/specular denoising, documented default configuration. [NRD](https://github.com/NVIDIA-RTX/NRD) |
| Screen-space radiance-cascade prototype | About 2 / 6 / 26 ms | Low/medium/high settings; accessible thesis abstract does not identify matching GPU/resolution. Not suitable for hardware-normalized comparison. [Chalmers thesis](https://odr.chalmers.se/items/9e42007a-bb60-440f-ba28-04410822758f) |

- **DDGI** traces rays from world-space probes and stores directional irradiance plus visibility information. Shading interpolates nearby probes; visibility weighting reduces the leaking inherent in ordinary irradiance interpolation. [DDGI paper](https://jcgt.org/published/0008/02/01/)

- Probe updates can reuse previous irradiance for subsequent bounces. Their tracing/update cost is decoupled from screen resolution; visible-surface evaluation still scales with pixels. The original prototype’s low cost included temporal amortization. [DDGI author’s explanation](https://morgan3d.github.io/articles/2019-04-01-ddgi/overview.html)

- Production DDGI work adds probe state management, faster response heuristics, and multiresolution cascaded volumes. These are directly relevant to a large editable world with sparse changes. [Scaling DDGI for production](https://arxiv.org/abs/2009.10796)

- **Project recommendation:** update a bounded set of nearby/dirty probes each frame; prioritize newly exposed caves, edits, and moving emissive sources. Preserve direct lighting responsiveness independently of slower diffuse-bounce convergence. [Production probe scheduling](https://arxiv.org/abs/2009.10796)

- **ReSTIR GI** reuses indirect-light path samples across time and neighboring pixels. The original paper reports large error reductions at comparable computation cost; these are quality improvements, not equivalent multipliers in frame rate. [Ouyang et al.](https://research.nvidia.com/publication/2021-06_restir-gi-path-resampling-real-time-path-tracing)

- Reservoir storage, neighbor reads, visibility validation, and denoising remain real costs. NVIDIA’s integration exposes explicit reservoir-buffer allocation and temporal/spatial reuse stages. A reduced-resolution experiment is more defensible here than native-resolution deployment. [RTXDI ReSTIR GI integration](https://github.com/NVIDIA-RTX/RTXDI/blob/main/Doc/RestirGI.md)

- **Radiance cascades** allocate finer spatial sampling to nearby transport and finer angular sampling to farther transport. Full 3D storage is difficult; the July 2026 *Split Radiance Cascades* preprint addresses it with sparse world-space probes and ray splitting. [Freeman and Sannikov](https://arxiv.org/abs/2607.20384)

- Screen-space radiance cascades inherit missing off-screen contributors/occluders. The Chalmers prototype also reports movement flicker and upscaling artifacts, despite avoiding a separate stochastic denoising pass. It is not evidence of complete world-space GI. [Prototype limitations](https://odr.chalmers.se/items/9e42007a-bb60-440f-ba28-04410822758f)

- **SVGF** combines temporal accumulation, variance estimation, and edge-aware à-trous filtering. Depth/normal rejection and motion reprojection are essential parts of its reconstruction, with history failure around disocclusion and changing lighting. [SVGF paper](https://cg.ivd.kit.edu/publications/2017/svgf/svgf_preprint.pdf)

- Metal provides `MPSSVGF` with controls for reprojection, temporal blending, variance, and bilateral weights. It is a useful implementation baseline, but Apple publishes no universal M3 Pro runtime for it. [MPSSVGF](https://developer.apple.com/documentation/metalperformanceshaders/mpssvgf)

- NRD supplies more recent denoising designs, but its documented integration targets D3D/Vulkan. A Metal implementation requires porting/integration work; its RTX measurements do not establish Metal performance. [NRD integration](https://github.com/NVIDIA-RTX/NRD)

## 6. Atmosphere, clouds, sun shadows, and water

**Hillaire atmosphere**

- Hillaire’s 2020 method separates atmospheric transmittance, multiple scattering, sky view, and aerial perspective into small LUTs, supporting changing atmosphere parameters without a large iterative precomputation. [Hillaire’s paper](https://sebh.github.io/publications/egsr2020.pdf)

| LUT | Published PC resolution | Published update cost, NVIDIA GTX 1080 |
|---|---:|---:|
| Transmittance | 256×64 | 0.01 ms |
| Multiple scattering | 32×32 | 0.07 ms |
| Sky view | 200×100 | 0.05 ms |
| Aerial perspective | 32³ | 0.04 ms |

These are the paper’s Table 2 settings. Final sky/aerial-perspective rendering adds **0.14 ms**, giving **0.31 ms total at 1280×720** for its daytime example. [Hillaire, performance section](https://sebh.github.io/publications/egsr2020.pdf)

- The author publishes the implementation and a reference path tracer. Use it to verify scattering equations and LUT parameterization; the 200×100 sky-view size above is the paper’s measured configuration, not a mandatory resolution. [Author’s implementation](https://github.com/sebh/UnrealEngineSkyAtmosphere)

- LUT atmosphere alone does not produce detailed mountain/cloud volumetric shadows. Hillaire adds shadowed ray marching with jitter/reprojection; its illustrated 32-sample case raises atmosphere rendering to approximately **1 ms**. [Volumetric-shadow extension](https://sebh.github.io/publications/egsr2020.pdf)

**Clouds and sunlight**

- Guerrilla reports its original cloud prototype rendering in **under 2 ms on PS4**. That is a historical cloud-system result, not a promise for native 3456×2234 rendering or a complete atmosphere pipeline. [Nubis presentation](https://www.guerrilla-games.com/read/nubis-authoring-real-time-volumetric-cloudscapes-with-the-decima-engine)

- Its later work distinguishes distant cloud skies from expensive near-field/fly-through clouds and explicitly discusses temporal artifacts. These should be separate quality modes in this renderer. [Nubis, Evolved](https://www.guerrilla-games.com/read/nubis-evolved)

- **Proposed initial cloud budget:** 0.4–0.8 ms, using reduced-resolution marching, history, and bounded steps; this is a project allocation requiring measurement. Keep a cheaper distant-cloud option when sky coverage dominates the screen. [Cloud-system tradeoffs](https://www.guerrilla-games.com/read/nubis-evolved)

- **Recommended sunlight baseline:** cascaded shadow maps with filtered sampling, bounded shadow distance, and cheaper distant geometry. Add selective ray-traced contact/shadow refinement only where it improves visible detail. Hybrid shadow-map/RT designs have shipping-engine precedents. [AMD hybrid rendering samples](https://gpuopen.com/learn/samples-library/)

**Water**

- **Gerstner waves** provide analytic displacement and derivatives from a small wave sum; separate fine normal detail from geometric undulation. This is a strong initial fit for Minecraft lakes and oceans. [GPU Gems: water simulation](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models)

- **FFT oceans** synthesize many frequencies from a wave spectrum and suit broad ocean surfaces. They introduce additional transform/storage work; use them only when the ocean appearance justifies the complexity. [Tessendorf-based simulation discussion](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-1-effective-water-simulation-physical-models)

- **Reflection hybrid:** try screen-space depth traversal first, trace selected misses against scene geometry, and use sky/environment fallback beyond the represented scene. AMD publishes classification and denoising stages for this combination. [Hybrid stochastic reflections](https://gpuopen.com/manuals/fidelityfx_sdk/samples/hybrid-reflections/)

- Classify reflection work by roughness and usefulness before tracing. A fixed ray budget and maximum distance are appropriate project controls; glossy water covering most of the screen is an important worst-case workload. [Reflection classifier](https://gpuopen.com/manuals/fidelityfx_sdk/techniques/classifier/)

- **Refraction baseline:** sample an opaque scene-color copy with normal-based distortion, check the displaced sample against scene depth, and reduce visibility with water depth. This is a practical approximation with missing-layer and screen-edge limitations. [Tardif’s water implementation](https://alextardif.com/Water.html)

- Water therefore normally needs access to opaque color/depth from another phase; reading arbitrary refracted screen positions is not equivalent to same-pixel framebuffer fetch. Plan that pass boundary explicitly. [Water inputs](https://alextardif.com/Water.html), [imageblock access scope](https://developer.apple.com/videos/play/tech-talks/603/)

- **Caustics:** begin with projected/analytic animated caustics restricted to shallow illuminated water. GPU Gems documents an explicitly approximate method; physically tracing refracted light paths is a separate, substantially harder transport problem. [GPU Gems: caustics](https://developer.nvidia.com/gpugems/gpugems/part-i-natural-effects/chapter-2-rendering-water-caustics)

## 7. macOS HDR/EDR and ProMotion presentation

The standard native EDR setup is:

```swift
metalLayer.wantsExtendedDynamicRangeContent = true
metalLayer.pixelFormat = .rgba16Float
metalLayer.colorspace =
    CGColorSpace(name: CGColorSpace.extendedLinearDisplayP3)
```

Apple demonstrates this combination. Shader output must actually use the selected linear color space; assigning a P3 label does not convert incorrectly encoded RGB values. [Explore HDR rendering with EDR](https://developer.apple.com/videos/play/wwdc2021/10161/)

- In EDR, **1.0 is SDR reference white**, while larger values represent brighter highlights. Available headroom is the current maximum luminance relative to SDR white, and varies with display conditions/settings. [EDR terminology](https://developer.apple.com/videos/play/wwdc2022/10114/)

- Query the `NSScreen` associated with the renderer’s window, rather than assuming the main display. Moving a window between screens can change the applicable EDR characteristics. [Screen-specific EDR](https://developer.apple.com/videos/play/wwdc2021/10161/)

| NSScreen property | Meaning |
|---|---|
| `maximumExtendedDynamicRangeColorComponentValue` | Current usable maximum EDR component |
| `maximumPotentialExtendedDynamicRangeColorComponentValue` | Potential maximum under suitable display conditions |
| `maximumReferenceExtendedDynamicRangeColorComponentValue` | Reference-rendering capability; zero where unsupported |

Apple distinguishes these values explicitly; potential headroom is not the current tone-mapping target. [EDR capability queries](https://developer.apple.com/videos/play/wwdc2021/10161/)

- Read current headroom during rendering and respond to screen-parameter/display changes. Values above current headroom can clip; even an EDR-capable display can currently expose only 1.0. [Custom tone mapping](https://developer.apple.com/documentation/metal/performing-your-own-tone-mapping)

- **Recommended mapping:** expose scene-linear lighting, preserve ordinary diffuse/UI reference levels, and smoothly compress highlights into current headroom. Avoid multiplying the entire SDR image by headroom, which changes midtones whenever display brightness changes. [EDR mapping guidance](https://developer.apple.com/documentation/metal/performing-your-own-tone-mapping)

- `CAEDRMetadata` supports system tone mapping for HDR media. For generated game imagery, a controlled custom output transform is usually easier to integrate; avoid accidentally applying two independent tone-mapping stages. [System tone mapping](https://developer.apple.com/documentation/metal/using-system-tone-mapping-on-video-content)

- **Calculated footprint:** one native-resolution `rgba16Float` image occupies approximately **58.9 MiB** before allocation overhead. Full-resolution color histories, ping-pong buffers, and drawables therefore need an explicit memory budget.

- `CAMetalDisplayLink` provides callbacks associated with a `CAMetalLayer`, with a preferred frame-rate range and preferred latency. Scheduling is best effort; requesting 120 Hz does not guarantee delivery. [CAMetalDisplayLink](https://developer.apple.com/documentation/quartzcore/cametaldisplaylink)

- Its update contains a drawable, `targetTimestamp` rendering deadline, and `targetPresentationTimestamp` for animation timing. Use those timestamps for pacing and late camera/input sampling. [Display-link update](https://developer.apple.com/documentation/quartzcore/cametaldisplaylink/update)

- `preferredFrameLatency` accepts **1 or 2 frames**. One frame is a useful low-latency target when the workload fits; macOS windowed presentation may require greater actual latency. [Latency property](https://developer.apple.com/documentation/quartzcore/cametaldisplaylink/preferredframelatency)

- MetalFX temporal upscaling consumes jittered color, depth, and motion information. It provides a Metal 3 route to native-sized output from fewer shaded pixels, but still needs correct history and a measured reconstruction budget. [MetalFX temporal upscaling](https://developer.apple.com/videos/play/wwdc2022/10103/)

## 8. Variable rasterization-rate maps

- `MTLRasterizationRateMap` describes a mapping from logical viewport coordinates to a physically smaller raster target. Horizontal rates are specified per column and vertical rates per row; it is not an arbitrary independent shading-rate value for every pixel. [Creating a rate map](https://developer.apple.com/documentation/metal/creating-a-rasterization-rate-map)

- A uniform rate of 0.5 in both dimensions produces approximately half-width, half-height intermediate rendering, or one-quarter the pixels. Obtain actual allocation dimensions from `physicalSize(layer:)`. [Rendering with a rate map](https://developer.apple.com/documentation/metal/rendering-with-a-rasterization-rate-map)

- Metal may choose actual rates at least as high as requested. Use the compiled map’s dimensions/mappings rather than assuming that every requested scale is represented exactly. [Rate-map guarantees](https://developer.apple.com/documentation/metal/creating-a-rasterization-rate-map)

- Render into intermediate color/depth textures, then reconstruct the destination using the map. Metal supplies `rasterization_rate_map_data` and its decoder for coordinate conversion. [Coordinate handling](https://developer.apple.com/documentation/metal/rendering-with-a-rasterization-rate-map)

- **Costs:** mapping/reconstruction, an additional intermediate/output operation, and integration changes for screen-space effects. Apple recommends the technique when saved rendering work exceeds the scaling cost. [When variable rates help](https://developer.apple.com/documentation/metal/rendering-at-different-rasterization-rates)

- Documented applications include blurred or otherwise less important regions and far shadow cascades. Fixed peripheral reduction may also be useful, but this laptop has no specified gaze-tracking input to drive eye-tracked foveation. [Rate-map use cases](https://developer.apple.com/documentation/metal/mtlrasterizationratemap)

- **Project caution:** HZB bounds, SSR traversal, motion vectors, and temporal history must agree about logical versus physical coordinates. Changing the map can invalidate assumptions about neighboring pixels; account for the mapping explicitly. [Rate-map coordinate model](https://developer.apple.com/documentation/metal/rendering-with-a-rasterization-rate-map)

- **Priority:** prototype ordinary dynamic resolution first. Add variable rates when a repeatable spatial quality pattern saves more than its reconstruction/integration cost; I found no broadly applicable M3 Pro percentage speedup.

## 9. Minecraft terrain ideas: Sodium, Nvidium, and Voxy

The formats below describe the inspected source versions; branch URLs can change.

| Project | Inspected representation | Transferable idea |
|---|---|---|
| Sodium | 20-byte terrain vertex | Quantized local positions and packed attributes |
| Nvidium | 16-byte terrain vertex | Compact decoding combined with GPU-driven mesh processing |
| Voxy | 8-byte emitted terrain quad record | Represent larger approximate surfaces through hierarchical LOD |

The respective implementations define these sizes directly. [Sodium encoder](https://raw.githubusercontent.com/CaffeineMC/sodium/dev/common/src/main/java/net/caffeinemc/mods/sodium/client/render/chunk/vertex/format/impl/CompactChunkVertex.java), [Nvidium encoder](https://raw.githubusercontent.com/MCRcortex/nvidium/dev/src/main/java/me/cortex/nvidium/sodiumCompat/NvidiumCompactChunkVertex.java), [Voxy mesher](https://raw.githubusercontent.com/MCRcortex/voxy/dev/src/main/java/me/cortex/voxy/client/core/rendering/building/RenderDataFactory.java)

- **Sodium layout:** 8 bytes encode three 20-bit positions; 4 bytes hold color; 4 hold UVs; 4 pack light, material, and section data. UV encoding includes a bias direction to control texture bleeding. [CompactChunkVertex](https://raw.githubusercontent.com/CaffeineMC/sodium/dev/common/src/main/java/net/caffeinemc/mods/sodium/client/render/chunk/vertex/format/impl/CompactChunkVertex.java)

- Its position mapping covers a 32-unit local range with an origin offset. Local quantization plus a separate section transform is the useful design principle; large world coordinates should not consume full precision in every vertex. [Sodium position encoding](https://raw.githubusercontent.com/CaffeineMC/sodium/dev/common/src/main/java/net/caffeinemc/mods/sodium/client/render/chunk/vertex/format/impl/CompactChunkVertex.java)

- **Nvidium layout:** 16-bit local X/Y/Z, packed material/light, RGB color with another light byte, and two packed UV components fit in four 32-bit words. That is **64 bytes for four vertices**, before indexing/other metadata. [Nvidium encoding](https://raw.githubusercontent.com/MCRcortex/nvidium/dev/src/main/java/me/cortex/nvidium/sodiumCompat/NvidiumCompactChunkVertex.java)

- Nvidium uses NVIDIA-specific mesh-shader functionality for a nearly GPU-driven terrain pipeline. Its public description also says it disables itself while Iris shaders are active, so its terrain results do not establish performance with SEUS-like lighting. [Nvidium description](https://modrinth.com/mod/nvidium)

- **Voxy** converts explored/imported terrain into LOD data. Its mesher emits packed 64-bit records containing face/position/extents plus model, light, and biome information; section data uses 32³ cells and hierarchical child state. [Mesher](https://raw.githubusercontent.com/MCRcortex/voxy/dev/src/main/java/me/cortex/voxy/client/core/rendering/building/RenderDataFactory.java), [WorldSection](https://raw.githubusercontent.com/MCRcortex/voxy/dev/src/main/java/me/cortex/voxy/common/world/WorldSection.java)

- Voxy’s eight-byte quad is not equivalent to eight bytes for an arbitrary detailed Minecraft model. The savings depend on its surface representation, model tables, merging, and reduced distant detail. [Voxy representation](https://raw.githubusercontent.com/MCRcortex/voxy/dev/src/main/java/me/cortex/voxy/client/core/rendering/building/RenderDataFactory.java)

**Published performance claims and their limits**

- **Sodium:** its official gallery describes **300% or greater FPS improvements** for historical 0.4.1/0.5.2 examples. These are whole-renderer claims, not an isolated compact-format speedup or a Minecraft 26.3 Metal comparison. [Official Sodium gallery](https://modrinth.com/mod/sodium/gallery)

- **Nvidium:** its official gallery labels a Hermitcraft season-eight demonstration **80 FPS at 4K**. The caption does not provide enough hardware/baseline information to compute a portable speedup. [Official Nvidium gallery](https://modrinth.com/mod/nvidium/gallery)

- **Voxy:** its official gallery reports Hermitcraft season-nine rendering at **1080p, 30 FPS on Intel Iris Xe**, and separately shows a **512-chunk-distance** scene. These demonstrate scalability, not shader-quality lighting parity. [Official Voxy gallery](https://modrinth.com/mod/voxy/gallery)

## Implications for our renderer

Ranked by expected project impact; these rankings are recommendations, not benchmark results.

1. **Bound distant-world work with LOD and streaming.** Keep detailed nearby terrain, progressively simplified distant surfaces, and fixed upload/residency budgets. Apply separate distance/detail policies to camera visibility, shadows, and GI. [Voxy’s approach](https://modrinth.com/mod/voxy)

2. **Separate output resolution from shading resolution.** Establish a dynamic-resolution/MetalFX path early; 70% linear resolution means roughly 49% as many scene pixels before reconstruction. Preserve native-resolution UI and test foliage, motion, and water carefully. [MetalFX](https://developer.apple.com/videos/play/wwdc2022/10103/)

3. **Make tile-resident deferred lighting the baseline.** Store only post-processing necessities, then compare a visibility buffer against that baseline. This targets bandwidth without assuming geometry reconstruction is always cheaper. [Apple deferred rendering](https://developer.apple.com/videos/play/wwdc2019/601/)

4. **Cull hierarchically and submit from the GPU.** Begin with section/meshlet culling and ICBs; add two-phase HZB where measured savings repay depth stores and synchronization. Compare mesh stages against compact vertex pulling on identical terrain. [GPU ICB sample](https://developer.apple.com/documentation/metal/encoding-indirect-command-buffers-on-the-gpu)

5. **Use bounded DDGI updates for bounced light.** Combine responsive shadowed direct lighting with cascaded probes and edit-driven invalidation. Reserve ReSTIR GI and full 3D radiance cascades for later quality/performance experiments. [Production DDGI](https://arxiv.org/abs/2009.10796)

6. **Spend RT selectively on water and difficult visibility.** Use SSR/sky fallback, bounded rays, persistent static acceleration structures, and separately updated dynamic geometry. Include builds and denoising in the RT budget. [Hybrid reflections](https://gpuopen.com/manuals/fidelityfx_sdk/samples/hybrid-reflections/), [Metal AS guidance](https://developer.apple.com/videos/play/wwdc2023/10128/)

7. **Implement Hillaire sky and correct EDR early.** They address large, visually important parts of the target appearance with controllable complexity. Give volumetric clouds their own adjustable budget. [Sky implementation](https://github.com/sebh/UnrealEngineSkyAtmosphere), [EDR tone mapping](https://developer.apple.com/documentation/metal/performing-your-own-tone-mapping)

8. **Judge success by sustained complete-frame delivery.** Measure CPU, GPU, streaming, AS maintenance, and presentation deadlines, including camera turns, forests, edits, and large water views. Preserve p99/stutter evidence; isolated terrain timings cannot establish locked 120 Hz. [Apple frame scheduling](https://developer.apple.com/documentation/quartzcore/cametaldisplaylink), [profiling guidance](https://developer.apple.com/videos/play/wwdc2019/606/)

