package metalmc.lod;

import net.minecraft.core.Holder;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeResolver;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraft.world.level.levelgen.Heightmap;
import net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator;
import net.minecraft.world.level.levelgen.NoiseRouter;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.densityfunction.DensitySampler;
import net.minecraft.world.level.levelgen.densityfunction.SamplerContext;

/**
 * Measures what seed-based far terrain would cost (-PfarProbe=1): the world generator's cheap surface
 * estimate (the noise router's chunk_surface_level) against its exact noise height, and biome lookups.
 * Runs once on the server thread after the world loads and logs METALMC_FAR lines.
 */
public final class FarTerrainProbe {
    private FarTerrainProbe() {
    }

    public static void run(MinecraftServer server) {
        ServerLevel level = server.overworld();
        ChunkGenerator gen = level.getChunkSource().getGenerator();
        if (!(gen instanceof NoiseBasedChunkGenerator noise)) {
            log("generator is " + gen.getClass().getSimpleName() + ", not noise-based");
            return;
        }
        RandomState rs = level.getChunkSource().randomState();
        NoiseRouter router = noise.generatorSettings().value().noiseRouter();
        DensitySampler surface = rs.getSampler(router.chunkSurfaceLevel());
        BiomeResolver biomes = gen.getBiomeSource().createUncachedResolver(rs);
        SamplerContext ctx = SamplerContext.EMPTY_UNCACHED;

        // 1. The estimate on a 256 x 256 grid, 16 blocks apart (4 km square).
        long t0 = System.nanoTime();
        double sum = 0, min = 1e9, max = -1e9;
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                float v = surface.sampleValue(ctx, (i - 128) * 16, 0, (j - 128) * 16);
                sum += v;
                min = Math.min(min, v);
                max = Math.max(max, v);
            }
        }
        long t1 = System.nanoTime();
        log(String.format("estimate: 65536 samples in %.1f ms (%.2f us each), mean %.1f min %.1f max %.1f",
            (t1 - t0) / 1e6, (t1 - t0) / 1e3 / 65536, sum / 65536, min, max));

        // 2. Exact noise height (the generator's own column scan) at 32 x 32 points 128 blocks apart, against
        // the estimate. The estimate has no jaggedness or 3D noise, so peaks differ most.
        long t2 = System.nanoTime();
        double err = 0, errMax = 0, bias = 0;
        int n = 0, within4 = 0, within8 = 0;
        for (int i = 0; i < 32; i++) {
            for (int j = 0; j < 32; j++) {
                int x = (i - 16) * 128 + 5, z = (j - 16) * 128 + 5;
                int h = gen.getBaseHeight(x, z, Heightmap.Types.OCEAN_FLOOR_WG, level, rs);
                float p = surface.sampleValue(ctx, x, 0, z);
                double d = p - h;
                err += Math.abs(d);
                bias += d;
                errMax = Math.max(errMax, Math.abs(d));
                if (Math.abs(d) <= 4) within4++;
                if (Math.abs(d) <= 8) within8++;
                n++;
            }
        }
        long t3 = System.nanoTime();
        log(String.format("exact: 1024 columns in %.1f ms (%.1f us each); estimate - exact: mean |d| %.1f, bias %.1f, max %.1f, within 4: %d%%, within 8: %d%%",
            (t3 - t2) / 1e6, (t3 - t2) / 1e3 / 1024, err / n, bias / n, errMax, 100 * within4 / n, 100 * within8 / n));

        // 2b. Surface from the terrain-shape density alone (sloped_cheese: no caves, aquifers or beardifier):
        // march down in 8-block steps from above the estimate until it's solid, then bisect to the block.
        net.minecraft.world.level.levelgen.densityfunction.DensityFunction cheeseFn = server.registryAccess()
            .lookupOrThrow(net.minecraft.core.registries.Registries.DENSITY_FUNCTION)
            .get(net.minecraft.resources.ResourceKey.create(net.minecraft.core.registries.Registries.DENSITY_FUNCTION,
                net.minecraft.resources.Identifier.withDefaultNamespace("overworld/sloped_cheese")))
            .map(Holder::value).orElse(null);
        if (cheeseFn != null) {
            DensitySampler cheese = rs.getSampler(cheeseFn);
            long t6 = System.nanoTime();
            double e2 = 0, e2max = 0, b2 = 0;
            int w4 = 0, w8 = 0, evals = 0;
            for (int i = 0; i < 32; i++) {
                for (int j = 0; j < 32; j++) {
                    int x = (i - 16) * 128 + 5, z = (j - 16) * 128 + 5;
                    int est = (int) surface.sampleValue(ctx, x, 0, z);
                    int y = Math.min(319, est + 96);
                    evals++;
                    while (y > -64 && cheese.sampleValue(ctx, x, y, z) <= 0) { y -= 8; evals++; }
                    int lo = y, hi = Math.min(319, y + 8);   // solid at lo, air above hi (or the top)
                    while (hi - lo > 1) {
                        int mid = (lo + hi) >> 1;
                        evals++;
                        if (cheese.sampleValue(ctx, x, mid, z) > 0) lo = mid; else hi = mid;
                    }
                    int h = gen.getBaseHeight(x, z, Heightmap.Types.OCEAN_FLOOR_WG, level, rs);
                    double d = (lo + 1) - h;
                    e2 += Math.abs(d); b2 += d; e2max = Math.max(e2max, Math.abs(d));
                    if (Math.abs(d) <= 4) w4++;
                    if (Math.abs(d) <= 8) w8++;
                }
            }
            long t7 = System.nanoTime();
            // Time the search alone (the exact heights above were for comparison).
            long t8 = System.nanoTime();
            for (int i = 0; i < 32; i++) {
                for (int j = 0; j < 32; j++) {
                    int x = (i - 16) * 128 + 69, z = (j - 16) * 128 + 69;
                    int est = (int) surface.sampleValue(ctx, x, 0, z);
                    int y = Math.min(319, est + 96);
                    while (y > -64 && cheese.sampleValue(ctx, x, y, z) <= 0) y -= 8;
                    int lo = y, hi = Math.min(319, y + 8);
                    while (hi - lo > 1) { int mid = (lo + hi) >> 1; if (cheese.sampleValue(ctx, x, mid, z) > 0) lo = mid; else hi = mid; }
                }
            }
            long t9 = System.nanoTime();
            log(String.format("sloped_cheese search: %.1f us per column (%.1f evaluations), vs exact: mean |d| %.1f, bias %.1f, max %.1f, within 4: %d%%, within 8: %d%%",
                (t9 - t8) / 1e3 / 1024, evals / 1024.0, e2 / 1024, b2 / 1024, e2max, 100 * w4 / 1024, 100 * w8 / 1024));
        } else {
            log("no overworld/sloped_cheese in the registry");
        }

        // 3. Biomes at the surface on the 256 x 256 grid.
        long t4 = System.nanoTime();
        java.util.Map<String, Integer> counts = new java.util.HashMap<>();
        for (int i = 0; i < 256; i++) {
            for (int j = 0; j < 256; j++) {
                int x = (i - 128) * 16, z = (j - 128) * 16;
                Holder<Biome> b = biomes.getNoiseBiome(x >> 2, 64 >> 2, z >> 2);
                counts.merge(b.getRegisteredName(), 1, Integer::sum);
            }
        }
        long t5 = System.nanoTime();
        log(String.format("biomes: 65536 in %.1f ms (%.2f us each): %s", (t5 - t4) / 1e6, (t5 - t4) / 1e3 / 65536, counts));
    }

    private static void log(String s) {
        System.out.println("METALMC_FAR " + s);
    }
}
