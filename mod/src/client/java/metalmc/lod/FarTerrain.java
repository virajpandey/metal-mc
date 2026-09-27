package metalmc.lod;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import metalmc.backend.MetalLod;
import net.minecraft.core.Holder;
import net.minecraft.core.registries.Registries;
import net.minecraft.resources.Identifier;
import net.minecraft.resources.ResourceKey;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerLevel;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.biome.BiomeResolver;
import net.minecraft.world.level.chunk.ChunkGenerator;
import net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator;
import net.minecraft.world.level.levelgen.RandomState;
import net.minecraft.world.level.levelgen.densityfunction.DensityFunction;
import net.minecraft.world.level.levelgen.densityfunction.DensitySampler;
import net.minecraft.world.level.levelgen.densityfunction.SamplerContext;

/**
 * Far terrain from the world generator (single-player, lod.generate). The LOD asks for nodes of levels 3 and
 * up that lack real chunks (MetalLod.farWanted, nearest first). For each node this samples 256 x 256 columns,
 * one per voxel column of the node's level, with the generator's own noise:
 * <ul>
 * <li>Ground height: start above the router's cheap surface estimate (chunk_surface_level, which runs about
 * 14 blocks low) and march down the terrain-shape density (overworld/sloped_cheese: no caves, aquifers or
 * structures) until it's solid, then bisect. Measured on the 4 km test world: 1.3 blocks from the
 * generator's exact height on average, 96% within 4 blocks, 110 us per column (6x faster than the exact
 * column scan).</li>
 * <li>Biome at that height, which the native side turns into surface materials, tints and tree canopy
 * (LodFar.swift).</li>
 * </ul>
 * Runs on a few low-priority threads; the samplers are the ones vanilla's parallel world generation shares.
 */
public final class FarTerrain {
    private FarTerrain() {
    }

    private static final int THREADS = 3;
    private static final int COLUMNS = 256;
    private static volatile Thread loop;
    private static volatile boolean running;
    private static final ConcurrentHashMap<Holder<Biome>, Integer> BIOME_IDS = new ConcurrentHashMap<>();

    public static synchronized void start(MinecraftServer server) {
        stop();
        ServerLevel level = server.overworld();
        ChunkGenerator gen = level.getChunkSource().getGenerator();
        if (!(gen instanceof NoiseBasedChunkGenerator noise)) {
            System.out.println("[metalmc-lod] far terrain: generator " + gen.getClass().getSimpleName() + " isn't noise-based; off");
            return;
        }
        RandomState rs = level.getChunkSource().randomState();
        DensityFunction cheeseFn = server.registryAccess().lookupOrThrow(Registries.DENSITY_FUNCTION)
            .get(ResourceKey.create(Registries.DENSITY_FUNCTION, Identifier.withDefaultNamespace("overworld/sloped_cheese")))
            .map(Holder::value).orElse(null);
        if (cheeseFn == null) {
            System.out.println("[metalmc-lod] far terrain: no overworld/sloped_cheese; off");
            return;
        }
        DensitySampler surface = rs.getSampler(noise.generatorSettings().value().noiseRouter().chunkSurfaceLevel());
        DensitySampler cheese = rs.getSampler(cheeseFn);
        int minY = level.getMinY(), maxY = level.getMaxY();
        BIOME_IDS.clear();
        // Cache per save and seed, outside the save: <game dir>/metalmc/lod/far/<save>-<seed>.
        java.nio.file.Path save = server.getWorldPath(net.minecraft.world.level.storage.LevelResource.ROOT).toAbsolutePath().normalize();
        String name = (save.getFileName() == null ? "world" : save.getFileName().toString()).replaceAll("[^A-Za-z0-9._-]", "_");
        java.nio.file.Path cache = net.minecraft.client.Minecraft.getInstance().gameDirectory.toPath()
            .resolve("metalmc").resolve("lod").resolve("far").resolve(name + "-" + Long.toHexString(level.getSeed()));
        MetalLod.farCache(cache.toAbsolutePath().toString());
        running = true;
        Thread t = new Thread(() -> run(gen, rs, surface, cheese, minY, maxY), "metalmc-far-terrain");
        t.setDaemon(true);
        t.setPriority(Thread.MIN_PRIORITY);
        loop = t;
        t.start();
        System.out.println("[metalmc-lod] far terrain: generating from the world's noise");
    }

    public static synchronized void stop() {
        running = false;
        Thread t = loop;
        loop = null;
        if (t != null) t.interrupt();
    }

    private static void run(ChunkGenerator gen, RandomState rs, DensitySampler surface, DensitySampler cheese, int minY, int maxY) {
        ExecutorService pool = Executors.newFixedThreadPool(THREADS, r -> {
            Thread t = new Thread(r, "metalmc-far-terrain-worker");
            t.setDaemon(true);
            t.setPriority(Thread.MIN_PRIORITY);
            return t;
        });
        ThreadLocal<BiomeResolver> resolvers = ThreadLocal.withInitial(() -> gen.getBiomeSource().createUncachedResolver(rs));
        long nodes = 0, columns = 0, nanos = 0;
        try {
            while (running) {
                int[] wanted = MetalLod.farWanted(THREADS * 2);
                if (wanted.length == 0) {
                    Thread.sleep(2000);
                    continue;
                }
                long t0 = System.nanoTime();
                java.util.List<Future<?>> jobs = new java.util.ArrayList<>();
                for (int i = 0; i + 2 < wanted.length; i += 3) {
                    int lv = wanted[i], nx = wanted[i + 1], nz = wanted[i + 2];
                    jobs.add(pool.submit(() -> generate(lv, nx, nz, surface, cheese, resolvers.get(), minY, maxY)));
                }
                for (Future<?> f : jobs) f.get();
                nodes += jobs.size();
                columns += (long) jobs.size() * COLUMNS * COLUMNS;
                nanos += System.nanoTime() - t0;
                System.out.println(String.format("[metalmc-lod] far terrain: %d nodes (%d columns) in %.1f s, %.0f us per column per thread",
                    nodes, columns, nanos / 1e9, nanos / 1e3 * THREADS / columns));
            }
        } catch (InterruptedException e) {
            // stopping
        } catch (Exception e) {
            System.out.println("[metalmc-lod] far terrain stopped: " + e);
        } finally {
            pool.shutdownNow();
        }
    }

    private static void generate(int level, int nx, int nz, DensitySampler surface, DensitySampler cheese, BiomeResolver biomes,
                                 int minY, int maxY) {
        if (!running) return;
        SamplerContext ctx = SamplerContext.EMPTY_UNCACHED;
        int s = 1 << level, size = COLUMNS * s;
        int x0 = nx * size, z0 = nz * size;
        // March in steps well under a voxel and stop once within a quarter voxel: enough to put the ground
        // in the right voxel at this level.
        int step = Math.max(8, s / 2), precision = Math.max(1, s / 4);
        short[] heights = new short[COLUMNS * COLUMNS];
        short[] ids = new short[COLUMNS * COLUMNS];
        for (int j = 0; j < COLUMNS; j++) {
            if (!running) return;
            for (int i = 0; i < COLUMNS; i++) {
                int x = x0 + i * s + s / 2, z = z0 + j * s + s / 2;
                int est = (int) surface.sampleValue(ctx, x, 0, z);
                int y = Math.min(maxY - 1, est + 96);
                while (y > minY && cheese.sampleValue(ctx, x, y, z) <= 0) y -= step;
                int lo = y, hi = Math.min(maxY, y + step);
                while (hi - lo > precision) {
                    int mid = (lo + hi) >> 1;
                    if (cheese.sampleValue(ctx, x, mid, z) > 0) lo = mid; else hi = mid;
                }
                int h = Math.max(minY, lo + 1);
                heights[j * COLUMNS + i] = (short) h;
                Holder<Biome> b = biomes.getNoiseBiome(x >> 2, Math.min(maxY - 1, h) >> 2, z >> 2);
                ids[j * COLUMNS + i] = (short) biomeId(b).intValue();
            }
        }
        MetalLod.farPut(level, nx, nz, heights, ids);
    }

    private static Integer biomeId(Holder<Biome> b) {
        return BIOME_IDS.computeIfAbsent(b, h -> Math.max(0, MetalLod.farBiome(h.getRegisteredName())));
    }
}
