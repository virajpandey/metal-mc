package metalmc.bench;

import java.util.concurrent.Semaphore;
import java.util.concurrent.atomic.AtomicInteger;
import net.minecraft.client.Minecraft;
import net.minecraft.server.MinecraftServer;
import net.minecraft.server.level.ServerChunkCache;
import net.minecraft.server.level.TicketType;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.chunk.status.ChunkStatus;

/**
 * Chunk pregenerator for building large test worlds (-Dmetalmc.pregen=RADIUS_IN_CHUNKS). Requests every
 * chunk in a square around chunk (0, 0), nearest first, with a bounded number in flight. Each request
 * uses a short-lived ticket, so generated chunks unload and are written to the region files, the same
 * way Chunky pregenerates worlds for Voxy.
 */
final class Pregen {
    private Pregen() {}

    static final int RADIUS = Integer.getInteger("metalmc.pregen", 0);
    private static final int IN_FLIGHT = 256;
    private static final int MAX_LOADED = 6000;

    private static volatile boolean done;
    private static final AtomicInteger completed = new AtomicInteger();
    private static final AtomicInteger failed = new AtomicInteger();

    private static int skipped;
    private static final java.util.Map<Long, int[]> headers = new java.util.HashMap<>();

    /** True if the region file's offset table has an entry for this chunk (so it was generated before). */
    private static boolean existsOnDisk(java.nio.file.Path regionDir, int x, int z) {
        long key = ((long) (x >> 5) << 32) | ((z >> 5) & 0xffffffffL);
        int[] table = headers.computeIfAbsent(key, k -> {
            java.nio.file.Path f = regionDir.resolve("r." + (x >> 5) + "." + (z >> 5) + ".mca");
            int[] t = new int[1024];
            try (java.io.DataInputStream in = new java.io.DataInputStream(new java.io.BufferedInputStream(java.nio.file.Files.newInputStream(f)))) {
                for (int i = 0; i < 1024; i++) t[i] = in.readInt();
            } catch (java.io.IOException e) {
                // No region file yet: nothing generated there.
            }
            return t;
        });
        return table[((z & 31) << 5) | (x & 31)] != 0;
    }

    static boolean done() {
        return done;
    }

    static void start(Minecraft mc) {
        MinecraftServer server = mc.getSingleplayerServer();
        if (server == null) {
            Bench.log("pregen: no integrated server");
            done = true;
            return;
        }
        ServerChunkCache cache = server.overworld().getChunkSource();
        int total = (2 * RADIUS + 1) * (2 * RADIUS + 1);
        java.nio.file.Path regionDir = server.getWorldPath(net.minecraft.world.level.storage.LevelResource.ROOT)
            .resolve("dimensions/minecraft/overworld/region");
        Thread t = new Thread(() -> {
            long t0 = System.nanoTime();
            Semaphore slots = new Semaphore(IN_FLIGHT);
            try {
                // Square rings outward from the center, so the explored area grows evenly.
                for (int r = 0; r <= RADIUS; r++) {
                    for (int x = -r; x <= r; x++) {
                        for (int z = -r; z <= r; z++) {
                            if (Math.max(Math.abs(x), Math.abs(z)) != r) continue;
                            if (existsOnDisk(regionDir, x, z)) {
                                skipped++;
                                continue;
                            }
                            slots.acquire();
                            // Let the server unload and save finished chunks before generating more; otherwise
                            // everything stays in memory until the final save (out of memory at ~60k chunks).
                            // Chunks waiting to be saved aren't in the loaded count, so watch the heap too.
                            Runtime rt = Runtime.getRuntime();
                            while (cache.getLoadedChunksCount() > MAX_LOADED || rt.totalMemory() - rt.freeMemory() > 0.6 * rt.maxMemory()) {
                                Thread.sleep(50);
                            }
                            ChunkPos pos = new ChunkPos(x, z);
                            // Hold a non-expiring ticket until the chunk is fully generated. getChunkFuture's own
                            // ticket expires after one tick, which saves most chunks half-generated.
                            server.submit(() -> cache.addTicketWithRadius(TicketType.PLAYER_LOADING, pos, 0)).join();
                            cache.getChunkFuture(x, z, ChunkStatus.FULL, true).whenComplete((result, error) -> {
                                server.execute(() -> cache.removeTicketWithRadius(TicketType.PLAYER_LOADING, pos, 0));
                                if (error != null || result == null || !result.isSuccess()) failed.incrementAndGet();
                                int n = completed.incrementAndGet();
                                if (n % 2000 == 0) {
                                    double s = (System.nanoTime() - t0) / 1e9;
                                    Bench.log(String.format(java.util.Locale.ROOT, "pregen %d/%d chunks (%.0f/s, %d failed)", n, total, n / s, failed.get()));
                                }
                                slots.release();
                            });
                        }
                    }
                }
                slots.acquire(IN_FLIGHT);
                Bench.log("pregen: skipped " + skipped + " chunks already on disk");
                double s = (System.nanoTime() - t0) / 1e9;
                Bench.log(String.format(java.util.Locale.ROOT, "pregen finished: %d chunks in %.0f s (%.0f/s), %d failed; saving",
                    completed.get(), s, completed.get() / s, failed.get()));
                server.submit(() -> server.saveAllChunks(false, true, true)).join();
                Bench.log("pregen saved");
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            } finally {
                done = true;
            }
        }, "MetalMC pregen");
        t.setDaemon(true);
        t.start();
        Bench.log("pregen started: radius " + RADIUS + " chunks (" + total + " chunks)");
    }
}
