package metalmc.lod;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import metalmc.backend.MetalLod;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientChunkEvents;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.core.Holder;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.biome.Biome;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.LevelChunk;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.chunk.PalettedContainer;
import org.lwjgl.system.MemoryUtil;

/**
 * Live chunk ingestion for the LOD. Every overworld chunk the client loads, and again when it unloads
 * (which captures edits made while it was loaded), is handed to the native LOD. The client thread only
 * copies the chunk's non-empty block sections and reads its surface biomes. A worker thread converts
 * the blocks to LOD material ids. This keeps the LOD current without waiting for the server to save,
 * and it's the only data source in multiplayer.
 */
public final class LiveIngest {
    private LiveIngest() {
    }

    /** Matches the native layout: overworld sections -4..19 (y -64..319). */
    private static final int MIN_SECTION = -4;
    private static final int SECTIONS = 24;
    private static final int BLOCK_BYTES = 16 * 16 * 16 * SECTIONS;
    private static final int MAX_QUEUED = 2048;
    private static final int SURFACE_QUART_Y = 96 >> 2;   // biome sample height, like the region reader

    private static volatile int generation;              // bumped when the LOD closes or reopens
    private static volatile boolean enabled;
    private static final AtomicInteger QUEUED = new AtomicInteger();
    private static final ExecutorService WORKER = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "MetalMC LOD ingest");
        t.setDaemon(true);
        t.setPriority(Thread.MIN_PRIORITY);
        return t;
    });

    // Client thread only.
    private static final Map<Holder<Biome>, Byte> TINTS = new HashMap<>();

    // Worker thread only: material id + 1 per block state id (0 = not classified yet), and native buffers.
    private static byte[] matByState = new byte[1 << 16];
    private static long blocksAddr;
    private static long tintsAddr;

    static void register() {
        ClientChunkEvents.CHUNK_LOAD.register(LiveIngest::capture);
        ClientChunkEvents.CHUNK_UNLOAD.register(LiveIngest::capture);
    }

    /** Called when the native LOD opens (true) or closes (false). Pending chunks from before are dropped. */
    static void setEnabled(boolean on) {
        generation++;
        enabled = on;
    }

    private static void capture(ClientLevel level, LevelChunk chunk) {
        if (!enabled || level.dimension() != Level.OVERWORLD) return;
        if (QUEUED.get() >= MAX_QUEUED) return;   // the worker is far behind; the unload will catch up
        LevelChunkSection[] sections = chunk.getSections();
        int first = level.getMinSectionY();
        @SuppressWarnings("unchecked")
        PalettedContainer<BlockState>[] copies = new PalettedContainer[SECTIONS];
        // All-air sections stay null; an all-air chunk still replaces whatever the LOD had there.
        for (int i = 0; i < sections.length; i++) {
            int slot = first + i - MIN_SECTION;
            if (slot < 0 || slot >= SECTIONS || sections[i] == null || sections[i].hasOnlyAir()) continue;
            copies[slot] = sections[i].getStates().copy();
        }
        ChunkPos pos = chunk.getPos();
        byte[] tints = new byte[16];
        for (int z = 0; z < 4; z++) {
            for (int x = 0; x < 4; x++) {
                Holder<Biome> b = chunk.getNoiseBiome((pos.x() << 2) + x, SURFACE_QUART_Y, (pos.z() << 2) + z);
                tints[z * 4 + x] = TINTS.computeIfAbsent(b, h -> (byte) h.unwrapKey()
                    .map(k -> MetalLod.tintIndex(k.identifier().toString())).orElse(0).intValue());
            }
        }
        int gen = generation;
        QUEUED.incrementAndGet();
        WORKER.execute(() -> {
            try {
                if (gen == generation && enabled) convert(pos.x(), pos.z(), copies, tints);
            } catch (Throwable t) {
                System.err.println("[metalmc-lod] live ingest failed for chunk " + pos + ": " + t);
            } finally {
                QUEUED.decrementAndGet();
            }
        });
    }

    private static void convert(int cx, int cz, PalettedContainer<BlockState>[] copies, byte[] tints) {
        if (blocksAddr == 0) {
            blocksAddr = MemoryUtil.nmemAlloc(BLOCK_BYTES);
            tintsAddr = MemoryUtil.nmemAlloc(16);
        }
        MemoryUtil.memSet(blocksAddr, 0, BLOCK_BYTES);
        for (int s = 0; s < SECTIONS; s++) {
            PalettedContainer<BlockState> c = copies[s];
            if (c == null) continue;
            long base = blocksAddr + (long) s * 4096;
            BlockState last = null;
            byte lastMat = 0;
            for (int y = 0; y < 16; y++) {
                for (int z = 0; z < 16; z++) {
                    for (int x = 0; x < 16; x++) {
                        BlockState state = c.get(x, y, z);
                        if (state != last) {
                            last = state;
                            lastMat = material(state);
                        }
                        if (lastMat != 0) MemoryUtil.memPutByte(base + (y << 8 | z << 4 | x), lastMat);
                    }
                }
            }
        }
        for (int i = 0; i < 16; i++) MemoryUtil.memPutByte(tintsAddr + i, tints[i]);
        MetalLod.ingest(cx, cz, blocksAddr, tintsAddr);
    }

    private static byte material(BlockState state) {
        int id = Block.BLOCK_STATE_REGISTRY.getId(state);
        if (id < 0) return 0;
        if (id >= matByState.length) matByState = java.util.Arrays.copyOf(matByState, Math.max(id + 1, matByState.length * 2));
        byte m = matByState[id];
        if (m == 0) {
            m = (byte) (MetalLod.classify(BuiltInRegistries.BLOCK.getKey(state.getBlock()).toString()) + 1);
            matByState[id] = m;
        }
        return (byte) (m - 1);
    }
}
