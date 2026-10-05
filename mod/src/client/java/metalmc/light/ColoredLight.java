package metalmc.light;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import metalmc.backend.MetalColoredLight;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientChunkEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.core.BlockPos;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.world.level.ChunkPos;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.chunk.LevelChunk;
import net.minecraft.world.level.chunk.LevelChunkSection;
import net.minecraft.world.level.chunk.PalettedContainer;
import org.lwjgl.system.MemoryUtil;

/**
 * Colored block light (METALMC_EXP=lit,coloredlight; Sources/MetalMCNative/ColoredLight.swift): feeds the native light
 * volume the overworld's blocks as vanilla describes them. Every chunk the client loads is sent whole (each block's light
 * dampening, light emission and color class: what the volume's flood fill needs), and every block change after that
 * alters light (ClientLevelLightMixin). The client thread only copies a loaded chunk's sections; a worker thread converts
 * them. A new level (world or dimension) starts over.
 */
public final class ColoredLight implements ClientModInitializer {
    private static final int MIN_SECTION = -4;   // the native store's sections: -4..19 (y -64..319)
    private static final int SECTIONS = 24;

    private static volatile int generation;
    private static ClientLevel currentLevel;   // client thread
    /** Chunk conversions queued and not yet sent, by chunk: a block change in one of them re-captures the chunk. */
    private static final ConcurrentHashMap<Long, AtomicInteger> PENDING = new ConcurrentHashMap<>();
    private static final ExecutorService WORKER = Executors.newSingleThreadExecutor(r -> {
        Thread t = new Thread(r, "MetalMC colored light");
        t.setDaemon(true);
        t.setPriority(Thread.MIN_PRIORITY);
        return t;
    });
    /** Color class by block (either thread). */
    private static final ConcurrentHashMap<Block, Integer> CLASSES = new ConcurrentHashMap<>();

    // Worker thread only: code + 1 by block state id (0: not yet), and the native buffers.
    private static int[] codeByState = new int[1 << 16];
    private static long ysAddr;
    private static long codesAddr;

    @Override
    public void onInitializeClient() {
        if (MetalColoredLight.ENABLED) ClientChunkEvents.CHUNK_LOAD.register(ColoredLight::capture);
    }

    private static long chunkKey(int cx, int cz) {
        return ((long) cx << 32) | (cz & 0xFFFFFFFFL);
    }

    /** Client thread: a new level starts a new generation (the native side forgets the old one's blocks). */
    private static void checkLevel(ClientLevel level) {
        if (level == currentLevel) return;
        currentLevel = level;
        generation++;
        PENDING.clear();
        MetalColoredLight.reset(generation);
    }

    private static void capture(ClientLevel level, LevelChunk chunk) {
        if (!MetalColoredLight.enabled() || level.dimension() != Level.OVERWORLD) return;
        checkLevel(level);
        queue(level, chunk);
    }

    private static void queue(ClientLevel level, LevelChunk chunk) {
        LevelChunkSection[] sections = chunk.getSections();
        int first = level.getMinSectionY();
        @SuppressWarnings("unchecked")
        PalettedContainer<BlockState>[] copies = new PalettedContainer[SECTIONS];
        for (int i = 0; i < sections.length; i++) {
            int slot = first + i - MIN_SECTION;
            if (slot < 0 || slot >= SECTIONS || sections[i] == null || sections[i].hasOnlyAir()) continue;
            copies[slot] = sections[i].getStates().copy();
        }
        ChunkPos pos = chunk.getPos();
        int cx = pos.x(), cz = pos.z();
        long key = chunkKey(cx, cz);
        AtomicInteger pending = PENDING.computeIfAbsent(key, k -> new AtomicInteger());
        pending.incrementAndGet();
        int gen = generation;
        WORKER.execute(() -> {
            try {
                if (gen == generation) convert(gen, cx, cz, copies);
            } catch (Throwable t) {
                System.err.println("[metalmc-light] colored light: chunk " + cx + ", " + cz + " failed: " + t);
            } finally {
                pending.decrementAndGet();
            }
        });
    }

    private static void convert(int gen, int cx, int cz, PalettedContainer<BlockState>[] copies) {
        if (codesAddr == 0) {
            ysAddr = MemoryUtil.nmemAlloc(SECTIONS * 4L);
            codesAddr = MemoryUtil.nmemAlloc(SECTIONS * 4096L * 2);
        }
        int count = 0;
        for (int s = 0; s < SECTIONS; s++) {
            PalettedContainer<BlockState> c = copies[s];
            if (c == null) continue;
            long base = codesAddr + (long) count * 8192;
            BlockState last = null;
            int code = 0;
            boolean any = false;
            for (int y = 0; y < 16; y++) {
                for (int z = 0; z < 16; z++) {
                    for (int x = 0; x < 16; x++) {
                        BlockState state = c.get(x, y, z);
                        if (state != last) {
                            last = state;
                            code = cachedCode(state);
                        }
                        if (code != 0) any = true;
                        MemoryUtil.memPutShort(base + 2L * ((y << 8) | (z << 4) | x), (short) code);
                    }
                }
            }
            if (!any) continue;   // nothing that dampens or gives off light: air to the volume
            MemoryUtil.memPutInt(ysAddr + 4L * count, MIN_SECTION + s);
            count++;
        }
        MetalColoredLight.chunk(gen, cx, cz, count, ysAddr, codesAddr);
    }

    /** Worker thread: the code of a block state, cached by its id. */
    private static int cachedCode(BlockState state) {
        int id = Block.BLOCK_STATE_REGISTRY.getId(state);
        if (id < 0) return code(state);
        if (id >= codeByState.length) codeByState = java.util.Arrays.copyOf(codeByState, Math.max(id + 1, codeByState.length * 2));
        int c = codeByState[id];
        if (c == 0) {
            c = code(state) + 1;
            codeByState[id] = c;
        }
        return c - 1;
    }

    /** A block state's code for the volume: light dampening (0-15, 15 opaque) | emission << 4 | color class << 8. */
    private static int code(BlockState state) {
        int damp = Math.max(Math.min(15, Math.max(0, state.getLightDampening())), state.isSolidRender() ? 15 : 0);
        int emission = Math.min(15, Math.max(0, state.getLightEmission()));
        int cls = emission > 0 ? CLASSES.computeIfAbsent(state.getBlock(),
            b -> MetalColoredLight.classify(BuiltInRegistries.BLOCK.getKey(b).toString())) : 0;
        return damp | emission << 4 | (cls & 63) << 8;
    }

    /**
     * Client thread (ClientLevelLightMixin, as the level announces a block change): sends the block's new code if its light
     * changed. If a conversion of its chunk is still queued (captured before the change), the chunk is captured again
     * instead, so the change isn't overwritten.
     */
    public static void blockChanged(ClientLevel level, BlockPos pos, BlockState oldState, BlockState newState) {
        if (level != currentLevel || !MetalColoredLight.enabled()) return;
        int was = code(oldState), now = code(newState);
        if (was == now) return;
        int cx = pos.getX() >> 4, cz = pos.getZ() >> 4;
        AtomicInteger pending = PENDING.get(chunkKey(cx, cz));
        if (pending != null && pending.get() > 0) {
            queue(level, level.getChunk(cx, cz));
            return;
        }
        MetalColoredLight.block(generation, pos.getX(), pos.getY(), pos.getZ(), now);
    }

    /** Render thread, each frame before lit mode's relight (GameRendererLodMixin): the volume's work for this camera. */
    public static void frame(double x, double y, double z) {
        if (!MetalColoredLight.enabled()) return;
        Minecraft mc = Minecraft.getInstance();
        if (mc.level == null || mc.level.dimension() != Level.OVERWORLD) return;
        checkLevel(mc.level);
        MetalColoredLight.frame(x, y, z);
    }
}
