package metalmc.backend;

import java.nio.ByteBuffer;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import org.joml.Matrix4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

/** Bridge from the LOD feature (metalmc.lod) to the native LOD renderer. Only works on the Metal backend. */
public final class MetalLod {
    private MetalLod() {
    }

    public static boolean available() {
        return MetalDevice.current != null;
    }

    /** Starts building LOD for a world save in the background. */
    public static boolean open(String worldDir, int farBlocks, int centerX, int centerZ) {
        if (!available()) return false;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            ByteBuffer path = stack.UTF8(worldDir);
            return Mtl.lodOpen(MemoryUtil.memAddress(path), farBlocks, centerX, centerZ) == 1;
        }
    }

    /**
     * Opens the LOD with a single-player save's region files ("" for none) and/or a directory where live
     * chunks are saved between sessions ("" to keep them in memory only).
     */
    public static boolean open2(String worldDir, String storeDir, int farBlocks, int centerX, int centerZ) {
        if (!available()) return false;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            ByteBuffer world = stack.UTF8(worldDir);
            ByteBuffer store = stack.UTF8(storeDir);
            return Mtl.lodOpen2(MemoryUtil.memAddress(world), MemoryUtil.memAddress(store), farBlocks, centerX, centerZ) == 1;
        }
    }

    /** LOD material id for a block name such as "minecraft:stone". Any thread. */
    public static int classify(String blockName) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            return Mtl.lodClassify(MemoryUtil.memAddress(stack.UTF8(blockName)));
        }
    }

    /** Biome tint class for a biome name such as "minecraft:savanna". Any thread. */
    public static int tintIndex(String biomeName) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            return Mtl.lodTintIndex(MemoryUtil.memAddress(stack.UTF8(biomeName)));
        }
    }

    /**
     * Hands one chunk to the LOD: {@code blocks} holds 16 x 16 x 384 material ids (y from the world bottom,
     * then z, then x) and {@code tints} the 4 x 4 surface biome tint classes (z * 4 + x). Both are native
     * addresses. Any thread.
     */
    public static void ingest(int chunkX, int chunkZ, long blocks, long tints) {
        Mtl.lodIngest(chunkX, chunkZ, blocks, tints);
    }

    /** Chunk sections (SectionPos.asLong) hidden in the newest completed occlusion test. Render thread. */
    public static long[] occlusionHidden(double camX, double camY, double camZ, int max) {
        if (!available()) return new long[0];
        try (MemoryStack stack = MemoryStack.stackPush()) {
            DoubleBuffer cam = stack.mallocDouble(3);
            cam.put(0, camX).put(1, camY).put(2, camZ);
            long out = MemoryUtil.nmemAlloc(8L * max);
            try {
                int n = Mtl.occHidden(MemoryUtil.memAddress(cam), out, max);
                long[] keys = new long[n];
                for (int i = 0; i < n; i++) keys[i] = MemoryUtil.memGetLong(out + 8L * i);
                return keys;
            } finally {
                MemoryUtil.nmemFree(out);
            }
        }
    }

    /** Box-tests chunk sections for occlusion in the current render pass (the main world pass). */
    public static void occlusionTest(Matrix4fc projection, Matrix4fc view, double camX, double camY, double camZ, long[] keys, int count) {
        MetalDevice device = MetalDevice.current;
        if (device == null) return;
        MetalRenderPass pass = device.encoder().currentRenderPass();
        if (pass == null) return;
        long keysAddr = MemoryUtil.nmemAlloc(8L * count);
        try (MemoryStack stack = MemoryStack.stackPush()) {
            for (int i = 0; i < count; i++) MemoryUtil.memPutLong(keysAddr + 8L * i, keys[i]);
            FloatBuffer p = stack.mallocFloat(32);
            projection.get(0, p);
            view.get(16, p);
            DoubleBuffer cam = stack.mallocDouble(3);
            cam.put(0, camX).put(1, camY).put(2, camZ);
            Mtl.occTest(MemoryUtil.memAddress(p), MemoryUtil.memAddress(cam), keysAddr, count);
        } finally {
            MemoryUtil.nmemFree(keysAddr);
        }
        pass.restoreAfterExternalDraw();
    }

    /** Stops streaming and releases the LOD (the player left the world). */
    public static void close() {
        if (available()) Mtl.lodClose();
    }

    /** Where the player is: the LOD keeps its finest level around this position. */
    public static void center(int x, int z, int vanillaRadius) {
        if (available()) Mtl.lodCenter(x, z, vanillaRadius);
    }

    /** {state (0 none, 1 building, 2 has nodes), nodes, quads}. */
    public static long[] status() {
        if (!available()) return new long[3];
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer out = stack.callocLong(3);
            Mtl.lodStatus(MemoryUtil.memAddress(out));
            return new long[]{out.get(0), out.get(1), out.get(2)};
        }
    }

    /**
     * Draws LOD into the currently open render pass (the main world pass), then restores Minecraft's
     * pipeline state. Returns the number of LOD draws.
     */
    public static int draw(Matrix4fc projection, Matrix4fc view, double camX, double camY, double camZ,
                           float fogR, float fogG, float fogB, float fogA, float envStart, float envEnd,
                           float rdStart, float rdEnd, float discardRadius, float sky) {
        MetalDevice device = MetalDevice.current;
        if (device == null) return 0;
        MetalRenderPass pass = device.encoder().currentRenderPass();
        if (pass == null) return 0;
        int draws;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer p = stack.mallocFloat(48);
            projection.get(0, p);
            view.get(16, p);
            p.put(32, fogR).put(33, fogG).put(34, fogB).put(35, fogA);
            p.put(36, envStart).put(37, envEnd).put(38, rdStart).put(39, rdEnd).put(40, discardRadius).put(41, sky);
            DoubleBuffer cam = stack.mallocDouble(3);
            cam.put(0, camX).put(1, camY).put(2, camZ);
            draws = Mtl.lodDraw(MemoryUtil.memAddress(p), MemoryUtil.memAddress(cam));
        }
        pass.restoreAfterExternalDraw();
        return draws;
    }
}
