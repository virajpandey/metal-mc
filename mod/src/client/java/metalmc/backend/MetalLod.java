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

    /** {state (0 idle, 1 building, 2 ready, 3 failed), sections, quads}. */
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
