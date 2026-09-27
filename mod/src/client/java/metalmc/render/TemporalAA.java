package metalmc.render;

import org.joml.Matrix4f;

/**
 * Temporal anti-aliasing (config taa): each frame the level's projection is shifted by a sub-pixel jitter (an
 * 8-step Halton 2,3 sequence) and, after the level is drawn, MetalFX blends it with the previous frames
 * (MetalTaa). The LOD draws with the same jittered projection.
 */
public final class TemporalAA {
    private TemporalAA() {
    }

    public static final boolean ENABLED = metalmc.MetalMCConfig.taa() && metalmc.MetalMCConfig.metalBackend();

    /** This frame's projection before the jitter, and the jitter in pixels (x right, y up in clip space). */
    public static final Matrix4f UNJITTERED = new Matrix4f();
    public static float jitterX, jitterY;
    private static int frame;

    private static float halton(int index, int base) {
        float f = 1, r = 0;
        for (int i = index; i > 0; i /= base) {
            f /= base;
            r += f * (i % base);
        }
        return r;
    }

    /** Jitters {@code projection} in place for a {@code width} x {@code height} target. */
    public static void jitter(Matrix4f projection, int width, int height) {
        UNJITTERED.set(projection);
        int i = (frame++ & 7) + 1;
        jitterX = halton(i, 2) - 0.5f;
        jitterY = halton(i, 3) - 0.5f;
        projection.translateLocal(2f * jitterX / width, 2f * jitterY / height, 0f);
    }
}
