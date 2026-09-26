package metalmc.terrain;

import it.unimi.dsi.fastutil.longs.LongOpenHashSet;
import metalmc.backend.MetalLod;
import org.joml.Matrix4fc;

/**
 * Occlusion culling for vanilla chunk sections (Metal backend). While the draw lists are built, the sections
 * vanilla would draw farther than {@link #NEAR} blocks become test candidates; after solid terrain their
 * boxes are tested on the GPU (native SectionOcclusion.swift). Sections found fully hidden are left out of
 * the draw lists until a later test sees them again.
 *
 * Hidden sections are tested every frame, so they come back 2-3 frames after coming into view. Visible ones
 * are re-tested every {@link #VISIBLE_RETEST} frames, staggered: a visible box runs the marking shader for
 * every pixel it covers, and testing all of them every frame cost more than culling saved when looking
 * down over terrain. Render thread only.
 */
public final class SectionOcclusion {
    private SectionOcclusion() {
    }

    public static final boolean ENABLED = metalmc.MetalMCConfig.occlusionCulling();
    /** Sections whose box comes within this many blocks of the camera are never tested or skipped. */
    public static final double NEAR = 48;
    private static final int MAX = 1 << 16;
    public static final int VISIBLE_RETEST = 8;
    private static int frame;

    private static final LongOpenHashSet HIDDEN = new LongOpenHashSet();
    private static long[] candidates = new long[4096];
    private static int candidateCount;
    private static long lastKey = Long.MIN_VALUE;
    private static boolean active;

    /** Start of draw-list building: fetch the newest results and reset the candidate list. */
    public static void beginFrame(double camX, double camY, double camZ) {
        candidateCount = 0;
        lastKey = Long.MIN_VALUE;
        frame++;
        HIDDEN.clear();
        active = ENABLED && MetalLod.available();
        if (!active) return;
        long[] hidden = MetalLod.occlusionHidden(camX, camY, camZ, MAX);
        for (long k : hidden) HIDDEN.add(k);
    }

    /**
     * Called for each section draw vanilla adds. Returns true if the draw should be skipped. The section at
     * block origin (x, y, z) is recorded as a test candidate if it's far enough from the camera.
     */
    public static boolean skip(int x, int y, int z, double camX, double camY, double camZ) {
        if (!active) return false;
        double dx = Math.max(0, Math.max(x - camX, camX - (x + 16)));
        double dy = Math.max(0, Math.max(y - camY, camY - (y + 16)));
        double dz = Math.max(0, Math.max(z - camZ, camZ - (z + 16)));
        if (dx * dx + dy * dy + dz * dz < NEAR * NEAR) return false;
        long key = net.minecraft.core.SectionPos.asLong(x >> 4, y >> 4, z >> 4);
        boolean hidden = HIDDEN.contains(key);
        if (key != lastKey) {
            lastKey = key;
            boolean due = hidden || ((it.unimi.dsi.fastutil.HashCommon.mix(key) + frame) % VISIBLE_RETEST) == 0;
            if (due && candidateCount < MAX) {
                if (candidateCount == candidates.length) candidates = java.util.Arrays.copyOf(candidates, candidateCount * 2);
                candidates[candidateCount++] = key;
            }
        }
        return hidden;
    }

    /** After solid terrain, inside the main pass: test this frame's candidates. */
    public static void test(Matrix4fc projection, Matrix4fc view, double camX, double camY, double camZ) {
        if (!active || candidateCount == 0) return;
        MetalLod.occlusionTest(projection, view, camX, camY, camZ, candidates, candidateCount);
    }

    public static int hiddenCount() {
        return HIDDEN.size();
    }
}
