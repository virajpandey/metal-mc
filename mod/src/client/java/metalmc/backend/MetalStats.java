package metalmc.backend;

import java.nio.DoubleBuffer;
import org.lwjgl.system.MemoryUtil;

/** Measurements the benchmark can read when the Metal backend is active. */
public final class MetalStats {
    private MetalStats() {
    }

    /**
     * {passes, draws, blit encoders, clear passes, submits, attachment pixels, LOD draws, LOD quads, LOD CPU ns}
     * since the last call.
     */
    public static long[] takeCounters() {
        java.nio.LongBuffer buf = MemoryUtil.memAllocLong(9);
        try {
            Mtl.statsTake(MemoryUtil.memAddress(buf));
            long[] out = new long[9];
            for (int i = 0; i < 9; i++) out[i] = buf.get(i);
            return out;
        } finally {
            MemoryUtil.memFree(buf);
        }
    }

    /** Logs every pass, blit encoder, and submit natively for the next n frames. */
    public static void traceFrames(int n) {
        Mtl.traceFrames(n);
    }

    /** Per-submit GPU times in milliseconds since the last call (command buffer start to end on the GPU). */
    public static double[] takeGpuMillis() {
        int max = 400_000;
        DoubleBuffer buf = MemoryUtil.memAllocDouble(max);
        try {
            int n = Mtl.gpuTimesTake(MemoryUtil.memAddress(buf), max);
            double[] out = new double[n];
            for (int i = 0; i < n; i++) out[i] = buf.get(i) * 1000.0;
            return out;
        } finally {
            MemoryUtil.memFree(buf);
        }
    }
}
