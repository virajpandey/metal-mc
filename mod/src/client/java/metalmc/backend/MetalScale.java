package metalmc.backend;

import com.mojang.renderpearl.api.textures.GpuTexture;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import org.joml.Matrix4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Bridge to the render scale (Sources/MetalMCNative/RenderScale.swift: a fixed METALMC_SCALE or its governor) and the
 * anti-aliasing's upscaling resolve (TaaUpscale.swift). Render thread. See metalmc.render.RenderScale.
 */
public final class MetalScale {
    private MetalScale() {
    }

    /** Resolved on first use, so nothing loads the library unless the Metal backend is running. */
    private static final class Native {
        private static final Linker LINKER = Linker.nativeLinker();
        private static final SymbolLookup LIB = SymbolLookup.libraryLookup(NativeLibrary.path(), Arena.global());

        private static MethodHandle h(String name, MemoryLayout ret, MemoryLayout... args) {
            FunctionDescriptor fd = ret == null ? FunctionDescriptor.ofVoid(args) : FunctionDescriptor.of(ret, args);
            var addr = LIB.find(name).orElseThrow(() -> new IllegalStateException("missing native symbol " + name));
            return LINKER.downcallHandle(addr, fd);
        }

        static final MethodHandle SCALE = h("mmc_render_scale", JAVA_FLOAT);
        static final MethodHandle FRAME_START = h("mmc_frame_start", null);
        static final MethodHandle TAA_SCALED = h("mmc_taa_apply_scaled", JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG,
            JAVA_FLOAT, JAVA_FLOAT, JAVA_INT);
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new IllegalStateException(t);
    }

    /** The scale to draw this frame's level at (1: the panel's resolution), once at the start of every frame. */
    public static float next() {
        if (MetalDevice.current == null) return 1f;
        try {
            return (float) Native.SCALE.invokeExact();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The start of the level, before any of its passes: work the last frame left that can run beside this frame's main pass
     * (the GI cache's deferred update, METALMC_EXP=gidefer; nothing without it).
     */
    public static void frameStart() {
        if (MetalDevice.current == null) return;
        try {
            Native.FRAME_START.invokeExact();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * The anti-aliasing for a level drawn into {@code color} and {@code depth} at a fraction of {@code out}'s size: resolves
     * it into the history at {@code out}'s size and leaves its copy into {@code out} waiting (pass folding: the hand's pass).
     * {@code projection} is the frame's projection without the jitter, {@code jitterX/Y} the jitter in the level's pixels.
     * Returns true if it ran.
     */
    public static boolean taa(GpuTexture color, GpuTexture depth, GpuTexture out, Matrix4fc projection, Matrix4fc viewRotation,
                              double camX, double camY, double camZ, float jitterX, float jitterY, boolean reset) {
        if (MetalDevice.current == null || !(color instanceof MetalTexture c) || !(depth instanceof MetalTexture d)
                || !(out instanceof MetalTexture o)) return false;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer p = stack.mallocFloat(32);
            projection.get(0, p);
            viewRotation.get(16, p);
            DoubleBuffer cam = stack.mallocDouble(3);
            cam.put(0, camX).put(1, camY).put(2, camZ);
            return (int) Native.TAA_SCALED.invokeExact(c.handle, d.handle, o.handle, MemoryUtil.memAddress(p), MemoryUtil.memAddress(cam),
                jitterX, jitterY, reset ? 1 : 0) == 1;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
