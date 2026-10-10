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

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Bridge to the volumetric clouds (Sources/MetalMCNative/Clouds.swift, METALMC_EXP=clouds with sky, off by default): a
 * cumulus layer that follows the planet's curve to the horizon, lit by our sky's tables, its shadows on everything lit
 * mode relights. While they're on, vanilla's clouds aren't drawn (metalmc.clouds.mixin.CloudRendererMixin). Render thread.
 */
public final class MetalClouds {
    private MetalClouds() {
    }

    /** METALMC_EXP=clouds (off by default). The native side reads the same variable (and needs sky too). */
    public static final boolean ENABLED = experiment("clouds");

    private static int nativeOn = -1;

    private static boolean experiment(String name) {
        String v = System.getenv("METALMC_EXP");
        // As MetalLit: the settings' switches reach the native side by a setenv Java's copy of the environment never sees.
        if (v == null) v = metalmc.MetalMCConfig.nativeExperiments();
        for (String s : v.split(",")) {
            if (s.trim().equals(name)) return true;
        }
        return false;
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

        static final MethodHandle ENABLED = h("mmc_clouds_enabled", JAVA_INT);
        static final MethodHandle FRAME = h("mmc_clouds_frame", JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG);
        static final MethodHandle COMPOSITE = h("mmc_clouds_composite", JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG);
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new IllegalStateException(t);
    }

    /** The clouds are on and their shaders compile (asked once, on the Metal backend). */
    private static boolean nativeOn() {
        if (nativeOn < 0) {
            if (!ENABLED || MetalDevice.current == null) return false;
            try {
                nativeOn = (int) Native.ENABLED.invokeExact();
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }
        return nativeOn == 1;
    }

    /** Ours replace vanilla's this frame: on, and our sky draws the frame (the overworld, the camera in air). */
    public static boolean active() {
        return ENABLED && metalmc.sky.Sky.frameActive && nativeOn();
    }

    /**
     * This frame's clouds, after the level and before the shadows (the march reads the level's depth; the shadows take
     * the clouds' shadows). {@code projection}: the level's projection without the anti-aliasing's jitter. Returns true if
     * they're on this frame.
     */
    public static boolean frame(GpuTexture color, GpuTexture depth, Matrix4fc projection, Matrix4fc viewRotation, double camX,
                                double camY, double camZ, float sunAngle, float rainBrightness, float thunder, float seaLevel) {
        if (!active() || !(color instanceof MetalTexture c) || !(depth instanceof MetalTexture d)) return false;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer p = stack.mallocFloat(36);
            projection.get(0, p);
            viewRotation.get(16, p);
            p.put(32, sunAngle).put(33, rainBrightness).put(34, thunder).put(35, seaLevel);
            DoubleBuffer cam = stack.mallocDouble(3);
            cam.put(0, camX).put(1, camY).put(2, camZ);
            return (int) Native.FRAME.invokeExact(c.handle, d.handle, MemoryUtil.memAddress(p), MemoryUtil.memAddress(cam)) == 1;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /**
     * Without anti-aliasing (with it, its resolve composites the clouds as it loads each pixel): the clouds over the level,
     * after the sky's aerial perspective. {@code projection}: as the level was drawn.
     */
    public static boolean composite(GpuTexture color, GpuTexture depth, Matrix4fc projection, Matrix4fc viewRotation) {
        if (!active() || !(color instanceof MetalTexture c) || !(depth instanceof MetalTexture d)) return false;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer p = stack.mallocFloat(32);
            projection.get(0, p);
            viewRotation.get(16, p);
            return (int) Native.COMPOSITE.invokeExact(c.handle, d.handle, MemoryUtil.memAddress(p)) == 1;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
