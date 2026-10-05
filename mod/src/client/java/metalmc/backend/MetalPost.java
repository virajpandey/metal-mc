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
 * Bridge to the post chain (Sources/MetalMCNative/Post.swift, METALMC_EXP=post, off by default): bloom, eye adaptation,
 * light shafts and a filmic tone curve over the level's scene-linear light, after the anti-aliasing and before the hand
 * and the HUD. Needs our sky (METALMC_EXP=sky) to have drawn the frame; meant for lit mode. With post the main target is
 * float (the native side reports it like HDR's, so MainTargetMixin makes it so). Render thread.
 */
public final class MetalPost {
    private MetalPost() {
    }

    /** METALMC_EXP=post (off by default). The native side reads the same variable. */
    public static final boolean ENABLED = experiment("post");

    private static boolean experiment(String name) {
        String v = System.getenv("METALMC_EXP");
        // As MetalLit: the settings' switches (config/metalmc.properties) reach the native side by a setenv Java's copy of
        // the environment never sees, so the same rule here (METALMC_EXP in the environment first).
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

        static final MethodHandle APPLY = h("mmc_post_apply", JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT);
        static final MethodHandle CLOUDS = h("mmc_post_clouds", null, JAVA_FLOAT);
    }

    /**
     * Before the sky's aerial perspective: where vanilla's clouds are this frame ({@code bottomRelative}: the bottom of
     * their 4-block slab, camera-relative), so the step that takes the level through the air lights them by day.
     */
    public static void clouds(float bottomRelative) {
        if (!ENABLED || MetalDevice.current == null) return;
        try {
            Native.CLOUDS.invokeExact(bottomRelative);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new IllegalStateException(t);
    }

    /**
     * The post chain on the level just resolved into {@code color}: after the anti-aliasing ({@code taa}: it ran this frame;
     * {@code projection} is then its unjittered one) or, without it, after the sky's aerial perspective (the projection the
     * level was drawn with). {@code sunAngle} is vanilla's. Its copy into the frame takes the place of the anti-aliasing's,
     * folded into the next pass on the frame (the hand's). Returns true if it ran.
     */
    public static boolean apply(GpuTexture color, GpuTexture depth, Matrix4fc projection, Matrix4fc viewRotation,
                                double camX, double camY, double camZ, float sunAngle, boolean taa) {
        if (!ENABLED || MetalDevice.current == null || !(color instanceof MetalTexture c) || !(depth instanceof MetalTexture d)) return false;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer p = stack.mallocFloat(33);
            projection.get(0, p);
            viewRotation.get(16, p);
            p.put(32, sunAngle);
            DoubleBuffer cam = stack.mallocDouble(3);
            cam.put(0, camX).put(1, camY).put(2, camZ);
            return (int) Native.APPLY.invokeExact(c.handle, d.handle, MemoryUtil.memAddress(p), MemoryUtil.memAddress(cam), taa ? 1 : 0) == 1;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
