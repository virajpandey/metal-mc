package metalmc.backend;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import static java.lang.foreign.ValueLayout.JAVA_DOUBLE;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Bridge to colored block light (Sources/MetalMCNative/ColoredLight.swift, METALMC_EXP=lit,coloredlight, off by default):
 * a light volume around the camera whose eight colors each spread like vanilla's block light, sampled by lit mode's
 * relight in place of vanilla's lightmap. The blocks come from metalmc.light.ColoredLight (loaded chunks and block
 * changes). Only works on the Metal backend; the bindings live here so the feature stays in its own files.
 */
public final class MetalColoredLight {
    private MetalColoredLight() {
    }

    /** METALMC_EXP=coloredlight with lit (off by default). The native side reads the same variable. */
    public static final boolean ENABLED = MetalLit.ENABLED && experiment("coloredlight");

    private static boolean experiment(String name) {
        String v = System.getenv("METALMC_EXP");
        // As MetalLit: the settings' switches (experiments=coloredlight) when the environment has none.
        if (v == null) v = metalmc.MetalMCConfig.nativeExperiments();
        for (String s : v.split(",")) {
            if (s.trim().equals(name)) return true;
        }
        return false;
    }

    public static boolean enabled() {
        return ENABLED && MetalDevice.current != null;
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

        static final MethodHandle CLASSIFY = h("mmc_cl_classify", JAVA_INT, JAVA_LONG);
        static final MethodHandle RESET = h("mmc_cl_reset", null, JAVA_INT);
        static final MethodHandle CHUNK = h("mmc_cl_chunk", null, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_LONG);
        static final MethodHandle BLOCK = h("mmc_cl_block", null, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT);
        static final MethodHandle FRAME = h("mmc_cl_frame", null, JAVA_DOUBLE, JAVA_DOUBLE, JAVA_DOUBLE);
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new IllegalStateException(t);
    }

    /** The light color class of a block, by its registry name ("minecraft:soul_torch"). */
    public static int classify(String blockName) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            return (int) Native.CLASSIFY.invokeExact(MemoryUtil.memAddress(stack.UTF8(blockName)));
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** A new world or dimension: the native side forgets every section; calls of other generations are ignored. */
    public static void reset(int generation) {
        try { Native.RESET.invokeExact(generation); } catch (Throwable t) { throw rethrow(t); }
    }

    /**
     * One chunk: {@code count} sections, their section y in {@code ys} (ints), 4096 codes each in {@code codes} (shorts,
     * y then z then x; dampening | emission << 4 | class << 8). Replaces whatever the chunk had.
     */
    public static void chunk(int generation, int cx, int cz, int count, long ys, long codes) {
        try { Native.CHUNK.invokeExact(generation, cx, cz, count, ys, codes); } catch (Throwable t) { throw rethrow(t); }
    }

    /** One block changed. */
    public static void block(int generation, int x, int y, int z, int code) {
        try { Native.BLOCK.invokeExact(generation, x, y, z, code); } catch (Throwable t) { throw rethrow(t); }
    }

    /** The frame's volume work, before the relight: the camera's world position. Render thread, no pass open. */
    public static void frame(double x, double y, double z) {
        if (!enabled()) return;
        try { Native.FRAME.invokeExact(x, y, z); } catch (Throwable t) { throw rethrow(t); }
    }
}
