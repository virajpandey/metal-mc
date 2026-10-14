package metalmc.backend;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;

/** Lab mode's native half (Sources/MetalMCNative/ShaderLab.swift; docs/lab-mode.md). */
public final class MetalLab {
    private MetalLab() {
    }

    private static MethodHandle reload, mask, maskStatus, burst, burstStatus;
    private static boolean looked;

    private static void look() {
        if (looked) return;
        looked = true;
        var lib = SymbolLookup.libraryLookup(NativeLibrary.path(), Arena.global());
        reload = lib.find("mmc_shaderlab_reload").map(a -> Linker.nativeLinker().downcallHandle(a,
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT))).orElse(null);
        mask = lib.find("mmc_lab_mask").map(a -> Linker.nativeLinker().downcallHandle(a,
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_FLOAT))).orElse(null);
        maskStatus = lib.find("mmc_lab_mask_status").map(a -> Linker.nativeLinker().downcallHandle(a,
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT))).orElse(null);
        burst = lib.find("mmc_lab_burst").map(a -> Linker.nativeLinker().downcallHandle(a,
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT))).orElse(null);
        burstStatus = lib.find("mmc_lab_burst_status").map(a -> Linker.nativeLinker().downcallHandle(a,
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT))).orElse(null);
    }

    /**
     * Lab mode's `mask` (Sources/MetalMCNative/LabMask.swift): the class mask of the next {@code frames} relit frames into
     * the PNG at {@code path}; {@code cloudHeight} is the bottom of vanilla's cloud slab (world y). Returns null if it was
     * taken, else why not.
     */
    public static synchronized String requestMask(String path, int frames, float cloudHeight) {
        try {
            look();
        } catch (RuntimeException e) {
            return "no native library (" + e.getMessage() + ")";
        }
        if (mask == null) return "the native library has no class mask (an older build)";
        try (Arena arena = Arena.ofConfined()) {
            int ok = (int) mask.invokeExact(arena.allocateFrom(path), frames, cloudHeight);
            return ok != 0 ? null : "lit mode is off (METALMC_EXP=lit): no G-buffer to classify";
        } catch (Throwable t) {
            return t.toString();
        }
    }

    /** The mask's state: 2 idle, 0 working, 1 written, -1 failed; and its summary (the class shares, or the error). */
    public static synchronized Object[] maskStatus() {
        if (maskStatus == null) return new Object[]{-1, "the native library has no class mask"};
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(1024);
            int s = (int) maskStatus.invokeExact(buf, 1024);
            return new Object[]{s, buf.getString(0)};
        } catch (Throwable t) {
            return new Object[]{-1, t.toString()};
        }
    }

    /**
     * Lab mode's `burst` (Sources/MetalMCNative/Flicker.swift): the next {@code frames} presented frames' flicker into
     * {@code base}-flicker.png and the rest, the first {@code save} frames kept. Returns null if it was taken, else why not.
     */
    public static synchronized String requestBurst(String base, int frames, int save) {
        try {
            look();
        } catch (RuntimeException e) {
            return "no native library (" + e.getMessage() + ")";
        }
        if (burst == null) return "the native library has no flicker meter (an older build)";
        try (Arena arena = Arena.ofConfined()) {
            int ok = (int) burst.invokeExact(arena.allocateFrom(base), frames, save);
            return ok != 0 ? null : "not taken";
        } catch (Throwable t) {
            return t.toString();
        }
    }

    /** The burst's state: 2 idle, 0 working, 1 written, -1 failed; and its summary (the numbers, or the error). */
    public static synchronized Object[] burstStatus() {
        if (burstStatus == null) return new Object[]{-1, "the native library has no flicker meter"};
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(2048);
            int s = (int) burstStatus.invokeExact(buf, 2048);
            return new Object[]{s, buf.getString(0)};
        } catch (Throwable t) {
            return new Object[]{-1, t.toString()};
        }
    }

    /**
     * Recompiles every shader file of the shader directory (METALMC_SHADERDIR) now, changed or not; their pipelines are
     * rebuilt between this frame and the next. Returns a summary for the log ("..." with what failed).
     */
    public static synchronized String reloadShaders() {
        try {
            look();
        } catch (RuntimeException e) {
            return "no native library (" + e.getMessage() + ")";
        }
        if (reload == null) return "the native library has no shader lab (an older build)";
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(4096);
            int failed = (int) reload.invokeExact(buf, 4096);
            return (failed != 0 ? "error: " : "") + buf.getString(0);
        } catch (Throwable t) {
            return "error: " + t;
        }
    }
}
