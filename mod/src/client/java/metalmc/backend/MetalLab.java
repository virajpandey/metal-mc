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

    private static MethodHandle reload;
    private static boolean looked;

    /**
     * Recompiles every shader file of the shader directory (METALMC_SHADERDIR) now, changed or not; their pipelines are
     * rebuilt between this frame and the next. Returns a summary for the log ("..." with what failed).
     */
    public static synchronized String reloadShaders() {
        if (!looked) {
            looked = true;
            try {
                var lib = SymbolLookup.libraryLookup(NativeLibrary.path(), Arena.global());
                reload = lib.find("mmc_shaderlab_reload").map(a -> Linker.nativeLinker().downcallHandle(a,
                    FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT))).orElse(null);
            } catch (RuntimeException e) {
                return "no native library (" + e.getMessage() + ")";
            }
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
