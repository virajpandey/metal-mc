package metalmc.backend;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.invoke.MethodHandle;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Java side of the native bridge (libMetalMCNative.dylib, built from Sources/MetalMCNative by SwiftPM),
 * called through the java.lang.foreign API. The dylib path comes from -Dmetalmc.native.
 */
public final class MetalNative {
    public static final int EXPECTED_ABI = 1;

    private final MethodHandle abiVersion;
    private final MethodHandle deviceName;
    private final MethodHandle supportsApple9;

    private MetalNative(SymbolLookup lib) {
        Linker linker = Linker.nativeLinker();
        abiVersion = linker.downcallHandle(find(lib, "mmc_abi_version"), FunctionDescriptor.of(ValueLayout.JAVA_INT));
        deviceName = linker.downcallHandle(find(lib, "mmc_device_name"),
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
        supportsApple9 = linker.downcallHandle(find(lib, "mmc_supports_apple9"), FunctionDescriptor.of(ValueLayout.JAVA_INT));
    }

    private static MemorySegment find(SymbolLookup lib, String name) {
        return lib.find(name).orElseThrow(() -> new IllegalStateException("missing native symbol " + name));
    }

    /** Loads the dylib, or throws with a readable reason. */
    public static MetalNative load() {
        String path = System.getProperty("metalmc.native");
        if (path == null || path.isEmpty()) throw new IllegalStateException("-Dmetalmc.native is not set");
        Path p = Path.of(path);
        if (!Files.isRegularFile(p)) throw new IllegalStateException("native library not found: " + p);
        return new MetalNative(SymbolLookup.libraryLookup(p, Arena.global()));
    }

    public int abiVersion() {
        try {
            return (int) abiVersion.invokeExact();
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    public String deviceName() {
        try (Arena arena = Arena.ofConfined()) {
            MemorySegment buf = arena.allocate(256);
            int ok = (int) deviceName.invokeExact(buf, 256);
            return ok == 1 ? buf.getString(0) : null;
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    public boolean supportsApple9() {
        try {
            return (int) supportsApple9.invokeExact() == 1;
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }
}
