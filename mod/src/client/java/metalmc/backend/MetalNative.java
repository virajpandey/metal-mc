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
 * called through the java.lang.foreign API. See NativeLibrary for where the dylib is found.
 */
public final class MetalNative {
    public static final int EXPECTED_ABI = 3;

    private final MethodHandle abiVersion;
    private final MethodHandle deviceName;
    private final MethodHandle supportsApple9;
    private final MethodHandle release;
    private final MethodHandle surfaceCreate;
    private final MethodHandle surfaceSize;
    private final MethodHandle surfaceRunFrames;

    private MetalNative(SymbolLookup lib) {
        Linker linker = Linker.nativeLinker();
        abiVersion = linker.downcallHandle(find(lib, "mmc_abi_version"), FunctionDescriptor.of(ValueLayout.JAVA_INT));
        deviceName = linker.downcallHandle(find(lib, "mmc_device_name"),
            FunctionDescriptor.of(ValueLayout.JAVA_INT, ValueLayout.ADDRESS, ValueLayout.JAVA_INT));
        supportsApple9 = linker.downcallHandle(find(lib, "mmc_supports_apple9"), FunctionDescriptor.of(ValueLayout.JAVA_INT));
        release = linker.downcallHandle(find(lib, "mmc_release"), FunctionDescriptor.ofVoid(ValueLayout.JAVA_LONG));
        surfaceCreate = linker.downcallHandle(find(lib, "mmc_surface_create"),
            FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT));
        surfaceSize = linker.downcallHandle(find(lib, "mmc_surface_size"),
            FunctionDescriptor.of(ValueLayout.JAVA_LONG, ValueLayout.JAVA_LONG));
        surfaceRunFrames = linker.downcallHandle(find(lib, "mmc_surface_run_frames"),
            FunctionDescriptor.of(ValueLayout.JAVA_DOUBLE, ValueLayout.JAVA_LONG, ValueLayout.JAVA_INT, ValueLayout.JAVA_INT));
    }

    public void release(long handle) {
        try {
            release.invokeExact(handle);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    /** Wraps a CAMetalLayer address (from SDL_Metal_GetLayer). Returns a handle, or 0. */
    public long surfaceCreate(long layerAddress, boolean displaySync) {
        try {
            return (long) surfaceCreate.invokeExact(layerAddress, displaySync ? 1 : 0);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    /** Drawable size as {width, height}. */
    public int[] surfaceSize(long handle) {
        try {
            long packed = (long) surfaceSize.invokeExact(handle);
            return new int[]{(int) (packed >>> 32), (int) packed};
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    /** Clears and presents `frames` frames as fast as presentation allows; returns elapsed seconds. */
    public double surfaceRunFrames(long handle, int frames, int frameOffset) {
        try {
            return (double) surfaceRunFrames.invokeExact(handle, frames, frameOffset);
        } catch (Throwable t) {
            throw new IllegalStateException(t);
        }
    }

    private static MemorySegment find(SymbolLookup lib, String name) {
        return lib.find(name).orElseThrow(() -> new IllegalStateException("missing native symbol " + name));
    }

    /** Loads the dylib, or throws with a readable reason. */
    public static MetalNative load() {
        Path p = NativeLibrary.path();
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
