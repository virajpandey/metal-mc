package metalmc.backend;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.util.HexFormat;
import net.fabricmc.loader.api.FabricLoader;

/**
 * Finds libMetalMCNative.dylib: -Dmetalmc.native in development, otherwise the copy bundled in the mod
 * jar, extracted to <game dir>/metalmc/natives under a content-hash name (so updates never collide).
 */
final class NativeLibrary {
    private NativeLibrary() {
    }

    private static final String RESOURCE = "/natives/macos-arm64/libMetalMCNative.dylib";
    private static Path cached;

    static synchronized Path path() {
        if (cached != null) return cached;
        passSettings();
        String dev = System.getProperty("metalmc.native");
        if (dev != null && !dev.isEmpty()) return cached = Path.of(dev);
        try (InputStream in = NativeLibrary.class.getResourceAsStream(RESOURCE)) {
            if (in == null) throw new IllegalStateException("the mod jar has no " + RESOURCE);
            byte[] bytes = in.readAllBytes();
            String hash = HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes)).substring(0, 16);
            Path dir = FabricLoader.getInstance().getGameDir().resolve("metalmc").resolve("natives");
            Files.createDirectories(dir);
            Path file = dir.resolve("libMetalMCNative-" + hash + ".dylib");
            if (!Files.exists(file)) {
                Path tmp = Files.createTempFile(dir, "libMetalMCNative", ".tmp");
                Files.write(tmp, bytes);
                Files.move(tmp, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            }
            return cached = file;
        } catch (IOException | java.security.NoSuchAlgorithmException e) {
            throw new IllegalStateException("couldn't extract the MetalMC native library: " + e, e);
        }
    }

    /**
     * The native code reads its switches from METALMC_EXP once, the first time it needs them, so the settings'
     * switches (MetalMCConfig.nativeExperiments) go into the process environment before any native call. A
     * METALMC_EXP already in the environment (the development runs and the benchmark harness) is left as it is.
     */
    private static void passSettings() {
        String exp = metalmc.MetalMCConfig.nativeExperiments();
        if (exp.isEmpty() || System.getenv("METALMC_EXP") != null) return;
        try {
            java.lang.foreign.Linker linker = java.lang.foreign.Linker.nativeLinker();
            java.lang.invoke.MethodHandle setenv = linker.downcallHandle(
                    linker.defaultLookup().find("setenv").orElseThrow(),
                    java.lang.foreign.FunctionDescriptor.of(java.lang.foreign.ValueLayout.JAVA_INT,
                            java.lang.foreign.ValueLayout.ADDRESS, java.lang.foreign.ValueLayout.ADDRESS,
                            java.lang.foreign.ValueLayout.JAVA_INT));
            try (java.lang.foreign.Arena arena = java.lang.foreign.Arena.ofConfined()) {
                int r = (int) setenv.invokeExact(arena.allocateFrom("METALMC_EXP"), arena.allocateFrom(exp), 0);
                System.out.println("[metalmc] settings turn on: " + exp + (r == 0 ? "" : " (setenv failed: " + r + ")"));
            }
        } catch (Throwable t) {
            System.err.println("[metalmc] couldn't pass the settings' switches to the native library: " + t);
        }
    }
}
