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
}
