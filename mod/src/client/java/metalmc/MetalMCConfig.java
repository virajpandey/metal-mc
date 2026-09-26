package metalmc;

import java.io.IOException;
import java.io.InputStream;
import java.io.Writer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;
import net.fabricmc.loader.api.FabricLoader;

/**
 * Settings from config/metalmc.properties (written with defaults on first launch). A system property
 * with the same key prefixed by "metalmc." overrides the file, which is how the benchmark harness
 * switches features per run.
 */
public final class MetalMCConfig {
    private MetalMCConfig() {
    }

    private static final String DEFAULTS = """
        # MetalMC settings. Restart Minecraft after changing them.

        # Graphics backend: "metal" draws through Apple's Metal directly (falls back to vanilla's
        # backends if Metal can't start); "off" leaves Minecraft's own choice (OpenGL or Vulkan).
        backend=metal

        # Skip block faces that point away from the camera (identical image, less GPU work).
        facingCulling=true

        # Skip distant chunk sections hidden behind terrain (tested on the GPU each frame; sections within
        # 48 blocks are always drawn).
        occlusionCulling=true

        # Far-terrain LOD beyond the render distance (Metal backend). In single-player it shows terrain the
        # world has already generated; on servers, terrain you've already seen there. It follows the player.
        lod=true
        # How far the LOD reaches, in blocks.
        lod.far=4096
        # Also build the LOD from chunks as the game loads them (keeps it current without waiting for saves).
        lod.live=true
        # LOD on servers, built from the chunks you've seen there and saved under metalmc/lod/ (needs lod.live).
        lod.multiplayer=true
        # Block texture detail on LOD terrain (about 2% slower than flat colors).
        lod.textures=true
        """;

    private static final Properties FILE = load();

    private static Properties load() {
        Properties p = new Properties();
        try {
            Path dir = FabricLoader.getInstance().getConfigDir();
            Path file = dir.resolve("metalmc.properties");
            if (!Files.exists(file)) {
                Files.createDirectories(dir);
                try (Writer w = Files.newBufferedWriter(file, StandardCharsets.UTF_8)) {
                    w.write(DEFAULTS);
                }
            }
            try (InputStream in = Files.newInputStream(file)) {
                p.load(in);
            }
        } catch (IOException | RuntimeException e) {
            System.err.println("[metalmc] couldn't read config/metalmc.properties, using defaults: " + e);
            try {
                p.load(new java.io.StringReader(DEFAULTS));
            } catch (IOException ignored) {
                // Unreachable for a string.
            }
        }
        return p;
    }

    private static String get(String key, String fallback) {
        String sys = System.getProperty("metalmc." + key);
        if (sys != null && !sys.isEmpty()) return sys.trim();
        return FILE.getProperty(key, fallback).trim();
    }

    private static boolean flag(String key, boolean fallback) {
        String v = get(key, fallback ? "true" : "false").toLowerCase();
        return v.equals("true") || v.equals("1") || v.equals("on") || v.equals("yes");
    }

    public static boolean metalBackend() {
        return get("backend", "metal").equalsIgnoreCase("metal");
    }

    public static boolean facingCulling() {
        return flag("facingCulling", true);
    }

    public static boolean occlusionCulling() {
        return flag("occlusionCulling", true);
    }

    public static boolean lod() {
        return flag("lod", true);
    }

    public static boolean lodLive() {
        return flag("lod.live", true);
    }

    public static boolean lodTextures() {
        return flag("lod.textures", true);
    }

    public static boolean lodMultiplayer() {
        return flag("lod.multiplayer", true);
    }

    public static int lodFar() {
        try {
            return Math.max(512, Integer.parseInt(get("lod.far", "4096")));
        } catch (NumberFormatException e) {
            return 4096;
        }
    }
}
