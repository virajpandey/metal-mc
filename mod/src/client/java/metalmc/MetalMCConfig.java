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
        # How far the LOD reaches, in blocks. 32 km costs about 2% more than 8 km (most of it is past the
        # saved world, generated from the seed in single-player; see lod.generate).
        lod.far=32768
        # Also build the LOD from chunks as the game loads them (keeps it current without waiting for saves).
        lod.live=true
        # LOD on servers, built from the chunks you've seen there and saved under metalmc/lod/ (needs lod.live).
        lod.multiplayer=true
        # Block texture detail on LOD terrain (about 2% slower than flat colors).
        lod.textures=true
        # Radius in blocks of full-resolution LOD (one voxel per block) past the render distance; 0 for none.
        lod.detail=768
        # Single-player: past the terrain the world has generated, show terrain sampled from the world's own
        # generator (heights and biomes, on a coarse grid in the background) out to lod.far.
        lod.generate=true

        # Temporal anti-aliasing (Metal backend): smooths jagged and shimmering edges, most visible on far
        # terrain, by blending each frame with the ones before it. About 1.8 ms of GPU per frame at the panel's
        # native resolution; a 120 Hz flight still drops no frames. Set to false for the raw image.
        taa=true

        # Ray-traced sun shadows (Metal backend, prototype): terrain, trees and mountains cast shadows from the sun, out
        # to the LOD's far terrain. About 1 ms of GPU per frame at the panel's native resolution (enough to drop some
        # frames at a locked 120 Hz), so off by default.
        shadows=false

        # The new look (Metal backend, work in progress; off by default, turn on what you like):
        # Lighting computed by MetalMC: sun, sky and block light on all terrain, the nearby chunks in its own format.
        lighting=false
        # A physically based sky and haze over distance (with lighting).
        sky=false
        # Bounce light: sunlight and sky light reflected off surfaces, traced on the GPU (with lighting).
        bounceLight=false
        # Water that reflects the sky and the sun (with lighting and sky).
        waterReflections=false
        # Colored block light: torches orange, soul lights cyan, redstone red, lava orange-red (with lighting).
        coloredLight=false
        # HDR output: real highlights on screens that show them, such as the MacBook Pro's XDR display.
        hdr=false
        # Extra native switches, comma separated (developer use; METALMC_EXP in the environment wins).
        experiments=
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

    public static int lodDetail() {
        try {
            return Math.max(0, Integer.parseInt(get("lod.detail", "768")));
        } catch (NumberFormatException e) {
            return 768;
        }
    }

    public static boolean lodGenerate() {
        return flag("lod.generate", true);
    }

    /** Temporal anti-aliasing (on by default): Taa.swift's resolve, about 1.8 ms of GPU per frame at the panel's resolution. */
    public static boolean taa() {
        return flag("taa", true);
    }

    /** Ray-traced sun shadows (off by default): RtShadows.swift, about 1 ms of GPU per frame at the panel's resolution. */
    public static boolean shadows() {
        return flag("shadows", false);
    }

    public static boolean lodMultiplayer() {
        return flag("lod.multiplayer", true);
    }

    /**
     * The native switches the settings above turn on (Sources/MetalMCNative reads them from METALMC_EXP), comma
     * separated; empty when none. NativeLibrary passes them on before any native code runs.
     */
    public static String nativeExperiments() {
        java.util.LinkedHashSet<String> exp = new java.util.LinkedHashSet<>();
        if (flag("lighting", false)) {
            exp.add("lit");
            exp.add("nearchunks");
        }
        if (flag("sky", false)) exp.add("sky");
        if (flag("bounceLight", false)) exp.add("gi");
        if (flag("waterReflections", false)) exp.add("water");
        if (flag("coloredLight", false)) exp.add("coloredlight");
        if (flag("hdr", false)) exp.add("hdr");
        for (String e : get("experiments", "").split(",")) {
            if (!e.isBlank()) exp.add(e.trim());
        }
        return String.join(",", exp);
    }

    public static int lodFar() {
        try {
            return Math.max(512, Integer.parseInt(get("lod.far", "32768")));
        } catch (NumberFormatException e) {
            return 32768;
        }
    }
}
