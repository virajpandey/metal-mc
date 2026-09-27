package metalmc.lod;

import metalmc.backend.MetalLod;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.resources.ResourceKey;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.level.Level;
import net.minecraft.world.level.storage.LevelResource;

/**
 * Far-terrain LOD (Voxy-style, clean-room) for the Metal backend. Controlled by config/metalmc.properties
 * (lod, lod.far, lod.live, lod.multiplayer). In single-player it builds from the save's region files, plus
 * chunks as the client loads them (LiveIngest). On a server there are no region files: it builds from
 * loaded chunks only, saved per server under {@code <game dir>/metalmc/lod/}, so it grows as the player
 * explores. While the LOD is ready, the mixins push the far plane and render-distance fog out to FAR and
 * draw the LOD after solid terrain.
 * <p>
 * Each dimension with a LOD (the overworld and the End) gets its own. The native side keeps the one the player
 * left (paused), so coming back from the End is instant. The Nether has none: its fog ends about 100 blocks
 * out, well inside vanilla's chunks.
 */
public final class Lod implements ClientModInitializer {
    public static final boolean ENABLED = metalmc.MetalMCConfig.lod();
    public static final int FAR = metalmc.MetalMCConfig.lodFar();
    public static final boolean LIVE = metalmc.MetalMCConfig.lodLive();
    public static final boolean MULTIPLAYER = metalmc.MetalMCConfig.lodMultiplayer();
    public static final boolean TEXTURES = metalmc.MetalMCConfig.lodTextures();
    public static final boolean GENERATE = metalmc.MetalMCConfig.lodGenerate();
    /** The projection vanilla draws the level with this frame, view bobbing included (GameRendererLodMixin). */
    public static final org.joml.Matrix4f LEVEL_PROJECTION = new org.joml.Matrix4f();

    private static boolean opened;
    private static volatile boolean ready;
    private static volatile boolean built;
    private static int statusTicks;
    private static String openedDir;
    private static ResourceKey<Level> openedDim;   // the dimension whose LOD is current

    /** Dimensions with a LOD. */
    private static boolean supported(ResourceKey<Level> dim) {
        return dim == Level.OVERWORLD || dim == Level.END;
    }

    /** True once the LOD of the player's dimension is built and should be drawn (and the far plane and fog extended). */
    public static boolean active() {
        if (!ENABLED || !ready) return false;
        Minecraft mc = Minecraft.getInstance();
        return mc.level != null && mc.level.dimension() == openedDim;
    }

    /** True once every LOD level has been built at least once (the benchmark waits for this). */
    public static boolean built() {
        return built;
    }

    @Override
    public void onInitializeClient() {
        if (!ENABLED) return;
        ClientTickEvents.END_CLIENT_TICK.register(Lod::tick);
        if (LIVE) LiveIngest.register();
    }

    /**
     * What the LOD should be built from right now: {worldDir, storeDir, identity}, or null for nothing
     * (menus, or a server with lod.multiplayer off).
     */
    private static String[] source(Minecraft mc) {
        if (mc.level == null) return null;
        MinecraftServer server = mc.getSingleplayerServer();
        if (server != null) {
            String dir = server.getWorldPath(LevelResource.ROOT).toAbsolutePath().normalize().toString();
            return new String[]{dir, "", dir};
        }
        net.minecraft.client.multiplayer.ServerData data = mc.getCurrentServer();
        if (data == null || !MULTIPLAYER || !LIVE) return null;
        String name = data.ip.toLowerCase(java.util.Locale.ROOT).replaceAll("[^a-z0-9._-]", "_");
        return new String[]{"", mc.gameDirectory.toPath().resolve("metalmc").resolve("lod").resolve(name).toAbsolutePath().normalize().toString(),
            "server:" + name};
    }

    /** A server's store for one dimension: <game dir>/metalmc/lod/<server>/overworld (the_end, ...). */
    private static String store(String serverDir, ResourceKey<Level> dim) {
        if (serverDir.isEmpty()) return "";
        net.minecraft.resources.Identifier id = dim.identifier();
        String sub = id.getNamespace().equals("minecraft") ? id.getPath() : id.getNamespace() + "_" + id.getPath();
        return java.nio.file.Path.of(serverDir).resolve(sub.replaceAll("[^a-z0-9._-]", "_")).toString();
    }

    private static void tick(Minecraft mc) {
        String[] src = source(mc);
        String current = src == null ? null : src[2];
        // A different save or server (or none) since the LOD was opened: start over.
        if (opened && !java.util.Objects.equals(current, openedDir)) {
            LiveIngest.setEnabled(false);
            FarTerrain.stop();
            MetalLod.close();
            opened = false;
            ready = false;
            built = false;
            openedDir = null;
            openedDim = null;
        }
        // Entering a dimension with a LOD: open (or resume) its own. Elsewhere (the Nether) the last one is left
        // as it was, neither drawn nor moved.
        ResourceKey<Level> dim = mc.level != null ? mc.level.dimension() : null;
        if (src != null && dim != null && dim != openedDim && supported(dim) && MetalLod.available()) {
            LiveIngest.setEnabled(false);
            FarTerrain.stop();
            int cx = mc.player != null ? mc.player.getBlockX() : 0, cz = mc.player != null ? mc.player.getBlockZ() : 0;
            long world = MetalLod.open3(src[0], store(src[1], dim), dim.identifier().toString(), FAR, cx, cz);
            System.out.println("[metalmc-lod] opening " + current + " " + dim.identifier() + " far=" + FAR + " live=" + LIVE + ": " + world);
            // Nothing to build from: drop every dimension's LOD, so the last one can't be drawn here.
            if (world == 0) MetalLod.close();
            opened = true;
            openedDir = current;
            openedDim = dim;
            ready = false;
            built = false;
            statusTicks = 19;   // check the status on the next tick: a resumed dimension draws right away
            if (world != 0 && LIVE) LiveIngest.setEnabled(true, world, dim);
            if (world != 0 && GENERATE && mc.getSingleplayerServer() != null) FarTerrain.start(mc.getSingleplayerServer(), dim, world);
        }
        if (!opened || ++statusTicks % 20 != 0) return;
        if (mc.player != null && dim == openedDim) {
            MetalLod.center(mc.player.getBlockX(), mc.player.getBlockZ(), mc.options.getEffectiveRenderDistance() * 16);
        }
        if (!ready || !built) {
            long[] s = MetalLod.status();
            if (!ready && s[0] == 2) {
                ready = true;
                System.out.println("[metalmc-lod] ready: " + s[1] + " nodes, " + s[2] + " quads");
            }
            if (!built && s[3] == 1) {
                built = true;
                System.out.println("[metalmc-lod] first build done: " + s[1] + " nodes, " + s[2] + " quads");
            }
        }
    }
}
