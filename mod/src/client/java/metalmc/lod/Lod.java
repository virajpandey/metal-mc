package metalmc.lod;

import metalmc.backend.MetalLod;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.level.storage.LevelResource;

/**
 * Far-terrain LOD (Voxy-style, clean-room) for the Metal backend. Controlled by config/metalmc.properties
 * (lod, lod.far, lod.live, lod.multiplayer). In single-player it builds from the save's region files, plus
 * chunks as the client loads them (LiveIngest). On a server there are no region files: it builds from
 * loaded chunks only, saved per server under {@code <game dir>/metalmc/lod/}, so it grows as the player
 * explores. While the LOD is ready, the mixins push the far plane and render-distance fog out to FAR and
 * draw the LOD after solid terrain.
 */
public final class Lod implements ClientModInitializer {
    public static final boolean ENABLED = metalmc.MetalMCConfig.lod();
    public static final int FAR = metalmc.MetalMCConfig.lodFar();
    public static final boolean LIVE = metalmc.MetalMCConfig.lodLive();
    public static final boolean MULTIPLAYER = metalmc.MetalMCConfig.lodMultiplayer();

    private static boolean opened;
    private static volatile boolean ready;
    private static int statusTicks;
    private static String openedDir;

    /**
     * True once the LOD is built and should be drawn (and the far plane and fog extended). Overworld only:
     * the LOD reads the overworld's region files, and the Nether and End don't benefit from it.
     */
    public static boolean active() {
        if (!ENABLED || !ready) return false;
        Minecraft mc = Minecraft.getInstance();
        return mc.level != null && mc.level.dimension() == net.minecraft.world.level.Level.OVERWORLD;
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
        String store = mc.gameDirectory.toPath().resolve("metalmc").resolve("lod").resolve(name).resolve("overworld")
            .toAbsolutePath().normalize().toString();
        return new String[]{"", store, "server:" + name};
    }

    private static void tick(Minecraft mc) {
        String[] src = source(mc);
        String current = src == null ? null : src[2];
        // A different save or server (or none) since the LOD was opened: start over.
        if (opened && !java.util.Objects.equals(current, openedDir)) {
            LiveIngest.setEnabled(false);
            MetalLod.close();
            opened = false;
            ready = false;
            openedDir = null;
        }
        if (!opened && src != null && MetalLod.available()) {
            opened = true;
            openedDir = current;
            int cx = mc.player != null ? mc.player.getBlockX() : 0, cz = mc.player != null ? mc.player.getBlockZ() : 0;
            boolean ok = MetalLod.open2(src[0], src[1], FAR, cx, cz);
            System.out.println("[metalmc-lod] opening " + current + " far=" + FAR + " live=" + LIVE + ": " + ok);
            if (ok && LIVE) LiveIngest.setEnabled(true);
        }
        if (!opened || ++statusTicks % 20 != 0) return;
        if (mc.player != null) {
            MetalLod.center(mc.player.getBlockX(), mc.player.getBlockZ(), mc.options.getEffectiveRenderDistance() * 16);
        }
        if (!ready) {
            long[] s = MetalLod.status();
            if (s[0] == 2) {
                ready = true;
                System.out.println("[metalmc-lod] ready: " + s[1] + " nodes, " + s[2] + " quads");
            }
        }
    }
}
