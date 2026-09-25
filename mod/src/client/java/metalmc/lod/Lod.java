package metalmc.lod;

import metalmc.backend.MetalLod;
import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.minecraft.client.Minecraft;
import net.minecraft.server.MinecraftServer;
import net.minecraft.world.level.storage.LevelResource;

/**
 * Far-terrain LOD (Voxy-style, clean-room) for the Metal backend. Off unless -Dmetalmc.lod=1.
 * On single-player world join it starts the native LOD build from the save's region files. While
 * the LOD is ready, the mixins push the far plane and render-distance fog out to FAR and draw the LOD
 * after solid terrain.
 */
public final class Lod implements ClientModInitializer {
    public static final boolean ENABLED = "1".equals(System.getProperty("metalmc.lod", "0"));
    public static final int FAR = Integer.getInteger("metalmc.lod.far", 2048);

    private static boolean opened;
    private static volatile boolean ready;
    private static int statusTicks;

    /** True once the LOD is built and should be drawn (and the far plane and fog extended). */
    public static boolean active() {
        return ENABLED && ready;
    }

    @Override
    public void onInitializeClient() {
        if (!ENABLED) return;
        ClientTickEvents.END_CLIENT_TICK.register(Lod::tick);
    }

    private static void tick(Minecraft mc) {
        MinecraftServer server = mc.getSingleplayerServer();
        if (!opened && server != null && mc.level != null && MetalLod.available()) {
            opened = true;
            String dir = server.getWorldPath(LevelResource.ROOT).toAbsolutePath().normalize().toString();
            int cx = mc.player != null ? mc.player.getBlockX() : 0, cz = mc.player != null ? mc.player.getBlockZ() : 0;
            System.out.println("[metalmc-lod] opening " + dir + " far=" + FAR + ": " + MetalLod.open(dir, FAR, cx, cz));
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
