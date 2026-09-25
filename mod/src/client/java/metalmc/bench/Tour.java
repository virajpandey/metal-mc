package metalmc.bench;

import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.components.debug.DebugScreenEntries;
import net.minecraft.client.gui.components.debug.DebugScreenEntryStatus;
import net.minecraft.client.gui.screens.PauseScreen;
import net.minecraft.client.gui.screens.inventory.InventoryScreen;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.server.MinecraftServer;

import java.util.List;
import java.util.function.Consumer;

/**
 * Rendering coverage tour (-Dmetalmc.tour=1): walks the game through weather, times of day, GUI
 * screens, debug overlays, block entities, entities with special shaders, particles, and the other
 * dimensions, taking a screenshot at the end of each step. Run it on every backend and diff the
 * screenshots pairwise. Scenes are built in the sky above the world so the backdrop is deterministic.
 */
final class Tour {
    private Tour() {}

    /** Platform in the sky for the close-up steps; the camera sits north of it looking south. */
    private static final int PX = 8, PY = 220, PZ = 24;

    record Pose(double x, double y, double z, float yaw, float pitch) {}

    static final Pose ORBIT = new Pose(Bench.CENTER_X + Bench.RADIUS, Bench.HEIGHT, Bench.CENTER_Z, 90f, 25f);
    static final Pose CLOSE = new Pose(PX + 0.5, PY + 3.5, PZ - 7.5, 0f, 22f);
    static final Pose NETHER = new Pose(0.5, 100, 0.5, 45f, 20f);
    static final Pose END = new Pose(60.5, 80, 0.5, 90f, 25f);

    record Step(String name, Pose pose, int ticks, Consumer<Minecraft> setup) {}

    private static void cmd(Minecraft mc, String... commands) {
        MinecraftServer server = mc.getSingleplayerServer();
        if (server == null) return;
        server.execute(() -> {
            for (String c : commands) server.getCommands().performPrefixedCommand(server.createCommandSourceStack(), c);
        });
    }

    private static String fill(int x0, int y0, int z0, int x1, int y1, int z1, String block) {
        return "fill " + x0 + " " + y0 + " " + z0 + " " + x1 + " " + y1 + " " + z1 + " " + block;
    }

    private static String set(int dx, int dy, int dz, String block) {
        return "setblock " + (PX + dx) + " " + (PY + dy) + " " + (PZ + dz) + " " + block;
    }

    private static String summon(String entity, double dx, double dz, String nbt) {
        return "summon " + entity + " " + (PX + dx + 0.5) + " " + (PY + 1) + " " + (PZ + dz + 0.5) + " " + nbt;
    }

    static final List<Step> STEPS = List.of(
        new Step("day", ORBIT, 60, mc -> {
            mc.debugEntries.setStatus(DebugScreenEntries.CHUNK_BORDERS, DebugScreenEntryStatus.NEVER);
            mc.debugEntries.setStatus(DebugScreenEntries.ENTITY_HITBOXES, DebugScreenEntryStatus.NEVER);
            cmd(mc, "time set 6000", "weather clear", "gamerule advance_time false", "gamerule advance_weather false");
        }),
        new Step("sunset", ORBIT, 40, mc -> cmd(mc, "time set 12600")),
        new Step("night", ORBIT, 40, mc -> cmd(mc, "time set 18000")),
        new Step("rain", ORBIT, 80, mc -> cmd(mc, "time set 6000", "weather rain")),
        new Step("scene", CLOSE, 100, mc -> cmd(mc, "weather clear",
            fill(PX - 6, PY, PZ - 2, PX + 6, PY, PZ + 8, "minecraft:smooth_stone"),
            set(-5, 1, 2, "minecraft:glass"), set(-4, 1, 2, "minecraft:red_stained_glass"), set(-3, 1, 2, "minecraft:water"),
            set(-2, 1, 2, "minecraft:lava"), set(-1, 1, 2, "minecraft:torch"), set(0, 1, 4, "minecraft:chest[facing=north]"),
            set(1, 1, 2, "minecraft:enchanting_table"), set(2, 1, 2, "minecraft:oak_leaves"), set(3, 1, 2, "minecraft:poppy"),
            set(4, 1, 2, "minecraft:end_portal"), set(5, 1, 2, "minecraft:glowstone"), set(-5, 1, 5, "minecraft:red_bed[facing=north,part=foot]"),
            set(-5, 1, 6, "minecraft:red_bed[facing=north,part=head]"), set(3, 1, 6, "minecraft:white_banner"),
            set(5, 1, 6, "minecraft:oak_sign[rotation=8]{front_text:{messages:['\"MetalMC\"','\"tour\"','\"\"','\"\"']}}"),
            summon("minecraft:pig", -3, 5, "{NoAI:1b,Silent:1b,Rotation:[180f,0f]}"),
            summon("minecraft:creeper", -1, 6, "{NoAI:1b,Silent:1b,powered:1b,Rotation:[180f,0f]}"),
            summon("minecraft:zombie", 1, 6, "{NoAI:1b,Silent:1b,Rotation:[180f,0f],equipment:{head:{id:\"minecraft:diamond_helmet\",components:{\"minecraft:enchantment_glint_override\":true}}}}"),
            summon("minecraft:villager", 3, 4, "{NoAI:1b,Silent:1b,Rotation:[180f,0f]}"),
            summon("minecraft:sheep", -1, 3, "{NoAI:1b,Silent:1b,Glowing:1b,Rotation:[180f,0f]}"),
            summon("minecraft:armor_stand", 5, 4, "{Rotation:[180f,0f],equipment:{chest:{id:\"minecraft:iron_chestplate\"}}}"))),
        new Step("particles", CLOSE, 6, mc -> cmd(mc,
            "particle minecraft:flame " + (PX + 0.5) + " " + (PY + 2) + " " + (PZ + 1) + " 1 0.5 0.5 0 200 force",
            "particle minecraft:happy_villager " + (PX - 2.5) + " " + (PY + 2) + " " + (PZ + 1) + " 1 0.5 0.5 0 60 force")),
        new Step("f3", CLOSE, 20, mc -> {
            // Explicit statuses: toggleStatus is a 3-state machine whose result depends on the status
            // persisted in debug-profile.json by earlier runs.
            mc.debugEntries.setOverlayVisible(true);
            mc.debugEntries.setStatus(DebugScreenEntries.CHUNK_BORDERS, DebugScreenEntryStatus.ALWAYS_ON);
            mc.debugEntries.setStatus(DebugScreenEntries.ENTITY_HITBOXES, DebugScreenEntryStatus.ALWAYS_ON);
        }),
        new Step("pause", CLOSE, 20, mc -> {
            mc.debugEntries.setOverlayVisible(false);
            mc.debugEntries.setStatus(DebugScreenEntries.CHUNK_BORDERS, DebugScreenEntryStatus.NEVER);
            mc.debugEntries.setStatus(DebugScreenEntries.ENTITY_HITBOXES, DebugScreenEntryStatus.NEVER);
            mc.gui.setScreen(new PauseScreen(true));
        }),
        new Step("inventory", CLOSE, 20, mc -> {
            cmd(mc, "give @a minecraft:diamond_sword[minecraft:enchantment_glint_override=true]", "give @a minecraft:grass_block 64",
                "give @a minecraft:oak_sapling 16", "give @a minecraft:potion[minecraft:potion_contents={potion:\"minecraft:healing\"}]");
            mc.gui.setScreen(null);
            if (mc.player != null) mc.gui.setScreen(new InventoryScreen(mc.player));
        }),
        new Step("nether", NETHER, 200, mc -> {
            mc.gui.setScreen(null);
            cmd(mc, "execute in minecraft:the_nether run tp @a 0.5 100 0.5");
        }),
        new Step("end", END, 200, mc -> cmd(mc, "execute in minecraft:the_end run tp @a 60.5 80 0.5"))
    );

    /** LOD showcase (-Dmetalmc.tour=lod): horizontal views at ground level and from high up, four directions each. */
    static final List<Step> LOD_STEPS = List.of(
        new Step("ground-north", ground(180f), 60, mc -> cmd(mc, "time set 6000", "weather clear", "gamerule advance_time false", "gamerule advance_weather false")),
        new Step("ground-east", ground(270f), 40, mc -> {}),
        new Step("ground-south", ground(0f), 40, mc -> {}),
        new Step("ground-west", ground(90f), 40, mc -> {}),
        new Step("high-north", new Pose(Bench.CENTER_X, 260, Bench.CENTER_Z, 180f, 12f), 60, mc -> {}),
        new Step("high-east", new Pose(Bench.CENTER_X, 260, Bench.CENTER_Z, 270f, 12f), 40, mc -> {}),
        new Step("high-south", new Pose(Bench.CENTER_X, 260, Bench.CENTER_Z, 0f, 12f), 40, mc -> {}),
        new Step("high-west", new Pose(Bench.CENTER_X, 260, Bench.CENTER_Z, 90f, 12f), 40, mc -> {})
    );

    /** A pose 3 blocks above the terrain at the tour center, looking horizontally (a little down). */
    private static Pose ground(float yaw) {
        return new Pose(Bench.CENTER_X, Double.NaN, Bench.CENTER_Z, yaw, 3f);
    }

    static final boolean LOD_TOUR = "lod".equals(System.getProperty("metalmc.tour"));

    static List<Step> steps() {
        return LOD_TOUR ? LOD_STEPS : STEPS;
    }

    private static int step = -1;
    private static int tick;
    private static boolean done;

    /** Returns true once the tour has finished. */
    static boolean onTick(Minecraft mc, LocalPlayer player) {
        if (done) return true;
        List<Step> steps = steps();
        if (step < 0 || ++tick >= steps.get(step).ticks()) {
            if (step >= 0) Bench.screenshotNow(mc, "tour-" + String.format("%02d", step) + "-" + steps.get(step).name());
            step++;
            tick = 0;
            if (step >= steps.size()) {
                done = true;
                return false;
            }
            Bench.log("tour step " + step + ": " + steps.get(step).name());
            steps.get(step).setup().accept(mc);
        }
        Pose p = steps.get(step).pose();
        if (Double.isNaN(p.y()) && mc.level != null) {
            // Ground pose: 3 blocks above the highest block at the center column.
            int top = mc.level.getHeight(net.minecraft.world.level.levelgen.Heightmap.Types.MOTION_BLOCKING, (int) Math.floor(p.x()), (int) Math.floor(p.z()));
            p = new Pose(p.x(), top + 3, p.z(), p.yaw(), p.pitch());
        }
        player.getAbilities().mayfly = true;
        player.getAbilities().flying = true;
        player.snapTo(p.x(), p.y(), p.z(), p.yaw(), p.pitch());
        player.setDeltaMovement(0, 0, 0);
        return false;
    }
}
