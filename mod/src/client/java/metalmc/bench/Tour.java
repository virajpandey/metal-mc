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
        if (server == null) {
            // Multiplayer: send them as chat commands (the bench player is op on the test server).
            if (mc.player != null) for (String c : commands) mc.player.connection.sendCommand(c);
            return;
        }
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

    /**
     * Multiplayer LOD test (-PbenchTour=mp, against a local server): the server moves the player with /tp
     * (a server rejects the bench's client-side moves), east across the world so the client loads those
     * chunks, then back to a high pose looking east over them. "mp2" only takes the final pose, to check
     * that the LOD comes back from its saved store after reconnecting.
     */
    static final boolean MP_TOUR = "mp".equals(System.getProperty("metalmc.tour")) || "mp2".equals(System.getProperty("metalmc.tour"));
    private static final String OVERVIEW = "tp @s 8 260 8 270 12";
    static final List<Step> MP_STEPS = List.of(
        new Step("mp-start", null, 200, mc -> cmd(mc, "gamemode creative", "time set 6000", "weather clear",
            "gamerule advance_time false", "gamerule advance_weather false", "tp @s 8 200 8 270 12")),
        new Step("mp-500", null, 200, mc -> cmd(mc, "tp @s 500 200 8 270 12")),
        new Step("mp-1000", null, 200, mc -> cmd(mc, "tp @s 1000 200 8 270 12")),
        new Step("mp-1500", null, 200, mc -> cmd(mc, "tp @s 1500 200 8 270 12")),
        new Step("mp-overview", null, 400, mc -> cmd(mc, OVERVIEW))
    );
    static final List<Step> MP2_STEPS = List.of(
        new Step("mp-overview", null, 300, mc -> cmd(mc, "gamemode creative", "time set 6000", "weather clear", OVERVIEW))
    );

    /**
     * Fidelity measurement (-PbenchTour=fidelity with -Pfidelity=1): fixed poses with no HUD, no mobs, frozen
     * noon and clear weather. Rendered once as vanilla RD 32 (the reference), once as vanilla RD 12 (the mask of
     * what the LOD must fill) and once per LOD variant; tools/fidscore.py compares them. The first step waits
     * 30 s so RD 32's chunks load and compile.
     */
    static final boolean FIDELITY_TOUR = "fidelity".equals(System.getProperty("metalmc.tour"));
    private static Pose at(double y, float yaw, float pitch) {
        return new Pose(Bench.CENTER_X, y, Bench.CENTER_Z, yaw, pitch);
    }
    static final List<Step> FIDELITY_STEPS = List.of(
        new Step("ground-north", ground(180f), 600, mc -> {
            // -PfidelityTime=<ticks> (6000 noon, 13000 dusk, 18000 midnight).
            cmd(mc, "time set " + Integer.getInteger("metalmc.fidelity.time", 6000), "weather clear", "gamerule advance_time false", "gamerule advance_weather false",
                "gamerule spawn_mobs false", "kill @e[type=!player]");
            if (!mc.gui.hud.isHidden()) mc.gui.hud.toggle();   // F1: no HUD, no hand
        }),
        new Step("ground-east", ground(270f), 80, mc -> {}),
        new Step("ground-south", ground(0f), 80, mc -> {}),
        new Step("ground-west", ground(90f), 80, mc -> {}),
        new Step("mid-north", at(150, 180f, 20f), 80, mc -> {}),
        new Step("mid-east", at(150, 270f, 20f), 80, mc -> {}),
        new Step("mid-south", at(150, 0f, 20f), 80, mc -> {}),
        new Step("mid-west", at(150, 90f, 20f), 80, mc -> {}),
        new Step("high-north", at(260, 180f, 12f), 80, mc -> {}),
        new Step("high-south", at(260, 0f, 12f), 80, mc -> {})
    );

    /**
     * End LOD test (-PbenchTour=end): the main island in four directions, high above it, then out among the outer
     * islands (1.5 km east, which vanilla generates on arrival). No HUD (the dragon's boss bar), noon.
     */
    static final boolean END_TOUR = "end".equals(System.getProperty("metalmc.tour"));
    private static Pose end(double x, double y, float yaw, float pitch) {
        return new Pose(x, y, 0.5, yaw, pitch);
    }
    private static final String KILL_DRAGON = "kill @e[type=minecraft:ender_dragon]";
    static final List<Step> END_STEPS = List.of(
        new Step("end-arrive", end(0.5, 90, 270f, 6f), 700, mc -> {
            cmd(mc, "execute in minecraft:the_end run tp @a 0.5 90 0.5", "time set 6000", "gamerule advance_time false");
            if (!mc.gui.hud.isHidden()) mc.gui.hud.toggle();
        }),
        // The dragon fight's boss bar fogs the End to 96 blocks (vanilla); the tour shows it after the fight.
        new Step("end-east", end(0.5, 90, 270f, 6f), 300, mc -> cmd(mc, KILL_DRAGON)),
        new Step("end-north", end(0.5, 90, 180f, 6f), 100, mc -> cmd(mc, KILL_DRAGON)),
        new Step("end-south", end(0.5, 90, 0f, 6f), 100, mc -> {}),
        new Step("end-west", end(0.5, 90, 90f, 6f), 100, mc -> {}),
        new Step("end-high", end(0.5, 300, 270f, 25f), 200, mc -> {}),
        new Step("end-outer", end(1500.5, 110, 270f, 8f), 600, mc -> cmd(mc, "execute in minecraft:the_end run tp @a 1500.5 110 0.5")),
        new Step("end-outer-high", end(1500.5, 300, 270f, 30f), 200, mc -> {}),
        new Step("end-back", end(0.5, 90, 270f, 6f), 300, mc -> cmd(mc, "execute in minecraft:the_end run tp @a 0.5 90 0.5"))
    );

    /**
     * View bobbing check (-PbenchTour=walk): walks north on the ground (toward open land) with bobbing on, screenshots at several points
     * of the walk cycle. Vanilla bobs by tilting its projection; the LOD has to tilt with it.
     */
    static final boolean WALK_TOUR = "walk".equals(System.getProperty("metalmc.tour"));
    private static void walk(Minecraft mc, boolean on) {
        if (mc.player != null) {
            mc.player.getAbilities().flying = false;
            mc.player.getAbilities().mayfly = false;
        }
        mc.options.keyUp.setDown(on);
    }
    static final List<Step> WALK_STEPS = List.of(
        new Step("walk-start", ground(180f), 200, mc -> {
            cmd(mc, "time set 6000", "weather clear", "gamerule advance_time false", "gamerule advance_weather false");
            mc.options.bobView().set(true);
            if (!mc.gui.hud.isHidden()) mc.gui.hud.toggle();
        }),
        new Step("walk-0", null, 30, mc -> walk(mc, true)),
        new Step("walk-1", null, 5, mc -> walk(mc, true)),
        new Step("walk-2", null, 4, mc -> walk(mc, true)),
        new Step("walk-3", null, 5, mc -> walk(mc, true)),
        new Step("walk-4", null, 4, mc -> walk(mc, true)),
        new Step("walk-end", null, 20, mc -> walk(mc, false))
    );

    /** Zoomed view (-PbenchTour=zoom): the mid pose north at field of view 30 (vanilla's minimum, 2.6x zoom). */
    static final boolean ZOOM_TOUR = "zoom".equals(System.getProperty("metalmc.tour"));
    static final List<Step> ZOOM_STEPS = List.of(
        new Step("zoom-wide", at(150, 180f, 8f), 400, mc -> {
            cmd(mc, "time set 6000", "weather clear", "gamerule advance_time false", "gamerule advance_weather false");
            if (!mc.gui.hud.isHidden()) mc.gui.hud.toggle();
        }),
        new Step("zoom-30", at(150, 180f, 8f), 120, mc -> mc.options.fov().set(30)),
        new Step("zoom-back", at(150, 180f, 8f), 20, mc -> mc.options.fov().set(70))
    );

    /**
     * TAA check (-PbenchTour=taa, with and without -Ptaa=true): rain, campfire smoke with animals nearby, third
     * person, and a fast turn (the camera turns 5 degrees per tick through the last step, screenshot mid-turn).
     */
    static final boolean TAA_TOUR = "taa".equals(System.getProperty("metalmc.tour"));
    static final List<Step> TAA_STEPS = List.of(
        new Step("taa-rain", ground(180f), 200, mc -> {
            cmd(mc, "time set 6000", "gamerule advance_time false", "gamerule advance_weather false", "weather rain");
            if (!mc.gui.hud.isHidden()) mc.gui.hud.toggle();
        }),
        new Step("taa-smoke-mobs", ground(180f), 160, mc -> cmd(mc, "weather clear",
            "execute at @p run setblock ~1 ~-3 ~-5 minecraft:campfire",
            "execute at @p run summon minecraft:horse ~3 ~-3 ~-7", "execute at @p run summon minecraft:pig ~-2 ~-3 ~-6",
            "execute at @p run summon minecraft:sheep ~0 ~-3 ~-9")),
        new Step("taa-third", ground(180f), 60, mc -> mc.options.setCameraType(net.minecraft.client.CameraType.THIRD_PERSON_BACK)),
        new Step("taa-turn", ground(180f), 30, mc -> mc.options.setCameraType(net.minecraft.client.CameraType.FIRST_PERSON))
    );

    /**
     * Shader pipeline showcase (-PbenchTour=shader, for the external pipeline, METALMC_EXTPIPE): a noon overview and a
     * ground view, the sunset, a pool of water built in the sky (open sky to reflect), a closed room lit by torches and
     * glowstone (no sky light), and the night. No HUD; each pose holds a few seconds so temporal filters settle.
     */
    static final boolean SHADER_TOUR = "shader".equals(System.getProperty("metalmc.tour"));
    private static final int WX0 = 30, WX1 = 78, WY = 196, WZ0 = -40, WZ1 = 8;      // the pool
    private static final int RX0 = -32, RX1 = -22, RY = 200, RZ0 = 28, RZ1 = 38;    // the room
    static final List<Step> SHADER_STEPS = List.of(
        new Step("noon-overview", at(190, 180f, 30f), 600, mc -> {
            cmd(mc, "time set 6000", "weather clear", "gamerule advance_time false", "gamerule advance_weather false",
                "gamerule spawn_mobs false");
            if (!mc.gui.hud.isHidden()) mc.gui.hud.toggle();
        }),
        new Step("noon-ground", ground(180f), 160, mc -> {}),
        new Step("afternoon-ground", ground(90f), 160, mc -> cmd(mc, "time set 9000")),
        new Step("sunset", ground(90f), 160, mc -> cmd(mc, "time set 12300")),
        new Step("water", new Pose(WX0 + 1.5, WY + 3.6, WZ1 - 1.5, 225f, 14f), 200, mc -> cmd(mc, "time set 7000",
            fill(WX0, WY, WZ0, WX1, WY + 2, WZ1, "minecraft:stone"),
            fill(WX0 + 1, WY + 1, WZ0 + 1, WX1 - 1, WY + 2, WZ1 - 1, "minecraft:water"))),
        new Step("interior", new Pose(RX0 + 2.5, RY + 0.2, RZ0 + 2.5, 315f, 10f), 200, mc -> cmd(mc, "time set 6000",
            fill(RX0, RY - 1, RZ0, RX1, RY + 5, RZ1, "minecraft:stone_bricks"),
            fill(RX0 + 1, RY, RZ0 + 1, RX1 - 1, RY + 4, RZ1 - 1, "minecraft:air"),
            "setblock " + (RX1 - 2) + " " + RY + " " + (RZ1 - 2) + " minecraft:glowstone",
            "setblock " + (RX1 - 1) + " " + (RY + 2) + " " + (RZ0 + 4) + " minecraft:wall_torch[facing=west]",
            "setblock " + (RX0 + 4) + " " + (RY + 2) + " " + (RZ1 - 1) + " minecraft:wall_torch[facing=north]",
            "setblock " + (RX0 + 6) + " " + RY + " " + (RZ0 + 6) + " minecraft:oak_planks",
            "setblock " + (RX0 + 5) + " " + RY + " " + (RZ0 + 6) + " minecraft:lantern")),
        new Step("night", at(190, 180f, 30f), 200, mc -> cmd(mc, "time set 18000"))
    );

    static List<Step> steps() {
        if (SHADER_TOUR) return SHADER_STEPS;
        if (TAA_TOUR) return TAA_STEPS;
        if (FIDELITY_TOUR) return FIDELITY_STEPS;
        if (ZOOM_TOUR) return ZOOM_STEPS;
        if (END_TOUR) return END_STEPS;
        if (WALK_TOUR) return WALK_STEPS;
        if (MP_TOUR) return "mp2".equals(System.getProperty("metalmc.tour")) ? MP2_STEPS : MP_STEPS;
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
        if (p == null) return false;   // the server places the player
        if (steps.get(step).name().equals("taa-turn")) p = new Pose(p.x(), p.y(), p.z(), p.yaw() + 5f * tick, p.pitch());
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
