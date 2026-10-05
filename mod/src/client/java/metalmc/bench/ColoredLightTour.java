package metalmc.bench;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import net.minecraft.client.Minecraft;
import net.minecraft.server.MinecraftServer;

/**
 * Colored block light check (-PbenchTour=coloredlight; Sources/MetalMCNative/ColoredLight.swift, docs/lighting-design.md):
 * builds the test gallery (metalmc/coloredlight_scene.txt, the same commands tools/litflow.swift applies offline) in the
 * sky, then takes screenshots at midnight from above and low over the mixing area, at noon, and sealed in (a cave: no
 * sky light inside), before and after its lights are swapped (block changes reaching the light volume). No HUD, no mobs.
 * Run it with and without METALMC_EXP=coloredlight for the before and after.
 */
final class ColoredLightTour {
    private ColoredLightTour() {}

    static final boolean ENABLED = "coloredlight".equals(System.getProperty("metalmc.tour"));

    private static final Tour.Pose OVERVIEW = new Tour.Pose(0.5, 226, 2.5, 0f, 47f);
    private static final Tour.Pose LOW = new Tour.Pose(0.5, 205.5, 15.5, 0f, 24f);
    private static final Tour.Pose CAVE = new Tour.Pose(0.5, 200.0, 14.5, 0f, 4f);
    private static final Tour.Pose CAVE_SIDE = new Tour.Pose(-22.5, 200.0, 30.5, 280f, 6f);
    /** Natural caves over lava in claudeworld-merged (found by tools/litflow.swift's LITFLOW_CL section: open around, an
     *  open line of sight to lava with air above it). */
    private static final Tour.Pose LAVA_CAVE = new Tour.Pose(24.5, 19.88, 57.5, 180f, 18f);
    private static final Tour.Pose DEEP_LAVA = new Tour.Pose(-26.5, -51.12, -75.5, 90f, 18f);

    private static void cmd(Minecraft mc, List<String> commands) {
        MinecraftServer server = mc.getSingleplayerServer();
        if (server == null) {
            if (mc.player != null) for (String c : commands) mc.player.connection.sendCommand(c);
            return;
        }
        server.execute(() -> {
            for (String c : commands) server.getCommands().performPrefixedCommand(server.createCommandSourceStack(), c);
        });
    }

    private static void cmd(Minecraft mc, String... commands) {
        cmd(mc, List.of(commands));
    }

    /** The gallery's commands (lines that aren't comments). */
    private static List<String> scene() {
        List<String> out = new ArrayList<>();
        try (InputStream in = ColoredLightTour.class.getResourceAsStream("/metalmc/coloredlight_scene.txt")) {
            if (in == null) {
                Bench.log("coloredlight tour: no scene resource");
                return out;
            }
            BufferedReader r = new BufferedReader(new InputStreamReader(in, StandardCharsets.UTF_8));
            for (String line; (line = r.readLine()) != null; ) {
                String t = line.trim();
                if (!t.isEmpty() && !t.startsWith("#")) out.add(t);
            }
        } catch (java.io.IOException e) {
            Bench.log("coloredlight tour: scene unreadable: " + e);
        }
        return out;
    }

    static final List<Tour.Step> STEPS = List.of(
        new Tour.Step("cl-build", OVERVIEW, 400, mc -> {
            List<String> c = new ArrayList<>(List.of("time set 18000", "weather clear", "gamerule advance_time false",
                "gamerule advance_weather false", "gamerule spawn_mobs false", "kill @e[type=!player]"));
            c.addAll(scene());
            cmd(mc, c);
            if (!mc.gui.hud.isHidden()) mc.gui.hud.toggle();
        }),
        new Tour.Step("cl-night", OVERVIEW, 80, mc -> cmd(mc, "kill @e[type=!player]")),
        new Tour.Step("cl-night-low", LOW, 80, mc -> {}),
        // Per-pass GPU times of three frames of the settled view (tools/bench/passes.py on the run's log).
        new Tour.Step("cl-night-low-traced", LOW, 20, mc -> metalmc.backend.MetalStats.traceFrames(3)),
        new Tour.Step("cl-noon", OVERVIEW, 80, mc -> cmd(mc, "time set 6000")),
        // Sealed in: a roof and outer walls (no sky light inside), at midnight.
        new Tour.Step("cl-cave", CAVE, 200, mc -> cmd(mc, "time set 18000",
            "fill -26 204 12 26 204 66 minecraft:stone_bricks", "fill -26 200 12 -26 203 66 minecraft:stone_bricks",
            "fill 26 200 12 26 203 66 minecraft:stone_bricks", "fill -26 200 12 26 203 12 minecraft:stone_bricks",
            "fill -26 200 66 26 203 66 minecraft:stone_bricks")),
        new Tour.Step("cl-cave-side", CAVE_SIDE, 60, mc -> {}),
        new Tour.Step("cl-cave-side-traced", CAVE_SIDE, 20, mc -> metalmc.backend.MetalStats.traceFrames(3)),
        // Block changes: the mixing area's three lights swapped around, a lava pool poured by the side.
        new Tour.Step("cl-cave-edit", CAVE, 60, mc -> cmd(mc, "setblock -6 200 26 minecraft:soul_torch",
            "setblock 0 200 26 minecraft:redstone_torch", "setblock 6 200 26 minecraft:torch",
            "fill -12 199 30 -10 199 32 minecraft:lava", "fill -12 198 30 -10 198 32 minecraft:smooth_stone")),
        new Tour.Step("cl-cave-edit-side", CAVE_SIDE, 40, mc -> {}),
        // Natural lava caves underground (no sky light): the volume follows the camera there and fills in a few frames.
        new Tour.Step("cl-lava-cave", LAVA_CAVE, 120, mc -> {}),
        new Tour.Step("cl-deep-lava", DEEP_LAVA, 120, mc -> {}),
        new Tour.Step("cl-deep-lava-traced", DEEP_LAVA, 20, mc -> metalmc.backend.MetalStats.traceFrames(3))
    );
}
