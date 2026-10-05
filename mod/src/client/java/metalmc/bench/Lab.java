package metalmc.bench;

import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.server.MinecraftServer;

import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.time.LocalTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;

/**
 * Lab mode (-PbenchLab=1, tools/bench/lab.sh; docs/lab-mode.md): once the world has loaded the game stays up and runs the
 * commands appended to run/metalmc-control.txt, one at a time. Each tick it takes the file's lines into its queue (the
 * file empties; run/metalmc-control.pending lists what hasn't started). Results go to run/metalmc-control.log and the game
 * log ([metalmc-lab]). Lines starting with # are skipped.
 * <pre>
 *   scene NAME        go to a scene of tools/bench/scenes.json (pose, time of day, weather, setup commands) and wait until
 *                     the terrain around it has loaded and the LOD has settled
 *   shot LABEL        screenshot into run/screenshots/LABEL.png (done once the file is written)
 *   reload            recompile every shader file now (-PshaderDir) and rebuild their pipelines
 *   wait TICKS        do nothing for TICKS ticks (20 a second)
 *   tour [PREFIX]     scene + shot for every scene: PREFIX-NAME.png (PREFIX: tour)
 *   quit              stop the game
 *   time TICKS, weather clear|rain|thunder, tp X Y Z [YAW PITCH], free (stop holding the pose), cmd COMMAND (any server
 *   command), hud on|off, fov DEGREES, status, scenes, echo TEXT
 * </pre>
 */
final class Lab {
    private Lab() {}

    static final boolean ENABLED = "1".equals(System.getProperty("metalmc.lab"));
    private static final Path SCENES = Path.of(System.getProperty("metalmc.lab.scenes", "../../tools/bench/scenes.json"));
    private static final DateTimeFormatter CLOCK = DateTimeFormatter.ofPattern("HH:mm:ss.SSS");

    private static Path file, log, pendingFile;
    /** Commands made by other commands (tour), run before the file's. */
    private static final ArrayDeque<String> front = new ArrayDeque<>();
    /** Lines taken from the control file and not run yet (listed in run/metalmc-control.pending). */
    private static final ArrayDeque<String> inbox = new ArrayDeque<>();
    /** The control file as it was taken (renamed away whole), read a tick later. */
    private static Path taken;
    /** The command in progress (one that takes ticks), its number and text. */
    private static Running current;
    private static int count;
    /** The pose held every tick (scene, tp), until `free`. */
    private static double[] hold;
    private static boolean quit;

    /** A command that takes ticks: tick() returns its result once done, null until then. */
    private interface Running {
        String tick(Minecraft mc);
    }

    private record Command(int n, String text) {}
    private static Command running;

    /** World loaded: the defaults for repeatable pictures, and the ready line. */
    static void start(Minecraft mc) {
        Path dir = mc.gameDirectory.toPath();
        file = dir.resolve("metalmc-control.txt");
        log = dir.resolve("metalmc-control.log");
        pendingFile = dir.resolve("metalmc-control.pending");
        server(mc, "gamerule advance_time false", "gamerule advance_weather false", "gamerule spawn_mobs false");
        if (!mc.gui.hud.isHidden()) mc.gui.hud.toggle();   // F1: no HUD, no hand in the pictures
        String shaders = System.getenv("METALMC_SHADERDIR");
        Map<String, JsonObject> scenes = scenes();
        say("ready: commands from " + file + "; " + (scenes == null ? "no scenes file at " + SCENES : scenes.size() + " scenes in " + SCENES)
            + "; shaders " + (shaders == null || shaders.isEmpty() ? "built in (no -PshaderDir)" : "from " + shaders));
    }

    /** Every tick in lab mode. Returns true once `quit` ran. */
    static boolean onTick(Minecraft mc, LocalPlayer player) {
        if (hold != null) place(player, hold);
        if (quit) return true;
        take();
        for (int i = 0; i < 32; i++) {   // commands that finish at once run back to back
            if (current != null) {
                String result;
                try {
                    result = current.tick(mc);
                } catch (RuntimeException e) {
                    result = "error: " + e;
                }
                if (result == null) return false;
                current = null;
                done(running, result);
            }
            String line = next();
            if (line == null) return false;
            running = new Command(++count, line);
            String result;
            try {
                result = run(mc, player, line);
            } catch (RuntimeException e) {
                result = "error: " + e;
            }
            if (result != null) done(running, result);
            if (quit) return true;
        }
        return false;
    }

    /** Starts a command; returns its result if it finished at once, or null with `current` set. */
    private static String run(Minecraft mc, LocalPlayer player, String line) {
        String[] w = line.trim().split("\\s+", 2);
        String verb = w[0].toLowerCase(Locale.ROOT), arg = w.length > 1 ? w[1].trim() : "";
        switch (verb) {
            case "scene": return scene(mc, arg);
            case "shot": return shot(mc, arg);
            case "reload": return metalmc.backend.MetalLab.reloadShaders();
            case "wait": {
                int[] left = {Math.max(0, Integer.parseInt(arg))};
                current = m -> left[0]-- > 0 ? null : "ok";
                return null;
            }
            case "tour": {
                Map<String, JsonObject> scenes = scenes();
                if (scenes == null) return "error: no scenes file at " + SCENES;
                String prefix = arg.isEmpty() ? "tour" : arg;
                // Queued in front, in order: scene a, shot PREFIX-a, scene b, ...
                var lines = new java.util.ArrayList<String>();
                for (String name : scenes.keySet()) {
                    lines.add("scene " + name);
                    lines.add("shot " + prefix + "-" + name);
                }
                for (int i = lines.size() - 1; i >= 0; i--) front.addFirst(lines.get(i));
                pending();
                return "ok: " + scenes.size() + " scenes queued";
            }
            case "time": server(mc, "time set " + Integer.parseInt(arg)); return "ok";
            case "weather": server(mc, "weather " + arg); return "ok";
            case "tp": {
                double[] v = Arrays.stream(arg.split("\\s+")).mapToDouble(Double::parseDouble).toArray();
                if (v.length != 3 && v.length != 5) return "error: tp X Y Z [YAW PITCH]";
                hold = new double[]{v[0], v[1], v[2], v.length == 5 ? v[3] : player.getYRot(), v.length == 5 ? v[4] : player.getXRot()};
                server(mc, String.format(Locale.ROOT, "tp @a %.3f %.3f %.3f %.2f %.2f", hold[0], hold[1], hold[2], hold[3], hold[4]));
                return "ok";
            }
            case "free": hold = null; return "ok";
            case "cmd": server(mc, arg); return "ok (sent to the server)";
            case "hud": {
                boolean on = arg.equals("on");
                if (mc.gui.hud.isHidden() == on) mc.gui.hud.toggle();
                return "ok";
            }
            case "fov": mc.options.fov().set(Integer.parseInt(arg)); return "ok";
            case "echo": return arg;
            case "scenes": {
                Map<String, JsonObject> scenes = scenes();
                return scenes == null ? "error: no scenes file at " + SCENES : String.join(" ", scenes.keySet());
            }
            case "status": return status(mc, player);
            case "quit": quit = true; mc.stop(); return "ok";
            default: return "error: unknown command (scene, shot, reload, wait, tour, quit, time, weather, tp, free, cmd, hud, fov, status, scenes, echo)";
        }
    }

    private static String scene(Minecraft mc, String name) {
        Map<String, JsonObject> scenes = scenes();
        if (scenes == null) return "error: no scenes file at " + SCENES;
        JsonObject s = scenes.get(name);
        if (s == null) return "error: no scene " + name + " (" + String.join(" ", scenes.keySet()) + ")";
        JsonArray pos = s.getAsJsonArray("pos");
        double fromX = mc.player == null ? 0 : mc.player.getX(), fromZ = mc.player == null ? 0 : mc.player.getZ();
        hold = new double[]{pos.get(0).getAsDouble(), pos.get(1).getAsDouble(), pos.get(2).getAsDouble(),
            s.has("yaw") ? s.get("yaw").getAsDouble() : 0, s.has("pitch") ? s.get("pitch").getAsDouble() : 0};
        // A jump past a LOD node or two: the LOD's update passes (every 2 s) have to move it before it counts as settled.
        boolean far = Math.hypot(hold[0] - fromX, hold[2] - fromZ) > 256;
        int time = s.has("time") ? s.get("time").getAsInt() : 6000;
        String weather = s.has("weather") ? s.get("weather").getAsString() : "clear";
        server(mc, String.format(Locale.ROOT, "tp @a %.3f %.3f %.3f %.2f %.2f", hold[0], hold[1], hold[2], hold[3], hold[4]),
            "time set " + time, "weather " + weather);
        // The setup's commands run once the server has loaded the chunks there: 2 s after the jump, and again at 5 s for a
        // slow load (a setblock in a chunk that isn't loaded does nothing; in the same tick as the jump none are).
        var setup = new java.util.ArrayList<String>();
        if (s.has("setup")) for (JsonElement e : s.getAsJsonArray("setup")) setup.add(e.getAsString());
        if (s.has("fov")) mc.options.fov().set(s.get("fov").getAsInt());
        // Settled: at least `settle` ticks (rain takes about 5 s to fade in; the GI cache and the anti-aliasing's history a
        // few seconds to converge); the LOD built and its quad count steady for 3 s (after a far jump, only once it has
        // changed: it can hold still for the 2 s before its first update pass; or after 20 s); and vanilla's sections
        // around the camera compiled (or 5 s more: with our near chunks vanilla's count may never say so). At most a minute.
        int min = Math.max(s.has("settle") ? s.get("settle").getAsInt() : 100, setup.isEmpty() ? 0 : 160), max = Math.max(min, 1200);
        String what = String.format(Locale.ROOT, "%.1f %.1f %.1f yaw %.1f pitch %.1f, time %d, %s", hold[0], hold[1], hold[2],
            hold[3], hold[4], time, weather);
        int[] t = {0, 0, 0};   // ticks, ticks the quad count held, times it changed
        long[] lastQuads = {-1};
        current = m -> {
            t[0]++;
            if (!setup.isEmpty() && (t[0] == 40 || t[0] == 100)) server(m, setup.toArray(new String[0]));
            boolean terrain = m.levelRenderer.hasRenderedAllSections() && m.levelRenderer.sectionRenderDispatcher().isQueueEmpty();
            boolean lod = true;
            if (metalmc.lod.Lod.ENABLED && metalmc.backend.MetalLod.available()) {
                long q = metalmc.backend.MetalLod.status()[2];
                if (lastQuads[0] >= 0 && q != lastQuads[0]) {
                    t[1] = 0;
                    t[2]++;
                } else {
                    t[1]++;
                }
                lastQuads[0] = q;
                lod = metalmc.lod.Lod.built() && t[1] >= 60 && (!far || t[2] > 0 || t[0] >= 400);
            }
            if (t[0] >= min && lod && (terrain || t[0] >= min + 100)) {
                return "ok: " + what + "; settled in " + t[0] + " ticks" + (terrain ? "" : " (vanilla's sections still compiling)");
            }
            if (t[0] >= max) return "ok: " + what + "; not settled after " + t[0] + " ticks (terrain " + (terrain ? "loaded" : "loading")
                + ", LOD " + (lod ? "steady" : "changing") + ")";
            return null;
        };
        return null;
    }

    private static String shot(Minecraft mc, String label) {
        String name = label.replaceAll("[^A-Za-z0-9._-]", "_");
        if (name.isEmpty()) return "error: shot LABEL";
        if (!name.endsWith(".png")) name += ".png";
        Path out = mc.gameDirectory.toPath().resolve("screenshots").resolve(name);
        String[] message = {null};
        net.minecraft.client.Screenshot.grab(mc.gameDirectory, name, mc.gameRenderer.mainRenderTarget(), 1,
            m -> message[0] = m.getString());
        int[] t = {0};
        current = m -> {
            if (message[0] != null) return (Files.exists(out) ? "ok: " + out.toAbsolutePath().normalize() : "error: " + message[0]);
            return ++t[0] > 400 ? "error: no screenshot after 20 s" : null;
        };
        return null;
    }

    private static String status(Minecraft mc, LocalPlayer p) {
        String lod = "off";
        if (metalmc.lod.Lod.ENABLED && metalmc.backend.MetalLod.available()) {
            long[] s = metalmc.backend.MetalLod.status();
            lod = (metalmc.lod.Lod.built() ? "built" : "building") + ", " + s[1] + " nodes, " + s[2] + " quads";
        }
        return String.format(Locale.ROOT, "ok: at %.1f %.1f %.1f yaw %.1f pitch %.1f, day time %d, rain %.2f, %d fps, %s, LOD %s, %s",
            p.getX(), p.getY(), p.getZ(), p.getYRot(), p.getXRot(), mc.level == null ? -1 : mc.level.getDefaultClockTime() % 24000,
            mc.level == null ? 0 : mc.level.getRainLevel(1f), mc.getFps(), hold == null ? "free" : "pose held",
            lod, mc.getWindow().getWidth() + "x" + mc.getWindow().getHeight());
    }

    /** The scenes file, read again each time (edits apply at once), by name in file order; null if it's missing or bad. */
    private static Map<String, JsonObject> scenes() {
        try {
            JsonObject root = JsonParser.parseString(Files.readString(SCENES)).getAsJsonObject();
            Map<String, JsonObject> out = new LinkedHashMap<>();
            for (JsonElement e : root.getAsJsonArray("scenes")) {
                JsonObject s = e.getAsJsonObject();
                out.put(s.get("name").getAsString(), s);
            }
            return out;
        } catch (IOException | RuntimeException e) {
            say("scenes file " + SCENES + ": " + e);
            return null;
        }
    }

    private static void place(LocalPlayer p, double[] pose) {
        p.getAbilities().mayfly = true;
        p.getAbilities().flying = true;
        p.snapTo(pose[0], pose[1], pose[2], (float) pose[3], (float) pose[4]);
        p.setDeltaMovement(0, 0, 0);
    }

    private static void server(Minecraft mc, String... commands) {
        MinecraftServer server = mc.getSingleplayerServer();
        if (server == null) {
            if (mc.player != null) for (String c : commands) mc.player.connection.sendCommand(c);
            return;
        }
        server.execute(() -> {
            for (String c : commands) server.getCommands().performPrefixedCommand(server.createCommandSourceStack(), c);
        });
    }

    /** The next command: one made by another command, or the control file's next line. */
    private static String next() {
        String s = !front.isEmpty() ? front.pollFirst() : inbox.pollFirst();
        if (s != null) pending();
        return s;
    }

    /**
     * Every tick: takes the control file's lines into the queue without ever rewriting the file. The file is renamed away
     * whole and read a tick later: an `echo ... >>` that opened it before the rename writes into the renamed file, and
     * those writes have landed by then; one that opens it after creates a new file, taken next time. (Rewriting the file
     * to drop the line just run lost lines appended at the same moment: three of seventeen, appended in a loop.)
     */
    private static void take() {
        try {
            if (taken != null) {
                for (String l : Files.readAllLines(taken, StandardCharsets.UTF_8)) {
                    String s = l.strip();
                    if (!s.isEmpty() && !s.startsWith("#")) inbox.add(s);
                }
                Files.deleteIfExists(taken);
                taken = null;
                pending();
            }
            if (Files.exists(file) && Files.size(file) > 0) {
                Path t = file.resolveSibling(file.getFileName() + ".taken");
                Files.move(file, t, StandardCopyOption.ATOMIC_MOVE);
                taken = t;
            }
        } catch (IOException e) {
            say("control file " + file + ": " + e);
        }
    }

    /** run/metalmc-control.pending: the commands queued and not started yet, in order. */
    private static void pending() {
        StringBuilder b = new StringBuilder();
        for (String s : front) b.append(s).append('\n');
        for (String s : inbox) b.append(s).append('\n');
        try {
            Files.writeString(pendingFile, b.toString(), StandardCharsets.UTF_8);
        } catch (IOException e) {
            say("can't write " + pendingFile + ": " + e);
        }
    }

    private static void done(Command c, String result) {
        say("#" + c.n() + " " + c.text() + " -> " + result);
    }

    /** The game log and run/metalmc-control.log. */
    static void say(String s) {
        String line = LocalTime.now().format(CLOCK) + " " + s;
        System.out.println("[metalmc-lab] " + line);
        if (log == null) return;
        try {
            Files.writeString(log, line + "\n", StandardCharsets.UTF_8, StandardOpenOption.CREATE, StandardOpenOption.APPEND);
        } catch (IOException e) {
            System.out.println("[metalmc-lab] can't write " + log + ": " + e);
        }
    }
}
