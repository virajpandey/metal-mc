package metalmc.bench;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.renderpearl.api.device.DeviceInfo;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Locale;

/**
 * Automated in-game benchmark. Launched with -Dmetalmc.bench=1: once the world loads it holds the
 * camera at the start pose to let chunks build, then flies one fixed orbit while recording the wall
 * time of every rendered frame, writes a CSV and a one-line summary, and quits the game.
 */
public final class Bench {
    static final boolean ENABLED = "1".equals(System.getProperty("metalmc.bench"));
    static final String LABEL = System.getProperty("metalmc.bench.label", "unlabeled");
    static final int WARMUP_TICKS = Integer.getInteger("metalmc.bench.warmupTicks", 400);   // 20 s
    static final int RUN_TICKS = Integer.getInteger("metalmc.bench.runTicks", 1200);        // 60 s
    static final double CENTER_X = Double.parseDouble(System.getProperty("metalmc.bench.cx", "8"));
    static final double CENTER_Z = Double.parseDouble(System.getProperty("metalmc.bench.cz", "8"));
    static final double RADIUS = Double.parseDouble(System.getProperty("metalmc.bench.radius", "140"));
    static final double HEIGHT = Double.parseDouble(System.getProperty("metalmc.bench.y", "140"));

    private enum State { WAITING, WARMUP, RUNNING, DONE }

    private static State state = State.WAITING;
    private static int tick;
    private static long lastFrameNs;
    private static final long[] frameNs = new long[400_000];
    private static int frameCount;

    private Bench() {}

    /** Called once per rendered frame (see FrameHookMixin). */
    public static void onFrame() {
        if (state != State.RUNNING) return;
        long now = System.nanoTime();
        if (lastFrameNs != 0 && frameCount < frameNs.length) frameNs[frameCount++] = now - lastFrameNs;
        lastFrameNs = now;
    }

    /** Called at the end of every client tick. */
    static void onTick(Minecraft mc) {
        if (state == State.DONE) return;
        LocalPlayer player = mc.player;
        if (mc.level == null || player == null) return;
        switch (state) {
            case WAITING -> {
                state = State.WARMUP;
                tick = 0;
                log("world loaded; warming up for " + WARMUP_TICKS + " ticks");
            }
            case WARMUP -> {
                place(player, 0);
                if (++tick >= WARMUP_TICKS) {
                    state = State.RUNNING;
                    tick = 0;
                    frameCount = 0;
                    lastFrameNs = 0;
                    log("running for " + RUN_TICKS + " ticks");
                }
            }
            case RUNNING -> {
                place(player, tick);
                if (++tick >= RUN_TICKS) {
                    state = State.DONE;
                    finish(mc);
                }
            }
            default -> { }
        }
    }

    /** Orbit pose at tick t: radius RADIUS around (CENTER_X, CENTER_Z), looking at the center, 25 degrees down. */
    private static void place(LocalPlayer p, int t) {
        double a = 2 * Math.PI * t / RUN_TICKS;
        double x = CENTER_X + RADIUS * Math.cos(a);
        double z = CENTER_Z + RADIUS * Math.sin(a);
        // Minecraft yaw: 0 faces +Z, 90 faces -X, so the direction (dx, dz) has yaw atan2(-dx, dz).
        float yaw = (float) Math.toDegrees(Math.atan2(-(CENTER_X - x), CENTER_Z - z));
        p.getAbilities().mayfly = true;
        p.getAbilities().flying = true;
        p.snapTo(x, HEIGHT, z, yaw, 25f);
        p.setDeltaMovement(0, 0, 0);
    }

    private static void finish(Minecraft mc) {
        long[] f = Arrays.copyOf(frameNs, frameCount);
        long[] sorted = f.clone();
        Arrays.sort(sorted);
        double sumMs = 0;
        for (long v : f) sumMs += v / 1e6;
        double meanMs = f.length == 0 ? 0 : sumMs / f.length;
        double p50 = pct(sorted, 50), p95 = pct(sorted, 95), p99 = pct(sorted, 99);
        double maxMs = sorted.length == 0 ? 0 : sorted[sorted.length - 1] / 1e6;
        int stutters = 0;
        for (long v : f) if (v / 1e6 > 2 * p50) stutters++;
        double seconds = sumMs / 1000;

        DeviceInfo info = RenderSystem.getDevice().getDeviceInfo();
        String summary = String.format(Locale.ROOT,
            "METALMC_BENCH label=%s backend=%s gpu=\"%s\" driver=\"%s\" frames=%d seconds=%.1f fps_mean=%.1f "
                + "ms_mean=%.3f ms_p50=%.3f ms_p95=%.3f ms_p99=%.3f ms_max=%.3f stutters_gt2x_median=%d render_distance=%d",
            LABEL, info.backendName(), info.name(), info.driverInfo(), f.length, seconds,
            seconds > 0 ? f.length / seconds : 0, meanMs, p50, p95, p99, maxMs, stutters,
            mc.options.renderDistance().get());

        try {
            Path dir = mc.gameDirectory.toPath().resolve("metalmc-bench");
            Files.createDirectories(dir);
            String stem = LABEL + "-" + System.currentTimeMillis();
            try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(dir.resolve(stem + ".csv"), StandardCharsets.UTF_8))) {
                w.println("frame,ms");
                for (int i = 0; i < f.length; i++) w.printf(Locale.ROOT, "%d,%.4f%n", i, f[i] / 1e6);
            }
            Files.writeString(dir.resolve(stem + ".txt"), summary + "\n", StandardCharsets.UTF_8);
        } catch (IOException e) {
            log("failed to write results: " + e);
        }
        System.out.println(summary);
        log("done; quitting");
        mc.stop();
    }

    private static double pct(long[] sorted, double p) {
        if (sorted.length == 0) return 0;
        int i = (int) Math.min(sorted.length - 1, Math.round(p / 100 * (sorted.length - 1)));
        return sorted[i] / 1e6;
    }

    private static void log(String msg) {
        System.out.println("[metalmc-bench] " + msg);
    }
}
