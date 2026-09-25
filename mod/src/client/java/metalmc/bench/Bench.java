package metalmc.bench;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.renderpearl.api.device.DeviceInfo;
import com.mojang.renderpearl.api.device.GpuSurface;
import net.minecraft.client.Minecraft;
import net.minecraft.client.gui.components.debug.DebugScreenEntries;
import net.minecraft.client.gui.components.debug.DebugScreenEntryStatus;
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
    /** Vanilla's GPU timer needs its debug-screen line enabled, which itself costs frame time; opt-in. */
    static final boolean GPU_TIMER = "1".equals(System.getProperty("metalmc.bench.gpuTimer", "0"));
    static final double CENTER_X = Double.parseDouble(System.getProperty("metalmc.bench.cx", "8"));
    static final double CENTER_Z = Double.parseDouble(System.getProperty("metalmc.bench.cz", "8"));
    static final double RADIUS = Double.parseDouble(System.getProperty("metalmc.bench.radius", "140"));
    // "ground" (-PbenchY=ground) flies the orbit 2 blocks above the terrain, looking nearly level.
    static final boolean GROUND = "ground".equals(System.getProperty("metalmc.bench.y"));
    static final double HEIGHT = GROUND ? Double.NaN : Double.parseDouble(System.getProperty("metalmc.bench.y", "140"));

    /** Screenshots at the start and middle of the run (tests readback too); off with -Dmetalmc.bench.screenshots=0. */
    static final boolean SCREENSHOTS = !"0".equals(System.getProperty("metalmc.bench.screenshots", "1"));

    /** Rendering coverage tour instead of the timed orbit (see Tour). */
    static final boolean TOUR = "1".equals(System.getProperty("metalmc.tour")) || "lod".equals(System.getProperty("metalmc.tour"))
        || Tour.MP_TOUR;

    private enum State { WAITING, WARMUP, RUNNING, TOUR, PREGEN, DONE }

    private static State state = State.WAITING;
    private static int tick;
    private static long lastFrameNs;
    private static final long[] frameNs = new long[400_000];
    /** Vanilla's GPU utilization (% of the profiled frame's CPU duration), sampled every frame. */
    private static final double[] gpuUtil = new double[400_000];
    private static int frameCount;

    private Bench() {}

    /** Called once per rendered frame (see FrameHookMixin). */
    public static void onFrame() {
        if (state != State.RUNNING) return;
        long now = System.nanoTime();
        if (lastFrameNs != 0 && frameCount < frameNs.length) {
            gpuUtil[frameCount] = Minecraft.getInstance().getGpuUtilization();
            frameNs[frameCount++] = now - lastFrameNs;
        }
        lastFrameNs = now;
    }

    /** Called at the end of every client tick. */
    static void onTick(Minecraft mc) {
        if (state == State.DONE) return;
        LocalPlayer player = mc.player;
        if (mc.level == null || player == null) return;
        switch (state) {
            case WAITING -> {
                if (Pregen.RADIUS > 0) {
                    state = State.PREGEN;
                    Pregen.start(mc);
                    return;
                }
                state = State.WARMUP;
                tick = 0;
                if ("1".equals(System.getProperty("metalmc.bench.noon"))) {
                    // Deterministic conditions for fixtures whose saved weather or time drifted (e.g. after pregen).
                    net.minecraft.server.MinecraftServer server = mc.getSingleplayerServer();
                    if (server != null) server.execute(() -> {
                        for (String c : new String[]{"time set 6000", "weather clear", "gamerule advance_time false", "gamerule advance_weather false"}) {
                            server.getCommands().performPrefixedCommand(server.createCommandSourceStack(), c);
                        }
                    });
                }
                // Vanilla only runs its GPU timer query while this debug entry is enabled. The status is saved
                // to disk, so set it explicitly either way.
                mc.debugEntries.setStatus(DebugScreenEntries.GPU_UTILIZATION,
                    GPU_TIMER ? DebugScreenEntryStatus.ALWAYS_ON : DebugScreenEntryStatus.NEVER);
                log("world loaded; warming up for " + WARMUP_TICKS + " ticks; " + presentInfo(mc));
            }
            case WARMUP -> {
                if (!Tour.MP_TOUR) place(player, 0);
                if (++tick >= WARMUP_TICKS && TOUR) {
                    state = State.TOUR;
                    tick = 0;
                    log("starting rendering tour (" + Tour.steps().size() + " steps)");
                } else if (tick >= WARMUP_TICKS) {
                    state = State.RUNNING;
                    tick = 0;
                    frameCount = 0;
                    lastFrameNs = 0;
                    log("running for " + RUN_TICKS + " ticks");
                    if (isMetal()) {
                        metalmc.backend.MetalStats.takeGpuMillis();
                        metalmc.backend.MetalStats.takeCounters();
                    }
                    screenshot(mc, "start");
                }
            }
            case RUNNING -> {
                place(player, tick);
                if (tick == RUN_TICKS / 2) screenshot(mc, "mid");
                if (tick == RUN_TICKS / 4 && isMetal() && "1".equals(System.getProperty("metalmc.bench.trace"))) {
                    metalmc.backend.MetalStats.traceFrames(3);
                }
                if (++tick >= RUN_TICKS) {
                    state = State.DONE;
                    finish(mc);
                }
            }
            case PREGEN -> {
                place(player, 0);
                if (Pregen.done()) {
                    state = State.DONE;
                    log("pregen done; quitting");
                    mc.stop();
                }
            }
            case TOUR -> {
                // After the last step, give the screenshot readbacks and file writes a moment, then quit.
                if (Tour.onTick(mc, player) && ++tick > 60) {
                    state = State.DONE;
                    log("tour done; quitting");
                    mc.stop();
                }
            }
            default -> { }
        }
    }

    private static void screenshot(Minecraft mc, String tag) {
        if (SCREENSHOTS) screenshotNow(mc, tag);
    }

    static void screenshotNow(Minecraft mc, String tag) {
        String name = LABEL + "-" + tag + ".png";
        net.minecraft.client.Screenshot.grab(mc.gameDirectory, name, mc.gameRenderer.mainRenderTarget(), 1,
            message -> log("screenshot " + name + ": " + message.getString()));
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
        if (GROUND) {
            int top = p.level().getHeight(net.minecraft.world.level.levelgen.Heightmap.Types.MOTION_BLOCKING, (int) Math.floor(x), (int) Math.floor(z));
            p.snapTo(x, top + 2, z, yaw, 3f);
        } else {
            p.snapTo(x, HEIGHT, z, yaw, 25f);
        }
        p.setDeltaMovement(0, 0, 0);
    }

    private static boolean isMetal() {
        return "Metal".equals(RenderSystem.getDevice().getDeviceInfo().backendName());
    }

    private static double pctD(double[] sorted, int p) {
        return sorted.length == 0 ? 0 : sorted[(int) Math.round(p / 100.0 * (sorted.length - 1))];
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
        // GPU time estimate per frame = utilization% x frame time (utilization is relative to CPU frame duration).
        double[] gpuMs = new double[f.length];
        double utilSum = 0;
        for (int i = 0; i < f.length; i++) {
            utilSum += gpuUtil[i];
            gpuMs[i] = gpuUtil[i] / 100.0 * (f[i] / 1e6);
        }
        double[] gpuSorted = gpuMs.clone();
        Arrays.sort(gpuSorted);
        double gpuMean = 0;
        for (double v : gpuMs) gpuMean += v;
        gpuMean = f.length == 0 ? 0 : gpuMean / f.length;
        double utilMean = f.length == 0 ? 0 : utilSum / f.length;
        double gpuP95 = gpuSorted.length == 0 ? 0 : gpuSorted[(int) Math.round(0.95 * (gpuSorted.length - 1))];

        DeviceInfo info = RenderSystem.getDevice().getDeviceInfo();
        String metalGpu = "";
        if (isMetal()) {
            // Measured GPU time per submit (one submit per frame): Metal command buffer gpuStart..gpuEnd.
            double[] g = metalmc.backend.MetalStats.takeGpuMillis();
            double[] gs = g.clone();
            Arrays.sort(gs);
            double gm = 0;
            for (double v : g) gm += v;
            gm = g.length == 0 ? 0 : gm / g.length;
            long[] c = metalmc.backend.MetalStats.takeCounters();
            double sub = Math.max(1, c[4]);
            metalGpu = String.format(Locale.ROOT, " metal_gpu_submits=%d metal_gpu_ms_mean=%.3f metal_gpu_ms_p50=%.3f metal_gpu_ms_p95=%.3f metal_gpu_ms_p99=%.3f"
                    + " per_frame_passes=%.1f per_frame_draws=%.1f per_frame_blits=%.1f per_frame_clears=%.1f per_frame_pass_mpix=%.1f"
                    + " per_frame_lod_draws=%.1f per_frame_lod_kquads=%.1f per_frame_lod_cpu_ms=%.3f",
                g.length, gm, pctD(gs, 50), pctD(gs, 95), pctD(gs, 99),
                c[0] / sub, c[1] / sub, c[2] / sub, c[3] / sub, c[5] / sub / 1e6,
                c[6] / sub, c[7] / sub / 1e3, c[8] / sub / 1e6);
        }
        String summary = String.format(Locale.ROOT,
            "METALMC_BENCH label=%s backend=%s gpu=\"%s\" driver=\"%s\" frames=%d seconds=%.1f fps_mean=%.1f "
                + "ms_mean=%.3f ms_p50=%.3f ms_p95=%.3f ms_p99=%.3f ms_max=%.3f stutters_gt2x_median=%d "
                + "gpu_timer=%s gpu_util_mean=%.1f gpu_ms_est_mean=%.3f gpu_ms_est_p95=%.3f render_distance=%d "
                + "fullscreen=%s window=%dx%d %s facing_culling=%s lod=%s lod_far=%d",
            LABEL, info.backendName(), info.name(), info.driverInfo(), f.length, seconds,
            seconds > 0 ? f.length / seconds : 0, meanMs, p50, p95, p99, maxMs, stutters,
            GPU_TIMER, utilMean, gpuMean, gpuP95, mc.options.renderDistance().get(),
            mc.options.fullscreen().get(), mc.getWindow().getWidth(), mc.getWindow().getHeight(), presentInfo(mc), metalmc.terrain.FacingSorter.ENABLED,
            metalmc.lod.Lod.active(), metalmc.lod.Lod.ENABLED ? metalmc.lod.Lod.FAR : 0) + metalGpu;

        try {
            Path dir = mc.gameDirectory.toPath().resolve("metalmc-bench");
            Files.createDirectories(dir);
            String stem = LABEL + "-" + System.currentTimeMillis();
            try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(dir.resolve(stem + ".csv"), StandardCharsets.UTF_8))) {
                w.println("frame,ms,gpu_util_pct,gpu_ms_est");
                for (int i = 0; i < f.length; i++) {
                    w.printf(Locale.ROOT, "%d,%.4f,%.2f,%.4f%n", i, f[i] / 1e6, gpuUtil[i], gpuMs[i]);
                }
            }
            Files.writeString(dir.resolve(stem + ".txt"), summary + "\n", StandardCharsets.UTF_8);
        } catch (IOException e) {
            log("failed to write results: " + e);
        }
        System.out.println(summary);
        log("done; quitting");
        mc.stop();
    }

    /** The swapchain present mode Minecraft picks: the first of its preference list that the surface supports. */
    private static String presentInfo(Minecraft mc) {
        try {
            var modes = mc.windowSurface().supportedPresentModes();
            var chosen = GpuSurface.PresentMode.getSupportedVsyncMode(modes, mc.options.enableVsync().get());
            return "present_supported=" + modes.toString().replace(" ", "") + " present=" + chosen;
        } catch (RuntimeException e) {
            return "present=unknown(" + e.getClass().getSimpleName() + ")";
        }
    }

    private static double pct(long[] sorted, double p) {
        if (sorted.length == 0) return 0;
        int i = (int) Math.min(sorted.length - 1, Math.round(p / 100 * (sorted.length - 1)));
        return sorted[i] / 1e6;
    }

    static void log(String msg) {
        System.out.println("[metalmc-bench] " + msg);
    }
}
