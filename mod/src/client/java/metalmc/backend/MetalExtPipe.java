package metalmc.backend;

import com.mojang.renderpearl.api.pipeline.DepthStencilState;
import com.mojang.renderpearl.api.pipeline.ShaderType;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import com.mojang.renderpearl.backend.api.BackendRenderPipeline;
import com.mojang.renderpearl.backend.api.SpvModule;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.util.HashMap;
import java.util.Map;
import net.minecraft.client.Minecraft;
import net.minecraft.client.player.LocalPlayer;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import net.minecraft.client.renderer.state.level.SkyRenderState;
import net.minecraft.core.BlockPos;
import net.minecraft.world.effect.MobEffects;
import net.minecraft.world.item.BlockItem;
import net.minecraft.world.item.ItemStack;
import net.minecraft.world.level.LightLayer;
import net.minecraft.world.level.material.FogType;
import org.joml.Matrix4f;
import org.joml.Matrix4fc;
import org.joml.Vector3fc;
import org.joml.Vector4f;
import org.joml.Vector4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Bridge to the external pipeline (Sources/MetalMCNative/ExtPipe.swift, docs/extpipe-design.md): with METALMC_EXTPIPE
 * naming a directory with a pipeline description, the level's sky and main passes draw into that pipeline's G-buffer
 * through its programs, its full-screen passes run after the opaque and after the translucent geometry, and its last pass
 * writes the frame. This class computes the standard OptiFine/Iris uniforms each frame (camera and shadow matrices, sun
 * and moon positions, time, weather, the player's state), marks the passes and tells the native side about each of
 * vanilla's pipelines. Off unless the variable is set. Render thread only.
 */
public final class MetalExtPipe {
    private MetalExtPipe() {
    }

    public static final String DIR = System.getenv("METALMC_EXTPIPE");
    public static final boolean ENABLED = DIR != null && !DIR.isEmpty();
    /** METALMC_EXTPIPE_NOCULL=1: no view-frustum culling, for a shadow pass with the terrain all around (FrustumExtMixin). */
    public static final boolean NO_CULL = ENABLED && "1".equals(System.getenv("METALMC_EXTPIPE_NOCULL"));

    /** Resolved on first use, so nothing loads the library unless the Metal backend is running. */
    private static final class Native {
        private static final Linker LINKER = Linker.nativeLinker();
        private static final SymbolLookup LIB = SymbolLookup.libraryLookup(NativeLibrary.path(), Arena.global());

        private static MethodHandle h(String name, MemoryLayout ret, MemoryLayout... args) {
            FunctionDescriptor fd = ret == null ? FunctionDescriptor.ofVoid(args) : FunctionDescriptor.of(ret, args);
            var addr = LIB.find(name).orElseThrow(() -> new IllegalStateException("missing native symbol " + name));
            return LINKER.downcallHandle(addr, fd);
        }

        static final MethodHandle ENABLED = h("mmc_ext_enabled", JAVA_INT);
        static final MethodHandle CONST = h("mmc_ext_const", JAVA_FLOAT, JAVA_LONG, JAVA_FLOAT);
        static final MethodHandle PIPELINE_INFO = h("mmc_ext_pipeline_info", null, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT);
        static final MethodHandle FRAME_BEGIN = h("mmc_ext_frame_begin", JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_LONG);
        static final MethodHandle REDIRECT = h("mmc_ext_redirect", null, JAVA_INT);
        static final MethodHandle TRANSLUCENT = h("mmc_ext_translucent", JAVA_INT);
        static final MethodHandle FRAME_END = h("mmc_ext_frame_end", null);
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new IllegalStateException(t);
    }

    /**
     * The standard uniforms in the order the native side reads them (ExtPipe.swift, extStdLayout): name and float count.
     * Matrices column-major; integers as floats.
     */
    private static final Object[][] STD = {
        {"gbufferModelView", 16}, {"gbufferModelViewInverse", 16},
        {"gbufferPreviousModelView", 16}, {"gbufferPreviousProjection", 16},
        {"gbufferProjection", 16}, {"gbufferProjectionInverse", 16},
        {"shadowModelView", 16}, {"shadowModelViewInverse", 16},
        {"shadowProjection", 16}, {"shadowProjectionInverse", 16},
        {"cameraPosition", 3}, {"previousCameraPosition", 3},
        {"sunPosition", 3}, {"moonPosition", 3}, {"upPosition", 3}, {"shadowLightPosition", 3},
        {"skyColor", 3}, {"fogColor", 3},
        {"eyeBrightness", 2}, {"eyeBrightnessSmooth", 2},
        {"aspectRatio", 1}, {"blindness", 1}, {"darknessFactor", 1}, {"far", 1}, {"near", 1},
        {"fogMode", 1}, {"fogStart", 1}, {"fogEnd", 1}, {"fogDensity", 1},
        {"frameCounter", 1}, {"frameTime", 1}, {"frameTimeCounter", 1},
        {"heldBlockLightValue", 1}, {"heldBlockLightValue2", 1}, {"heldItemId", 1}, {"heldItemId2", 1},
        {"isEyeInWater", 1}, {"moonPhase", 1}, {"nightVision", 1}, {"rainStrength", 1},
        {"sunAngle", 1}, {"shadowAngle", 1}, {"viewHeight", 1}, {"viewWidth", 1}, {"wetness", 1},
        {"worldTime", 1}, {"worldDay", 1}, {"screenBrightness", 1}, {"eyeAltitude", 1},
        {"centerDepthSmooth", 1}, {"hideGUI", 1}, {"thunderStrength", 1}, {"playerMood", 1},
    };
    private static final Map<String, Integer> OFFSET = new HashMap<>();
    private static final int STD_COUNT;

    static {
        int at = 0;
        for (Object[] e : STD) {
            OFFSET.put((String) e[0], at);
            at += (Integer) e[1];
        }
        STD_COUNT = at;
    }

    private static final float[] values = new float[STD_COUNT];
    private static boolean nativeReady;
    private static boolean frameActive;
    private static int frameCounter;
    private static long startNanos;
    private static long lastNanos;
    private static final Matrix4f prevModelView = new Matrix4f();
    private static final Matrix4f prevProjection = new Matrix4f();
    private static double prevX, prevY, prevZ;
    private static boolean havePrev;
    private static float smoothBlock = -1, smoothSky = -1, wetness;

    /** True if the external pipeline is on and loaded (it loads on the first call). */
    public static boolean enabled() {
        if (!ENABLED || MetalDevice.current == null) return false;
        if (!nativeReady) {
            try {
                nativeReady = (int) Native.ENABLED.invokeExact() == 1;
            } catch (Throwable t) {
                throw rethrow(t);
            }
        }
        return nativeReady;
    }

    /** True between beginFrame and the end of the level's main pass. */
    public static boolean frameActive() {
        return frameActive;
    }

    private static float constant(String name, float fallback) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            return (float) Native.CONST.invokeExact(MemoryUtil.memAddress(stack.UTF8(name)), fallback);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** From MetalRenderPipeline: names of the new pipeline's uniforms and vertex inputs, and its depth test. */
    static void pipelineInfo(long handle, BackendRenderPipeline.CreateInfo info) {
        if (!ENABLED) return;
        StringBuilder uniforms = new StringBuilder();
        for (int i = 0; i < info.uniforms().size(); i++) {
            if (i > 0) uniforms.append('\n');
            uniforms.append(info.uniforms().get(i).name());
        }
        StringBuilder attribs = new StringBuilder();
        for (BackendRenderPipeline.CreateInfo.Shader s : info.shaders()) {
            if (s.module().type() != ShaderType.VERTEX) continue;
            try {
                for (SpvModule.Reflection.InterfaceVariable v : s.module().reflect().inputs()) {
                    attribs.append(v.location()).append('=').append(v.name()).append('\n');
                }
            } catch (Exception e) {
                // No names: the pipeline's draws can only go to programs that don't read vertex attributes by name.
            }
        }
        DepthStencilState depth = info.depthStencilState();
        int compare = depth != null ? MetalConst.compare(depth.depthTest()) : -1;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            ByteBuffer u = stack.UTF8(uniforms.toString());
            ByteBuffer a = stack.UTF8(attribs.toString());
            Native.PIPELINE_INFO.invokeExact(handle, MemoryUtil.memAddress(u), MemoryUtil.memAddress(a), compare, depth != null && depth.writeDepth() ? 1 : 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    private static void put(String name, float... v) {
        Integer at = OFFSET.get(name);
        if (at == null) return;
        System.arraycopy(v, 0, values, at, v.length);
    }

    private static void put(String name, Matrix4fc m) {
        Integer at = OFFSET.get(name);
        if (at != null) m.get(values, at);
    }

    /** OptiFine's view-space position of a celestial body at height `y` (Iris's CelestialUniforms). */
    private static float[] celestial(Matrix4fc modelView, float skyAngle, float sunPathRotation, float y) {
        Matrix4f m = new Matrix4f(modelView)
            .rotateY((float) Math.toRadians(-90.0))
            .rotateZ((float) Math.toRadians(sunPathRotation))
            .rotateX((float) Math.toRadians(skyAngle * 360.0));
        Vector4f p = m.transform(new Vector4f(0f, y, 0f, 0f));
        return new float[]{p.x, p.y, p.z};
    }

    /**
     * Starts the frame: the standard uniforms for this frame, and whether the level is redirected. `levelProjection` is
     * the projection vanilla draws the level with (reverse-Z, with view bobbing); the pipeline gets it in GL's convention.
     */
    public static void beginFrame(CameraRenderState cam, LevelRenderState level, Matrix4fc levelProjection, Vector4f fogColor,
                                  int width, int height, GpuTextureView blockAtlas) {
        frameActive = false;
        if (!enabled()) return;
        Minecraft mc = Minecraft.getInstance();
        long now = System.nanoTime();
        if (startNanos == 0) {
            startNanos = now;
            lastNanos = now;
        }
        float frameTime = (now - lastNanos) / 1e9f;
        lastNanos = now;

        // Camera: vanilla's view rotation is OptiFine's gbufferModelView (positions are camera-relative). The projection
        // turned to GL's depth range: its z row becomes -a times the w row plus b, which holds for any affine transform
        // vanilla multiplied in (bobbing, the hurt tilt, nausea).
        Matrix4f view = new Matrix4f(cam.viewRotationMatrix);
        float n = 0.05f, f = Math.max(cam.depthFar, 1f);
        float a = -(f + n) / (f - n), b = -2f * f * n / (f - n);
        Matrix4f proj = new Matrix4f(levelProjection);
        proj.m02(-a * levelProjection.m03());
        proj.m12(-a * levelProjection.m13());
        proj.m22(-a * levelProjection.m23());
        proj.m32(-a * levelProjection.m33() + b);
        if (!havePrev) {
            prevModelView.set(view);
            prevProjection.set(proj);
            prevX = cam.pos.x;
            prevY = cam.pos.y;
            prevZ = cam.pos.z;
            havePrev = true;
        }
        put("gbufferModelView", view);
        put("gbufferModelViewInverse", new Matrix4f(view).invert());
        put("gbufferPreviousModelView", prevModelView);
        put("gbufferPreviousProjection", prevProjection);
        put("gbufferProjection", proj);
        put("gbufferProjectionInverse", new Matrix4f(proj).invert());
        put("cameraPosition", (float) cam.pos.x, (float) cam.pos.y, (float) cam.pos.z);
        put("previousCameraPosition", (float) prevX, (float) prevY, (float) prevZ);

        // Sun, moon and the shadow camera (OptiFine's conventions, as Iris has them).
        SkyRenderState sky = level.skyRenderState;
        double turns = sky.sunAngle / (2.0 * Math.PI);
        float skyAngle = (float) (turns - Math.floor(turns));   // vanilla's celestial angle, 0 at noon
        float sunAngle = skyAngle < 0.75f ? skyAngle + 0.25f : skyAngle - 0.75f;
        boolean day = sunAngle <= 0.5f;
        float shadowAngle = day ? sunAngle : sunAngle - 0.5f;
        float sunPathRotation = constant("sunPathRotation", 0f);
        float shadowDistance = constant("shadowDistance", 160f);
        float interval = constant("shadowIntervalSize", 2f);
        float[] sun = celestial(view, skyAngle, sunPathRotation, 100f);
        float[] moon = celestial(view, skyAngle, sunPathRotation, -100f);
        Vector4f up = new Matrix4f(view).rotateY((float) Math.toRadians(-90.0)).transform(new Vector4f(0f, 100f, 0f, 0f));
        put("sunPosition", sun);
        put("moonPosition", moon);
        put("upPosition", up.x, up.y, up.z);
        put("shadowLightPosition", day ? sun : moon);
        put("sunAngle", sunAngle);
        put("shadowAngle", shadowAngle);
        float shadowSkyAngle = shadowAngle < 0.25f ? shadowAngle + 0.75f : shadowAngle - 0.25f;
        Matrix4f shadowView = new Matrix4f()
            .translate(0f, 0f, -100f)
            .rotateX((float) Math.toRadians(90.0))
            .rotateZ((float) Math.toRadians(shadowSkyAngle * -360.0))
            .rotateX((float) Math.toRadians(sunPathRotation));
        if (interval > 0f) {
            float half = interval / 2f;
            float ox = (float) (cam.pos.x % interval), oy = (float) (cam.pos.y % interval), oz = (float) (cam.pos.z % interval);
            if (ox > half) ox -= interval; else if (ox < -half) ox += interval;
            if (oy > half) oy -= interval; else if (oy < -half) oy += interval;
            if (oz > half) oz -= interval; else if (oz < -half) oz += interval;
            shadowView.translate(ox, oy, oz);
        }
        float sn = 0.05f, sf = 256f;
        Matrix4f shadowProj = new Matrix4f(
            1f / shadowDistance, 0f, 0f, 0f,
            0f, 1f / shadowDistance, 0f, 0f,
            0f, 0f, 2f / (sn - sf), 0f,
            0f, 0f, -(sf + sn) / (sf - sn), 1f);
        put("shadowModelView", shadowView);
        put("shadowModelViewInverse", new Matrix4f(shadowView).invert());
        put("shadowProjection", shadowProj);
        put("shadowProjectionInverse", new Matrix4f(shadowProj).invert());

        Vector3fc skyColor = sky.skyColor;
        if (skyColor != null) put("skyColor", skyColor.x(), skyColor.y(), skyColor.z());
        put("fogColor", fogColor.x, fogColor.y, fogColor.z);
        put("moonPhase", sky.moonPhase.index());
        float rain = 1f - sky.rainBrightness;
        put("rainStrength", rain);
        // wetnessHalflife / drynessHalflife (ticks; OptiFine's default 600).
        float halfLife = (rain > wetness ? constant("wetnessHalflife", 600f) : constant("drynessHalflife", 200f)) / 20f;
        wetness += (rain - wetness) * (1f - (float) Math.pow(0.5, frameTime / Math.max(halfLife, 1e-3f)));
        put("wetness", wetness);

        // The player and the world.
        LocalPlayer player = mc.player;
        int block = 0, skyLight = 15;
        if (mc.level != null) {
            BlockPos eye = BlockPos.containing(cam.pos.x, cam.pos.y, cam.pos.z);
            block = mc.level.getBrightness(LightLayer.BLOCK, eye);
            skyLight = mc.level.getBrightness(LightLayer.SKY, eye);
            long time = mc.level.getOverworldClockTime();
            put("worldTime", time % 24000L);
            put("worldDay", time / 24000L);
            put("thunderStrength", mc.level.getThunderLevel(1f));
        }
        put("eyeBrightness", block * 16, skyLight * 16);
        if (smoothBlock < 0) {
            smoothBlock = block * 16;
            smoothSky = skyLight * 16;
        }
        float k = 1f - (float) Math.pow(0.5, frameTime / Math.max(constant("eyeBrightnessHalflife", 10f) / 20f, 1e-3f));
        smoothBlock += (block * 16 - smoothBlock) * k;
        smoothSky += (skyLight * 16 - smoothSky) * k;
        put("eyeBrightnessSmooth", Math.round(smoothBlock), Math.round(smoothSky));
        put("isEyeInWater", cam.fogType == FogType.WATER ? 1 : cam.fogType == FogType.LAVA ? 2 : cam.fogType == FogType.POWDER_SNOW ? 3 : 0);
        float nightVision = 0f, blindness = 0f, darkness = 0f;
        int held = 0, held2 = 0;
        if (player != null) {
            nightVision = player.hasEffect(MobEffects.NIGHT_VISION) ? 1f : 0f;
            blindness = player.hasEffect(MobEffects.BLINDNESS) ? 1f : 0f;
            darkness = player.hasEffect(MobEffects.DARKNESS) ? 1f : 0f;
            held = light(player.getMainHandItem());
            held2 = light(player.getOffhandItem());
        }
        put("nightVision", nightVision);
        put("blindness", blindness);
        put("darknessFactor", darkness);
        put("heldBlockLightValue", held);
        put("heldBlockLightValue2", held2);
        put("eyeAltitude", (float) cam.pos.y);
        put("viewWidth", width);
        put("viewHeight", height);
        put("aspectRatio", width / (float) Math.max(height, 1));
        put("near", n);
        put("far", mc.options.getEffectiveRenderDistance() * 16f);
        put("fogStart", cam.fogData.environmentalStart);
        put("fogEnd", cam.fogData.environmentalEnd);
        put("frameCounter", frameCounter);
        put("frameTime", frameTime);
        put("frameTimeCounter", (float) (((now - startNanos) / 1e9) % 3600.0));
        put("screenBrightness", mc.options.gamma().get().floatValue());
        put("hideGUI", mc.gui.hud.isHidden() ? 1 : 0);
        frameCounter = (frameCounter + 1) % 720720;

        long atlas = blockAtlas instanceof MetalTextureView v ? v.handle : 0L;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer buf = stack.mallocFloat(STD_COUNT);
            buf.put(0, values);
            frameActive = (int) Native.FRAME_BEGIN.invokeExact(MemoryUtil.memAddress(buf), STD_COUNT, width, height, atlas) == 1;
        } catch (Throwable t) {
            throw rethrow(t);
        }
        prevModelView.set(view);
        prevProjection.set(proj);
        prevX = cam.pos.x;
        prevY = cam.pos.y;
        prevZ = cam.pos.z;
    }

    private static int light(ItemStack stack) {
        if (stack != null && stack.getItem() instanceof BlockItem bi) return bi.getBlock().defaultBlockState().getLightEmission();
        return 0;
    }

    /** The next render pass is the level's sky pass (1) or main pass (2): it draws into the external G-buffer. */
    public static void redirect(int kind) {
        if (!frameActive) return;
        try {
            Native.REDIRECT.invokeExact(kind);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    /** Between the main pass's opaque and translucent geometry: the deferred passes run, and the pass reopens. */
    public static void translucent() {
        if (!frameActive) return;
        int reopened;
        try {
            reopened = (int) Native.TRANSLUCENT.invokeExact();
        } catch (Throwable t) {
            throw rethrow(t);
        }
        if (reopened == 1 && MetalDevice.current != null) {
            MetalRenderPass pass = MetalDevice.current.encoder().currentRenderPass();
            if (pass != null) pass.rebindAll();
        }
    }

    /** The level is done. */
    public static void endFrame() {
        if (!frameActive) return;
        frameActive = false;
        try {
            Native.FRAME_END.invokeExact();
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
