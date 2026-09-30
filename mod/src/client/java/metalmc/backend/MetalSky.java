package metalmc.backend;

import com.mojang.blaze3d.systems.RenderSystem;
import com.mojang.renderpearl.api.GpuFormat;
import com.mojang.renderpearl.api.textures.GpuTexture;
import java.nio.FloatBuffer;
import org.joml.Matrix4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

/**
 * Bridge to the native sky, atmosphere and HDR output (Sky.swift, Hdr.swift; METALMC_EXP=sky and METALMC_EXP=hdr, both
 * off by default). Render thread.
 */
public final class MetalSky {
    private MetalSky() {
    }

    private static int enabled = -1;
    private static int hdr = -1;
    private static GpuTexture snapshot;

    /** The physically based sky is on (METALMC_EXP=sky) and the Metal backend runs. */
    public static boolean enabled() {
        if (enabled < 0) {
            if (MetalDevice.current == null) return false;
            enabled = Mtl.skyEnabled();
        }
        return enabled == 1;
    }

    /** HDR output is on (METALMC_EXP=hdr) and the Metal backend runs: the main target is RGBA16Float. */
    public static boolean hdr() {
        if (hdr < 0) {
            if (MetalDevice.current == null) return false;
            hdr = Mtl.hdrEnabled();
        }
        return hdr == 1;
    }

    /**
     * Start of a level frame with the sky on, before any render pass: this frame's sun (vanilla's sun angle, radians),
     * rain brightness (1 clear), the camera's height above sea level, and the render distance's edge (fade start and
     * end, blocks; end <= start for none). Rebuilds the tables that changed. Returns true if the sky is ready to draw.
     */
    public static boolean prepare(float sunAngle, float rainBrightness, float height, float fadeStart, float fadeEnd) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer p = stack.mallocFloat(5);
            p.put(0, sunAngle).put(1, rainBrightness).put(2, height).put(3, fadeStart).put(4, fadeEnd);
            return Mtl.skyPrepare(MemoryUtil.memAddress(p)) == 1;
        }
    }

    /**
     * Before vanilla's sky pass opens: the sky at a quarter of the resolution for this frame's view, which the sky pass
     * then filters up ({@code color} is the target it'll be drawn into). Without it the pass works out every pixel.
     */
    public static boolean prepareView(GpuTexture color, Matrix4fc projection, Matrix4fc viewRotation) {
        if (MetalDevice.current == null || !(color instanceof MetalTexture c)) return false;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer p = stack.mallocFloat(32);
            projection.get(0, p);
            viewRotation.get(16, p);
            return Mtl.skyView(c.handle, MemoryUtil.memAddress(p)) == 1;
        }
    }

    /**
     * Draws the sky into the currently open render pass (vanilla's sky pass), then restores Minecraft's pipeline state.
     * {@code projection} is the one the level is drawn with. Returns false if it didn't draw (vanilla's sky then should).
     */
    public static boolean draw(Matrix4fc projection, Matrix4fc viewRotation) {
        MetalDevice device = MetalDevice.current;
        if (device == null) return false;
        MetalRenderPass pass = device.encoder().currentRenderPass();
        if (pass == null) return false;
        int drawn;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer p = stack.mallocFloat(32);
            projection.get(0, p);
            viewRotation.get(16, p);
            drawn = Mtl.skyDraw(MemoryUtil.memAddress(p));
        }
        pass.restoreAfterExternalDraw();
        return drawn == 1;
    }

    /**
     * The level just drawn into {@code color} seen through the air: aerial perspective, the render distance's edge faded
     * into the sky, the tone curve. After the ray-traced shadows, before the anti-aliasing; with {@code taa} the
     * anti-aliasing applies it as it loads the color.
     */
    public static boolean aerial(GpuTexture color, GpuTexture depth, Matrix4fc projection, Matrix4fc viewRotation, boolean taa) {
        if (MetalDevice.current == null || !(color instanceof MetalTexture c) || !(depth instanceof MetalTexture d)) return false;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer p = stack.mallocFloat(32);
            projection.get(0, p);
            viewRotation.get(16, p);
            return Mtl.skyAerial(c.handle, d.handle, MemoryUtil.memAddress(p), taa ? 1 : 0) == 1;
        }
    }

    /**
     * Screenshots with HDR: vanilla reads the main target as 8-bit RGBA, so it gets an 8-bit copy with the frame rolled
     * into SDR instead. Any other texture comes back as it is.
     */
    public static GpuTexture screenshotSource(GpuTexture source) {
        if (!hdr() || !(source instanceof MetalTexture src) || source.getFormat() != GpuFormat.RGBA16_FLOAT) return source;
        int w = source.getWidth(0), h = source.getHeight(0);
        if (snapshot == null || snapshot.isClosed() || snapshot.getWidth(0) != w || snapshot.getHeight(0) != h) {
            if (snapshot != null) snapshot.close();
            snapshot = RenderSystem.getDevice().createTexture("MetalMC HDR screenshot", GpuTexture.USAGE_COPY_SRC | GpuTexture.USAGE_COPY_DST
                | GpuTexture.USAGE_TEXTURE_BINDING | GpuTexture.USAGE_RENDER_ATTACHMENT, GpuFormat.RGBA8_UNORM, w, h, 1, 1);
        }
        if (!(snapshot instanceof MetalTexture dst)) return source;
        return Mtl.hdrSnapshot(src.handle, dst.handle) == 1 ? snapshot : source;
    }
}
