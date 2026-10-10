package metalmc.render;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.pipeline.TextureTarget;
import com.mojang.renderpearl.api.GpuFormat;
import metalmc.backend.MetalScale;

/**
 * Render scale (config renderScale, METALMC_SCALE; Sources/MetalMCNative/RenderScale.swift and TaaUpscale.swift): the
 * level drawn into a target of its own at a fraction of the main target's size, and the anti-aliasing's upscaling resolve
 * writing the main target at full size, so the hand, the HUD, the screen effects and post-processing stay at the panel's
 * resolution.
 * <p>
 * vanilla's level reaches the main target only through GameRenderer.mainRenderTarget() (LevelRenderer's frame graph, its
 * clears and composites, the sky renderer), while the hand, the screen effects, the GUI and vanilla's post effects use
 * GameRenderer's field. So from the start of the level (renderLevel) until the anti-aliasing (render3dHud), the accessor
 * answers the level's target (GameRendererScaleMixin), and our passes before the anti-aliasing take it too
 * (GameRendererLodMixin). Everything sized from it follows: the G-buffer, the relight, the traced shadows, the GI cache's
 * per-pixel work, the water's rays, the sky's quarter-resolution view and aerial perspective.
 * <p>
 * The scale is asked for at the start of every frame (a fixed setting or the governor's), and the level's target is
 * resized when it changes. Frames that can't be scaled draw at full resolution as before: without temporal
 * anti-aliasing, with vanilla's post effects (they need the level's depth at full size), outside the level, with the
 * external pipeline. Render thread.
 */
public final class RenderScale {
    private RenderScale() {
    }

    /** This frame's level is drawn into {@link #level} (decided at the start of the frame). */
    private static boolean frameScaled;
    /** From the start of the level to its anti-aliasing: GameRenderer.mainRenderTarget() answers {@link #level}. */
    private static boolean inLevel;
    private static TextureTarget level;
    private static float scale = 1f;
    private static boolean failed;

    /**
     * The start of a frame (GameRenderer.render, where it clears the main target): whether this frame's level is drawn at a
     * scale, with the level's target sized for it. {@code renderLevel}: the frame draws the level; {@code postEffects}: it
     * applies vanilla's post effects. When it returns true the main target needn't be cleared: the anti-aliasing's copy
     * writes all of its color and vanilla clears its depth before the hand.
     */
    public static boolean beginFrame(RenderTarget main, boolean renderLevel, boolean postEffects) {
        frameScaled = false;
        inLevel = false;
        if (failed || !TemporalAA.ENABLED || !renderLevel || postEffects || metalmc.backend.MetalExtPipe.ENABLED) return false;
        if (main.getColorTexture() == null || main.getDepthTexture() == null) return false;
        float s = MetalScale.next();
        if (!(s > 0f) || s >= 0.999f) return false;
        int w = Math.max(1, Math.round(main.width * s)), h = Math.max(1, Math.round(main.height * s));
        GpuFormat colorFormat = main.getColorTexture().getFormat(), depthFormat = main.getDepthTexture().getFormat();
        try {
            if (level == null || level.getColorTexture() == null || level.getDepthTexture() == null
                    || level.getColorTexture().getFormat() != colorFormat || level.getDepthTexture().getFormat() != depthFormat) {
                if (level != null) level.destroyBuffers();
                level = new TextureTarget("MetalMC level", w, h, colorFormat, depthFormat);
                System.out.println("[metalmc] render scale: the level drawn at " + w + "x" + h + " (" + s + " of " + main.width + "x" + main.height + ")");
            } else if (level.width != w || level.height != h) {
                level.resize(w, h);
            }
        } catch (RuntimeException e) {
            failed = true;
            System.err.println("[metalmc] render scale: couldn't make the level's target, drawing at full resolution: " + e);
            return false;
        }
        scale = s;
        frameScaled = true;
        return true;
    }

    /** The start of the level (GameRenderer.renderLevel): from here the main target's accessor answers the level's target. */
    public static void beginLevel() {
        inLevel = frameScaled;
    }

    /** The level's target while the level is being drawn (GameRenderer.mainRenderTarget() answers it), else null. */
    public static RenderTarget levelTarget() {
        return inLevel ? level : null;
    }

    /** The target this frame's level is drawn into: the level's own with a scale, else {@code main}. */
    public static RenderTarget level(RenderTarget main) {
        return frameScaled && level != null ? level : main;
    }

    /** True if this frame's level is drawn at a scale. */
    public static boolean scaled() {
        return frameScaled && level != null;
    }

    /** The scale this frame's level is drawn at (1 without one). */
    public static float scale() {
        return scaled() ? scale : 1f;
    }

    /** After the anti-aliasing: the main target's accessor answers the main target again. */
    public static void endLevel() {
        inLevel = false;
    }

    /**
     * The upscaling resolve failed (its shaders, or the native side declined): this frame's main target has no level, so
     * scaling stops for the rest of the session.
     */
    public static void fail() {
        if (!failed) System.err.println("[metalmc] render scale: the upscaling resolve didn't run; drawing at full resolution from now on");
        failed = true;
    }
}
