package metalmc.sky;

import metalmc.backend.MetalSky;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.fog.FogData;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import net.minecraft.client.renderer.state.level.SkyRenderState;
import net.minecraft.world.level.dimension.DimensionType;
import net.minecraft.world.level.material.FogType;
import org.joml.Matrix4f;

/**
 * Our own sky and atmosphere (METALMC_EXP=sky, off by default; Sky.swift, docs/lighting-design.md) in place of
 * vanilla's sky disc, sunrise fan, sun and distance fog.
 * <p>
 * Each level frame (SkyGameRendererMixin, at the start of GameRenderer.renderLevel) decides whether it's on: in the
 * overworld's sky, with the camera in air (not in water, lava or powder snow), no blinding effect and no boss fog.
 * Everything else keeps vanilla's sky and fog. While it's on:
 * <ul>
 *   <li>the native side rebuilds the atmosphere's tables that changed (sun, altitude, rain);</li>
 *   <li>the sky is drawn in vanilla's sky pass in place of its sky disc (SkyRendererMixin); the sunrise fan and the sun
 *       aren't drawn, the moon and stars are vanilla's;</li>
 *   <li>vanilla's distance fog, and the LOD's, which reads the same fog data, is pushed out of reach: the atmosphere's
 *       aerial perspective replaces it. The render distance's own fade goes to the native side instead, which fades the
 *       level's edge (vanilla's chunks, or the LOD's when it's on) into the sky behind it rather than into vanilla's fog
 *       color;</li>
 *   <li>after the level, aerial perspective and the tone curve (GameRendererLodMixin, before the anti-aliasing).</li>
 * </ul>
 */
public final class Sky {
    private Sky() {
    }

    /** Whether our sky replaces vanilla's this frame (decided at the start of each level frame). */
    public static volatile boolean frameActive;
    /** Whether it drew in this frame's sky pass (if not, vanilla's sunrise fan and sun stay). */
    public static volatile boolean drawn;
    /** The view rotation the level is drawn with this frame (the projection is Lod.LEVEL_PROJECTION). */
    public static final Matrix4f VIEW_ROTATION = new Matrix4f();
    /** Fog start and end that no pixel reaches: vanilla's shaders and the LOD's stop fogging. */
    private static final float NO_FOG = 1e6f;

    /** Start of GameRenderer.renderLevel, before any of the level's passes. */
    public static void beginLevel(LevelRenderState level) {
        frameActive = false;
        drawn = false;
        if (!MetalSky.enabled()) return;
        Minecraft mc = Minecraft.getInstance();
        CameraRenderState cam = level.cameraRenderState;
        SkyRenderState sky = level.skyRenderState;
        boolean inAir = cam.fogType == FogType.NONE || cam.fogType == FogType.ATMOSPHERIC;
        if (mc.level == null || sky.skybox != DimensionType.Skybox.OVERWORLD || !inAir || cam.entityRenderState.doesMobEffectBlockSky
            || mc.gui.hud.getBossOverlay().shouldCreateWorldFog()) return;
        FogData fog = cam.fogData;
        float height = (float) (cam.pos.y - mc.level.getSeaLevel());
        if (!MetalSky.prepare(sky.sunAngle, sky.rainBrightness, height, fog.renderDistanceStart, fog.renderDistanceEnd)) return;
        VIEW_ROTATION.set(cam.viewRotationMatrix);
        fog.environmentalStart = NO_FOG - 1;
        fog.environmentalEnd = NO_FOG;
        fog.renderDistanceStart = NO_FOG - 1;
        fog.renderDistanceEnd = NO_FOG;
        frameActive = true;
    }
}
