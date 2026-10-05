package metalmc.lod.mixin;

import com.mojang.blaze3d.pipeline.RenderTarget;
import metalmc.lod.Lod;
import metalmc.render.TemporalAA;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.state.OptionsRenderState;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.PlayerRenderState;
import org.joml.Matrix4f;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyArg;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * The projection vanilla draws the level with: the camera's, times view bobbing, the hurt tilt and the nausea and
 * portal distortion (GameRenderer.renderLevel works on a copy). The LOD and the section occlusion test must use the
 * same one, or while the player walks the LOD sways against vanilla's terrain (bobbing rolls the view up to 3
 * degrees and pitches it up to 5) and the occlusion boxes miss the depth buffer.
 * <p>
 * With temporal anti-aliasing on, this is also where the level's projection gets its sub-pixel jitter, and where
 * the anti-aliasing runs: after the level, before the hand and the screen effects.
 */
@Mixin(GameRenderer.class)
abstract class GameRendererLodMixin {
    @Shadow
    @Final
    private RenderTarget mainRenderTarget;

    @ModifyArg(method = "renderLevel", at = @At(value = "INVOKE",
        target = "Lnet/minecraft/client/renderer/ProjectionMatrixBuffer;getBuffer(Lorg/joml/Matrix4f;)Lcom/mojang/renderpearl/api/buffers/GpuBufferSlice;"))
    private Matrix4f metalmc$levelProjection(Matrix4f projection) {
        if (TemporalAA.ENABLED) TemporalAA.jitter(projection, mainRenderTarget.width, mainRenderTarget.height);
        Lod.LEVEL_PROJECTION.set(projection);
        return projection;
    }

    /**
     * No section fade-in while the LOD draws. Vanilla fades a new section in from the fog color over
     * chunkSectionFadeInTime (0.75 s by default), which hides pop-in at the edge of its render distance. With the
     * LOD that edge is mid-landscape and the LOD already shows the terrain, so fading sections flashed as fog-colored
     * ghosts while flying; drawn at once, a section replaces the LOD's nearly identical level-0 geometry.
     */
    @Inject(method = "extractOptions", at = @At("TAIL"))
    private void metalmc$noSectionFade(CallbackInfo ci) {
        if (Lod.active()) ((GameRenderer) (Object) this).gameRenderState().optionsRenderState.chunkSectionFadeInTime = 0.0;
    }

    @Inject(method = "render3dHud", at = @At("HEAD"))
    private void metalmc$temporalAA(CameraRenderState cameraState, PlayerRenderState playerState, OptionsRenderState optionsState,
                                    boolean consistentDepthRequired, CallbackInfo ci) {
        if (mainRenderTarget.getColorTexture() == null || mainRenderTarget.getDepthTexture() == null) return;
        if (Lod.active() && Lod.SUN_SKY) {
            // Full strength while the sun is more than about 6 degrees up, fading out as it sets; weaker in rain.
            float strength = 0.42f * Math.clamp((float) Math.cos(Lod.SUN_ANGLE) * 10f, 0f, 1f) * Lod.SUN_CLEAR;
            metalmc.backend.MetalShadows.apply(mainRenderTarget.getColorTexture(), mainRenderTarget.getDepthTexture(), Lod.LEVEL_PROJECTION,
                cameraState.viewRotationMatrix, cameraState.pos.x, cameraState.pos.y, cameraState.pos.z, Lod.SUN_ANGLE, strength, Lod.CLOUD_HEIGHT, TemporalAA.ENABLED);
        }
        // Lit hook (METALMC_EXP=lit, metalmc.backend.MetalLit): the terrain relit from the G-buffer the level's main pass
        // wrote, after the shadows (it takes their visibility) and before the aerial perspective; with anti-aliasing, in
        // its resolve as it loads the color.
        if (metalmc.backend.MetalLit.ENABLED && Lod.SUN_SKY) {
            // Colored block light (METALMC_EXP=coloredlight, metalmc.light.ColoredLight): the light volume's work, which the
            // relight samples.
            if (metalmc.backend.MetalColoredLight.ENABLED) metalmc.light.ColoredLight.frame(cameraState.pos.x, cameraState.pos.y, cameraState.pos.z);
            metalmc.backend.MetalLit.relight(mainRenderTarget.getColorTexture(), mainRenderTarget.getDepthTexture(), Lod.LEVEL_PROJECTION,
                cameraState.viewRotationMatrix, Lod.SUN_ANGLE, cameraState.fogData,
                net.minecraft.client.Minecraft.getInstance().gameRenderer.levelLightmap(), metalmc.sky.Sky.frameActive, TemporalAA.ENABLED);
        }
        // Sky hook (METALMC_EXP=sky, metalmc.sky.Sky): the level through the air (aerial perspective, the render distance's
        // fade into the sky, the tone curve), after the shadows; with anti-aliasing, as it loads the color.
        if (metalmc.sky.Sky.frameActive) {
            // Post hook (METALMC_EXP=post): where vanilla's clouds are, so this step lights them by day.
            if (metalmc.backend.MetalPost.ENABLED) metalmc.backend.MetalPost.clouds(Lod.CLOUD_HEIGHT - (float) cameraState.pos.y);
            metalmc.backend.MetalSky.aerial(mainRenderTarget.getColorTexture(), mainRenderTarget.getDepthTexture(), Lod.LEVEL_PROJECTION,
                cameraState.viewRotationMatrix, TemporalAA.ENABLED);
        }
        if (TemporalAA.ENABLED) {
            metalmc.backend.MetalTaa.apply(mainRenderTarget.getColorTexture(), mainRenderTarget.getDepthTexture(), TemporalAA.UNJITTERED,
                cameraState.viewRotationMatrix, cameraState.pos.x, cameraState.pos.y, cameraState.pos.z, TemporalAA.jitterX, TemporalAA.jitterY, false);
        }
        // Post hook (METALMC_EXP=post, metalmc.backend.MetalPost): bloom, eye adaptation, light shafts and the tone curve
        // over the level's scene-linear light (our sky drew it), after the anti-aliasing, before the hand.
        if (metalmc.backend.MetalPost.ENABLED && metalmc.sky.Sky.frameActive) {
            metalmc.backend.MetalPost.apply(mainRenderTarget.getColorTexture(), mainRenderTarget.getDepthTexture(),
                TemporalAA.ENABLED ? TemporalAA.UNJITTERED : Lod.LEVEL_PROJECTION, cameraState.viewRotationMatrix,
                cameraState.pos.x, cameraState.pos.y, cameraState.pos.z, Lod.SUN_ANGLE, TemporalAA.ENABLED);
        }
    }
}
