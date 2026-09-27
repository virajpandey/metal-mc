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

    @Inject(method = "render3dHud", at = @At("HEAD"))
    private void metalmc$temporalAA(CameraRenderState cameraState, PlayerRenderState playerState, OptionsRenderState optionsState,
                                    boolean consistentDepthRequired, CallbackInfo ci) {
        if (!TemporalAA.ENABLED || mainRenderTarget.getColorTexture() == null || mainRenderTarget.getDepthTexture() == null) return;
        metalmc.backend.MetalTaa.apply(mainRenderTarget.getColorTexture(), mainRenderTarget.getDepthTexture(), TemporalAA.UNJITTERED,
            cameraState.viewRotationMatrix, cameraState.pos.x, cameraState.pos.y, cameraState.pos.z, TemporalAA.jitterX, TemporalAA.jitterY, false);
    }
}
