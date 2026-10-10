package metalmc.render.mixin;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.renderpearl.api.commands.CommandEncoder;
import com.mojang.renderpearl.api.textures.GpuTexture;
import metalmc.render.RenderScale;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.state.GameRenderState;
import org.joml.Vector4fc;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Render scale (metalmc.render.RenderScale): the frame's decision where it clears the main target, and the main target's
 * accessor answering the level's own target while the level is drawn.
 */
@Mixin(GameRenderer.class)
abstract class GameRendererScaleMixin {
    @Shadow
    @Final
    private RenderTarget mainRenderTarget;

    @Shadow
    @Final
    private GameRenderState gameRenderState;

    /**
     * The start of the frame clears the main target. With a scale this frame the level goes into its own target, the
     * anti-aliasing's copy then writes every pixel of the main target's color, and vanilla clears its depth before the hand
     * and again before the GUI, so the clear is left out: with pass folding it would run as a pass of its own as soon as
     * the frame's first other pass begins (the whole target stored for nothing).
     */
    @Redirect(method = "render", at = @At(value = "INVOKE", ordinal = 0,
        target = "Lcom/mojang/renderpearl/api/commands/CommandEncoder;clearColorAndDepthTextures(Lcom/mojang/renderpearl/api/textures/GpuTexture;Lorg/joml/Vector4fc;Lcom/mojang/renderpearl/api/textures/GpuTexture;D)V"))
    private void metalmc$frameStart(CommandEncoder encoder, GpuTexture color, Vector4fc clearColor, GpuTexture depth, double clearDepth) {
        boolean scaled = RenderScale.beginFrame(mainRenderTarget, gameRenderState.shouldRenderLevel, !gameRenderState.requestedPostEffects.isEmpty());
        if (!scaled) encoder.clearColorAndDepthTextures(color, clearColor, depth, clearDepth);
    }

    @Inject(method = "renderLevel", at = @At("HEAD"))
    private void metalmc$levelStart(CallbackInfo ci) {
        RenderScale.beginLevel();
        // Work the last frame left to run beside this frame's main pass (the GI cache's deferred update, gidefer).
        metalmc.backend.MetalScale.frameStart();
    }

    /** While the level is drawn with a scale: the level's target (LevelRenderer reaches the main target only through here). */
    @Inject(method = "mainRenderTarget", at = @At("HEAD"), cancellable = true)
    private void metalmc$levelTarget(CallbackInfoReturnable<RenderTarget> cir) {
        RenderTarget level = RenderScale.levelTarget();
        if (level != null) cir.setReturnValue(level);
    }
}
