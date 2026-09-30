package metalmc.sky.mixin;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.vertex.PoseStack;
import com.mojang.renderpearl.api.buffers.GpuBufferSlice;
import com.mojang.renderpearl.api.commands.RenderPass;
import metalmc.backend.MetalSky;
import metalmc.lod.Lod;
import metalmc.sky.Sky;
import net.minecraft.client.renderer.SkyRenderer;
import net.minecraft.client.renderer.state.level.SkyRenderState;
import org.joml.Vector3fc;
import org.joml.Vector4fc;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Our sky (metalmc.sky.Sky) in vanilla's sky pass: drawn in place of the sky disc, first in the pass, so vanilla's moon
 * and stars still draw over it. Its sun and sunset glow are part of it, so vanilla's sunrise fan and sun aren't drawn.
 * If it didn't draw, vanilla's sky stays as it is.
 */
@Mixin(SkyRenderer.class)
abstract class SkyRendererMixin {
    @Shadow
    @Final
    private RenderTarget renderTarget;

    /** Before the sky pass opens: the sky at a quarter of the resolution for this frame's view, which the pass filters up. */
    @Inject(method = "render", at = @At("HEAD"))
    private void metalmc$skyView(GpuBufferSlice skyFog, SkyRenderState state, CallbackInfo ci) {
        if (Sky.frameActive) MetalSky.prepareView(renderTarget.getColorTexture(), Lod.LEVEL_PROJECTION, Sky.VIEW_ROTATION);
    }

    @Inject(method = "renderSkyDisc", at = @At("HEAD"), cancellable = true)
    private void metalmc$sky(RenderPass renderPass, Vector3fc skyColor, CallbackInfo ci) {
        Sky.drawn = Sky.frameActive && MetalSky.draw(Lod.LEVEL_PROJECTION, Sky.VIEW_ROTATION);
        if (Sky.drawn) ci.cancel();
    }

    @Inject(method = "renderSunriseAndSunset", at = @At("HEAD"), cancellable = true)
    private void metalmc$noSunriseFan(RenderPass renderPass, PoseStack poseStack, float sunAngle, Vector4fc sunriseAndSunsetColor, CallbackInfo ci) {
        if (Sky.drawn) ci.cancel();
    }

    @Inject(method = "renderSun", at = @At("HEAD"), cancellable = true)
    private void metalmc$noSun(RenderPass renderPass, float rainBrightness, PoseStack poseStack, CallbackInfo ci) {
        if (Sky.drawn) ci.cancel();
    }
}
