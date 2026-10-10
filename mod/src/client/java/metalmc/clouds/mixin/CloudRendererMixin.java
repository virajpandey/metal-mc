package metalmc.clouds.mixin;

import metalmc.backend.MetalClouds;
import net.minecraft.client.renderer.CloudRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Volumetric clouds (METALMC_EXP=clouds, metalmc.backend.MetalClouds): while ours are drawn, vanilla's aren't, in either
 * of its paths (the main pass, and the order-independent transparency one). Whether ours are on is decided at the start of
 * the level's frame (our sky draws it), before vanilla's clouds would draw.
 */
@Mixin(CloudRenderer.class)
abstract class CloudRendererMixin {
    @Inject(method = "render(Lnet/minecraft/client/CloudStatus;Lcom/mojang/renderpearl/api/commands/RenderPass;)V", at = @At("HEAD"), cancellable = true)
    private void metalmc$noVanillaClouds(CallbackInfo ci) {
        if (MetalClouds.active()) ci.cancel();
    }

    @Inject(method = "renderOit", at = @At("HEAD"), cancellable = true)
    private void metalmc$noVanillaCloudsOit(CallbackInfo ci) {
        if (MetalClouds.active()) ci.cancel();
    }
}
