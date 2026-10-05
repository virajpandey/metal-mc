package metalmc.extpipe.mixin;

import metalmc.backend.MetalExtPipe;
import net.minecraft.client.renderer.SkyRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** The external pipeline (METALMC_EXTPIPE): vanilla's sky pass draws into its G-buffer too, through its sky programs. */
@Mixin(SkyRenderer.class)
abstract class SkyRendererExtMixin {
    @Inject(method = "render", at = @At(value = "INVOKE",
        target = "Lcom/mojang/renderpearl/api/commands/CommandEncoder;createRenderPass(Ljava/util/function/Supplier;Lcom/mojang/renderpearl/api/textures/GpuTextureView;Ljava/util/Optional;Lcom/mojang/renderpearl/api/textures/GpuTextureView;Ljava/util/OptionalDouble;)Lcom/mojang/renderpearl/api/commands/RenderPass;"))
    private void metalmc$extSkyPass(CallbackInfo ci) {
        if (MetalExtPipe.ENABLED) MetalExtPipe.redirect(1);
    }
}
