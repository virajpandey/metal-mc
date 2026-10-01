package metalmc.lod.mixin;

import metalmc.backend.MetalLit;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import net.minecraft.world.level.dimension.DimensionType;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Lit mode (METALMC_EXP=lit, metalmc.backend.MetalLit): right before the level's main pass opens (the "Solid" or "Main"
 * pass of LevelRenderer.addMainPass, where vanilla's terrain, the near chunks, the LOD and the far field are drawn), the
 * native side is told to give it the terrain G-buffer as a second color target. Overworld only: the relight needs the
 * sun. Not required: if 26.3's lambda moves, lit mode only logs that the level pass had no G-buffer.
 */
@Mixin(LevelRenderer.class)
abstract class LevelRendererLitMixin {
    @Shadow
    @Final
    private LevelRenderState levelRenderState;

    @Inject(method = "lambda$addMainPass$0", require = 0, at = @At(value = "INVOKE",
        target = "Lcom/mojang/renderpearl/api/commands/CommandEncoder;createRenderPass(Ljava/util/function/Supplier;Lcom/mojang/renderpearl/api/textures/GpuTextureView;Ljava/util/Optional;Lcom/mojang/renderpearl/api/textures/GpuTextureView;Ljava/util/OptionalDouble;)Lcom/mojang/renderpearl/api/commands/RenderPass;"))
    private void metalmc$litGbuffer(CallbackInfo ci) {
        if (MetalLit.ENABLED && levelRenderState.skyRenderState.skybox == DimensionType.Skybox.OVERWORLD) MetalLit.markLevelPass();
    }
}
