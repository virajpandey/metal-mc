package metalmc.sky.mixin;

import com.llamalad7.mixinextras.injector.ModifyExpressionValue;
import com.llamalad7.mixinextras.sugar.Local;
import com.mojang.renderpearl.api.GpuFormat;
import com.mojang.renderpearl.api.commands.RenderPassDescriptor;
import com.mojang.renderpearl.frontend.FrontendRenderPass;
import metalmc.backend.MetalSky;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;

/**
 * HDR (METALMC_EXP=hdr): a render pass refuses a pipeline whose declared color format isn't its attachment's, and
 * vanilla's pipelines declare RGBA8. The main target is float then (MainTargetMixin: RGBA16Float, or RG11B10Float with
 * METALMC_HDRFORMAT=rg11b10) and the backend gives each pipeline a variant for it (PipelineBox in Backend.swift), so an
 * RGBA8 pipeline is accepted on the float attachment.
 */
@Mixin(FrontendRenderPass.class)
abstract class FrontendRenderPassMixin {
    @ModifyExpressionValue(method = "setPipeline", at = @At(value = "INVOKE",
        target = "Lcom/mojang/renderpearl/api/pipeline/ColorTargetState;format()Lcom/mojang/renderpearl/api/GpuFormat;"))
    private GpuFormat metalmc$hdrTarget(GpuFormat declared, @Local RenderPassDescriptor.Attachment<?> attachment) {
        if (declared == GpuFormat.RGBA8_UNORM && MetalSky.hdr() && attachment.textureView().texture().getFormat() == MetalSky.hdrFormat()) {
            return MetalSky.hdrFormat();
        }
        return declared;
    }
}
