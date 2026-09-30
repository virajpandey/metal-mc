package metalmc.sky.mixin;

import com.mojang.blaze3d.pipeline.MainTarget;
import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.renderpearl.api.GpuFormat;
import metalmc.backend.MetalSky;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyArg;

/**
 * HDR (METALMC_EXP=hdr): when the window is resized, the main target is rebuilt through RenderTarget.createBuffers
 * (its second texture is the color one); it stays RGBA16Float, like MainTargetMixin makes it at first. Other targets
 * are untouched.
 */
@Mixin(RenderTarget.class)
abstract class RenderTargetMixin {
    @ModifyArg(method = "createBuffers", at = @At(value = "INVOKE",
        target = "Lcom/mojang/renderpearl/api/device/GpuDevice;createTexture(Ljava/util/function/Supplier;ILcom/mojang/renderpearl/api/GpuFormat;IIII)Lcom/mojang/renderpearl/api/textures/GpuTexture;",
        ordinal = 1), index = 2)
    private GpuFormat metalmc$hdrFormat(GpuFormat format) {
        return (Object) this instanceof MainTarget && MetalSky.hdr() ? GpuFormat.RGBA16_FLOAT : format;
    }
}
