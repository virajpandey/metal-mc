package metalmc.sky.mixin;

import com.mojang.blaze3d.pipeline.MainTarget;
import com.mojang.renderpearl.api.GpuFormat;
import metalmc.backend.MetalSky;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyArg;

/**
 * HDR (METALMC_EXP=hdr, Hdr.swift): the main target's color is RGBA16Float, so the level can hold light brighter than
 * SDR white (the sun, the sky around it). Its values stay sRGB-encoded like vanilla's; the backend gives vanilla's
 * pipelines, which declare RGBA8 targets, variants for it. RenderTargetMixin covers the window being resized.
 */
@Mixin(MainTarget.class)
abstract class MainTargetMixin {
    @ModifyArg(method = "allocateColorAttachment", at = @At(value = "INVOKE",
        target = "Lcom/mojang/renderpearl/api/device/GpuDevice;createTexture(Ljava/util/function/Supplier;ILcom/mojang/renderpearl/api/GpuFormat;IIII)Lcom/mojang/renderpearl/api/textures/GpuTexture;"),
        index = 2)
    private GpuFormat metalmc$hdrFormat(GpuFormat format) {
        return MetalSky.hdr() ? GpuFormat.RGBA16_FLOAT : format;
    }
}
