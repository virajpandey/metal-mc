package metalmc.sky.mixin;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.renderpearl.api.textures.GpuTexture;
import metalmc.backend.MetalSky;
import net.minecraft.client.Screenshot;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

/**
 * HDR (METALMC_EXP=hdr): screenshots read the main target as 8-bit RGBA, so with the float target they read an 8-bit
 * copy of it, rolled into SDR (MetalSky.screenshotSource). Covers F2, the world icon and the benchmark's screenshots.
 */
@Mixin(Screenshot.class)
abstract class ScreenshotMixin {
    @Redirect(method = "takeScreenshot(Lcom/mojang/blaze3d/pipeline/RenderTarget;ILjava/util/function/Consumer;)V",
        at = @At(value = "INVOKE", target = "Lcom/mojang/blaze3d/pipeline/RenderTarget;getColorTexture()Lcom/mojang/renderpearl/api/textures/GpuTexture;"))
    private static GpuTexture metalmc$sdrCopy(RenderTarget target) {
        return MetalSky.screenshotSource(target.getColorTexture());
    }
}
