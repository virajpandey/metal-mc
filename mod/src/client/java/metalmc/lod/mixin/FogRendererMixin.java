package metalmc.lod.mixin;

import metalmc.lod.Lod;
import net.minecraft.client.renderer.fog.FogData;
import net.minecraft.client.renderer.fog.FogRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * With LOD active, the render-distance fog moves from vanilla's chunk edge to the LOD's edge, so vanilla
 * chunks are no longer fogged where the LOD continues them. Weather and biome fog are left alone.
 */
@Mixin(FogRenderer.class)
abstract class FogRendererMixin {
    @Inject(method = "setupFog", at = @At("RETURN"))
    private void metalmc$extendFog(CallbackInfoReturnable<FogData> cir) {
        if (!Lod.active()) return;
        FogData fog = cir.getReturnValue();
        float oldEnd = fog.renderDistanceEnd;
        float span = Math.max(64f, Lod.FAR / 8f);
        fog.renderDistanceStart = Lod.FAR - span;
        fog.renderDistanceEnd = Lod.FAR;
        if (fog.skyEnd >= oldEnd - 1) fog.skyEnd = Lod.FAR;
    }
}
