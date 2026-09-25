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
 * chunks are no longer fogged where the LOD continues them, and clear-weather haze stretches to the LOD edge.
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
        // Clear-weather atmospheric haze runs 0 -> 1024 blocks (FOG_END_DISTANCE default), which would hide
        // everything past 1 km. Stretch it to the LOD distance. Shorter special fogs (rain, water, lava,
        // blindness, the Nether's 96) stay as they are.
        if (fog.environmentalEnd >= 1000f && fog.environmentalEnd < Lod.FAR) {
            float scale = Lod.FAR / fog.environmentalEnd;
            fog.environmentalStart *= scale;
            fog.environmentalEnd = Lod.FAR;
        }
    }
}
