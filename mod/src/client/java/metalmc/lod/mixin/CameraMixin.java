package metalmc.lod.mixin;

import metalmc.lod.Lod;
import net.minecraft.client.Camera;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Pushes the projection's far plane past the LOD. Reverse-Z float depth keeps precision at any range. */
@Mixin(Camera.class)
abstract class CameraMixin {
    @Shadow
    private float depthFar;

    @Inject(method = "update", at = @At("RETURN"))
    private void metalmc$extendFar(CallbackInfo ci) {
        if (Lod.active()) depthFar = Math.max(depthFar, Lod.FAR * 1.5f);
    }
}
