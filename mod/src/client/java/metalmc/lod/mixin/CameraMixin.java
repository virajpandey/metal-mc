package metalmc.lod.mixin;

import metalmc.lod.Lod;
import net.minecraft.client.Camera;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyArg;

/**
 * Pushes the projection's far plane past the LOD. Reverse-Z float depth keeps precision at any range. Camera.update sets
 * depthFar and builds the perspective from it in one go, so the far plane is changed where it's passed in: changed after
 * update returned (as before 26.3), the projection kept vanilla's few kilometres, and terrain past them (the far field's,
 * which isn't clipped) wrote depth 0, so the sky's aerial perspective took it for sky and left it without haze.
 */
@Mixin(Camera.class)
abstract class CameraMixin {
    @Shadow
    private float depthFar;

    @ModifyArg(method = "update", at = @At(value = "INVOKE",
        target = "Lnet/minecraft/client/Camera;setupPerspective(FFFFF)V"), index = 1)
    private float metalmc$extendFar(float far) {
        if (!Lod.active()) return far;
        depthFar = Math.max(depthFar, Lod.FAR * 1.5f);
        return depthFar;
    }
}
