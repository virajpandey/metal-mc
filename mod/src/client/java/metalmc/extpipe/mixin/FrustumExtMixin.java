package metalmc.extpipe.mixin;

import metalmc.backend.MetalExtPipe;
import net.minecraft.client.renderer.culling.Frustum;
import net.minecraft.world.level.levelgen.structure.BoundingBox;
import net.minecraft.world.phys.AABB;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * External pipeline with METALMC_EXTPIPE_NOCULL=1 (-PextPipeNoCull=1): no view-frustum culling, so vanilla draws every
 * section in range (behind the camera too) and the shadow pass, which replays the main pass's terrain draws, has the
 * terrain all around: shadows from behind the camera, and a pack's voxel volume around it. Costs the extra vertex work.
 */
@Mixin(Frustum.class)
abstract class FrustumExtMixin {
    @Inject(method = "cubeInFrustum(Lnet/minecraft/world/level/levelgen/structure/BoundingBox;)I", at = @At("HEAD"), cancellable = true)
    private void metalmc$extNoCullBox(BoundingBox bb, CallbackInfoReturnable<Integer> cir) {
        if (MetalExtPipe.NO_CULL) cir.setReturnValue(-2);
    }

    @Inject(method = "isVisible", at = @At("HEAD"), cancellable = true)
    private void metalmc$extNoCull(AABB bb, CallbackInfoReturnable<Boolean> cir) {
        if (MetalExtPipe.NO_CULL) cir.setReturnValue(true);
    }
}
