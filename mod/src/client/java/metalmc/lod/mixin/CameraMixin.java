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
 * The plane is 6 x the LOD's reach: the far field's outermost ring (FarField.swift: a ring per LOD level from 1 to the
 * top, at most 8, each 1024 cells either side of the camera) runs to about 4 x the reach along the axes and 5.7 x at its
 * corners. At 1.5 x
 * (before 2026-10-10) its far hits wrote depth 0 all the same: a speckled, unhazed strip along the horizon with the LOD
 * at 8 km (the lab's), every hit there at the same depth 0, so they z-fought.
 */
@Mixin(Camera.class)
abstract class CameraMixin {
    /** -PlodExtendFar=0: the perspective keeps vanilla's far plane (only depthFar, which the 3D HUD uses, is pushed). */
    private static final boolean EXTEND = !"0".equals(System.getProperty("metalmc.lod.extendFar", "1"));

    @Shadow
    private float depthFar;

    @ModifyArg(method = "update", at = @At(value = "INVOKE",
        target = "Lnet/minecraft/client/Camera;setupPerspective(FFFFF)V"), index = 1)
    private float metalmc$extendFar(float far) {
        if (!Lod.active()) return far;
        depthFar = Math.max(depthFar, Lod.FAR * 6f);
        return EXTEND ? depthFar : far;
    }
}
