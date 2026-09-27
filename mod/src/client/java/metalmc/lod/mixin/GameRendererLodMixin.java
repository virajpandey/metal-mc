package metalmc.lod.mixin;

import metalmc.lod.Lod;
import net.minecraft.client.renderer.GameRenderer;
import org.joml.Matrix4f;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyArg;

/**
 * The projection vanilla draws the level with: the camera's, times view bobbing, the hurt tilt and the nausea and
 * portal distortion (GameRenderer.renderLevel works on a copy). The LOD and the section occlusion test must use the
 * same one, or while the player walks the LOD sways against vanilla's terrain (bobbing rolls the view up to 3
 * degrees and pitches it up to 5) and the occlusion boxes miss the depth buffer.
 */
@Mixin(GameRenderer.class)
abstract class GameRendererLodMixin {
    @ModifyArg(method = "renderLevel", at = @At(value = "INVOKE",
        target = "Lnet/minecraft/client/renderer/ProjectionMatrixBuffer;getBuffer(Lorg/joml/Matrix4f;)Lcom/mojang/renderpearl/api/buffers/GpuBufferSlice;"))
    private Matrix4f metalmc$levelProjection(Matrix4f projection) {
        Lod.LEVEL_PROJECTION.set(projection);
        return projection;
    }
}
