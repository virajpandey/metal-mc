package metalmc.terrain.mixin;

import com.llamalad7.mixinextras.sugar.Local;
import java.nio.ByteBuffer;
import metalmc.terrain.NearChunks;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.renderer.chunk.CompiledSectionMesh;
import net.minecraft.client.renderer.chunk.SectionRenderDispatcher;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.ModifyArg;

/**
 * nearslim (METALMC_EXP=nearchunks,nearslim): a layer already repacked into the near-chunk arena uploads a 28-byte
 * placeholder into vanilla's vertex heap instead of its vertices. Vanilla's bookkeeping (upload callbacks, the mesh swap)
 * runs as before. Runs once per section layer upload, on the mesh worker (plain @Local captures don't allocate).
 */
@Mixin(SectionRenderDispatcher.RenderSection.class)
abstract class RenderSectionNearMixin {
    @ModifyArg(method = "addSectionBuffersToUberBuffer", at = @At(value = "INVOKE",
        target = "Lcom/mojang/blaze3d/vertex/UberGpuBuffer;addAllocation(Ljava/lang/Object;Lcom/mojang/blaze3d/vertex/UberGpuBuffer$UploadCallback;Ljava/nio/ByteBuffer;)Z",
        ordinal = 0), index = 2)
    private ByteBuffer metalmc$slimUpload(ByteBuffer vertices, @Local(argsOnly = true) ChunkSectionLayer layer,
                                          @Local(argsOnly = true) CompiledSectionMesh mesh) {
        return NearChunks.uploadVertices(layer, mesh, vertices);
    }
}
