package metalmc.terrain.mixin;

import com.mojang.blaze3d.vertex.MeshData;
import java.util.Map;
import metalmc.terrain.FacingData;
import metalmc.terrain.FacingSorter;
import metalmc.terrain.NearChunks;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.renderer.chunk.SectionCompiler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Sorts each opaque/cutout section layer's quads into facing buckets on the mesh worker thread, then (near chunks) repacks
 * those layers into the near-chunk arena, in their final order.
 */
@Mixin(SectionCompiler.class)
abstract class SectionCompilerMixin {
    @Inject(method = "compile", at = @At("RETURN"))
    private void metalmc$sortFacings(CallbackInfoReturnable<SectionCompiler.Results> cir) {
        SectionCompiler.Results results = cir.getReturnValue();
        if (FacingSorter.ENABLED) {
            for (Map.Entry<ChunkSectionLayer, MeshData> e : results.renderedLayers.entrySet()) {
                if (e.getKey().translucent()) continue; // translucent quads keep their depth order
                ((FacingData) (Object) results).metalmc$setFacings(e.getKey(), FacingSorter.sort(e.getValue()));
            }
        }
        // Near chunks (METALMC_EXP=nearchunks): after the facing sort, so the facing buckets' quad ranges hold in the arena.
        if (NearChunks.ENABLED) NearChunks.afterCompile(results);
    }
}
