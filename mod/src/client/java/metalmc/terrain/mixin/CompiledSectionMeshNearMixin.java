package metalmc.terrain.mixin;

import metalmc.terrain.NearChunks;
import metalmc.terrain.NearData;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.renderer.chunk.CompiledSectionMesh;
import net.minecraft.client.renderer.chunk.SectionCompiler;
import net.minecraft.client.renderer.chunk.TranslucencyPointOfView;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * The near-chunk entries of a section mesh (NearChunks): taken over from the compile results, and freed when vanilla
 * closes the mesh (replaced by a newer compile, the section reset, or the compile cancelled), the same moment vanilla
 * frees the mesh's own heap allocations. Vanilla closes meshes under its copy lock, which its draw extraction also holds.
 */
@Mixin(CompiledSectionMesh.class)
abstract class CompiledSectionMeshNearMixin implements NearData {
    @Unique
    private int metalmc$nearSolid;
    @Unique
    private int metalmc$nearCutout;
    @Unique
    private boolean metalmc$slimSolid;
    @Unique
    private boolean metalmc$slimCutout;

    @Inject(method = "<init>", at = @At("RETURN"))
    private void metalmc$takeNear(TranslucencyPointOfView pointOfView, SectionCompiler.Results results, long startTimeNs, CallbackInfo ci) {
        NearData from = (NearData) (Object) results;
        metalmc$nearSolid = from.metalmc$nearEntry(ChunkSectionLayer.SOLID);
        metalmc$nearCutout = from.metalmc$nearEntry(ChunkSectionLayer.CUTOUT);
    }

    @Inject(method = "close", at = @At("HEAD"))
    private void metalmc$releaseNear(CallbackInfo ci) {
        NearChunks.release(this);
    }

    @Override
    public int metalmc$nearEntry(ChunkSectionLayer layer) {
        return layer == ChunkSectionLayer.SOLID ? metalmc$nearSolid : layer == ChunkSectionLayer.CUTOUT ? metalmc$nearCutout : 0;
    }

    @Override
    public void metalmc$setNearEntry(ChunkSectionLayer layer, int id) {
        if (layer == ChunkSectionLayer.SOLID) metalmc$nearSolid = id;
        else if (layer == ChunkSectionLayer.CUTOUT) metalmc$nearCutout = id;
    }

    @Override
    public boolean metalmc$nearSlim(ChunkSectionLayer layer) {
        return layer == ChunkSectionLayer.SOLID ? metalmc$slimSolid : layer == ChunkSectionLayer.CUTOUT && metalmc$slimCutout;
    }

    @Override
    public void metalmc$setNearSlim(ChunkSectionLayer layer) {
        if (layer == ChunkSectionLayer.SOLID) metalmc$slimSolid = true;
        else if (layer == ChunkSectionLayer.CUTOUT) metalmc$slimCutout = true;
    }
}
