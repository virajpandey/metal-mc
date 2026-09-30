package metalmc.terrain.mixin;

import metalmc.terrain.NearData;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.renderer.chunk.SectionCompiler;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;

/** Near-chunk entries for a compile's solid and cutout layers, until CompiledSectionMeshNearMixin takes them over. */
@Mixin(SectionCompiler.Results.class)
abstract class SectionCompilerResultsNearMixin implements NearData {
    @Unique
    private int metalmc$nearSolid;
    @Unique
    private int metalmc$nearCutout;

    @Override
    public int metalmc$nearEntry(ChunkSectionLayer layer) {
        return layer == ChunkSectionLayer.SOLID ? metalmc$nearSolid : layer == ChunkSectionLayer.CUTOUT ? metalmc$nearCutout : 0;
    }

    @Override
    public void metalmc$setNearEntry(ChunkSectionLayer layer, int id) {
        if (layer == ChunkSectionLayer.SOLID) metalmc$nearSolid = id;
        else if (layer == ChunkSectionLayer.CUTOUT) metalmc$nearCutout = id;
    }
}
