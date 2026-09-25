package metalmc.terrain.mixin;

import java.util.EnumMap;
import metalmc.terrain.FacingData;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.renderer.chunk.SectionCompiler;
import org.jspecify.annotations.Nullable;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;

@Mixin(SectionCompiler.Results.class)
abstract class SectionCompilerResultsMixin implements FacingData {
    @Unique
    private final EnumMap<ChunkSectionLayer, int[]> metalmc$facings = new EnumMap<>(ChunkSectionLayer.class);

    @Override
    public int @Nullable [] metalmc$facings(ChunkSectionLayer layer) {
        return metalmc$facings.get(layer);
    }

    @Override
    public void metalmc$setFacings(ChunkSectionLayer layer, int @Nullable [] counts) {
        if (counts == null) metalmc$facings.remove(layer);
        else metalmc$facings.put(layer, counts);
    }
}
