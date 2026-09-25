package metalmc.terrain;

import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import org.jspecify.annotations.Nullable;

/** Per-layer facing bucket counts, attached by mixin to SectionCompiler.Results and CompiledSectionMesh. */
public interface FacingData {
    int @Nullable [] metalmc$facings(ChunkSectionLayer layer);

    void metalmc$setFacings(ChunkSectionLayer layer, int @Nullable [] counts);
}
