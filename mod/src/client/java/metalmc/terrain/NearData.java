package metalmc.terrain;

import net.minecraft.client.renderer.chunk.ChunkSectionLayer;

/**
 * Near-chunk arena entries (NearChunks) per solid/cutout layer, attached by mixin to SectionCompiler.Results and
 * CompiledSectionMesh. An id of 0 means the layer is vanilla's.
 */
public interface NearData {
    int metalmc$nearEntry(ChunkSectionLayer layer);

    void metalmc$setNearEntry(ChunkSectionLayer layer, int id);

    /** True if the layer's vertices were left out of vanilla's heap (nearslim): vanilla must never draw it. */
    default boolean metalmc$nearSlim(ChunkSectionLayer layer) {
        return false;
    }

    default void metalmc$setNearSlim(ChunkSectionLayer layer) {
    }
}
