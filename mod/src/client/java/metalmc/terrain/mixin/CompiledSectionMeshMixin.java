package metalmc.terrain.mixin;

import java.util.EnumMap;
import metalmc.terrain.FacingData;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.renderer.chunk.CompiledSectionMesh;
import net.minecraft.client.renderer.chunk.SectionCompiler;
import net.minecraft.client.renderer.chunk.TranslucencyPointOfView;
import org.jspecify.annotations.Nullable;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Unique;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Carries the facing bucket counts from the compile results onto the section mesh used for drawing. */
@Mixin(CompiledSectionMesh.class)
abstract class CompiledSectionMeshMixin implements FacingData {
    @Unique
    private final EnumMap<ChunkSectionLayer, int[]> metalmc$facings = new EnumMap<>(ChunkSectionLayer.class);

    @Inject(method = "<init>", at = @At("RETURN"))
    private void metalmc$copyFacings(TranslucencyPointOfView pointOfView, SectionCompiler.Results results, long startTimeNs, CallbackInfo ci) {
        for (ChunkSectionLayer layer : ChunkSectionLayer.values()) {
            int[] counts = ((FacingData) (Object) results).metalmc$facings(layer);
            if (counts != null) metalmc$facings.put(layer, counts);
        }
    }

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
