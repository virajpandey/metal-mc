package metalmc.terrain.mixin;

import com.mojang.renderpearl.api.buffers.GpuBuffer;
import com.mojang.renderpearl.api.buffers.GpuBufferSlice;
import com.mojang.renderpearl.api.commands.RenderPass;
import com.mojang.renderpearl.api.pipeline.IndexType;
import com.mojang.renderpearl.api.pipeline.RenderPipeline;
import metalmc.terrain.NearChunks;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.renderer.chunk.ChunkSectionsToRender;
import org.jspecify.annotations.Nullable;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Right after vanilla's multi-draw-indirect draws of a terrain layer: the near-chunk draws of the same layer (NearChunks),
 * in the same render pass with the uniforms vanilla just bound, and vanilla's per-section stream for their positions and
 * fade. The wireframe debug view draws them as lines too.
 */
@Mixin(ChunkSectionsToRender.DrawIndirect.class)
abstract class DrawIndirectNearMixin {
    @Shadow
    @Final
    private GpuBufferSlice chunkSectionInfos;

    @Inject(method = "render", at = @At("TAIL"))
    private void metalmc$drawNear(ChunkSectionLayer layer, RenderPass renderPass, @Nullable GpuBuffer defaultIndexBuffer, @Nullable IndexType defaultIndexType,
                                  @Nullable RenderPipeline renderPipelineOverride, @Nullable RenderPipeline renderPipelineOverrideMultidraw, CallbackInfo ci) {
        if (NearChunks.ENABLED) NearChunks.draw(layer, chunkSectionInfos, renderPipelineOverrideMultidraw != null);
    }
}
