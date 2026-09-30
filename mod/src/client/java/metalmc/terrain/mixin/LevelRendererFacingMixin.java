package metalmc.terrain.mixin;

import java.util.List;
import metalmc.terrain.FacingData;
import metalmc.terrain.FacingSorter;
import metalmc.terrain.NearChunks;
import metalmc.terrain.SectionOcclusion;
import net.minecraft.client.renderer.DynamicGpuData;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.renderer.chunk.CompiledSectionMesh;
import net.minecraft.client.renderer.chunk.SectionRenderDispatcher;
import net.minecraft.client.renderer.chunk.SectionMesh;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import net.minecraft.core.BlockPos;
import net.minecraft.world.phys.Vec3;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyArg;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Replaces each section layer's single draw with draws of only the facing buckets that can face the
 * camera, merged where adjacent. Applies to both the indirect and the per-section terrain paths. Also
 * leaves out sections that the GPU occlusion test found hidden (SectionOcclusion).
 */
@Mixin(LevelRenderer.class)
abstract class LevelRendererFacingMixin {
    @Shadow
    @Final
    private LevelRenderState levelRenderState;

    @Shadow
    private net.minecraft.client.renderer.ViewArea viewArea;

    @Shadow
    private boolean usingMultiDrawIndirectForTerrain;

    @Inject(method = "extractSectionDrawGroups", at = @At("HEAD"))
    private void metalmc$beginOcclusion(CallbackInfoReturnable<Integer> cir) {
        Vec3 cam = levelRenderState.cameraRenderState.pos;
        SectionOcclusion.beginFrame(cam.x, cam.y, cam.z);
        if (viewArea != null && metalmc.lod.Lod.active()) SectionOcclusion.recordCompiled(((ViewAreaAccessor) viewArea).metalmc$sections());
        // Near chunks draw from their own per-section stream only on the multi-draw-indirect path.
        if (NearChunks.ENABLED) NearChunks.beginFrame(usingMultiDrawIndirectForTerrain);
    }

    // The section and layer vanilla is working on, recorded by the hooks below as it walks the visible sections
    // (render thread). Redirects and ModifyArg rather than WrapOperation with @Local: those allocated a holder
    // per local and a varargs array per call, per section per layer per frame (about a tenth of all allocation
    // while flying, and G1 pauses are most of the frames that miss 120 Hz).
    private static SectionMesh currentMesh;
    private static int currentX, currentY, currentZ;
    private static ChunkSectionLayer currentLayer;

    /**
     * A section the occlusion test found hidden gets an empty mesh here, so vanilla skips it before it
     * creates draw groups or section data for it.
     */
    @Redirect(method = "extractSectionDrawGroups", at = @At(value = "INVOKE",
        target = "Lnet/minecraft/client/renderer/chunk/SectionRenderDispatcher$RenderSection;getSectionMesh()Lnet/minecraft/client/renderer/chunk/SectionMesh;"))
    private SectionMesh metalmc$skipHidden(SectionRenderDispatcher.RenderSection section) {
        SectionMesh mesh = section.getSectionMesh();
        BlockPos o = section.getRenderOrigin();
        currentX = o.getX();
        currentY = o.getY();
        currentZ = o.getZ();
        if (mesh != CompiledSectionMesh.UNCOMPILED) SectionOcclusion.recordVanilla(currentX, currentY, currentZ);
        if (mesh.hasRenderableLayers()) {
            Vec3 c = levelRenderState.cameraRenderState.pos;
            if (SectionOcclusion.skip(currentX, currentY, currentZ, c.x, c.y, c.z)) mesh = CompiledSectionMesh.EMPTY;
        }
        currentMesh = mesh;
        return mesh;
    }

    @ModifyArg(method = "extractSectionDrawGroups", at = @At(value = "INVOKE",
        target = "Lnet/minecraft/client/renderer/chunk/SectionMesh;getSectionDraw(Lnet/minecraft/client/renderer/chunk/ChunkSectionLayer;)Lnet/minecraft/client/renderer/chunk/SectionMesh$SectionDraw;"))
    private ChunkSectionLayer metalmc$recordLayer(ChunkSectionLayer layer) {
        currentLayer = layer;
        return layer;
    }

    /**
     * The section layer's draw (the third List.add in the method): split into the facing buckets that can face the camera.
     * Near chunks (METALMC_EXP=nearchunks): a layer repacked into the near-chunk arena goes to NearChunks' list instead,
     * split the same way.
     */
    @Redirect(method = "extractSectionDrawGroups", at = @At(value = "INVOKE", target = "Ljava/util/List;add(Ljava/lang/Object;)Z", ordinal = 2))
    private boolean metalmc$splitByFacing(List<Object> list, Object element) {
        int near = 0;
        if (NearChunks.ENABLED && element instanceof DynamicGpuData.IndexedDraw) {
            near = NearChunks.divertedEntry(currentMesh, currentLayer);
            // Vanilla's heap holds a placeholder for a slimmed layer: draw nothing rather than garbage.
            if (near == 0 && NearChunks.slimmed(currentMesh, currentLayer)) return true;
        }
        int[] counts = FacingSorter.ENABLED && currentMesh instanceof FacingData data ? data.metalmc$facings(currentLayer) : null;
        if (!(element instanceof DynamicGpuData.IndexedDraw draw) || (counts == null && near == 0)) {
            return list.add(element);
        }
        if (counts == null) {
            NearChunks.add(currentLayer, near, draw.firstIndex() / 6, draw.indexCount() / 6, draw.baseInstance());
            return true;
        }
        Vec3 cam = levelRenderState.cameraRenderState.pos;
        int mask = FacingSorter.visibleMask(cam.x, cam.y, cam.z, currentX, currentY, currentZ);
        int quad = 0;
        int runStart = -1;
        for (int b = 0; b < FacingSorter.BUCKETS; b++) {
            boolean visible = (mask & (1 << b)) != 0 || counts[b] == 0;
            if (visible) {
                if (runStart < 0) runStart = quad;
            } else if (runStart >= 0) {
                emit(list, draw, runStart, quad, near);
                runStart = -1;
            }
            quad += counts[b];
        }
        if (runStart >= 0) emit(list, draw, runStart, quad, near);
        return true;
    }

    private static void emit(List<Object> list, DynamicGpuData.IndexedDraw draw, int fromQuad, int toQuad, int near) {
        if (toQuad <= fromQuad) return;
        if (near != 0) {
            NearChunks.add(currentLayer, near, draw.firstIndex() / 6 + fromQuad, toQuad - fromQuad, draw.baseInstance());
            return;
        }
        if (fromQuad == 0 && (toQuad - fromQuad) * 6 == draw.indexCount()) {
            list.add(draw);   // every bucket visible: keep vanilla's draw
            return;
        }
        list.add(new DynamicGpuData.IndexedDraw((toQuad - fromQuad) * 6, draw.instanceCount(), draw.firstIndex() + fromQuad * 6,
            draw.baseVertex(), draw.baseInstance()));
    }
}
