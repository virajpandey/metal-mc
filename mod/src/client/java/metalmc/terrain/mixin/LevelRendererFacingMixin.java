package metalmc.terrain.mixin;

import java.util.List;
import metalmc.terrain.FacingData;
import metalmc.terrain.FacingSorter;
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

    @Inject(method = "extractSectionDrawGroups", at = @At("HEAD"))
    private void metalmc$beginOcclusion(CallbackInfoReturnable<Integer> cir) {
        Vec3 cam = levelRenderState.cameraRenderState.pos;
        SectionOcclusion.beginFrame(cam.x, cam.y, cam.z);
        if (viewArea != null && metalmc.lod.Lod.active()) SectionOcclusion.recordCompiled(((ViewAreaAccessor) viewArea).metalmc$sections());
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

    /** The section layer's draw (the third List.add in the method): split into the facing buckets that can face the camera. */
    @Redirect(method = "extractSectionDrawGroups", at = @At(value = "INVOKE", target = "Ljava/util/List;add(Ljava/lang/Object;)Z", ordinal = 2))
    private boolean metalmc$splitByFacing(List<Object> list, Object element) {
        if (!FacingSorter.ENABLED || !(element instanceof DynamicGpuData.IndexedDraw draw) || !(currentMesh instanceof FacingData data)) {
            return list.add(element);
        }
        int[] counts = data.metalmc$facings(currentLayer);
        if (counts == null) return list.add(element);
        Vec3 cam = levelRenderState.cameraRenderState.pos;
        int mask = FacingSorter.visibleMask(cam.x, cam.y, cam.z, currentX, currentY, currentZ);
        int quad = 0;
        int runStart = -1;
        for (int b = 0; b < FacingSorter.BUCKETS; b++) {
            boolean visible = (mask & (1 << b)) != 0 || counts[b] == 0;
            if (visible) {
                if (runStart < 0) runStart = quad;
            } else if (runStart >= 0) {
                emit(list, draw, runStart, quad);
                runStart = -1;
            }
            quad += counts[b];
        }
        if (runStart >= 0) emit(list, draw, runStart, quad);
        return true;
    }

    private static void emit(List<Object> list, DynamicGpuData.IndexedDraw draw, int fromQuad, int toQuad) {
        if (toQuad <= fromQuad) return;
        if (fromQuad == 0 && (toQuad - fromQuad) * 6 == draw.indexCount()) {
            list.add(draw);   // every bucket visible: keep vanilla's draw
            return;
        }
        list.add(new DynamicGpuData.IndexedDraw((toQuad - fromQuad) * 6, draw.instanceCount(), draw.firstIndex() + fromQuad * 6,
            draw.baseVertex(), draw.baseInstance()));
    }
}
