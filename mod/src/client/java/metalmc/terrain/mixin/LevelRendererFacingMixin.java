package metalmc.terrain.mixin;

import com.llamalad7.mixinextras.injector.wrapoperation.Operation;
import com.llamalad7.mixinextras.injector.wrapoperation.WrapOperation;
import com.llamalad7.mixinextras.sugar.Local;
import java.util.List;
import metalmc.terrain.FacingData;
import metalmc.terrain.FacingSorter;
import net.minecraft.client.renderer.DynamicGpuData;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.renderer.chunk.SectionMesh;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import net.minecraft.core.BlockPos;
import net.minecraft.world.phys.Vec3;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;

/**
 * Replaces each section layer's single draw with draws of only the facing buckets that can face the
 * camera, merged where adjacent. Applies to both the indirect and the per-section terrain paths.
 */
@Mixin(LevelRenderer.class)
abstract class LevelRendererFacingMixin {
    @Shadow
    @Final
    private LevelRenderState levelRenderState;

    @WrapOperation(method = "extractSectionDrawGroups", at = @At(value = "INVOKE", target = "Ljava/util/List;add(Ljava/lang/Object;)Z"))
    private boolean metalmc$splitByFacing(List<Object> list, Object element, Operation<Boolean> original,
                                          @Local SectionMesh sectionMesh, @Local ChunkSectionLayer layer, @Local BlockPos renderOffset) {
        if (!FacingSorter.ENABLED || !(element instanceof DynamicGpuData.IndexedDraw draw) || !(sectionMesh instanceof FacingData data)) {
            return original.call(list, element);
        }
        int[] counts = data.metalmc$facings(layer);
        if (counts == null) return original.call(list, element);
        Vec3 cam = levelRenderState.cameraRenderState.pos;
        int mask = FacingSorter.visibleMask(cam.x, cam.y, cam.z, renderOffset.getX(), renderOffset.getY(), renderOffset.getZ());
        int quad = 0;
        int runStart = -1;
        for (int b = 0; b < FacingSorter.BUCKETS; b++) {
            boolean visible = (mask & (1 << b)) != 0 || counts[b] == 0;
            if (visible) {
                if (runStart < 0) runStart = quad;
            } else if (runStart >= 0) {
                emit(list, original, draw, runStart, quad);
                runStart = -1;
            }
            quad += counts[b];
        }
        if (runStart >= 0) emit(list, original, draw, runStart, quad);
        return true;
    }

    private static void emit(List<Object> list, Operation<Boolean> original, DynamicGpuData.IndexedDraw draw, int fromQuad, int toQuad) {
        if (toQuad <= fromQuad) return;
        original.call(list, new DynamicGpuData.IndexedDraw((toQuad - fromQuad) * 6, draw.instanceCount(), draw.firstIndex() + fromQuad * 6,
            draw.baseVertex(), draw.baseInstance()));
    }
}
