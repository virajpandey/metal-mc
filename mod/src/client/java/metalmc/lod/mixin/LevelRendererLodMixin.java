package metalmc.lod.mixin;

import com.mojang.renderpearl.api.commands.RenderPass;
import metalmc.backend.MetalLod;
import metalmc.lod.Lod;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.chunk.ChunkSectionsToRender;
import net.minecraft.client.renderer.feature.FeatureRenderDispatcher;
import net.minecraft.client.renderer.fog.FogData;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Draws the LOD in the main world pass right after vanilla's solid terrain. */
@Mixin(LevelRenderer.class)
abstract class LevelRendererLodMixin {
    @Shadow
    @Final
    private LevelRenderState levelRenderState;

    @Inject(method = "executeSolid", at = @At(value = "INVOKE",
        target = "Lnet/minecraft/client/renderer/chunk/ChunkSectionsToRender;renderGroup(Lnet/minecraft/client/renderer/chunk/ChunkSectionLayerGroup;Lcom/mojang/renderpearl/api/commands/RenderPass;Lcom/mojang/renderpearl/api/textures/GpuSampler;Lcom/mojang/renderpearl/api/textures/GpuTextureView;Z)V",
        shift = At.Shift.AFTER))
    private void metalmc$drawLod(ChunkSectionsToRender chunks, FeatureRenderDispatcher.PreparedFrame featureFrame, RenderPass renderPass, CallbackInfo ci) {
        if (!Lod.active()) return;
        CameraRenderState cam = levelRenderState.cameraRenderState;
        FogData fog = cam.fogData;
        Minecraft mc = Minecraft.getInstance();
        int renderDistance = mc.options.getEffectiveRenderDistance();
        float discard = Math.max(0, (renderDistance - 1) * 16f);
        float sky = mc.level == null ? 1f : 1f - mc.level.getSkyDarken() / 15f;
        MetalLod.draw(cam.projectionMatrix, cam.viewRotationMatrix, cam.pos.x, cam.pos.y, cam.pos.z,
            fog.color.x(), fog.color.y(), fog.color.z(), fog.color.w(), fog.environmentalStart, fog.environmentalEnd,
            fog.renderDistanceStart, fog.renderDistanceEnd, discard, sky);
    }
}
