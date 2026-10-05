package metalmc.extpipe.mixin;

import com.mojang.blaze3d.pipeline.RenderTarget;
import com.mojang.blaze3d.resource.GraphicsResourceAllocator;
import com.mojang.renderpearl.api.buffers.GpuBufferSlice;
import metalmc.backend.MetalExtPipe;
import metalmc.extpipe.ExtHand;
import metalmc.lod.Lod;
import net.minecraft.client.renderer.GameRenderer;
import net.minecraft.client.renderer.LevelRenderer;
import net.minecraft.client.renderer.SubmitNodeStorage;
import net.minecraft.client.renderer.state.level.CameraRenderState;
import net.minecraft.client.renderer.state.level.LevelRenderState;
import net.minecraft.client.renderer.texture.TextureAtlas;
import net.minecraft.client.renderer.texture.TextureManager;
import org.joml.Vector4f;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * The external pipeline (METALMC_EXTPIPE, metalmc.backend.MetalExtPipe) in the level: its frame starts with the level
 * (the standard uniforms), the main pass ("Main" in LevelRenderer.addMainPass) draws into its G-buffer, its deferred passes
 * run between that pass's opaque and translucent geometry (classic transparency; with improved transparency they run at
 * the end of the pass), and its composite and final passes when the main pass ends; the first-person hand goes in with
 * the level's features when the description asks for it (ExtHand). Does nothing without the variable.
 */
@Mixin(LevelRenderer.class)
abstract class LevelRendererExtMixin {
    @Shadow
    @Final
    private LevelRenderState levelRenderState;

    @Shadow
    @Final
    private GameRenderer gameRenderer;

    @Shadow
    @Final
    private TextureManager textureManager;

    @Shadow
    @Final
    private SubmitNodeStorage submitNodeStorage;

    @Inject(method = "render", at = @At("HEAD"))
    private void metalmc$extBegin(GraphicsResourceAllocator resourceAllocator, boolean renderOutline, CameraRenderState cameraState,
                                  GpuBufferSlice terrainFog, Vector4f fogColor, boolean shouldRenderSky, boolean consistentDepthRequired,
                                  CallbackInfo ci) {
        if (!MetalExtPipe.ENABLED) return;
        RenderTarget main = gameRenderer.mainRenderTarget();
        MetalExtPipe.beginFrame(cameraState, levelRenderState, Lod.LEVEL_PROJECTION, fogColor, main.width, main.height,
            textureManager.getTexture(TextureAtlas.LOCATION_BLOCKS).getTextureView());
    }

    @Inject(method = "render", at = @At("TAIL"))
    private void metalmc$extEnd(CallbackInfo ci) {
        if (MetalExtPipe.ENABLED) MetalExtPipe.endFrame();
    }

    @Inject(method = "lambda$addMainPass$0", require = 0, at = @At(value = "INVOKE",
        target = "Lcom/mojang/renderpearl/api/commands/CommandEncoder;createRenderPass(Ljava/util/function/Supplier;Lcom/mojang/renderpearl/api/textures/GpuTextureView;Ljava/util/Optional;Lcom/mojang/renderpearl/api/textures/GpuTextureView;Ljava/util/OptionalDouble;)Lcom/mojang/renderpearl/api/commands/RenderPass;"))
    private void metalmc$extMainPass(CallbackInfo ci) {
        if (MetalExtPipe.ENABLED) MetalExtPipe.redirect(2);
    }

    @Inject(method = "executeClassicTransparency", at = @At("HEAD"))
    private void metalmc$extTranslucent(CallbackInfo ci) {
        if (MetalExtPipe.ENABLED) MetalExtPipe.translucent();
    }

    /** The first-person hand with the level's features (ExtHand), before the level prepares them. */
    @Inject(method = "render", at = @At(value = "INVOKE",
        target = "Lnet/minecraft/client/renderer/feature/FeatureRenderDispatcher;prepareFrame(Lnet/minecraft/client/renderer/SubmitNodeStorage;)Lnet/minecraft/client/renderer/feature/FeatureRenderDispatcher$PreparedFrame;"))
    private void metalmc$extHand(CallbackInfo ci) {
        if (MetalExtPipe.ENABLED) ExtHand.submit(gameRenderer, submitNodeStorage);
    }
}
