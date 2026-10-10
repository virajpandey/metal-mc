package metalmc.render.mixin;

import com.mojang.blaze3d.pipeline.RenderTarget;
import metalmc.render.RenderScale;
import net.minecraft.client.renderer.SkyRenderer;
import org.objectweb.asm.Opcodes;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Redirect;

/**
 * Render scale (metalmc.render.RenderScale): vanilla's sky renderer keeps the main target it was made with (LevelRenderer
 * makes it once), so its pass draws into this frame's level target instead when the level is drawn at a scale.
 */
@Mixin(SkyRenderer.class)
abstract class SkyRendererScaleMixin {
    @Shadow
    @Final
    private RenderTarget renderTarget;

    @Redirect(method = "render", at = @At(value = "FIELD", opcode = Opcodes.GETFIELD,
        target = "Lnet/minecraft/client/renderer/SkyRenderer;renderTarget:Lcom/mojang/blaze3d/pipeline/RenderTarget;"))
    private RenderTarget metalmc$levelTarget(SkyRenderer self) {
        return RenderScale.level(renderTarget);
    }
}
