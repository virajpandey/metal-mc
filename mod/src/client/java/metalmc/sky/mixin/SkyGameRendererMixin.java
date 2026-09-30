package metalmc.sky.mixin;

import metalmc.sky.Sky;
import net.minecraft.client.renderer.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * The sky's per-frame start (metalmc.sky.Sky): before the level's passes, and before GameRenderer.renderLevel hands the
 * camera's fog to the fog buffer, so the fog it turns off is off for vanilla's shaders and the LOD alike.
 */
@Mixin(GameRenderer.class)
abstract class SkyGameRendererMixin {
    @Inject(method = "renderLevel", at = @At("HEAD"))
    private void metalmc$skyBegin(CallbackInfo ci) {
        Sky.beginLevel(((GameRenderer) (Object) this).gameRenderState().levelRenderState);
    }
}
