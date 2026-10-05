package metalmc.extpipe.mixin;

import metalmc.extpipe.ExtHand;
import net.minecraft.client.renderer.GameRenderer;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Vanilla's own first-person hand pass is skipped in a frame whose hand already went into the external G-buffer. */
@Mixin(GameRenderer.class)
abstract class GameRendererExtMixin {
    @Inject(method = "renderItemInHand", at = @At("HEAD"), cancellable = true)
    private void metalmc$extHand(CallbackInfo ci) {
        if (ExtHand.drawn) {
            ExtHand.drawn = false;
            ci.cancel();
        }
    }
}
