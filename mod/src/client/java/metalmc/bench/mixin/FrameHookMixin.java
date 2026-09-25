package metalmc.bench.mixin;

import metalmc.bench.Bench;
import net.minecraft.client.Minecraft;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/** Timestamps the start of every rendered frame. */
@Mixin(Minecraft.class)
public class FrameHookMixin {
    @Inject(at = @At("HEAD"), method = "renderFrame")
    private void metalmc$onRenderFrame(boolean renderLevel, CallbackInfo info) {
        Bench.onFrame();
    }
}
