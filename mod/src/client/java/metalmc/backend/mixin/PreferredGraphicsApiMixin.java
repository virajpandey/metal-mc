package metalmc.backend.mixin;

import com.mojang.renderpearl.api.device.GpuBackend;
import metalmc.backend.MetalGpuBackend;
import net.minecraft.client.PreferredGraphicsApi;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/** With -Dmetalmc.backend=metal, puts the MetalMC backend first in the list Minecraft tries at startup. */
@Mixin(PreferredGraphicsApi.class)
public class PreferredGraphicsApiMixin {
    @Inject(at = @At("RETURN"), method = "getBackendsToTry", cancellable = true)
    private void metalmc$preferMetal(CallbackInfoReturnable<GpuBackend[]> cir) {
        if (!"metal".equals(System.getProperty("metalmc.backend"))) return;
        GpuBackend[] vanilla = cir.getReturnValue();
        GpuBackend[] withMetal = new GpuBackend[vanilla.length + 1];
        withMetal[0] = new MetalGpuBackend();
        System.arraycopy(vanilla, 0, withMetal, 1, vanilla.length);
        cir.setReturnValue(withMetal);
    }
}
