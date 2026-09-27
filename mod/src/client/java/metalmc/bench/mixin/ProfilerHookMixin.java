package metalmc.bench.mixin;

import metalmc.bench.Bench;
import metalmc.bench.HitchProfiler;
import net.minecraft.client.Minecraft;
import net.minecraft.util.profiling.ProfilerFiller;
import net.minecraft.util.profiling.SingleTickProfiler;
import org.jspecify.annotations.Nullable;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/** -PbenchHitches=1: vanilla's per-frame profiler also feeds HitchProfiler during the timed run. */
@Mixin(Minecraft.class)
abstract class ProfilerHookMixin {
    @Inject(method = "constructProfiler", at = @At("RETURN"), cancellable = true)
    private void metalmc$hitchProfiler(boolean shouldCollectFrameProfile, @Nullable SingleTickProfiler tickProfiler,
                                       CallbackInfoReturnable<ProfilerFiller> cir) {
        if (Bench.HITCHES && Bench.running()) cir.setReturnValue(ProfilerFiller.combine(cir.getReturnValue(), HitchProfiler.INSTANCE));
    }
}
