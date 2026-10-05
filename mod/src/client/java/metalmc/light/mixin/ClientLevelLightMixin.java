package metalmc.light.mixin;

import metalmc.backend.MetalColoredLight;
import metalmc.light.ColoredLight;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.state.BlockState;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Colored block light (METALMC_EXP=lit,coloredlight, metalmc.light.ColoredLight): every block change the client's level
 * announces (from the server, or predicted by the player's own actions), after the chunk holds the new state, goes to the
 * light volume if it changes how the block gives off or lets through light.
 */
@Mixin(ClientLevel.class)
abstract class ClientLevelLightMixin {
    @Inject(method = "sendBlockUpdated", at = @At("HEAD"))
    private void metalmc$coloredLight(BlockPos pos, BlockState oldState, BlockState newState, int flags, CallbackInfo ci) {
        if (MetalColoredLight.ENABLED) ColoredLight.blockChanged((ClientLevel) (Object) this, pos, oldState, newState);
    }
}
