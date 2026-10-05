package metalmc.extpipe.mixin;

import metalmc.extpipe.BlockIds;
import net.minecraft.client.renderer.block.BlockAndTintGetter;
import net.minecraft.client.renderer.block.FluidRenderer;
import net.minecraft.client.renderer.chunk.SectionCompiler;
import net.minecraft.core.BlockPos;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.material.FluidState;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.ModifyVariable;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfoReturnable;

/**
 * Block ids for the external pipeline (metalmc.extpipe.BlockIds, only with its block.properties): the section compiler
 * says which block's vertices come next (a fluid's are its own block's, not the waterlogged block's), and BufferBuilder
 * tags each of them (BufferBuilderExtMixin).
 */
@Mixin(SectionCompiler.class)
abstract class SectionCompilerExtMixin {
    @ModifyVariable(method = "compile", at = @At("STORE"), ordinal = 0)
    private BlockState metalmc$blockId(BlockState state) {
        if (BlockIds.ENABLED) BlockIds.begin(state);
        return state;
    }

    @Redirect(method = "compile", at = @At(value = "INVOKE",
        target = "Lnet/minecraft/client/renderer/block/FluidRenderer;tesselate(Lnet/minecraft/client/renderer/block/BlockAndTintGetter;Lnet/minecraft/core/BlockPos;Lnet/minecraft/client/renderer/block/FluidRenderer$Output;Lnet/minecraft/world/level/block/state/BlockState;Lnet/minecraft/world/level/material/FluidState;)V"))
    private void metalmc$fluidId(FluidRenderer renderer, BlockAndTintGetter level, BlockPos pos, FluidRenderer.Output output,
                                 BlockState blockState, FluidState fluidState) {
        if (BlockIds.ENABLED) BlockIds.begin(fluidState.createLegacyBlock());
        renderer.tesselate(level, pos, output, blockState, fluidState);
        if (BlockIds.ENABLED) BlockIds.begin(blockState);
    }

    @Inject(method = "compile", at = @At("RETURN"))
    private void metalmc$blockIdsDone(CallbackInfoReturnable<SectionCompiler.Results> cir) {
        if (BlockIds.ENABLED) BlockIds.end();
    }
}
