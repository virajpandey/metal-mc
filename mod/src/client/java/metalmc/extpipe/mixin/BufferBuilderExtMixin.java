package metalmc.extpipe.mixin;

import com.mojang.blaze3d.vertex.BufferBuilder;
import metalmc.extpipe.BlockIds;
import org.spongepowered.asm.mixin.Final;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.Shadow;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

/**
 * Block ids for the external pipeline (metalmc.extpipe.BlockIds): each BLOCK-format vertex written while the section
 * compiler has a block with an id in hand gets that id (blocks' quads and fluids both come through this overload).
 */
@Mixin(BufferBuilder.class)
abstract class BufferBuilderExtMixin {
    @Shadow
    private long vertexPointer;

    @Shadow
    @Final
    private boolean blockFormat;

    @Inject(method = "addVertex(FFFIFFIIFFF)V", at = @At("TAIL"))
    private void metalmc$blockId(CallbackInfo ci) {
        if (blockFormat && BlockIds.ENABLED) BlockIds.tag(vertexPointer);
    }
}
