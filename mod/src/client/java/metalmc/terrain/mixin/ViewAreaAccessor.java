package metalmc.terrain.mixin;

import net.minecraft.client.RotatingSectionStorage;
import net.minecraft.client.renderer.ViewArea;
import net.minecraft.client.renderer.chunk.SectionRenderDispatcher;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** Vanilla's section storage, to list the sections it has compiled (SectionOcclusion.recordCompiled). */
@Mixin(ViewArea.class)
public interface ViewAreaAccessor {
    @Accessor("sections")
    RotatingSectionStorage<SectionRenderDispatcher.RenderSection> metalmc$sections();
}
