package metalmc.lod.mixin;

import com.mojang.blaze3d.platform.NativeImage;
import net.minecraft.client.renderer.texture.SpriteContents;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** A sprite's texture as loaded (kept in memory while the atlas is), for the LOD's resource-pack colors. */
@Mixin(SpriteContents.class)
public interface SpriteContentsAccessor {
    @Accessor("originalImage")
    NativeImage metalmc$originalImage();
}
