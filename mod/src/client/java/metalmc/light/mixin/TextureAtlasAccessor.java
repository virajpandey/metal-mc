package metalmc.light.mixin;

import java.util.Map;
import net.minecraft.client.renderer.texture.TextureAtlas;
import net.minecraft.client.renderer.texture.TextureAtlasSprite;
import net.minecraft.resources.Identifier;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

/** The block atlas's sprites by name and its size, for foliage's sprite classes (metalmc.backend.MetalFoliage). */
@Mixin(TextureAtlas.class)
public interface TextureAtlasAccessor {
    @Accessor("texturesByName")
    Map<Identifier, TextureAtlasSprite> metalmc$texturesByName();

    @Accessor("width")
    int metalmc$width();

    @Accessor("height")
    int metalmc$height();
}
