package metalmc.lod;

import com.mojang.renderpearl.api.textures.GpuTextureView;
import java.util.ArrayList;
import java.util.List;
import metalmc.backend.MetalLod;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.texture.AbstractTexture;
import net.minecraft.client.renderer.texture.TextureAtlas;
import net.minecraft.client.renderer.texture.TextureAtlasSprite;
import net.minecraft.resources.Identifier;

/**
 * Hands the LOD Minecraft's block atlas and where each LOD material's textures sit in it, for texture
 * detail on far terrain. Sent again whenever the atlas changes (resource reloads). Render thread only.
 */
public final class LodTextures {
    private LodTextures() {
    }

    private static GpuTextureView sent;

    public static void ensure() {
        AbstractTexture tex = Minecraft.getInstance().getTextureManager().getTexture(TextureAtlas.LOCATION_BLOCKS);
        if (!(tex instanceof TextureAtlas atlas)) return;
        GpuTextureView view = atlas.getTextureView();
        if (view == null || view == sent) return;
        List<float[]> rows = new ArrayList<>();
        for (int i = 0; ; i++) {
            String top = MetalLod.spriteName(i, true), side = MetalLod.spriteName(i, false);
            if (top == null) break;
            float[] r = new float[8];
            put(atlas, top, r, 0);
            put(atlas, side, r, 4);
            rows.add(r);
        }
        float[] rects = new float[rows.size() * 8];
        for (int i = 0; i < rows.size(); i++) System.arraycopy(rows.get(i), 0, rects, 8 * i, 8);
        if (MetalLod.setAtlas(view, rects, rows.size())) sent = view;
    }

    private static void put(TextureAtlas atlas, String name, float[] out, int at) {
        if (name.isEmpty()) return;
        TextureAtlasSprite s = atlas.getSprite(Identifier.withDefaultNamespace("block/" + name));
        out[at] = s.getU0();
        out[at + 1] = s.getV0();
        out[at + 2] = s.getU1();
        out[at + 3] = s.getV1();
    }
}
