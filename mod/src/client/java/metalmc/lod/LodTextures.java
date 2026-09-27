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
        List<float[]> meanRows = new ArrayList<>();
        for (int i = 0; ; i++) {
            String top = MetalLod.spriteName(i, true), side = MetalLod.spriteName(i, false);
            if (top == null) break;
            float[] r = new float[8];
            put(atlas, top, r, 0);
            put(atlas, side, r, 4);
            rows.add(r);
            float[] m = {-1, -1, -1, -1, -1, -1};
            mean(atlas, top, m, 0);
            mean(atlas, side, m, 3);
            meanRows.add(m);
        }
        float[] rects = new float[rows.size() * 8], means = new float[rows.size() * 6];
        for (int i = 0; i < rows.size(); i++) {
            System.arraycopy(rows.get(i), 0, rects, 8 * i, 8);
            System.arraycopy(meanRows.get(i), 0, means, 6 * i, 6);
        }
        if (MetalLod.setAtlas(view, rects, means, rows.size())) sent = view;
    }

    /**
     * The mean color of a texture's opaque texels (all animation frames) in the loaded resource pack, like
     * tools/lod_colors.py computes vanilla's; the LOD scales its colors by the ratio.
     */
    private static void mean(TextureAtlas atlas, String name, float[] out, int at) {
        if (name.isEmpty()) return;
        TextureAtlasSprite s = atlas.getSprite(Identifier.withDefaultNamespace("block/" + name));
        com.mojang.blaze3d.platform.NativeImage img = ((metalmc.lod.mixin.SpriteContentsAccessor) s.contents()).metalmc$originalImage();
        if (img == null) return;
        // Every animation frame, as tools/lod_colors.py averages vanilla's textures.
        int w = img.getWidth(), h = img.getHeight();
        long r = 0, g = 0, b = 0, n = 0;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int p = img.getPixel(x, y);
                if ((p >>> 24) < 128) continue;
                r += (p >> 16) & 255;
                g += (p >> 8) & 255;
                b += p & 255;
                n++;
            }
        }
        if (n == 0) return;
        out[at] = r / 255f / n;
        out[at + 1] = g / 255f / n;
        out[at + 2] = b / 255f / n;
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
