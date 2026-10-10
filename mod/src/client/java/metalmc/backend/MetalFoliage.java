package metalmc.backend;

import com.mojang.renderpearl.api.textures.GpuTextureView;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.util.Map;
import java.util.Set;
import net.minecraft.client.Minecraft;
import net.minecraft.client.renderer.texture.AbstractTexture;
import net.minecraft.client.renderer.texture.TextureAtlas;
import net.minecraft.client.renderer.texture.TextureAtlasSprite;
import net.minecraft.resources.Identifier;
import org.lwjgl.system.MemoryUtil;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Bridge to foliage (Sources/MetalMCNative/Foliage.swift, METALMC_EXP=leaflight with lit, and wave): the near chunks
 * look up each quad's class (leaf, plant) and how it sways by the block atlas sprite its texture comes from, in a class
 * map the native side builds from the sprites this sends: every leaf and plant sprite's rectangle in the atlas, with its
 * code (class | sway << 2). Sent from the near chunks' draw whenever the atlas they sample changes (a resource reload).
 * Render thread. The bindings live here so the feature stays in its own files.
 */
public final class MetalFoliage {
    private MetalFoliage() {
    }

    // The G-buffer's classes and the ways a sprite sways (Foliage.swift).
    private static final int LEAF = 1, PLANT = 2;
    private static final int SWAY_LEAF = 1, SWAY_PLANT = 2, SWAY_UPPER = 3, SWAY_HANGING = 4;

    /** The upper halves of double plants (their bottom sways as much as the lower half's top). */
    private static final Set<String> UPPER = Set.of("tall_grass_top", "large_fern_top", "sunflower_top", "sunflower_front",
        "sunflower_back", "lilac_top", "rose_bush_top", "peony_top", "pitcher_plant_top", "pitcher_crop_top_stage_3",
        "pitcher_crop_top_stage_4");
    /** Plants that stand on the ground and sway from their roots. */
    private static final Set<String> PLANTS = Set.of("short_grass", "grass", "fern", "dead_bush", "bush", "firefly_bush",
        "short_dry_grass", "tall_dry_grass", "dandelion", "poppy", "blue_orchid", "allium", "azure_bluet", "red_tulip",
        "orange_tulip", "white_tulip", "pink_tulip", "oxeye_daisy", "cornflower", "lily_of_the_valley", "wither_rose",
        "torchflower", "open_eyeblossom", "closed_eyeblossom", "cactus_flower", "tall_grass_bottom", "large_fern_bottom",
        "sunflower_bottom", "lilac_bottom", "rose_bush_bottom", "peony_bottom", "pitcher_plant_bottom", "mangrove_propagule",
        "brown_mushroom", "red_mushroom", "sweet_berry_bush_stage0", "sweet_berry_bush_stage1", "sweet_berry_bush_stage2",
        "sweet_berry_bush_stage3");
    /** Crops by their stage textures' prefixes. */
    private static final String[] CROPS = {"wheat_stage", "carrots_stage", "potatoes_stage", "beetroots_stage",
        "torchflower_crop_stage", "pitcher_crop_bottom_stage"};
    /** Flat plants on the ground and thin columns: lit as plants, they don't sway. */
    private static final Set<String> STILL = Set.of("pink_petals", "pink_petals_stem", "wildflowers", "wildflowers_stem",
        "leaf_litter", "lily_pad", "sugar_cane", "big_dripleaf_top", "big_dripleaf_stem", "small_dripleaf_top",
        "small_dripleaf_side", "small_dripleaf_stem_top", "small_dripleaf_stem_bottom");
    /** Plants that hang (their bottom sways). */
    private static final Set<String> HANGING = Set.of("pale_hanging_moss", "pale_hanging_moss_tip", "hanging_roots");

    /** The code of a sprite (its name under the atlas, e.g. block/oak_leaves): class | sway << 2, 0 for anything else. */
    static int code(String path) {
        if (!path.startsWith("block/")) return 0;
        String n = path.substring(6);
        if (n.endsWith("_leaves") || n.equals("vine") || n.equals("azalea_top") || n.equals("azalea_side")
            || n.equals("flowering_azalea_top") || n.equals("flowering_azalea_side")) return LEAF | SWAY_LEAF << 2;
        if (UPPER.contains(n)) return PLANT | SWAY_UPPER << 2;
        if (PLANTS.contains(n) || n.endsWith("_sapling")) return PLANT | SWAY_PLANT << 2;
        for (String c : CROPS) if (n.startsWith(c)) return PLANT | SWAY_PLANT << 2;
        if (STILL.contains(n)) return PLANT;
        if (HANGING.contains(n)) return PLANT | SWAY_HANGING << 2;
        return 0;
    }

    private static Boolean enabled;
    private static GpuTextureView sent;

    /** Resolved on first use, so nothing loads the library unless the Metal backend is running. */
    private static final class Native {
        private static final Linker LINKER = Linker.nativeLinker();
        private static final SymbolLookup LIB = SymbolLookup.libraryLookup(NativeLibrary.path(), Arena.global());

        private static MethodHandle h(String name, MemoryLayout ret, MemoryLayout... args) {
            FunctionDescriptor fd = ret == null ? FunctionDescriptor.ofVoid(args) : FunctionDescriptor.of(ret, args);
            var addr = LIB.find(name).orElseThrow(() -> new IllegalStateException("missing native symbol " + name));
            return LINKER.downcallHandle(addr, fd);
        }

        static final MethodHandle ENABLED = h("mmc_foliage_enabled", JAVA_INT);
        static final MethodHandle SET_SPRITES = h("mmc_foliage_set_sprites", null, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_INT);
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new IllegalStateException(t);
    }

    /**
     * The near chunks' draw, with the atlas view it samples: sends the block atlas's leaf and plant sprites when that view
     * changed (and leaflight or wave is on). Render thread.
     */
    public static void ensure(GpuTextureView view) {
        if (view == sent || MetalDevice.current == null) return;
        if (enabled == null) {
            try { enabled = (int) Native.ENABLED.invokeExact() != 0; } catch (Throwable t) { throw rethrow(t); }
        }
        sent = view;
        if (!enabled || !(view instanceof MetalTextureView mv)) return;
        AbstractTexture tex = Minecraft.getInstance().getTextureManager().getTexture(TextureAtlas.LOCATION_BLOCKS);
        if (!(tex instanceof TextureAtlas atlas)) return;
        var acc = (metalmc.light.mixin.TextureAtlasAccessor) atlas;
        Map<Identifier, TextureAtlasSprite> sprites = acc.metalmc$texturesByName();
        int[] rects = new int[5 * sprites.size()];
        int n = 0;
        for (var e : sprites.entrySet()) {
            int code = code(e.getKey().getPath());
            if (code == 0) continue;
            TextureAtlasSprite s = e.getValue();
            rects[5 * n] = s.getX();
            rects[5 * n + 1] = s.getY();
            rects[5 * n + 2] = s.contents().width();
            rects[5 * n + 3] = s.contents().height();
            rects[5 * n + 4] = code;
            n++;
        }
        long addr = MemoryUtil.nmemAlloc(4L * Math.max(1, 5 * n));
        try {
            for (int i = 0; i < 5 * n; i++) MemoryUtil.memPutInt(addr + 4L * i, rects[i]);
            Native.SET_SPRITES.invokeExact(mv.handle, acc.metalmc$width(), acc.metalmc$height(), addr, n);
        } catch (Throwable t) {
            throw rethrow(t);
        } finally {
            MemoryUtil.nmemFree(addr);
        }
    }
}
