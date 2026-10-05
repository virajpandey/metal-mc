package metalmc.extpipe;

import com.mojang.logging.LogUtils;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import metalmc.backend.MetalExtPipe;
import net.minecraft.core.registries.BuiltInRegistries;
import net.minecraft.resources.Identifier;
import net.minecraft.world.level.block.Block;
import net.minecraft.world.level.block.state.BlockState;
import net.minecraft.world.level.block.state.properties.Property;
import org.lwjgl.system.MemoryUtil;
import org.slf4j.Logger;

/**
 * Block ids for the external pipeline's terrain programs (OptiFine's mc_Entity.x): read from a block.properties file in
 * the pipeline's directory (OptiFine's format, preprocessed: lines {@code block.<id>=<block>[:<property>=<values>]...},
 * the namespace optional). While a chunk section is meshed, each vertex of a block gets its block's id: the low 8 bits
 * in the vertex color's alpha (always 255 in vanilla's terrain), bits 8-14 in the block light's high byte, and bit 14 of
 * the sky light says an id is there (light values only use the low byte). Only the external pipeline's terrain programs
 * read these vertices then; vanilla's own terrain shaders aren't used for the level while it runs. Off without the file.
 */
public final class BlockIds {
    private BlockIds() {
    }

    private static final Logger LOGGER = LogUtils.getLogger();
    private static final Map<BlockState, Integer> IDS = new IdentityHashMap<>();
    public static final boolean ENABLED = MetalExtPipe.ENABLED && load();
    /** The id of the block being meshed on this thread, or -1. */
    private static final ThreadLocal<int[]> CURRENT = ThreadLocal.withInitial(() -> new int[]{-1});

    private record Entry(int id, Block block, List<String[]> predicates) {
    }

    private static boolean load() {
        Path file = Path.of(MetalExtPipe.DIR).resolve("block.properties");
        if (!Files.isRegularFile(file)) return false;
        List<Entry> entries = new ArrayList<>();
        try {
            for (String line : Files.readAllLines(file, StandardCharsets.UTF_8)) {
                line = line.strip();
                if (!line.startsWith("block.")) continue;
                int eq = line.indexOf('=');
                if (eq < 0) continue;
                int id;
                try {
                    id = Integer.parseInt(line.substring(6, eq).strip());
                } catch (NumberFormatException e) {
                    continue;
                }
                for (String item : line.substring(eq + 1).strip().split("\\s+")) {
                    if (item.isEmpty()) continue;
                    String[] parts = item.split(":");
                    int at = 0;
                    String ns = "minecraft";
                    if (parts.length > 1 && !parts[1].contains("=")) {
                        ns = parts[0];
                        at = 1;
                    }
                    Identifier key = Identifier.tryParse(ns + ":" + parts[at]);
                    if (key == null) continue;
                    Block block = BuiltInRegistries.BLOCK.getOptional(key).orElse(null);
                    if (block == null) continue;
                    List<String[]> preds = new ArrayList<>();
                    for (int i = at + 1; i < parts.length; i++) {
                        int e = parts[i].indexOf('=');
                        if (e > 0) preds.add(new String[]{parts[i].substring(0, e), parts[i].substring(e + 1)});
                    }
                    entries.add(new Entry(id, block, preds));
                }
            }
        } catch (IOException e) {
            LOGGER.warn("external pipeline: couldn't read {}", file, e);
            return false;
        }
        // Every block state's id: an entry with property conditions wins over a plain one, else the first match.
        int states = 0;
        for (Block block : BuiltInRegistries.BLOCK) {
            for (BlockState state : block.getStateDefinition().getPossibleStates()) {
                int best = -1;
                boolean specific = false;
                for (Entry e : entries) {
                    if (e.block != block) continue;
                    if (!matches(state, e.predicates)) continue;
                    boolean s = !e.predicates.isEmpty();
                    if (best < 0 || (s && !specific)) {
                        best = e.id;
                        specific = s;
                    }
                }
                if (best >= 0 && best < 32768) {
                    IDS.put(state, best);
                    states++;
                }
            }
        }
        LOGGER.info("external pipeline: block ids for {} block states from {}", states, file);
        return states > 0;
    }

    private static boolean matches(BlockState state, List<String[]> predicates) {
        for (String[] p : predicates) {
            Property<?> prop = state.getBlock().getStateDefinition().getProperty(p[0]);
            if (prop == null) return false;
            String value = valueName(state, prop);
            boolean any = false;
            for (String v : p[1].split(",")) {
                if (v.equals(value)) {
                    any = true;
                    break;
                }
            }
            if (!any) return false;
        }
        return true;
    }

    private static <T extends Comparable<T>> String valueName(BlockState state, Property<T> prop) {
        return prop.getName(state.getValue(prop));
    }

    /** From the section compiler: the block whose vertices come next on this thread (null: none). */
    public static void begin(BlockState state) {
        Integer id = state == null ? null : IDS.get(state);
        CURRENT.get()[0] = id == null ? -1 : id;
    }

    public static void end() {
        CURRENT.get()[0] = -1;
    }

    /** From BufferBuilder, after a vertex of the BLOCK format was written at {@code pointer}. */
    public static void tag(long pointer) {
        int id = CURRENT.get()[0];
        if (id < 0) return;
        MemoryUtil.memPutByte(pointer + 15, (byte) id);
        MemoryUtil.memPutByte(pointer + 25, (byte) ((id >> 8) & 0x7f));
        MemoryUtil.memPutByte(pointer + 27, (byte) (MemoryUtil.memGetByte(pointer + 27) | 0x40));
    }
}
