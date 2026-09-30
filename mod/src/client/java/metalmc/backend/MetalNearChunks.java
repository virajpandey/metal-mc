package metalmc.backend;

import com.mojang.renderpearl.api.buffers.GpuBufferSlice;
import com.mojang.renderpearl.util.TextureViewAndSampler;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.LongBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Bridge to the native near-chunk renderer (Sources/MetalMCNative/NearChunks.swift, METALMC_EXP=nearchunks): vanilla's
 * solid and cutout section layers repacked into 32-byte quad records in one arena and drawn from it. Only works on the
 * Metal backend. The bindings live here rather than in Mtl so this feature stays in its own files.
 */
public final class MetalNearChunks {
    private MetalNearChunks() {
    }

    /** METALMC_EXP=nearchunks (off by default). The native side reads the same variable. */
    public static final boolean ENABLED = experiment("nearchunks");
    /**
     * METALMC_EXP=nearchunks,nearslim: a repacked layer's vertices are left out of vanilla's vertex heaps (a 28-byte
     * placeholder goes there instead), which is where the memory saving comes from. Needs multi-draw-indirect terrain
     * (the default); see docs/near-chunks-design.md.
     */
    public static final boolean SLIM = ENABLED && experiment("nearslim");

    private static boolean experiment(String name) {
        String v = System.getenv("METALMC_EXP");
        if (v == null) return false;
        for (String s : v.split(",")) {
            if (s.trim().equals(name)) return true;
        }
        return false;
    }

    public static boolean available() {
        return ENABLED && MetalDevice.current != null;
    }

    /** Resolved on first use, so nothing loads the library unless the Metal backend is running. */
    private static final class Native {
        private static final Linker LINKER = Linker.nativeLinker();
        private static final SymbolLookup LIB = SymbolLookup.libraryLookup(NativeLibrary.path(), Arena.global());

        private static MethodHandle h(String name, MemoryLayout ret, MemoryLayout... args) {
            FunctionDescriptor fd = ret == null ? FunctionDescriptor.ofVoid(args) : FunctionDescriptor.of(ret, args);
            return LINKER.downcallHandle(LIB.find(name).orElseThrow(() -> new IllegalStateException("missing native symbol " + name)), fd);
        }

        // Not critical downcalls: add encodes a whole layer (up to a millisecond on a mesh worker), draw encodes a frame's draws.
        static final MethodHandle READY = h("mmc_near_ready", JAVA_INT);
        static final MethodHandle ADD = h("mmc_near_add", JAVA_INT, JAVA_LONG, JAVA_INT);
        static final MethodHandle FREE = h("mmc_near_free", null, JAVA_INT);
        static final MethodHandle DRAW = h("mmc_near_draw", JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT);
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new IllegalStateException(t);
    }

    /** True once the near-chunk shaders are compiled (the first call starts compiling them). Render thread. */
    public static boolean ready() {
        if (!available()) return false;
        try { return (int) Native.READY.invokeExact() == 1; } catch (Throwable t) { throw rethrow(t); }
    }

    /**
     * Repacks one section layer of vanilla BLOCK vertices ({@code vertexCount}, 28 bytes each, at native {@code address}).
     * Returns its entry id (always positive), or 0 if the layer stays vanilla's. Any thread (the mesh workers).
     */
    public static int add(long address, int vertexCount) {
        if (!available()) return 0;
        try { return (int) Native.ADD.invokeExact(address, vertexCount); } catch (Throwable t) { throw rethrow(t); }
    }

    /** Frees an entry (its section mesh closed). Any thread. */
    public static void free(int id) {
        if (id == 0 || !available()) return;
        try { Native.FREE.invokeExact(id); } catch (Throwable t) { throw rethrow(t); }
    }

    /**
     * Draws {@code count} records (4 ints each: entry, first quad, quad count, section index into {@code sections}) of
     * one layer (0 solid, 1 cutout) into the open render pass, with the uniforms vanilla bound for its own draw of that
     * layer, then restores Minecraft's pipeline state. Returns the records drawn, or -1 if they couldn't be drawn.
     * Render thread.
     */
    public static int draw(int layer, long records, int count, GpuBufferSlice sections, boolean wireframe) {
        MetalDevice device = MetalDevice.current;
        if (device == null || count == 0) return 0;
        MetalRenderPass pass = device.encoder().currentRenderPass();
        if (pass == null) return -1;
        if (!(pass.uniformValue("Projection") instanceof GpuBufferSlice projection)
            || !(pass.uniformValue("TerrainUniform") instanceof GpuBufferSlice terrain)
            || !(pass.uniformValue("Globals") instanceof GpuBufferSlice globals)
            || !(pass.uniformValue("Fog") instanceof GpuBufferSlice fog)
            || !(pass.uniformValue("Sampler0") instanceof TextureViewAndSampler atlas)
            || !(pass.uniformValue("Sampler2") instanceof TextureViewAndSampler lightmap)) {
            return -1;
        }
        int drawn;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer res = stack.mallocLong(14);
            slice(res, 0, sections);
            slice(res, 2, projection);
            slice(res, 4, terrain);
            slice(res, 6, globals);
            slice(res, 8, fog);
            res.put(10, ((MetalTextureView) atlas.view()).handle).put(11, ((MetalSampler) atlas.sampler()).handle);
            res.put(12, ((MetalTextureView) lightmap.view()).handle).put(13, ((MetalSampler) lightmap.sampler()).handle);
            drawn = (int) Native.DRAW.invokeExact(layer, records, count, device.encoder().currentSubmit(), MemoryUtil.memAddress(res), wireframe ? 1 : 0);
        } catch (Throwable t) {
            throw rethrow(t);
        }
        pass.restoreAfterExternalDraw();
        return drawn;
    }

    private static void slice(LongBuffer res, int i, GpuBufferSlice s) {
        res.put(i, ((MetalBuffer) s.buffer()).handle).put(i + 1, s.offset());
    }
}
