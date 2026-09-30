package metalmc.terrain;

import com.mojang.blaze3d.vertex.DefaultVertexFormat;
import com.mojang.blaze3d.vertex.MeshData;
import com.mojang.renderpearl.api.buffers.GpuBufferSlice;
import java.nio.ByteBuffer;
import metalmc.backend.MetalNearChunks;
import net.minecraft.client.renderer.chunk.ChunkSectionLayer;
import net.minecraft.client.renderer.chunk.CompiledSectionMesh;
import net.minecraft.client.renderer.chunk.SectionCompiler;
import net.minecraft.client.renderer.chunk.SectionMesh;
import org.lwjgl.system.MemoryUtil;

/**
 * Near chunks in our own compact format (METALMC_EXP=nearchunks, off by default; docs/near-chunks-design.md).
 * <ul>
 * <li>On the mesh worker, right after vanilla builds a section (and the facing sort, so the quad order is final), its
 * solid and cutout layers are repacked into the native arena (NearChunks.swift). The entry ids ride on the section's
 * mesh, which frees them when vanilla closes it.</li>
 * <li>While vanilla extracts its draws, those layers' draws (split by facing like vanilla's) go into per-layer record
 * lists instead of vanilla's indirect draws.</li>
 * <li>Right after vanilla's own draws of each layer, the lists are drawn from the arena with a shader equivalent to
 * vanilla's, reading the same uniforms and per-section data vanilla's draws use.</li>
 * </ul>
 * A layer the codec can't hold, and every frame the diversion is off (shaders still compiling, terrain drawn without
 * multi-draw-indirect), stays vanilla's. Translucent terrain is always vanilla's.
 */
public final class NearChunks {
    private NearChunks() {
    }

    public static final boolean ENABLED = MetalNearChunks.ENABLED;

    private static final ChunkSectionLayer[] LAYERS = {ChunkSectionLayer.SOLID, ChunkSectionLayer.CUTOUT};
    private static final int RECORD_BYTES = 16;

    // Render thread: whether this frame's solid and cutout draws are diverted, and each layer's records (entry, first
    // quad, quad count, section index), in native memory.
    private static boolean diverting;
    private static final long[] lists = new long[2];
    private static final int[] capacity = new int[2];
    private static final int[] counts = new int[2];
    private static boolean warned;

    /** A 28-byte placeholder for vanilla's heap in place of a repacked layer's vertices (nearslim). */
    private static final ByteBuffer SLIM_PLACEHOLDER = MemoryUtil.memCalloc(DefaultVertexFormat.BLOCK.getVertexSize());

    private static int index(ChunkSectionLayer layer) {
        return layer == ChunkSectionLayer.SOLID ? 0 : layer == ChunkSectionLayer.CUTOUT ? 1 : -1;
    }

    /**
     * Mesh worker (or the render thread for a synchronous compile), when SectionCompiler.compile returns. A section's solid
     * and cutout layers are repacked together or not at all: the grass side overlay (cutout) must come out of the same
     * vertex shader as the dirt side under it (solid), since it only passes the depth test at exactly equal depth.
     */
    public static void afterCompile(SectionCompiler.Results results) {
        if (!MetalNearChunks.available()) return;
        NearData data = (NearData) (Object) results;
        int solid = repack(results.renderedLayers.get(ChunkSectionLayer.SOLID));
        int cutout = solid < 0 ? -1 : repack(results.renderedLayers.get(ChunkSectionLayer.CUTOUT));
        if (solid < 0 || cutout < 0) {
            if (solid > 0) MetalNearChunks.free(solid);
            if (cutout > 0) MetalNearChunks.free(cutout);
            return;
        }
        data.metalmc$setNearEntry(ChunkSectionLayer.SOLID, solid);
        data.metalmc$setNearEntry(ChunkSectionLayer.CUTOUT, cutout);
    }

    /** Repacks one layer: its entry id, 0 if there's no layer, or -1 if it has to stay vanilla's. */
    private static int repack(MeshData mesh) {
        if (mesh == null) return 0;
        MeshData.DrawState state = mesh.drawState();
        int vertices = state.vertexCount();
        if (!state.format().equals(DefaultVertexFormat.BLOCK) || mesh.indexBuffer() != null || vertices == 0 || vertices % 4 != 0) return -1;
        ByteBuffer vb = mesh.vertexBuffer();
        if (vb.remaining() < vertices * DefaultVertexFormat.BLOCK.getVertexSize()) return -1;
        int id = MetalNearChunks.add(MemoryUtil.memAddress(vb), vertices);   // positive, or 0 if refused
        return id > 0 ? id : -1;
    }

    /** Frees a closing mesh's entries (any thread; vanilla closes meshes under its copy lock). */
    public static void release(NearData data) {
        for (ChunkSectionLayer layer : LAYERS) {
            int id = data.metalmc$nearEntry(layer);
            if (id != 0) {
                data.metalmc$setNearEntry(layer, 0);
                MetalNearChunks.free(id);
            }
        }
    }

    /**
     * nearslim: what vanilla uploads for a layer. A repacked layer (while the near-chunk shaders are ready) gets a
     * placeholder instead of its vertices, and is marked so vanilla never draws it. Mesh worker.
     */
    public static ByteBuffer uploadVertices(ChunkSectionLayer layer, CompiledSectionMesh mesh, ByteBuffer vertices) {
        if (!MetalNearChunks.SLIM || vertices == null || index(layer) < 0) return vertices;
        NearData data = (NearData) (Object) mesh;
        if (data.metalmc$nearEntry(layer) == 0 || !MetalNearChunks.ready()) return vertices;
        data.metalmc$setNearSlim(layer);
        return SLIM_PLACEHOLDER.duplicate();
    }

    /** Render thread, as vanilla starts extracting this frame's section draws. */
    public static void beginFrame(boolean multiDrawIndirect) {
        counts[0] = 0;
        counts[1] = 0;
        diverting = multiDrawIndirect && MetalNearChunks.available() && MetalNearChunks.ready();
    }

    /** The mesh's arena entry for a layer if this frame draws it from the arena, else 0 (vanilla draws it). */
    public static int divertedEntry(SectionMesh mesh, ChunkSectionLayer layer) {
        if (!diverting || !(mesh instanceof NearData data)) return 0;
        return data.metalmc$nearEntry(layer);
    }

    /** True if vanilla's heap holds only a placeholder for this layer (nearslim), so vanilla must not draw it. */
    public static boolean slimmed(SectionMesh mesh, ChunkSectionLayer layer) {
        return mesh instanceof NearData data && data.metalmc$nearSlim(layer);
    }

    /** Adds a run of a section layer's quads to this frame's list. Render thread. */
    public static void add(ChunkSectionLayer layer, int entry, int firstQuad, int quads, int sectionIndex) {
        int l = index(layer);
        if (l < 0 || quads <= 0) return;
        if (counts[l] == capacity[l]) {
            int grown = Math.max(1024, capacity[l] * 2);
            lists[l] = MemoryUtil.nmemRealloc(lists[l], (long) grown * RECORD_BYTES);
            if (lists[l] == 0L) throw new OutOfMemoryError("near chunk draw list");
            capacity[l] = grown;
        }
        long a = lists[l] + (long) counts[l]++ * RECORD_BYTES;
        MemoryUtil.memPutInt(a, entry);
        MemoryUtil.memPutInt(a + 4, firstQuad);
        MemoryUtil.memPutInt(a + 8, quads);
        MemoryUtil.memPutInt(a + 12, sectionIndex);
    }

    /**
     * Draws a layer's list right after vanilla's own draws of the layer, with vanilla's per-section stream ({@code
     * sections}). Render thread.
     */
    public static void draw(ChunkSectionLayer layer, GpuBufferSlice sections, boolean wireframe) {
        int l = index(layer);
        if (l < 0 || counts[l] == 0) return;
        int drawn = MetalNearChunks.draw(l, lists[l], counts[l], sections, wireframe);
        if (drawn < 0 && !warned) {
            warned = true;
            System.err.println("[metalmc] near chunks: couldn't draw the " + layer.label() + " layer's " + counts[l]
                + " diverted draws this frame (pipeline or uniforms missing)");
        }
        counts[l] = 0;
    }
}
