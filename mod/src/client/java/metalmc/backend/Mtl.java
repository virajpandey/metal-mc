package metalmc.backend;

import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;

import static java.lang.foreign.ValueLayout.JAVA_FLOAT;
import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Static bindings to the backend half of libMetalMCNative (Sources/MetalMCNative/Backend.swift).
 * Handles and raw pointers are passed as longs. Short non-blocking calls use critical downcalls,
 * which skip the Java thread-state transition. Calls that can block (waits, compiles, nextDrawable)
 * don't use them.
 */
final class Mtl {
    private Mtl() {
    }

    private static final Linker LINKER = Linker.nativeLinker();
    private static final SymbolLookup LIB = SymbolLookup.libraryLookup(NativeLibrary.path(), Arena.global());

    private static MethodHandle h(String name, boolean critical, MemoryLayout ret, MemoryLayout... args) {
        FunctionDescriptor fd = ret == null ? FunctionDescriptor.ofVoid(args) : FunctionDescriptor.of(ret, args);
        var addr = LIB.find(name).orElseThrow(() -> new IllegalStateException("missing native symbol " + name));
        return critical ? LINKER.downcallHandle(addr, fd, Linker.Option.critical(false)) : LINKER.downcallHandle(addr, fd);
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new IllegalStateException(t);
    }

    private static final MethodHandle CTX_INIT = h("mmc_ctx_init", false, JAVA_INT);
    private static final MethodHandle CTX_LIMITS = h("mmc_ctx_limits", false, null, JAVA_LONG);
    private static final MethodHandle RELEASE = h("mmc_handle_release", false, null, JAVA_LONG);
    private static final MethodHandle BUFFER_CREATE = h("mmc_buffer_create", false, JAVA_LONG, JAVA_LONG);
    private static final MethodHandle BUFFER_CONTENTS = h("mmc_buffer_contents", true, JAVA_LONG, JAVA_LONG);
    private static final MethodHandle BUFFER_LABEL = h("mmc_buffer_label", false, null, JAVA_LONG, JAVA_LONG);
    private static final MethodHandle TEXTURE_CREATE = h("mmc_texture_create", false, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle TEXTURE_LABEL = h("mmc_texture_label", false, null, JAVA_LONG, JAVA_LONG);
    private static final MethodHandle TEXTURE_VIEW = h("mmc_texture_view", false, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT);
    private static final MethodHandle SAMPLER_CREATE = h("mmc_sampler_create", false, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_FLOAT);
    private static final MethodHandle PIPELINE_CREATE = h("mmc_pipeline_create", false, JAVA_LONG,
        JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_INT);
    private static final MethodHandle PASS_BEGIN = h("mmc_pass_begin", true, JAVA_INT,
        JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_FLOAT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle PASS_END = h("mmc_pass_end", true, null);
    private static final MethodHandle RP_SCISSOR = h("mmc_rp_scissor", true, null, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle RP_SET_PIPELINE = h("mmc_rp_set_pipeline", true, JAVA_INT, JAVA_LONG);
    private static final MethodHandle RP_BIND_BUFFER = h("mmc_rp_bind_buffer", true, null, JAVA_INT, JAVA_LONG, JAVA_LONG);
    private static final MethodHandle RP_BIND_TEXTURE = h("mmc_rp_bind_texture", true, null, JAVA_INT, JAVA_LONG, JAVA_LONG);
    private static final MethodHandle RP_BIND_TEXEL = h("mmc_rp_bind_texel_buffer", true, null, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT);
    private static final MethodHandle RP_PUSH_CONSTANTS = h("mmc_rp_push_constants", true, null, JAVA_LONG, JAVA_INT);
    private static final MethodHandle RP_SET_VERTEX_BUFFER = h("mmc_rp_set_vertex_buffer", true, null, JAVA_INT, JAVA_LONG, JAVA_LONG);
    private static final MethodHandle RP_SET_INDEX_BUFFER = h("mmc_rp_set_index_buffer", true, null, JAVA_LONG, JAVA_INT);
    private static final MethodHandle RP_DRAW_INDEXED = h("mmc_rp_draw_indexed", true, null, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle RP_MULTI_DRAW_INDEXED = h("mmc_rp_multi_draw_indexed", true, null, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle RP_DRAW_INDEXED_INDIRECT = h("mmc_rp_draw_indexed_indirect", true, null, JAVA_LONG, JAVA_LONG, JAVA_INT);
    private static final MethodHandle RP_DRAW = h("mmc_rp_draw", true, null, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle RP_MULTI_DRAW = h("mmc_rp_multi_draw", true, null, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle RP_DRAW_INDIRECT = h("mmc_rp_draw_indirect", true, null, JAVA_LONG, JAVA_LONG, JAVA_INT);
    private static final MethodHandle COPY_BUFFER = h("mmc_copy_buffer", true, null, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG);
    private static final MethodHandle COPY_BUFFER_TO_TEXTURE = h("mmc_copy_buffer_to_texture", true, null,
        JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle COPY_TEXTURE_TO_BUFFER = h("mmc_copy_texture_to_buffer", true, null,
        JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_INT);
    private static final MethodHandle COPY_TEXTURE_TO_TEXTURE = h("mmc_copy_texture_to_texture", true, null,
        JAVA_LONG, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle CLEAR_TEXTURES = h("mmc_clear_textures", true, null, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_FLOAT);
    private static final MethodHandle CLEAR_REGION = h("mmc_clear_region", true, null,
        JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_FLOAT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle SUBMIT = h("mmc_submit", false, null, JAVA_LONG);
    private static final MethodHandle WAIT_SUBMIT = h("mmc_wait_submit", false, JAVA_INT, JAVA_LONG, JAVA_LONG);
    private static final MethodHandle GPU_TIMES_TAKE = h("mmc_gpu_times_take", false, JAVA_INT, JAVA_LONG, JAVA_INT);
    private static final MethodHandle STATS_TAKE = h("mmc_stats_take", false, null, JAVA_LONG);
    private static final MethodHandle TRACE_FRAMES = h("mmc_trace_frames", false, null, JAVA_INT);
    private static final MethodHandle LOD_OPEN = h("mmc_lod_open", false, JAVA_INT, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle LOD_CENTER = h("mmc_lod_center", false, null, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle LOD_STATUS = h("mmc_lod_status", false, null, JAVA_LONG);
    private static final MethodHandle LOD_DRAW = h("mmc_lod_draw", false, JAVA_INT, JAVA_LONG, JAVA_LONG);
    private static final MethodHandle COMPLETED_SUBMIT = h("mmc_completed_submit", true, JAVA_LONG);
    private static final MethodHandle SURFACE_CREATE = h("mmc_surface2_create", false, JAVA_LONG, JAVA_LONG);
    private static final MethodHandle SURFACE_CONFIGURE = h("mmc_surface2_configure", false, null, JAVA_LONG, JAVA_INT, JAVA_INT, JAVA_INT);
    private static final MethodHandle SURFACE_ACQUIRE = h("mmc_surface2_acquire", false, JAVA_INT, JAVA_LONG);
    private static final MethodHandle SURFACE_BLIT = h("mmc_surface2_blit", false, null, JAVA_LONG, JAVA_LONG);
    private static final MethodHandle SURFACE_PRESENT = h("mmc_surface2_present", false, null, JAVA_LONG);

    static int init() {
        try { return (int) CTX_INIT.invokeExact(); } catch (Throwable t) { throw rethrow(t); }
    }

    static void limits(long out) {
        try { CTX_LIMITS.invokeExact(out); } catch (Throwable t) { throw rethrow(t); }
    }

    static void release(long handle) {
        try { RELEASE.invokeExact(handle); } catch (Throwable t) { throw rethrow(t); }
    }

    static long bufferCreate(long size) {
        try { return (long) BUFFER_CREATE.invokeExact(size); } catch (Throwable t) { throw rethrow(t); }
    }

    static long bufferContents(long handle) {
        try { return (long) BUFFER_CONTENTS.invokeExact(handle); } catch (Throwable t) { throw rethrow(t); }
    }

    static void bufferLabel(long handle, long cString) {
        try { BUFFER_LABEL.invokeExact(handle, cString); } catch (Throwable t) { throw rethrow(t); }
    }

    static long textureCreate(int format, int width, int height, int layers, int mips, int flags) {
        try { return (long) TEXTURE_CREATE.invokeExact(format, width, height, layers, mips, flags); } catch (Throwable t) { throw rethrow(t); }
    }

    static void textureLabel(long handle, long cString) {
        try { TEXTURE_LABEL.invokeExact(handle, cString); } catch (Throwable t) { throw rethrow(t); }
    }

    static long textureView(long handle, int baseMip, int mipCount) {
        try { return (long) TEXTURE_VIEW.invokeExact(handle, baseMip, mipCount); } catch (Throwable t) { throw rethrow(t); }
    }

    static long samplerCreate(int addrU, int addrV, int minFilter, int magFilter, int anisotropy, float maxLod) {
        try { return (long) SAMPLER_CREATE.invokeExact(addrU, addrV, minFilter, magFilter, anisotropy, maxLod); } catch (Throwable t) { throw rethrow(t); }
    }

    static long pipelineCreate(long name, long vs, long vsEntry, long fs, long fsEntry, long params, int count, long err, int errLen) {
        try {
            return (long) PIPELINE_CREATE.invokeExact(name, vs, vsEntry, fs, fsEntry, params, count, err, errLen);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static int passBegin(long colors, int count, int clearMask, long clearColors, long depth, int depthClear, float depthValue,
                         int x, int y, int w, int h) {
        try {
            return (int) PASS_BEGIN.invokeExact(colors, count, clearMask, clearColors, depth, depthClear, depthValue, x, y, w, h);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void passEnd() {
        try { PASS_END.invokeExact(); } catch (Throwable t) { throw rethrow(t); }
    }

    static void rpScissor(int x, int y, int w, int h) {
        try { RP_SCISSOR.invokeExact(x, y, w, h); } catch (Throwable t) { throw rethrow(t); }
    }

    static int rpSetPipeline(long handle) {
        try { return (int) RP_SET_PIPELINE.invokeExact(handle); } catch (Throwable t) { throw rethrow(t); }
    }

    static void rpBindBuffer(int index, long handle, long offset) {
        try { RP_BIND_BUFFER.invokeExact(index, handle, offset); } catch (Throwable t) { throw rethrow(t); }
    }

    static void rpBindTexture(int index, long texture, long sampler) {
        try { RP_BIND_TEXTURE.invokeExact(index, texture, sampler); } catch (Throwable t) { throw rethrow(t); }
    }

    static void rpBindTexelBuffer(int index, long handle, long offset, long length, int format, int bpp) {
        try { RP_BIND_TEXEL.invokeExact(index, handle, offset, length, format, bpp); } catch (Throwable t) { throw rethrow(t); }
    }

    static void rpPushConstants(long ptr, int len) {
        try { RP_PUSH_CONSTANTS.invokeExact(ptr, len); } catch (Throwable t) { throw rethrow(t); }
    }

    static void rpSetVertexBuffer(int slot, long handle, long offset) {
        try { RP_SET_VERTEX_BUFFER.invokeExact(slot, handle, offset); } catch (Throwable t) { throw rethrow(t); }
    }

    static void rpSetIndexBuffer(long handle, int type) {
        try { RP_SET_INDEX_BUFFER.invokeExact(handle, type); } catch (Throwable t) { throw rethrow(t); }
    }

    static void rpDrawIndexed(int indexCount, int instanceCount, int firstIndex, int baseVertex, int baseInstance) {
        try {
            RP_DRAW_INDEXED.invokeExact(indexCount, instanceCount, firstIndex, baseVertex, baseInstance);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void rpMultiDrawIndexed(long params, int instanceCount, int firstInstance, int drawCount) {
        try { RP_MULTI_DRAW_INDEXED.invokeExact(params, instanceCount, firstInstance, drawCount); } catch (Throwable t) { throw rethrow(t); }
    }

    static void rpDrawIndexedIndirect(long handle, long offset, int drawCount) {
        try { RP_DRAW_INDEXED_INDIRECT.invokeExact(handle, offset, drawCount); } catch (Throwable t) { throw rethrow(t); }
    }

    static void rpDraw(int vertexCount, int instanceCount, int firstVertex, int firstInstance) {
        try { RP_DRAW.invokeExact(vertexCount, instanceCount, firstVertex, firstInstance); } catch (Throwable t) { throw rethrow(t); }
    }

    static void rpMultiDraw(long params, int instanceCount, int firstInstance, int drawCount) {
        try { RP_MULTI_DRAW.invokeExact(params, instanceCount, firstInstance, drawCount); } catch (Throwable t) { throw rethrow(t); }
    }

    static void rpDrawIndirect(long handle, long offset, int drawCount) {
        try { RP_DRAW_INDIRECT.invokeExact(handle, offset, drawCount); } catch (Throwable t) { throw rethrow(t); }
    }

    static void copyBuffer(long src, long srcOffset, long dst, long dstOffset, long size) {
        try { COPY_BUFFER.invokeExact(src, srcOffset, dst, dstOffset, size); } catch (Throwable t) { throw rethrow(t); }
    }

    static void copyBufferToTexture(long src, long srcOffset, int bytesPerRow, long tex, int mip, int layer, int x, int y, int w, int h) {
        try {
            COPY_BUFFER_TO_TEXTURE.invokeExact(src, srcOffset, bytesPerRow, tex, mip, layer, x, y, w, h);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void copyTextureToBuffer(long tex, int mip, int x, int y, int w, int h, long dst, long dstOffset, int bytesPerRow) {
        try {
            COPY_TEXTURE_TO_BUFFER.invokeExact(tex, mip, x, y, w, h, dst, dstOffset, bytesPerRow);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void copyTextureToTexture(long src, long dst, int mip, int dx, int dy, int sx, int sy, int w, int h) {
        try {
            COPY_TEXTURE_TO_TEXTURE.invokeExact(src, dst, mip, dx, dy, sx, sy, w, h);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void clearTextures(long color, long rgba, long depth, float depthValue) {
        try { CLEAR_TEXTURES.invokeExact(color, rgba, depth, depthValue); } catch (Throwable t) { throw rethrow(t); }
    }

    static void clearRegion(long color, long rgba, long depth, float depthValue, int x, int y, int w, int h, int mip) {
        try {
            CLEAR_REGION.invokeExact(color, rgba, depth, depthValue, x, y, w, h, mip);
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }

    static void submit(long index) {
        try { SUBMIT.invokeExact(index); } catch (Throwable t) { throw rethrow(t); }
    }

    static boolean waitSubmit(long index, long timeoutNs) {
        try { return (int) WAIT_SUBMIT.invokeExact(index, timeoutNs) == 1; } catch (Throwable t) { throw rethrow(t); }
    }

    static int gpuTimesTake(long out, int max) {
        try { return (int) GPU_TIMES_TAKE.invokeExact(out, max); } catch (Throwable t) { throw rethrow(t); }
    }

    static void statsTake(long out) {
        try { STATS_TAKE.invokeExact(out); } catch (Throwable t) { throw rethrow(t); }
    }

    static void traceFrames(int n) {
        try { TRACE_FRAMES.invokeExact(n); } catch (Throwable t) { throw rethrow(t); }
    }

    static int lodOpen(long worldDir, int far, int centerX, int centerZ) {
        try { return (int) LOD_OPEN.invokeExact(worldDir, far, centerX, centerZ); } catch (Throwable t) { throw rethrow(t); }
    }

    static void lodCenter(int x, int z, int vanillaRadius) {
        try { LOD_CENTER.invokeExact(x, z, vanillaRadius); } catch (Throwable t) { throw rethrow(t); }
    }

    static void lodStatus(long out) {
        try { LOD_STATUS.invokeExact(out); } catch (Throwable t) { throw rethrow(t); }
    }

    static int lodDraw(long params, long cam) {
        try { return (int) LOD_DRAW.invokeExact(params, cam); } catch (Throwable t) { throw rethrow(t); }
    }

    static long completedSubmit() {
        try { return (long) COMPLETED_SUBMIT.invokeExact(); } catch (Throwable t) { throw rethrow(t); }
    }

    static long surfaceCreate(long layer) {
        try { return (long) SURFACE_CREATE.invokeExact(layer); } catch (Throwable t) { throw rethrow(t); }
    }

    static void surfaceConfigure(long handle, int w, int h, int displaySync) {
        try { SURFACE_CONFIGURE.invokeExact(handle, w, h, displaySync); } catch (Throwable t) { throw rethrow(t); }
    }

    static boolean surfaceAcquire(long handle) {
        try { return (int) SURFACE_ACQUIRE.invokeExact(handle) == 1; } catch (Throwable t) { throw rethrow(t); }
    }

    static void surfaceBlit(long handle, long srcTexture) {
        try { SURFACE_BLIT.invokeExact(handle, srcTexture); } catch (Throwable t) { throw rethrow(t); }
    }

    static void surfacePresent(long handle) {
        try { SURFACE_PRESENT.invokeExact(handle); } catch (Throwable t) { throw rethrow(t); }
    }
}
