package metalmc.backend;

import com.mojang.renderpearl.api.buffers.GpuBuffer;
import com.mojang.renderpearl.api.buffers.GpuBufferSlice;
import com.mojang.renderpearl.api.commands.GpuQueryPool;
import com.mojang.renderpearl.api.commands.RenderPass;
import com.mojang.renderpearl.api.pipeline.IndexType;
import com.mojang.renderpearl.backend.api.BackendRenderPipeline;
import com.mojang.renderpearl.backend.api.RenderPassBackend;
import com.mojang.renderpearl.util.TextureViewAndSampler;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.util.function.Supplier;
import org.jspecify.annotations.Nullable;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

/**
 * Records into the native MTLRenderCommandEncoder. Uniforms are bound lazily at the next draw, and
 * only the ones that changed, to the stages whose shaders read them.
 */
final class MetalRenderPass implements RenderPassBackend {
    private final RenderPass.RenderArea area;
    private @Nullable MetalRenderPipeline pipeline;
    private final Object[] uniforms = new Object[32];
    private int dirty;

    MetalRenderPass(RenderPass.RenderArea area) {
        this.area = area;
    }

    /** Native code drew into this pass with its own pipeline: re-apply Minecraft's pipeline state. */
    void restoreAfterExternalDraw() {
        if (pipeline != null) Mtl.rpSetPipeline(pipeline.handle);
    }

    @Override
    public void pushDebugGroup(Supplier<String> label) {
    }

    @Override
    public void popDebugGroup() {
    }

    @Override
    public void setPipeline(BackendRenderPipeline pipeline) {
        if (!(pipeline instanceof MetalRenderPipeline p)) throw new IllegalArgumentException("Pipeline must be instance of MetalRenderPipeline");
        this.pipeline = Mtl.rpSetPipeline(p.handle) == 1 ? p : null;
        java.util.Arrays.fill(uniforms, null);
        dirty = 0;
    }

    @Override
    public void setUniform(int index, @Nullable Object value) {
        uniforms[index] = value;
        dirty |= 1 << index;
    }

    @Override
    public void pushConstants(ByteBuffer value) {
        if (value.isDirect()) {
            Mtl.rpPushConstants(MemoryUtil.memAddress(value), value.remaining());
            return;
        }
        try (MemoryStack stack = MemoryStack.stackPush()) {
            ByteBuffer copy = stack.malloc(value.remaining());
            copy.put(value.duplicate()).flip();
            Mtl.rpPushConstants(MemoryUtil.memAddress(copy), copy.remaining());
        }
    }

    @Override
    public void enableScissor(int x, int y, int width, int height) {
        Mtl.rpScissor(x, y, width, height);
    }

    @Override
    public void disableScissor() {
        Mtl.rpScissor(area.x(), area.y(), area.width(), area.height());
    }

    @Override
    public void setVertexBuffer(int slot, @Nullable GpuBufferSlice vertexBuffer) {
        if (vertexBuffer != null) Mtl.rpSetVertexBuffer(slot, ((MetalBuffer) vertexBuffer.buffer()).handle, vertexBuffer.offset());
    }

    @Override
    public void setIndexBuffer(GpuBuffer indexBuffer, IndexType indexType) {
        Mtl.rpSetIndexBuffer(((MetalBuffer) indexBuffer).handle, indexType == IndexType.INT ? 1 : 0);
    }

    /** Binds changed uniforms. Returns false if the pipeline failed to compile (the draw is skipped natively too). */
    private boolean flush() {
        MetalRenderPipeline p = pipeline;
        if (p == null) return false;
        int bits = dirty;
        if (bits == 0) return true;
        dirty = 0;
        int n = p.uniformKinds.length;
        for (int i = 0; i < n; i++) {
            if ((bits & (1 << i)) == 0) continue;
            Object value = uniforms[i];
            if (value == null) {
                throw new IllegalStateException("Missing uniform " + p.uniforms.get(i).name() + " (should be " + p.uniforms.get(i).type() + ")");
            }
            switch (p.uniformKinds[i]) {
                case 0 -> {
                    GpuBufferSlice s = (GpuBufferSlice) value;
                    Mtl.rpBindBuffer(i, ((MetalBuffer) s.buffer()).handle, s.offset());
                }
                case 1 -> {
                    TextureViewAndSampler t = (TextureViewAndSampler) value;
                    Mtl.rpBindTexture(i, ((MetalTextureView) t.view()).handle, ((MetalSampler) t.sampler()).handle);
                }
                default -> {
                    GpuBufferSlice s = (GpuBufferSlice) value;
                    Mtl.rpBindTexelBuffer(i, ((MetalBuffer) s.buffer()).handle, s.offset(), s.length(), p.texelFormats[i], p.texelBytes[i]);
                }
            }
        }
        return true;
    }

    @Override
    public void drawIndexed(int indexCount, int instanceCount, int firstIndex, int vertexOffset, int firstInstance) {
        if (flush()) Mtl.rpDrawIndexed(indexCount, instanceCount, firstIndex, vertexOffset, firstInstance);
    }

    @Override
    public void multiDrawIndexed(IntBuffer drawParameters, int instanceCount, int firstInstance, int drawCount) {
        if (flush()) Mtl.rpMultiDrawIndexed(MemoryUtil.memAddress(drawParameters), instanceCount, firstInstance, drawCount);
    }

    @Override
    public void multiDrawIndexed(PointerBuffer firstIndexOffsets, IntBuffer indexCounts, IntBuffer vertexOffsets, int drawCount) {
        throw new UnsupportedOperationException("Metal backend does not support the multiDrawDirectSeparate device feature");
    }

    @Override
    public void drawIndexedIndirect(GpuBufferSlice commands, int drawCount) {
        if (flush()) Mtl.rpDrawIndexedIndirect(((MetalBuffer) commands.buffer()).handle, commands.offset(), drawCount);
    }

    @Override
    public void draw(int vertexCount, int instanceCount, int firstVertex, int firstInstance) {
        if (flush()) Mtl.rpDraw(vertexCount, instanceCount, firstVertex, firstInstance);
    }

    @Override
    public void multiDraw(IntBuffer drawParameters, int instanceCount, int firstInstance, int drawCount) {
        if (flush()) Mtl.rpMultiDraw(MemoryUtil.memAddress(drawParameters), instanceCount, firstInstance, drawCount);
    }

    @Override
    public void multiDraw(IntBuffer firstVertices, IntBuffer vertexCounts, int drawCount) {
        throw new UnsupportedOperationException("Metal backend does not support the multiDrawDirectSeparate device feature");
    }

    @Override
    public void drawIndirect(GpuBufferSlice commands, int drawCount) {
        if (flush()) Mtl.rpDrawIndirect(((MetalBuffer) commands.buffer()).handle, commands.offset(), drawCount);
    }

    @Override
    public void writeTimestamp(GpuQueryPool pool, int index) {
    }
}
