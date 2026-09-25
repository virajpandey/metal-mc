package metalmc.backend;

import com.mojang.renderpearl.api.buffers.GpuBuffer;
import com.mojang.renderpearl.api.buffers.GpuBufferSlice;
import com.mojang.renderpearl.api.buffers.TransientMemory;
import com.mojang.renderpearl.api.commands.GpuFence;
import com.mojang.renderpearl.api.commands.GpuQueryPool;
import com.mojang.renderpearl.api.commands.RenderPass;
import com.mojang.renderpearl.api.commands.RenderPassDescriptor;
import com.mojang.renderpearl.api.textures.GpuTexture;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import com.mojang.renderpearl.backend.api.CommandEncoderBackend;
import com.mojang.renderpearl.backend.api.RenderPassBackend;
import java.nio.ByteBuffer;
import java.nio.FloatBuffer;
import java.nio.LongBuffer;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.OptionalDouble;
import org.joml.Vector4fc;
import org.jspecify.annotations.Nullable;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

/**
 * The one command stream. Commands record into a single MTLCommandBuffer per submit (render passes,
 * blits, clears in order; Metal's hazard tracking orders them). Like the Vulkan backend, up to two
 * submits are in flight, and destruction is deferred two submits.
 */
final class MetalCommandEncoder implements CommandEncoderBackend {
    static final int MAX_SUBMITS_IN_FLIGHT = 2;

    private final MetalDevice device;
    private final MetalTransientMemory transientMemory;
    private long currentSubmitIndex = 2;
    private long completedSubmitIndex = 1;
    @SuppressWarnings("unchecked")
    private final List<Runnable>[] destroyQueues = new List[]{new ArrayList<Runnable>(), new ArrayList<Runnable>()};
    private int destroyQueueIndex;
    private @Nullable MetalRenderPass currentRenderPass;

    MetalCommandEncoder(MetalDevice device) {
        this.device = device;
        this.transientMemory = new MetalTransientMemory(device, this);
    }

    @Nullable MetalRenderPass currentRenderPass() {
        return currentRenderPass;
    }

    void queueForDestroy(Runnable r) {
        destroyQueues[destroyQueueIndex].add(r);
    }

    private boolean rotateDestroyQueue() {
        destroyQueueIndex = (destroyQueueIndex + 1) % destroyQueues.length;
        List<Runnable> queue = destroyQueues[destroyQueueIndex];
        if (queue.isEmpty()) return false;
        destroyQueues[destroyQueueIndex] = new ArrayList<>();
        queue.forEach(Runnable::run);
        return true;
    }

    void destroy() {
        transientMemory.endSubmit();
        Mtl.submit(currentSubmitIndex);
        Mtl.waitSubmit(currentSubmitIndex, -1);
        for (int i = 0; i < destroyQueues.length; i++) {
            if (rotateDestroyQueue()) i = 0;
        }
        transientMemory.destroy();
        for (int i = 0; i < destroyQueues.length; i++) {
            if (rotateDestroyQueue()) i = 0;
        }
    }

    @Override
    public void submit() {
        if (currentRenderPass != null) throw new IllegalStateException("Cannot submit while inside a RenderPass");
        transientMemory.endSubmit();
        Mtl.submit(currentSubmitIndex);
        currentSubmitIndex++;
        if (!awaitSubmitCompletion(currentSubmitIndex - MAX_SUBMITS_IN_FLIGHT, 5_000_000_000L)) {
            throw new IllegalStateException("5s timeout reached waiting for Metal submit " + (currentSubmitIndex - MAX_SUBMITS_IN_FLIGHT));
        }
        rotateDestroyQueue();
    }

    @Override
    public TransientMemory transientMemory() {
        return transientMemory;
    }

    @Override
    public RenderPassBackend createRenderPass(RenderPassDescriptor descriptor) {
        if (currentRenderPass != null) throw new IllegalStateException("Close the existing render pass before creating a new one");
        List<RenderPassDescriptor.Attachment<Optional<Vector4fc>>> colors = descriptor.colorAttachments();
        int n = colors.size();
        RenderPass.RenderArea area = descriptor.renderArea();
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer handles = stack.callocLong(Math.max(n, 1));
            FloatBuffer clears = stack.callocFloat(Math.max(4 * n, 4));
            int clearMask = 0;
            for (int i = 0; i < n; i++) {
                RenderPassDescriptor.Attachment<Optional<Vector4fc>> att = colors.get(i);
                if (att == null) continue;
                handles.put(i, ((MetalTextureView) att.textureView()).handle);
                if (att.clearValue().isPresent()) {
                    Vector4fc c = att.clearValue().get();
                    clearMask |= 1 << i;
                    clears.put(4 * i, c.x()).put(4 * i + 1, c.y()).put(4 * i + 2, c.z()).put(4 * i + 3, c.w());
                }
            }
            RenderPassDescriptor.Attachment<OptionalDouble> depth = descriptor.depthAttachment();
            long depthHandle = depth != null ? ((MetalTextureView) depth.textureView()).handle : 0L;
            boolean depthClear = depth != null && depth.clearValue().isPresent();
            float depthValue = depthClear ? (float) depth.clearValue().getAsDouble() : 0f;
            int ok = Mtl.passBegin(MemoryUtil.memAddress(handles), n, clearMask, MemoryUtil.memAddress(clears), depthHandle,
                depthClear ? 1 : 0, depthValue, area.x(), area.y(), area.width(), area.height());
            if (ok != 1) throw new IllegalStateException("Metal render pass creation failed: " + descriptor.label().get());
        }
        currentRenderPass = new MetalRenderPass(area);
        return currentRenderPass;
    }

    @Override
    public void submitRenderPass() {
        if (currentRenderPass == null) throw new IllegalStateException("Cannot submit a renderpass if one hasn't been started!");
        Mtl.passEnd();
        currentRenderPass = null;
    }

    private static long rgba(MemoryStack stack, Vector4fc c) {
        FloatBuffer f = stack.mallocFloat(4);
        f.put(0, c.x()).put(1, c.y()).put(2, c.z()).put(3, c.w());
        return MemoryUtil.memAddress(f);
    }

    @Override
    public void clearColorTexture(GpuTexture colorTexture, Vector4fc clearColor) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            Mtl.clearTextures(((MetalTexture) colorTexture).handle, rgba(stack, clearColor), 0L, 0f);
        }
    }

    @Override
    public void clearColorAndDepthTextures(GpuTexture colorTexture, Vector4fc clearColor, GpuTexture depthTexture, double clearDepth) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            Mtl.clearTextures(((MetalTexture) colorTexture).handle, rgba(stack, clearColor), ((MetalTexture) depthTexture).handle, (float) clearDepth);
        }
    }

    @Override
    public void clearColorAndDepthTextures(GpuTexture colorTexture, Vector4fc clearColor, GpuTexture depthTexture, double clearDepth,
                                           int regionX, int regionY, int regionWidth, int regionHeight, int mipLevel) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            Mtl.clearRegion(((MetalTexture) colorTexture).handle, rgba(stack, clearColor), ((MetalTexture) depthTexture).handle, (float) clearDepth,
                regionX, regionY, regionWidth, regionHeight, mipLevel);
        }
    }

    @Override
    public void clearDepthTexture(GpuTexture depthTexture, double clearDepth) {
        try (MemoryStack stack = MemoryStack.stackPush()) {
            Mtl.clearTextures(0L, stack.nfloat(0f), ((MetalTexture) depthTexture).handle, (float) clearDepth);
        }
    }

    @Override
    public void writeToBuffer(GpuBufferSlice destination, ByteBuffer data) {
        int size = data.remaining();
        if (size == 0) return;
        GpuBufferSlice staging = transientMemory.uploadStaging(data, 4L, GpuBuffer.USAGE_COPY_SRC);
        Mtl.copyBuffer(((MetalBuffer) staging.buffer()).handle, staging.offset(), ((MetalBuffer) destination.buffer()).handle, destination.offset(), size);
    }

    @Override
    public void copyToBuffer(GpuBufferSlice source, GpuBufferSlice target) {
        Mtl.copyBuffer(((MetalBuffer) source.buffer()).handle, source.offset(), ((MetalBuffer) target.buffer()).handle, target.offset(), source.length());
    }

    @Override
    public void writeToTexture(GpuTexture destination, ByteBuffer source, int mipLevel, int depthOrLayer, int destX, int destY, int width, int height) {
        GpuBufferSlice staging = transientMemory.uploadStaging(source, 16L, GpuBuffer.USAGE_COPY_SRC);
        int bpp = destination.getFormat().blockSize();
        Mtl.copyBufferToTexture(((MetalBuffer) staging.buffer()).handle, staging.offset(), width * bpp, ((MetalTexture) destination).handle,
            mipLevel, depthOrLayer, destX, destY, width, height);
    }

    @Override
    public void copyBufferToTexture(GpuBufferSlice source, int sourceX, int sourceY, int sourceWidth, int sourceHeight, GpuTexture destination,
                                    int destinationX, int destinationY, int copyWidth, int copyHeight, int mipLevel, int arrayLayer) {
        int bpp = destination.getFormat().blockSize();
        long skip = (sourceX + (long) sourceY * sourceWidth) * bpp;
        Mtl.copyBufferToTexture(((MetalBuffer) source.buffer()).handle, source.offset() + skip, sourceWidth * bpp, ((MetalTexture) destination).handle,
            mipLevel, arrayLayer, destinationX, destinationY, copyWidth, copyHeight);
    }

    @Override
    public void copyTextureToBuffer(GpuTexture source, GpuBuffer destination, long offset, Runnable callback, int mipLevel) {
        copyTextureToBuffer(source, destination, offset, callback, mipLevel, 0, 0, source.getWidth(mipLevel), source.getHeight(mipLevel));
    }

    @Override
    public void copyTextureToBuffer(GpuTexture source, GpuBuffer destination, long offset, Runnable callback, int mipLevel, int x, int y, int width, int height) {
        int bpp = source.getFormat().blockSize();
        Mtl.copyTextureToBuffer(((MetalTexture) source).handle, mipLevel, x, y, width, height, ((MetalBuffer) destination).handle, offset, width * bpp);
        queueForDestroy(callback);
    }

    @Override
    public void copyTextureToTexture(GpuTexture source, GpuTexture destination, int mipLevel, int destX, int destY, int sourceX, int sourceY, int width, int height) {
        Mtl.copyTextureToTexture(((MetalTexture) source).handle, ((MetalTexture) destination).handle, mipLevel, destX, destY, sourceX, sourceY, width, height);
    }

    private boolean awaitSubmitCompletion(long submitIndex, long timeoutNs) {
        if (completedSubmitIndex >= submitIndex) return true;
        if (submitIndex >= currentSubmitIndex) {
            if (timeoutNs == 0) return false;
            throw new IllegalStateException("Cannot wait on a fence for the current submit");
        }
        boolean done = Mtl.waitSubmit(submitIndex, timeoutNs);
        if (done) completedSubmitIndex = Math.max(completedSubmitIndex, submitIndex);
        return done;
    }

    @Override
    public GpuFence createFence() {
        long submitIndex = currentSubmitIndex;
        return new GpuFence() {
            private boolean completed;

            @Override
            public boolean awaitCompletion(long timeoutNs) {
                if (!completed) completed = awaitSubmitCompletion(submitIndex, timeoutNs);
                return completed;
            }

            @Override
            public void close() {
                completed = true;
            }
        };
    }

    @Override
    public void writeTimestamp(GpuQueryPool pool, int index) {
    }

    /** Surface blits go into this command stream; see MetalSurface. */
    void blitToSurface(long surface, GpuTextureView view) {
        Mtl.surfaceBlit(surface, ((MetalTextureView) view).handle);
    }
}
