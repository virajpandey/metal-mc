package metalmc.backend;

import com.mojang.renderpearl.api.buffers.GpuBuffer;
import com.mojang.renderpearl.api.buffers.GpuBufferSlice;
import com.mojang.renderpearl.api.buffers.TransientMemory;
import com.mojang.renderpearl.backend.util.TransientBlockAllocator;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import it.unimi.dsi.fastutil.ints.IntComparator;
import it.unimi.dsi.fastutil.objects.ReferenceArrayList;
import java.nio.ByteBuffer;
import java.util.List;
import java.util.stream.IntStream;
import org.lwjgl.system.MemoryUtil;

/**
 * Per-submit scratch memory. Mirrors the Vulkan backend's allocator, minus the copies: on unified
 * memory a "GPU" block is CPU-writable, so uploadGpu is a memcpy into a shared MTLBuffer and needs no
 * transfer command. Blocks return to the pool two submits later, after the GPU is done with them.
 */
final class MetalTransientMemory implements TransientMemory {
    private static final long BLOCK_SIZE = 524288L;
    private static final long MAX_GPU_ALIGNMENT = Long.highestOneBit(Long.MAX_VALUE);

    private final MetalDevice device;
    private final MetalCommandEncoder encoder;
    private final TransientBlockAllocator<TransientBlockAllocator.Allocator.CpuBlock> cpu =
        new TransientBlockAllocator<>(BLOCK_SIZE, 16L, TransientBlockAllocator.Allocator.CpuBlock.memalloc());
    private final TransientBlockAllocator<Block> staging;
    private final TransientBlockAllocator<Block> gpu;
    private long submitIndex;

    record Block(long handle, long contents, long size) implements TransientBlockAllocator.Allocator.Block {
        @Override
        public boolean suboptimal() {
            return false;
        }
    }

    MetalTransientMemory(MetalDevice device, MetalCommandEncoder encoder) {
        this.device = device;
        this.encoder = encoder;
        TransientBlockAllocator.Allocator<Block> blocks = TransientBlockAllocator.Allocator.create(this::allocateBlock, b -> Mtl.release(b.handle()));
        this.staging = new TransientBlockAllocator<>(BLOCK_SIZE, MAX_GPU_ALIGNMENT, blocks);
        this.gpu = new TransientBlockAllocator<>(BLOCK_SIZE, MAX_GPU_ALIGNMENT, blocks);
    }

    private Block allocateBlock(long size) {
        long handle = Mtl.bufferCreate(size);
        if (handle == 0) throw new IllegalStateException("Metal transient block allocation failed (" + size + " bytes)");
        return new Block(handle, Mtl.bufferContents(handle), size);
    }

    void endSubmit() {
        cpu.rotate().run();
        encoder.queueForDestroy(staging.rotate());
        encoder.queueForDestroy(gpu.rotate());
        submitIndex++;
    }

    void destroy() {
        cpu.close();
        staging.close();
        gpu.close();
    }

    private GpuBufferSlice slice(TransientBlockAllocator.Allocation<Block> alloc, @GpuBuffer.Usage int usage) {
        Block b = alloc.block();
        return new GpuBufferSlice(new TransientBuffer(b, usage, submitIndex), alloc.offset(), alloc.size());
    }

    @Override
    public ByteBuffer allocateCpu(long size, long alignment, long minimumAllocation, long elementSize) {
        TransientBlockAllocator.Allocation<TransientBlockAllocator.Allocator.CpuBlock> alloc = cpu.allocate(size, alignment, minimumAllocation, elementSize);
        return MemoryUtil.memByteBuffer(alloc.block().address() + alloc.offset(), (int) alloc.size());
    }

    @Override
    public GpuBufferSlice.MappedView allocateStaging(long size, long alignment, @GpuBuffer.Usage int usage, long minimumAllocation, long elementSize) {
        TransientBlockAllocator.Allocation<Block> alloc = staging.allocate(size, alignment, minimumAllocation, elementSize);
        ByteBuffer cpuView = MemoryUtil.memByteBuffer(alloc.block().contents() + alloc.offset(), (int) alloc.size());
        return new GpuBufferSlice.MappedView(slice(alloc, usage), cpuView, () -> {});
    }

    @Override
    public GpuBufferSlice allocateGpu(long size, long alignment, @GpuBuffer.Usage int usage, long minimumAllocation, long elementSize) {
        return slice(gpu.allocate(size, alignment, minimumAllocation, elementSize), usage);
    }

    @Override
    public GpuBufferSlice.MappedView allocateGpuMapped(long size, long alignment, @GpuBuffer.Usage int usage, long minimumAllocation, long elementSize) {
        TransientBlockAllocator.Allocation<Block> alloc = gpu.allocate(size, alignment, minimumAllocation, elementSize);
        ByteBuffer cpuView = MemoryUtil.memByteBuffer(alloc.block().contents() + alloc.offset(), (int) alloc.size());
        return new GpuBufferSlice.MappedView(slice(alloc, usage), cpuView, () -> {});
    }

    @Override
    public GpuBufferSlice uploadStaging(List<ByteBuffer> data, long alignment, @GpuBuffer.Usage int usage, long minimumAllocation, long elementSize) {
        return upload(data, alignment, usage, minimumAllocation, elementSize, true);
    }

    @Override
    public GpuBufferSlice uploadGpu(List<ByteBuffer> data, long alignment, @GpuBuffer.Usage int usage, long minimumAllocation, long elementSize) {
        return upload(data, alignment, usage, minimumAllocation, elementSize, false);
    }

    private static long roundUp(long value, long alignment) {
        return alignment <= 1 ? value : (value + alignment - 1) / alignment * alignment;
    }

    private GpuBufferSlice upload(List<ByteBuffer> data, long alignment, @GpuBuffer.Usage int usage, long minimumAllocation, long elementSize, boolean toStaging) {
        long totalSize = 0;
        for (ByteBuffer b : data) totalSize = roundUp(totalSize + b.remaining(), alignment);
        try (GpuBufferSlice.MappedView mapped = toStaging
            ? allocateStaging(totalSize, alignment, usage, minimumAllocation, elementSize)
            : allocateGpuMapped(totalSize, alignment, usage, minimumAllocation, elementSize)) {
            long base = MemoryUtil.memAddress(mapped.data());
            long offset = 0;
            for (ByteBuffer b : data) {
                long n = Math.min(mapped.slice().length() - offset, b.remaining());
                if (b.isDirect()) {
                    MemoryUtil.memCopy(MemoryUtil.memAddress(b), base + offset, n);
                } else {
                    MemoryUtil.memByteBuffer(base + offset, (int) n).put(b.duplicate().limit(b.position() + (int) n));
                }
                offset = roundUp(offset + b.remaining(), alignment);
                if (offset >= mapped.slice().length()) break;
            }
            return mapped.slice();
        }
    }

    @Override
    public List<GpuBufferSlice> multiUploadStaging(List<ByteBuffer> data, long alignment, @GpuBuffer.Usage int usage) {
        return multiUpload(data, alignment, usage, true);
    }

    @Override
    public List<GpuBufferSlice> multiUploadGpu(List<ByteBuffer> data, long alignment, @GpuBuffer.Usage int usage) {
        return multiUpload(data, alignment, usage, false);
    }

    /** Packs the largest buffers first into the current block (same strategy as the Vulkan backend). */
    private List<GpuBufferSlice> multiUpload(List<ByteBuffer> data, long alignment, @GpuBuffer.Usage int usage, boolean toStaging) {
        ReferenceArrayList<GpuBufferSlice> out = new ReferenceArrayList<>();
        out.size(data.size());
        TransientBlockAllocator<Block> allocator = toStaging ? staging : gpu;
        IntArrayList order = IntArrayList.toList(IntStream.range(0, data.size()));
        order.sort(IntComparator.comparing(i -> data.get(i).remaining()));
        while (!order.isEmpty()) {
            int pick = -1;
            for (int i = order.size() - 1; i >= 0; i--) {
                if (allocator.canAllocateInCurrentBlock(data.get(order.getInt(i)).remaining(), alignment)) {
                    pick = order.removeInt(i);
                    break;
                }
            }
            if (pick == -1) pick = order.popInt();
            ByteBuffer b = data.get(pick);
            try (GpuBufferSlice.MappedView view = toStaging
                ? allocateStaging(b.remaining(), alignment, usage)
                : allocateGpuMapped(b.remaining(), alignment, usage)) {
                MetalBuffer.copyInto(MemoryUtil.memAddress(view.data()), b);
                out.set(pick, view.slice());
            }
        }
        return out;
    }

    /** A view of a transient block for one submit. It reports closed once that submit ends. */
    private final class TransientBuffer extends MetalBuffer {
        private final long bufferSubmitIndex;
        private boolean closed;

        TransientBuffer(Block block, @GpuBuffer.Usage int usage, long bufferSubmitIndex) {
            super(MetalTransientMemory.this.device, usage, block.size(), block.handle(), block.contents());
            this.bufferSubmitIndex = bufferSubmitIndex;
        }

        @Override
        public boolean isClosed() {
            if (!closed) closed = bufferSubmitIndex < submitIndex;
            return closed;
        }

        @Override
        public void close() {
            closed = true;
        }

        @Override
        public GpuBufferSlice.MappedView map(long offset, long length, boolean read, boolean write) {
            throw new IllegalStateException("Cannot map transient buffer");
        }

        @Override
        public GpuBufferSlice slice(long offset, long length) {
            throw new IllegalStateException("Cannot slice transient buffer");
        }

        @Override
        public GpuBufferSlice slice() {
            throw new IllegalStateException("Cannot slice transient buffer");
        }
    }
}
