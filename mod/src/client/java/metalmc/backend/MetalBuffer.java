package metalmc.backend;

import com.mojang.renderpearl.api.buffers.GpuBuffer;
import com.mojang.renderpearl.api.buffers.GpuBufferSlice;
import com.mojang.renderpearl.backend.common.BaseGpuBuffer;
import java.nio.ByteBuffer;
import org.lwjgl.system.MemoryUtil;

/**
 * A shared-storage MTLBuffer. On Apple Silicon the CPU and GPU see the same memory, so mapping is
 * just a pointer into the buffer's contents and initial data is a memcpy (no staging copy).
 */
class MetalBuffer extends BaseGpuBuffer {
    protected final MetalDevice device;
    final long handle;
    final long contents;
    private boolean closed;
    private int mappings;

    MetalBuffer(MetalDevice device, @GpuBuffer.Usage int usage, long size, long handle, long contents) {
        super(usage, size);
        this.device = device;
        this.handle = handle;
        this.contents = contents;
    }

    static MetalBuffer create(MetalDevice device, @GpuBuffer.Usage int usage, long size) {
        long handle = Mtl.bufferCreate(size);
        if (handle == 0) throw new IllegalStateException("Metal buffer allocation failed (" + size + " bytes)");
        return new MetalBuffer(device, usage, size, handle, Mtl.bufferContents(handle));
    }

    @Override
    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            if (mappings != 0) throw new IllegalStateException("Attempt to close a mapped buffer");
            device.encoder().queueForDestroy(() -> Mtl.release(handle));
        }
    }

    @Override
    public GpuBufferSlice.MappedView map(long offset, long length, boolean read, boolean write) {
        if (closed) throw new IllegalStateException("Buffer already closed");
        if (!read && !write) throw new IllegalArgumentException("At least read or write must be true");
        if (read && (usage() & GpuBuffer.USAGE_MAP_READ) == 0) throw new IllegalStateException("Buffer is not readable");
        if (write && (usage() & GpuBuffer.USAGE_MAP_WRITE) == 0) throw new IllegalStateException("Buffer is not writable");
        if (offset < 0 || length < 0 || offset + length > size()) {
            throw new IllegalArgumentException("Cannot map " + length + " bytes at offset " + offset + " of a " + size() + " byte buffer");
        }
        if (length > Integer.MAX_VALUE) throw new IllegalArgumentException("Mapping buffer slice larger than 2GB is not supported");
        mappings++;
        ByteBuffer view = MemoryUtil.memByteBuffer(contents + offset, (int) length);
        return new GpuBufferSlice.MappedView(slice(offset, length), view, new Runnable() {
            private boolean done;

            @Override
            public void run() {
                if (!done) {
                    done = true;
                    mappings--;
                }
            }
        });
    }

    /** Copies `data` (position to limit) to `address`. */
    static void copyInto(long address, ByteBuffer data) {
        int n = data.remaining();
        if (data.isDirect()) {
            MemoryUtil.memCopy(MemoryUtil.memAddress(data), address, n);
        } else {
            MemoryUtil.memByteBuffer(address, n).put(data.duplicate());
        }
    }
}
