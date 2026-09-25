package metalmc.backend;

import com.mojang.renderpearl.api.GpuFormat;
import com.mojang.renderpearl.api.textures.GpuTexture;
import com.mojang.renderpearl.backend.common.BaseGpuTexture;

/** A private-storage MTLTexture (uploads go through blits, which keeps lossless compression available). */
final class MetalTexture extends BaseGpuTexture {
    private final MetalDevice device;
    final long handle;
    private boolean closed;

    MetalTexture(MetalDevice device, @GpuTexture.Usage int usage, String label, GpuFormat format, int width, int height, int depthOrLayers, int mipLevels) {
        super(usage, label, format, width, height, depthOrLayers, mipLevels);
        this.device = device;
        int pixelFormat = MetalConst.pixelFormat(format);
        if (pixelFormat == 0) throw new IllegalArgumentException("No Metal texture format for " + format);
        // Every texture can be a render target: clears are render passes, and Apple GPUs pay nothing for it.
        int flags = 1 | ((usage & GpuTexture.USAGE_CUBEMAP_COMPATIBLE) != 0 ? 2 : 0);
        this.handle = Mtl.textureCreate(pixelFormat, width, height, depthOrLayers, mipLevels, flags);
        if (handle == 0) throw new IllegalStateException("Metal texture creation failed: " + label + " " + format + " " + width + "x" + height);
    }

    @Override
    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            // Views retain the MTLTexture on the native side, so this can go independently of them.
            device.encoder().queueForDestroy(() -> Mtl.release(handle));
        }
    }
}
