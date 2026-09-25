package metalmc.backend;

import com.mojang.renderpearl.backend.common.BaseGpuTextureView;

final class MetalTextureView extends BaseGpuTextureView {
    private final MetalDevice device;
    final long handle;
    private boolean closed;

    MetalTextureView(MetalDevice device, MetalTexture texture, int baseMipLevel, int mipLevels) {
        super(texture, baseMipLevel, mipLevels);
        this.device = device;
        this.handle = Mtl.textureView(texture.handle, baseMipLevel, mipLevels);
        if (handle == 0) throw new IllegalStateException("Metal texture view creation failed: " + texture.getLabel());
    }

    @Override
    public boolean isClosed() {
        return closed;
    }

    @Override
    public void close() {
        if (!closed) {
            closed = true;
            device.encoder().queueForDestroy(() -> Mtl.release(handle));
        }
    }
}
