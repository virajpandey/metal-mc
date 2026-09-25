package metalmc.backend;

import com.mojang.renderpearl.api.textures.AddressMode;
import com.mojang.renderpearl.api.textures.FilterMode;
import com.mojang.renderpearl.api.textures.GpuSampler;
import java.util.OptionalDouble;

final class MetalSampler implements GpuSampler {
    private final MetalDevice device;
    final long handle;
    private final AddressMode addressModeU;
    private final AddressMode addressModeV;
    private final FilterMode minFilter;
    private final FilterMode magFilter;
    private final int maxAnisotropy;
    private final OptionalDouble maxLod;
    private boolean closed;

    MetalSampler(MetalDevice device, AddressMode u, AddressMode v, FilterMode min, FilterMode mag, int maxAnisotropy, OptionalDouble maxLod) {
        this.device = device;
        this.addressModeU = u;
        this.addressModeV = v;
        this.minFilter = min;
        this.magFilter = mag;
        this.maxAnisotropy = maxAnisotropy;
        this.maxLod = maxLod;
        this.handle = Mtl.samplerCreate(u == AddressMode.REPEAT ? 0 : 1, v == AddressMode.REPEAT ? 0 : 1,
            min == FilterMode.NEAREST ? 0 : 1, mag == FilterMode.NEAREST ? 0 : 1, maxAnisotropy,
            maxLod.isPresent() ? (float) maxLod.getAsDouble() : -1f);
        if (handle == 0) throw new IllegalStateException("Metal sampler creation failed");
    }

    @Override
    public AddressMode getAddressModeU() {
        return addressModeU;
    }

    @Override
    public AddressMode getAddressModeV() {
        return addressModeV;
    }

    @Override
    public FilterMode getMinFilter() {
        return minFilter;
    }

    @Override
    public FilterMode getMagFilter() {
        return magFilter;
    }

    @Override
    public int getMaxAnisotropy() {
        return maxAnisotropy;
    }

    @Override
    public OptionalDouble getMaxLod() {
        return maxLod;
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
