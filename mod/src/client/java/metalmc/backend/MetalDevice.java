package metalmc.backend;

import com.mojang.renderpearl.api.GpuFormat;
import com.mojang.renderpearl.api.buffers.GpuBuffer;
import com.mojang.renderpearl.api.commands.GpuQueryPool;
import com.mojang.renderpearl.api.device.DeviceFeatures;
import com.mojang.renderpearl.api.device.DeviceInfo;
import com.mojang.renderpearl.api.device.DeviceLimits;
import com.mojang.renderpearl.api.device.DeviceType;
import com.mojang.renderpearl.api.device.HintsAndWorkarounds;
import com.mojang.renderpearl.api.textures.AddressMode;
import com.mojang.renderpearl.api.textures.FilterMode;
import com.mojang.renderpearl.api.textures.GpuSampler;
import com.mojang.renderpearl.api.textures.GpuTexture;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import com.mojang.renderpearl.backend.api.BackendRenderPipeline;
import com.mojang.renderpearl.backend.api.GpuDeviceBackend;
import com.mojang.renderpearl.backend.api.GpuSurfaceBackend;
import java.nio.ByteBuffer;
import java.nio.LongBuffer;
import java.util.List;
import java.util.OptionalDouble;
import java.util.Set;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;
import org.jspecify.annotations.Nullable;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

/** Minecraft's GpuDeviceBackend on Metal. */
public final class MetalDevice implements GpuDeviceBackend {
    /** The live device, for features outside the backend interface (LOD). Null when Metal isn't active. */
    static volatile MetalDevice current;

    private final DeviceInfo info;
    private final MetalCommandEncoder encoder;
    private final boolean debug;

    MetalDevice(String deviceName, boolean debug) {
        this.debug = debug;
        long maxBuffer;
        boolean apple9;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            LongBuffer limits = stack.callocLong(4);
            Mtl.limits(MemoryUtil.memAddress(limits));
            maxBuffer = limits.get(0);
            apple9 = limits.get(2) == 1;
        }
        this.info = new DeviceInfo(
            deviceName,
            "Apple",
            "MetalMC (Metal 3, MSL 3.0), macOS " + System.getProperty("os.version"),
            true,
            "Metal",
            1.0f,
            new DeviceLimits(16, 16, 16384, maxBuffer, 1 << 20, 8, Integer.MAX_VALUE),
            // wireframe, draw parameters, interleaved multi-draw, (no separate), multi-draw indirect,
            // draw indirect, non-zero first instance, persistent mapping
            new DeviceFeatures(true, true, true, false, true, true, true, true),
            Set.of("MTLGPUFamilyMetal3", apple9 ? "MTLGPUFamilyApple9" : "MTLGPUFamilyApple8"),
            // Same hints the Vulkan backend uses on Apple Silicon (explicit depth for OIT invariance).
            new HintsAndWorkarounds(false, false, true, false),
            DeviceType.INTEGRATED);
        this.encoder = new MetalCommandEncoder(this);
        current = this;
    }

    MetalCommandEncoder encoder() {
        return encoder;
    }

    @Override
    public GpuSurfaceBackend createSurface(long windowHandle, BooleanSupplier isIconified) {
        return new MetalSurface(windowHandle);
    }

    @Override
    public MetalCommandEncoder createCommandEncoder() {
        return encoder;
    }

    @Override
    public GpuSampler createSampler(AddressMode addressModeU, AddressMode addressModeV, FilterMode minFilter, FilterMode magFilter,
                                    int maxAnisotropy, OptionalDouble maxLod) {
        return new MetalSampler(this, addressModeU, addressModeV, minFilter, magFilter, maxAnisotropy, maxLod);
    }

    @Override
    public GpuTexture createTexture(@Nullable String label, @GpuTexture.Usage int usage, GpuFormat format, int width, int height, int depthOrLayers, int mipLevels) {
        MetalTexture t = new MetalTexture(this, usage, label != null ? label : "", format, width, height, depthOrLayers, mipLevels);
        if (debug && label != null) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                Mtl.textureLabel(t.handle, MemoryUtil.memAddress(stack.UTF8(label)));
            }
        }
        return t;
    }

    @Override
    public GpuTextureView createTextureView(GpuTexture texture, int baseMipLevel, int mipLevels) {
        return new MetalTextureView(this, (MetalTexture) texture, baseMipLevel, mipLevels);
    }

    @Override
    public GpuBuffer createBuffer(@Nullable Supplier<String> label, @GpuBuffer.Usage int usage, long size) {
        MetalBuffer b = MetalBuffer.create(this, usage, size);
        if (debug && label != null) {
            try (MemoryStack stack = MemoryStack.stackPush()) {
                Mtl.bufferLabel(b.handle, MemoryUtil.memAddress(stack.UTF8(label.get())));
            }
        }
        return b;
    }

    @Override
    public GpuBuffer createBuffer(@Nullable Supplier<String> label, @GpuBuffer.Usage int usage, ByteBuffer data) {
        MetalBuffer b = (MetalBuffer) createBuffer(label, usage | GpuBuffer.USAGE_COPY_DST, data.remaining());
        // Shared memory: a new buffer can't be in use by the GPU yet, so write it directly.
        MetalBuffer.copyInto(b.contents, data);
        return b;
    }

    @Override
    public List<String> getLastDebugMessages() {
        return List.of();
    }

    @Override
    public boolean isDebuggingEnabled() {
        return debug;
    }

    @Override
    public BackendRenderPipeline.Pending compilePipeline(BackendRenderPipeline.CreateInfo pipelineCreateInfo) {
        return MetalRenderPipeline.compile(this, pipelineCreateInfo);
    }

    @Override
    public void close() {
        encoder.destroy();
    }

    @Override
    public GpuQueryPool createTimestampQueryPool(int size) {
        return new MetalQueryPool(size);
    }

    @Override
    public long getTimestampCalibrationOffset() {
        return 0L;
    }

    @Override
    public DeviceInfo getDeviceInfo() {
        return info;
    }
}
