package metalmc.backend;

import com.mojang.renderpearl.api.device.BackendCreationException;
import com.mojang.renderpearl.api.device.GpuBackend;
import com.mojang.renderpearl.api.device.GpuDebugOptions;
import com.mojang.renderpearl.api.device.GpuDevice;
import org.jspecify.annotations.Nullable;
import org.lwjgl.sdl.SDLVideo;

/**
 * M3 spike, step 1a: a backend that Minecraft tries before OpenGL/Vulkan. It loads the native bridge
 * and creates a real Metal device, then declines (BackendCreationException), so the game falls back
 * to the next backend. Each later step replaces a piece of the "decline" with a real implementation.
 */
public final class MetalGpuBackend implements GpuBackend {
    /** SDL3 SDL_WINDOW_METAL. */
    static final long SDL_WINDOW_METAL = 0x20000000L;

    private @Nullable MetalNative nativeLib;

    @Override
    public String getName() {
        return "Metal (MetalMC)";
    }

    @Override
    public void loadLibrary() throws BackendCreationException {
        if (nativeLib != null) return;
        try {
            MetalNative lib = MetalNative.load();
            int abi = lib.abiVersion();
            if (abi != MetalNative.EXPECTED_ABI) {
                throw new BackendCreationException("MetalMC native ABI " + abi + " != expected " + MetalNative.EXPECTED_ABI,
                    BackendCreationException.Reason.OTHER);
            }
            nativeLib = lib;
        } catch (RuntimeException e) {
            throw new BackendCreationException("MetalMC native bridge unavailable: " + e.getMessage(),
                BackendCreationException.Reason.PLATFORM_ERROR);
        }
    }

    @Override
    public void unloadLibrary() {
        nativeLib = null;
    }

    @Override
    public long createWindow(@Nullable String title, int width, int height, long flags) {
        return SDLVideo.SDL_CreateWindow(title, width, height, SDL_WINDOW_METAL | flags);
    }

    @Override
    public GpuDevice createDevice(GpuDebugOptions debugOptions) throws BackendCreationException {
        if (nativeLib == null) throw new BackendCreationException("MetalMC library not loaded", BackendCreationException.Reason.OTHER);
        String name = nativeLib.deviceName();
        boolean apple9 = nativeLib.supportsApple9();
        System.out.println("[metalmc-backend] Metal device via native bridge: name=\"" + name + "\" apple9=" + apple9
            + " (step 1a: declining so the game falls back)");
        throw new BackendCreationException("MetalMC backend: device \"" + name + "\" OK, rendering not implemented yet (spike step 1a)",
            BackendCreationException.Reason.OTHER);
    }
}
