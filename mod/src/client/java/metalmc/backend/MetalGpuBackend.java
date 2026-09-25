package metalmc.backend;

import com.mojang.renderpearl.api.device.BackendCreationException;
import com.mojang.renderpearl.api.device.GpuBackend;
import com.mojang.renderpearl.api.device.GpuDebugOptions;
import com.mojang.renderpearl.api.device.GpuDevice;
import com.mojang.renderpearl.frontend.FrontendGpuDevice;
import org.jspecify.annotations.Nullable;
import org.lwjgl.sdl.SDLEvents;
import org.lwjgl.sdl.SDLMetal;
import org.lwjgl.sdl.SDLVideo;

/**
 * A backend that Minecraft tries before OpenGL/Vulkan (see PreferredGraphicsApiMixin). It loads the
 * native bridge and returns a Metal device; if anything fails it declines with a
 * BackendCreationException and the game falls back to the next backend.
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
        System.out.println("[metalmc-backend] Metal device via native bridge: name=\"" + name + "\" apple9=" + apple9);
        if ("1".equals(System.getProperty("metalmc.backend.presentTest"))) {
            presentRateTest(nativeLib, true);
            presentRateTest(nativeLib, false);
        }
        if ("0".equals(System.getProperty("metalmc.backend.render", "1"))) {
            throw new BackendCreationException("MetalMC backend: device \"" + name + "\" OK, rendering disabled (-Dmetalmc.backend.render=0)",
                BackendCreationException.Reason.OTHER);
        }
        try {
            if (Mtl.init() != 1) {
                throw new BackendCreationException("MetalMC: Metal enum values don't match the Java side", BackendCreationException.Reason.OTHER);
            }
            boolean debug = "1".equals(System.getProperty("metalmc.backend.debug"));
            return new FrontendGpuDevice(new MetalDevice(name, debug));
        } catch (RuntimeException | LinkageError e) {
            throw new BackendCreationException("MetalMC device creation failed: " + e, BackendCreationException.Reason.OTHER);
        }
    }

    /** SDL3 SDL_WINDOW_HIGH_PIXEL_DENSITY (Retina-resolution drawables). */
    static final long SDL_WINDOW_HIGH_PIXEL_DENSITY = 0x2000L;

    /**
     * Opens a 854x480-point window with a native CAMetalLayer, clears and presents frames as fast as
     * presentation allows, and logs the rate. Compares with MoltenVK's windowed 120 Hz pacing.
     */
    private static void presentRateTest(MetalNative lib, boolean displaySync) {
        long window = SDLVideo.SDL_CreateWindow("MetalMC present test", 854, 480, SDL_WINDOW_METAL | SDL_WINDOW_HIGH_PIXEL_DENSITY);
        if (window == 0) {
            System.out.println("[metalmc-backend] present test: SDL_CreateWindow failed");
            return;
        }
        long view = 0, surface = 0;
        try {
            SDLVideo.SDL_ShowWindow(window);
            SDLEvents.SDL_PumpEvents();
            view = SDLMetal.SDL_Metal_CreateView(window);
            long layer = view == 0 ? 0 : SDLMetal.SDL_Metal_GetLayer(view);
            surface = layer == 0 ? 0 : lib.surfaceCreate(layer, displaySync);
            if (surface == 0) {
                System.out.println("[metalmc-backend] present test: no CAMetalLayer surface");
                return;
            }
            int[] size = lib.surfaceSize(surface);
            // Warm up, then measure 1200 frames in chunks, pumping window events between chunks.
            lib.surfaceRunFrames(surface, 60, 0);
            SDLEvents.SDL_PumpEvents();
            double seconds = 0;
            int frames = 0;
            for (int chunk = 0; chunk < 20; chunk++) {
                seconds += lib.surfaceRunFrames(surface, 60, 60 + frames);
                frames += 60;
                SDLEvents.SDL_PumpEvents();
            }
            System.out.printf(java.util.Locale.ROOT,
                "METALMC_PRESENT displaySync=%s drawable=%dx%d frames=%d seconds=%.3f fps=%.1f%n",
                displaySync, size[0], size[1], frames, seconds, frames / seconds);
        } finally {
            if (surface != 0) lib.release(surface);
            if (view != 0) SDLMetal.SDL_Metal_DestroyView(view);
            SDLVideo.SDL_DestroyWindow(window);
        }
    }
}
