package metalmc.backend;

import com.mojang.renderpearl.api.device.GpuSurface;
import com.mojang.renderpearl.api.device.SurfaceException;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import com.mojang.renderpearl.backend.api.CommandEncoderBackend;
import com.mojang.renderpearl.backend.api.GpuSurfaceBackend;
import java.util.Collection;
import java.util.EnumSet;
import java.util.Set;
import org.lwjgl.sdl.SDLError;
import org.lwjgl.sdl.SDLMetal;

/**
 * The window's CAMetalLayer (SDL_Metal_CreateView). Minecraft's frame is acquire, render, blit,
 * submit, present: the blit records a flipped copy into the drawable and the submit schedules its
 * presentation on the command buffer.
 */
final class MetalSurface implements GpuSurfaceBackend {
    private static final Set<GpuSurface.PresentMode> MODES = EnumSet.of(GpuSurface.PresentMode.IMMEDIATE, GpuSurface.PresentMode.FIFO);

    private final long view;
    private final long handle;

    MetalSurface(long window) {
        this.view = SDLMetal.SDL_Metal_CreateView(window);
        if (view == 0) throw new IllegalStateException("SDL_Metal_CreateView failed: " + SDLError.SDL_GetError());
        long layer = SDLMetal.SDL_Metal_GetLayer(view);
        this.handle = layer == 0 ? 0 : Mtl.surfaceCreate(layer);
        if (handle == 0) throw new IllegalStateException("No CAMetalLayer for the window");
    }

    @Override
    public void configure(GpuSurface.Configuration config) throws SurfaceException {
        if (config.width() <= 0 || config.height() <= 0) throw new SurfaceException("Invalid surface size " + config.width() + "x" + config.height());
        Mtl.surfaceConfigure(handle, config.width(), config.height(), config.presentMode() == GpuSurface.PresentMode.IMMEDIATE ? 0 : 1);
    }

    @Override
    public boolean isSuboptimal() {
        return false;
    }

    @Override
    public void acquireNextTexture() throws SurfaceException {
        if (!Mtl.surfaceAcquire(handle)) throw new SurfaceException("CAMetalLayer.nextDrawable timed out");
    }

    @Override
    public void blitFromTexture(CommandEncoderBackend commandEncoder, GpuTextureView textureView) {
        ((MetalCommandEncoder) commandEncoder).blitToSurface(handle, textureView);
    }

    @Override
    public void present() {
        Mtl.surfacePresent(handle);
    }

    @Override
    public Collection<GpuSurface.PresentMode> supportedPresentModes() {
        return MODES;
    }

    @Override
    public void close() {
        Mtl.release(handle);
        SDLMetal.SDL_Metal_DestroyView(view);
    }
}
