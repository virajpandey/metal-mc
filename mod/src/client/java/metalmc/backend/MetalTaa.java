package metalmc.backend;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import org.joml.Matrix4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

/** Bridge to the native temporal anti-aliasing (Taa.swift). Render thread. */
public final class MetalTaa {
    private MetalTaa() {
    }

    /**
     * Blends the level just drawn into {@code color} with its history (MetalFX). {@code projection} is this frame's
     * projection without the jitter, {@code jitterX/Y} the jitter it was drawn with, in pixels.
     */
    public static boolean apply(com.mojang.renderpearl.api.textures.GpuTexture color, com.mojang.renderpearl.api.textures.GpuTexture depth,
                                Matrix4fc projection, Matrix4fc viewRotation, double camX, double camY, double camZ,
                                float jitterX, float jitterY, boolean reset) {
        if (MetalDevice.current == null || !(color instanceof MetalTexture c) || !(depth instanceof MetalTexture d)) return false;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer p = stack.mallocFloat(32);
            projection.get(0, p);
            viewRotation.get(16, p);
            DoubleBuffer cam = stack.mallocDouble(3);
            cam.put(0, camX).put(1, camY).put(2, camZ);
            return Mtl.taaApply(c.handle, d.handle, MemoryUtil.memAddress(p), MemoryUtil.memAddress(cam), jitterX, jitterY, reset ? 1 : 0) != 0;
        }
    }
}
