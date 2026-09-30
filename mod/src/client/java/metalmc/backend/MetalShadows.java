package metalmc.backend;

import java.nio.DoubleBuffer;
import java.nio.FloatBuffer;
import org.joml.Matrix4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

/** Bridge to the native ray-traced sun shadows (RtShadows.swift; METALMC_EXP=rtshadows). Render thread. */
public final class MetalShadows {
    private MetalShadows() {
    }

    /**
     * Darkens the parts of the level just drawn into {@code color} that the sun doesn't reach. {@code projection} is the
     * projection the level was drawn with (jittered when anti-aliasing is on), {@code sunAngle} vanilla's sun angle in
     * radians, {@code strength} how dark full shadow is (0 turns them off). With {@code taa} the shadows are left for the
     * anti-aliasing to apply, which reads every pixel anyway.
     */
    public static boolean apply(com.mojang.renderpearl.api.textures.GpuTexture color, com.mojang.renderpearl.api.textures.GpuTexture depth,
                                Matrix4fc projection, Matrix4fc viewRotation, double camX, double camY, double camZ,
                                float sunAngle, float strength, float cloudHeight, boolean taa) {
        if (MetalDevice.current == null || !(color instanceof MetalTexture c) || !(depth instanceof MetalTexture d)) return false;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer p = stack.mallocFloat(32);
            projection.get(0, p);
            viewRotation.get(16, p);
            DoubleBuffer cam = stack.mallocDouble(3);
            cam.put(0, camX).put(1, camY).put(2, camZ);
            return Mtl.shadowsApply(c.handle, d.handle, MemoryUtil.memAddress(p), MemoryUtil.memAddress(cam), sunAngle, strength, cloudHeight, taa ? 1 : 0) != 0;
        }
    }
}
