package metalmc.backend;

import com.mojang.renderpearl.api.textures.GpuTexture;
import com.mojang.renderpearl.api.textures.GpuTextureView;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemoryLayout;
import java.lang.foreign.SymbolLookup;
import java.lang.invoke.MethodHandle;
import java.nio.FloatBuffer;
import net.minecraft.client.renderer.fog.FogData;
import org.joml.Matrix4fc;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;

import static java.lang.foreign.ValueLayout.JAVA_INT;
import static java.lang.foreign.ValueLayout.JAVA_LONG;

/**
 * Bridge to lit mode (Sources/MetalMCNative/Lit.swift, METALMC_EXP=lit, off by default): the terrain shaders we own write a
 * G-buffer in the level's main pass, and after the level a full-screen pass relights the terrain from it with the sun, the
 * sky, the moon and block light. Meant to run with METALMC_EXP=nearchunks (near terrain in our shaders) and rtshadows (sun
 * visibility). Only works on the Metal backend; the bindings live here so the feature stays in its own files. Render thread.
 */
public final class MetalLit {
    private MetalLit() {
    }

    /** METALMC_EXP=lit (off by default). The native side reads the same variable. */
    public static final boolean ENABLED = experiment("lit");

    private static boolean experiment(String name) {
        String v = System.getenv("METALMC_EXP");
        // The settings' switches (config/metalmc.properties) reach the native side by a setenv that Java's copy of the
        // environment, taken at startup, never sees: the same rule here (METALMC_EXP in the environment first).
        if (v == null) v = metalmc.MetalMCConfig.nativeExperiments();
        for (String s : v.split(",")) {
            if (s.trim().equals(name)) return true;
        }
        return false;
    }

    public static boolean enabled() {
        return ENABLED && MetalDevice.current != null;
    }

    /** Resolved on first use, so nothing loads the library unless the Metal backend is running. */
    private static final class Native {
        private static final Linker LINKER = Linker.nativeLinker();
        private static final SymbolLookup LIB = SymbolLookup.libraryLookup(NativeLibrary.path(), Arena.global());

        private static MethodHandle h(String name, boolean critical, MemoryLayout ret, MemoryLayout... args) {
            FunctionDescriptor fd = ret == null ? FunctionDescriptor.ofVoid(args) : FunctionDescriptor.of(ret, args);
            var addr = LIB.find(name).orElseThrow(() -> new IllegalStateException("missing native symbol " + name));
            return critical ? LINKER.downcallHandle(addr, fd, Linker.Option.critical(false)) : LINKER.downcallHandle(addr, fd);
        }

        static final MethodHandle LEVEL_PASS = h("mmc_lit_level_pass", true, null);
        static final MethodHandle RELIGHT = h("mmc_lit_relight", false, JAVA_INT, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_LONG, JAVA_INT);
    }

    private static RuntimeException rethrow(Throwable t) {
        if (t instanceof RuntimeException r) return r;
        if (t instanceof Error e) throw e;
        return new IllegalStateException(t);
    }

    /** The next render pass is the level's main pass (LevelRenderer's, where the terrain is drawn): it gets the G-buffer. */
    public static void markLevelPass() {
        if (!enabled()) return;
        try { Native.LEVEL_PASS.invokeExact(); } catch (Throwable t) { throw rethrow(t); }
    }

    /**
     * Relights the terrain of the level just drawn into {@code color}, after the ray-traced shadows (it uses their
     * visibility) and before the aerial perspective and the anti-aliasing. {@code projection} is the one the level was drawn
     * with (jittered with anti-aliasing), {@code sunAngle} vanilla's, {@code fog} the camera's fog as the level used it,
     * {@code lightmap} vanilla's lightmap, {@code skyActive} whether our sky drew this frame (its atmosphere then gives the
     * sun's and sky's light; otherwise a daylight curve scaled by vanilla's). With {@code taa} it's left for the
     * anti-aliasing's resolve, which applies it as it loads each pixel. Returns true if it ran (or was left for it).
     */
    public static boolean relight(GpuTexture color, GpuTexture depth, Matrix4fc projection, Matrix4fc viewRotation, float sunAngle,
                                  FogData fog, GpuTextureView lightmap, boolean skyActive, boolean taa) {
        if (!enabled() || !(color instanceof MetalTexture c) || !(depth instanceof MetalTexture d)) return false;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            FloatBuffer p = stack.mallocFloat(42);
            projection.get(0, p);
            viewRotation.get(16, p);
            p.put(32, sunAngle);
            p.put(33, fog.color.x()).put(34, fog.color.y()).put(35, fog.color.z()).put(36, fog.color.w());
            p.put(37, fog.environmentalStart).put(38, fog.environmentalEnd).put(39, fog.renderDistanceStart).put(40, fog.renderDistanceEnd);
            p.put(41, skyActive ? 1f : 0f);
            long lm = lightmap instanceof MetalTextureView v ? v.handle : 0L;
            return (int) Native.RELIGHT.invokeExact(c.handle, d.handle, MemoryUtil.memAddress(p), lm, taa ? 1 : 0) == 1;
        } catch (Throwable t) {
            throw rethrow(t);
        }
    }
}
