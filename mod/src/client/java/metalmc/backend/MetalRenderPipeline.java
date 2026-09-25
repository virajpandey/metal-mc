package metalmc.backend;

import com.mojang.logging.LogUtils;
import com.mojang.renderpearl.api.pipeline.BindGroupLayout;
import com.mojang.renderpearl.api.pipeline.BlendFunction;
import com.mojang.renderpearl.api.pipeline.ColorTargetState;
import com.mojang.renderpearl.api.pipeline.DepthStencilState;
import com.mojang.renderpearl.api.pipeline.PolygonMode;
import com.mojang.renderpearl.api.pipeline.ShaderType;
import com.mojang.renderpearl.api.pipeline.UniformType;
import com.mojang.renderpearl.backend.api.BackendRenderPipeline;
import com.mojang.renderpearl.util.ShaderCompileException;
import it.unimi.dsi.fastutil.ints.IntArrayList;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.IntBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.jspecify.annotations.Nullable;
import org.lwjgl.PointerBuffer;
import org.lwjgl.system.MemoryStack;
import org.lwjgl.system.MemoryUtil;
import org.lwjgl.util.spvc.Spvc;
import org.lwjgl.util.spvc.SpvcMslResourceBinding;
import org.lwjgl.util.spvc.SpvcReflectedResource;
import org.slf4j.Logger;

/**
 * A render pipeline: the SPIR-V from Minecraft's frontend is translated to MSL with SPIRV-Cross
 * (bundled with the game through LWJGL), then compiled into Metal pipeline states on the native side.
 */
final class MetalRenderPipeline implements BackendRenderPipeline {
    private static final Logger LOGGER = LogUtils.getLogger();
    /** Metal buffer index for push constants (see Backend.swift). */
    static final int PUSH_CONSTANTS_INDEX = 24;
    private static final @Nullable String DUMP_DIR = emptyToNull(System.getProperty("metalmc.dumpMsl"));

    private static @Nullable String emptyToNull(@Nullable String s) {
        return s == null || s.isEmpty() ? null : s;
    }

    private final MetalDevice device;
    final long handle;
    final List<BindGroupLayout.UniformDescription> uniforms;
    /** Per uniform: 0 buffer, 1 texture+sampler, 2 texel buffer. */
    final int[] uniformKinds;
    final int[] texelFormats;
    final int[] texelBytes;
    private boolean closed;

    private MetalRenderPipeline(MetalDevice device, long handle, List<BindGroupLayout.UniformDescription> uniforms) {
        this.device = device;
        this.handle = handle;
        this.uniforms = uniforms;
        int n = uniforms.size();
        this.uniformKinds = new int[n];
        this.texelFormats = new int[n];
        this.texelBytes = new int[n];
        for (int i = 0; i < n; i++) {
            BindGroupLayout.UniformDescription u = uniforms.get(i);
            uniformKinds[i] = switch (u.type()) {
                case UNIFORM_BUFFER -> 0;
                case COMBINED_IMAGE_SAMPLER -> 1;
                case TEXEL_BUFFER -> 2;
            };
            if (u.type() == UniformType.TEXEL_BUFFER && u.gpuFormat() != null) {
                texelFormats[i] = MetalConst.pixelFormat(u.gpuFormat());
                texelBytes[i] = u.gpuFormat().blockSize();
            }
        }
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

    record Msl(String source, String entryPoint, int uniformMask) {
    }

    static BackendRenderPipeline.Pending compile(MetalDevice device, BackendRenderPipeline.CreateInfo info) {
        if (info.uniforms().size() > PUSH_CONSTANTS_INDEX || info.uniforms().size() > 16) {
            LOGGER.error("Pipeline {} has {} uniforms; the Metal backend supports 16", info.name(), info.uniforms().size());
            return BackendRenderPipeline.Pending.NULL;
        }
        Msl vertex = null;
        Msl fragment = null;
        try {
            for (BackendRenderPipeline.CreateInfo.Shader shader : info.shaders()) {
                Msl msl = translate(shader, info.name());
                if (shader.module().type() == ShaderType.VERTEX) vertex = msl;
                else fragment = msl;
            }
        } catch (ShaderCompileException e) {
            LOGGER.error("Couldn't translate pipeline {} to MSL", info.name(), e);
            return BackendRenderPipeline.Pending.NULL;
        }
        if (vertex == null) {
            LOGGER.error("Pipeline {} has no vertex shader", info.name());
            return BackendRenderPipeline.Pending.NULL;
        }

        IntArrayList p = new IntArrayList();
        p.add(MetalConst.topology(info.primitiveTopology()));
        p.add(info.cull() ? 1 : 0);
        p.add(info.polygonMode() == PolygonMode.WIREFRAME ? 1 : 0);
        DepthStencilState depth = info.depthStencilState();
        p.add(depth != null ? 1 : 0);
        p.add(depth != null ? MetalConst.compare(depth.depthTest()) : 7);
        p.add(depth != null && depth.writeDepth() ? 1 : 0);
        p.add(Float.floatToRawIntBits(depth != null ? depth.depthBiasConstant() : 0f));
        p.add(Float.floatToRawIntBits(depth != null ? depth.depthBiasScaleFactor() : 0f));
        p.add(vertex.uniformMask());
        p.add(fragment != null ? fragment.uniformMask() : 0);
        p.add(0);
        List<@Nullable ColorTargetState> targets = info.colorTargetStates();
        p.add(targets.size());
        for (ColorTargetState t : targets) {
            if (t == null) {
                for (int i = 0; i < 9; i++) p.add(0);
                continue;
            }
            p.add(MetalConst.pixelFormat(t.format()));
            p.add(MetalConst.writeMask(t));
            BlendFunction blend = t.blendFunction().orElse(null);
            p.add(blend != null ? 1 : 0);
            p.add(blend != null ? MetalConst.blendOp(blend.color().op()) : 0);
            p.add(blend != null ? MetalConst.blendOp(blend.alpha().op()) : 0);
            p.add(blend != null ? MetalConst.blendFactor(blend.color().sourceFactor()) : 1);
            p.add(blend != null ? MetalConst.blendFactor(blend.color().destFactor()) : 0);
            p.add(blend != null ? MetalConst.blendFactor(blend.alpha().sourceFactor()) : 1);
            p.add(blend != null ? MetalConst.blendFactor(blend.alpha().destFactor()) : 0);
        }
        p.add(info.vertexBuffers().size());
        for (BackendRenderPipeline.CreateInfo.VertexBuffer vb : info.vertexBuffers()) {
            p.add(vb.bufferSlot());
            p.add(vb.stride());
            p.add(vb.stepRate());
        }
        p.add(info.attribBindings().size());
        for (BackendRenderPipeline.CreateInfo.AttribBinding a : info.attribBindings()) {
            p.add(a.location());
            p.add(a.bufferSlot());
            p.add(a.offset());
            int fmt = MetalConst.vertexFormat(a.format());
            if (fmt == 0) {
                LOGGER.error("Pipeline {}: no Metal vertex format for {}", info.name(), a.format());
                return BackendRenderPipeline.Pending.NULL;
            }
            p.add(fmt);
        }

        long handle;
        String error;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            IntBuffer params = stack.mallocInt(p.size());
            for (int i = 0; i < p.size(); i++) params.put(i, p.getInt(i));
            ByteBuffer err = stack.calloc(4096);
            ByteBuffer name = stack.UTF8(info.name());
            ByteBuffer vsEntry = stack.UTF8(vertex.entryPoint());
            ByteBuffer fsEntry = fragment != null ? stack.UTF8(fragment.entryPoint()) : null;
            ByteBuffer vsSrc = MemoryUtil.memUTF8(vertex.source());
            ByteBuffer fsSrc = fragment != null ? MemoryUtil.memUTF8(fragment.source()) : null;
            try {
                handle = Mtl.pipelineCreate(MemoryUtil.memAddress(name), MemoryUtil.memAddress(vsSrc), MemoryUtil.memAddress(vsEntry),
                    fsSrc != null ? MemoryUtil.memAddress(fsSrc) : 0L, fsEntry != null ? MemoryUtil.memAddress(fsEntry) : 0L,
                    MemoryUtil.memAddress(params), p.size(), MemoryUtil.memAddress(err), err.capacity());
            } finally {
                MemoryUtil.memFree(vsSrc);
                if (fsSrc != null) MemoryUtil.memFree(fsSrc);
            }
            error = MemoryUtil.memUTF8Safe(err);
        }
        if (handle == 0) {
            LOGGER.error("Couldn't compile Metal pipeline {}: {}", info.name(), error);
            return BackendRenderPipeline.Pending.NULL;
        }
        MetalRenderPipeline pipeline = new MetalRenderPipeline(device, handle, info.uniforms());
        return () -> pipeline;
    }

    private static void check(int result, long context, String what) throws ShaderCompileException {
        if (result != Spvc.SPVC_SUCCESS) {
            throw new ShaderCompileException(what + ": " + Spvc.spvc_context_get_last_error_string(context));
        }
    }

    private static final int[] DESCRIPTOR_RESOURCE_TYPES = {
        Spvc.SPVC_RESOURCE_TYPE_UNIFORM_BUFFER, Spvc.SPVC_RESOURCE_TYPE_STORAGE_BUFFER, Spvc.SPVC_RESOURCE_TYPE_STORAGE_IMAGE,
        Spvc.SPVC_RESOURCE_TYPE_SAMPLED_IMAGE, Spvc.SPVC_RESOURCE_TYPE_SEPARATE_IMAGE, Spvc.SPVC_RESOURCE_TYPE_SEPARATE_SAMPLERS
    };

    /** SPIR-V to MSL. Also returns which uniform bindings this stage reads. */
    static Msl translate(BackendRenderPipeline.CreateInfo.Shader shader, String pipelineName) throws ShaderCompileException {
        ShaderType type = shader.module().type();
        int model = type == ShaderType.VERTEX ? 0 : 4; // SpvExecutionModelVertex / Fragment
        IntBuffer spirv = shader.module().spv().asIntBuffer();
        long context = 0;
        try (MemoryStack stack = MemoryStack.stackPush()) {
            PointerBuffer ret = stack.callocPointer(1);
            if (Spvc.spvc_context_create(ret) != Spvc.SPVC_SUCCESS) throw new ShaderCompileException("spvc_context_create failed");
            context = ret.get(0);
            check(Spvc.spvc_context_parse_spirv(context, spirv, spirv.remaining(), ret), context, "parse SPIR-V");
            long ir = ret.get(0);
            check(Spvc.spvc_context_create_compiler(context, Spvc.SPVC_BACKEND_MSL, ir, Spvc.SPVC_CAPTURE_MODE_TAKE_OWNERSHIP, ret), context, "create MSL compiler");
            long compiler = ret.get(0);
            check(Spvc.spvc_compiler_create_compiler_options(compiler, ret), context, "create options");
            long options = ret.get(0);
            Spvc.spvc_compiler_options_set_uint(options, Spvc.SPVC_COMPILER_OPTION_MSL_VERSION, 30000);
            Spvc.spvc_compiler_options_set_uint(options, Spvc.SPVC_COMPILER_OPTION_MSL_PLATFORM, Spvc.SPVC_MSL_PLATFORM_MACOS);
            Spvc.spvc_compiler_options_set_bool(options, Spvc.SPVC_COMPILER_OPTION_FLIP_VERTEX_Y, true);
            Spvc.spvc_compiler_options_set_bool(options, Spvc.SPVC_COMPILER_OPTION_MSL_TEXTURE_BUFFER_NATIVE, true);
            Spvc.spvc_compiler_options_set_bool(options, Spvc.SPVC_COMPILER_OPTION_MSL_PAD_FRAGMENT_OUTPUT_COMPONENTS, true);
            check(Spvc.spvc_compiler_install_compiler_options(compiler, options), context, "install options");

            // Which bindings does this stage use? (The frontend already rewrote them to set 0, binding i.)
            int mask = 0;
            check(Spvc.spvc_compiler_create_shader_resources(compiler, ret), context, "reflect resources");
            long resources = ret.get(0);
            PointerBuffer count = stack.callocPointer(1);
            for (int resourceType : DESCRIPTOR_RESOURCE_TYPES) {
                check(Spvc.spvc_resources_get_resource_list_for_type(resources, resourceType, ret, count), context, "list resources");
                int n = (int) count.get(0);
                if (n == 0) continue;
                SpvcReflectedResource.Buffer list = SpvcReflectedResource.create(ret.get(0), n);
                for (int i = 0; i < n; i++) {
                    int id = list.get(i).id();
                    int set = Spvc.spvc_compiler_get_decoration(compiler, id, 34);
                    int binding = Spvc.spvc_compiler_get_decoration(compiler, id, 33);
                    if (set != 0 || binding >= 32) {
                        throw new ShaderCompileException("unexpected descriptor set " + set + " binding " + binding);
                    }
                    mask |= 1 << binding;
                }
            }

            for (int binding = 0; binding < 32; binding++) {
                if ((mask & (1 << binding)) == 0) continue;
                SpvcMslResourceBinding b = SpvcMslResourceBinding.calloc(stack);
                Spvc.spvc_msl_resource_binding_init(b);
                b.stage(model).desc_set(0).binding(binding).msl_buffer(binding).msl_texture(binding).msl_sampler(binding);
                check(Spvc.spvc_compiler_msl_add_resource_binding(compiler, b), context, "add resource binding");
            }
            SpvcMslResourceBinding pc = SpvcMslResourceBinding.calloc(stack);
            Spvc.spvc_msl_resource_binding_init(pc);
            pc.stage(model).desc_set(Spvc.SPVC_MSL_PUSH_CONSTANT_DESC_SET).binding(Spvc.SPVC_MSL_PUSH_CONSTANT_BINDING)
                .msl_buffer(PUSH_CONSTANTS_INDEX);
            check(Spvc.spvc_compiler_msl_add_resource_binding(compiler, pc), context, "add push constant binding");

            check(Spvc.spvc_compiler_set_entry_point(compiler, shader.entryPoint(), model), context, "set entry point");
            check(Spvc.spvc_compiler_compile(compiler, ret), context, "compile MSL");
            String source = MemoryUtil.memUTF8(ret.get(0));
            String entry = Spvc.spvc_compiler_get_cleansed_entry_point_name(compiler, shader.entryPoint(), model);
            if (DUMP_DIR != null) dump(pipelineName, shader.name(), type, source);
            return new Msl(source, entry, mask);
        } finally {
            if (context != 0) Spvc.spvc_context_destroy(context);
        }
    }

    private static void dump(String pipeline, String shader, ShaderType type, String source) {
        try {
            Path dir = Path.of(DUMP_DIR);
            Files.createDirectories(dir);
            String file = (pipeline + "__" + shader).replaceAll("[^A-Za-z0-9_.-]", "_") + (type == ShaderType.VERTEX ? ".vert" : ".frag") + ".metal";
            Files.writeString(dir.resolve(file), source, StandardCharsets.UTF_8);
        } catch (IOException e) {
            LOGGER.warn("Couldn't dump MSL", e);
        }
    }
}
