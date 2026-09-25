package metalmc.terrain;

import com.mojang.blaze3d.vertex.MeshData;
import java.nio.ByteBuffer;
import org.jspecify.annotations.Nullable;
import org.lwjgl.system.MemoryUtil;

/**
 * Face-direction buckets for terrain (the idea behind Sodium's biggest terrain win). At mesh-build time
 * each section layer's quads are reordered so quads facing the same axis direction are contiguous.
 * At draw time only the buckets that can face the camera are drawn. The result is pixel-identical,
 * because back faces would be culled anyway, but the GPU skips their vertex work.
 */
public final class FacingSorter {
    private FacingSorter() {
    }

    public static final boolean ENABLED = !"0".equals(System.getProperty("metalmc.facingCulling", "1"));

    // Draw order: buckets that are usually visible together are adjacent, so they merge into one draw.
    public static final int UNASSIGNED = 0, POS_Y = 1, NEG_X = 2, NEG_Z = 3, POS_X = 4, POS_Z = 5, NEG_Y = 6;
    public static final int BUCKETS = 7;

    /**
     * Reorders the quads of a sequentially indexed quad mesh by facing, in place, and returns the quad
     * count of each bucket. Returns null if the mesh isn't a plain quad list. Position must be the first
     * vertex attribute (float3), as in DefaultVertexFormat.BLOCK.
     */
    public static int @Nullable [] sort(MeshData mesh) {
        MeshData.DrawState state = mesh.drawState();
        if (mesh.indexBuffer() != null || state.vertexCount() % 4 != 0 || state.vertexCount() == 0) return null;
        int stride = state.format().getVertexSize();
        int quads = state.vertexCount() / 4;
        int quadBytes = 4 * stride;
        ByteBuffer vb = mesh.vertexBuffer();
        if (vb.remaining() < quads * quadBytes) return null;
        long base = MemoryUtil.memAddress(vb);

        byte[] bucketOf = new byte[quads];
        int[] counts = new int[BUCKETS];
        for (int q = 0; q < quads; q++) {
            long a = base + (long) q * quadBytes;
            // Normal from the diagonals: robust even if two corners coincide. Quads wind counter-clockwise
            // seen from the front, so this points out of the visible side.
            float d1x = MemoryUtil.memGetFloat(a + 2L * stride) - MemoryUtil.memGetFloat(a);
            float d1y = MemoryUtil.memGetFloat(a + 2L * stride + 4) - MemoryUtil.memGetFloat(a + 4);
            float d1z = MemoryUtil.memGetFloat(a + 2L * stride + 8) - MemoryUtil.memGetFloat(a + 8);
            float d2x = MemoryUtil.memGetFloat(a + 3L * stride) - MemoryUtil.memGetFloat(a + stride);
            float d2y = MemoryUtil.memGetFloat(a + 3L * stride + 4) - MemoryUtil.memGetFloat(a + stride + 4);
            float d2z = MemoryUtil.memGetFloat(a + 3L * stride + 8) - MemoryUtil.memGetFloat(a + stride + 8);
            float nx = d1y * d2z - d1z * d2y;
            float ny = d1z * d2x - d1x * d2z;
            float nz = d1x * d2y - d1y * d2x;
            int b = classify(nx, ny, nz);
            bucketOf[q] = (byte) b;
            counts[b]++;
        }

        // Already in bucket order (e.g. one bucket)? Then nothing to move.
        boolean sorted = true;
        for (int q = 1; q < quads && sorted; q++) sorted = bucketOf[q - 1] <= bucketOf[q];
        if (!sorted) {
            int[] next = new int[BUCKETS];
            for (int b = 1; b < BUCKETS; b++) next[b] = next[b - 1] + counts[b - 1];
            long tmp = MemoryUtil.nmemAlloc((long) quads * quadBytes);
            try {
                for (int q = 0; q < quads; q++) {
                    MemoryUtil.memCopy(base + (long) q * quadBytes, tmp + (long) next[bucketOf[q]]++ * quadBytes, quadBytes);
                }
                MemoryUtil.memCopy(tmp, base, (long) quads * quadBytes);
            } finally {
                MemoryUtil.nmemFree(tmp);
            }
        }
        return counts;
    }

    static int classify(float nx, float ny, float nz) {
        float ax = Math.abs(nx), ay = Math.abs(ny), az = Math.abs(nz);
        float eps = 1e-4f * (ax + ay + az);
        if (ax + ay + az == 0) return UNASSIGNED;
        if (ay > 0 && ax <= eps && az <= eps) return ny > 0 ? POS_Y : NEG_Y;
        if (ax > 0 && ay <= eps && az <= eps) return nx > 0 ? POS_X : NEG_X;
        if (az > 0 && ax <= eps && ay <= eps) return nz > 0 ? POS_Z : NEG_Z;
        return UNASSIGNED;
    }

    /**
     * Bitmask of buckets that may face a camera at (cx, cy, cz), for a section with origin (ox, oy, oz).
     * Conservative by one block, because some block models reach outside their block.
     */
    public static int visibleMask(double cx, double cy, double cz, int ox, int oy, int oz) {
        int m = 1 << UNASSIGNED;
        if (cy > oy - 1) m |= 1 << POS_Y;
        if (cy < oy + 17) m |= 1 << NEG_Y;
        if (cx > ox - 1) m |= 1 << POS_X;
        if (cx < ox + 17) m |= 1 << NEG_X;
        if (cz > oz - 1) m |= 1 << POS_Z;
        if (cz < oz + 17) m |= 1 << NEG_Z;
        return m;
    }
}
