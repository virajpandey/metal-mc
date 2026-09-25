package metalmc.backend;

import com.mojang.renderpearl.api.commands.GpuQueryPool;
import java.util.Arrays;
import java.util.OptionalLong;

/** Timestamp queries aren't implemented yet; every value reads as unavailable. */
final class MetalQueryPool implements GpuQueryPool {
    private final int size;

    MetalQueryPool(int size) {
        this.size = size;
    }

    @Override
    public int size() {
        return size;
    }

    @Override
    public OptionalLong getValue(int index) {
        return OptionalLong.empty();
    }

    @Override
    public OptionalLong[] getValues(int index, int count) {
        OptionalLong[] values = new OptionalLong[count];
        Arrays.fill(values, OptionalLong.empty());
        return values;
    }

    @Override
    public void close() {
    }
}
