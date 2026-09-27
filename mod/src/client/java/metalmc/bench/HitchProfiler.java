package metalmc.bench;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.function.Supplier;
import net.minecraft.util.profiling.ProfilerFiller;
import net.minecraft.util.profiling.metrics.MetricCategory;

/**
 * -PbenchHitches=1: times vanilla's profiler sections on the render thread during the timed run, and for every
 * frame whose Java work misses a 120 Hz refresh, logs where the time went (METALMC_SLOWFRAME, sections of at
 * least 0.3 ms, four levels deep). Sections are a trie of the names vanilla pushes, so a steady frame allocates
 * nothing; names vanilla builds on demand (entity types and the like) aren't evaluated and count as "~".
 */
public final class HitchProfiler implements ProfilerFiller {
    public static final HitchProfiler INSTANCE = new HitchProfiler();
    private static final long THRESHOLD_NS = 8_333_333L;
    private static final long MIN_SECTION_NS = 300_000L;
    private static final int MAX_DEPTH = 4;

    private static final class Node {
        final String name;
        final Node parent;
        final int depth;
        final HashMap<String, Node> children = new HashMap<>(4);
        long time, start, frame;

        Node(String name, Node parent) {
            this.name = name;
            this.parent = parent;
            this.depth = parent == null ? 0 : parent.depth + 1;
        }
    }

    private final Node root = new Node("", null);
    private Node current = root;
    private int overflow;        // pushes below MAX_DEPTH, not recorded
    private long frameStart, frameId;

    private HitchProfiler() {
    }

    @Override
    public void startTick() {
        current = root;
        overflow = 0;
        frameId++;
        frameStart = System.nanoTime();
    }

    @Override
    public void endTick() {
        long frame = System.nanoTime() - frameStart;
        if (frame < THRESHOLD_NS || !Bench.running()) return;
        ArrayList<Node> slow = new ArrayList<>();
        collect(root, slow);
        slow.sort((a, b) -> Long.compare(b.time, a.time));
        StringBuilder sb = new StringBuilder(String.format(java.util.Locale.ROOT, "METALMC_SLOWFRAME %s %.1f ms:",
            java.time.LocalTime.now(), frame / 1e6));
        for (int i = 0; i < Math.min(14, slow.size()); i++) {
            Node n = slow.get(i);
            sb.append(' ').append(path(n)).append(String.format(java.util.Locale.ROOT, " %.1f;", n.time / 1e6));
        }
        System.out.println(sb);
    }

    private void collect(Node n, ArrayList<Node> out) {
        for (Node c : n.children.values()) {
            if (c.frame != frameId || c.time < MIN_SECTION_NS) continue;
            out.add(c);
            collect(c, out);
        }
    }

    private static String path(Node n) {
        return n.parent == null || n.parent.parent == null ? n.name : path(n.parent) + "/" + n.name;
    }

    @Override
    public void push(String name) {
        if (current.depth >= MAX_DEPTH || overflow > 0) {
            overflow++;
            return;
        }
        Node c = current.children.get(name);
        if (c == null) {
            c = new Node(name, current);
            current.children.put(name, c);
        }
        if (c.frame != frameId) {
            c.frame = frameId;
            c.time = 0;
        }
        c.start = System.nanoTime();
        current = c;
    }

    @Override
    public void push(Supplier<String> name) {
        push("~");
    }

    @Override
    public void pop() {
        if (overflow > 0) {
            overflow--;
            return;
        }
        if (current == root) return;
        current.time += System.nanoTime() - current.start;
        current = current.parent;
    }

    @Override
    public void popPush(String name) {
        pop();
        push(name);
    }

    @Override
    public void popPush(Supplier<String> name) {
        pop();
        push("~");
    }

    @Override
    public void markForCharting(MetricCategory category) {
    }

    @Override
    public void incrementCounter(String name, int amount) {
    }

    @Override
    public void incrementCounter(Supplier<String> name, int amount) {
    }
}
