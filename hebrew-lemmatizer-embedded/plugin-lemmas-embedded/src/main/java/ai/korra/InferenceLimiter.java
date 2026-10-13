package ai.korra;

import java.util.concurrent.Semaphore;

/**
 * Caps the native (off-heap) memory the model may use at once, the way Elasticsearch's ML nodes
 * give their native inference a fixed memory limit instead of letting it grow with load.
 *
 * <p>ONNX Runtime allocates every running window outside the JVM heap, where Elasticsearch's
 * circuit breakers cannot see it, and keeps its peak allocation. Without a cap, every write thread
 * that analyzes a long value runs a full window at the same time; on a 6 GB node with a 3 GB heap
 * and 8 write threads that was enough for the kernel to kill Elasticsearch.
 *
 * <p>The cap is counted in model positions, because a window's memory grows with its length.
 * Short values take few positions and still run side by side; only long windows wait for each
 * other. A thread that waits is a write thread that is busy, so a node at its cap pushes back on
 * indexing the normal Elasticsearch way: the write queue fills and clients get 429 responses.
 *
 * <p>Short windows, which is nearly every search query, have a small budget of their own, so they
 * never queue behind long windows. Sharing one budget made a 4-word query wait up to a second
 * while the node indexed long values. ML nodes get the same separation by deploying a model twice,
 * once for ingest and once for search.
 */
final class InferenceLimiter {

    /**
     * Peak native memory per position of a running window: 0.3-0.4 GB was measured per
     * 512-position window, most of it the [positions, 128000] vocabulary scores computed before
     * the top-3 predictions are taken.
     */
    static final long BYTES_PER_POSITION = 700L * 1024;

    /** JVM memory outside the heap (metaspace, thread stacks, network buffers) and the loaded model. */
    static final long RESERVED_BYTES = 1536L * 1024 * 1024;

    /** Windows of at most this many positions use the short-window budget. */
    static final int SHORT_WINDOW = 64;

    /** Positions short windows may use at once, on top of the main budget: about 175 MB. */
    static final int SHORT_WINDOW_BUDGET = 256;

    private final int capacity;
    private final Semaphore positions;
    private final Semaphore shortWindowPositions;

    InferenceLimiter(int capacity) {
        if (capacity <= SHORT_WINDOW) {
            throw new IllegalArgumentException("capacity must be above " + SHORT_WINDOW + ", was " + capacity);
        }
        this.capacity = capacity;
        // Fair, so a full window is not starved by a steady stream of shorter ones.
        this.positions = new Semaphore(capacity, true);
        this.shortWindowPositions = new Semaphore(SHORT_WINDOW_BUDGET, true);
    }

    /**
     * Positions this node lets the model process at once.
     *
     * <p>Half of the memory outside the heap that is not already reserved goes to the model; the
     * other half stays for the OS page cache that Lucene searches depend on. There is no point in
     * going past what the CPUs can run anyway (each window uses {@code threadsPerWindow} threads),
     * since more concurrent windows would only add memory, not throughput. A full window is always
     * allowed, so every value can still be lemmatized on a small node.
     *
     * @param totalMemoryBytes node memory as Elasticsearch sees it, or 0 if unknown
     */
    static int budgetFor(long totalMemoryBytes, long heapBytes, int processors, int threadsPerWindow,
                         int positionsPerWindow) {
        long spare = Math.max(0, totalMemoryBytes - heapBytes - RESERVED_BYTES);
        long byMemory = spare / 2 / BYTES_PER_POSITION;
        long byCpu = (long) Math.max(1, processors / threadsPerWindow) * positionsPerWindow;
        return (int) Math.max(positionsPerWindow, Math.min(byMemory, byCpu));
    }

    int capacity() {
        return capacity;
    }

    /**
     * Waits until {@code requested} positions are free and takes them. Returns the number taken,
     * which must be passed to {@link #release(int)}. Short windows take from their own budget,
     * longer ones from the main budget (all of it at most).
     */
    int acquire(int requested) {
        int taken = Math.max(1, requested);
        if (taken <= SHORT_WINDOW) {
            shortWindowPositions.acquireUninterruptibly(taken);
        } else {
            taken = Math.min(taken, capacity);
            positions.acquireUninterruptibly(taken);
        }
        return taken;
    }

    void release(int taken) {
        // A long window always takes more than SHORT_WINDOW, since the capacity is above it.
        (taken <= SHORT_WINDOW ? shortWindowPositions : positions).release(taken);
    }
}
