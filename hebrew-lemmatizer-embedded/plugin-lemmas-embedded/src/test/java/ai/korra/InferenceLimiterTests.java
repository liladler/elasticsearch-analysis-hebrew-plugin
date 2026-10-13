package ai.korra;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.ThreadLocalRandom;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.junit.Test;

public class InferenceLimiterTests {

    private static final long GB = 1L << 30;
    private static final int WINDOW = OnnxLemmatizer.MAX_SEQUENCE_LENGTH;
    private static final int THREADS = OnnxLemmatizer.THREADS_PER_WINDOW;

    @Test
    public void theReportedSixGbNodeGetsAboutTwoFullWindows() {
        // 6 GB container, 3 GB heap, 8 CPUs: the node that was OOM-killed with 8 full windows.
        assertEquals(1123, InferenceLimiter.budgetFor(6 * GB, 3 * GB, 8, THREADS, WINDOW));
    }

    @Test
    public void anEightGbNodeIsLimitedByMemory() {
        assertEquals(1872, InferenceLimiter.budgetFor(8 * GB, 4 * GB, 8, THREADS, WINDOW));
    }

    @Test
    public void aSixteenGbNodeIsLimitedByItsCpus() {
        // Memory would allow 4868 positions, but 8 CPUs only run 4 windows of 2 threads at once.
        assertEquals(4 * WINDOW, InferenceLimiter.budgetFor(16 * GB, 8 * GB, 8, THREADS, WINDOW));
    }

    @Test
    public void moreCpusAllowMoreWindowsWhenMemoryAllows() {
        assertEquals(8 * WINDOW, InferenceLimiter.budgetFor(64 * GB, 31 * GB, 16, THREADS, WINDOW));
    }

    @Test
    public void aSmallNodeCanStillRunOneFullWindow() {
        assertEquals(WINDOW, InferenceLimiter.budgetFor(4 * GB, 2 * GB, 8, THREADS, WINDOW));
        assertEquals(WINDOW, InferenceLimiter.budgetFor(2 * GB, 3 * GB, 8, THREADS, WINDOW));
        assertEquals(WINDOW, InferenceLimiter.budgetFor(64 * GB, 31 * GB, 1, THREADS, WINDOW));
    }

    @Test
    public void unknownMemoryRunsOneWindowAtATime() {
        assertEquals(WINDOW, InferenceLimiter.budgetFor(0, 3 * GB, 8, THREADS, WINDOW));
    }

    @Test
    public void rejectsABudgetThatCannotTellLongWindowsFromShortOnes() {
        assertThrows(IllegalArgumentException.class, () -> new InferenceLimiter(0));
        assertThrows(IllegalArgumentException.class, () -> new InferenceLimiter(InferenceLimiter.SHORT_WINDOW));
    }

    @Test
    public void aShortWindowDoesNotWaitBehindLongOnes() throws Exception {
        InferenceLimiter limiter = new InferenceLimiter(2 * WINDOW);
        int held = limiter.acquire(2 * WINDOW);

        ExecutorService threads = Executors.newFixedThreadPool(2);
        try {
            // A long window that has to wait for the whole budget...
            AtomicReference<Thread> longWaiter = new AtomicReference<>();
            Future<?> longWindow = threads.submit(() -> {
                longWaiter.set(Thread.currentThread());
                limiter.release(limiter.acquire(WINDOW));
            });
            while (longWaiter.get() == null || longWaiter.get().getState() != Thread.State.WAITING) {
                Thread.onSpinWait();
            }

            // ...must not hold up a short query that arrives after it.
            Future<Integer> shortWindow = threads.submit(() -> limiter.acquire(6));
            assertEquals(Integer.valueOf(6), shortWindow.get(10, TimeUnit.SECONDS));
            limiter.release(6);

            limiter.release(held);
            longWindow.get(10, TimeUnit.SECONDS);
        } finally {
            threads.shutdownNow();
        }
    }

    @Test
    public void shortWindowsShareABudgetOfTheirOwn() throws Exception {
        InferenceLimiter limiter = new InferenceLimiter(2 * WINDOW);
        int perWindow = InferenceLimiter.SHORT_WINDOW;
        int fit = InferenceLimiter.SHORT_WINDOW_BUDGET / perWindow;
        for (int i = 0; i < fit; i++) {
            limiter.acquire(perWindow);
        }

        ExecutorService threads = Executors.newSingleThreadExecutor();
        try {
            Future<Integer> oneMore = threads.submit(() -> limiter.acquire(perWindow));
            assertThrows(TimeoutException.class, () -> oneMore.get(200, TimeUnit.MILLISECONDS));

            limiter.release(perWindow);
            assertEquals(Integer.valueOf(perWindow), oneMore.get(10, TimeUnit.SECONDS));
        } finally {
            threads.shutdownNow();
        }
    }

    @Test
    public void aRequestLargerThanTheBudgetTakesTheWholeBudgetInsteadOfWaitingForever() {
        InferenceLimiter limiter = new InferenceLimiter(100);

        int taken = limiter.acquire(512);

        assertEquals(100, taken);
        limiter.release(taken);
        assertEquals(100, limiter.acquire(100));
    }

    @Test
    public void concurrentWindowsNeverExceedTheBudget() throws Exception {
        int capacity = 2 * WINDOW;
        InferenceLimiter limiter = new InferenceLimiter(capacity);
        AtomicInteger inFlight = new AtomicInteger();
        AtomicInteger peak = new AtomicInteger();
        AtomicInteger shortInFlight = new AtomicInteger();
        AtomicInteger shortPeak = new AtomicInteger();

        ExecutorService threads = Executors.newFixedThreadPool(16);
        try {
            List<Future<?>> done = new ArrayList<>();
            for (int t = 0; t < 16; t++) {
                done.add(threads.submit(() -> {
                    for (int i = 0; i < 200; i++) {
                        int taken = limiter.acquire(ThreadLocalRandom.current().nextInt(1, WINDOW + 1));
                        boolean isShort = taken <= InferenceLimiter.SHORT_WINDOW;
                        AtomicInteger lane = isShort ? shortInFlight : inFlight;
                        try {
                            (isShort ? shortPeak : peak).accumulateAndGet(lane.addAndGet(taken), Math::max);
                            Thread.onSpinWait();
                        } finally {
                            lane.addAndGet(-taken);
                            limiter.release(taken);
                        }
                    }
                    return null;
                }));
            }
            for (Future<?> future : done) {
                future.get(1, TimeUnit.MINUTES);
            }
        } finally {
            threads.shutdownNow();
        }

        assertTrue("peak " + peak.get() + " exceeded " + capacity, peak.get() <= capacity);
        assertTrue("short-window peak " + shortPeak.get() + " exceeded " + InferenceLimiter.SHORT_WINDOW_BUDGET,
                shortPeak.get() <= InferenceLimiter.SHORT_WINDOW_BUDGET);
        assertEquals(0, inFlight.get());
        assertEquals(0, shortInFlight.get());
    }
}
