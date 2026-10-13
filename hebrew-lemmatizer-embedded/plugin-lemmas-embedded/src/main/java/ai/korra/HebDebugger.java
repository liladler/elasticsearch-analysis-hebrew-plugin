package ai.korra;

/**
 * Debug utility for Hebrew lemmatizer plugin.
 * Enable by setting environment variable: KORRA_HEB_DEBUG=TRUE
 */
public final class HebDebugger {

    private static final boolean DEBUG_ENABLED;

    static {
        String debugEnv = System.getenv("KORRA_HEB_DEBUG");
        DEBUG_ENABLED = "TRUE".equalsIgnoreCase(debugEnv);
    }

    private HebDebugger() {
    }

    public static boolean isDebugEnabled() {
        return DEBUG_ENABLED;
    }

    public static void log(String message) {
        if (DEBUG_ENABLED) {
            System.out.println("[HebLemmatizer] " + message);
        }
    }
}
