package ai.korra;

public class HebDebugger {


    private boolean debugEnabled;

    // Constructor checks the environment variable
    public HebDebugger() {
        String debugEnv = System.getenv("KORRA_HEB_DEBUG");
        this.debugEnabled = "TRUE".equalsIgnoreCase(debugEnv);
    }

    // Method to print a message if debugging is enabled
    public void debugPrint(String message) {
        if (this.debugEnabled) {
            System.out.println("\u200E" + message + "\u200E");
        }
    }

}