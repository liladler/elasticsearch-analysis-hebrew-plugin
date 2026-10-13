package ai.korra;

import java.util.ArrayList;
import java.util.List;

/**
 * Splits a value that is longer than the model's window into windows, preferring to end each one
 * at a sentence end.
 */
final class WindowPlanner {

    private WindowPlanner() {
    }

    /**
     * Splits tokens into consecutive windows of at most {@code maxPieces} word pieces. Returns
     * {@code [startInclusive, endExclusive)} token index pairs covering every token.
     *
     * <p>When a window would overflow it ends after the last sentence-ending token it contains,
     * if there is one, otherwise at the last token that still fits.
     */
    static List<int[]> plan(List<int[]> pieces, List<String> tokens, int maxPieces) {
        List<int[]> windows = new ArrayList<>();
        int start = 0;
        int count = 0;
        int lastSentenceEnd = -1;

        for (int i = 0; i < tokens.size(); i++) {
            int size = pieces.get(i).length;
            while (count + size > maxPieces && count > 0) {
                int cut = lastSentenceEnd >= start ? lastSentenceEnd + 1 : i;
                windows.add(new int[]{start, cut});
                start = cut;
                count = 0;
                lastSentenceEnd = -1;
                for (int j = start; j < i; j++) {
                    count += pieces.get(j).length;
                    if (endsSentence(tokens.get(j))) {
                        lastSentenceEnd = j;
                    }
                }
            }
            count += size;
            if (endsSentence(tokens.get(i))) {
                lastSentenceEnd = i;
            }
        }
        windows.add(new int[]{start, tokens.size()});
        return windows;
    }

    /**
     * True if the token ends a sentence: its last character is . ! ? the ellipsis or sof pasuq,
     * ignoring any closing quote, bracket or invisible mark after it (a sentence end is often
     * written as {@code ."} or {@code .)} or {@code ?"}).
     */
    static boolean endsSentence(String token) {
        int end = token.length();
        while (end > 0 && isClosingCharacter(token.charAt(end - 1))) {
            end--;
        }
        if (end == 0) {
            return false;
        }
        char last = token.charAt(end - 1);
        return last == '.' || last == '!' || last == '?' || last == '\u2026' || last == '\u05C3';
    }

    private static boolean isClosingCharacter(char c) {
        return "\"')]}\u00BB\u201D\u2019\u05F3\u05F4".indexOf(c) >= 0
                || Character.getType(c) == Character.FORMAT
                || Character.getType(c) == Character.SPACE_SEPARATOR;
    }
}
