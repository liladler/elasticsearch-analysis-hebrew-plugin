package ai.korra;

import java.util.ArrayList;
import java.util.List;

/**
 * Splits a token into its Hebrew words and everything else, so that only the Hebrew words are
 * lemmatized and the rest (numbers, Latin text, hyphens between words) is kept.
 *
 * <p>Used when {@code preserve_mixed_tokens} is enabled. Without it a value such as {@code אלף-7}
 * is lemmatized as a whole and comes out as {@code אלף}: the number is silently lost, so
 * {@code אלף-7} and {@code אלף-9} become the same term.
 */
final class MixedTokens {

    static final class Part {
        final String text;
        final boolean hebrew;

        Part(String text, boolean hebrew) {
            this.text = text;
            this.hebrew = hebrew;
        }
    }

    private MixedTokens() {
    }

    /**
     * Returns the parts of the token between its leading and trailing punctuation, which is
     * dropped exactly as lemmatization drops it today ("הילדים." becomes "ילד", not "ילד.").
     * Returns {@code null} when the token contains no Hebrew letter: such a token must not be sent
     * to the model and is kept as is.
     */
    static List<Part> split(String token) {
        int start = 0;
        int end = token.length();
        while (start < end && isEdgePunctuation(token.codePointAt(start))) {
            start += Character.charCount(token.codePointAt(start));
        }
        while (end > start && isEdgePunctuation(token.codePointBefore(end))) {
            end -= Character.charCount(token.codePointBefore(end));
        }
        if (start >= end) {
            return null;
        }
        String core = token.substring(start, end);

        List<Part> parts = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean currentHebrew = false;
        boolean previousWasHebrew = false;
        boolean anyHebrewLetter = false;

        for (int i = 0; i < core.length(); ) {
            int codePoint = core.codePointAt(i);
            int next = i + Character.charCount(codePoint);
            boolean hebrew = isHebrewLetter(codePoint) || isHebrewMark(codePoint)
                    || (isEmbeddedQuote(codePoint) && previousWasHebrew
                            && next < core.length() && isHebrewLetter(core.codePointAt(next)));
            if (isHebrewLetter(codePoint)) {
                anyHebrewLetter = true;
            }

            if (current.length() > 0 && hebrew != currentHebrew) {
                parts.add(new Part(current.toString(), currentHebrew));
                current.setLength(0);
            }
            current.appendCodePoint(codePoint);
            currentHebrew = hebrew;
            previousWasHebrew = hebrew;
            i = next;
        }
        if (current.length() > 0) {
            parts.add(new Part(current.toString(), currentHebrew));
        }
        return anyHebrewLetter ? parts : null;
    }

    /** Joins the parts back together, replacing each Hebrew part by the next lemma. */
    static String rebuild(List<Part> parts, List<String> lemmas) {
        StringBuilder result = new StringBuilder();
        int lemmaIndex = 0;
        for (Part part : parts) {
            result.append(part.hebrew ? lemmas.get(lemmaIndex++) : part.text);
        }
        return result.toString();
    }

    static int hebrewPartCount(List<Part> parts) {
        int count = 0;
        for (Part part : parts) {
            if (part.hebrew) {
                count++;
            }
        }
        return count;
    }

    private static boolean isHebrewLetter(int codePoint) {
        return (codePoint >= 0x05D0 && codePoint <= 0x05F2) // letters and Yiddish digraphs
                || (codePoint >= 0xFB1D && codePoint <= 0xFB4F); // presentation forms
    }

    private static boolean isHebrewMark(int codePoint) {
        return codePoint >= 0x0591 && codePoint <= 0x05C7
                && Character.getType(codePoint) == Character.NON_SPACING_MARK; // niqqud, cantillation
    }

    private static boolean isEmbeddedQuote(int codePoint) {
        return codePoint == '\'' || codePoint == '"' || codePoint == '’'
                || codePoint == '׳' || codePoint == '״';
    }

    private static boolean isEdgePunctuation(int codePoint) {
        if (Character.isWhitespace(codePoint)) {
            return true;
        }
        switch (Character.getType(codePoint)) {
            case Character.CONNECTOR_PUNCTUATION:
            case Character.DASH_PUNCTUATION:
            case Character.START_PUNCTUATION:
            case Character.END_PUNCTUATION:
            case Character.INITIAL_QUOTE_PUNCTUATION:
            case Character.FINAL_QUOTE_PUNCTUATION:
            case Character.OTHER_PUNCTUATION:
                return true;
            default:
                return false;
        }
    }
}
