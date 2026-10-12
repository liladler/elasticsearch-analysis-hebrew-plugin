package ai.korra;

import java.util.ArrayList;
import java.util.List;

/**
 * Splits a token into its Hebrew words and everything else, so that only the Hebrew words are
 * lemmatized and the rest (numbers, Latin text, hyphens between words) is kept.
 *
 * <p>Lemmatizing a whole token such as {@code אלף-7} keeps only the first Hebrew word, so it
 * comes out as {@code אלף}: the number is silently lost and {@code אלף-7} and {@code אלף-9}
 * become the same term.
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

    /** Sentence punctuation, quotes and brackets that may surround a token without being part of it. */
    private static final String SAFE_EDGE_CHARS = ".,;:!?\"'()[]{}\u00AB\u00BB\u201C\u201D\u201E\u2018\u2019\u201A"
            + "\u2026\u2013\u2014\u2015-\u05BE\u05C3\u05F3\u05F4";

    /**
     * Drops sentence punctuation, quotes, brackets and invisible characters from both ends of a
     * token that contains no Hebrew letter, so "2020," is the same term as "2020". Symbols that can
     * be part of a term ("C#", "50%", "@user", "AT&T") are left alone. If nothing would remain the
     * token is returned unchanged.
     */
    static String trimEdges(String token) {
        int start = 0;
        int end = token.length();
        while (start < end && isSafeEdge(token.codePointAt(start))) {
            start += Character.charCount(token.codePointAt(start));
        }
        while (end > start && isSafeEdge(token.codePointBefore(end))) {
            end -= Character.charCount(token.codePointBefore(end));
        }
        return start >= end ? token : token.substring(start, end);
    }

    private static boolean isSafeEdge(int codePoint) {
        return isSpaceOrInvisible(codePoint) || SAFE_EDGE_CHARS.indexOf(codePoint) >= 0;
    }

    /**
     * Whitespace (including the no-break space, which Java does not treat as whitespace) and
     * invisible format characters such as the right-to-left mark and zero width joiner. They are
     * not part of any term, but they would otherwise stay inside it and break exact matches.
     */
    private static boolean isSpaceOrInvisible(int codePoint) {
        if (Character.isWhitespace(codePoint)) {
            return true;
        }
        int type = Character.getType(codePoint);
        return type == Character.SPACE_SEPARATOR || type == Character.FORMAT;
    }

    private static boolean isEdgePunctuation(int codePoint) {
        if (isSpaceOrInvisible(codePoint)) {
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
