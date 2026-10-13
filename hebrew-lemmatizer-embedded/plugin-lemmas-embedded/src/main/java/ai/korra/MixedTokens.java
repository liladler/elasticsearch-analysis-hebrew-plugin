package ai.korra;

import java.util.ArrayList;
import java.util.List;
import java.util.function.IntPredicate;

/**
 * Splits a token into its Hebrew words and everything else, so that only the Hebrew words are
 * lemmatized and the rest (numbers, Latin text, hyphens between words) is kept.
 *
 * <p>Lemmatizing a whole token such as {@code אלף-7} keeps only the first Hebrew word, so it
 * comes out as {@code אלף}: the number is silently lost and {@code אלף-7} and {@code אלף-9}
 * become the same term.
 */
final class MixedTokens {

    record Part(String text, boolean hebrew) {
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
        String core = trim(token, MixedTokens::isEdgePunctuation);
        if (core.isEmpty()) {
            return null;
        }
        if (core.codePoints().anyMatch(HebrewChars::isLetter)) {
            // Right-to-left and left-to-right marks, joiners and the like are invisible and mean
            // nothing in a term, but inside a word they split it for the model and break exact
            // matches, so a word with one stray mark never matched the same word without it.
            core = removeInvisibleCharacters(core);
        }

        List<Part> parts = new ArrayList<>();
        StringBuilder current = new StringBuilder();
        boolean currentHebrew = false;
        boolean previousWasHebrew = false;
        boolean anyHebrewLetter = false;

        for (int i = 0; i < core.length(); ) {
            int codePoint = core.codePointAt(i);
            int next = i + Character.charCount(codePoint);
            boolean hebrew = HebrewChars.isLetter(codePoint) || HebrewChars.isMark(codePoint)
                    || (HebrewChars.isQuoteLike(codePoint) && previousWasHebrew
                            && next < core.length() && HebrewChars.isLetter(core.codePointAt(next)));
            if (HebrewChars.isLetter(codePoint)) {
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
        return anyHebrewLetter ? treatLoneLettersNextToDigitsAsText(parts) : null;
    }

    /**
     * A single Hebrew letter next to a number ("ה-26", "ב-1985") is a prefix, not a word. Sent to
     * the model on its own, with nothing to go on, its lemma depended on the sentence around it
     * ("את-26", "מה-26", "ה-26" for the same token), so the same value produced different terms.
     * Such a letter is kept as written; if no real Hebrew word is left the token skips the model.
     */
    private static List<Part> treatLoneLettersNextToDigitsAsText(List<Part> parts) {
        List<Part> result = new ArrayList<>(parts.size());
        boolean anyHebrew = false;
        for (int i = 0; i < parts.size(); i++) {
            Part part = parts.get(i);
            boolean lone = part.hebrew() && letterCount(part.text()) == 1
                    && (hasDigit(parts, i - 1) || hasDigit(parts, i + 1));
            if (lone) {
                part = new Part(part.text(), false);
            }
            anyHebrew |= part.hebrew();
            Part previous = result.isEmpty() ? null : result.get(result.size() - 1);
            if (previous != null && previous.hebrew() == false && part.hebrew() == false) {
                result.set(result.size() - 1, new Part(previous.text() + part.text(), false));
            } else {
                result.add(part);
            }
        }
        return anyHebrew ? result : null;
    }

    private static boolean hasDigit(List<Part> parts, int index) {
        if (index < 0 || index >= parts.size() || parts.get(index).hebrew()) {
            return false;
        }
        return parts.get(index).text().codePoints().anyMatch(Character::isDigit);
    }

    private static int letterCount(String text) {
        return (int) text.codePoints().filter(HebrewChars::isLetter).count();
    }

    /** Joins the parts back together, replacing each Hebrew part by the next lemma. */
    static String rebuild(List<Part> parts, List<String> lemmas) {
        StringBuilder result = new StringBuilder();
        int lemmaIndex = 0;
        for (Part part : parts) {
            result.append(part.hebrew() ? lemmas.get(lemmaIndex++) : part.text());
        }
        return result.toString();
    }

    static int hebrewPartCount(List<Part> parts) {
        int count = 0;
        for (Part part : parts) {
            if (part.hebrew()) {
                count++;
            }
        }
        return count;
    }

    private static String removeInvisibleCharacters(String text) {
        StringBuilder builder = new StringBuilder(text.length());
        text.codePoints().filter(codePoint -> Character.getType(codePoint) != Character.FORMAT)
                .forEach(builder::appendCodePoint);
        return builder.toString();
    }

    /** Sentence punctuation, quotes and brackets that may surround a token without being part of it. */
    private static final String SAFE_EDGE_CHARS = ".,;:!?\"'()[]{}\u00AB\u00BB\u201C\u201D\u201E\u2018\u2019\u201A"
            + "\u2026\u2013\u2014\u2015-\u05BE\u05C3\u05F3\u05F4";

    /**
     * Drops sentence punctuation, quotes, brackets and invisible characters from both ends of a
     * token that contains no Hebrew letter, so "2020," is the same term as "2020". Symbols that can
     * be part of a term ("C#", "50%", "@user", "AT&T") are left alone. A token made only of
     * punctuation keeps its punctuation (an invisible mark next to it is still dropped), and a
     * token made only of invisible marks is returned unchanged rather than emptied.
     */
    static String trimEdges(String token) {
        String trimmed = trim(token, MixedTokens::isSafeEdge);
        if (trimmed.isEmpty() == false) {
            return trimmed;
        }
        // Only punctuation and invisible marks were left. Keep the punctuation, but an invisible
        // mark is never part of a term, so it is dropped here as well ("\u2026" + RLM -> "\u2026").
        String withoutInvisibles = trim(token, HebrewChars::isSpaceOrInvisible);
        return withoutInvisibles.isEmpty() ? token : withoutInvisibles;
    }

    /** Removes the characters matching {@code trimmable} from both ends of the token. */
    private static String trim(String token, IntPredicate trimmable) {
        int start = 0;
        int end = token.length();
        while (start < end && trimmable.test(token.codePointAt(start))) {
            start += Character.charCount(token.codePointAt(start));
        }
        while (end > start && trimmable.test(token.codePointBefore(end))) {
            end -= Character.charCount(token.codePointBefore(end));
        }
        return token.substring(start, end);
    }

    private static boolean isSafeEdge(int codePoint) {
        return HebrewChars.isSpaceOrInvisible(codePoint) || SAFE_EDGE_CHARS.indexOf(codePoint) >= 0;
    }

    private static boolean isEdgePunctuation(int codePoint) {
        if (HebrewChars.isSpaceOrInvisible(codePoint)) {
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
