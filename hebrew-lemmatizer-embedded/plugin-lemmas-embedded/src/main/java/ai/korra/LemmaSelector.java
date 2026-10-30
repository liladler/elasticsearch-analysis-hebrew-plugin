package ai.korra;

import java.util.HashSet;
import java.util.Set;

/** Picks the lemma for a token from the model's top predictions. */
final class LemmaSelector {

    private static final Set<Character> WEAK_LETTERS = Set.of('א', 'ה', 'ו', 'י');
    private static final int HEBREW_START = 0x0590;
    private static final int HEBREW_END = 0x05FF;
    private static final int HEBREW_RANGE = HEBREW_END - HEBREW_START + 1;

    private final String[] vocab;

    LemmaSelector(String[] vocab) {
        this.vocab = vocab;
    }

    /**
     * Returns the first of the model's top predictions that shares enough significant letters
     * with the token (weak letters {@code א ה ו י} are ignored), or the token without niqqud if
     * none does.
     *
     * <p>A Hebrew letter of the candidate adds to the overlap each time it appears in it; any
     * other character only the first time.
     */
    String select(String originalToken, long[] topK) {
        return WordPieceTokenizer.withoutDoubleQuotes(choose(originalToken, topK));
    }

    private String choose(String originalToken, long[] topK) {
        boolean[] significantHebrew = new boolean[HEBREW_RANGE];
        int significantCount = 0;
        Set<Character> extraSignificant = null;

        // Compare letters in the form the model saw them (no niqqud). That form is also what we
        // return when no candidate matches: a token with niqqud left in would never match a
        // query written without vowels.
        String normalizedToken = WordPieceTokenizer.normalize(originalToken);
        for (char c : normalizedToken.toCharArray()) {
            if (WEAK_LETTERS.contains(c)) {
                continue;
            }
            if (c >= HEBREW_START && c <= HEBREW_END) {
                int idx = c - HEBREW_START;
                if (!significantHebrew[idx]) {
                    significantHebrew[idx] = true;
                    significantCount++;
                }
            } else {
                if (extraSignificant == null) {
                    extraSignificant = new HashSet<>();
                }
                if (extraSignificant.add(c)) {
                    significantCount++;
                }
            }
        }

        for (long predictedId : topK) {
            int predId = Math.toIntExact(predictedId);
            if (predId < 0 || predId >= vocab.length) {
                continue;
            }

            String candidate = vocab[predId];
            if (candidate.startsWith("[") || candidate.startsWith("##")) {
                continue;
            }

            int overlap = 0;
            int candCount = 0;
            boolean[] candHebrew = new boolean[HEBREW_RANGE];
            Set<Character> candExtra = null;

            for (char c : candidate.toCharArray()) {
                if (WEAK_LETTERS.contains(c)) {
                    continue;
                }
                if (c >= HEBREW_START && c <= HEBREW_END) {
                    int idx = c - HEBREW_START;
                    if (!candHebrew[idx]) {
                        candHebrew[idx] = true;
                        candCount++;
                    }
                    if (significantHebrew[idx]) {
                        overlap++;
                    }
                } else {
                    if (candExtra == null) {
                        candExtra = new HashSet<>();
                    }
                    if (candExtra.add(c)) {
                        candCount++;
                        if (extraSignificant != null && extraSignificant.contains(c)) {
                            overlap++;
                        }
                    }
                }
            }

            int minRequired = Math.min(2, Math.min(significantCount, candCount));
            if (overlap >= minRequired) {
                return candidate;
            }
        }

        return normalizedToken;
    }
}
