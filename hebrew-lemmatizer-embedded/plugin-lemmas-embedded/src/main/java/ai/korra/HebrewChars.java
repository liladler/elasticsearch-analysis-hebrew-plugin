package ai.korra;

/**
 * Character rules shared by token splitting ({@link MixedTokens}) and the vocabulary lookup
 * ({@link WordPieceTokenizer}). They must agree: a character one of them treats differently from
 * the other makes the same word produce different terms.
 */
final class HebrewChars {

    private HebrewChars() {
    }

    /** Hebrew letters, Yiddish digraphs and presentation forms. */
    static boolean isLetter(int codePoint) {
        return (codePoint >= 0x05D0 && codePoint <= 0x05F2) || isPresentationForm(codePoint);
    }

    /** Precomposed letters with a point, such as U+FB2A. */
    static boolean isPresentationForm(int codePoint) {
        return codePoint >= 0xFB1D && codePoint <= 0xFB4F;
    }

    /** Niqqud and cantillation marks. */
    static boolean isMark(int codePoint) {
        return codePoint >= 0x0591 && codePoint <= 0x05C7
                && Character.getType(codePoint) == Character.NON_SPACING_MARK;
    }

    /**
     * The characters Hebrew writes an acronym or loanword with: gershayim (U+05F4) and geresh
     * (U+05F3), the ASCII quote and apostrophe, and the curly quotes word processors substitute for
     * them.
     */
    static boolean isQuoteLike(int codePoint) {
        return codePoint == '׳' || codePoint == '״' || codePoint == '"' || codePoint == '\''
                || codePoint == '‘' || codePoint == '’' || codePoint == '“' || codePoint == '”';
    }

    /**
     * The quote-like marks that Hebrew writes an acronym with in the middle of a word, as opposed to
     * the single ones that mark a loanword: gershayim, the ASCII double quote and the curly ones.
     */
    static boolean isDoubleQuoteLike(int codePoint) {
        return codePoint == '״' || codePoint == '"' || codePoint == '“' || codePoint == '”';
    }

    /**
     * Whitespace (including the no-break space, which Java does not treat as whitespace) and
     * invisible format characters such as the right-to-left mark and zero width joiner. They are
     * not part of any term, but they would otherwise stay inside it and break exact matches.
     */
    static boolean isSpaceOrInvisible(int codePoint) {
        if (Character.isWhitespace(codePoint)) {
            return true;
        }
        int type = Character.getType(codePoint);
        return type == Character.SPACE_SEPARATOR || type == Character.FORMAT;
    }
}
