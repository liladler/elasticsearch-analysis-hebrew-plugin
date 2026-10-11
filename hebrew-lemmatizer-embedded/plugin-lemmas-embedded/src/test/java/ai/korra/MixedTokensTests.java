package ai.korra;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

public class MixedTokensTests {

    /** Renders parts as H:text for Hebrew and O:text for everything else, e.g. "H:אלף|O:-7". */
    private static String describe(String token) {
        List<MixedTokens.Part> parts = MixedTokens.split(token);
        if (parts == null) {
            return null;
        }
        List<String> rendered = new ArrayList<>();
        for (MixedTokens.Part part : parts) {
            rendered.add((part.hebrew ? "H:" : "O:") + part.text);
        }
        return String.join("|", rendered);
    }

    @Test
    public void hebrewWordWithANumberKeepsTheNumber() {
        assertEquals("H:אלף|O:-7", describe("אלף-7"));
        assertEquals("H:יחידה|O:-8200", describe("יחידה-8200"));
        assertEquals("H:אלף|O:7", describe("אלף7"));
    }

    @Test
    public void hyphenatedHebrewNameKeepsBothParts() {
        assertEquals("H:בן|O:-|H:גוריון", describe("בן-גוריון"));
        assertEquals("H:חיל|O:־|H:האוויר", describe("חיל־האוויר"));
    }

    @Test
    public void tokensWithoutHebrewLettersAreNotForwardedToTheModel() {
        assertNull(describe("T-204"));
        assertNull(describe("iPhone"));
        assertNull(describe("12/05/2024"));
        assertNull(describe("03-5551234"));
        assertNull(describe("8200"));
    }

    @Test
    public void punctuationAtTheEdgesIsTrimmedLikeBefore() {
        // Today "הילדים." is indexed as "ילד", not "ילד."; that must not change.
        assertEquals("H:הילדים", describe("הילדים."));
        assertEquals("H:הילדים", describe("הילדים,"));
        assertEquals("H:אלף|O:-7", describe("(אלף-7)"));
    }

    @Test
    public void gershayimAndGereshInsideAHebrewWordStayPartOfIt() {
        assertEquals("H:צה\"ל", describe("צה\"ל"));
        assertEquals("H:ג'ורג", describe("ג'ורג'"));
    }

    @Test
    public void niqqudStaysAttachedToItsWord() {
        assertEquals("H:שְׁלוֹם|O:-7", describe("שְׁלוֹם-7"));
    }

    @Test
    public void aPlainHebrewWordIsASingleHebrewPart() {
        assertEquals("H:הילדים", describe("הילדים"));
    }

    @Test
    public void rebuildReplacesOnlyTheHebrewParts() {
        List<MixedTokens.Part> parts = MixedTokens.split("קריית-שמונה");

        assertEquals("קריה-שמונה", MixedTokens.rebuild(parts, List.of("קריה", "שמונה")));
    }
}
