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
    public void invisibleMarksInsideAHebrewTokenAreRemoved() {
        // Right-to-left and left-to-right marks and joiners carry no meaning in a term.
        assertEquals("H:שלום", describe("של\u200Fום"));
        assertEquals("H:שלום", describe("שלו\u200Dם"));
        assertEquals("H:אלף|O:-7", describe("אלף\u200E-7"));
        assertEquals("H:אלף|O:-7", describe("אלף-\u200F7"));
    }

    @Test
    public void invisibleMarksInsideATokenWithoutHebrewAreLeftAlone() {
        // Only edges are trimmed there; the inside of e.g. an emoji sequence must stay intact.
        assertEquals("2\u200F0", MixedTokens.trimEdges("2\u200F0"));
    }

    @Test
    public void rebuildReplacesOnlyTheHebrewParts() {
        List<MixedTokens.Part> parts = MixedTokens.split("קריית-שמונה");

        assertEquals("קריה-שמונה", MixedTokens.rebuild(parts, List.of("קריה", "שמונה")));
    }

    @Test
    public void edgePunctuationIsTrimmedFromTokensWithoutHebrewToo() {
        // "2020," used to come back as "2020"; keeping the comma made it a different term.
        assertEquals("2020", MixedTokens.trimEdges("2020,"));
        assertEquals("2020", MixedTokens.trimEdges("2020."));
        assertEquals("2020", MixedTokens.trimEdges("(2020)"));
        assertEquals("1985", MixedTokens.trimEdges("1985;"));
        assertEquals("T-204", MixedTokens.trimEdges("T-204."));
        assertEquals("iPhone", MixedTokens.trimEdges("\"iPhone\""));
    }

    @Test
    public void trimEdgesKeepsSymbolsThatAreNotSentencePunctuation() {
        // Stripping these would change terms such as C# or 50%.
        assertEquals("C#", MixedTokens.trimEdges("C#"));
        assertEquals("50%", MixedTokens.trimEdges("50%"));
        assertEquals("@user", MixedTokens.trimEdges("@user"));
        assertEquals("AT&T", MixedTokens.trimEdges("AT&T"));
    }

    @Test
    public void aTokenOfOnlyPunctuationIsKeptRatherThanEmptied() {
        assertEquals("...", MixedTokens.trimEdges("..."));
        assertEquals("-", MixedTokens.trimEdges("-"));
        assertEquals("\u200F", MixedTokens.trimEdges("\u200F"));
    }

    @Test
    public void invisibleCharactersAtTheEdgesAreTrimmed() {
        assertEquals("H:הילדים", describe("הילדים.\u200F"));   // right-to-left mark after the period
        assertEquals("H:הילדים", describe("הילדים\u200E"));   // left-to-right mark
        assertEquals("H:הילדים", describe("הילדים\u00A0"));   // no-break space
        assertEquals("H:הילדים", describe("\u200Dהילדים\u200D")); // zero width joiner
        assertEquals("2020", MixedTokens.trimEdges("2020,\u200F"));
    }
    @Test
    public void aLoneLetterNextToDigitsIsAPrefixNotAWordForTheModel() {
        // "ה-26" and "ב-1985": a single letter glued to a number. Sent to the model on its own the
        // lemma depended on the surrounding sentence (את-26, מה-26, ...), so it is kept as written.
        assertNull(describe("ה-26"));
        assertNull(describe("ב-1985"));
        assertNull(describe("ו7"));
        assertNull(describe("ל-3/4"));
    }

    @Test
    public void aLoneLetterThatIsNotNextToDigitsIsStillAWord() {
        assertEquals("H:א|O:-|H:ב", describe("א-ב"));
        assertEquals("H:בן|O:-|H:גוריון", describe("בן-גוריון"));
    }

    @Test
    public void longerHebrewWordsNextToDigitsAreStillLemmatized() {
        assertEquals("H:אלף|O:-7", describe("אלף-7"));
        assertEquals("H:הילדים|O:-7", describe("הילדים-7"));
        assertEquals("H:קו|O:-480", describe("קו-480"));
    }

    @Test
    public void onlyTheLoneLetterIsTreatedAsAPrefix() {
        // "במאי" is a real word and stays one; it is not mistaken for a prefix.
        assertEquals("H:במאי|O:-26", describe("במאי-26"));
    }

    @Test
    public void invisibleCharactersAreDroppedEvenWhenOnlyPunctuationRemains() {
        // A token of punctuation and an invisible mark: the mark goes, the punctuation stays.
        assertEquals("\u2026", MixedTokens.trimEdges("\u2026\u200F"));
        assertEquals("\u2026", MixedTokens.trimEdges("\u200F\u2026"));
        assertEquals("...", MixedTokens.trimEdges("...\u200E"));
        assertEquals("-", MixedTokens.trimEdges("\u200F-\u200F"));
    }

    @Test
    public void aTokenOfOnlyInvisibleCharactersIsKeptRatherThanEmptied() {
        assertEquals("\u200F", MixedTokens.trimEdges("\u200F"));
        assertEquals("\u200F\u200E", MixedTokens.trimEdges("\u200F\u200E"));
    }
}
