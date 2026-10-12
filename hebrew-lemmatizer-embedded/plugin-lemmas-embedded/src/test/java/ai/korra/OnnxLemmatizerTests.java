package ai.korra;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class OnnxLemmatizerTests {

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void acceptsInt64TopThreeOutput() throws OrtException {
        OnnxLemmatizer.validateModelOutputContract(OnnxJavaType.INT64, new long[]{-1, -1, 3});
    }

    @Test
    public void rejectsFullLogitsOutput() {
        OrtException exception = assertThrows(OrtException.class,
                () -> OnnxLemmatizer.validateModelOutputContract(
                        OnnxJavaType.FLOAT, new long[]{-1, -1, 128000}));

        assertTrue(exception.getMessage().contains("expected INT64 [batch, sequence, 3]"));
    }

    @Test
    public void rejectsWrongTopThreeElementType() {
        assertThrows(OrtException.class,
                () -> OnnxLemmatizer.validateModelOutputContract(
                        OnnxJavaType.FLOAT, new long[]{-1, -1, 3}));
    }

    @Test
    public void removesOnlyStaleCacheEntries() throws Exception {
        Path cacheDir = temporaryFolder.newFolder("heb-lemmatizer").toPath();
        Path activeDir = Files.createDirectory(cacheDir.resolve("active-key"));
        Path activeModel = Files.writeString(activeDir.resolve("model.onnx"), "active");

        Path staleDir = Files.createDirectories(cacheDir.resolve("stale-key/nested"));
        Files.writeString(staleDir.resolve("model.onnx"), "stale");
        Path legacyModel = Files.writeString(cacheDir.resolve("model.onnx"), "legacy");

        OnnxLemmatizer.cleanupStaleCacheEntries(cacheDir, activeDir);

        assertTrue(Files.exists(activeDir));
        assertTrue(Files.exists(activeModel));
        assertFalse(Files.exists(cacheDir.resolve("stale-key")));
        assertFalse(Files.exists(legacyModel));
    }

    @Test
    public void lemmatizesVowelizedWordsLikePlainWords() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data").toPath());

        List<String> plain = lemmatizer.lemmatize(List.of("הילדים", "אוכלים"));
        List<String> vowelized = lemmatizer.lemmatize(List.of("הַיְלָדִים", "אוֹכְלִים"));

        assertEquals(List.of("ילד", "אוכל"), plain);
        assertEquals("niqqud must not change the lemma", plain, vowelized);
    }

    @Test
    public void latinIdentifiersAreNotAlteredByNiqqudHandling() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data11").toPath());

        // These came back as "t" and "f" when the normalization also lowercased Latin text.
        List<String> identifiers = List.of("T-204", "F-16", "F-16I");

        assertEquals(identifiers, lemmatizer.lemmatize(identifiers));
    }

    @Test
    public void aTokenMadeOnlyOfMarksDoesNotShiftTheFollowingLemmas() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data2").toPath());

        List<String> lemmas = lemmatizer.lemmatize(List.of("\u05B0\u05B1", "הילדים"));

        assertEquals(List.of("\u05B0\u05B1", "ילד"), lemmas);
    }

    @Test
    public void lemmatizesValuesLongerThanTheModelWindow() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data3").toPath());
        // 700 words is well past the 512-position limit of the model.
        List<String> tokens = Collections.nCopies(700, "הילדים");

        List<String> lemmas = lemmatizer.lemmatize(tokens);

        assertEquals(700, lemmas.size());
        assertEquals("every token must be lemmatized, not left as is", Collections.nCopies(700, "ילד"), lemmas);
    }

    @Test
    public void valuesThatFitInOneWindowAreStillLemmatizedTogether() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data4").toPath());
        List<String> tokens = new ArrayList<>(Collections.nCopies(510, "הילדים"));

        assertEquals(Collections.nCopies(510, "ילד"), lemmatizer.lemmatize(tokens));
    }

    @Test
    public void shortInputsAreNotSplitAtSentenceEnds() {
        // Three sentences that all fit: one window, exactly like before this change.
        List<String> tokens = List.of("אחד.", "שניים!", "שלושה?");
        List<int[]> pieces = List.of(new int[1], new int[1], new int[1]);

        assertWindows(List.of(new int[]{0, 3}), OnnxLemmatizer.planWindows(pieces, tokens, 510));
    }

    @Test
    public void overflowSplitsAtTheLastSentenceEnd() {
        // 4 tokens of 3 pieces each, limit 10: the 4th does not fit, so the window ends after the
        // sentence-ending token (index 1) instead of mid-sentence.
        List<String> tokens = List.of("a", "b.", "c", "d");
        List<int[]> pieces = List.of(new int[3], new int[3], new int[3], new int[3]);

        assertWindows(List.of(new int[]{0, 2}, new int[]{2, 4}), OnnxLemmatizer.planWindows(pieces, tokens, 10));
    }

    @Test
    public void overflowWithoutAnySentenceEndSplitsAtATokenBoundary() {
        List<String> tokens = List.of("a", "b", "c", "d", "e");
        List<int[]> pieces = List.of(new int[4], new int[4], new int[4], new int[4], new int[4]);

        assertWindows(List.of(new int[]{0, 2}, new int[]{2, 4}, new int[]{4, 5}),
                OnnxLemmatizer.planWindows(pieces, tokens, 10));
    }

    @Test
    public void everyPlannedWindowRespectsTheLimitAndCoversAllTokens() {
        List<String> tokens = new ArrayList<>();
        List<int[]> pieces = new ArrayList<>();
        for (int i = 0; i < 2000; i++) {
            tokens.add(i % 17 == 0 ? "word." : "word");
            pieces.add(new int[1 + (i % 5)]);
        }

        List<int[]> windows = OnnxLemmatizer.planWindows(pieces, tokens, 510);

        int expectedStart = 0;
        for (int[] window : windows) {
            assertEquals("windows must be contiguous", expectedStart, window[0]);
            int total = 0;
            for (int i = window[0]; i < window[1]; i++) {
                total += pieces.get(i).length;
            }
            assertTrue("window of " + total + " pieces exceeds the limit", total <= 510);
            expectedStart = window[1];
        }
        assertEquals("every token must be in a window", tokens.size(), expectedStart);
    }

    @Test
    public void mixedTokensKeepTheirNumbers() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data5").toPath());

        assertEquals(List.of("אלף-7"), lemmatizer.lemmatize(List.of("אלף-7")));
        assertEquals(List.of("יחידה-8200"), lemmatizer.lemmatize(List.of("יחידה-8200")));
        assertEquals(List.of("בן-גוריון"), lemmatizer.lemmatize(List.of("בן-גוריון")));
        assertEquals(List.of("אלף-7"), lemmatizer.lemmatize(List.of("(אלף-7)")));
    }

    @Test
    public void hebrewPartsOfAMixedTokenAreStillLemmatized() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data6").toPath());

        // "הילדים" -> "ילד" even when glued to a number.
        assertEquals(List.of("ילד-7"), lemmatizer.lemmatize(List.of("הילדים-7")));
    }

    @Test
    public void tokensWithoutHebrewAreReturnedUntouched() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data7").toPath());

        List<String> tokens = List.of("iPhone", "T-204", "12/05/2024", "03-5551234");

        assertEquals(tokens, lemmatizer.lemmatize(tokens));
    }

    @Test
    public void plainHebrewWithEdgePunctuationIsLemmatized() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data8").toPath());

        // Punctuation at the edges of a token is dropped, as it always was.
        assertEquals(List.of("ילד", "אוכל", "בננה"),
                lemmatizer.lemmatize(List.of("הילדים,", "אוכלים", "הבננות.")));
    }

    @Test
    public void neighboursOfAMixedTokenAreLemmatizedAsUsual() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data9").toPath());

        assertEquals(List.of("ילד", "אלף-7", "אוכל"), lemmatizer.lemmatize(List.of("הילדים", "אלף-7", "אוכלים")));
    }

    @Test
    public void edgePunctuationIsDroppedFromNumbersAndLatinTokens() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data12").toPath());

        assertEquals(List.of("2020", "2020", "2020", "T-204", "C#"),
                lemmatizer.lemmatize(List.of("2020,", "2020.", "(2020)", "T-204.", "C#")));
    }

    @Test
    public void invisibleMarksAroundAHebrewWordAreDropped() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data13").toPath());

        assertEquals(List.of("ילד", "ילד", "ילד"),
                lemmatizer.lemmatize(List.of("הילדים.\u200F", "הילדים\u00A0", "הילדים\u200E")));
    }

    @Test
    public void theSameLetterAndNumberGivesTheSameTermInAnyContext() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data14").toPath());

        // "ה-26" used to come out as ה-26, את-26, מה-26 or וה-26 depending on its neighbours.
        for (List<String> context : List.of(
                List.of("ה-26", "במאי"),
                List.of("בתאריך", "ה-26"),
                List.of("הוא", "אמר", "ה-26", "היום"),
                List.of("את", "ה-26"))) {
            List<String> lemmas = lemmatizer.lemmatize(context);
            assertEquals("ה-26 in " + context, "ה-26", lemmas.get(context.indexOf("ה-26")));
        }
        assertEquals(List.of("ב-1985"), lemmatizer.lemmatize(List.of("ב-1985")));
    }

    @Test
    public void aVowelizedTokenTheModelCannotMatchIsReturnedWithoutItsMarks() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data15").toPath());

        // When no candidate matches, the token itself is returned. It must not carry niqqud,
        // otherwise a query written without vowels can never match it.
        for (String token : List.of("וַיִּקְרָא", "וַיַּבְדֵּל", "מְרַחֶפֶת", "בְּרֵאשִׁית")) {
            String term = lemmatizer.lemmatize(List.of(token)).get(0);
            assertEquals(token + " -> " + term, -1, firstHebrewMark(term));
        }
    }

    private static int firstHebrewMark(String text) {
        for (int i = 0; i < text.length(); i++) {
            char c = text.charAt(i);
            if (c >= 0x0591 && c <= 0x05C7 && c != 0x05BE && c != 0x05C0 && c != 0x05C3 && c != 0x05C6) {
                return i;
            }
        }
        return -1;
    }

    @Test
    public void sentenceEndsFollowedByAClosingQuoteOrBracketStillEndTheSentence() {
        // 4 tokens of 3 pieces, limit 10: the window must end after the sentence end at index 1,
        // whatever closing character follows the punctuation.
        for (String end : List.of(".", "!", "?", "\u2026", "\u05C3", ".\"", ".)", "?\"", "!)", ".\u201D", ".\u200F")) {
            List<String> tokens = List.of("a", "b" + end, "c", "d");
            List<int[]> pieces = List.of(new int[3], new int[3], new int[3], new int[3]);

            assertWindows(List.of(new int[]{0, 2}, new int[]{2, 4}), OnnxLemmatizer.planWindows(pieces, tokens, 10));
        }
    }

    @Test
    public void aTokenThatMerelyContainsPunctuationIsNotASentenceEnd() {
        for (String token : List.of("3.5", "e.g", "a,b", "\"", ")", "...x")) {
            List<String> tokens = List.of("a", token, "c", "d");
            List<int[]> pieces = List.of(new int[4], new int[4], new int[4], new int[4]);

            // no sentence end anywhere: plain token-boundary split, 10 pieces per window
            assertWindows(List.of(new int[]{0, 2}, new int[]{2, 4}), OnnxLemmatizer.planWindows(pieces, tokens, 10));
        }
    }

    private static void assertWindows(List<int[]> expected, List<int[]> actual) {
        assertEquals("window count", expected.size(), actual.size());
        for (int i = 0; i < expected.size(); i++) {
            assertEquals("window " + i + " start", expected.get(i)[0], actual.get(i)[0]);
            assertEquals("window " + i + " end", expected.get(i)[1], actual.get(i)[1]);
        }
    }
}
