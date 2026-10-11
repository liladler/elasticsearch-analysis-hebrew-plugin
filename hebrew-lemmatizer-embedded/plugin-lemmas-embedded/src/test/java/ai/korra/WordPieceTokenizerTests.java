package ai.korra;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertSame;

import java.io.IOException;
import java.io.InputStream;

import org.junit.BeforeClass;
import org.junit.Test;

/**
 * The model's own tokenizer.json normalizes text with NFD, Lowercase and StripAccents before it
 * looks words up. These tests pin the Java port to the same behaviour.
 */
public class WordPieceTokenizerTests {

    private static WordPieceTokenizer tokenizer;
    private static int unkId;

    @BeforeClass
    public static void loadTokenizer() throws IOException {
        try (InputStream stream = WordPieceTokenizerTests.class.getClassLoader()
                .getResourceAsStream("model/tokenizer.json")) {
            tokenizer = new WordPieceTokenizer(stream);
        }
        unkId = tokenizer.getTokenId("[UNK]");
    }

    @Test
    public void vowelizedWordEncodesLikeThePlainWord() {
        int[] plain = tokenizer.encode("שלום");

        assertArrayEquals(plain, tokenizer.encode("שְׁלוֹם"));
        assertNotEquals("plain word should be in the vocabulary", unkId, plain[0]);
    }

    @Test
    public void anotherVowelizedWordEncodesLikeThePlainWord() {
        assertArrayEquals(tokenizer.encode("ספר"), tokenizer.encode("סֵפֶר"));
    }

    @Test
    public void cantillationMarksAreIgnored() {
        assertArrayEquals(tokenizer.encode("בראשית"), tokenizer.encode("בְּרֵאשִׁ֖ית"));
    }

    @Test
    public void presentationFormsAreDecomposed() {
        // U+FB2A is a precomposed shin with a shin dot.
        assertArrayEquals(tokenizer.encode("שלום"), tokenizer.encode("שׁלום"));
    }

    @Test
    public void latinTextIsLeftAloneSoExistingTermsDoNotChange() {
        // Lowercasing or stripping accents here would change how identifiers such as T-204 are
        // lemmatized for every existing index.
        for (String token : new String[]{"T-204", "F-16I", "iPhone", "Café", "12/05/2024"}) {
            assertSame(token, WordPieceTokenizer.normalize(token));
        }
    }

    @Test
    public void plainHebrewIsLeftAlone() {
        String word = "הילדים";

        assertSame("plain Hebrew must not be rewritten", word, WordPieceTokenizer.normalize(word));
    }

    @Test
    public void aTokenMadeOnlyOfMarksEncodesToNothing() {
        assertEquals(0, tokenizer.encode("ְֱ").length);
    }
}
