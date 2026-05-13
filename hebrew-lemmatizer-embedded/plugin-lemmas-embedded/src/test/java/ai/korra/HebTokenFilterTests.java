package ai.korra;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.junit.Test;

public class HebTokenFilterTests {

    @Test
    public void preservesOffsetsForEachEmittedLemma() throws IOException {
        List<TokenData> tokens = analyze("הילדים אוכלים", input -> List.of("ילד", "אוכל"));

        assertEquals(2, tokens.size());
        assertToken(tokens.get(0), "ילד", 0, 6, 1);
        assertToken(tokens.get(1), "אוכל", 7, 13, 1);
    }

    @Test
    public void preservesOffsetsAcrossLongTokenStreams() throws IOException {
        List<String> words = repeatedWords(300);
        String text = String.join(" ", words);

        List<TokenData> tokens = analyze(text, input -> input);

        assertEquals(words.size(), tokens.size());

        int offset = 0;
        for (int i = 0; i < words.size(); i++) {
            String word = words.get(i);
            TokenData token = tokens.get(i);
            assertEquals("token " + i + " term", word, token.term);
            assertEquals("token " + i + " start offset", offset, token.startOffset);
            assertEquals("token " + i + " end offset", offset + word.length(), token.endOffset);
            assertEquals("token " + i + " position increment", 1, token.positionIncrement);
            assertTrue("token " + i + " should not have a zero-length offset",
                    token.startOffset < token.endOffset);
            offset += word.length() + 1;
        }
    }

    private static List<TokenData> analyze(String text, HebTokenFilter.LemmatizerProvider lemmatizer)
            throws IOException {
        Tokenizer tokenizer = new WhitespaceTokenizer();
        tokenizer.setReader(new StringReader(text));

        try (TokenStream stream = new HebTokenFilter(new LowerCaseFilter(tokenizer), lemmatizer)) {
            CharTermAttribute termAttr = stream.addAttribute(CharTermAttribute.class);
            OffsetAttribute offsetAttr = stream.addAttribute(OffsetAttribute.class);
            PositionIncrementAttribute posAttr = stream.addAttribute(PositionIncrementAttribute.class);

            List<TokenData> tokens = new ArrayList<>();
            stream.reset();
            while (stream.incrementToken()) {
                tokens.add(new TokenData(
                        termAttr.toString(),
                        offsetAttr.startOffset(),
                        offsetAttr.endOffset(),
                        posAttr.getPositionIncrement()));
            }
            stream.end();
            return tokens;
        }
    }

    private static List<String> repeatedWords(int count) {
        List<String> baseWords = List.of(
                "הילדים", "אוכלים", "את", "הבננות", "בבית", "הספר",
                "המורה", "כתבה", "סיפור", "ארוך", "על", "המשפחה",
                "והחברים", "בבוקר", "הם", "הלכו", "לשוק", "וקנו",
                "לחם", "טרי", "אחר", "כך", "ישבו", "בגינה",
                "ודיברו", "על", "הספרים", "החדשים", "שקראו", "בלילה");

        List<String> words = new ArrayList<>(count);
        for (int i = 0; i < count; i++) {
            words.add(baseWords.get(i % baseWords.size()));
        }
        return words;
    }

    private static void assertToken(TokenData token, String term, int startOffset, int endOffset,
            int positionIncrement) {
        assertEquals(term, token.term);
        assertEquals(startOffset, token.startOffset);
        assertEquals(endOffset, token.endOffset);
        assertEquals(positionIncrement, token.positionIncrement);
    }

    private static class TokenData {
        private final String term;
        private final int startOffset;
        private final int endOffset;
        private final int positionIncrement;

        private TokenData(String term, int startOffset, int endOffset, int positionIncrement) {
            this.term = term;
            this.startOffset = startOffset;
            this.endOffset = endOffset;
            this.positionIncrement = positionIncrement;
        }
    }
}
