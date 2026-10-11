package ai.korra;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.Tokenizer;
import org.apache.lucene.analysis.core.LowerCaseFilter;
import org.apache.lucene.analysis.core.WhitespaceTokenizer;
import org.apache.lucene.analysis.miscellaneous.ConditionalTokenFilter;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.TokenFilter;
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

    @Test
    public void doesNotAskUpstreamForTokensAfterItHasEnded() throws IOException {
        Tokenizer tokenizer = new WhitespaceTokenizer();
        tokenizer.setReader(new StringReader("aa bbbb cc"));
        EndOfStreamProbe probe = new EndOfStreamProbe(tokenizer);

        try (TokenStream stream = new HebTokenFilter(probe, input -> input)) {
            stream.reset();
            while (stream.incrementToken()) {
                // drain
            }
            stream.end();
        }

        assertEquals("upstream must not be read again after it returned false", 0, probe.callsAfterEnd);
    }

    @Test
    public void keepsTokensThatBypassAConditionFilter() throws IOException {
        // Mirrors Elasticsearch's `condition` filter: only tokens of 4+ characters reach the
        // lemmatizer, shorter ones must pass through untouched and in order.
        String text = "aa bbbb cc dddd eeee ff gggg hhhh ii";

        List<TokenData> tokens = analyzeInsideCondition(text, 4, input -> upperCase(input));

        assertEquals(List.of("aa", "BBBB", "cc", "DDDD", "EEEE", "ff", "GGGG", "HHHH", "ii"), terms(tokens));
        for (TokenData token : tokens) {
            assertEquals("token " + token.term + " position increment", 1, token.positionIncrement);
        }
    }

    @Test
    public void conditionFilterBehavesLikeAStockFilterWhenNothingIsFiltered() throws IOException {
        // No token reaches the lemmatizer, so every token must come out unchanged.
        List<TokenData> tokens = analyzeInsideCondition("aa bb cc", 4, input -> upperCase(input));

        assertEquals(List.of("aa", "bb", "cc"), terms(tokens));
    }

    @Test
    public void canBeReusedForASecondInputAfterReset() throws IOException {
        // Analyzers reuse token streams: reset() must clear any end-of-input state.
        Tokenizer tokenizer = new WhitespaceTokenizer();
        TokenStream stream = new HebTokenFilter(tokenizer, input -> upperCase(input));
        CharTermAttribute termAttr = stream.addAttribute(CharTermAttribute.class);

        for (String text : List.of("aa bb", "cc dd ee")) {
            tokenizer.setReader(new StringReader(text));
            List<String> terms = new ArrayList<>();
            stream.reset();
            while (stream.incrementToken()) {
                terms.add(termAttr.toString());
            }
            stream.end();
            stream.close();
            assertEquals(upperCase(List.of(text.split(" "))), terms);
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

    private static List<TokenData> analyzeInsideCondition(String text, int minLength,
            HebTokenFilter.LemmatizerProvider lemmatizer) throws IOException {
        Tokenizer tokenizer = new WhitespaceTokenizer();
        tokenizer.setReader(new StringReader(text));

        try (TokenStream stream = new LengthCondition(tokenizer, minLength, in -> new HebTokenFilter(in, lemmatizer))) {
            CharTermAttribute termAttr = stream.addAttribute(CharTermAttribute.class);
            OffsetAttribute offsetAttr = stream.addAttribute(OffsetAttribute.class);
            PositionIncrementAttribute posAttr = stream.addAttribute(PositionIncrementAttribute.class);

            List<TokenData> tokens = new ArrayList<>();
            stream.reset();
            while (stream.incrementToken()) {
                tokens.add(new TokenData(termAttr.toString(), offsetAttr.startOffset(),
                        offsetAttr.endOffset(), posAttr.getPositionIncrement()));
            }
            stream.end();
            return tokens;
        }
    }

    private static List<String> upperCase(List<String> input) {
        List<String> out = new ArrayList<>(input.size());
        for (String term : input) {
            out.add(term.toUpperCase());
        }
        return out;
    }

    private static List<String> terms(List<TokenData> tokens) {
        List<String> out = new ArrayList<>(tokens.size());
        for (TokenData token : tokens) {
            out.add(token.term);
        }
        return out;
    }

    /** Counts how often upstream is read after it has already reported the end of the stream. */
    private static class EndOfStreamProbe extends TokenFilter {
        private boolean ended;
        private int callsAfterEnd;

        private EndOfStreamProbe(TokenStream input) {
            super(input);
        }

        @Override
        public boolean incrementToken() throws IOException {
            if (ended) {
                callsAfterEnd++;
            }
            boolean hasToken = input.incrementToken();
            if (hasToken == false) {
                ended = true;
            }
            return hasToken;
        }

        @Override
        public void reset() throws IOException {
            ended = false;
            super.reset();
        }
    }

    /** Applies the wrapped filter only to tokens with at least {@code minLength} characters. */
    private static class LengthCondition extends ConditionalTokenFilter {
        private final CharTermAttribute termAttr = addAttribute(CharTermAttribute.class);
        private final int minLength;

        private LengthCondition(TokenStream input, int minLength, Function<TokenStream, TokenStream> factory) {
            super(input, factory);
            this.minLength = minLength;
        }

        @Override
        protected boolean shouldFilter() {
            return termAttr.length() >= minLength;
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
