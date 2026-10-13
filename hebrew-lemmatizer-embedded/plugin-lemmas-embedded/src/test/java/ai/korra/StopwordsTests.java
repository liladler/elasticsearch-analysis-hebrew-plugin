package ai.korra;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

public class StopwordsTests {

    private final Stopwords stopwords = new Stopwords();

    @Test
    public void functionWordsAreStillRemoved() {
        for (String word : new String[]{"את", "של", "על", "עם", "לא", "הוא", "כי"}) {
            assertFalse(word + " should be a stopword", stopwords.isNotStopword(word));
        }
    }

    @Test
    public void theWordForMoreIsKept() {
        // "עוד" is also what the acronym for a lawyer (עו"ד) becomes once its quote mark is dropped,
        // so removing it as a stopword deleted every lawyer from the index.
        assertTrue(stopwords.isNotStopword("עוד"));
    }
}
