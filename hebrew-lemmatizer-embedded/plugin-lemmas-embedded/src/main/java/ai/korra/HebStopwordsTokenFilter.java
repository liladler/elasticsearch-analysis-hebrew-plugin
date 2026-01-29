package ai.korra;

import java.io.IOException;

import org.apache.lucene.analysis.FilteringTokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

/**
 * Hebrew stopwords filter.
 */
public class HebStopwordsTokenFilter extends FilteringTokenFilter {

    private final Stopwords stopwords = new Stopwords();
    private final CharTermAttribute termAttr = addAttribute(CharTermAttribute.class);

    public HebStopwordsTokenFilter(TokenStream input) {
        super(input);
    }

    @Override
    public boolean accept() throws IOException {
        String term = termAttr.toString();
        return stopwords.isNotStopword(term);
    }
}
