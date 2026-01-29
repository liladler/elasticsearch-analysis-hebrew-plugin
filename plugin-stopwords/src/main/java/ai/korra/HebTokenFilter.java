package ai.korra;

import java.io.IOException;

import org.apache.lucene.analysis.FilteringTokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;

public class HebTokenFilter extends FilteringTokenFilter {

    private Stopwords objStopwords = new Stopwords();

    private final CharTermAttribute termAttr = addAttribute(CharTermAttribute.class);

    public HebTokenFilter(TokenStream input) {
        super(input);
    }

    @Override
    public boolean accept() throws IOException {

      String term = termAttr.toString();
      return objStopwords.isNotStopword(term);

    }

}