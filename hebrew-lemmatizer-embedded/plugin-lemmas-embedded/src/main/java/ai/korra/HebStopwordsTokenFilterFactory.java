package ai.korra;

import org.apache.lucene.analysis.TokenStream;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.env.Environment;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.analysis.AbstractTokenFilterFactory;

/**
 * Factory for Hebrew stopwords token filter.
 */
public class HebStopwordsTokenFilterFactory extends AbstractTokenFilterFactory {

    public HebStopwordsTokenFilterFactory(IndexSettings indexSettings, Environment environment, String name, Settings settings) {
        super(name);
    }

    @Override
    public TokenStream create(TokenStream tokenStream) {
        return new HebStopwordsTokenFilter(tokenStream);
    }
}
