package ai.korra;

import org.apache.lucene.analysis.TokenStream;
import org.elasticsearch.plugin.analysis.TokenFilterFactory;
import org.elasticsearch.plugin.NamedComponent;

@NamedComponent(value = "heb_stopwords")
public class HebTokenFilterFactory implements TokenFilterFactory {

    @Override
    public TokenStream create(TokenStream tokenStream) {
        return new HebTokenFilter(tokenStream);
    }

}