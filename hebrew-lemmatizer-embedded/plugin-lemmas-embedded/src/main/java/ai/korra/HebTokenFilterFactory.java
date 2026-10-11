package ai.korra;

import org.apache.lucene.analysis.TokenStream;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.env.Environment;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.analysis.AbstractTokenFilterFactory;
import org.elasticsearch.index.analysis.TokenFilterFactory;

import java.nio.file.Path;

/**
 * Factory for Hebrew lemmatization token filter.
 */
public class HebTokenFilterFactory extends AbstractTokenFilterFactory {

    private final Path dataPath;

    public HebTokenFilterFactory(IndexSettings indexSettings, Environment environment, String name, Settings settings) {
        super(name);
        this.dataPath = environment.dataDirs()[0];
    }

    @Override
    public TokenStream create(TokenStream tokenStream) {
        return new HebTokenFilter(tokenStream, dataPath);
    }
}
