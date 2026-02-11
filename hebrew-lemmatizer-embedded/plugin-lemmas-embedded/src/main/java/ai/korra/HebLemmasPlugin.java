package ai.korra;

import org.elasticsearch.index.analysis.TokenFilterFactory;
import org.elasticsearch.indices.analysis.AnalysisModule;
import org.elasticsearch.plugins.AnalysisPlugin;
import org.elasticsearch.plugins.Plugin;

import java.util.Map;

/**
 * Classic plugin entry point for Hebrew lemmatizer.
 */
public class HebLemmasPlugin extends Plugin implements AnalysisPlugin {

    @Override
    public Map<String, AnalysisModule.AnalysisProvider<TokenFilterFactory>> getTokenFilters() {
        return Map.of(
                "heb_lemmas", HebTokenFilterFactory::new,
                "heb_stopwords", HebStopwordsTokenFilterFactory::new
        );
    }
}
