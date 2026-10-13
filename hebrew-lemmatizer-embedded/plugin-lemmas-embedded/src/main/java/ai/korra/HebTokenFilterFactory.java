package ai.korra;

import org.apache.lucene.analysis.TokenStream;
import org.elasticsearch.common.settings.Settings;
import org.elasticsearch.common.util.concurrent.EsExecutors;
import org.elasticsearch.env.Environment;
import org.elasticsearch.index.IndexSettings;
import org.elasticsearch.index.analysis.AbstractTokenFilterFactory;
import org.elasticsearch.index.analysis.TokenFilterFactory;
import org.elasticsearch.monitor.jvm.JvmInfo;
import org.elasticsearch.monitor.os.OsProbe;

import java.nio.file.Path;

/**
 * Factory for Hebrew lemmatization token filter.
 */
public class HebTokenFilterFactory extends AbstractTokenFilterFactory {

    private final Path dataPath;
    private final int positionBudget;

    public HebTokenFilterFactory(IndexSettings indexSettings, Environment environment, String name, Settings settings) {
        super(name);
        this.dataPath = environment.dataDirs()[0];
        // The same node memory (honoring es.total_memory_bytes) and processor count that
        // Elasticsearch's ML nodes size their native memory and threads from.
        this.positionBudget = InferenceLimiter.budgetFor(
                OsProbe.getInstance().getAdjustedTotalMemorySize(),
                JvmInfo.jvmInfo().getMem().getHeapMax().getBytes(),
                EsExecutors.allocatedProcessors(environment.settings()),
                OnnxLemmatizer.THREADS_PER_WINDOW,
                OnnxLemmatizer.MAX_SEQUENCE_LENGTH);
    }

    @Override
    public TokenStream create(TokenStream tokenStream) {
        return new HebTokenFilter(tokenStream, dataPath, positionBudget);
    }
}
