package ai.korra;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.KeywordAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;

/**
 * Hebrew token filter that performs lemmatization using an embedded ONNX model.
 */
public class HebTokenFilter extends TokenFilter {

    private OnnxLemmatizer lemmatizer;
    private final HebDebugger debugger = new HebDebugger();

    private final CharTermAttribute termAttr = addAttribute(CharTermAttribute.class);
    private final OffsetAttribute offsetAttr = addAttribute(OffsetAttribute.class);
    private final TypeAttribute typeAttr = addAttribute(TypeAttribute.class);
    private final PositionIncrementAttribute posAttr = addAttribute(PositionIncrementAttribute.class);
    private final PositionLengthAttribute posLenAttr = addAttribute(PositionLengthAttribute.class);
    private final KeywordAttribute keywordAttr = addAttribute(KeywordAttribute.class);

    private boolean emitExtraToken;
    private List<String> lemmaList = new ArrayList<>();
    private int lemmaIndex;
    private final List<CapturedToken> captured = new ArrayList<>();
    private final LemmatizerProvider lemmatizerProvider;
    private final Path dataPath;
    private final int positionBudget;
    private boolean initialized = false;
    // Set once upstream has reported the end of the stream. TokenStream consumers must not be
    // read again after that, and doing so loses tokens inside delegating filters such as ES's
    // `condition` filter. Cleared by reset().
    private boolean inputExhausted;

    HebTokenFilter(TokenStream input, Path dataPath, int positionBudget) {
        super(input);
        this.lemmatizerProvider = null;
        this.dataPath = dataPath;
        this.positionBudget = positionBudget;
    }

    HebTokenFilter(TokenStream input, LemmatizerProvider lemmatizerProvider) {
        super(input);
        this.lemmatizerProvider = lemmatizerProvider;
        this.dataPath = null;
        this.positionBudget = 0;
    }

    private void initializeLemmatizer() throws IOException {
        if (lemmatizerProvider != null) {
            return;
        }
        if (!initialized) {
            try {
                debugger.debugPrint("Initializing embedded ONNX lemmatizer");
                lemmatizer = OnnxLemmatizer.getInstance(dataPath, positionBudget);
                initialized = true;
                debugger.debugPrint("ONNX lemmatizer ready");
            } catch (Exception e) {
                debugger.debugPrint("Initialization failed: " + e.getMessage());
                throw new IOException("Failed to initialize embedded lemmatizer", e);
            }
        }
    }

    @Override
    public void reset() throws IOException {
        emitExtraToken = false;
        inputExhausted = false;
        lemmaList.clear();
        lemmaIndex = 0;
        captured.clear();
        super.reset();
    }

    @Override
    public final boolean incrementToken() throws IOException {
        initializeLemmatizer();

        if (emitExtraToken) {
            produceTerm();
            return true;
        }

        if (inputExhausted) {
            return false;
        }

        captured.clear();
        if (input.incrementToken()) {
            captureCurrentToken();
        } else {
            inputExhausted = true;
            return false;
        }

        while (input.incrementToken()) {
            captureCurrentToken();
        }
        inputExhausted = true;

        List<String> terms = terms();
        try {
            // Keyword tokens stay in the model input so their neighbours are lemmatized with the
            // same context as before; only their own output is replaced below. Skip the model
            // entirely when there is nothing to lemmatize.
            lemmaList = allKeywords()
                    ? new ArrayList<>(terms)
                    : lemmatizerProvider != null
                            ? lemmatizerProvider.lemmatize(terms)
                            : lemmatizer.lemmatize(terms);
            lemmaIndex = 0;
        } catch (Exception e) {
            debugger.debugPrint("Lemmatization error: " + e.getMessage());
            lemmaList = new ArrayList<>(terms);
            lemmaIndex = 0;
        }
        keepKeywordTokens();

        if (lemmaList.isEmpty()) {
            return false;
        }

        produceTerm();
        return true;
    }

    private void captureCurrentToken() {
        captured.add(new CapturedToken(termAttr.toString(), offsetAttr.startOffset(), offsetAttr.endOffset(),
                posAttr.getPositionIncrement(), typeAttr.type(), posLenAttr.getPositionLength(),
                keywordAttr.isKeyword()));
    }

    private List<String> terms() {
        List<String> terms = new ArrayList<>(captured.size());
        for (CapturedToken token : captured) {
            terms.add(token.term());
        }
        return terms;
    }

    private boolean allKeywords() {
        for (CapturedToken token : captured) {
            if (token.keyword() == false) {
                return false;
            }
        }
        return true;
    }

    /** Honors keyword_marker: a token flagged as keyword is emitted unchanged, not as its lemma. */
    private void keepKeywordTokens() {
        if (lemmaList.size() != captured.size()) {
            return;
        }
        List<String> result = new ArrayList<>(lemmaList);
        for (int i = 0; i < result.size(); i++) {
            if (captured.get(i).keyword()) {
                result.set(i, captured.get(i).term());
            }
        }
        lemmaList = result;
    }

    private void produceTerm() {
        String lemma = lemmaList.get(lemmaIndex);
        CapturedToken token = captured.get(lemmaIndex);

        clearAttributes();
        termAttr.setEmpty().append(lemma);
        offsetAttr.setOffset(token.startOffset(), token.endOffset());
        posAttr.setPositionIncrement(token.positionIncrement());
        typeAttr.setType(token.type());
        posLenAttr.setPositionLength(token.positionLength());
        keywordAttr.setKeyword(token.keyword());

        lemmaIndex++;
        emitExtraToken = lemmaIndex < lemmaList.size();
    }

    /** One token from upstream, with every attribute this filter emits again. */
    private record CapturedToken(String term, int startOffset, int endOffset, int positionIncrement,
                                 String type, int positionLength, boolean keyword) {
    }

    @FunctionalInterface
    interface LemmatizerProvider {
        List<String> lemmatize(List<String> tokens) throws Exception;
    }
}
