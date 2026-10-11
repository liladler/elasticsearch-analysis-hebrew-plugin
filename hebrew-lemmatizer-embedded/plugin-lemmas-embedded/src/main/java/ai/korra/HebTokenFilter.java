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
    private final List<String> tokenList = new ArrayList<>();
    private final List<int[]> offsetList = new ArrayList<>();
    private final List<Integer> posIncrementList = new ArrayList<>();
    private final List<String> typeList = new ArrayList<>();
    private final List<Integer> posLengthList = new ArrayList<>();
    private final List<Boolean> keywordList = new ArrayList<>();
    private final LemmatizerProvider lemmatizerProvider;
    private final Path dataPath;
    private boolean initialized = false;
    // Set once upstream has reported the end of the stream. TokenStream consumers must not be
    // read again after that, and doing so loses tokens inside delegating filters such as ES's
    // `condition` filter. Cleared by reset().
    private boolean inputExhausted;

    HebTokenFilter(TokenStream input, Path dataPath) {
        super(input);
        this.lemmatizerProvider = null;
        this.dataPath = dataPath;
    }

    HebTokenFilter(TokenStream input, LemmatizerProvider lemmatizerProvider) {
        super(input);
        this.lemmatizerProvider = lemmatizerProvider;
        this.dataPath = null;
    }

    private void initializeLemmatizer() throws IOException {
        if (lemmatizerProvider != null) {
            return;
        }
        if (!initialized) {
            try {
                debugger.debugPrint("Initializing embedded ONNX lemmatizer");
                lemmatizer = OnnxLemmatizer.getInstance(dataPath);
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
        tokenList.clear();
        offsetList.clear();
        posIncrementList.clear();
        typeList.clear();
        posLengthList.clear();
        keywordList.clear();
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

        tokenList.clear();
        offsetList.clear();
        posIncrementList.clear();
        typeList.clear();
        posLengthList.clear();
        keywordList.clear();
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

        try {
            // Keyword tokens stay in the model input so their neighbours are lemmatized with the
            // same context as before; only their own output is replaced below. Skip the model
            // entirely when there is nothing to lemmatize.
            lemmaList = allKeywords()
                    ? new ArrayList<>(tokenList)
                    : lemmatizerProvider != null
                            ? lemmatizerProvider.lemmatize(tokenList)
                            : lemmatizer.lemmatize(tokenList);
            lemmaIndex = 0;
        } catch (Exception e) {
            debugger.debugPrint("Lemmatization error: " + e.getMessage());
            lemmaList = new ArrayList<>(tokenList);
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
        tokenList.add(termAttr.toString());
        offsetList.add(new int[] { offsetAttr.startOffset(), offsetAttr.endOffset() });
        posIncrementList.add(posAttr.getPositionIncrement());
        typeList.add(typeAttr.type());
        posLengthList.add(posLenAttr.getPositionLength());
        keywordList.add(keywordAttr.isKeyword());
    }

    private boolean allKeywords() {
        for (boolean keyword : keywordList) {
            if (keyword == false) {
                return false;
            }
        }
        return true;
    }

    /** Honors keyword_marker: a token flagged as keyword is emitted unchanged, not as its lemma. */
    private void keepKeywordTokens() {
        if (lemmaList.size() != tokenList.size()) {
            return;
        }
        List<String> result = new ArrayList<>(lemmaList);
        for (int i = 0; i < result.size(); i++) {
            if (keywordList.get(i)) {
                result.set(i, tokenList.get(i));
            }
        }
        lemmaList = result;
    }

    private void produceTerm() {
        String lemma = lemmaList.get(lemmaIndex);
        int[] offsets = offsetList.get(lemmaIndex);

        clearAttributes();
        termAttr.setEmpty().append(lemma);
        offsetAttr.setOffset(offsets[0], offsets[1]);
        posAttr.setPositionIncrement(posIncrementList.get(lemmaIndex));
        typeAttr.setType(typeList.get(lemmaIndex));
        posLenAttr.setPositionLength(posLengthList.get(lemmaIndex));
        keywordAttr.setKeyword(keywordList.get(lemmaIndex));

        lemmaIndex++;
        emitExtraToken = lemmaIndex < lemmaList.size();
    }

    @FunctionalInterface
    interface LemmatizerProvider {
        List<String> lemmatize(List<String> tokens) throws Exception;
    }
}
