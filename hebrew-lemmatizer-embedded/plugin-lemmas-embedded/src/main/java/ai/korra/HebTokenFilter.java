package ai.korra;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;
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

    private boolean emitExtraToken;
    private List<String> lemmaList = new ArrayList<>();
    private int lemmaIndex;
    private final List<String> tokenList = new ArrayList<>();
    private final List<int[]> offsetList = new ArrayList<>();
    private final List<Integer> posIncrementList = new ArrayList<>();
    private final List<String> typeList = new ArrayList<>();
    private final LemmatizerProvider lemmatizerProvider;
    private boolean initialized = false;

    public HebTokenFilter(TokenStream input) {
        super(input);
        this.lemmatizerProvider = null;
    }

    HebTokenFilter(TokenStream input, LemmatizerProvider lemmatizerProvider) {
        super(input);
        this.lemmatizerProvider = lemmatizerProvider;
    }

    private void initializeLemmatizer() throws IOException {
        if (lemmatizerProvider != null) {
            return;
        }
        if (!initialized) {
            try {
                debugger.debugPrint("Initializing embedded ONNX lemmatizer");
                lemmatizer = OnnxLemmatizer.getInstance();
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
        lemmaList.clear();
        lemmaIndex = 0;
        tokenList.clear();
        offsetList.clear();
        posIncrementList.clear();
        typeList.clear();
        super.reset();
    }

    @Override
    public final boolean incrementToken() throws IOException {
        initializeLemmatizer();

        if (emitExtraToken) {
            produceTerm();
            return true;
        }

        tokenList.clear();
        offsetList.clear();
        posIncrementList.clear();
        typeList.clear();
        if (input.incrementToken()) {
            captureCurrentToken();
        } else {
            return false;
        }

        while (input.incrementToken()) {
            captureCurrentToken();
        }

        try {
            lemmaList = lemmatizerProvider != null
                    ? lemmatizerProvider.lemmatize(tokenList)
                    : lemmatizer.lemmatize(tokenList);
            lemmaIndex = 0;
        } catch (Exception e) {
            debugger.debugPrint("Lemmatization error: " + e.getMessage());
            lemmaList = new ArrayList<>(tokenList);
            lemmaIndex = 0;
        }

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
    }

    private void produceTerm() {
        String lemma = lemmaList.get(lemmaIndex);
        int[] offsets = offsetList.get(lemmaIndex);

        clearAttributes();
        termAttr.setEmpty().append(lemma);
        offsetAttr.setOffset(offsets[0], offsets[1]);
        posAttr.setPositionIncrement(posIncrementList.get(lemmaIndex));
        typeAttr.setType(typeList.get(lemmaIndex));

        lemmaIndex++;
        emitExtraToken = lemmaIndex < lemmaList.size();
    }

    @FunctionalInterface
    interface LemmatizerProvider {
        List<String> lemmatize(List<String> tokens) throws Exception;
    }
}
