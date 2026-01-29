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
    private final List<String> tokenList = new ArrayList<>();
    private boolean initialized = false;

    public HebTokenFilter(TokenStream input) {
        super(input);
    }

    private void initializeLemmatizer() throws IOException {
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
        tokenList.clear();
        super.reset();
    }

    @Override
    public boolean incrementToken() throws IOException {
        initializeLemmatizer();

        if (emitExtraToken) {
            produceTerm();
            return true;
        }

        tokenList.clear();
        if (input.incrementToken()) {
            tokenList.add(termAttr.toString());
        } else {
            return false;
        }

        while (input.incrementToken()) {
            tokenList.add(termAttr.toString());
        }

        try {
            lemmaList = lemmatizer.lemmatize(tokenList);
        } catch (Exception e) {
            debugger.debugPrint("Lemmatization error: " + e.getMessage());
            lemmaList = new ArrayList<>(tokenList);
        }

        if (lemmaList.isEmpty()) {
            return false;
        }

        produceTerm();
        return true;
    }

    private void produceTerm() {
        int origStart = offsetAttr.startOffset();
        int origEnd = offsetAttr.endOffset();
        String lemma = lemmaList.get(0);

        offsetAttr.setOffset(origStart, origEnd);
        typeAttr.setType(typeAttr.type());
        termAttr.setEmpty().append(lemma);
        posAttr.setPositionIncrement(1);

        lemmaList.remove(0);
        emitExtraToken = !lemmaList.isEmpty();
    }
}
