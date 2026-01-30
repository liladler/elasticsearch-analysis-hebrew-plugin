package ai.korra;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.lang.reflect.Type;

import org.apache.lucene.analysis.TokenFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.analysis.tokenattributes.OffsetAttribute;
import org.apache.lucene.analysis.tokenattributes.TypeAttribute;
import org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

public class HebTokenFilter extends TokenFilter {

    private HebStemmer stemmer = new HebStemmer();
    private HebDebugger debugger = new HebDebugger();
    private final CharTermAttribute termAttr = addAttribute(CharTermAttribute.class);
    private final OffsetAttribute offsetAttr = addAttribute(OffsetAttribute.class);
	  private final TypeAttribute typeAttr = addAttribute(TypeAttribute.class);
    private final PositionIncrementAttribute posAttr = addAttribute(PositionIncrementAttribute.class);

    private boolean emitExtraToken;
    private List<String> lemmaList = new ArrayList<>();
    private StringBuilder concatBuilder = new StringBuilder();

    public HebTokenFilter(TokenStream input) {
        super(input);
    }

    @Override
    public void reset() throws IOException {
      emitExtraToken = false;
      super.reset();
    }

    @Override
    public boolean incrementToken() throws IOException {
      debugger.debugPrint("HebTokenFilter.incrementToken started ");
      if (emitExtraToken) {
        
        produceTerm();
        debugger.debugPrint("HebTokenFilter.incrementToken finished with emitExtraToken");
        return true;
      }

      if (input.incrementToken()) {
        concatBuilder.setLength(0);
        String term = termAttr.toString();
        concatBuilder.append(term);
        concatBuilder.append(" ");
      } else {
        return false;
      }

      while (input.incrementToken()) {
        String term = termAttr.toString();
        concatBuilder.append(term);
        concatBuilder.append(" ");
      }

      String concatString = concatBuilder.toString();
      if (concatString.endsWith(" ")) {
        concatString = concatString.substring(0, concatString.length() - 1);
      }
      //prrint concat String senteces
      String reversedText = new StringBuilder(concatString).reverse().toString();
      debugger.debugPrint("HebTokenFilter.incrementToken concated String : "+reversedText);
      String jsonString = stemmer.stem(concatString);
      Gson gson = new Gson();
      Type listType = new TypeToken<List<String>>() {}.getType();
      lemmaList = gson.fromJson(jsonString, listType);
      //lemma list print
      debugger.debugPrint("HebTokenFilter.incrementToken lemmaList : "+lemmaList);
      produceTerm();
      debugger.debugPrint("HebTokenFilter.incrementToken finished ");
      return true;

    }

    private void produceTerm() {
      debugger.debugPrint("HebTokenFilter.produceTerm called ");
      int origStart = offsetAttr.startOffset();
      int origEnd = offsetAttr.endOffset();
      String extraTerm = lemmaList.get(0);
      debugger.debugPrint("HebTokenFilter.produceTerm extraTerm: "+extraTerm);
      offsetAttr.setOffset(origStart, origEnd);
      typeAttr.setType(typeAttr.type());
      termAttr.setEmpty().append(extraTerm);
      posAttr.setPositionIncrement(1);

      lemmaList.remove(0);
      if (lemmaList.isEmpty()) {
        emitExtraToken = false;
      } else {
        emitExtraToken = true;
      }
      debugger.debugPrint("HebTokenFilter.produceTerm finished ");
    }

}