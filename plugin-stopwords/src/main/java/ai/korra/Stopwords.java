package ai.korra;

import java.util.Arrays;
import java.util.List;

public class Stopwords {

    private HebDebugger debugger = new HebDebugger();
    private final List<String> stopwordList = Arrays.asList("אבל", "או", "אחר", "אך", "אל", "אם", "את", "בין", "גם", "דרך", "הוא", "היה", "זאת", "זה", "יותר", "יש", "כי", "כך", "כן", "לא", "לפני", "מה", "מי", "עד", "על", "עם", "רק", "של", "שם", "אדם", "אותה", "אותו", "אותם", "אחד", "אחרי", "אין", "אלא", "אלה", "אני", "אף", "אשר", "ב", "בגלל", "בית", "בן", "דבר", "היא", "הם", "זו", "זמן", "חלק", "יום", "יכול", "ישראל", "כדי", "כל", "כמו", "ל", "לה", "להיות", "לו", "לפי", "מן", "נגד", "עוד", "פה", "שלו", "שנה", "ש", "מ", "ל", "כ", "ו", "ה", "ב", "אז", "_","אילו","אלו");

    public boolean isNotStopword(String term) {
        if (stopwordList.contains(term)) {
            String reversedText = new StringBuilder(term).reverse().toString();
            debugger.debugPrint("Heb stopwords found and removed : " + reversedText);
            return false;
        }
        return true;
    }
}
