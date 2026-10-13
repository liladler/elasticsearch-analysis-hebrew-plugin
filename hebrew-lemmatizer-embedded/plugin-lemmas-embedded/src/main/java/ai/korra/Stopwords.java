package ai.korra;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

public class Stopwords {

    private static final Set<String> STOPWORDS = new HashSet<>(Arrays.asList(
            "אבל", "או", "אחר", "אך", "אל", "אם", "את", "בין", "גם", "דרך", "הוא",
            "היה", "זאת", "זה", "יותר", "יש", "כי", "כך", "כן", "לא", "לפני", "מה",
            "מי", "עד", "על", "עם", "רק", "של", "שם", "אדם", "אותה", "אותו", "אותם",
            "אחד", "אחרי", "אין", "אלא", "אלה", "אני", "אף", "אשר", "ב", "בגלל",
            "בן", "דבר", "היא", "הם", "זו", "זמן", "חלק", "יום", "יכול",
            "ישראל", "כדי", "כל", "כמו", "ל", "לה", "להיות", "לו", "לפי", "מן",
            "נגד", "פה", "שלו", "שנה", "ש", "מ", "ל", "כ", "ו", "ה", "ב",
            "אז", "_", "אילו", "אלו"
    ));

    public boolean isNotStopword(String term) {
        if (STOPWORDS.contains(term)) {
            if (HebDebugger.isDebugEnabled()) {
                String reversedText = new StringBuilder(term).reverse().toString();
                HebDebugger.log("Heb stopwords found and removed : " + reversedText);
            }
            return false;
        }
        return true;
    }
}
