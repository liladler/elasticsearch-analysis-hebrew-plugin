package ai.korra;

import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Pure Java WordPiece tokenizer for BERT models.
 * Reads vocabulary from HuggingFace tokenizer.json format.
 */
public class WordPieceTokenizer {

    private final Map<String, Integer> vocab;
    private final String[] reverseVocab;
    private final int unkTokenId;
    private final int maxInputCharsPerWord;

    public WordPieceTokenizer(String tokenizerJsonPath) throws java.io.IOException {
        this(Files.newInputStream(Path.of(tokenizerJsonPath)));
    }

    public WordPieceTokenizer(InputStream inputStream) throws java.io.IOException {
        this.maxInputCharsPerWord = 100;
        this.vocab = new HashMap<>();

        Gson gson = new Gson();
        try (Reader reader = new InputStreamReader(inputStream, StandardCharsets.UTF_8)) {
            JsonObject root = gson.fromJson(reader, JsonObject.class);
            JsonObject model = root.getAsJsonObject("model");
            JsonObject vocabObj = model.getAsJsonObject("vocab");
            for (Map.Entry<String, JsonElement> entry : vocabObj.entrySet()) {
                vocab.put(entry.getKey(), entry.getValue().getAsInt());
            }
        }

        this.reverseVocab = new String[vocab.size()];
        for (Map.Entry<String, Integer> entry : vocab.entrySet()) {
            reverseVocab[entry.getValue()] = entry.getKey();
        }

        this.unkTokenId = vocab.getOrDefault("[UNK]", 100);
    }

    /**
     * Removes Hebrew niqqud and cantillation marks, and decomposes Hebrew presentation forms, so
     * vowelized text reaches the vocabulary lookup the way the model's own tokenizer.json
     * (NFD, StripAccents) presents it. Without this a word such as "שְׁלוֹם" fails to match and
     * collapses to a single letter.
     *
     * <p>Deliberately limited to Hebrew: Latin letters are left exactly as they are. The model's
     * tokenizer also lowercases and strips Latin accents, but doing that here changes how tokens
     * such as "T-204" are handled (the lowercased form matches the vocabulary, and the plugin then
     * keeps only its first piece), which would alter the terms of existing indices.
     *
     * <p>Text without any such mark is returned as is, without allocating.
     */
    static String normalize(String word) {
        boolean needed = false;
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if ((c >= 0x0591 && c <= 0x05C7) || (c >= 0xFB1D && c <= 0xFB4F)) {
                needed = true;
                break;
            }
        }
        if (needed == false) {
            return word;
        }

        StringBuilder builder = new StringBuilder(word.length());
        for (int i = 0; i < word.length(); i++) {
            char c = word.charAt(i);
            if (c >= 0xFB1D && c <= 0xFB4F) {
                // Precomposed letter with a point, e.g. U+FB2A: split into letter + mark first.
                for (char part : Normalizer.normalize(String.valueOf(c), Normalizer.Form.NFD).toCharArray()) {
                    appendUnlessHebrewMark(builder, part);
                }
            } else {
                appendUnlessHebrewMark(builder, c);
            }
        }
        return builder.toString();
    }

    private static void appendUnlessHebrewMark(StringBuilder builder, char c) {
        boolean hebrewMark = c >= 0x0591 && c <= 0x05C7
                && Character.getType(c) == Character.NON_SPACING_MARK;
        if (hebrewMark == false) {
            builder.append(c);
        }
    }

    public int[] encode(String word) {
        if (word == null || word.isEmpty()) {
            return new int[0];
        }
        word = normalize(word);
        if (word.isEmpty()) {
            return new int[0];
        }

        List<Integer> tokens = new ArrayList<>(Math.min(word.length(), 8));

        Integer directMatch = vocab.get(word);
        if (directMatch != null) {
            return new int[]{directMatch};
        }

        char[] chars = word.toCharArray();
        if (chars.length > maxInputCharsPerWord) {
            tokens.add(unkTokenId);
            return toIntArray(tokens);
        }

        int start = 0;
        while (start < chars.length) {
            int end = chars.length;
            String curSubstr = null;
            while (start < end) {
                String substr = new String(chars, start, end - start);
                if (start > 0) {
                    substr = "##" + substr;
                }
                if (vocab.containsKey(substr)) {
                    curSubstr = substr;
                    break;
                }
                end--;
            }

            if (curSubstr == null) {
                tokens.add(unkTokenId);
                break;
            }

            tokens.add(vocab.get(curSubstr));
            start = end;
        }

        return toIntArray(tokens);
    }

    public int getTokenId(String token) {
        return vocab.getOrDefault(token, unkTokenId);
    }

    public String[] getVocab() {
        return reverseVocab.clone();
    }

    private int[] toIntArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for (int i = 0; i < list.size(); i++) {
            arr[i] = list.get(i);
        }
        return arr;
    }
}
