package ai.korra;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Hebrew lemmatizer using ONNX Runtime for inference.
 */
public class OnnxLemmatizer implements AutoCloseable {

    private static volatile OnnxLemmatizer instance;
    private static final Object LOCK = new Object();

    private final OrtEnvironment env;
    private final OrtSession session;
    private final WordPieceTokenizer tokenizer;
    private final String[] vocab;

    private final int clsTokenId;
    private final int sepTokenId;

    private static final Set<Character> WEAK_LETTERS = Set.of('א', 'ה', 'ו', 'י');

    private OnnxLemmatizer(Path modelDir) throws OrtException, IOException {
        HebDebugger.log("Initializing OnnxLemmatizer from: " + modelDir);

        this.env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        opts.setInterOpNumThreads(1);
        opts.setIntraOpNumThreads(2);

        Path modelPath = modelDir.resolve("model.onnx");
        this.session = env.createSession(modelPath.toString(), opts);

        Path tokenizerPath = modelDir.resolve("tokenizer.json");
        this.tokenizer = new WordPieceTokenizer(tokenizerPath.toString());
        this.vocab = tokenizer.getVocab();

        this.clsTokenId = tokenizer.getTokenId("[CLS]");
        this.sepTokenId = tokenizer.getTokenId("[SEP]");
    }

    public static OnnxLemmatizer getInstance() throws OrtException, IOException {
        if (instance == null) {
            synchronized (LOCK) {
                if (instance == null) {
                    Path modelDir = extractModelResources();
                    instance = new OnnxLemmatizer(modelDir);
                }
            }
        }
        return instance;
    }

    public List<String> lemmatize(List<String> tokens) throws OrtException {
        if (tokens == null || tokens.isEmpty()) {
            return Collections.emptyList();
        }

        List<int[]> wordPieceIds = new ArrayList<>();
        List<Integer> allIds = new ArrayList<>();
        allIds.add(clsTokenId);

        for (String token : tokens) {
            int[] ids = tokenizer.encode(token);
            wordPieceIds.add(ids);
            for (int id : ids) {
                allIds.add(id);
            }
        }
        allIds.add(sepTokenId);

        int seqLen = allIds.size();
        long[][] inputIds = new long[1][seqLen];
        long[][] attentionMask = new long[1][seqLen];
        long[][] tokenTypeIds = new long[1][seqLen];

        for (int i = 0; i < seqLen; i++) {
            inputIds[0][i] = allIds.get(i);
            attentionMask[0][i] = 1;
            tokenTypeIds[0][i] = 0;
        }

        float[][] logits = runInference(inputIds, attentionMask, tokenTypeIds);

        List<String> lemmas = new ArrayList<>();
        int logitIdx = 1;

        for (int i = 0; i < tokens.size(); i++) {
            int[] wpIds = wordPieceIds.get(i);
            String originalToken = tokens.get(i);
            if (logitIdx < logits.length) {
                int[] topK = getTopK(logits[logitIdx], 3);
                lemmas.add(selectBestLemma(originalToken, topK));
            } else {
                lemmas.add(originalToken);
            }
            logitIdx += wpIds.length;
        }

        return lemmas;
    }

    private float[][] runInference(long[][] inputIds, long[][] attentionMask, long[][] tokenTypeIds)
            throws OrtException {
        try (OnnxTensor idsTensor = OnnxTensor.createTensor(env, inputIds);
             OnnxTensor maskTensor = OnnxTensor.createTensor(env, attentionMask);
             OnnxTensor typeTensor = OnnxTensor.createTensor(env, tokenTypeIds)) {

            Map<String, OnnxTensor> inputs = new LinkedHashMap<>();
            inputs.put("input_ids", idsTensor);
            inputs.put("attention_mask", maskTensor);
            inputs.put("token_type_ids", typeTensor);

            try (OrtSession.Result results = session.run(inputs)) {
                OnnxValue output = results.get(0);
                if (output instanceof OnnxTensor tensor) {
                    long[] shape = tensor.getInfo().getShape();
                    if (shape.length == 3) {
                        float[][][] data = (float[][][]) tensor.getValue();
                        return data[0];
                    }
                }
                throw new OrtException("Unexpected output tensor format");
            }
        }
    }

    private int[] getTopK(float[] scores, int k) {
        int vocabSize = scores.length;
        boolean[] used = new boolean[vocabSize];
        int[] topK = new int[k];

        for (int i = 0; i < k; i++) {
            float maxVal = Float.NEGATIVE_INFINITY;
            int maxIdx = 0;
            for (int j = 0; j < vocabSize; j++) {
                if (!used[j] && scores[j] > maxVal) {
                    maxVal = scores[j];
                    maxIdx = j;
                }
            }
            topK[i] = maxIdx;
            used[maxIdx] = true;
        }
        return topK;
    }

    private String selectBestLemma(String originalToken, int[] topK) {
        Set<Character> significantChars = new HashSet<>();
        for (char c : originalToken.toCharArray()) {
            if (!WEAK_LETTERS.contains(c)) {
                significantChars.add(c);
            }
        }

        for (int predId : topK) {
            if (predId < 0 || predId >= vocab.length) {
                continue;
            }

            String candidate = vocab[predId];
            if (candidate.startsWith("[") || candidate.startsWith("##")) {
                continue;
            }

            int overlap = 0;
            Set<Character> candSignificant = new HashSet<>();
            for (char c : candidate.toCharArray()) {
                if (!WEAK_LETTERS.contains(c)) {
                    candSignificant.add(c);
                    if (significantChars.contains(c)) {
                        overlap++;
                    }
                }
            }

            int minRequired = Math.min(2, Math.min(significantChars.size(), candSignificant.size()));
            if (overlap >= minRequired) {
                return candidate;
            }
        }

        return originalToken;
    }

    private static Path extractModelResources() throws IOException {
        String dataPath = System.getProperty("es.path.data");
        if (dataPath == null) {
            dataPath = System.getProperty("java.io.tmpdir");
        }

        Path cacheDir = Path.of(dataPath, "heb-lemmatizer");
        Files.createDirectories(cacheDir);

        String[] resources = {"model.onnx", "tokenizer.json"};
        for (String resource : resources) {
            Path targetPath = cacheDir.resolve(resource);
            if (!Files.exists(targetPath)) {
                String resourcePath = "model/" + resource;
                InputStream is = OnnxLemmatizer.class.getModule().getResourceAsStream(resourcePath);
                if (is == null) {
                    is = OnnxLemmatizer.class.getClassLoader().getResourceAsStream(resourcePath);
                }
                try (InputStream stream = is) {
                    if (stream == null) {
                        throw new IOException("Resource not found in JAR: " + resourcePath);
                    }
                    Files.copy(stream, targetPath, StandardCopyOption.REPLACE_EXISTING);
                }
            }
        }

        return cacheDir;
    }

    @Override
    public void close() throws Exception {
        if (session != null) {
            session.close();
        }
        if (env != null) {
            env.close();
        }
    }
}
