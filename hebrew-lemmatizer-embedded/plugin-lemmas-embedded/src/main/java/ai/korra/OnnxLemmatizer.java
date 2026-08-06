package ai.korra;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OnnxValue;
import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import ai.onnxruntime.ValueInfo;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
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
    private static final System.Logger LOGGER = System.getLogger(OnnxLemmatizer.class.getName());
    private static final String CACHE_DIRECTORY = "heb-lemmatizer";
    private static final String CACHE_KEY_RESOURCE = "model-cache-key.txt";

    private final OrtEnvironment env;
    private final OrtSession session;
    private final WordPieceTokenizer tokenizer;
    private final String[] vocab;

    private final int clsTokenId;
    private final int sepTokenId;

    private static final Set<Character> WEAK_LETTERS = Set.of('א', 'ה', 'ו', 'י');
    private static final int HEBREW_START = 0x0590;
    private static final int HEBREW_END = 0x05FF;
    private static final int HEBREW_RANGE = HEBREW_END - HEBREW_START + 1;
    private static final int PREDICTIONS_PER_TOKEN = 3;

    private OnnxLemmatizer(Path modelDir) throws OrtException, IOException {
        HebDebugger.log("Initializing OnnxLemmatizer from: " + modelDir);

        this.env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        opts.setInterOpNumThreads(1);
        opts.setIntraOpNumThreads(2);

        Path modelPath = modelDir.resolve("model.onnx");
        this.session = env.createSession(modelPath.toString(), opts);

        try {
            validateModelOutputContract(session);

            Path tokenizerPath = modelDir.resolve("tokenizer.json");
            this.tokenizer = new WordPieceTokenizer(tokenizerPath.toString());
            this.vocab = tokenizer.getVocab();

            this.clsTokenId = tokenizer.getTokenId("[CLS]");
            this.sepTokenId = tokenizer.getTokenId("[SEP]");

            // Validate the complete Java/ONNX contract once before publishing the singleton.
            long[][] canaryIds = {{clsTokenId, sepTokenId}};
            runInference(canaryIds, new long[][]{{1, 1}}, new long[][]{{0, 0}});
        } catch (OrtException | IOException | RuntimeException e) {
            try {
                session.close();
            } catch (OrtException closeException) {
                e.addSuppressed(closeException);
            }
            throw e;
        }
    }

    public static OnnxLemmatizer getInstance(Path dataPath) throws OrtException, IOException {
        if (instance == null) {
            synchronized (LOCK) {
                if (instance == null) {
                    Path modelDir = extractModelResources(dataPath);
                    OnnxLemmatizer candidate = new OnnxLemmatizer(modelDir);
                    try {
                        cleanupStaleCacheEntries(modelDir.getParent(), modelDir);
                    } catch (IOException e) {
                        LOGGER.log(System.Logger.Level.WARNING,
                                "Unable to remove stale Hebrew lemmatizer cache entries", e);
                    }
                    instance = candidate;
                }
            }
        }
        return instance;
    }

    private static void validateModelOutputContract(OrtSession session) throws OrtException {
        Map<String, NodeInfo> outputs = session.getOutputInfo();
        if (outputs.size() != 1) {
            throw new OrtException("Expected exactly one Top-3 model output, found " + outputs.size());
        }

        ValueInfo outputInfo = outputs.values().iterator().next().getInfo();
        if (!(outputInfo instanceof TensorInfo tensorInfo)) {
            throw new OrtException("Expected the Top-3 model output to be a tensor");
        }
        validateModelOutputContract(tensorInfo.type, tensorInfo.getShape());
    }

    static void validateModelOutputContract(OnnxJavaType type, long[] shape) throws OrtException {
        if (type != OnnxJavaType.INT64
                || shape.length != 3
                || shape[2] != PREDICTIONS_PER_TOKEN) {
            throw new OrtException("Incompatible ONNX model output: expected INT64 [batch, sequence, 3], found "
                    + type + " " + Arrays.toString(shape));
        }
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

        long[][] topPredictions = runInference(inputIds, attentionMask, tokenTypeIds);

        List<String> lemmas = new ArrayList<>();
        int logitIdx = 1;

        for (int i = 0; i < tokens.size(); i++) {
            int[] wpIds = wordPieceIds.get(i);
            String originalToken = tokens.get(i);
            if (logitIdx < topPredictions.length) {
                lemmas.add(selectBestLemma(originalToken, topPredictions[logitIdx]));
            } else {
                lemmas.add(originalToken);
            }
            logitIdx += wpIds.length;
        }

        return lemmas;
    }

    private long[][] runInference(long[][] inputIds, long[][] attentionMask, long[][] tokenTypeIds)
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
                    if (shape.length == 3 && shape[2] == PREDICTIONS_PER_TOKEN) {
                        long[][][] data = (long[][][]) tensor.getValue();
                        return data[0];
                    }
                }
                throw new OrtException("Unexpected output tensor format");
            }
        }
    }

    private String selectBestLemma(String originalToken, long[] topK) {
        boolean[] significantHebrew = new boolean[HEBREW_RANGE];
        int significantCount = 0;
        Set<Character> extraSignificant = null;

        for (char c : originalToken.toCharArray()) {
            if (WEAK_LETTERS.contains(c)) {
                continue;
            }
            if (c >= HEBREW_START && c <= HEBREW_END) {
                int idx = c - HEBREW_START;
                if (!significantHebrew[idx]) {
                    significantHebrew[idx] = true;
                    significantCount++;
                }
            } else {
                if (extraSignificant == null) {
                    extraSignificant = new HashSet<>();
                }
                if (extraSignificant.add(c)) {
                    significantCount++;
                }
            }
        }

        for (long predictedId : topK) {
            int predId = Math.toIntExact(predictedId);
            if (predId < 0 || predId >= vocab.length) {
                continue;
            }

            String candidate = vocab[predId];
            if (candidate.startsWith("[") || candidate.startsWith("##")) {
                continue;
            }

            int overlap = 0;
            int candCount = 0;
            boolean[] candHebrew = new boolean[HEBREW_RANGE];
            Set<Character> candExtra = null;

            for (char c : candidate.toCharArray()) {
                if (WEAK_LETTERS.contains(c)) {
                    continue;
                }
                if (c >= HEBREW_START && c <= HEBREW_END) {
                    int idx = c - HEBREW_START;
                    if (!candHebrew[idx]) {
                        candHebrew[idx] = true;
                        candCount++;
                    }
                    if (significantHebrew[idx]) {
                        overlap++;
                    }
                } else {
                    if (candExtra == null) {
                        candExtra = new HashSet<>();
                    }
                    if (candExtra.add(c)) {
                        candCount++;
                        if (extraSignificant != null && extraSignificant.contains(c)) {
                            overlap++;
                        }
                    }
                }
            }

            int minRequired = Math.min(2, Math.min(significantCount, candCount));
            if (overlap >= minRequired) {
                return candidate;
            }
        }

        return originalToken;
    }

    private static Path extractModelResources(Path dataPath) throws IOException {
        String cacheKey = readResourceText(CACHE_KEY_RESOURCE).trim();
        if (!cacheKey.matches("[0-9a-f]{64}")) {
            throw new IOException("Invalid embedded model cache key: " + cacheKey);
        }

        Path cacheDir = dataPath.resolve(CACHE_DIRECTORY);
        Path modelDir = cacheDir.resolve(cacheKey);
        Files.createDirectories(modelDir);

        String[] resources = {"model.onnx", "tokenizer.json"};
        for (String resource : resources) {
            extractResourceAtomically(resource, modelDir);
        }

        return modelDir;
    }

    private static void extractResourceAtomically(String resource, Path modelDir) throws IOException {
        Path targetPath = modelDir.resolve(resource);
        if (Files.isRegularFile(targetPath)) {
            return;
        }
        if (Files.exists(targetPath)) {
            throw new IOException("Model cache target is not a regular file: " + targetPath);
        }

        Path temporaryPath = Files.createTempFile(modelDir, resource + ".", ".tmp");
        try (InputStream stream = openResource(resource)) {
            Files.copy(stream, temporaryPath, StandardCopyOption.REPLACE_EXISTING);
            try {
                Files.move(temporaryPath, targetPath,
                        StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
            } catch (AtomicMoveNotSupportedException e) {
                Files.move(temporaryPath, targetPath, StandardCopyOption.REPLACE_EXISTING);
            }
        } finally {
            Files.deleteIfExists(temporaryPath);
        }
    }

    private static String readResourceText(String resource) throws IOException {
        try (InputStream stream = openResource(resource)) {
            return new String(stream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8);
        }
    }

    private static InputStream openResource(String resource) throws IOException {
        String resourcePath = "model/" + resource;
        InputStream stream = OnnxLemmatizer.class.getModule().getResourceAsStream(resourcePath);
        if (stream == null) {
            stream = OnnxLemmatizer.class.getClassLoader().getResourceAsStream(resourcePath);
        }
        if (stream == null) {
            throw new IOException("Resource not found in JAR: " + resourcePath);
        }
        return stream;
    }

    static void cleanupStaleCacheEntries(Path cacheDir, Path activeModelDir) throws IOException {
        Path activePath = activeModelDir.toAbsolutePath().normalize();
        try (var entries = Files.list(cacheDir)) {
            for (Path entry : entries.toList()) {
                if (!entry.toAbsolutePath().normalize().equals(activePath)) {
                    deleteRecursively(entry);
                }
            }
        }
    }

    private static void deleteRecursively(Path path) throws IOException {
        try (var paths = Files.walk(path)) {
            for (Path entry : paths.sorted(Comparator.reverseOrder()).toList()) {
                Files.deleteIfExists(entry);
            }
        }
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
