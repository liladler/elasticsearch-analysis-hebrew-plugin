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

    private final java.util.concurrent.atomic.AtomicLong windowFailures = new java.util.concurrent.atomic.AtomicLong();

    private final int clsTokenId;
    private final int sepTokenId;

    private static final Set<Character> WEAK_LETTERS = Set.of('א', 'ה', 'ו', 'י');
    private static final int HEBREW_START = 0x0590;
    private static final int HEBREW_END = 0x05FF;
    private static final int HEBREW_RANGE = HEBREW_END - HEBREW_START + 1;
    private static final int PREDICTIONS_PER_TOKEN = 3;
    // BERT position embeddings stop at 512; two positions go to [CLS] and [SEP].
    static final int MAX_SEQUENCE_LENGTH = 512;
    static final int MAX_PIECES_PER_WINDOW = MAX_SEQUENCE_LENGTH - 2;

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

    /**
     * Lemmatizes the tokens of one field value.
     *
     * <p>Only the Hebrew words inside a token are lemmatized. Digits, Latin text and separators
     * between words are kept, so "אלף-7" stays "אלף-7" (lemmatizing the whole token would keep only
     * "אלף" and lose the number), and a token without any Hebrew letter is returned unchanged
     * without going through the model. Punctuation at the edges of a token is dropped.
     *
     * <p>The model accepts at most {@value #MAX_SEQUENCE_LENGTH} positions, two of which are the
     * [CLS] and [SEP] markers. Values that fit are lemmatized in a single call. Longer values are
     * split into windows of at most {@value #MAX_PIECES_PER_WINDOW} word pieces, preferring to end
     * a window after a sentence-ending token, so every token still gets a lemma instead of the
     * whole value silently being left as is.
     */
    public List<String> lemmatize(List<String> tokens) throws OrtException {
        if (tokens == null || tokens.isEmpty()) {
            return Collections.emptyList();
        }

        // Each Hebrew word of each token becomes one model word; the rest of the token is kept.
        List<List<MixedTokens.Part>> partsPerToken = new ArrayList<>(tokens.size());
        List<String> words = new ArrayList<>();
        List<String> sentenceView = new ArrayList<>();
        for (String token : tokens) {
            List<MixedTokens.Part> parts = MixedTokens.split(token);
            partsPerToken.add(parts);
            if (parts == null) {
                continue;
            }
            for (MixedTokens.Part part : parts) {
                if (part.hebrew) {
                    words.add(part.text);
                    sentenceView.add(part.text);
                }
            }
            if (endsSentence(token) && MixedTokens.hebrewPartCount(parts) > 0) {
                int last = sentenceView.size() - 1;
                sentenceView.set(last, sentenceView.get(last) + ".");
            }
        }

        List<String> wordLemmas = words.isEmpty() ? Collections.emptyList() : lemmatizeWords(words, sentenceView);

        List<String> result = new ArrayList<>(tokens.size());
        int next = 0;
        for (int i = 0; i < tokens.size(); i++) {
            List<MixedTokens.Part> parts = partsPerToken.get(i);
            if (parts == null) {
                result.add(MixedTokens.trimEdges(tokens.get(i)));
                continue;
            }
            int count = MixedTokens.hebrewPartCount(parts);
            result.add(MixedTokens.rebuild(parts, wordLemmas.subList(next, next + count)));
            next += count;
        }
        return result;
    }

    private List<String> lemmatizeWords(List<String> tokens, List<String> sentenceView) throws OrtException {
        List<int[]> wordPieceIds = new ArrayList<>(tokens.size());
        for (String token : tokens) {
            wordPieceIds.add(tokenizer.encode(token));
        }

        List<String> lemmas = new ArrayList<>(tokens.size());
        for (int[] window : planWindows(wordPieceIds, sentenceView, MAX_PIECES_PER_WINDOW)) {
            lemmas.addAll(lemmatizeWindow(tokens, wordPieceIds, window[0], window[1]));
        }
        return lemmas;
    }

    /**
     * Splits tokens into consecutive windows of at most {@code maxPieces} word pieces. Returns
     * {@code [startInclusive, endExclusive)} token index pairs covering every token.
     *
     * <p>When a window would overflow it ends after the last sentence-ending token it contains,
     * if there is one, otherwise at the last token that still fits.
     */
    static List<int[]> planWindows(List<int[]> pieces, List<String> tokens, int maxPieces) {
        List<int[]> windows = new ArrayList<>();
        int start = 0;
        int count = 0;
        int lastSentenceEnd = -1;

        for (int i = 0; i < tokens.size(); i++) {
            int size = pieces.get(i).length;
            while (count + size > maxPieces && count > 0) {
                int cut = lastSentenceEnd >= start ? lastSentenceEnd + 1 : i;
                windows.add(new int[]{start, cut});
                start = cut;
                count = 0;
                lastSentenceEnd = -1;
                for (int j = start; j < i; j++) {
                    count += pieces.get(j).length;
                    if (endsSentence(tokens.get(j))) {
                        lastSentenceEnd = j;
                    }
                }
            }
            count += size;
            if (endsSentence(tokens.get(i))) {
                lastSentenceEnd = i;
            }
        }
        windows.add(new int[]{start, tokens.size()});
        return windows;
    }

    private static boolean endsSentence(String token) {
        if (token.isEmpty()) {
            return false;
        }
        char last = token.charAt(token.length() - 1);
        return last == '.' || last == '!' || last == '?' || last == '\u05C3';
    }

    private List<String> lemmatizeWindow(List<String> tokens, List<int[]> wordPieceIds, int from, int to)
            throws OrtException {
        List<Integer> allIds = new ArrayList<>();
        allIds.add(clsTokenId);
        for (int i = from; i < to; i++) {
            for (int id : wordPieceIds.get(i)) {
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

        long[][] topPredictions;
        try {
            topPredictions = runInference(inputIds, attentionMask, tokenTypeIds);
        } catch (OrtException | RuntimeException e) {
            // Keep the rest of the value lemmatized, and make the failure visible: before this
            // the caller swallowed it silently and indexed the whole value unlemmatized.
            warnWindowFailed(seqLen, from, to, e);
            return new ArrayList<>(tokens.subList(from, to));
        }

        List<String> lemmas = new ArrayList<>(to - from);
        int logitIdx = 1;

        for (int i = from; i < to; i++) {
            int[] wpIds = wordPieceIds.get(i);
            String originalToken = tokens.get(i);
            if (wpIds.length == 0) {
                // Nothing to look up (e.g. a token made only of niqqud marks). It owns no model
                // position, so it must not read, or advance past, the next token's prediction.
                lemmas.add(originalToken);
                continue;
            }
            if (logitIdx < topPredictions.length) {
                lemmas.add(selectBestLemma(originalToken, topPredictions[logitIdx]));
            } else {
                lemmas.add(originalToken);
            }
            logitIdx += wpIds.length;
        }

        return lemmas;
    }

    private void warnWindowFailed(int sequenceLength, int from, int to, Exception cause) {
        long failures = windowFailures.incrementAndGet();
        // Failures should be rare (the model contract is validated at startup); cap the noise so a
        // systematic problem cannot flood the log, but still keep reporting periodically.
        if (failures <= 5 || failures % 1000 == 0) {
            LOGGER.log(System.Logger.Level.WARNING,
                    "Hebrew lemmatizer could not process a window of " + sequenceLength + " positions"
                            + " (tokens " + from + "-" + (to - 1) + "); leaving them unlemmatized"
                            + " (failure #" + failures + ")",
                    cause);
        }
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

        // Compare letters in the form the model saw them (no niqqud, lowercase); the original
        // token is still what we return when no candidate matches.
        for (char c : WordPieceTokenizer.normalize(originalToken).toCharArray()) {
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
