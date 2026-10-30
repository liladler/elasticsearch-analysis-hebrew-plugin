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
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Hebrew lemmatizer using ONNX Runtime for inference.
 */
public class OnnxLemmatizer implements AutoCloseable {

    private static volatile OnnxLemmatizer instance;
    private static final Object LOCK = new Object();
    private static final System.Logger LOGGER = System.getLogger(OnnxLemmatizer.class.getName());

    private final OrtEnvironment env;
    private final OrtSession session;
    private final WordPieceTokenizer tokenizer;
    private final LemmaSelector selector;
    private final InferenceLimiter limiter;

    private final java.util.concurrent.atomic.AtomicLong windowFailures = new java.util.concurrent.atomic.AtomicLong();

    private final int clsTokenId;
    private final int sepTokenId;

    private static final int PREDICTIONS_PER_TOKEN = 3;
    // BERT position embeddings stop at 512; two positions go to [CLS] and [SEP].
    static final int MAX_SEQUENCE_LENGTH = 512;
    static final int MAX_PIECES_PER_WINDOW = MAX_SEQUENCE_LENGTH - 2;
    static final int THREADS_PER_WINDOW = 2;

    private OnnxLemmatizer(Path modelDir, int positionBudget) throws OrtException, IOException {
        HebDebugger.log("Initializing OnnxLemmatizer from: " + modelDir);
        this.limiter = new InferenceLimiter(positionBudget);

        this.env = OrtEnvironment.getEnvironment();

        OrtSession.SessionOptions opts = new OrtSession.SessionOptions();
        opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT);
        opts.setInterOpNumThreads(1);
        opts.setIntraOpNumThreads(THREADS_PER_WINDOW);
        // On x86 CPUs without VNNI (AMD Zen 2 and 3, Intel before Cascade Lake) the default INT8
        // matmul kernel overflows its 16-bit accumulators, which silently corrupts the model's output:
        // about 62% of the lemmas of the Tiny model came out wrong. This setting makes it use a
        // kernel that does not overflow. It has no effect on ARM.
        opts.addConfigEntry("session.x64quantprecision", "1");

        Path modelPath = modelDir.resolve("model.onnx");
        this.session = env.createSession(modelPath.toString(), opts);

        try {
            validateModelOutputContract(session);

            Path tokenizerPath = modelDir.resolve("tokenizer.json");
            this.tokenizer = new WordPieceTokenizer(tokenizerPath.toString());
            this.selector = new LemmaSelector(tokenizer.getVocab());

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

    /**
     * Returns the node's lemmatizer, creating it on first use.
     *
     * @param positionBudget model positions that may be processed at once; see
     *                       {@link InferenceLimiter#budgetFor}. Only the first call's value is used,
     *                       since there is one model per node.
     */
    public static OnnxLemmatizer getInstance(Path dataPath, int positionBudget) throws OrtException, IOException {
        if (instance == null) {
            synchronized (LOCK) {
                if (instance == null) {
                    Path modelDir = ModelCache.extract(dataPath);
                    OnnxLemmatizer candidate = new OnnxLemmatizer(modelDir, positionBudget);
                    int totalPositions = positionBudget + InferenceLimiter.SHORT_WINDOW_BUDGET;
                    LOGGER.log(System.Logger.Level.INFO,
                            "Hebrew lemmatizer limited to " + positionBudget + " model positions at once, plus "
                                    + InferenceLimiter.SHORT_WINDOW_BUDGET + " for windows of up to "
                                    + InferenceLimiter.SHORT_WINDOW + " positions (about "
                                    + (totalPositions * InferenceLimiter.BYTES_PER_POSITION >> 20)
                                    + " MB of native memory at peak)");
                    try {
                        ModelCache.cleanupStaleEntries(modelDir.getParent(), modelDir);
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
                if (part.hebrew()) {
                    words.add(part.text());
                    sentenceView.add(part.text());
                }
            }
            if (WindowPlanner.endsSentence(token) && MixedTokens.hebrewPartCount(parts) > 0) {
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
        for (int[] window : WindowPlanner.plan(wordPieceIds, sentenceView, MAX_PIECES_PER_WINDOW)) {
            lemmas.addAll(lemmatizeWindow(tokens, wordPieceIds, window[0], window[1]));
        }
        return lemmas;
    }

    private List<String> lemmatizeWindow(List<String> tokens, List<int[]> wordPieceIds, int from, int to)
            throws OrtException {
        int seqLen = 2;
        for (int i = from; i < to; i++) {
            seqLen += wordPieceIds.get(i).length;
        }

        // [CLS] pieces... [SEP], every position attended to, one segment (token type 0).
        long[][] inputIds = new long[1][seqLen];
        long[][] attentionMask = new long[1][seqLen];
        long[][] tokenTypeIds = new long[1][seqLen];
        int position = 0;
        inputIds[0][position++] = clsTokenId;
        for (int i = from; i < to; i++) {
            for (int id : wordPieceIds.get(i)) {
                inputIds[0][position++] = id;
            }
        }
        inputIds[0][position] = sepTokenId;
        Arrays.fill(attentionMask[0], 1);

        long[][] topPredictions;
        int taken = limiter.acquire(seqLen);
        try {
            topPredictions = runInference(inputIds, attentionMask, tokenTypeIds);
        } catch (OrtException | RuntimeException e) {
            // Keep the rest of the value lemmatized, and make the failure visible: before this
            // the caller swallowed it silently and indexed the whole value unlemmatized.
            warnWindowFailed(seqLen, from, to, e);
            return new ArrayList<>(tokens.subList(from, to));
        } finally {
            limiter.release(taken);
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
                lemmas.add(selector.select(originalToken, topPredictions[logitIdx]));
            } else {
                lemmas.add(WordPieceTokenizer.withoutDoubleQuotes(WordPieceTokenizer.normalize(originalToken)));
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
