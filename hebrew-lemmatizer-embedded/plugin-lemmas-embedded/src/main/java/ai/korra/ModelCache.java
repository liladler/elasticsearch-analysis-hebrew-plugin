package ai.korra;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;

/**
 * The model files, extracted from the plugin jar into a directory under the node's data path
 * named after their content, so an upgraded model never reuses an old extraction.
 */
final class ModelCache {

    private static final String CACHE_DIRECTORY = "heb-lemmatizer";
    private static final String CACHE_KEY_RESOURCE = "model-cache-key.txt";

    private ModelCache() {
    }

    static Path extract(Path dataPath) throws IOException {
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
        InputStream stream = ModelCache.class.getModule().getResourceAsStream(resourcePath);
        if (stream == null) {
            stream = ModelCache.class.getClassLoader().getResourceAsStream(resourcePath);
        }
        if (stream == null) {
            throw new IOException("Resource not found in JAR: " + resourcePath);
        }
        return stream;
    }

    static void cleanupStaleEntries(Path cacheDir, Path activeModelDir) throws IOException {
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
}
