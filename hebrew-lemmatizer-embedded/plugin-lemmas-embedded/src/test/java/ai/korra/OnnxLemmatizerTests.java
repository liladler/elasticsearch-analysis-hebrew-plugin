package ai.korra;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import ai.onnxruntime.OnnxJavaType;
import ai.onnxruntime.OrtException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

public class OnnxLemmatizerTests {

    @Rule
    public TemporaryFolder temporaryFolder = new TemporaryFolder();

    @Test
    public void acceptsInt64TopThreeOutput() throws OrtException {
        OnnxLemmatizer.validateModelOutputContract(OnnxJavaType.INT64, new long[]{-1, -1, 3});
    }

    @Test
    public void rejectsFullLogitsOutput() {
        OrtException exception = assertThrows(OrtException.class,
                () -> OnnxLemmatizer.validateModelOutputContract(
                        OnnxJavaType.FLOAT, new long[]{-1, -1, 128000}));

        assertTrue(exception.getMessage().contains("expected INT64 [batch, sequence, 3]"));
    }

    @Test
    public void rejectsWrongTopThreeElementType() {
        assertThrows(OrtException.class,
                () -> OnnxLemmatizer.validateModelOutputContract(
                        OnnxJavaType.FLOAT, new long[]{-1, -1, 3}));
    }

    @Test
    public void removesOnlyStaleCacheEntries() throws Exception {
        Path cacheDir = temporaryFolder.newFolder("heb-lemmatizer").toPath();
        Path activeDir = Files.createDirectory(cacheDir.resolve("active-key"));
        Path activeModel = Files.writeString(activeDir.resolve("model.onnx"), "active");

        Path staleDir = Files.createDirectories(cacheDir.resolve("stale-key/nested"));
        Files.writeString(staleDir.resolve("model.onnx"), "stale");
        Path legacyModel = Files.writeString(cacheDir.resolve("model.onnx"), "legacy");

        OnnxLemmatizer.cleanupStaleCacheEntries(cacheDir, activeDir);

        assertTrue(Files.exists(activeDir));
        assertTrue(Files.exists(activeModel));
        assertFalse(Files.exists(cacheDir.resolve("stale-key")));
        assertFalse(Files.exists(legacyModel));
    }
    @Test
    public void lemmatizesVowelizedWordsLikePlainWords() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data").toPath());

        List<String> plain = lemmatizer.lemmatize(List.of("הילדים", "אוכלים"));
        List<String> vowelized = lemmatizer.lemmatize(List.of("הַיְלָדִים", "אוֹכְלִים"));

        assertEquals(List.of("ילד", "אוכל"), plain);
        assertEquals("niqqud must not change the lemma", plain, vowelized);
    }

    @Test
    public void latinIdentifiersAreNotAlteredByNiqqudHandling() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data11").toPath());

        // These came back as "t" and "f" when the normalization also lowercased Latin text.
        List<String> identifiers = List.of("T-204", "F-16", "F-16I");

        assertEquals(identifiers, lemmatizer.lemmatize(identifiers));
    }

    @Test
    public void aTokenMadeOnlyOfMarksDoesNotShiftTheFollowingLemmas() throws Exception {
        OnnxLemmatizer lemmatizer = OnnxLemmatizer.getInstance(temporaryFolder.newFolder("data2").toPath());

        List<String> lemmas = lemmatizer.lemmatize(List.of("\u05B0\u05B1", "הילדים"));

        assertEquals(List.of("\u05B0\u05B1", "ילד"), lemmas);
    }
}
