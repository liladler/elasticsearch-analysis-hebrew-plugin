# elasticsearch-analysis-hebrew-plugin

This repo is a **fork** focused on an embedded, optimized Hebrew analyzer for Elasticsearch 9.x.

## What is different in this fork

- The DictaBERT model is embedded in the plugin and runs inside Elasticsearch with ONNX Runtime,
  with no external service
- INT8 quantized model: DictaBERT-Lex on `main`, DictaBERT Tiny on `dictabert-tiny`
- Stopwords included in the same plugin (`heb_stopwords`)

## Choose between Lex and Tiny

The repository provides two alternative builds with the same analyzer API, filter names and
plugin name. Install **one** of them on every node; the two zips cannot be installed side by side.

| Variant | Source branch | Zip size | Best fit |
| --- | --- | ---: | --- |
| DictaBERT-Lex | `main` | about 167 MB | Lemma quality first |
| DictaBERT Tiny | [`dictabert-tiny`](https://github.com/liladler/elasticsearch-analysis-hebrew-plugin/tree/dictabert-tiny) | about 62 MB | Indexing throughput first |

Tiny is several times faster than Lex, but the two models do not always produce the same lemma, so
switching between them changes indexed terms. Validate both on representative text (domain terms,
proper names, inflected words) before choosing.

### Release downloads

Download the zips from [Releases](https://github.com/liladler/elasticsearch-analysis-hebrew-plugin/releases).
Each release is built for one Elasticsearch version and has two zips:

- `heb-lemmas-embedded-plugin-<ES_VERSION>.zip` (DictaBERT-Lex)
- `heb-lemmas-embedded-tiny-plugin-<ES_VERSION>.zip` (DictaBERT Tiny)

The zip's version must match your Elasticsearch version exactly. The zips contain the Linux
ONNX Runtime libraries only, so they run on Linux nodes (x86-64 and ARM), including Elastic Cloud.

## Build

Prerequisites:

- `python3` + `pip`
- Python deps: `hebrew-lemmatizer-embedded/model-export/requirements.txt`
- Docker (to build a Linux-compatible zip)

Install Python deps:

```
cd hebrew-lemmatizer-embedded/model-export
python3 -m pip install -r requirements.txt
```

Run the build script (exports INT8 ONNX + builds Linux-only zip):

```
./scripts/build_plugin_linux.sh
```

Use the Linux-only build for Linux deployments and Elastic Cloud.

The model file (`model.onnx`) is not stored in git. The build script generates it with
`export_model.py`.

Optional version overrides:

```
./scripts/build_plugin_linux.sh --es-version 9.3.0
```

Output:

```
hebrew-lemmatizer-embedded/plugin-lemmas-embedded/build/distributions/heb-lemmas-embedded-plugin-<ES_VERSION>.zip
```

The automated release workflow builds this Lex artifact from `main` and the
corresponding `heb-lemmas-embedded-tiny-plugin-<ES_VERSION>.zip` artifact from
the `dictabert-tiny` branch.

## Install & test (local ES)

```
/path/to/elasticsearch/bin/elasticsearch-plugin remove heb-lemmas-embedded-plugin
/path/to/elasticsearch/bin/elasticsearch-plugin install file:///path/to/heb-lemmas-embedded-plugin-<ES_VERSION>.zip
/path/to/elasticsearch/bin/elasticsearch
```

```
curl -k -X POST "https://localhost:9200/_analyze" \
  -H "Content-Type: application/json" \
  -u "elastic:<password>" \
  -d '{"tokenizer":"whitespace","filter":["heb_lemmas","heb_stopwords"],"text":"הילדים אוכלים את הבננות"}'
```

## Create index example (settings + mappings)

```
curl -k -u "elastic:<password>" -X PUT "https://localhost:9200/hebrew_test" \
  -H "Content-Type: application/json" \
  -d '{
    "settings": {
      "analysis": {
        "analyzer": {
          "hebrew_analyzer": {
            "char_filter": ["html_strip"],
            "tokenizer": "whitespace",
            "filter": ["heb_lemmas", "heb_stopwords"]
          }
        }
      }
    },
    "mappings": {
      "properties": {
        "text": {
          "type": "text",
          "analyzer": "hebrew_analyzer"
        }
      }
    }
  }'
```

## Bug fixes and upgrading

These are bug fixes, and they are also included in the earlier releases of this plugin. There are
no changes to the API, settings, mappings or filter names, so a new zip is a drop-in replacement.

**We recommend reindexing your data after upgrading.** These are bug fixes, and they change the
terms produced for a small set of inputs that were previously handled wrongly: tokens that carry
punctuation, digits, hyphens, Latin letters or niqqud. Plain Hebrew words are lemmatized as
before. Documents that are already indexed keep their old terms, so reindexing makes them match
the same way newly indexed documents do. Without it everything keeps working, and only those
tokens may miss.

| Fixed | Before | After |
| --- | --- | --- |
| `heb_lemmas` inside a `condition` filter lost the tokens that skip the condition | `condition` over "הילדים אוכלים את הבננות בגן" returned 4 tokens, `את` was dropped | all 5 tokens are returned |
| Hebrew with niqqud or cantillation marks collapsed to one letter | `שְׁלוֹם` became `ש` | `שְׁלוֹם` becomes `שלום` |
| Values longer than 510 word pieces were indexed without any lemmatization, silently | 511 words: no lemmas | long values are split into windows (at sentence ends where possible) and fully lemmatized |
| `keyword_marker` before `heb_lemmas` was ignored | the keyword was lemmatized and lost its keyword flag | the token is kept unchanged and stays flagged |
| `positionLength` was reset to 1 | graph filters before `heb_lemmas` (for example `synonym_graph`) lost multi-word paths | `positionLength` is preserved |
| Tokens mixing Hebrew with numbers, Latin text or hyphens lost everything after the first Hebrew word | `אלף-7` became `אלף`, so `אלף-7` and `אלף-9` were the same term | `אלף-7` stays `אלף-7` |
| The quote inside an acronym was removed before the model saw the word, so a prefixed acronym was not recognized | `בארה"ב` became `בארהב`, while `ארה"ב` became `ארה` | `ארה"ב`, `ארה״ב`, `בארה"ב` and `בארה״ב` all become `ארהב` |
| On x86 CPUs without VNNI (AMD EPYC Zen 2 and 3, Intel before Cascade Lake), the INT8 model's sums overflowed and lemmas came out wrong, with no error | Tiny: `ישראל` became `לישראל`, `תאילנד` became `מתאילנד` | the same lemmas as on any other CPU |

### Tokens that mix Hebrew with numbers, Latin text or hyphens

Only the Hebrew words inside a token are lemmatized. Digits, Latin text and the separators
between words are kept, and a token without any Hebrew letter is not sent to the model at all.
Punctuation at the edges of a token is still dropped.

| Token | Term |
| --- | --- |
| `אלף-7` | `אלף-7` |
| `יחידה-8200` | `יחידה-8200` |
| `בן-גוריון` | `בן-גוריון` |
| `קריית-שמונה` | `קריה-שמונה` |
| `12/05/2024` | `12/05/2024` |
| `iPhone` | `iPhone` |
| `(אלף-7)` | `אלף-7` |
| `הילדים.` | `ילד` |

### Keeping specific tokens as they are

Put a `keyword_marker` before `heb_lemmas` to leave chosen tokens untouched, for example every
token that contains a digit, a Latin letter, a slash or a hyphen:

```json
"filter": {
  "keep_codes": { "type": "keyword_marker", "keywords_pattern": ".*[0-9A-Za-z/\\-].*" }
}
```

### Splitting hyphenated tokens into separate terms

The `whitespace` tokenizer keeps a hyphenated token such as `אלף-7` in one piece, and the filter
keeps it as one term. If you would rather index the parts as separate terms, split on the hyphen
before `heb_lemmas` with a `pattern_replace` char filter:

```json
"analysis": {
  "char_filter": {
    "split_hyphens": {
      "type": "pattern_replace",
      "pattern": "(?<=[\\p{L}\\p{N}])[-־](?=[\\p{L}\\p{N}])",
      "replacement": " "
    }
  },
  "analyzer": {
    "hebrew_analyzer": {
      "char_filter": ["split_hyphens"],
      "tokenizer": "whitespace",
      "filter": ["heb_lemmas", "heb_stopwords"]
    }
  }
}
```

| Text | Without the char filter | With the char filter |
| --- | --- | --- |
| `ה-26` | `ה-26` | `26` |
| `ב-1985` | `ב-1985` | `1985` |
| `אלף-7` | `אלף-7` | `אלף`, `7` |
| `קריית-שמונה` | `קריה-שמונה` | `קריה`, `שמונה` |
| `T-204` | `T-204` | `T`, `204` |

- The parts get consecutive positions, so a `match_phrase` query keeps them side by side.
- The query is analyzed the same way, so searching with a hyphen or with a space finds the same
  documents.
- A part that is a stopword is removed by `heb_stopwords`, which is why the one-letter prefix in
  `ה-26` disappears.
- For identifiers that must match exactly (call signs, unit ids), use a `keyword` field rather than
  this analyzer: after splitting, a search for `204` also finds `F-204`.

### Native memory

The model runs outside the JVM heap, so its memory is not covered by the heap size or by
Elasticsearch's circuit breakers. The DictaBERT-Lex build limits how much text the model
processes at once, based on the node's memory, heap and processors, so a busy node slows down
(the write queue fills and clients get 429 responses) instead of running out of memory.

- Keep the heap at the default 50% of memory; a larger heap leaves less room for the model.
- To index more long documents per second, add data nodes or CPUs, or set `number_of_replicas`
  to 0 during a bulk load, since every replica runs the model again.

## Upgrade to a newer ES version

1. Download the zip for the new version from
   [Releases](https://github.com/liladler/elasticsearch-analysis-hebrew-plugin/releases), or build
   it with `./scripts/build_plugin_linux.sh --es-version <ES_VERSION>`.
2. On every node, remove the old plugin, install the new zip and restart.
