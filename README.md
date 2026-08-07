# elasticsearch-analysis-hebrew-plugin

This repo is a **fork** focused on an embedded, optimized Hebrew analyzer for Elasticsearch 9.x.

## What is different in this fork

- Embedded DictaBERT model (no external Docker service)
- ONNX Runtime 1.26 in-process inference
- INT8 quantized DictaBERT-Lex model
- Top-3 predictions emitted inside the ONNX graph to avoid copying full logits into Java
- Fail-fast validation of the Java/ONNX output contract before indexing
- Content-addressed model extraction under the configured Elasticsearch data path
- Stopwords included **in the same plugin** (`heb_stopwords`)
- ES 9.x classic plugin with entitlements

## Choose between Lex and Tiny

The repository provides two alternative builds with the same analyzer API,
plugin name, ONNX Runtime 1.26 integration, Top-3 output optimization, startup
validation, and content-addressed model cache. Their lemma predictions are not
equivalent. Install **one** of them on every node; the two ZIPs are alternatives
and cannot be installed side by side.

> **Direct output comparison:** Across 8,827 surface tokens from the public
> Hebrew UD test text, Lex and Tiny emitted the same lemma for **8,121 tokens
> (92.0%)** and different lemmas for **706 tokens (8.0%)**. In other words,
> switching models changed about **1 in 12 lemma outputs** in this test. This is
> an output-disagreement measurement, not an accuracy score: it does not use the
> corpus's gold labels and does not say which model is correct when they differ.

| Variant | Source branch | 35-document bulk, c8 | 100-document bulk, c8 | Best fit |
| --- | --- | ---: | ---: | --- |
| DictaBERT-Lex | `main` | 50.3 docs/s | 51.8 docs/s | Larger model when lemma quality is the priority and has been validated on representative text |
| DictaBERT Tiny, per-channel INT8 | [`dictabert-tiny`](https://github.com/liladler/elasticsearch-analysis-hebrew-plugin/tree/dictabert-tiny) | 181.4 docs/s | 184.2 docs/s | Throughput-first indexing after reviewing the model's different outputs on representative text |

| Lex compared with Tiny | Tokens | Share |
| --- | ---: | ---: |
| Same lemma output | 8,121 | 92.0% |
| Different lemma output | 706 | 8.0% |
| Total compared | 8,827 | 100.0% |

In this test Tiny was about 3.6x faster, but its output was not interchangeable
with Lex: the models produced different indexed lemmas for 8.0% of tokens. Do
not interpret 92.0% agreement as 92.0% accuracy, or the 8.0% disagreement as an
8.0% Tiny error rate. Review the disagreement examples and validate both
variants on representative text, especially domain terms, proper names, and
inflected tokens, before choosing between them.

Throughput was measured locally on ES 9.4.4 with an 8-vCPU, 4-GB Docker
container, a 2-GB heap, 10 lines per document, and concurrency 8. Treat it as a
relative comparison, not a production capacity guarantee.

### Release downloads for ES 9.4.4

- [DictaBERT-Lex plugin](https://github.com/liladler/elasticsearch-analysis-hebrew-plugin/releases/download/v9.4.4/heb-lemmas-embedded-plugin-9.4.4.zip)
- [DictaBERT Tiny plugin](https://github.com/liladler/elasticsearch-analysis-hebrew-plugin/releases/download/v9.4.4/heb-lemmas-embedded-tiny-plugin-9.4.4.zip)

Each release uses the same two filename patterns with its Elasticsearch version.

### Uploading the Lex plugin to Elastic Cloud

The DictaBERT-Lex plugin ZIP is larger than the Elastic Cloud UI's 200 MB
upload limit. Create it through the Elastic Cloud Extensions API using a
direct, publicly accessible download URL instead:

First, create an [Elastic Cloud API key](https://www.elastic.co/docs/deploy-manage/api-keys/elastic-cloud-api-keys)
from **Organization > API keys** in the Elastic Cloud Console. The key needs a
role that can manage the target hosted deployment. Store it in
`CLOUD_API_KEY`; this is a Cloud organization key, not an Elasticsearch API
key created inside the deployment.

```bash
ES_VERSION=9.4.3

curl --fail-with-body --silent --show-error --request POST \
  'https://api.elastic-cloud.com/api/v1/deployments/extensions' \
  --header "Authorization: ApiKey $CLOUD_API_KEY" \
  --header 'Content-Type: application/json' \
  --data "{\
    \"name\": \"heb-lemmas-embedded-plugin\",\
    \"extension_type\": \"plugin\",\
    \"version\": \"${ES_VERSION}\",\
    \"description\": \"DictaBERT-Lex Hebrew analyzer (heb_lemmas + heb_stopwords), ES ${ES_VERSION}\",\
    \"download_url\": \"$(curl --fail --silent --show-error --output /dev/null --write-out '%{redirect_url}' "https://github.com/liladler/elasticsearch-analysis-hebrew-plugin/releases/download/v${ES_VERSION}/heb-lemmas-embedded-plugin-${ES_VERSION}.zip")\"\
  }"
```

Set `ES_VERSION` once to the version being installed. It must exactly match both
the target Elasticsearch version and the value in the plugin's
`plugin-descriptor.properties`. The nested `curl` resolves the GitHub release
redirect and passes its direct download URL to Elastic Cloud. The resolved URL
is temporary, so run the complete command together as shown. The API-created
extension can then be selected from the deployment's Extensions page.

See Elastic's [Extensions API documentation](https://www.elastic.co/docs/deploy-manage/deploy/elastic-cloud/manage-plugins-extensions-through-api)
for listing, updating, and attaching extensions to deployment plans.

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

Note: the model file (`model.onnx`) is not stored in git. It is generated by
`export_model.py`. If you want the prebuilt model, download `model-lex.onnx`
from the GitHub release assets and save it as
`hebrew-lemmatizer-embedded/plugin-lemmas-embedded/src/main/resources/model/model.onnx`.
The DictaBERT Tiny variant uses `model-tiny.onnx` instead; the build verifies
whichever model is present against the committed `model-cache-key.txt`, so the
two are not interchangeable.

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
/path/to/elasticsearch/bin/elasticsearch-plugin install file:///path/to/heb-lemmas-embedded-plugin-9.2.4.zip
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

## Upgrade to a newer ES version (e.g., 9.3)

1. Run `./scripts/build_plugin_linux.sh --es-version <ES>`
3. Reinstall the new zip into ES and restart

   
   
