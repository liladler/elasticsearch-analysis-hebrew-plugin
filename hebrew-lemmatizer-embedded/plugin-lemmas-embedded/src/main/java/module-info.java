/**
 * Hebrew Lemmatizer classic plugin module.
 */
module ai.korra.heb.lemmatizer {
    requires org.elasticsearch.server;
    requires org.apache.lucene.core;
    requires org.apache.lucene.analysis.common;

    requires com.microsoft.onnxruntime;
    requires com.google.gson;

    exports ai.korra;
}
