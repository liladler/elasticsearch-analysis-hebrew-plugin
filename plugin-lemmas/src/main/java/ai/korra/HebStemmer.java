package ai.korra;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class HebStemmer {
    protected String stem(String term)
    {
        HebDebugger debugger = new HebDebugger();

        debugger.debugPrint("HebStemmer stem started with term: " + term);
        String host = System.getenv("KORRA_HEB_URL");
        
        if (host == null || host.trim().isEmpty()) {
            host = "http://dicta:8000/lemmas";
        };

        debugger.debugPrint("HebStemmer host : " +host);
        // create a client
        
        var client = HttpClient.newBuilder()
                .version(HttpClient.Version.HTTP_1_1)
                .build();

        // create a request
        var request = HttpRequest.newBuilder()
                .uri(URI.create(host))
                .header("accept", "*/*")
                .header("Content-Type", "text/plain;charset=UTF-8")
                .POST(HttpRequest.BodyPublishers.ofString(term))
                .build();

        // use the client to send the request
        try {
            //debug mode
            debugger.debugPrint("HebStemmer api call to dicta sent");
            long startTime = System.nanoTime();
            // send api call to dicta
            var response = client.send(request, HttpResponse.BodyHandlers.ofString());
            // debug mode
            long endTime = System.nanoTime();
            long duration = endTime - startTime;
            long durationInMs = duration / 1000_000_000;
            debugger.debugPrint("HebStemmer api call to dicta run time in seconds: " + durationInMs);
            debugger.debugPrint("HebStemmer api call response body: " + response.body());
            
            return response.body();
        } catch (Exception e) {
            debugger.debugPrint("Heb.stemmer api call Exception while fetching response: " + e);
            return "Exception while fetching response: " + e;
        }
    }
}
