package org.example;

import org.apache.lucene.analysis.ro.RomanianAnalyzer;
import org.apache.lucene.queryparser.classic.QueryParser;
import org.apache.lucene.queryparser.classic.ParseException;
import org.apache.lucene.document.Document;
import org.apache.lucene.index.IndexReader;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.store.FSDirectory;

import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;


public class Searcher {

    public static void searchQuery(String indexPath, String queryStr) {
    try {
        String normalizedQuery = TextUtils.removeDiacritics(queryStr);

        try (IndexReader reader = DirectoryReader.open(FSDirectory.open(Paths.get(indexPath)))) {
            IndexSearcher searcher = new IndexSearcher(reader);

            QueryParser parser = new QueryParser("contents", new RomanianAnalyzer(TextUtils.getStopwordsSet()));
            Query query = parser.parse(normalizedQuery);
            TopDocs topDocs = searcher.search(query, 5); 

            for (ScoreDoc scoreDoc : topDocs.scoreDocs) {
                Document doc = searcher.storedFields().document(scoreDoc.doc);
                String path = doc.get("path");

                if (path != null) {
                    String filename = new File(path).getName();
                    System.out.println(filename);
                }
            }
        }
    } catch (IOException | ParseException e) {
        e.printStackTrace();
    }
}
}