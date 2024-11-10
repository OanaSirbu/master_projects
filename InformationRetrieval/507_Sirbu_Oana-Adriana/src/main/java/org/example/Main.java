package org.example;


public class Main {
    public static void main(String[] args) {
        if (args.length < 3) {
            System.out.println("Usage:");
            System.out.println("To index: java -jar docsearch-1.0-SNAPSHOT.jar -index -directory <path to docs>");
            System.out.println("To search: java -jar docsearch-1.0-SNAPSHOT.jar -search -directory <path to docs> -query <keyword>");
            return;
        }

        String docsPath = null;
        String query = null;
        boolean isIndexing = false;
        boolean isSearching = false;

        for (int i = 0; i < args.length; i++) {
            switch (args[i]) {
                case "-index":
                    isIndexing = true;
                    break;
                case "-search":
                    isSearching = true;
                    break;
                case "-directory":
                    if (i + 1 < args.length) {
                        docsPath = args[i + 1];
                    }
                    break;
                case "-query":
                    if (i + 1 < args.length) {
                        query = args[i + 1];
                    }
                    break;
            }
        }

        if (docsPath == null) {
            System.out.println("Error: Document directory not specified.");
            return;
        }

        try {
            String indexPath = "index";

            if (isIndexing) {
                Indexer.indexDocuments(docsPath, indexPath);
            } else if (isSearching) {
                if (query == null) {
                    System.out.println("Error: Search query not specified.");
                    return;
                }
                Searcher.searchQuery(indexPath, query);
            } else {
                System.out.println("Error: Invalid operation. Use -index or -search.");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
