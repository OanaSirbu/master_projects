# Romanian Information Retrieval System

This project provides functionality for indexing and searching documents using Apache Lucene. It supports indexing text-based documents (TXT, PDF, DOCX) and allows searching through the indexed content with specific keywords.

## Project Structure

- `pom.xml`: Maven configuration file for managing dependencies and building the project.
- `src/main/java/org/example/`: Contains the main source code files.
  - `Main.java`: Main class for running the application.
  - `Indexer.java`: Class for indexing documents.
  - `Searcher.java`: Class for searching indexed documents.
  - `TextUtils.java`: Utility class for handling diacritics and stopwords in the Romanian language.

## Classes Overview

- **Main.java**: The entry point of the application. Handles user input for indexing and searching operations.
- **Indexer.java**: Indexes documents by extracting text from supported file formats (TXT, PDF, DOCX) and storing it in a Lucene index.
- **Searcher.java**: Allows searching for a query term in the indexed documents.
- **TextUtils.java**: Provides utility functions for removing diacritics and handling Romanian stopwords in the indexing and search process.

## Setup 

To build the project and create the executable .jar file, use the following command:

```bash
mvn package
 ```

This will generate a docsearch-1.0-SNAPSHOT.jar file in the target directory.

Run the following command to index a directory of documents:

```bash
java -jar target/docsearch-1.0-SNAPSHOT.jar -index -directory <path to docs>
 ```

Run the following command to search the indexed documents for a specific keyword:

```bash
java -jar target/docsearch-1.0-SNAPSHOT.jar -search -directory <path to docs> -query <keyword>
 ```
