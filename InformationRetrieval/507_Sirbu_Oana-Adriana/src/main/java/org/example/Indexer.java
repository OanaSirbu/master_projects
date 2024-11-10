package org.example;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.ro.RomanianAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;
import org.apache.poi.xwpf.usermodel.XWPFDocument;  
import org.apache.pdfbox.pdmodel.PDDocument;   
import org.apache.pdfbox.text.PDFTextStripper;
import org.apache.tika.exception.TikaException;
import org.xml.sax.SAXException;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.IOException;
import java.nio.file.Paths;
import java.util.Iterator;

public class Indexer {

    public static void indexDocuments(String docsPath, String indexPath) throws Exception {
        Analyzer analyzer = new RomanianAnalyzer(getStopwordsSet());
        Directory indexDir = FSDirectory.open(Paths.get(indexPath));
        
        IndexWriterConfig config = new IndexWriterConfig(analyzer);
        config.setOpenMode(IndexWriterConfig.OpenMode.CREATE);  
        
        IndexWriter writer = new IndexWriter(indexDir, config);

        File folder = new File(docsPath);
        File[] files = folder.listFiles();
        if (files != null) {
            for (File file : files) {
                if (file.isFile()) {
                    indexFile(writer, file);  
                }
            }
        }
        writer.close();
    }

    private static String extractText(File file) throws IOException, TikaException, SAXException {
        String fileName = file.getName();
        String extractedText = "";

        if (fileName.endsWith(".txt")) {
            extractedText = extractTextFromTXT(file);
        } 
        else if (fileName.endsWith(".pdf")) {
            extractedText = extractTextFromPDF(file);
        } 
        else if (fileName.endsWith(".docx")) {
            extractedText = extractTextFromDOCX(file);
        }

        if (extractedText.isEmpty()) {
            System.err.println("Warning: No content extracted from file: " + file.getName());
        }

        return extractedText;
    }

    private static String extractTextFromTXT(File file) throws IOException {
        StringBuilder content = new StringBuilder();

        try (BufferedReader reader = new BufferedReader(new FileReader(file))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line).append(System.lineSeparator());  
            }
        }

        return content.toString();  
    }

    private static String extractTextFromPDF(File file) throws IOException {
        PDDocument document = PDDocument.load(file);
        PDFTextStripper stripper = new PDFTextStripper();
        
        String text = stripper.getText(document);
        
        document.close();
        return text;
    }

    private static String extractTextFromDOCX(File file) throws IOException {
        FileInputStream fis = new FileInputStream(file);
        XWPFDocument doc = new XWPFDocument(fis);
        StringBuilder text = new StringBuilder();

        Iterator<org.apache.poi.xwpf.usermodel.XWPFParagraph> paragraphs = doc.getParagraphsIterator();
        while (paragraphs.hasNext()) {
            text.append(paragraphs.next().getText()).append("\n");
        }

        doc.close();
        fis.close();
        return text.toString();
    }

    private static void indexFile(IndexWriter writer, File file) throws IOException, TikaException, SAXException {
        String text = extractText(file);  

        if (text == null || text.isEmpty()) {
            System.out.println("Skipping file due to empty or unsupported content: " + file.getAbsolutePath());
            return;
        }

        String normalizedText = TextUtils.removeDiacritics(text);

        Document doc = new Document();
        doc.add(new TextField("contents", normalizedText, Field.Store.YES));  
        doc.add(new TextField("path", file.getAbsolutePath(), Field.Store.YES));  
        writer.addDocument(doc);
    }

    private static CharArraySet getStopwordsSet() {
        return TextUtils.getStopwordsSet();
    }
}
