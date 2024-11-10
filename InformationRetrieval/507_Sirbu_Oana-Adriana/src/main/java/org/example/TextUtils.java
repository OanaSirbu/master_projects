package org.example;

import org.apache.lucene.analysis.CharArraySet;
import org.apache.lucene.analysis.ro.RomanianAnalyzer;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import java.text.Normalizer;
import java.util.regex.Pattern;

public class TextUtils {

    private static final Pattern DIACRITICS_PATTERN = Pattern.compile("\\p{InCombiningDiacriticalMarks}+");

    public static String removeDiacritics(String input) {
        String normalized = Normalizer.normalize(input, Normalizer.Form.NFD);
        return DIACRITICS_PATTERN.matcher(normalized).replaceAll("");
    }

    public static List<String> generateDiacriticFree(List<String> stopwords) {
        List<String> diacriticFreeStopwords = new ArrayList<>();
        for (String word : stopwords) {
            diacriticFreeStopwords.add(removeDiacritics(word));
        }
        return diacriticFreeStopwords;
    }

    public static CharArraySet getStopwordsSet() {
        CharArraySet luceneStopwords = RomanianAnalyzer.getDefaultStopSet();
        Set<String> stopwords = new HashSet<>(luceneStopwords.size());

        for (Object obj : luceneStopwords) {
            if (obj instanceof char[]) {
                stopwords.add(new String((char[]) obj));
            } else {
                stopwords.add(obj.toString());
            }
        }

        stopwords.addAll(generateDiacriticFree(new ArrayList<>(stopwords)));

        return new CharArraySet(stopwords, true);
    }
}
