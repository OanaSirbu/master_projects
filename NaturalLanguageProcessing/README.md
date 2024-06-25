# Text Preprocessing and NLP Techniques

## Table of Contents
1. [Introduction](#introduction)
2. [Text Preprocessing](#text-preprocessing)
    - [Tokenization](#tokenization)
    - [Removing Stopwords](#removing-stopwords)
    - [Stemming and Lemmatization](#stemming-and-lemmatization)
3. [Advanced NLP Techniques](#advanced-nlp-techniques)
    - [Stanford Parser and POS Tagging](#stanford-parser-and-pos-tagging)
    - [WordNet](#wordnet)
    - [Word2Vec](#word2vec)
    - [Relatedness through Extended Lesk](#relatedness-through-extended-lesk)
    - [WordNet++](#wordnet-plus-plus)
    - [WSD with Bayesian Methods](#wsd-with-bayesian-methods)
    - [Ant Colony Optimization](#ant-colony-optimization)
    - [BERT](#bert)

## Introduction
This README provides an overview of text preprocessing techniques and various natural language processing (NLP) methods used for analyzing and processing text data in the 1st year, 2nd semester of the AI master at UniBuc.

## Text Preprocessing

### Tokenization
Tokenization is the process of splitting text into individual words or tokens. It is the first step in text preprocessing. (Lab 1)

### Removing Stopwords
Stopwords are common words (e.g., "the", "is", "in") that are usually removed from text to focus on the important words. (Lab 1)

### Stemming and Lemmatization
- **Stemming:** Reduces words to their root form (e.g., "running" to "run").
- **Lemmatization:** Reduces words to their base or dictionary form (e.g., "better" to "good"). (also Lab 1)

## Advanced NLP Techniques

### Stanford Parser and POS Tagging
The Stanford Parser analyzes the grammatical structure of sentences and assigns parts of speech (POS) tags to each word. (Lab 2-3)

### Parsing algorithms 
Implementations for Well formed substrings tabel and left corner parser can be found in Lab 2-3.

### WordNet
WordNet is a lexical database that groups words into sets of synonyms (synsets) and provides definitions and relations between them. (Lab 4)

### Word2Vec
Word2Vec is a model that represents words in vector space, capturing semantic relationships between words through their co-occurrence in a corpus. (Lab 4)

### Relatedness through Extended Lesk
The Extended Lesk algorithm measures word relatedness by comparing the overlap of dictionary definitions of words. (Lab 5)

### WordNet++
WordNet++ enhances WordNet by incorporating additional lexical resources and semantic relations. (Lab 5)

### WSD with Bayesian Methods
Word Sense Disambiguation (WSD) using Bayesian methods involves probabilistic approaches to determine the correct meaning of a word in context. (Lab 5 - true)

### Ant Colony Optimization
Ant Colony Optimization (ACO) is a computational algorithm inspired by the behavior of ants, used for solving complex optimization problems in NLP. (Lab 6)

### BERT
BERT (Bidirectional Encoder Representations from Transformers) is a pre-trained transformer model that provides deep contextual understanding of language.
