import nltk
import wikipedia
import string
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt

# Exercise 1
wikipedia.set_lang('en')
article_title = 'Interstellar medium'
article = wikipedia.page(article_title)
content = article.content[:200]

print("Title:", article_title)
print("First 200 words:", content)

sentences = nltk.sent_tokenize(article.content)
for i, sentence in enumerate(sentences[:20]):
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(sentence)
    pos_tags = nltk.pos_tag(words)
    print(f"Sentence {i+1}: {pos_tags}")


# Exercise 2
def words_with_pos(text, pos_tag):
    sentences = nltk.sent_tokenize(text)
    
    words_with_pos_tag = []
    
    for sentence in sentences:
        sentence = sentence.translate(str.maketrans('', '', string.punctuation))
        
        words = nltk.word_tokenize(sentence)
        tagged_words = nltk.pos_tag(words)
    
        for word, tag in tagged_words:
            if tag == pos_tag:
                words_with_pos_tag.append(word)
    
    return words_with_pos_tag

def words_with_any_pos(text, pos_tags):
    words_with_any_pos_tags = []
    
    for pos_tag in pos_tags:
        words = words_with_pos(text, pos_tag)
        words_with_any_pos_tags.extend(words)
    
    return words_with_any_pos_tags

article_text = article.content.translate(str.maketrans('', '', string.punctuation))

print("Nouns: ", words_with_pos(article_text, 'NN'))

print("Verbs in gerund: ", words_with_pos(article_text, 'VBG'))

print("Both NNs and VBGs: ", words_with_any_pos(article_text, ['NN', 'VBG']))


# Exercise 3
noun_tags = ['NN', 'NNS', 'NNP', 'NNPS']
verb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

nouns = words_with_any_pos(article_text, noun_tags)
verbs = words_with_any_pos(article_text, verb_tags)

total_words = nltk.word_tokenize(article_text)
specific_words = nouns + verbs
percentage_specific_words = (len(specific_words) / len(total_words)) * 100
print(percentage_specific_words)


# Exercise 4
lemmatizer = WordNetLemmatizer()

def lemmatize_with_pos(word, pos):
    if pos.startswith('J'):
        return lemmatizer.lemmatize(word, pos='a')
    elif pos.startswith('V'):
        return lemmatizer.lemmatize(word, pos='v')
    elif pos.startswith('R'):
        return lemmatizer.lemmatize(word, pos='r')
    elif pos.startswith('N'):
        return lemmatizer.lemmatize(word, pos='n')
    else:
        return lemmatizer.lemmatize(word)

distinct_results = set()

N = 7

for sentence in sentences[:30]:
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    words = nltk.word_tokenize(sentence)
    tagged_words = nltk.pos_tag(words)

    for word, pos in tagged_words:
        # Lemmatize without POS tagging
        simple_lemmatization = lemmatizer.lemmatize(word)
        # Lemmatize with POS tagging
        lemmatization_with_pos = lemmatize_with_pos(word, pos)
        if simple_lemmatization != lemmatization_with_pos:
            distinct_results.add((word, pos, simple_lemmatization, lemmatization_with_pos))

max_word_length = max(len(word) for word, _, _, _ in distinct_results)
max_pos_length = max(len(pos) for _, pos, _, _ in distinct_results)
max_simple_lemmatization_length = max(len(simple_lemmatization) for _, _, simple_lemmatization, _ in distinct_results)
max_lemmatization_with_pos_length = max(len(lemmatization_with_pos) for _, _, _, lemmatization_with_pos in distinct_results)

print(f"{'Original word':<{max_word_length}} | {'POS':<{max_pos_length}} | {'Simple lemmatization':<{max_simple_lemmatization_length}} | {'Lemmatization with POS':<{max_lemmatization_with_pos_length}}")
for word, pos, simple_lemmatization, lemmatization_with_pos in distinct_results:
    print(f"{word:<{max_word_length}} | {pos:<{max_pos_length}} | {simple_lemmatization:<{max_simple_lemmatization_length}} | {lemmatization_with_pos:<{max_lemmatization_with_pos_length}}")


# Exercise 5
def clean_text(text):
    translator = str.maketrans('', '', string.punctuation)
    
    cleaned_text = text.translate(translator)
    
    return cleaned_text

cleaned_text = clean_text(article_text)

def count_words_by_pos(text):
    words = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(words)
    
    pos_counts = {}
    
    for word, pos in tagged_words:
        if pos in pos_counts:
            pos_counts[pos] += 1
        else:
            pos_counts[pos] = 1
    
    return pos_counts

pos_counts = count_words_by_pos(cleaned_text)

sorted_pos_counts = sorted(pos_counts.items(), key=lambda x: x[1], reverse=True)

plt.figure(figsize=(10, 6))
plt.bar([pos[0] for pos in sorted_pos_counts], [pos[1] for pos in sorted_pos_counts], color='skyblue')
plt.xlabel('Part of Speech')
plt.ylabel('Number of Words')
plt.title('Number of Words for Each Part of Speech')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Exercise 6
grammar = nltk.CFG.fromstring("""  
S -> NP VP
NP -> NN | NN VP | Det NN | JJ NP | NN CC NP
VP -> VB | VB NP
                         
NN -> 'person' |'researcher' | 'supernovas' | 'galaxies'
VB -> 'studies' | 'looks'
JJ -> 'bright' | 'great'
CC -> 'and'
Det -> 'the'
""")

sentence = ["the", "researcher", "studies", "bright", "supernovas", "and", "great", "galaxies"]
desc_parser = nltk.RecursiveDescentParser(grammar)
for branch in desc_parser.parse(sentence):
    print(branch)


# Exercise 7
sr_parser = nltk.ShiftReduceParser(grammar)

sentence_equal = ["researcher", "looks"]
sentence_different = ["the", "researcher", "studies", "bright", "supernovas"]

rd_trees_equal = list(desc_parser.parse(sentence_equal))
sr_trees_equal = list(sr_parser.parse(sentence_equal))

rd_trees_different = list(desc_parser.parse(sentence_different))
sr_trees_different = list(sr_parser.parse(sentence_different))


def compare_parse_trees(rd_trees, sr_trees, sentence):
    if rd_trees == sr_trees and rd_trees != [] and sr_trees != []:
        print(f"For sentence '{' '.join(sentence)}', trees produced by recursive descent parsing and shift-reduce parsing are equal.")
    elif rd_trees == [] and sr_trees == []:
        print(f"For sentence '{' '.join(sentence)}', both of the trees produced by recursive descent parsing and shift-reduce parsing are null.")
    else:
        print(f"For sentence '{' '.join(sentence)}', trees produced by recursive descent parsing and shift-reduce parsing are different.")

# Call the function for the equal sentence
compare_parse_trees(rd_trees_equal, sr_trees_equal, sentence_equal)

# Call the function for the different sentence
compare_parse_trees(rd_trees_different, sr_trees_different, sentence_different)


