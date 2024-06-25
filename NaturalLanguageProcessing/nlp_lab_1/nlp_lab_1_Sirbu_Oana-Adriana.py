import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder
import string
from urllib import request
import re
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, LancasterStemmer, SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from num2words import num2words

# 1.
url_addr = "https://www.gutenberg.org/cache/epub/73057/pg73057.txt"
response = request.urlopen(url_addr)
text = response.read().decode('utf8')

path = "nlp_text_lab_1.txt"
with open(path, "w", encoding="utf-8") as file:
    file.write(text)

print(len(text))
print(text[:90])

# 2.
start_index = text.find("CHAPTER I")
text_without_header = text[start_index+len("CHAPTER I."):].lstrip()
print(text_without_header)

# 3.
sentences = sent_tokenize(text_without_header)
no_of_sentences = len(sentences)
print(no_of_sentences)

total_words = sum(len(word_tokenize(sentence)) for sentence in sentences)
average_len = total_words / no_of_sentences if no_of_sentences > 0 else 0
print(average_len)

# 4.
all_words = word_tokenize(text_without_header)

bigram_finder = BigramCollocationFinder.from_words(all_words)
trigram_finder = TrigramCollocationFinder.from_words(all_words)

all_bigrams = bigram_finder.ngram_fd.items()
all_trigrams = trigram_finder.ngram_fd.items()

all_unique_bigrams = set(bigram_finder.ngram_fd)
all_uique_trigrams = set(trigram_finder.ngram_fd)

print(all_unique_bigrams)

# 5.
filtered_words = [word.lower() for word in all_words if re.match("^[a-zA-Z0-9]*$", word)]
print(filtered_words)

# 6.
def extract_most_freq(lw, N):
    words_counter = Counter(lw)
    return words_counter.most_common(N)

print(extract_most_freq(filtered_words, 5))

# 7.
stop_words = set(stopwords.words('english'))
lws = [word for word in filtered_words if word not in stop_words]
print(lws)
print(extract_most_freq(lws, 5))

# 8.
ps = PorterStemmer()

list_stemmed = [ps.stem(ws) for ws in lws]
print(list_stemmed[:200])

# we can see with our own eyes that there are some words in the list of stemmed words that do not
# appear in the dictionary, as they are missing one last letter (e.g. 'e' from 'attitud' or 'peopl')
# I will also use a more automate way to find these words and print them
not_in_dictionary = [word for word in list_stemmed if not wordnet.synsets(word)]
print(not_in_dictionary)

# 9.
ls = LancasterStemmer()
snb = SnowballStemmer('english')

print(ps.stem('being'))
print(ls.stem('being'))
print(snb.stem('being'))

stemming_res = {}
NW = 500

for word in lws:
    porter_stem = ps.stem(word)
    lancaster_stem = ls.stem(word)
    snowball_stem = snb.stem(word)
    
    if porter_stem != lancaster_stem or lancaster_stem != snowball_stem:
        stemming_res[word] = (porter_stem, lancaster_stem, snowball_stem)


max_stem_length = max(len(stem) for stems in stemming_res.values() for stem in stems) + 2

print(f"{'Porter':<{max_stem_length}} | {'Lancaster':<{max_stem_length}} | {'Snowball':<{max_stem_length}}")
print(f"{'-' * (3 * max_stem_length + 2)}")
for word, stems in stemming_res.items():
    print(f"{stems[0]:<{max_stem_length}} | {stems[1]:<{max_stem_length}} | {stems[2]:<{max_stem_length}}")

# 10.
wl = WordNetLemmatizer()

comparison = {}

max_stem_length = 0
max_lemma_length = 0

for word in lws:
    snb_word = snb.stem(word)
    max_stem_length = max(max_stem_length, len(snb_word))

    wordnet_word = wl.lemmatize(word)
    max_lemma_length = max(max_lemma_length, len(wordnet_word))

    if snb_word != wordnet_word:
        comparison[word] = (snb_word, wordnet_word)


print(f"{'Snowball':<{max_stem_length}} | {'WordNetLemmatizer':<{max_lemma_length}}")
print(f"{'-' * (max_stem_length + max_lemma_length + 2)}")

for word, (snb_word, wordnet_word) in comparison.items():
    print(f"{snb_word:<{max_stem_length}} | {wordnet_word:<{max_lemma_length}}")

# 11.
lemma_words = [wl.lemmatize(word) for word in lws]
lemma_counter = Counter(lemma_words)

N = 10 

for lemma_word, count in lemma_counter.most_common(N):
    print(f"{lemma_word}: {count}")

# 12.
list_change_index = []
no_of_changes = 0
N = 10

for i, word in enumerate(lws):
    if str(word).isdigit():
        lws[i] = num2words(str(word))
        list_change_index.append(i)
        no_of_changes += 1

print("Total number of changes is: ", no_of_changes)
lws_N_changes = []
for idx in range(N):
    lws_N_changes.append(lws[list_change_index[idx]])
print(lws_N_changes)

# 13.
def find_ngrams(W, N):
    idxs = [i for i, word in enumerate(lws) if word.lower() == W.lower()]
    
    for idx in idxs:
        if N % 2 != 0:
            s_idx = max(0, idx - (N // 2))
            e_idx = min(len(lws), idx + (N // 2) + 1)
            
        else:
            s_idx = max(0, idx - (N // 2) + 1)
            e_idx = min(len(lws), idx + (N // 2) + 1)
            
        ngram_words = lws[s_idx:e_idx]
        print(ngram_words)

# if we would have wanted for N % 2 == 0, the chosen word to be on the even position, then we would modify the code as follows:
        # else:
        #     s_idx = max(0, idx - (N // 2))
        #     e_idx = min(len(lws), idx + (N // 2))

print(find_ngrams('boundary', 5))

def ngrams_in_same_sentence(W, N):
    words_in_sentences = [[word.lower() for word in word_tokenize(sentence) if word.isalnum() and word.lower() not in stopwords.words('english')] for sentence in sentences]

    ngrams = []
    for sent_words in words_in_sentences:
        idxs = [i for i, word in enumerate(sent_words) if word.lower() == W.lower()]
        for idx in idxs:
            if N % 2 != 0:
                s_index = max(0, idx - (N // 2))
                e_index = min(len(sent_words), idx + (N // 2) + 1)
            else:
                s_index = max(0, idx - (N // 2) + 1)
                e_index = min(len(sent_words), idx + (N // 2) + 1)
            ngram = sent_words[s_index:e_index]
            ngrams.append(ngram)

    for ngram in ngrams:
        print(ngram)

print(ngrams_in_same_sentence('dignity', 4))


