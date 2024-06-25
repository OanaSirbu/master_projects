from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.wsd import lesk


def get_gloss_intersection(gloss_1, gloss_2):
    stop_words = set(stopwords.words('english'))
    words_gloss_1 = {word.lower() for word in word_tokenize(gloss_1) if word.lower() not in stop_words}
    words_gloss_2 = {word.lower() for word in word_tokenize(gloss_2) if word.lower() not in stop_words}

    return sum(1 for word in words_gloss_1 if word in words_gloss_2)


def my_lesk(word, context):
    synsets = wordnet.synsets(word)
    stop_words = set(stopwords.words('english'))
    context_words = {word.lower() for word in word_tokenize(context) if word.lower() not in stop_words}

    best_def = max(synsets, key=lambda synset: sum(get_gloss_intersection(definition, ' '.join(context_words)) 
                                                   for definition in synset.definition().split(';')))
    
    return best_def


text = "She gazed at the stars shining brightly in the night sky"
word = "stars"


def check_lesk(lesk_alg, word, text):
    word_meaning = lesk_alg(word, text)
    print(f"For {lesk_alg.__name__}")
    if word_meaning:
        print(f"The best word meaning in the provided context is: \n {word_meaning}")
        print(f"With the associated definition: \n {word_meaning.definition()}")
    else:
        print("Nothing found")
    return '-----------------------------------'

print(check_lesk(my_lesk, word, text))
print(check_lesk(lesk, word_tokenize(text), word))