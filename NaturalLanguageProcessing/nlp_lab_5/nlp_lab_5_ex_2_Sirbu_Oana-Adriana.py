import nltk
from nltk.corpus import wordnet as wn


def get_extended_glosses(synset):
    extended_glosses = set()
    extended_synsets = synset.hypernyms() + synset.hyponyms() + synset.part_meronyms() + synset.part_holonyms() + synset.attributes() + synset.also_sees() + synset.similar_tos()
    for extended_synset in extended_synsets:
        extended_glosses.update(extended_synset.definition().split())
    return extended_glosses


def calculate_score(target_synset, tagged_words, pos_wn, context):
        target_glosses = set(gloss for lemma in target_synset.lemmas() 
                            for gloss in lemma.synset().definition().split())

        score = 0
        
        for word, pos in tagged_words:
            if pos not in ['PRP', 'IN', 'DT', 'CC']:
                word_synsets = wn.synsets(word, pos=pos_wn.get(pos[0].upper()))
                
                for word_synset in word_synsets:
                    word_glosses = set(gloss for lemma in word_synset.lemmas() 
                                    for gloss in lemma.synset().definition().split())
                    extended_glosses = get_extended_glosses(word_synset) 
                    all_glosses = word_glosses.union(extended_glosses)
                    
                    overlap = all_glosses & target_glosses
                    if overlap:
                        score += len(overlap) ** 2
                        common_phrase = ' '.join(overlap)
                        context = context.replace(common_phrase, '###')
        
        return score


def extended_lesk(target_word, context):
    tagged_word = nltk.pos_tag([target_word])[0]
    pos = tagged_word[1]
    
    wn_pos = {'N': 'n', 'V': 'v', 'J': 'a', 'R': 'r'}
    
    target_synsets = wn.synsets(target_word, pos=wn_pos.get(pos[0].upper()))
    
    tagged_words = nltk.pos_tag(nltk.word_tokenize(context))

    filtered_tagged_words = [(word, pos) for word, pos in tagged_words if word != target_word]
    
    best_synset = max(target_synsets, key=lambda synset: calculate_score(synset, filtered_tagged_words, wn_pos, context))

    return best_synset


sentence = "She gazed at the stars shining brightly in the night sky"
sentence_2 = "Yesterday I saw a movie in which roles were played by famous stars"
chosen_word = "stars"
result_synset_1 = extended_lesk(chosen_word, sentence)
result_synset_2 = extended_lesk(chosen_word, sentence_2)

print("Word meaning as computed by extended lesk is:", result_synset_1.definition())
print('-------------------------------------------')
print("Word meaning as computed by extended lesk is:", result_synset_2.definition())
