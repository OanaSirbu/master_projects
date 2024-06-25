from nltk.corpus import wordnet as wn


# Ex 1
def print_all_glosses(word):
    synsets = wn.synsets(word)
    if synsets:
        print(f"Glosses for '{word}':")
        for synset in synsets:
            print(f"- {synset.definition()}")
    else:
        print(f"No senses found for this word")

word = "star"
print(f'\n')
print_all_glosses(word)

print(f"\n")

# Ex 2
def find_synset_and_print_gloss(word_1, word_2): 
    synsets_word_1 = wn.synsets(word_1)
    synsets_word_2 = wn.synsets(word_2)
    
    common_synsets = set(synsets_word_1).intersection(synsets_word_2)
    
    if common_synsets:
        print("Common synsets: ", common_synsets)
        print("Gloss for the common synsets:")
        for synset in common_synsets:
            print("-" * 30)
            print("Def:", synset.definition())
    else:
        print("No common synsets found. Try different words.")

word1 = "star"
word2 = "asterisk"
print_all_glosses("asterisk")
find_synset_and_print_gloss(word1, word2)


# Ex 3
def get_holonyms_meronyms(synset):
    holonyms = []
    meronyms = []
    
    holonyms.extend(synset.member_holonyms())
    holonyms.extend(synset.substance_holonyms())
    holonyms.extend(synset.part_holonyms())

    meronyms.extend(synset.member_meronyms())
    meronyms.extend(synset.substance_meronyms())
    meronyms.extend(synset.part_meronyms())

    return holonyms, meronyms

def print_holonyms_meronyms(holonyms, meronyms):
    print("Holonyms:")
    for holonym in holonyms:
        print("-", holonym.name().split('.')[0])
    print()

    print("Meronyms:")
    for meronym in meronyms:
        print("-", meronym.name().split('.')[0])
    print()

word = "star.n.01"
synset = wn.synset(word)

holonyms, meronyms = get_holonyms_meronyms(synset)

print(f'\n')
print_holonyms_meronyms(holonyms, meronyms)

print(f'\n')
print("All Holonyms and Meronyms:")
print("Holonyms:", [', '.join(holonym.lemma_names()) for holonym in holonyms])
print("Meronyms:", [', '.join(meronym.lemma_names()) for meronym in meronyms])


# Ex 4
def get_hypernym_path(synset):
    hypernym_paths = synset.hypernym_paths()  
    for path in hypernym_paths:
        hypernym_names = [hypernym.name().split('.')[0] for hypernym in path]
        print(" -> ".join(hypernym_names))


print(f'\n')
word = "star"
synsets = wn.synsets(word)
if synset:
    print(f"Hypernym paths for '{word}':")
    for synset in synsets:
        get_hypernym_path(synset)
else:
    print(f"No synsets found for '{word}'.")


# Ex 5
def shortest_path_length(synset1, synset2):
    shortest_length = float('inf')
    shortest_hypernyms = []

    for hypernym1 in synset1.hypernyms():
        for hypernym2 in synset2.hypernyms():
            path_length = synset1.shortest_path_distance(hypernym1) + synset2.shortest_path_distance(hypernym2)
            if path_length < shortest_length:
                shortest_length = path_length
                shortest_hypernyms = [(hypernym1, hypernym2)]
            elif path_length == shortest_length:
                shortest_hypernyms.append((hypernym1, hypernym2))

    return shortest_hypernyms

synset1 = wn.synset('car.n.01')
synset2 = wn.synset('bicycle.n.01')
print(f'\n')
print(shortest_path_length(synset1, synset2))


# Ex 6
def sort_synsets(first_synset, synsets_list):
    similarity_scores = [(synset, first_synset.path_similarity(synset)) for synset in synsets_list]
    similarity_scores.sort(key=lambda elem: elem[1], reverse=True)
    return similarity_scores

word = "cat"
synsets_cat = wn.synsets(word)[:1] 
synsets_list = wn.synsets("animal")[:1] + wn.synsets("tree")[:1] + wn.synsets("house")[:1] + wn.synsets("object")[:1] + wn.synsets("public_school")[:1] + wn.synsets("mouse")[:1]

print(f"\n")

sorted_synsets = sort_synsets(synsets_cat[0], synsets_list)
for synset, similarity_score in sorted_synsets:
    print(f"{synset.name()} - Similarity: {similarity_score}")


# Ex 7
def find_common_meronyms(synset):
    meronyms = set()

    part_meronyms = synset.part_meronyms()
    substance_meronyms = synset.substance_meronyms()
    member_meronyms = synset.member_meronyms()
    
    meronyms.update(part_meronyms)
    meronyms.update(substance_meronyms)
    meronyms.update(member_meronyms)
    
    for meronym in list(meronyms):
        list_of_meronyms = find_common_meronyms(meronym)
        meronyms.update(list_of_meronyms)

    return meronyms

def check_indirect_meronyms(synset1, synset2):
    common_meronyms = find_common_meronyms(synset1).intersection(find_common_meronyms(synset2))
    return bool(common_meronyms)

synset1 = wn.synset('tree.n.01')
synset2 = wn.synset('root.n.01')
print(f'\n')
print(check_indirect_meronyms(synset1, synset2))

# Ex 8
def print_syns_ants(adjective):
    synsets = wn.synsets(adjective, pos=wn.ADJ)
    for synset in synsets:
        print("Sense:", synset.name())
        print("Gloss:", synset.definition())
        synonyms = []
        antonyms = []

        for lemma in synset.lemmas():
            synonyms.append(lemma.name())
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())

        print("Synonyms:", synonyms)
        print("Antonyms:", antonyms)
        print()

adjective = "smart"
print(f'\n')
print("My chosen adjective is: ", adjective)
print_syns_ants(adjective)