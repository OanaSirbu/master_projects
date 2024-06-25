import wikipedia
from nltk.tokenize import word_tokenize
import gensim.downloader as api
import re
from nltk.corpus import stopwords
import heapq
import spacy
from sklearn.cluster import KMeans
import numpy as np


# Ex 1
google_model = api.load("word2vec-google-news-300")
print("\nThe total number of words in the model's vocabulary:", len(google_model))

article_title = "Supernova"
text = wikipedia.page(article_title).content

def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)
    return text

cleaned_text = clean_text(text)
words = word_tokenize(cleaned_text)[:500]

extracted_text = ' '.join(words)

# print("Extracted text:\n", extracted_text)


# Ex 2
not_in_vocab = [word for word in words if word not in google_model and word not in stopwords.words('english')]
print("Words that are out of vocabulary:", not_in_vocab)


# Ex 3
closest_words = []
farthest_words = []


def get_similarity(word_1, word_2):
    if word_1 in google_model and word_2 in google_model:
        return google_model.similarity(word_1, word_2)
    else:
        return None


for i in range(len(words)-1):
    for j in range(i + 1, len(words)):
        word_1 = words[i]
        word_2 = words[j]

        if word_1 == word_2:
            continue

        similarity_dist = get_similarity(word_1, word_2)

        if similarity_dist is not None:
            heapq.heappush(closest_words, (-similarity_dist, word_1, word_2))
            heapq.heappush(farthest_words, (similarity_dist, word_1, word_2))


closest_pair = heapq.heappop(closest_words)
farthest_pair = heapq.heappop(farthest_words)

print(f"The similarity for the closest pair computed is: {-closest_pair[0]}: {closest_pair[1]} & {closest_pair[2]}")
print(f"And the one for the farthest pair is: {farthest_pair[0]}: {farthest_pair[1]} & {farthest_pair[2]}")


# Ex 4
nlp = spacy.load("en_core_web_sm")
doc = nlp(extracted_text)
named_entities = [ent.text for ent in doc.ents]

print("All named entities:", named_entities)

filtered_words = set([word for word in words if word.lower() not in stopwords.words('english') and word.lower() in google_model])

similar_words = {}
for entity in named_entities:
    if len(entity.split()) >= 3:  
        print(f"Skipping too specific entity (we won't find any match for them): {entity}")
        continue
    similar_words[entity] = []
    for word in list(filtered_words):
        try:
            similarity_dist = google_model.similarity(entity.lower(), word.lower())
            similar_words[entity].append((similarity_dist, word))
        except KeyError:
            pass


for entity, similar_words_list in similar_words.items():
    similar_words_list.sort(reverse=True)
    print(f"Current named entity: {entity}")
    if similar_words_list:
        for i in range(min(5, len(similar_words_list))):
            similar_word = similar_words_list[i][1]
            similarity = similar_words_list[i][0]
            print(f"Original word - similar): {similar_word}, similarity: {similarity}")
            print(f"Lowercase word - similar): {similar_word.lower()}, similarity: {similarity}")
    else:
        print("No similar words found! :(")
    print()


# Ex 5
word_vectors = [google_model[word] for word in filtered_words]

X = np.array(word_vectors)

kmeans = KMeans(n_clusters=5)
kmeans.fit(X)

cluster_labels = kmeans.labels_

clusters = {i: [] for i in range(5)}
for i, word in enumerate(filtered_words):
    clusters[cluster_labels[i]].append(word)

for cluster_id, words_in_cluster in clusters.items():
    print(f"Cluster {cluster_id + 1}:")
    print(words_in_cluster)
    print()