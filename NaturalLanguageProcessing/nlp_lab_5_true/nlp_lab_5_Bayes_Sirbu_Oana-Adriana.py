from nltk.corpus import senseval
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import random
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords


lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

words_to_process = ['interest', 'hard', 'line', 'serve']

output_file = "nlp_lab_5_true/nlp_lab_5_Bayes_Sirbu_Oana-Adriana.txt"

with open(output_file, "w") as file:
    for word in words_to_process:
        instances = list(senseval.instances(f'{word}.pos'))
        random.shuffle(instances)

        data = []
        labels = []
        for instance in instances:
            context = instance.context
            content_words = []
            for item in context:
                if len(item) == 2:
                    word, pos = item
                    if pos.startswith(('N', 'V', 'J', 'R')) and word.lower() not in stop_words:
                        content_words.append(lemmatizer.lemmatize(word.lower()))
            target_word = instance.word[:len(instance.word) - 2]
            label = instance.senses[0]
            target_index = instance.position
            left_context = content_words[max(0, target_index - 3):target_index]
            right_context = content_words[target_index + 1:min(target_index + 4, len(content_words))]
            if left_context or right_context:
                sentence = ' '.join(left_context + [target_word] + right_context)
                data.append(sentence)
                labels.append(label)

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform(data)

        X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=1)

        clf = MultinomialNB()
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        file.write(f"\n")
        file.write("Predictions vs Correct Labels:\n")
        for i in range(len(predictions)):
            file.write(f"Prediction: {predictions[i]} | Correct Label: {y_test[i]} | Correct: {predictions[i] == y_test[i]}\n")
        file.write(f"Accuracy: {accuracy}\n\n")

        print(f"\n")
        print("Predictions vs Correct Labels:")
        for i in range(len(predictions)):
            print(f"Prediction: {predictions[i]} | Correct Label: {y_test[i]} | Correct: {predictions[i] == y_test[i]}")
        print(f"Accuracy: {accuracy}\n")