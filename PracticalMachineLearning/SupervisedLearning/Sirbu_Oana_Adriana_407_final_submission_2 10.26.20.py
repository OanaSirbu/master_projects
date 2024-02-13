# In the first few lines, all the modules that were used to solve the classification task were imported
import pandas as pd # used to easily manipulate data
import string # used to access all ASCII characters
from sklearn.feature_extraction.text import CountVectorizer # used to extract features from text
from sklearn.svm import LinearSVC # second best model trained 
from sklearn.preprocessing import MaxAbsScaler # used to scale features
from sklearn.metrics import classification_report # used to display evaluation scores
import matplotlib.pyplot as plt # tool used to visualize data through graphs and histrograms
from sklearn.metrics import confusion_matrix # used to create a confusion matrix 
import seaborn as sns # used to illustrate heatmaps


# by using pandas library, data available for training, validation and test phases is stored
# into the following Dataframes 
train_data = pd.read_json("train.json")
validation_data = pd.read_json("validation.json")
test_data = pd.read_json("test.json")


# a new column was added, in order to merge the information from those two sentences
# this was used for data cleaning and further for model training
train_data['combined_text'] = train_data['sentence1'] + ' ' + train_data['sentence2']
validation_data['combined_text'] = validation_data['sentence1'] + ' ' + validation_data['sentence2']
test_data['combined_text'] = test_data['sentence1'] + ' ' + test_data['sentence2']


# custom function created to 'clean' the sentences
def clean_text(text):
    # first, punctuation was removed from the text
    text_without_punct = ''.join(ch for ch in text if ch not in string.punctuation)
    # then, by running different test, it was observed that this special quote is not removed
    # therefore a manual elimination feature was added through the next line of code
    text_without_special_char = text_without_punct.replace("„", "")
    # in the last step, all text was converted to lower case letters
    # so that the algorithm will recognize the same words even if one is capitalized
    cleaned_text = text_without_special_char.lower()
    return cleaned_text


# although it could have been passed as an argument to the preprocessing step of the Count Vectorizer instance
# this function was manually applied to the 'combined_text' column because it was easier for debugging
train_data['clean_combined_text'] = train_data['combined_text'].apply(clean_text)
validation_data['clean_combined_text'] = validation_data['combined_text'].apply(clean_text)
test_data['clean_combined_text'] = test_data['combined_text'].apply(clean_text)

# a check to see if the column contains the expected information
# (positive result)
print(test_data['clean_combined_text'])

# the training labels are passed into y_train variable
# and also the column containing the clean set of the 2 sentences is passed to train_vocab
y_train = train_data['label']  
train_vocab = train_data['clean_combined_text']

# an instance of MaxAbsScaler() is created
BoW_features_scaler = MaxAbsScaler()

# an instance of CountVectorizer is created in order to take into account all the sentences 
# across the four labels and create a vocabulary containing each unique word in the entire available text
# min_df=2 means it will not add to the vocabulary words that do not appear at least twice across the entire text
# ngram_range=(1,2) makes that, in addition to single words, to also use combinations of 2 neighbouring words
BoW_vectorizer = CountVectorizer(ngram_range=(1,2), min_df=2)

# the features are first extracted by fitting the BoW_vectorizer. We will work with these features from now
# they are also scaled, in order to help the optimization algorithms of the models to converge easier
X_train_BoW = BoW_vectorizer.fit_transform(train_vocab)
X_train_BoW_scaled = BoW_features_scaler.fit_transform(X_train_BoW)

# same steps as for the training data are followed for the test data for consistency
# a remark: identic instances of the scaler and of the vectorizer were used 
test_vocab = test_data['clean_combined_text']
X_test_BoW = BoW_vectorizer.transform(test_vocab)
X_test_BoW_scaled = BoW_features_scaler.transform(X_test_BoW)

# another print to check the results
print(X_test_BoW_scaled)

# after tuning its parameters, the best Linear SVC model for this task is created
# then fitted on the training data
linear_svc_model = LinearSVC(C=1, max_iter=1000)
linear_svc_model.fit(X_train_BoW, y_train)

# by using the previous fitting, we extracted the predicted results made on unseen data
y_test_predicted = linear_svc_model.predict(X_test_BoW)

# printed the results
print(y_test_predicted)
# Result: [3 2 3 ... 2 2 2]

# save the predictions into a csv file that contains 2 columns
# guid column that contains an ID for the data to be recognized
# and a label column that contains the predictions
sub = pd.DataFrame({'guid': test_data['guid'], 'label': y_test_predicted})
sub.to_csv('submission_test_16_final.csv', index=False)

# the order of the steps used to achieve those results is a little bit changed
# during the project, the models were initially tested on the validation data
# and this is what will follow here.
# validation labels are passed to the y_val 
y_val = validation_data['label']

# the same preprocessing steps are applied to the validation data
val_vocab = validation_data['clean_combined_text']
X_val_BoW = BoW_vectorizer.transform(val_vocab)
X_val_BoW_scaled = BoW_features_scaler.transform(X_val_BoW)

# the model that is fitted on the training data is used to make new predictions
y_val_predicted_logisticr = linear_svc_model.predict(X_val_BoW)

# because the validation data contains labels for the text documents,
# a classification report to evaluate the performance of the model used
# can be created
class_rep = classification_report(y_val, y_val_predicted_logisticr)
print(class_rep)
# Result: precision    recall  f1-score   support
#            0       0.10      0.05      0.07        74
#            1       0.39      0.17      0.23        72
#            2       0.52      0.68      0.59      1135
#            3       0.75      0.64      0.69      1778

#     accuracy                           0.63      3059
#    macro avg       0.44      0.39      0.40      3059
# weighted avg       0.64      0.63      0.63      3059


# to better visualize the way the model performs, one can use a confusion matrix
confusion_m = confusion_matrix(y_val, y_val_predicted_logisticr)

# the following steps are used to actually display that matrix with custom attributes
plt.figure(figsize=(8, 8))
# annot=True is used in order to display the corresponding number for each square in the matrix
# cmap='Blues' generates blue shades of color for the heatmap (lighter blue for lower values -> darker for bigger values)
# fmt='d' means the number have a decimal form (integers)
# cbar=False removes the color bar near the heatmap; I considered it already intuitive enough
sns.heatmap(confusion_m, annot=True, cmap='Blues', fmt='d', cbar=False)
# assigning labels to the axis and giving a name to the image
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
# displaying the plot 
plt.show()
