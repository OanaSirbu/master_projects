# import the necessary libraries
import os
import pandas as pd
import numpy as np
import umap.umap_ as umap
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, make_scorer, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


# define the constants of the file.
f1_scores_path = 'conclusions/f1_scores_macro.csv'  # the file in which we draw the F1 scores
accuracies_path = 'conclusions/accuracies.csv'  # the file in which we draw the accuracies
raw_data = pd.read_csv('data/diabetic_data.csv')  # the file from which we extract the data

def write_conclusions(model, f1_score, accuracy):
    """
    Draws the results into the files in which we store the conclusions of this project.
    The results are displayed at:
    ROW = `the preprocessing method`
    COLUMN = `the supervised model used for predictions`
    """

    # we store the name of the current folder. It symbolizes the preprocessing method we're using
    folder_name = os.getcwd().split('/')[-1]  
    model_name = type(model).__name__  # the name of the model
    f1_df = pd.read_csv(f1_scores_path)  # load the macro F1 scores data into a pd.DataFrame
    acc_df = pd.read_csv(accuracies_path)  # load the accuracies data into a pd.DataFrame

    # extract the ROW index for the macro F1 score and accuracy metrics, according to the preprocessing method we're using
    f1_row_index = f1_df.loc[f1_df['PreparationMethod'] == folder_name].index[0]
    acc_row_index = acc_df.loc[acc_df['PreparationMethod'] == folder_name].index[0]

    # we draw the results at coordinates ROW and COLUMN
    f1_df.loc[f1_row_index, model_name] = f1_score
    acc_df.loc[acc_row_index, model_name] = accuracy
    
    # dump the updated data into the corresponding csv file
    f1_df.to_csv(f1_scores_path, index=False) 
    acc_df.to_csv(accuracies_path, index=False)


def impute_question_marks(df, columns):
    for column in columns:
        mode_value = df[column].mode()[0]  # Calculate the mode of the column
        df[column] = df[column].replace('?', mode_value)  # Replace '?' values with the mode


def change_cat(text):
    if text == '>30' or text == '<30':
        return 'Yes'
    else:
        return 'No'


def preprocess_data(df):
    raw_data['readmitted_2'] = raw_data['readmitted'].apply(change_cat) 
    df['race'] = df['race'].apply(lambda x: 'Other' if x == '?' else x)
    # Drop unnecessary columns
    df = df.drop(columns=['encounter_id', 'patient_nbr', 'weight', 'payer_code', 'medical_specialty', 'readmitted'])
    # this is the part where we drop the medicine-related columns, as we've seen that they do not offer any relevant information
    df.drop(columns = ['acetohexamide', 'tolbutamide', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
                   'glipizide-metformin', 'glimepiride-pioglitazone', 'metformin-rosiglitazone',
                   'metformin-pioglitazone'], inplace = True)

    df['admission_type_id']=df['admission_type_id'].apply(lambda x : 5 if x in (6,8) else x)
    df['admission_type_id']=df['admission_type_id'].apply(lambda x : 1 if x == 4 else 2 if x == 7 else x)

    df['admission_source_id']= df['admission_source_id'].apply(lambda x : 11 if x in (12, 13, 14) else x)

    impute_question_marks(df, ['diag_1', 'diag_2', 'diag_3'])

    df['change'] = df['change'].apply(lambda x: 'Yes' if x == 'Ch' else x)

    return df


def encode_object_features(df):
    """
    Encode all the object-typed columns from the dataset.
    Return the updated pd.DataFrame.
    """
    encoded_df = df.copy()
    label_encoder = LabelEncoder()
    categorical_features = ['race', 'gender', 'age',
       'admission_type_id', 'discharge_disposition_id', 'admission_source_id', 'diag_1', 'diag_2', 'diag_3', 'number_diagnoses',
       'max_glu_serum', 'A1Cresult', 'metformin', 'repaglinide', 'nateglinide',
       'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'insulin',
       'glyburide-metformin', 'change', 'diabetesMed', 'readmitted_2'] 
    for cat in categorical_features:
        encoded_df[cat] = label_encoder.fit_transform(df[cat])

    return encoded_df


def apply_umap_projector(encoded_df, n_components=None, threshold=None, scale=True):
    """
    UMAP (Uniform Manifold Approximation and Projection)
    Details:
    1. UMAP is a nonlinear dimensionality reduction technique that is effective for visualizing high-dimensional data in lower dimensions.
    2. UMAP operates by constructing a high-dimensional graph representation of the data, and then optimizing a low-dimensional graph to be as structurally similar as possible.
    3. UMAP can capture both local and global structures of the data, making it suitable for complex datasets, such as medical records.
    """

    if n_components is not None and threshold is not None:
        raise ValueError("Only one of 'n_components' or 'threshold' should be provided.")

    umap_projector = umap.UMAP(n_components=n_components)

    if scale:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('umap', umap_projector)
        ])
    else:
        pipeline = Pipeline([
            ('umap', umap_projector)
        ])

    reduced_df = pipeline.fit_transform(encoded_df)

    return pd.DataFrame(reduced_df)


backup_data = raw_data

preprocessed_data = preprocess_data(raw_data)

# encode the data set
encoded_data = encode_object_features(preprocessed_data)

correlation_matrix = encoded_data[['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications', 
   'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']].corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Extract features and labels and then split them into train and test data
raw_X, y = encoded_data.drop(columns=['readmitted_2']), encoded_data['readmitted_2']

X = apply_umap_projector(raw_X, 3)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Number of features: {X.shape[1]}")

# Select the scoring metric as being the macro F1 score
scoring_metric = make_scorer(f1_score, average='macro')
