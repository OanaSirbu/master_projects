import os # useful to play with paths of files
import matplotlib.pyplot as plt # used for plots
import numpy as np # used to store large arrays
from tensorflow.keras.applications import InceptionV3 # import the InceptionV3 pre-trained model
from tensorflow.keras.preprocessing import image # used for image processing
from tensorflow.keras.applications.inception_v3 import preprocess_input # used to prepare input for InceptionV3
from tensorflow.keras.models import Model # used to define a neural network model
from sklearn.decomposition import PCA # to reduce the features to a lower dimensional representation
from sklearn.mixture import GaussianMixture # to import the GMM model
from collections import Counter # will help with counting the num of points from each cluster
from sklearn.metrics import silhouette_score, calinski_harabasz_score # used to compute these metrics for model evaluation
from sklearn.cluster import DBSCAN # import DBSCAN model
from sklearn.cluster import KMeans # import K-means model
from skimage import feature # helps with feature extraction from images
from skimage import io, color # used to read and change the color space of an image
from sklearn.model_selection import GridSearchCV # used to perform Grid Search
from sklearn.base import clone # used in Grid Search 
import pandas as pd # to visualize results from Grid Search
import seaborn as sns # used to represent heatmaps
from sklearn.ensemble import RandomForestClassifier # import the RFC model
from sklearn.model_selection import train_test_split # used to split the dataset into training and test sets
from sklearn.metrics import accuracy_score, classification_report # compute metrics for the supervised approach
from sklearn.dummy import DummyClassifier # used for comparison with random choice


# set the folder containing all the images (with horses, donkeys and zebras)
animals_directory_path = '/Users/oana/Downloads/pml-exam/master-AI/PML/Project 2/archive/train/animals'


# we make an instance of the InceptionV3 model, choose the last convolutional layer (named mixed10)
inceptionv3_model = InceptionV3(weights='imagenet', include_top=False)
mixed10_layer = inceptionv3_model.get_layer('mixed10')
# the features of interest will be extracted from the mixed10 layer
feature_extraction_model = Model(inputs=inceptionv3_model.input, outputs=mixed10_layer.output)


# this is a function to extract the relevant features from only one image
def extract_inception_features(image_path):
    # we used the (224, 224) target size to match the input requirements for the Inception model
    image_data = image.load_img(image_path, target_size=(224, 224))
    # transform the image into an array
    image_array = image.img_to_array(image_data)
    image_array = np.expand_dims(image_array, axis=0)
    # further preprocess the original image
    image_array = preprocess_input(image_array)
    # finally, extract relevant features from the mixed10 layer 
    features = feature_extraction_model.predict(image_array)
    # return the flattened array of features
    return features.flatten()


# once we have the function that extracts features from one image
# we should extract features from all the training images existing in the animal directory
# therefore we create the following function
def extract_inception_features_from_directory(directory_path):
    # create a list to store all the features
    features_list = []
    # loop through all the files that have specific extensions
    for filename in os.listdir(directory_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.JPG')):
            img_path = os.path.join(directory_path, filename)
            # for each file extract the features from the image
            img_features = extract_inception_features(img_path)
            # add the new features to the final list
            features_list.append(img_features)
    # simply return the features list
    return np.array(features_list)


# store the features list into the inception_features variable
inception_features = extract_inception_features_from_directory(animals_directory_path)


# define a Principal component analysis instance, that captures the most important 50 features
pca = PCA(n_components=50)
# apply the previous PCA instance on the inception_features to reduce their dimensionality
inception_features_pca = pca.fit_transform(inception_features)
# print the result (just a check)
print(inception_features_pca)


# define a GMM model which has 3 components (because we want an exact number of 3 clusters)
gmm_model_inc = GaussianMixture(n_components=3, random_state=1, init_params='kmeans')
# fit the model on the extracted inception features, that were also reduce by a PCA
gmm_model_inc.fit(inception_features_pca)
# predict the cluster associated with those efatures
clusters_for_inception = gmm_model_inc.predict(inception_features_pca)
# here we count the number of instances corresponding to each cluster
cluster_counts_inc = Counter(clusters_for_inception)
print("Cluster points:", cluster_counts_inc)

# make a plot for clusters using only the first principal component (0) and the second one (1) from the PCA
plt.scatter(inception_features_pca[:, 0], inception_features_pca[:, 1], c=clusters_for_inception, cmap='plasma', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
# here we set a title for the plot
plt.title('GMM Clustering - Inception features')
plt.show()

# we compute the evaluation metrics for the model 
# a silhouette score closer to 1 is better
gmm_silhouette_inc = silhouette_score(inception_features_pca, clusters_for_inception)
print(f'Silhouette score: {gmm_silhouette_inc}')
# a higher Calinski Harabasc score is also better
gmm_calinski_harabasz_inc = calinski_harabasz_score(inception_features_pca, clusters_for_inception)
print(f'Calinski-Harabasz score: {gmm_calinski_harabasz_inc}')


# as we analysed the performance of the GMM model, we also define a DBSCAN model and
# apply it on the same Inception features reduced by a PCA
# the hyperparameters were chosen after a careful grid search that can be found at the end of the code
dbscan = DBSCAN(eps=0.4, min_samples=30, metric='cosine')  
# predict the clusters associated with the inception_features_pca
clusters_for_dbscan_inc = dbscan.fit_predict(inception_features_pca)

# just as before, we make a plot to better visualize and understand the clusters 
plt.scatter(inception_features_pca[:, 0], inception_features_pca[:, 1], c=clusters_for_dbscan_inc, cmap='plasma', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('DBSCAN Clustering - Inception features')
plt.colorbar(label='Cluster')
plt.show()

# count the number of points distributed in each cluster and print the result
cluster_counts_dbscan_inc = Counter(clusters_for_dbscan_inc)
print("Cluster points:", cluster_counts_dbscan_inc)
# compute the silhouette score of the DBSCAN on the inception features
dbscan_silhouette_inc = silhouette_score(inception_features_pca, clusters_for_dbscan_inc)
print(f'Silhouette score: {dbscan_silhouette_inc}')
# and also another metric (the higher the better)
dbscan_calinski_harabasz_inc = calinski_harabasz_score(inception_features_pca, clusters_for_dbscan_inc)
print(f'Calinski-Harabasz score: {dbscan_calinski_harabasz_inc}')


# I tested another model on the same features for comparison purposes
# the number of clusters was set to 3 from the very beginning because we knew the dataset very well
kmeans = KMeans(n_clusters=3, random_state=1)
# fit the features and predict the corresponding clusters
clusters_for_kmeans_inc = kmeans.fit_predict(inception_features_pca)
# for a better visualisation, i also made a plot for these, using the same style for consistency
plt.scatter(inception_features_pca[:, 0], inception_features_pca[:, 1], c=clusters_for_kmeans_inc, cmap='plasma', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
# set a title to the plot 
plt.title('K-means Clustering - InceptionV3 features')
plt.show()

# if needed, one can also print the centers of the clusters
print("Cluster Centers:")
print(kmeans.cluster_centers_)
# as previously seen, we can count the num of points in each cluster
cluster_counts_kmeans_inc = Counter(clusters_for_kmeans_inc)
# this helps us understand the performance of the algorithm
print("Cluster points:", cluster_counts_kmeans_inc)

# some metrics were also computed for evaluation
kmeans_silhouette_inc = silhouette_score(inception_features_pca, clusters_for_kmeans_inc)
print(f'Silhouette score: {kmeans_silhouette_inc}')
kmeans_calinski_harabasz_inc = calinski_harabasz_score(inception_features_pca, 
                                                    clusters_for_kmeans_inc)
print(f'Calinski-Harabasz score: {kmeans_calinski_harabasz_inc}')


# now that we trained a GMM, a DBSCAN and a K-means on the Inception features, 
# we try another set of features: Histograms of oriented gradieds (short: hog)

# first, we extract the hog features from an image
# the hyperparameters were chosen empirically to be those that helps us achieve the best performance
def extract_hog_features(image_path, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1),
                         block_norm='L2-Hys', visualize=False):
    # simply read the image
    image_data = io.imread(image_path)
    # change the color space of the image to gray
    gray_image = color.rgb2gray(image_data)
    # compute hog features by taking the parameters of the main function
    hog_features = feature.hog(gray_image, orientations=orientations,
                               pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,
                               block_norm=block_norm, visualize=False)
    return hog_features
    
# create a list to store all the hog features
hog_features = []

# loop through all the files from the animals directory
for filename in os.listdir(animals_directory_path):
    # check if it is an image
    if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.webp', '.JPG')):
        image_path = os.path.join(animals_directory_path, filename)
        # use the function which extracts hog features from an image
        image_features = extract_hog_features(image_path)
        # just a check to make sure the features variables is not null 
        if image_features is not None:
            hog_features.append(image_features)
        else:
            # just print a message to detect a problem if it exists
            print(f"No features extracted for {image_path}. You better inspect it!!")

# for better efficiency, transform the list into a numpy array
hog_features_array = np.array(hog_features)
print(hog_features_array.shape)

# same as before, we create an instance of a GMM model
gmm_hog = GaussianMixture(n_components=3, random_state=1, init_params='kmeans')
# create a PCA that would reduce the dimension of the hog features space
pca = PCA(n_components=50)
hog_features_pca = pca.fit_transform(hog_features_array)
# then we fit the GMM model on the features
gmm_hog.fit(hog_features_pca)
# and predict the new clusters
gmm_clusters_for_hog = gmm_hog.predict(hog_features_pca)
# same procedure as above: count the instances from the clusters
cluster_counts_hog = Counter(gmm_clusters_for_hog)
print("Cluster points:", cluster_counts_hog)

# same graph to have a better understanding of the model's performance (are the clusters distinctive?)
plt.scatter(hog_features_pca[:, 0], hog_features_pca[:, 1], c=gmm_clusters_for_hog, cmap='plasma', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('GMM Clustering - HOG features')
plt.colorbar(label='Cluster')
# show the plot
plt.show()

# here we simply compute the silhouette and the calinski harabasz scores to observe the performance of the model
# and compare with the previous ones, obtained for the InceptionV3 features
gmm_silhouette_hog = silhouette_score(hog_features_pca, gmm_clusters_for_hog)
print(f'Silhouette score: {gmm_silhouette_hog}')
gmm_calinski_harabasz_hog = calinski_harabasz_score(hog_features_pca, gmm_clusters_for_hog)
print(f'Calinski-Harabasz Index: {gmm_calinski_harabasz_hog}')


# in a similar manner, we compute the clusters with a DBSCAN applied on HOG features 
dbscan = DBSCAN(eps=0.4, min_samples=18, metric='cosine')  
dbscan_clusters_for_hog = dbscan.fit_predict(hog_features_pca)
# then we visualize the DBSCAN clusters, using PCA 
plt.scatter(hog_features_pca[:, 0], hog_features_pca[:, 1], c=dbscan_clusters_for_hog, cmap='plasma', alpha=0.5)
# set the corresponding labels for the principal components from PCA
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('DBSCAN Clustering - HOG features')
plt.colorbar(label='Cluster')
plt.show()

# i use the same structure and compute the performance metrics after counting the instances forming each cluster
cluster_counts_dbscan_hog = Counter(dbscan_clusters_for_hog)
print("Cluster points:", cluster_counts_dbscan_hog)
dbscan_silhouette_hog = silhouette_score(hog_features_pca, dbscan_clusters_for_hog)
print(f'Silhouette score: {dbscan_silhouette_hog}')
dbscan_calinski_harabasz_hog = calinski_harabasz_score(hog_features_pca, dbscan_clusters_for_hog)
print(f'Calinski-Harabasz score: {dbscan_calinski_harabasz_hog}')


# and once again I also trained a 3rd model (Kmeans) on the same features to make comparisons 
kmeans = KMeans(n_clusters=3, random_state=1)
clusters_for_kmeans_hog = kmeans.fit_predict(hog_features_pca)
plt.scatter(hog_features_pca[:, 0], hog_features_pca[:, 1], c=clusters_for_kmeans_hog, cmap='plasma', alpha=0.5)
plt.title('K-means Clustering - for HOG features')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

# the corresponding metrics were computed
cluster_counts_kmeans_hog = Counter(clusters_for_kmeans_hog)
print("Cluster points:", cluster_counts_kmeans_hog)
kmeans_silhouette_hog = silhouette_score(hog_features_pca, clusters_for_kmeans_hog)
print(f'Silhouette score: {kmeans_silhouette_hog}')
kmeans_calinski_harabasz_hog = calinski_harabasz_score(hog_features_pca, clusters_for_kmeans_hog)
print(f'Calinski-Harabasz score: {kmeans_calinski_harabasz_hog}')


# extract the images names from folder (used to find the associated labels)
images_names = [file for file in os.listdir(animals_directory_path) if os.path.isfile(os.path.join(animals_directory_path, file))]
print(images_names)
images_labels = [file.split()[0] for file in images_names]
print(images_labels)


#######################################
# The comparison with the random choice
#######################################
print('We start the comparison with a random choice')

# we tested the performance of a Dummy classifier on the same featues (both Inception an HOG ones)
# split the dataset into train and test sets and use inception_features_pca as X_train 
features_train_dummy, features_test_dummy, labels_train_dummy, labels_test_dummy = train_test_split(
    inception_features_pca, images_labels, test_size=0.2, random_state=1
)

# make an instance of a dummy algorithm 
dummy_classifier_comp = DummyClassifier(strategy='stratified')

# first we 'train' the model on the Inception features
dummy_classifier_comp.fit(features_train_dummy, labels_train_dummy)
# and we make predictions on the test set
labels_pred = dummy_classifier_comp.predict(features_test_dummy)
# evaluate the performance using a classification report
classification_report_dummy = classification_report(labels_test_dummy, labels_pred)
print(classification_report_dummy)

# secondly, we 'train' the model on the HOG features 
features_train_dummy_2, features_test_dummy_2, labels_train_dummy_2, labels_test_dummy_2 = train_test_split(
    hog_features_pca, images_labels, test_size=0.2, random_state=1
)
dummy_classifier_comp_2 = DummyClassifier(strategy='stratified')
dummy_classifier_comp_2.fit(features_train_dummy_2, labels_train_dummy_2)
# we also make predictions on the test set of features 
labels_pred_2 = dummy_classifier_comp_2.predict(features_test_dummy_2)
# evaluate the performance using a comprehensive classification report
classification_report_dummy_2 = classification_report(labels_test_dummy_2, labels_pred_2)
print(classification_report_dummy_2)


# another 'dummy' approach was implemented for fun

num_samples = 1023  
# generate a random label for each image (out of 1023 images)
random_choice_labels = np.random.randint(0, 3, num_samples)
# compute the scores reached through the random choice method
random_choice_silhouette_score = silhouette_score(inception_features_pca, random_choice_labels)
random_choice_ch_score = calinski_harabasz_score(inception_features_pca, random_choice_labels)
# visualize and compare the results between the GMM, DBSCAN and random choice 
print(f"Random silhouette score: {random_choice_silhouette_score}")
print(f"Random calinski_harabasz_score: {random_choice_ch_score}")
print(f"GMM silhouette score: {gmm_silhouette_inc}")
print(f"GMM calinski_harabasz_score: {gmm_calinski_harabasz_inc}")
print(f"DBscan Silhouette Score: {dbscan_silhouette_inc}")
print(f"DBScan calinski_harabasz_score: {dbscan_calinski_harabasz_inc}")


#######################################
# comparison with a supervised baseline
#######################################

# split the dataset into train and test sets and use inception_features_pca as X_train 
features_train, features_test, labels_train, labels_test = train_test_split(
    inception_features_pca, images_labels, test_size=0.2, random_state=1
)

# make an instance of a supervised algorithm (I chose the Random Forest Classifier)
randomf_classifier = RandomForestClassifier(n_estimators=100, random_state=1)
# also created another GMM instance so I can directly compare these two
gmm_for_comparison = GaussianMixture(n_components=3, random_state=1)

# first we train the supervised model
randomf_classifier.fit(features_train, labels_train)
# and we make predictions on the test set
labels_pred = randomf_classifier.predict(features_test)
# evaluate the performance using a classification report
accuracy = accuracy_score(labels_test, labels_pred)
classification_report_comp = classification_report(labels_test, labels_pred)
print(classification_report_comp)

# then we train a GMM model on the same train features
gmm_for_comparison.fit(features_train)
# create clusters for the test features
gmm_clusters_for_comparison = gmm_for_comparison.predict(features_test)
# count the instances and evaluate the model with proper metrics
cluster_counts_for_comparison = Counter(gmm_clusters_for_comparison)
print("Cluster points:", cluster_counts_for_comparison)
gmm_silhouette_for_comparison = silhouette_score(features_test, gmm_clusters_for_comparison)
print(f'Silhouette score: {gmm_silhouette_for_comparison}')

# visualize the clusters obtained by GMM on test Inception features
plt.scatter(features_test[:, 0], features_test[:, 1], c=gmm_clusters_for_comparison, cmap='plasma', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.title('GMM Clustering - for test features')
plt.show()


# make the same comparison, with the same models and the same logic but for HOG features
features_train_hog, features_test_hog, labels_train_hog, labels_test_hog = train_test_split(
    hog_features_pca, images_labels, test_size=0.2, random_state=1
)

randomf_classifier_2 = RandomForestClassifier(n_estimators=100, random_state=1)
gmm_for_comparison_2 = GaussianMixture(n_components=3, random_state=1)

# here we train another instance of the RFC model on the train hog features
randomf_classifier_2.fit(features_train_hog, labels_train_hog)
# also make predictions on the test set
labels_pred_hog = randomf_classifier_2.predict(features_test_hog)

# evaluate the performance of the supervised method
accuracy_2 = accuracy_score(labels_test, labels_pred_hog)
classification_report_2 = classification_report(labels_test, labels_pred_hog)
print(f"Accuracy: {accuracy_2}")
print("Classification Report:\n", classification_report_2)

# train the GMM model, evaluate its performance with proper metrics
gmm_for_comparison_2.fit(features_train)
gmm_clusters_for_comparison_2 = gmm_for_comparison_2.predict(features_test)
cluster_counts_for_comparison_2 = Counter(gmm_clusters_for_comparison_2)
print("Cluster points:", cluster_counts_for_comparison_2)
gmm_silhouette_for_comparison_2 = silhouette_score(features_test, gmm_clusters_for_comparison_2)
print(f'Silhouette score: {gmm_silhouette_for_comparison_2}')
# visualize clusters for a better understanding of the model's performance
plt.scatter(features_test[:, 0], features_test[:, 1], c=gmm_clusters_for_comparison_2, cmap='plasma', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('GMM Clustering - for HOG test features')
plt.colorbar(label='Cluster')
plt.show()


# last but not least, we perform a Grid Search in order to find out the best parameters 
# of a DBSCAN trained on our dataset
param_grid_dbscan = {
    # we play with many different values for the most important hyperparameters
    'eps': [0.2, 0.3, 0.4, 0.5, 0.6],          
    'min_samples': [5, 10, 15, 20, 25, 30, 35],       
    'metric': ['euclidean', 'cosine'] 
}

# make an instance of the DBSCAN model
dbscan_for_grid = DBSCAN()

# a custom function to pass to the Grid Search 'scoring' parameter
# (the silhouette score wasn't an option in the list of default scores)
def silhouette_scorer(model_to_test, some_features):
    # we will chose the model that we want to evaluate on our features
    labels = model_to_test.fit_predict(some_features)
    if len(set(labels)) < 2:
        return 0  # we can't define a silhouette score if we have a single cluster
    else:
        # if we have more clusters then we compute the corresponding silhouette score
        return silhouette_score(some_features, labels)

# the basic syntax for performing a grid search with the previously defined grid parameters
# and the custom scoring function, with 3 folds in cross validation
grid_search_dbscan = GridSearchCV(clone(dbscan_for_grid), param_grid_dbscan, scoring=silhouette_scorer, cv=3)
# we train the model on the Inception features (but the same results were obtained for the HOG features)
grid_search_dbscan.fit(inception_features_pca)
# a variable in which we store the best parameters found by the grid search
best_params_dbscan = grid_search_dbscan.best_params_
results_dbscan = grid_search_dbscan.cv_results_
# the simply print them
print(best_params_dbscan)

# we store the results in a pandas dataframe because it will help us create a heatmap
crossv_results_df = pd.DataFrame(results_dbscan)
# find the mean scores for each combination of eps and min_samples using pandas methods
mean_scores = crossv_results_df.groupby(['param_eps', 'param_min_samples'])['mean_test_score'].mean().reset_index()
# create the heatmap for visualisation (there were too many relevant pairs of eps and min_values to make a simple graph)
heatmap_data = mean_scores.pivot(index='param_eps', columns='param_min_samples', values='mean_test_score')
# this is customizable
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap_data, annot=True, cmap='plasma', cbar_kws={'label': 'Mean Silhouette Score'})
plt.title('Mean silhouette score for different eps and min_samples pairs')
# on the OX axis we chose to put the values for the min_samples parameter
plt.xlabel('min_samples')
# on the OY we chose the eps param, that also highly influences the scores
plt.ylabel('eps')
plt.show()


# we also perform a Grid Search for the GMM models
# we start by defining a GMM instance
gmm_for_grid = GaussianMixture()
# perform the grid search only to find the best covariance_type
# (also tried for different params but they didn't influence the results)
param_grid_gmm = {
    'covariance_type': ['full', 'tied', 'diag', 'spherical'],
}

# used the previously defined silhouette_scorer function as a criterion for choosing the best model
grid_search_gmm = GridSearchCV(clone(gmm_for_grid), param_grid_gmm, scoring=silhouette_scorer, cv=3)
# tested on the Incpetion features (but same params were kept for the HOG ones in this case, too)
grid_search_gmm.fit(inception_features_pca)  
# extract the best parameter and print it
best_params_gmm = grid_search_gmm.best_params_
print("Best Parameters:", best_params_gmm)

# extract the mean score from each fold from the cross validation (for a simpler representation on a plot)
results_gmm = grid_search_gmm.cv_results_
covariance_types = results_gmm['param_covariance_type']
mean_silhouette_scores = results_gmm['mean_test_score']
# create a mapping for those attributes so that we can represent them on a graph
covariance_mapping = {'full': 0, 'tied': 1, 'diag': 2, 'spherical': 3}
digits_covariance = [covariance_mapping[cov_type] for cov_type in covariance_types]
# this is also customizable
plt.figure(figsize=(10, 6))
# create a graph for each mean silhouette score obtained as a function of the covariance type
plt.scatter(digits_covariance, mean_silhouette_scores, s=50, c='blue', alpha=0.7)
plt.plot(digits_covariance, mean_silhouette_scores, linestyle='-', marker='o', color='blue')
plt.xticks(list(covariance_mapping.values()), list(covariance_mapping.keys()))
# for a better understanding, we set the variables names on the axis 
plt.xlabel('Covariance Type')
plt.ylabel('Mean Silhouette Score')
# and a comprehensive title was chosen
plt.title('Silhouette Score vs. Covariance Type for GMM')
plt.show()