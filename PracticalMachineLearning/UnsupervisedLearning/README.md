# Project 2: Unsupervised Learning with Image Data

## Dataset
This project utilizes images sourced from Kaggle, organized into three folders representing horses, donkeys, and zebras. The dataset can be accessed [here](https://www.kaggle.com/datasets/ifeanyinneji/donkeys-horses-zebra-images-dataset).

## Feature Extraction Methods
Feature vectors were extracted from images using the Inception pre-trained model and Histogram of Oriented Gradients (HOG) features.

## Unsupervised Learning Methods
The project used Gaussian Mixture Model (GMM), K-means, and Density-Based Spatial Clustering of Applications with Noise (DBSCAN) to cluster the feature vectors.

## Results
All unsupervised models successfully represented the input images as three different clusters. The models identified two clusters with similar features, likely corresponding to horses and donkeys, while the third class is distinguished by very different principal components after PCA dimensionality reduction. The unsupervised algorithms performed better than the random choice method.

