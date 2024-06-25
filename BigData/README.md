This project aims to predict readmission rates of diabetic patients, from the following [dataset](https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008).

Over this data set, we used 4 preprocessing methods, each of them being preceded by 5 supervised models. The performance was measured using accuracy and macro F1 score as metrics of evaluation.

The folder `performance_measurements/` contains the actual code of this project. This folder, in turn, is composed of 1 file (`EDA.ipynb`, in which we take a first glimpse of the data) and 4 subfolders representing the preprocessing methods used:

- Simple encoding - just encoding the object-typed features.

- PCA - apply Principal Component Analysis over the encoded features.

- Random Projection - apply a dimensionality reduction technique that projects high-dimensional data onto a lower-dimensional subspace using randomly generated matrices.

- UMAP (Uniform Manifold Approximation and Projection) - nonlinear technique for visualizing high-dimensional data in lower dimensions, capturing both local and global structures effectively.

Each of these subfolders contain 6 files. The first file contains the data preprocessing and is run at the beginning of all the other four files. The other 5 files contain one supervised model each, and evaluate it according to the true labels. The supervised models that we used are the following:

- Random Forest Classifier.

- XGBClassifier.

- K Nearest Neighbors.

- Naive Bayes.

- CatBoost.

Finally, the results and the project report can be found in the folder `conclusions/`.
