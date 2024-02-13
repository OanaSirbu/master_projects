# Recommendation System using Hierarchical Poisson Factorization

This repository contains an implementation of a Recommendation System based on Hierarchical Poisson Factorization (HPF). The system is designed to offer personalized recommendations by using the hierarchical structure inherent in user-item interaction data.

## Dataset
For this project, we utilize the Book Recommendation Dataset, which can be accessed [here](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset). 

## Algorithm
The structure of the recommendation algorithm is inspired by the paper titled ['Scalable Recommendation with Poisson Factorization'](https://arxiv.org/abs/1311.1704). It uses statistical distributions like the Poisson distribution and the Gamma distribution to model data characteristics and relationships.

## Analysis
In the accompanying Jupyter notebook, we also provide a detailed analysis of the dataset. This includes a sanity check to ensure data integrity and thorough hyperparameter tuning to optimize the recommendation system's performance.

## Prerequisites
To run the code in this notebook, you'll need to have the following modules installed:

- [pymc3](https://docs.pymc.io/)
- [numpy](https://numpy.org/doc/)
- [pandas](https://pandas.pydata.org/docs/)
- [seaborn](https://seaborn.pydata.org/)
- [matplotlib](https://matplotlib.org/3.3.3/contents.html)
