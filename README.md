# An Ensemble Method for Classifying Reddit Posts
#### ECSE 551: Introduction to Machine Learning, Department of ECE, McGill University  

Authors: Gian Favero, Maxime Favreau-Vachon, Hieu Thien Hong, 2023

## Results
The methods outlined achieved 2nd place in the course Kaggle competition.

## Paper Abstract
We present a comprehensive exploration of text classification in the context of Reddit posts 
originating from four subreddit groups: Toronto, Paris, Montreal, and London. The class of posts 
from these groups is attempted to be learned using various classification models, including a 
Naive Bayes model constructed from scratch, and models made available via scikit-learn such as Support 
Vector Machines (SVM), Logistic Regression, and Random Forest. These models are coupled with a 
meticulous preprocessing pipeline involving, stop-word removal, lemmatization, and information gain. 
A detailed analysis of model performance is conducted through a 5-fold cross-validation process, revealing 
notable improvements with preprocessing steps such as common word removal, mutual information filtering,
and word replacement. Each model demonstrates similar challenges in accurately classifying posts from the
classes with similarities in linguistic and geopolitical context (e.g., Montreal versus Toronto or Paris). 
This challenge was best overcome with a model stacking approach that combined SVM, Logistic Regression, and 
Random Forest classifiers with a majority vote. This ensemble approach, combined with strict preprocessing, 
yielded a substantial increase in accuracy. The final model stack achieves an accuracy of 72% on the Kaggle 
evaluation set.

The full paper can be seen [here](An-Ensemble-Approach-for-Reddit-Post-Classification.pdf).
