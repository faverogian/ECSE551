from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from NestedCrossValidation import NestedCV
import pandas as pd

# Load the data
mushroom_set = pd.read_csv('Mushroom.csv')
mushroom_headers = ['Poisonous', 'Cap-shape', 'Cap-surface', 'Cap-color', 'Bruises', 'Odor', 'Gill-attachment',  'Gill-spacing', 'Gill-size', 'Gill-color', 'Stalk-color-below-ring', 'Class']
mushroom_set.columns = mushroom_headers

# Get the labels from last column
X = mushroom_set.iloc[:, :-1]
Y = mushroom_set.iloc[:, -1]

# Do a PCA on the X values
from sklearn.decomposition import PCA
pca = PCA(n_components=5)

# Fit the PCA to the data
pca.fit(X)

# Get the transformed data as a pandas dataframe
X = pd.DataFrame(pca.transform(X))

# Create L1 model
l1_model = LogisticRegression(penalty='l1', solver='liblinear')
l1_accuracy = NestedCV(outer_folds=10).k_fold_cross_validation2(l1_model, X, Y)

# Create L2 model
l2_model = LogisticRegression(penalty='l2', solver='liblinear')
l2_accuracy = NestedCV(outer_folds=10).k_fold_cross_validation2(l2_model, X, Y)

# Create no regularization model
no_reg_model = LogisticRegression(penalty='none', solver='lbfgs')
no_reg_accuracy = NestedCV(outer_folds=10).k_fold_cross_validation2(no_reg_model, X, Y)

# Print the accuracies
print("L1 Accuracy: ", l1_accuracy)
print("L2 Accuracy: ", l2_accuracy)
print("No Regularization Accuracy: ", no_reg_accuracy)




