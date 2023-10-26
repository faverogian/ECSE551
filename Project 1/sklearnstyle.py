from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from NestedCrossValidation import NestedCV
import pandas as pd

# Load the data
hepatitis_set = pd.read_csv('Hepatits.csv')
hepatitis_headers = ['Ascites', 'Varices', 'Bilirubin', 'Alk Phosphate', 'Sgot', 'Albumin', 'Protime', 'Histology', 'Class']
hepatitis_set.columns = hepatitis_headers

# Get the labels from last column
X = hepatitis_set.iloc[:, :-1]
Y = hepatitis_set.iloc[:, -1]

# Remove class feature from headers
hepatitis_headers = hepatitis_headers[:-1]

# Create no regularization model
model = LogisticRegression(penalty='l2', solver='liblinear')

# Find accuracy of no reg model
nested_cv = NestedCV(outer_folds=10)

# Evaluate the candidate models using a 10-fold cross validation
candidate_scores = nested_cv.k_fold_cross_validation(model, X, Y)

# Print the average accuracy for the model
print("No regularization accuracy: ", candidate_scores)




