from LogisticRegression import LogisticRegression
from NestedCrossValidation import NestedCV
import pandas as pd
import itertools

''' Data Preprocessing '''
# Load the data
mushroom_set = pd.read_csv('Mushroom.csv')
mushroom_headers = ['Poisonous', 'Cap-shape', 'Cap-surface', 'Cap-color', 'Bruises', 'Odor', 'Gill-attachment',  'Gill-spacing', 'Gill-size', 'Gill-color', 'Stalk-color-below-ring', 'Class']
mushroom_set.columns = mushroom_headers

# Get the labels from last column
X = mushroom_set.iloc[:, :-1]
Y = mushroom_set.iloc[:, -1]

''' Training the model '''
hyperparameters = {
    'lr': [0.001, 0.03, 0.1],
    'reg': ['l1', 'l2'],
    'norm_penalty': [0, 0.3, 0.5, 0.8]
}
hyper_perms = list(itertools.product(*hyperparameters.values()))
hyper_perms = [perm for perm in hyper_perms if perm[1] != 'l2' and perm[2] != 0] # Remove l2 with 0 norm penalty since it's the same as l1 with 0 norm penalty

# Perform nested cross validation to get candidate models
nested_cv = NestedCV(outer_folds=10, inner_folds=3, model=LogisticRegression, hyperparameter_grid=hyper_perms)
candidate_models = nested_cv.get_candidate_models(X, Y)

# Evaluate the candidate models using a 10-fold cross validation
candidate_scores = nested_cv.k_fold_cross_validation(candidate_models, X, Y)
