from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load the data
mushroom_set = pd.read_csv('Mushroom.csv')
mushroom_headers = ['Poisonous', 'Cap-shape', 'Cap-surface', 'Cap-color', 'Bruises', 'Odor', 'Gill-attachment',  'Gill-spacing', 'Gill-size', 'Gill-color', 'Stalk-color-below-ring', 'Class']
mushroom_set.columns = mushroom_headers

# Get the labels from last column
X = mushroom_set.iloc[:, :-1]
Y = mushroom_set.iloc[:, -1]

# Define hyperparameters for tuning
hyperparameters = {
    'C': [0.001, 0.01, 0.1, 0.3, 0.5, 0.8],
    'penalty': ['l1', 'l2']
}

# Create a logistic regression model
model = LogisticRegression(max_iter=2000, solver='liblinear')

# Create a GridSearchCV object with 10-fold cross-validation
grid_search = GridSearchCV(model, hyperparameters, scoring='accuracy', cv=10)

# Fit the GridSearchCV object to your data
grid_search.fit(X, Y)

# Print the best hyperparameters and the corresponding accuracy
best_hyperparameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(f"Best Hyperparameters: {best_hyperparameters}")
print(f"Best Accuracy: {best_accuracy}")