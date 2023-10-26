import numpy as np
from SplitSet import KFoldSplitSet, SplitSet

def Accu_eval(Y_pred, Y_test):
    return np.mean(Y_pred == Y_test)

class NestedCV:
    def __init__(self, outer_folds):
        self.outer_folds = outer_folds
    
    def k_fold_cross_validation(self, model, X, Y):
        # Create a fold generator
        outer_K_Folds = KFoldSplitSet(folds=self.outer_folds)

        # Keep track of the accuracies for each model
        accuracies = []

        # For each of the 10 folds, get the train and test sets
        for X_train, X_test, Y_train, Y_test in outer_K_Folds.split(X, Y):
            # Split the training set into train and validation sets
            splits = SplitSet()
            x_train, x_val, y_train, y_val = splits.split(X_train, Y_train, test_size=0.2)

            model.fit(x_train, y_train, x_val, y_val)

            # Get the predictions
            Y_pred = model.predict(X_test)

            # Get the accuracy on the test set
            accuracy = Accu_eval(Y_pred, Y_test)

            # Add the accuracy to the list
            accuracies.append(accuracy)

        # Print the average accuracy for the model
        avg_accuracy = np.mean(accuracies)
        return avg_accuracy
    
    def k_fold_cross_validation2(self, model, X, Y):
        # Create a fold generator
        outer_K_Folds = KFoldSplitSet(folds=self.outer_folds)

        # Keep track of the accuracies for each model
        accuracies = []

        # For each of the 10 folds, get the train and test sets
        for X_train, X_test, Y_train, Y_test in outer_K_Folds.split(X, Y):
            # Split the training set into train and validation sets
            model.fit(X_train, Y_train)

            # Get the predictions
            Y_pred = model.predict(X_test)

            # Get the accuracy on the test set
            accuracy = self.accuracy_score(Y_pred, Y_test)

            # Add the accuracy to the list
            accuracies.append(accuracy)

        # Print the average accuracy for the model
        avg_accuracy = np.mean(accuracies)
        return avg_accuracy
