import numpy as np
from SplitSet import KFoldSplitSet, SplitSet

class NestedCV:
    def __init__(self, outer_folds, inner_folds, model, hyperparameter_grid):
        self.outer_folds = outer_folds
        self.inner_folds = inner_folds
        self.model = model
        self.hyperparameters = hyperparameter_grid

    def accuracy_score(self, Y_pred, Y_test):
        accuracy = np.mean(Y_pred == Y_test)
        return accuracy
    
    def get_candidate_models(self, X, Y):
        # Create a fold generator
        outer_K_Folds = KFoldSplitSet(folds=self.outer_folds)
        inner_K_Folds = KFoldSplitSet(folds=self.inner_folds)

        # Keep track of the 10 best model hyperparameters
        model_hyperparameters = []
        
        # For each of the 10 folds, get the train and test sets
        for X_train, _, Y_train, _ in outer_K_Folds.split(X, Y):

            # Keep track of the best accuracy and hyperparameter combination of the inner folds
            best_accuracy = 0
            best_hyperparameters = None

            # Split the training set into inner folds for generating the 10 best models
            for x_train, x_val, y_train, y_val in inner_K_Folds.split(X_train, Y_train):
                # Create a list to store the accuracies for each hyperparameter combination
                inner_accuracies = []
                hyperparameter_combinations = []

                # For each hyperparameter combination, train the model and get the accuracy
                for perm in self.hyperparameters:
                    # Add the hyperparameter combination to the list
                    hyperparameter_combinations.append(perm)

                    # Train the model
                    model = self.model(*perm)
                    model.fit(x_train, y_train, x_val, y_val)

                    # Get the predictions
                    Y_pred = model.predict(x_val)

                    # Get the accuracy on the validation set
                    accuracy = self.accuracy_score(Y_pred, y_val)

                    # Add the accuracy to the list
                    inner_accuracies.append(accuracy)
                
                # Get the best hyperparameter combination on the inner folds
                average_inner_accuracy = np.mean(inner_accuracies)

                if average_inner_accuracy > best_accuracy:
                    best_accuracy = average_inner_accuracy
                    best_hyperparameters = hyperparameter_combinations[np.argmax(inner_accuracies)]

            # Append the model hyperparameters to the list
            model_hyperparameters.append(best_hyperparameters)
        
        return list(set(model_hyperparameters))

    def k_fold_cross_validation(self, hyperparameters, X, Y):
        # Keep track of the accuracies for each model
        accuracies = []

        for hyperparams in hyperparameters:
            # Create a fold generator
            outer_K_Folds = KFoldSplitSet(folds=self.outer_folds)

            # Keep track of the accuracies for each model
            accuracies = []

            # For each of the 10 folds, get the train and test sets
            for X_train, X_test, Y_train, Y_test in outer_K_Folds.split(X, Y):
                # Split the training set into train and validation sets
                splits = SplitSet()
                x_train, x_val, y_train, y_val = splits.split(X_train, Y_train, test_size=0.2)

                # Train the model
                model = self.model(*hyperparams)
                model.fit(x_train, y_train, x_val, y_val)

                # Get the predictions
                Y_pred = model.predict(X_test)

                # Get the accuracy on the test set
                accuracy = self.accuracy_score(Y_pred, Y_test)

                # Add the accuracy to the list
                accuracies.append(accuracy)

            # Print the average accuracy for the model
            print('Average accuracy for model with hyperparameters {}: {}'.format(hyperparams, np.mean(accuracies)))
