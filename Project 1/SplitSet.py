import numpy as np

class KFoldSplitSet:
    def __init__(self, folds):
        self.folds = folds

    def split(self, X, Y):
        # Shuffle the data rows
        indices = np.random.permutation(len(X))

        # Set the fold size
        fold_size = len(X) // self.folds

        # For each fold, get the train and test sets
        for i in range(self.folds):
            start, end = i * fold_size, (i + 1) * fold_size
            test_indices = indices[start:end]
            train_indices = np.concatenate((indices[:start], indices[end:]))
            
            # Get the labels
            Y_train = Y.iloc[train_indices]
            Y_test = Y.iloc[test_indices]

            # Get the features
            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]

            yield X_train, X_test, Y_train, Y_test

class SplitSet:
    def __init__(self):
        pass

    def split(self, X, Y, test_size):
        # Shuffle the data rows
        indices = np.random.permutation(len(X))

        stop_idx = round(test_size*X.shape[0])
        test_indices = indices[0:stop_idx]
        train_indices = indices[stop_idx:]

        # Get the labels
        Y_train = Y.iloc[train_indices]
        Y_test = Y.iloc[test_indices]

        # Get the features
        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        return X_train, X_test, Y_train, Y_test

