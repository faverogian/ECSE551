import numpy as np
from NestedCrossValidation import Accu_eval
    
class LogisticRegression:
    def __init__(self, lr, reg, norm_penalty, early_stopping=True):
        # Initialize parameters
        self.lr = lr
        self.max_iter = 2000
        self.epoch_size = 50
        self.stopping_threshold = 0.01
        self.patience = 5
        self.early_stopping = early_stopping
        self.w = None

        # Set the gradient function based on the regularization
        if reg == 'l1':
            self.grad = lambda X_train, Y_train, s: -np.dot(X_train.T, (Y_train - s)) + norm_penalty*np.sign(self.w)
        elif reg == 'l2':
            self.grad = lambda X_train, Y_train, s: -np.dot(X_train.T, (Y_train - s)) + 2*self.w*norm_penalty

    def intercept(self, X):
        # Creates the "dummy" variable for the bias term
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, a):
        # Shortcut sigmoid function
        return 1 / (1 + np.exp(-a))

    def fit(self, X_train, Y_train, X_val, Y_val):
        # Add intercept to training data
        x_train = self.intercept(X_train)

        # weights initialization to random values with a random seed
        np.random.seed(15)
        self.w = np.random.randn(x_train.shape[1])

        # Check for early stopping if accuracy on validation set stops improving
        accuracy = 0
        patience = self.patience
        iterations = []
        val_accuracies = []
        train_accuracies = []

        for i in range(self.max_iter):
            # Update weights using gradient descent
            a = np.dot(x_train, self.w)
            s = self.sigmoid(a)
            grad = self.grad(x_train, Y_train, s)

            self.w -= self.lr*grad

            # At regular intervals check accuracy
            if i % self.epoch_size == 0:
                # Validation accuracy
                Y_pred_val = self.predict(X_val)
                new_val_accuracy = Accu_eval(Y_pred_val, Y_val)
                val_accuracies.append(new_val_accuracy)

                # Training accuracy
                Y_pred_train = self.predict(X_train)
                train_accuracies.append(Accu_eval(Y_pred_train, Y_train))

                iterations.append(i)
                
                # Check early stopping criteria
                if self.early_stopping:
                    if new_val_accuracy - accuracy < self.stopping_threshold:
                        patience -= 1
                    else:
                        accuracy = new_val_accuracy
                        patience = self.patience

                    if patience == 0:
                        return iterations, val_accuracies, train_accuracies
        
        # Return accuracy data for observation
        return iterations, val_accuracies, train_accuracies

    def predict(self, X_test):
        # Function used to generate predictions from model
        X_test = self.intercept(X_test)
        pred = self.sigmoid(np.dot(X_test, self.w)) >= 0.5

        return 1*pred