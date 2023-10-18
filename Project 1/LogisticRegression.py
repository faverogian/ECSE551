import numpy as np
    
class LogisticRegression:
    def __init__(self, lr, reg, norm_penalty):
        # Initialize the weights and learning rate
        self.w = None
        self.lr = lr

        # Set the gradient function based on the regularization
        if reg == 'l1':
            self.grad = lambda X_train, Y_train, s: -np.dot(X_train.T, (Y_train - s)) + norm_penalty*np.sign(self.w)
        elif reg == 'l2':
            self.grad = lambda X_train, Y_train, s: -np.dot(X_train.T, (Y_train - s)) + 2*self.w*norm_penalty

    def intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def fit(self, X_train, Y_train, X_val, Y_val):
        # Add intercept to training data
        X_train = self.intercept(X_train)

        # weights initialization
        self.w = np.zeros(X_train.shape[1])

        # Check for early stopping if accuracy on validation set stops improving
        accuracy = 0
        patience = 10
        threshold = 0.001
        epoch_size = 100

        for i in range(5000):
            # Update weights using gradient descent
            a = np.dot(X_train, self.w)
            s = self.sigmoid(a)
            grad = self.grad(X_train, Y_train, s)

            self.w -= self.lr*grad

            # Check for early stopping every 100 iterations
            if i % epoch_size == 0:
                Y_pred = self.predict(X_val)
                new_accuracy = np.mean(Y_pred == Y_val)

                if new_accuracy - accuracy < threshold:
                    patience -= 1
                else:
                    accuracy = new_accuracy
                    patience = 10

                if patience == 0:
                    break

    def predict(self, X_test):
        X_test = self.intercept(X_test)
        pred = self.sigmoid(np.dot(X_test, self.w)) >= 0.5

        return 1*pred
  
