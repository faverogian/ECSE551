import numpy as np

class LogisticRegression:
    def __init__(self, lr=0.01, reg=None):
        self.lr = lr

    def intercept(self,X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    def sigmoid(self,a):
        return 1 / (1 + np.exp(-a))

    def cross_entropy(self,s, y):
        return -(y*np.log(s)+(1-y)*np.log(1-s))

    def fit(self,X, Y):
        X = self.intercept(X)

        # weights initialization
        w = np.zeros(X.shape[1])

        for i in range(1000):
            a = np.dot(X, w)

            s = self.sigmoid(a)
            g = -np.dot(X.T, (Y-s))

            #Update weights
            w -= self.lr*g

            print('gradient: ',g)
            print('Weights: ', w)
        return w

    def predict_prob(self,X,w):
        X = self.intercept(X)
        pred = self.sigmoid(np.dot(X, w)) >= 0.5
        return pred
  
