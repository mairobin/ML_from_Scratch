import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import datasets

class LinearRegression:

    def __init__(self, learning_rate=0.001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None #Array containing the weights for every feature

        # For Bias add Feature X0 with only value=1 [1,1,...,1]
        # DONT DO: self.bias = 0 # Bias is a single number for all features. Start with Bias == 0

    def fit(self, X, Y):
        X = np.vstack((np.ones((X.shape[0],)), X.T)).T
        self.weights = np.zeros(X.shape[1])
        no_datapoints = Y.size
        for i in range(self.n_iterations):

            # Matrix Vector Multiplication. Result is Column Vector with predictions
            y_pred = np.dot(X, self.weights)


            # MSE vector -> contains all mse for every training example
            mse = list()
            y_pred_list = y_pred.flatten()
            Y_list = Y.flatten()
            for p, a in zip(y_pred.flatten(), Y.flatten()):
                calc = (p-a)**2
                mse.append(calc)
            # Accumulated MSE
            mse = sum(mse)


            # Calculate Residuals Vector
            y_diff = y_pred - Y

            gradient = self.compute_gradient(X, y_diff) # Vector
            adjustment = self.learning_rate * gradient
            self.weights = self.weights - self.learning_rate * gradient
            print()



    def predict(self, X):
        X = np.vstack((np.ones((X.shape[0],)), X.T)).T
        y_predictions = np.dot(X, self.weights)
        return y_predictions

    def compute_gradient(self, X, y_diff):
        """Compute the gradient of the cost function J(theta) for linear regression."""
        m = len(y_diff)
        sum_weighted_errors = np.zeros_like(self.weights)

        for i in range(len(y_diff)):
            error = y_diff[i] # scalar
            xi = X[i]  # training sample row
            weighted_error = error * X[i] # Scalar Vector Multiplication of error with data row returns vector
            sum_weighted_errors += error * X[i] # Sum of weighted errors for all training rows returns vector
            print()
        gradient = sum_weighted_errors / len(y_diff)  # Average over all training examples returns vector
        return gradient


# Test Data set contains only one Feature

X, y = datasets.make_regression(n_samples=10, n_features=2, noise=20, random_state=8)


# Make y a column vector
#y = y.reshape(y.shape[0], 1)

# Round the features and target values to two decimal places
X = np.round(X, 2)
y = np.round(y, 2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=134)

reg = LinearRegression(learning_rate=0.1)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
print("PREDICTIONS")
print(predictions)

print("ACTUAL")
print(y_test)

print(f"Weights (Theta):\n{reg.weights}")
