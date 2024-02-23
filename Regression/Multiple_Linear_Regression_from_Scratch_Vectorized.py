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
        X = np.vstack((np.ones((X.shape[0],)), X.T)).T # Add Bias Column
        self.weights = np.zeros((X.shape[1], 1)) # Initialize weights with 0
        n_samples = Y.size
        for i in range(self.n_iterations):

            ### Calculate Predictions (Col Vector)

            # Matrix Vector Multiplication. Result is Column Vector with predictions
            y_pred = np.dot(X, self.weights)

            ### Calculate Residuals (Col Vector)
            y_diff = y_pred - Y

            ### Calculate Gradient
            gradient = (1 / n_samples) * np.dot(X.T, y_diff)
            self.weights = self.weights - self.learning_rate * gradient


    def predict(self, X):
        # Use the precalculated weights for the predictions
        X = np.vstack((np.ones((X.shape[0],)), X.T)).T # Add Bias Column
        y_predictions = np.dot(X, self.weights)
        return y_predictions


if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets


    def r2_score(y_true, y_pred):
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        n = len(y_true)
        mean_true = np.mean(y_true)
        mean_pred = np.mean(y_pred)

        ss_tot = 0  # total sum of squares
        ss_res = 0  # residual sum of squares

        for i in range(n):
            ss_tot += (y_true[i] - mean_true) ** 2
            ss_res += (y_true[i] - y_pred[i]) ** 2

        if ss_tot == 0:
            raise ValueError("Total sum of squares is zero, cannot compute R^2 score")

        r_squared = 1 - (ss_res / ss_tot)
        return r_squared

    # Utils
    """
    def r2_score(y_true, y_pred):
        corr_matrix = np.corrcoef(y_true, y_pred)
        corr = corr_matrix[0, 1]
        return corr ** 2
    """
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Linear Regression
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )
    # Make y a column vector
    y = y.reshape(y.shape[0], 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinearRegression(learning_rate=0.01, n_iterations=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    accu = r2_score(y_test, predictions)
    print("Linear reg Accuracy:", accu)


"""



# Test Data set contains only one Feature

X, y = datasets.make_regression(n_samples=10, n_features=2, noise=20, random_state=8)


# Make y a column vector
y = y.reshape(y.shape[0], 1)

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
"""
print("End")