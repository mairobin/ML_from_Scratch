import numpy as np

class BaseRegression:
    def __init__(self, learning_rate: float = 0.001, n_iters: int = 1000):
        # Assign the variables
        self.learning_rate = learning_rate
        self.n_iters = n_iters

        # Weights
        self.weights = None
        # For Bias add Feature X0 with only value=1 [1,1,...,1]
        # DONT DO: self.bias = 0 # Bias is a single number for all features. Start with Bias == 0


    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Add a column of ones to X to handle bias implicitly
        X = np.column_stack([np.ones(n_samples), X])

        # Initialize Weights
        self.weights = np.zeros(n_features + 1)

        # Minimizing loss, and finding the correct Weights using Gradient Descent
        for _ in range(self.n_iters):
            ### Calculate Predictions (Col Vector)

            # Matrix Vector Multiplication. Result is Column Vector with predictions

            # The Linear Model is fundamental for Linear AND Logistic Regression
            # For Linear Regression, the linear Model is just returned
            # For Logistic Regression, the linear Model is passed to sigmoid function
            y_predicted = self._approximation(X)

            ### Calculate Residuals (Col Vector)
            y_diff = y_predicted - y

            # calculate gradient
            gradient = (1 / n_samples) * np.dot(X.T, y_diff)

            # Go step gradient descent
            self.weights = self.weights - self.learning_rate * gradient


    def predict(self, X):
        # Add a column of ones to X to handle bias implicitly
        X = np.column_stack([np.ones(X.shape[0]), X])
        return self._predict(X)

    def _predict(self, X):
        raise NotImplementedError

    def _approximation(self, X):
        raise NotImplementedError

class LinearRegression(BaseRegression):
    def _approximation(self, X):
        return np.dot(X, self.weights)

    def _predict(self, X):
        return np.dot(X, self.weights)

class LogisticRegression(BaseRegression):
    def _approximation(self, X):
        linear_model = np.dot(X, self.weights)
        return self._sigmoid(linear_model)

    def _predict(self, X):
        linear_model = np.dot(X, self.weights)
        y_predicted = self._sigmoid(linear_model)
        y_predicted_labeles = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_labeles)

    def _sigmoid(self, x):
        return 1 / (np.exp(-x) + 1)

# Testing
if __name__ == "__main__":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    # Utils
    def r2_score(y_true, y_pred):
        corr_matrix = np.corrcoef(y_true, y_pred)
        corr = corr_matrix[0, 1]
        return corr ** 2

    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    # Linear Regression
    X, y = datasets.make_regression(
        n_samples=100, n_features=1, noise=20, random_state=4
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LinearRegression(learning_rate=0.01, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    accu = r2_score(y_test, predictions)
    print("Linear reg Accuracy:", accu)

    # Logistic reg
    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    regressor = LogisticRegression(learning_rate=0.0001, n_iters=1000)
    regressor.fit(X_train, y_train)
    predictions = regressor.predict(X_test)

    print("Logistic reg classification accuracy:", accuracy(y_test, predictions))
