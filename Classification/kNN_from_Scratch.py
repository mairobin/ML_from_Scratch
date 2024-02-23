import numpy as np
from collections import Counter

class kNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        # Method needs no logic as no model needs to be trained
        # Predictions comes only from scanning the neighbors
        self.X = X
        self.y = y

    def _euclidean_dist(self, point1, point2):
        # Point can be n-dimensional
        # Therefore for every dimension find absolute difference of p1-p2
        # Sum the distances of every dimension with np.sum()
        return np.sqrt(np.sum(point1-point2)**2)


    def predict(self, X):
        # X = data points with no labels
        # Use helper function to predict class of a single point
        labels = [self._predict_point(x) for x in X]
        return np.array(labels)

    def _predict_point(self, x):
        # Calculate distances
        distances = [self._euclidean_dist(x, p) for p in self.X]

        # Find nearest neighbors
        nn = np.argsort(distances) # sorts indices
        kNN_indices = nn[:self.k]
        kNN_labels = [self.y[i] for i in kNN_indices]

        # classifaction by majority vote
        counter = Counter(kNN_labels)
        winner = counter.most_common(1)  # Give the top n=1 most counted elements.
        return winner[0][0]  # returns a list of tuple. In Tuple Index 0 returns the element, Index 1 would return the count


from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
cmap = ListedColormap(['red', 'green', 'blue'])
iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

k = 5
model = kNN(k)
model.fit(X_train, y_train)
predictions = model.predict(X_test)


## Accuracy - where prediction matches actual

acc = np.sum(predictions==y_test) / len(y_test)

print(f"Accuracy: {acc}")
