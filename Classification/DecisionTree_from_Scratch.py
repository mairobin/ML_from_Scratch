from collections import Counter

import numpy as np

def entropy(labels):
    # Count occurences of each label and store in dict
    label_counts = {} # empty dict
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1 # retrieving the current count (or 0 if the label is not found) and add 1

    # Calculate probability p(A) for each class
    entropy = 0
    for count in label_counts.values():
        p = count / len(labels)
        entropy = entropy - p * np.log2(p)
    return entropy

class Node:
    def __init__(
        self, split_feature=None, split_threshold=None, left=None, right=None, *, label=None
    ):
        self.split_feature = split_feature
        self.split_threshold = split_threshold
        self.left_child = left
        self.right_child = right
        self.label = label # most common label in leaf node

    def is_leaf_node(self):
        return self.label is not None


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, considered_features=None):
        """
        :param min_samples_split:  minimum number of samples required to split a node further
        :param max_depth: depth of the tree
        :param considered_features: number of features that the algorithm considers at each split, preventing overfitting, complexity
        :param root: reference to root of the tree
        """
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.considered_features = considered_features
        self.root = None

    def fit(self, X, y):
        considered_features = X.shape[1] # if considered_features not initialized, consider all features
        self.considered_features = considered_features if not self.considered_features else min(self.considered_features, considered_features)
        self.root = self._grow_tree(X, y) # fit the model and return the root

    def predict(self, X):
        # for every data row traverse tree and return label of the leaf node
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # recursively grow tree
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Base Case - stopping Criteria
        if (
            depth >= self.max_depth
            or n_labels == 1
            or n_samples < self.min_samples_split
        ):
            leaf_value = self._most_common_label(y)
            return Node(label=leaf_value)

        # As long as Stopping Criteria not met
        # consider only a subset of features for splitting.
        # returns the indices of the considered features
        feat_idxs = np.random.choice(n_features, self.considered_features, replace=False)

        # greedily select the best split according to information gain
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        # grow the children that result from the split
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)

        # recursively grow the subtrees

        #consider only the rows matching the split criteria
        # left - all rows where best feature is below or equal threshold
        # right - all rows where best feature is greater than threshold
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1) # recursively grow the left subtree, slice rows keep columns
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1) # recursively grow the right subtree
        return Node(best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        # for every considered feature
        for feat_idx in feat_idxs:
            # Get the feature column
            X_column = X[:, feat_idx]
            # get the possible thresholds => occurring values without duplicates
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # Get the Information Gain for every possible threshold
                gain = self._information_gain(y, X_column, threshold)

                # Keep track of the best feature and its threshold
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        # parent loss
        parent_entropy = entropy(y)

        # generate split
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        # If the threshold is not splitting anything the gain is zero
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # compute the weighted avg. of the loss for the children
        n_all_elements = len(y)
        n_elements_left, n_elements_right = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = entropy(y[left_idxs]), entropy(y[right_idxs])

        ## weighting the entropy of each branch by how many elements it has
        child_entropy = (n_elements_left / n_all_elements) * entropy_left + (n_elements_right / n_all_elements) * entropy_right

        # information gain is difference in loss before vs. after split
        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        # split the feature values with respect to the threshold

        left_idxs = np.argwhere(X_column <= split_thresh).flatten() # less than or equal to the threshold
        right_idxs = np.argwhere(X_column > split_thresh).flatten() # greater than the threshold
        return left_idxs, right_idxs

    def _traverse_tree(self, x, node):
        # recursively traverse tree starting from the root ending in a lead
        if node.is_leaf_node():
            return node.label

        if x[node.split_feature] <= node.split_threshold:
            return self._traverse_tree(x, node.left_child)
        return self._traverse_tree(x, node.right_child)

    def _most_common_label(self, y):
        # get most common label in a Node

        # Calculate the number of occurences for every label
        counter = Counter(y)
        # get most common
        most_common = counter.most_common(1)[0][0]
        return most_common


if __name__ == "__main__":
    # Imports
    from sklearn import datasets
    from sklearn.model_selection import train_test_split

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    data = datasets.load_breast_cancer()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1234
    )

    clf = DecisionTree(max_depth=10)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy(y_test, y_pred)

    print("Accuracy:", acc)
