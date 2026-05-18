import numpy as np


class DecisionTreeClassifier:
    def __init__(
        self,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=None,
        random_state=None
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.tree = None
        self.classes_ = None
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.tree = self._build_tree(X, y, depth=0)

    def predict(self, X):
        return np.array([
            self._predict_one(row, self.tree)
            for row in X
        ])

    def _build_tree(self, X, y, depth):
        prediction = self._majority_class(y)

        node = {
            "type": "leaf",
            "value": prediction
        }

        if self._should_stop(X, y, depth):
            return node

        feature, threshold = self._best_split(X, y)

        if feature is None:
            return node

        left_idx = X[:, feature] <= threshold
        right_idx = ~left_idx

        if (
            np.sum(left_idx) < self.min_samples_leaf
            or np.sum(right_idx) < self.min_samples_leaf
        ):
            return node

        return {
            "type": "node",
            "feature": feature,
            "threshold": threshold,
            "left": self._build_tree(X[left_idx], y[left_idx], depth + 1),
            "right": self._build_tree(X[right_idx], y[right_idx], depth + 1)
        }

    def _should_stop(self, X, y, depth):
        return (
            len(y) < self.min_samples_split
            or len(np.unique(y)) == 1
            or (
                self.max_depth is not None
                and depth >= self.max_depth
            )
        )

    def _best_split(self, X, y):
        best_feature = None
        best_threshold = None
        best_gini = self._gini(y)

        n_features = X.shape[1]
        feature_indices = self._feature_indices(n_features)

        for feature in feature_indices:
            sorted_idx = np.argsort(X[:, feature])
            X_sorted = X[sorted_idx, feature]
            y_sorted = y[sorted_idx]

            for i in range(1, len(y)):
                if X_sorted[i] == X_sorted[i - 1]:
                    continue

                threshold = (X_sorted[i] + X_sorted[i - 1]) / 2

                left_y = y_sorted[:i]
                right_y = y_sorted[i:]

                if (
                    len(left_y) < self.min_samples_leaf
                    or len(right_y) < self.min_samples_leaf
                ):
                    continue

                weighted_gini = self._weighted_gini(left_y, right_y)

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    def _feature_indices(self, n_features):
        if self.max_features is None:
            return np.arange(n_features)

        if self.max_features == "sqrt":
            size = max(1, int(np.sqrt(n_features)))
        else:
            size = int(self.max_features)

        return self.rng.choice(n_features, size=size, replace=False)

    def _gini(self, y):
        gini = 1.0

        for cls in np.unique(y):
            p = np.sum(y == cls) / len(y)
            gini -= p ** 2

        return gini

    def _weighted_gini(self, left_y, right_y):
        total = len(left_y) + len(right_y)

        return (
            len(left_y) / total * self._gini(left_y)
            + len(right_y) / total * self._gini(right_y)
        )

    def _majority_class(self, y):
        labels, counts = np.unique(y, return_counts=True)
        return labels[np.argmax(counts)]

    def _predict_one(self, row, node):
        if node["type"] == "leaf":
            return node["value"]

        if row[node["feature"]] <= node["threshold"]:
            return self._predict_one(row, node["left"])

        return self._predict_one(row, node["right"])