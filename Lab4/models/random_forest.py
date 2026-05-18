import numpy as np

from models.decision_tree import DecisionTreeClassifier


class RandomForestClassifier:
    def __init__(
        self,
        n_estimators=10,
        max_depth=6,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=42
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        self.trees = []
        self.rng = np.random.default_rng(random_state)

    def fit(self, X, y):
        self.trees = []

        for _ in range(self.n_estimators):
            X_sample, y_sample = self._bootstrap(X, y)
            seed = int(self.rng.integers(1_000_000))

            tree = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                random_state=seed
            )

            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_predictions = np.array([
            tree.predict(X)
            for tree in self.trees
        ])

        tree_predictions = tree_predictions.T

        return np.array([
            self._majority_vote(predictions)
            for predictions in tree_predictions
        ])

    def _bootstrap(self, X, y):
        n_samples = X.shape[0]
        indices = self.rng.choice(
            n_samples,
            size=n_samples,
            replace=True
        )

        return X[indices], y[indices]

    def _majority_vote(self, predictions):
        labels, counts = np.unique(predictions, return_counts=True)
        return labels[np.argmax(counts)]