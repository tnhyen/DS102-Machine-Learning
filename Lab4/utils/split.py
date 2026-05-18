import numpy as np


def train_test_split(X, y, test_size=0.2, random_state=42):
    rng = np.random.default_rng(random_state)

    indices = np.arange(len(X))
    rng.shuffle(indices)

    split_idx = int(len(X) * (1 - test_size))

    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]