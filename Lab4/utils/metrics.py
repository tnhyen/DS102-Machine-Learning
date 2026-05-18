import numpy as np


def accuracy_score(y_true, y_pred):
    return np.mean(y_true == y_pred)


def f1_score_macro(y_true, y_pred):
    classes = np.unique(y_true)
    scores = []

    for cls in classes:
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)

        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        scores.append(f1)

    return np.mean(scores)