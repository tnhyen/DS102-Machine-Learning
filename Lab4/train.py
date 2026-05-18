import numpy as np

from sklearn.tree import DecisionTreeClassifier as SklearnDecisionTree
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest

from data.loader import load_data
from utils.split import train_test_split
from utils.metrics import accuracy_score, f1_score_macro

from models.decision_tree import DecisionTreeClassifier
from models.random_forest import RandomForestClassifier


MAX_DEPTH = 6
MIN_SAMPLES_SPLIT = 5
MIN_SAMPLES_LEAF = 2
N_TREES = 10


def print_class_distribution(y):
    labels, counts = np.unique(y, return_counts=True)

    print("\nClass Distribution:")
    for label, count in zip(labels, counts):
        print(f"Quality {label}: {count}")


def evaluate_model(name, y_true, y_pred):
    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("F1 Score:", round(f1_score_macro(y_true, y_pred), 4))


def main():
    X, y = load_data()

    print_class_distribution(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    custom_dt = DecisionTreeClassifier(
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF
    )
    custom_dt.fit(X_train, y_train)

    evaluate_model(
        "Custom Decision Tree",
        y_test,
        custom_dt.predict(X_test)
    )

    custom_rf = RandomForestClassifier(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features="sqrt",
        random_state=42
    )
    custom_rf.fit(X_train, y_train)

    evaluate_model(
        "Custom Random Forest",
        y_test,
        custom_rf.predict(X_test)
    )

    sklearn_dt = SklearnDecisionTree(
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=42
    )
    sklearn_dt.fit(X_train, y_train)

    evaluate_model(
        "Sklearn Decision Tree",
        y_test,
        sklearn_dt.predict(X_test)
    )

    sklearn_rf = SklearnRandomForest(
        n_estimators=N_TREES,
        max_depth=MAX_DEPTH,
        min_samples_split=MIN_SAMPLES_SPLIT,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_features="sqrt",
        random_state=42
    )
    sklearn_rf.fit(X_train, y_train)

    evaluate_model(
        "Sklearn Random Forest",
        y_test,
        sklearn_rf.predict(X_test)
    )


if __name__ == "__main__":
    main()