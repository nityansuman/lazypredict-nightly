import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from lazypredict import LazyClassifier


def test_lazy_classifier_runs_on_dummy_data():
    X, y = make_classification(n_samples=80, n_features=6, n_informative=4, random_state=42)
    split = 60
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        classifiers=[LogisticRegression, DecisionTreeClassifier],
        predictions=True,
    )

    scores, predictions = clf.fit(X_train, X_test, y_train, y_test)

    assert not scores.empty
    assert list(predictions.columns) == ["LogisticRegression", "DecisionTreeClassifier"]
    assert np.isfinite(scores["Accuracy"]).all()


def test_lazy_classifier_handles_imbalanced_multiclass():
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.8, 0.15, 0.05],
        class_sep=1.5,
        random_state=42,
    )

    split = 90
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    clf = LazyClassifier(
        verbose=0,
        ignore_warnings=True,
        classifiers=[LogisticRegression, DecisionTreeClassifier],
        predictions=False,
    )

    scores, _ = clf.fit(X_train, X_test, y_train, y_test)

    assert not scores.empty
    assert np.isfinite(scores["Balanced Accuracy"]).all()
    assert scores["Balanced Accuracy"].between(0, 1).all()