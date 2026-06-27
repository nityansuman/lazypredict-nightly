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