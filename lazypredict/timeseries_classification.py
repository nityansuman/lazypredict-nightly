import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from .classifier import LazyClassifier


def make_classification_windows(series, labels, lookback=5, horizon=1):
    """Create aligned feature and label windows for time-series classification."""
    series_values = np.asarray(series)
    if series_values.ndim == 1:
        series_values = series_values.reshape(-1, 1)

    label_values = np.asarray(labels)
    if len(label_values) < lookback + horizon:
        raise ValueError("labels must be long enough for the requested lookback and horizon")

    features = []
    targets = []
    limit = len(series_values) - lookback - horizon + 1
    for index in range(limit):
        features.append(series_values[index : index + lookback].reshape(-1))
        targets.append(label_values[index + lookback : index + lookback + horizon])

    X = np.asarray(features)
    y = np.asarray(targets)
    if horizon == 1:
        y = y.reshape(-1)

    return X, y


def evaluate_ts_classification(y_true, y_pred):
    """Compute accuracy, balanced accuracy, and weighted F1 for classification."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred, average="weighted"),
    }


class LazyTimeSeriesClassifier:
    """Benchmark classification models on supervised time-series windows."""

    def __init__(self, lookback=5, horizon=1, test_size=0.2, random_state=42, **classifier_kwargs):
        self.lookback = lookback
        self.horizon = horizon
        self.test_size = test_size
        self.random_state = random_state
        self.classifier_kwargs = classifier_kwargs
        self.models = {}

    def fit(self, series, labels):
        """Fit all supported classifiers on windowed time-series data."""
        X, y = make_classification_windows(series, labels, lookback=self.lookback, horizon=self.horizon)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, shuffle=False
        )

        classifier = LazyClassifier(**self.classifier_kwargs)
        scores, predictions = classifier.fit(X_train, X_test, y_train, y_test)
        self.models = classifier.models
        return scores, predictions


class LazyTimeSeriesClassification(LazyTimeSeriesClassifier):
    """Alias for LazyTimeSeriesClassifier for a more explicit API name."""
