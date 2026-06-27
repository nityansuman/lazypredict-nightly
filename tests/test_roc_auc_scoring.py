import numpy as np
from sklearn.metrics import roc_auc_score

from lazypredict.supervised import get_roc_auc_score


class ProbabilityEstimator:
    def predict_proba(self, X):
        return np.array(
            [
                [0.9, 0.1],
                [0.7, 0.3],
                [0.2, 0.8],
                [0.1, 0.9],
            ]
        )


class DecisionEstimator:
    def decision_function(self, X):
        return np.array([-2.0, -1.0, 1.0, 2.0])


class LabelEstimator:
    def predict(self, X):
        return np.array([0, 0, 1, 1])


def test_get_roc_auc_score_uses_predict_proba():
    estimator = ProbabilityEstimator()
    y_test = np.array([0, 0, 1, 1])

    score = get_roc_auc_score(estimator, np.zeros((4, 1)), y_test)

    assert score == roc_auc_score(y_test, np.array([0.1, 0.3, 0.8, 0.9]))


def test_get_roc_auc_score_uses_decision_function():
    estimator = DecisionEstimator()
    y_test = np.array([0, 0, 1, 1])

    score = get_roc_auc_score(estimator, np.zeros((4, 1)), y_test)

    assert score == roc_auc_score(y_test, np.array([-2.0, -1.0, 1.0, 2.0]))


def test_get_roc_auc_score_falls_back_to_labels():
    estimator = LabelEstimator()
    y_test = np.array([0, 0, 1, 1])

    score = get_roc_auc_score(estimator, np.zeros((4, 1)), y_test)

    assert score == roc_auc_score(y_test, np.array([0, 0, 1, 1]))