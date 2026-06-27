import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score


class ClassificationMetrics:
	"""Classification metric helpers."""

	@staticmethod
	def accuracy(y_true, y_pred):
		"""Compute plain accuracy."""
		return accuracy_score(y_true, y_pred)

	@staticmethod
	def balanced_accuracy(y_true, y_pred):
		"""Compute balanced accuracy."""
		return balanced_accuracy_score(y_true, y_pred)

	@staticmethod
	def f1(y_true, y_pred):
		"""Compute weighted F1."""
		return f1_score(y_true, y_pred, average="weighted")

	@staticmethod
	def roc_auc(estimator, X_test, y_test):
		"""Compute ROC AUC from probabilistic or margin-based outputs."""
		if hasattr(estimator, "predict_proba"):
			y_score = estimator.predict_proba(X_test)
			if y_score.ndim == 2 and y_score.shape[1] > 2:
				return roc_auc_score(y_test, y_score, multi_class="ovr")
			return roc_auc_score(y_test, y_score[:, 1])

		if hasattr(estimator, "decision_function"):
			y_score = estimator.decision_function(X_test)
			if np.ndim(y_score) == 2 and y_score.shape[1] > 2:
				return roc_auc_score(y_test, y_score, multi_class="ovr")
			return roc_auc_score(y_test, y_score)

		return roc_auc_score(y_test, estimator.predict(X_test))