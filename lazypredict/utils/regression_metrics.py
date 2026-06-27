import numpy as np
from sklearn.metrics import mean_squared_error, r2_score


class RegressionMetrics:
	"""Regression metric helpers."""

	@staticmethod
	def r_squared(y_true, y_pred):
		"""Compute R-squared."""
		return r2_score(y_true, y_pred)

	@staticmethod
	def adjusted_r_squared(y_true, y_pred, n_features):
		"""Compute adjusted R-squared from targets and predictions."""
		r_squared = r2_score(y_true, y_pred)
		n_samples = len(y_true)
		return 1 - (1 - r_squared) * ((n_samples - 1) / (n_samples - n_features - 1))

	@staticmethod
	def mse(y_true, y_pred):
		"""Compute mean squared error."""
		return mean_squared_error(y_true, y_pred)

	@staticmethod
	def rmse(y_true, y_pred):
		"""Compute root mean squared error."""
		return float(np.sqrt(mean_squared_error(y_true, y_pred)))