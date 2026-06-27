import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score


class TimeSeriesMetrics:
	"""Time-series metric helpers."""

	@staticmethod
	def accuracy(y_true, y_pred):
		"""Compute plain accuracy for time-series classification."""
		return accuracy_score(y_true, y_pred)

	@staticmethod
	def balanced_accuracy(y_true, y_pred):
		"""Compute balanced accuracy for time-series classification."""
		return balanced_accuracy_score(y_true, y_pred)

	@staticmethod
	def f1(y_true, y_pred):
		"""Compute weighted F1 for time-series classification."""
		return f1_score(y_true, y_pred, average="weighted")

	@staticmethod
	def mae(y_true, y_pred):
		"""Compute mean absolute error for forecasting."""
		return mean_absolute_error(y_true, y_pred)

	@staticmethod
	def mse(y_true, y_pred):
		"""Compute mean squared error for forecasting."""
		return mean_squared_error(y_true, y_pred)

	@staticmethod
	def rmse(y_true, y_pred):
		"""Compute root mean squared error for forecasting."""
		return float(np.sqrt(mean_squared_error(y_true, y_pred)))

	@staticmethod
	def r2(y_true, y_pred):
		"""Compute R-squared for forecasting."""
		return r2_score(y_true, y_pred)

	@staticmethod
	def classification(y_true, y_pred):
		"""Compute accuracy, balanced accuracy, and weighted F1 for classification."""
		return {
			"Accuracy": TimeSeriesMetrics.accuracy(y_true, y_pred),
			"Balanced Accuracy": TimeSeriesMetrics.balanced_accuracy(y_true, y_pred),
			"F1 Score": TimeSeriesMetrics.f1(y_true, y_pred),
		}

	@staticmethod
	def forecasting(y_true, y_pred):
		"""Compute standard forecasting metrics for time-series predictions."""
		y_true = np.asarray(y_true)
		y_pred = np.asarray(y_pred)

		metrics = {
			"MAE": TimeSeriesMetrics.mae(y_true, y_pred),
			"RMSE": TimeSeriesMetrics.rmse(y_true, y_pred),
			"R2": TimeSeriesMetrics.r2(y_true, y_pred),
		}
		if y_true.ndim == 2 and y_true.shape[1] > 1:
			metrics["MSE"] = TimeSeriesMetrics.mse(y_true, y_pred)

		return metrics