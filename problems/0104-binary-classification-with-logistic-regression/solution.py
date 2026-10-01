import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	def sig(x):
		x = np.clip(x, -500, 500)
		return 1 / (1 + np.exp(-x))

	z = X @ weights + bias
	y_prob = sig(z)
	return (y_prob >= 0.5).astype(int)