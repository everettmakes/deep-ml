import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	def sig(x):
		x = np.clip(x, -500, 500)
		return 1 / (1 + np.exp(-x))

	X = np.column_stack([np.ones(len(X)), X])   # intercept column
	n, d = X.shape
	beta = np.zeros(d)
	losses = []
	eps = 1e-15

	for _ in range(iterations):
		p = sig(X @ beta)
		loss = -np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))
		losses.append(round(float(loss), 4))
		grad = X.T @ (p - y)
		beta = beta - learning_rate * grad

	return np.round(beta, 4).tolist(), losses