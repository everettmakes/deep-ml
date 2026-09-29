import numpy as np

def focal_loss(y_true, y_pred, gamma=2.0, alpha=None):
	"""
	Compute Focal Loss for multi-class classification.
	
	Args:
		y_true: Ground truth labels as class indices (list or 1D array)
		y_pred: Predicted probabilities (2D array, shape: [n_samples, n_classes])
		gamma: Focusing parameter (default: 2.0)
		alpha: Class weights (optional, list or 1D array of length n_classes)
	
	Returns:
		float: Average focal loss
	"""
	y_true = np.asarray(y_true)
	y_pred = np.asarray(y_pred)
	n_samples, n_classes = y_pred.shape
	y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
	p_ts = y_pred[np.arange(n_samples), y_true]
	if alpha is None:
		alpha_t = np.ones(n_samples)
	else:
		alpha_t = np.asarray(alpha)[y_true]
	fl = np.mean(-alpha_t * (1 - p_ts)**gamma * np.log(p_ts))
	return float(fl)