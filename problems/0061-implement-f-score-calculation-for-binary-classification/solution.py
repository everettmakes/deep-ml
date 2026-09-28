import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	tp = sum((y_true == 1) & (y_pred == 1))
	fp = sum((y_true == 0) & (y_pred == 1))
	fn = sum((y_true == 1) & (y_pred == 0))
	precision = tp / (tp + fp)
	recall = tp / (tp + fn)
	return (1 + beta ** 2) * (precision * recall) / ((precision * beta ** 2) + recall)
