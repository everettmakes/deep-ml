import numpy as np
def precision(y_true, y_pred):
	tp = sum((y_true == 1) & (y_pred == 1))
	fp = sum((y_true == 0) & (y_pred == 1))
	if tp + fp == 0:
		return 0
	return tp / (tp + fp)
