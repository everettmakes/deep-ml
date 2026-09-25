import numpy as np
import pandas as pd
def stratified_kfold_indices(y, n_splits):
	"""
	Generate train/test indices for stratified K-fold cross-validation.

	Args:
		y: 1D array-like of integer class labels
		n_splits: number of folds

	Returns:
		A list of [train_indices, test_indices] pairs, one per fold.
	"""
	y = np.array(y)
	idxs = np.arange(len(y))

	groups = []
	for c in np.unique(y):
		idx = np.where(y == c)[0]
		groups.append(np.array_split(idx, n_splits))
	
	folds = []
	for i in range(n_splits):
		test = np.sort(np.concatenate([group[i] for group in groups]))
		train = np.setdiff1d(idxs, test)
		folds.append([train.tolist(), test.tolist()])
	
	return folds





