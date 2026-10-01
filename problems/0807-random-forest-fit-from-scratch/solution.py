import numpy as np

def random_forest_fit_predict(X_train, y_train, X_test, n_trees: int = 5, max_depth: int = 3, max_features: int = 2, seed: int = 0):
	"""
	Train a random forest classifier and predict on X_test.
	Returns a list of integer class predictions.
	"""

	def gini(y):
		_, counts = np.unique(y, return_counts=True)
		p = counts / len(y)
		return 1 - np.sum(p ** 2)

	def majority(y):
		vals, counts = np.unique(y, return_counts=True)
		return int(vals[np.argmax(counts)])

	def best_split(X, y, features):
		best_score, best = np.inf, None
		for f in features:
			vals = np.unique(X[:, f])
			for t in (vals[:-1] + vals[1:]) / 2:
				left = X[:, f] <= t
				yl, yr = y[left], y[~left]
				score = (len(yl) * gini(yl) + len(yr) * gini(yr)) / len(y)
				if score < best_score:
					best_score, best = score, (f, t)
		return best

	def build(X, y, depth, max_depth, max_features, rs):
		if len(np.unique(y)) == 1 or len(y) < 2 or depth == max_depth:
			return {'leaf': majority(y)}
		features = rs.choice(X.shape[1], max_features, replace=False)
		split = best_split(X, y, features)
		if split is None:
			return {'leaf': majority(y)}
		f, t = split
		left = X[:, f] <= t
		return {'f': f, 't': t,
			'L': build(X[left], y[left], depth + 1, max_depth, max_features, rs),
			'R': build(X[~left], y[~left], depth + 1, max_depth, max_features, rs)}

	def predict_one(node, x):
		while 'leaf' not in node:
			node = node['L'] if x[node['f']] <= node['t'] else node['R']
		return node['leaf']

	X, y = np.asarray(X_train, float), np.asarray(y_train)
	rs = np.random.RandomState(seed)
	n = len(X)
	trees = []
	for _ in range(n_trees):
		idx = rs.randint(0, n, n)
		trees.append(build(X[idx], y[idx], 0, max_depth, max_features, rs))
	return [majority([predict_one(t, x) for t in trees])
			for x in np.asarray(X_test, float)]