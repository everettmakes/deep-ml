import numpy as np

def silhouette_score(X: np.ndarray, labels: np.ndarray) -> float:
	X, labels = np.asarray(X, float), np.asarray(labels)
	clusters = np.unique(labels)
	n = len(X)
	if len(clusters) < 2 or len(clusters) == n:          # edge cases from the spec
		return 0.0

	D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)   # all pairwise distances

	s = np.zeros(n)
	for i in range(n):
		same = labels == labels[i]
		if same.sum() == 1:                              # alone in its cluster → s = 0
			continue
		a = D[i, same].sum() / (same.sum() - 1)          # exclude distance to itself
		b = min(D[i, labels == c].mean() for c in clusters if c != labels[i])
		s[i] = (b - a) / max(a, b)

	return round(float(s.mean()), 4)