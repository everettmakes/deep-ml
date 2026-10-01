import numpy as np

def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	X = np.asarray(points, float)
	C = np.asarray(initial_centroids, float).copy()

	for _ in range(max_iterations):
		# 1. Assign: distance from every point to every centroid
		d = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)   # shape (n_points, k)
		labels = d.argmin(axis=1)                                     # nearest centroid per point

		# 2. Update: move each centroid to the mean of its points
		new_C = C.copy()
		for j in range(k):
			if np.any(labels == j):                  # empty cluster → keep old centroid
				new_C[j] = X[labels == j].mean(axis=0)

		if np.allclose(new_C, C):                    # nothing moved → converged
			break
		C = new_C

	return [tuple(round(float(v), 4) for v in c) for c in C]