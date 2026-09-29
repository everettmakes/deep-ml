import numpy as np

def smote(X_minority: np.ndarray, n_synthetic: int, k: int = 5) -> np.ndarray:
    X = np.asarray(X_minority, dtype=float)
    n_samples, n_features = X.shape
    k_actual = min(k, n_samples - 1)
    if k_actual == 0 or n_synthetic == 0:
        return np.zeros((0, n_features))

    out = []
    for _ in range(n_synthetic):
        i = np.random.randint(0, n_samples)                 # pick a minority sample
        x_i = X[i]
        dists = np.linalg.norm(X - x_i, axis=1)             # distance to every sample
        nbrs = np.argsort(dists)[1:k_actual + 1]            # k nearest, skipping itself
        x_nn = X[nbrs[np.random.randint(0, k_actual)]]      # one random neighbour
        gap = np.random.random()
        out.append(x_i + gap * (x_nn - x_i))                # point on the line between them
    return np.array(out)