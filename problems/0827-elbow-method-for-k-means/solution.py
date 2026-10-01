import numpy as np

def elbow_wcss(X: np.ndarray, k_values: list, max_iters: int = 100) -> list:
    X = np.asarray(X, float)
    out = []
    for k in k_values:
        C = X[:k].copy()                                          # first k rows as centroids
        labels = None
        for _ in range(max_iters):
            d = ((X[:, None, :] - C[None, :, :]) ** 2).sum(axis=2)   # squared distances
            new_labels = d.argmin(axis=1)
            if labels is not None and np.array_equal(new_labels, labels):
                break                                             # assignments stable → stop
            labels = new_labels
            for j in range(k):
                if np.any(labels == j):                           # empty cluster → unchanged
                    C[j] = X[labels == j].mean(axis=0)
        out.append(round(float(((X - C[labels]) ** 2).sum()), 4))   # WCSS
    return out