import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    X = np.asarray(data, float)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    cov = np.cov(X, rowvar=False)

    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1][:k]
    pcs = vecs[:, order]

    for j in range(pcs.shape[1]):
        v = pcs[:, j]
        if v[np.abs(v) > 1e-10][0] < 0:
            pcs[:, j] = -v

    return np.round(pcs, 4)