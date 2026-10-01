import numpy as np

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, gamma: float) -> np.ndarray:
    X1, X2 = np.asarray(X1, float), np.asarray(X2, float)
    sq1 = np.sum(X1 ** 2, axis=1)[:, None]          # ‖x‖² as a column, shape (n1, 1)
    sq2 = np.sum(X2 ** 2, axis=1)[None, :]          # ‖y‖² as a row, shape (1, n2)
    sq_dists = sq1 + sq2 - 2 * X1 @ X2.T            # ‖x − y‖² for every pair, shape (n1, n2)
    sq_dists = np.maximum(sq_dists, 0)              # remove tiny negatives from rounding
    return np.exp(-gamma * sq_dists)