import numpy as np

def permutation_importance(X, y, weights, bias, permutations):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.asarray(weights, dtype=float)

    def r2(y_true, y_pred):
        return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

    baseline = r2(y, X @ w + bias)
    importances = []
    for j, perms_j in enumerate(permutations):
        scores = []
        for p in perms_j:
            X_perm = X.copy()
            X_perm[:, j] = X[p, j]
            scores.append(r2(y, X_perm @ w + bias))
        importances.append(float(baseline - np.mean(scores)))
    return importances