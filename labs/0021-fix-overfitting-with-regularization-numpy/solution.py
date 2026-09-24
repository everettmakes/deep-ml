import numpy as np

def train(X_train, y_train, X_val, y_val):
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float).ravel()
    X_val = np.asarray(X_val, dtype=float)
    y_val = np.asarray(y_val, dtype=float).ravel()

    def ridge(X, y, lam):
        p = X.shape[1]
        return np.linalg.solve(X.T @ X + lam * np.eye(p), X.T @ y)

    # Centre y so the intercept isn't penalised
    y_mean = y_train.mean()
    yc = y_train - y_mean

    # Pick lambda by validation error
    best_lam, best_err = None, np.inf
    for lam in np.logspace(-2, 4, 25):
        w = ridge(X_train, yc, lam)
        err = np.mean((X_val @ w + y_mean - y_val) ** 2)
        if err < best_err:
            best_lam, best_err = lam, err

    w = ridge(X_train, yc, best_lam)

    def predict(X):
        return np.asarray(X, dtype=float) @ w + y_mean

    return predict