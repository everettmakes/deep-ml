import numpy as np

def bagging_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, n_estimators: int = 10, seed: int = 42) -> np.ndarray:

    def fit_stump(X, y):
        best_err, best = np.inf, None
        for f in range(X.shape[1]):
            for t in np.unique(X[:, f]):
                for direction in (1, -1):
                    pred = np.where(X[:, f] <= t, 1, 0)
                    if direction == -1:
                        pred = 1 - pred
                    err = np.sum(pred != y)
                    if err < best_err:
                        best_err, best = err, (f, t, direction)
        return best

    def predict_stump(stump, X):
        f, t, direction = stump
        pred = np.where(X[:, f] <= t, 1, 0)
        return pred if direction == 1 else 1 - pred

    X, y = np.asarray(X_train, float), np.asarray(y_train)
    Xt = np.asarray(X_test, float)
    rs = np.random.RandomState(seed)
    n = len(X)

    preds = []
    for _ in range(n_estimators):
        idx = rs.randint(0, n, n)
        preds.append(predict_stump(fit_stump(X[idx], y[idx]), Xt))

    preds = np.array(preds)                       # (n_estimators, n_test)
    return (preds.mean(axis=0) > 0.5).astype(int)