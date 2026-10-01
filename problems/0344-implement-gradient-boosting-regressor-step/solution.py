import numpy as np

def gradient_boosting_step(X, y, current_predictions, learning_rate=0.1):
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    pred = np.asarray(current_predictions, float)

    r = y - pred                                   # 1. residuals: what's still wrong

    best_mse, best = np.inf, None                  # 2. fit a stump to the residuals
    for f in range(X.shape[1]):
        vals = np.unique(X[:, f])
        for t in (vals[:-1] + vals[1:]) / 2:       # midpoints, as in the random forest
            left = X[:, f] <= t
            fix = np.where(left, r[left].mean(), r[~left].mean())   # each side → its average
            mse = np.mean((r - fix) ** 2)          # how well does that fit the residuals?
            if mse < best_mse:
                best_mse, best = mse, fix

    if best is None:                               # every feature value identical: no split
        best = np.full(len(r), r.mean())

    return np.round(pred + learning_rate * best, 4).tolist()   # 3. nudge the predictions