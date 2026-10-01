import numpy as np

def explained_variance_ratio(X):
    X = np.asarray(X, float)
    Xc = X - X.mean(axis=0)
    cov = Xc.T @ Xc / (len(X) - 1)
    vals = np.linalg.eigvalsh(cov)[::-1]
    vals = np.clip(vals, 0, None)
    return (vals / vals.sum()).tolist()