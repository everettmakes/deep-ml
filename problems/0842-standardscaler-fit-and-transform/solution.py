import numpy as np

def standard_scaler(X_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Fit a standard scaler on X_train and transform X_test.
    Returns the standardized X_test as a numpy array.
    """
    mu = X_train.mean(axis=0)
    s = X_train.std(axis=0)
    s[s == 0] = 1

    return (X_test - mu) / s
