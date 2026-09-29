import numpy as np

def impute(X: np.ndarray) -> np.ndarray:
    '''
    Fill in missing values (NaN) in the input array.
    
    Args:
        X: Array with possible NaN values, shape (n_samples, n_features)
    
    Returns:
        X_clean: Array with no NaN values, same shape as X
    '''
    X_clean = X.copy()
    
    rows, cols = np.where(np.isnan(X_clean)) 
    fill = np.nanmean(X_clean)
    X_clean[rows, cols] = fill
    
    return X_clean
