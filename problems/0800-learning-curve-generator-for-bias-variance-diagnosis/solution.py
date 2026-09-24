import numpy as np

def learning_curve(X_train, y_train, X_val, y_val, train_sizes, degree, bias_threshold=0.5, variance_threshold=0.5):
    """
    Generate a learning curve and diagnose bias vs variance.

    Returns a dict with 'train_errors', 'val_errors', and 'diagnosis'.
    """
    y_train = np.asarray(y_train).ravel()
    y_val = np.asarray(y_val).ravel()
    X_train = np.asarray(X_train).ravel()
    X_val = np.asarray(X_val).ravel()

    x_val = np.vander(X_val, degree+1, increasing=True)
    train_mse, val_mse = [], []

    for n in train_sizes:
        X = np.vander(X_train[:n], degree+1, increasing=True)
        y = y_train[:n]

        w = np.linalg.pinv(X) @ y

        train_mse.append(float(np.mean((X @ w - y) ** 2)))
        val_mse.append(float(np.mean((x_val @ w - y_val) ** 2)))
    
    if train_mse[-1] > bias_threshold:
        diagnosis = 'high_bias'
    elif val_mse[-1] - train_mse[-1] > variance_threshold:
        diagnosis = 'high_variance'
    else:
        diagnosis = 'good_fit'
    
    return {'train_errors': train_mse, 'val_errors': val_mse, 'diagnosis': diagnosis}
    
