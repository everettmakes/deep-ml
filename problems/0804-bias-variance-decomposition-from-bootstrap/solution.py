import numpy as np

def bias_variance_decomp(predictions, y_true):
    """
    Compute the empirical bias-variance decomposition from bootstrap predictions.

    Args:
        predictions: array-like of shape (B, M) - predictions from B models at M test points
        y_true: array-like of shape (M,) - true target values

    Returns:
        dict with keys 'bias_squared', 'variance', 'mse'
    """
    ps = np.asarray(predictions)
    results = []
    for i in range(ps.shape[1]):
        t = ps[:, i]
        mean_prediction = np.mean(t)
        squared_bias = (mean_prediction - y_true[i]) ** 2
        variance = np.sum(((t - mean_prediction) ** 2)) / ps.shape[0]
        mse = np.mean(((t - y_true[i])**2))
        results.append([squared_bias, variance, mse])
    results = np.asarray(results)
    return {'bias_squared': float(np.mean(results[:, 0])), 'variance': float(np.mean(results[:, 1])), 'mse': float(np.mean(results[:, 2]))}

