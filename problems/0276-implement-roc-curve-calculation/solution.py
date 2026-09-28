import numpy as np

def compute_roc_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute ROC curve points (FPR, TPR) for binary classification.
    
    Args:
        y_true: Binary ground truth labels (0 or 1)
        y_scores: Predicted scores/probabilities for the positive class
    
    Returns:
        Tuple of (fpr, tpr) where each is a list of floats
    """
    y_true = np.array(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype=float)

    P = int(y_true.sum())
    N = len(y_true) - P

    fpr, tpr = [0.0], [0.0]
    for t in np.unique(y_scores)[::-1]:
        y_pred = y_scores >= t
        tp = int(np.sum(y_pred & (y_true == 1)))
        fp = int(np.sum(y_pred & (y_true == 0)))
        tpr.append(tp / P if P else 0.0)
        fpr.append(fp / N if N else 0.0)
    
    return fpr, tpr