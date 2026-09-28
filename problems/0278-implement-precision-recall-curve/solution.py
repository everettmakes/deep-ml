import numpy as np

def precision_recall_curve(y_true: list, y_scores: list) -> tuple:
    """
    Compute precision-recall pairs for different probability thresholds.
    
    Args:
        y_true: List of true binary labels (0 or 1)
        y_scores: List of predicted probabilities or confidence scores
    
    Returns:
        Tuple of (precisions, recalls, thresholds) where each is a list
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype=float)
    PT = int(y_true.sum())

    precisions, recalls, thresh = [], [], []
    for t in np.unique(y_scores)[::-1]:
        y_pred = y_scores >= t
        PP = int(y_pred.sum())
        TP = int(np.sum(y_pred & (y_true == 1)))
        precisions.append(TP / PP)
        recalls.append(TP / PT if PT else 0.0)
        thresh.append(float(t))

    return precisions, recalls, thresh