import numpy as np

def calculate_auc(y_true, y_scores):
    """
    Calculate the Area Under the ROC Curve (AUC).
    
    Args:
        y_true: List or array of binary ground truth labels (0 or 1)
        y_scores: List or array of predicted probabilities or confidence scores
        
    Returns:
        AUC value as a float
    """
    y_true = np.asarray(y_true).astype(int)
    y_scores = np.asarray(y_scores, dtype=float)

    P = int(y_true.sum())
    N = len(y_true) - P 
    if P == 0 or N == 0:
        return 0.0

    fpr, tpr = [0.0], [0.0]
    for t in np.unique(y_scores)[::-1]:
        y_pred = y_scores >= t 
        tpr.append(np.sum(y_pred & (y_true == 1)) / P) 
        fpr.append(np.sum(y_pred & (y_true == 0)) / N) 
    
    auc = 0.0
    for i in range(1, len(fpr)):
        auc += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2

    return float(auc)