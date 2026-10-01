import numpy as np

def xgboost_objective(gradients: np.ndarray, hessians: np.ndarray,
                      left_indices: np.ndarray, right_indices: np.ndarray,
                      lambda_reg: float = 1.0, gamma: float = 0.0) -> dict:
    g, h = np.asarray(gradients, float), np.asarray(hessians, float)
    left, right = np.asarray(left_indices, int), np.asarray(right_indices, int)

    G_L, H_L = g[left].sum(), h[left].sum()
    G_R, H_R = g[right].sum(), h[right].sum()
    G, H = G_L + G_R, H_L + H_R

    def score(G, H):
        return G ** 2 / (H + lambda_reg)

    gain = 0.5 * (score(G_L, H_L) + score(G_R, H_R) - score(G, H)) - gamma

    return {
        'left_weight': round(float(-G_L / (H_L + lambda_reg)), 4),
        'right_weight': round(float(-G_R / (H_R + lambda_reg)), 4),
        'gain': round(float(gain), 4),
    }