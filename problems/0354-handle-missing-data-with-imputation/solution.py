import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    data = np.asarray(data, dtype=float)

    def nanmode(x):
        x = x[~np.isnan(x)]
        if x.size == 0:
            return np.nan                        # all-NaN column: leave as NaN
        vals, counts = np.unique(x, return_counts=True)
        return vals[np.argmax(counts)]

    if strategy == 'mean':
        fill = np.nanmean(data, axis=0)          # one value per column
    elif strategy == 'median':
        fill = np.nanmedian(data, axis=0)
    elif strategy == 'mode':
        fill = np.array([nanmode(data[:, j]) for j in range(data.shape[1])])
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    return np.where(np.isnan(data), fill, data)  # fill broadcasts across rows