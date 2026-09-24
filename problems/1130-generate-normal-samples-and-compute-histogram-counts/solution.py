import numpy as np

def fn(seed, mean, std, n, bins):
    # return (counts, edges) as plain Python lists
    np.random.seed(seed)
    samples = np.random.normal(mean, std, size=n)
    counts, edges = np.histogram(samples, bins)
    return counts, edges