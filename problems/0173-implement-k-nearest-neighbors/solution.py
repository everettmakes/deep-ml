import numpy as np

def k_nearest_neighbors(points, query_point, k):
    P = np.asarray(points, float)
    q = np.asarray(query_point, float)
    dists = np.linalg.norm(P - q, axis=1)            # distance from query to every point
    idx = np.argsort(dists, kind='stable')[:k]       # stable → ties keep input order
    return [tuple(points[i]) for i in idx]