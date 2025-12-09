import numpy as np

def silhouette_score_numpy(X, labels):
    n = len(X)
    unique = np.unique(labels)
    if len(unique) == 1:
        return 0.0

    # full distance matrix
    dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))
    scores = []

    for i in range(n):
        own = labels[i]
        same_mask = (labels == own)
        same_mask[i] = False

        if same_mask.sum() == 0:
            a = 0.0
        else:
            a = dist[i][same_mask].mean()

        b_vals = []
        for other in unique:
            if other == own: continue
            mask = (labels == other)
            if mask.sum() > 0:
                b_vals.append(dist[i][mask].mean())

        b = min(b_vals) if b_vals else 0.0

        if max(a, b) == 0:
            s = 0.0
        else:
            s = (b - a) / max(a, b)
        scores.append(s)

    return float(np.mean(scores))
