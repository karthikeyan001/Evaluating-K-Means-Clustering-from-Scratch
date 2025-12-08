def silhouette_score_numpy(X, labels):
    n = len(X)
    unique = np.unique(labels)
    dist = np.sqrt(((X[:, None, :] - X[None, :, :]) ** 2).sum(axis=2))

    scores = []

    for i in range(n):
        own = labels[i]

        same_cluster = labels == own
        if same_cluster.sum() > 1:
            a = dist[i][same_cluster].sum() / (same_cluster.sum() - 1)
        else:
            a = 0

        b = min(
            dist[i][labels == other].mean()
            for other in unique if other != own
        )

        s = (b - a) / max(a, b)
        scores.append(s)

    return np.mean(scores)
