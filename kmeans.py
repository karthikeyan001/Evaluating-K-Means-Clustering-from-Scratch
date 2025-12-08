import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, tol=1e-4, random_state=42):
        self.k = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        np.random.seed(self.random_state)

        indices = np.random.choice(len(X), self.k, replace=False)
        self.centroids = X[indices]

        for _ in range(self.max_iter):

            distances = np.sqrt(
                ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
            )
            labels = np.argmin(distances, axis=1)

            new_centroids = np.array([
                X[labels == i].mean(axis=0) if np.any(labels == i)
                else self.centroids[i]
                for i in range(self.k)
            ])

            shift = np.sqrt(((new_centroids - self.centroids) ** 2).sum(axis=1)).max()
            self.centroids = new_centroids

            if shift < self.tol:
                break

        self.labels_ = labels
        self.inertia_ = ((X - self.centroids[labels]) ** 2).sum()

        return self

    def predict(self, X):
        distances = np.sqrt(
            ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=2)
        )
        return np.argmin(distances, axis=1)
