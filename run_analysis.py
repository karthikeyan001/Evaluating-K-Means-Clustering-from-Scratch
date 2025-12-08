import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans
from silhouette_score_numpy import silhouette_score_numpy

# Load correct dataset
X = np.load("features.npy")

Ks = range(1, 11)
SSE = []
SIL = []

for k in Ks:
    model = KMeans(n_clusters=k)
    model.fit(X)

    SSE.append(model.inertia_)

    # K=1 → silhouette = 0 (undefined)
    if k == 1:
        SIL.append(0)
    else:
        SIL.append(silhouette_score_numpy(X, model.labels_))

# Save plots
plt.plot(Ks, SSE, marker="o")
plt.xlabel("K")
plt.ylabel("SSE")
plt.title("Elbow Method (K=1 to 10)")
plt.savefig("elbow.png")
plt.close()

plt.plot(Ks, SIL, marker="o")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis (K=1 to 10)")
plt.savefig("silhouette.png")
plt.close()

print("Analysis completed for K=1 to 10.")
