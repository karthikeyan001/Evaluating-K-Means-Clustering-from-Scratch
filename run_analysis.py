import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans
from silhouette_score_numpy import silhouette_score_numpy
import csv

# Load the dataset saved by generate_dataset.py
X = np.load("features.npy")

Ks = range(1, 11)
SSE = []
SIL = []

for k in Ks:
    model = KMeans(n_clusters=k, max_iter=300, random_state=42)
    model.fit(X)
    SSE.append(model.inertia_)
    if k == 1:
        SIL.append(0.0)
    else:
        SIL.append(silhouette_score_numpy(X, model.labels_))

# Elbow plot (SSE)
plt.figure()
plt.plot(list(Ks), SSE, marker="o")
plt.xlabel("K")
plt.ylabel("SSE")
plt.title("Elbow Method (K = 1..10)")
plt.savefig("elbow.png", dpi=200, bbox_inches="tight")
plt.close()

# Silhouette plot
plt.figure()
plt.plot(list(Ks), SIL, marker="o")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette (K = 1..10)")
plt.savefig("silhouette.png", dpi=200, bbox_inches="tight")
plt.close()

# Save metrics numerically
with open("metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["K", "SSE", "Silhouette"])
    for k, sse, sil in zip(Ks, SSE, SIL):
        writer.writerow([k, sse, sil])

print("Produced: elbow.png, silhouette.png, metrics.csv")
