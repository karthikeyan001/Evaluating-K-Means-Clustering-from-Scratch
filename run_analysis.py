import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kmeans import KMeans
from silhouette import silhouette_score_numpy

df = pd.read_csv("data.csv")
X = df.values

Ks = range(2, 8)
SSE = []
SIL = []

for k in Ks:
    model = KMeans(n_clusters=k)
    model.fit(X)

    SSE.append(model.inertia_)
    SIL.append(silhouette_score_numpy(X, model.labels_))

plt.plot(Ks, SSE, marker="o")
plt.xlabel("K")
plt.ylabel("SSE")
plt.title("Elbow Method")
plt.savefig("elbow.png")
plt.close()

plt.plot(Ks, SIL, marker="o")
plt.xlabel("K")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Analysis")
plt.savefig("silhouette.png")
plt.close()

print("Analysis completed.")
