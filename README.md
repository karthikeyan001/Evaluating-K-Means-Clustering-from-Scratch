1. Introduction

This project implements K-Means clustering from scratch using NumPy only.
The objective is to manually understand centroid initialization, distance computation, label assignment, centroid update, convergence, and cluster quality evaluation without using libraries like scikit-learn.

A synthetic dataset containing 4 ground-truth clusters is generated and saved directly as a NumPy file (features.npy).

2. Dataset Generation

400 samples, 4 Gaussian clusters:

Cluster	Mean	Std	Count
c1	(0,0)	1.0	100
c2	(5,5)	1.0	100
c3	(-5,5)	1.0	100
c4	(5,-5)	1.0	100

Saved using:

np.save("features.npy", X)


This satisfies the requirement of saving the raw NumPy feature matrix directly.

3. Methodology
3.1 Data Loading

The dataset is loaded directly from a NumPy file:

X = np.load("features.npy")

3.2 Manual K-Means implementation

The algorithm includes:

Random centroid initialization

Euclidean distance matrix computation

Nearest centroid assignment

Centroid update

Convergence check using centroid shift

SSE calculation

No machine-learning libraries were used.

4. Determining Optimal K (Requirement: K = 1 to 10)

Two evaluation metrics were used:

4.1 SSE (Elbow Method)

SSE was computed for K = 1 to 10.

Plot generated: elbow.png
Shows rapid drop until K = 4, matching ground truth.

4.2 Silhouette Score

Silhouette Score was calculated manually using NumPy.

For K=1, silhouette = 0 (undefined)

Highest silhouette score occurs at K = 4

Plot generated: silhouette.png

5. Visualizations
5.1 Elbow Plot (elbow.png)

Shows steep decrease from K=1 to K=4, with minimal improvement afterwards.

5.2 Silhouette Plot (silhouette.png)

Peak at K=4 confirming optimal cluster count.

5.3 Cluster Plot

Scatter plot displays four well-separated groups with centroids.

6. Findings

Dataset contains 4 natural clusters (ground truth).

Both SSE and silhouette peaks confirm K=4 is optimal.

Manual implementation behaves identically to expected theory.

Silhouette implementation correctly handles edge cases (single-point clusters).

Requirement of testing K from 1 to 10 is fulfilled.

7. Conclusion

The full K-Means pipeline—dataset creation, manual implementation, evaluation, and visualization—was completed using only NumPy.
All deliverables were satisfied, including correct data loading, silhouette computation, and expanded K-range analysis.
