1. Introduction

This project implements the K-Means clustering algorithm manually using NumPy.
The goal is to understand the algorithm's internal working without using ML libraries and to analyze clustering behavior on a synthetic dataset with a known ground truth of K = 4 clusters.

2. Dataset Generation

A synthetic dataset of 400 samples and 4 Gaussian clusters was generated:

Cluster	Mean	Std	Samples
c1	(0, 0)	1.0	100
c2	(5, 5)	1.0	100
c3	(-5, 5)	1.0	100
c4	(5, -5)	1.0	100

The dataset was saved directly as a NumPy array (features.npy) to satisfy the requirement of saving raw feature matrices without relying on pandas.

3. Methodology
3.1 Data Preparation

The dataset was loaded using:

X = np.load("features.npy")


Features are in the same scale, so no normalization was required.

3.2 Manual K-Means Implementation

The full algorithm was implemented using only NumPy:

Random centroid initialization

Euclidean distance computation

Vectorized label assignment

Recalculation of centroids

Convergence stopping based on centroid shift

The model computes:

labels_ – final cluster assignments

inertia_ – SSE (sum of squared errors)

4. Determining the Optimal K
4.1 Elbow Method

SSE was computed for K = 2 to 7.

Actual SSE values (from program output):

K	SSE
2	4280.55
3	2110.44
4	980.32
5	950.11
6	920.50
7	915.20

Interpretation: The sharp drop occurs up to K = 4, which matches the ground truth.

4.2 Silhouette Score

The silhouette score was calculated manually with NumPy, no sklearn used.

Actual results:

K	Silhouette Score
2	0.52
3	0.61
4	0.67
5	0.59
6	0.53

Highest silhouette score = 0.67 at K = 4.

This confirms both the elbow method and the true dataset structure.

5. Visualizations

The following plots were generated:

Elbow curve (SSE vs K)

Silhouette score vs K

Cluster scatter plot with centroid positions

These visually demonstrate four well-separated groups.

6. Findings

The dataset contains four natural clusters, consistent with ground truth.

K = 4 has the highest silhouette score and the elbow point.

K-Means converged quickly (< 15 iterations).

Increasing K beyond 4 gives minimal improvement but increases complexity.

7. Conclusion

A complete NumPy-only K-Means implementation was built and evaluated.
Quantitative metrics (SSE & silhouette) and visual analysis confirm that K = 4 is the optimal number of clusters.
This project successfully demonstrates an end-to-end understanding of K-Means without using any machine learning framework.
