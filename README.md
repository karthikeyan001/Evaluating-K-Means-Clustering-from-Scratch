1. Introduction

This project focuses on implementing the K-Means clustering algorithm manually using NumPy. The aim is to understand how the algorithm works internally and to analyze its behavior by running it on a synthetic two-dimensional dataset. The study includes dataset creation, clustering, evaluation of different values of K, and interpretation of results without relying on high-level machine learning libraries.

2. Dataset Generation

A synthetic dataset was generated to allow controlled experimentation. Three Gaussian groups were created with the following characteristics:

Total samples: 300

Features: 2 (x and y)

True underlying clusters: 3

Standard deviation: 1.0

The dataset was stored in a CSV file with two columns:

x — Feature 1

y — Feature 2

These values do not represent real-world metrics. They are simply two-dimensional coordinates used to visualize clustering performance.

3. Methodology
3.1 Data Preparation

The dataset was loaded using pandas, checked for missing values, and converted into a NumPy array. Standardization was not required because all features share a similar scale.

3.2 Manual K-Means Implementation

The K-Means algorithm was implemented entirely using NumPy. The main components include:

Random selection of K initial centroids.

Distance computation using Euclidean distance.

Assignment of each data point to the nearest centroid.

Recalculation of centroids based on cluster membership.

Iteration until convergence or until a maximum number of iterations is reached.

The implementation avoids any machine learning library and performs all operations using vectorized NumPy expressions.

4. Determining the Optimal Number of Clusters

Two evaluation techniques were used to analyze the choice of K.

4.1 Elbow Method

The sum of squared errors (SSE) was calculated for K values from 2 to 7. The SSE decreases sharply up to K = 3. After that point, the improvement becomes small. This indicates that K = 3 is a natural balance between model complexity and accuracy.

4.2 Silhouette Analysis

The silhouette score was computed manually using only NumPy. The formula compares:

The average distance between a point and others in its own cluster.

The lowest average distance between the point and any other cluster.

Scores were highest at K = 3, confirming the elbow method. Higher values of K caused unnecessary splitting, while lower values produced overly broad clusters.

5. Visualizations

Scatter plots were generated to display cluster assignments and centroid positions. Additional plots were produced to show SSE values and silhouette scores across different K values. The visualizations clearly show well-separated cluster regions and validate the analytical findings.

6. Findings
The dataset naturally forms three compact groups.
Both evaluation methods support selecting K = 3.
The algorithm converges in a small number of iterations due to the simplicity of the data.
Increasing K increases computational cost without improving cluster quality.
Overall, the experiment demonstrates the full K-Means workflow and confirms that careful selection of K is essential for producing meaningful clusters.
7. Conclusion

A complete NumPy-only implementation of the K-Means algorithm was developed and tested on a synthetic dataset. The project includes dataset creation, clustering, evaluation of cluster quality, and visualization. The analysis consistently indicates that three clusters provide the best fit. This study reinforces fundamental concepts in unsupervised learning and demonstrates the internal mechanics of K-Means without relying on external machine learning libraries.

