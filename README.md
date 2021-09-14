# K-means clustering
In a machine learning interview setting, you might be asked how the output from K-means clustering might be used to assess its performance as the best algorithm.

In this exercise you'll practice K-means clustering. Using the `.inertia_` attribute to compare models with different numbers of clusters, `k`, you'll then also use this information to assess cluster number in the next exercise.

Recall that the target variable in the `diabetes` dataset is `progression`.
``` python
# Import module
from sklearn.cluster import KMeans

# Create feature matrix
X = diabetes.drop("progression", axis=1)

# Instantiate
kmeans = KMeans(n_clusters=20, random_state=123)

# Fit
fit = kmeans.fit(X)

# Print inertia
print("Sum of squared distances for 20 clusters is", fit.inertia_)
```

# Hierarchical agglomerative clustering
In the last exercise, you saw how the number of clusters while performing K-means clustering could impact your results allowing you to discuss K-means in a machine learning interview. However, another clustering model you can use is hierarchical agglomerative clustering. In Python, you could derive the optimal number of clusters for this technique both visually and mathematically. You will the `scipy` and `sklearn` modules to do both.

Recall that selecting the optimal number of clusters from a dendrogram depends on both the linkage criteria and distance threshold. Here, you'll create a dendrogram with the `X` matrix from `diabetes`, then extend an imaginary line at length 1.50, counting the number of vertical lines crossed to represent the optimal number of clusters for your hierarchical clustering algorithm going forward.
