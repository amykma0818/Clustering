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
