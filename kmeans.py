import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Step 1: Initialize cluster centroids randomly
        print("Step 1: Initialize cluster centroids randomly")
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            print("Iteration")
            # Step 2: Assign each data point to the nearest cluster centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Step 3: Update cluster centroids based on the mean of the assigned data points
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print("Converged")
                break
            
            self.centroids = new_centroids
        print("Returning labels")
        return labels

class KMeansCosine:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        n_samples, n_features = X.shape
        
        # Step 1: Initialize cluster centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            # Step 2: Assign each data point to the nearest cluster centroid
            similarities = X.dot(self.centroids.T) / (np.linalg.norm(X, axis=1, keepdims=True) * np.linalg.norm(self.centroids, axis=1))
            labels = np.argmax(similarities, axis=1)
            
            # Step 3: Update cluster centroids based on the mean of the assigned data points
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break
            
            self.centroids = new_centroids
        
        return labels
