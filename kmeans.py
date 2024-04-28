import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import datetime

class KMeans:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.X = X
        n_samples, n_features = X.shape
        
        # Step 1: Initialize cluster centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        i = 0
        for _ in range(self.max_iter):
            i += 1
            # Step 2: Assign each data point to the nearest cluster centroid
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            labels = np.argmin(distances, axis=0)
            
            # Step 3: Update cluster centroids based on the mean of the assigned data points
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                print(f"Converged after {i} iterations")
                break
            
            self.centroids = new_centroids
        self.labels = labels
        return labels, self.centroids
    
    def visualize_clusters(self):
        # Perform PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)

        # Plot data points colored by cluster assignments
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.labels, cmap='viridis', s=10)

        # Plot centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=100)

        plt.title('Clusters using Euclidian Distance')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.colorbar(label='Cluster Label')
        
        # Save the figure with the current time as the filename
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f'clusters_{current_time}.png')  # Save the figure with the current time
    

class KMeansCosine:
    def __init__(self, n_clusters, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        self.X = X
        n_samples, n_features = X.shape
        
        # Step 1: Initialize cluster centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        i = 0
        for _ in range(self.max_iter):
            i+=1
            # Step 2: Assign each data point to the nearest cluster centroid
            similarities = X.dot(self.centroids.T) / (np.linalg.norm(X, axis=1, keepdims=True) * np.linalg.norm(self.centroids, axis=1))
            labels = np.argmax(similarities, axis=1)
            
            # Step 3: Update cluster centroids based on the mean of the assigned data points
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                print(f"Converged after {i} iterations")
                break
            
            self.centroids = new_centroids
        self.labels = labels
        return labels, self.centroids
    

    def visualize_clusters(self):
        # Perform PCA for dimensionality reduction
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X)

        # Plot data points colored by cluster assignments
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.labels, cmap='viridis', s=10)

        # Plot centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=100)

        plt.title('Clusters using Cosine Similarity')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        
        # Save the figure with the current time as the filename
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f'clusters_{current_time}.png')  # Save the figure with the current time
        