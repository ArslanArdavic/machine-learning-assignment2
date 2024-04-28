import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import datetime

from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

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

    def reduce_dimensionality_with_tsne(self, train_images, n_components=3, random_state=42):
        print("Reducing dimensionality with t-SNE...")
        tsne = TSNE(n_components=n_components, random_state=random_state)
        return tsne.fit_transform(train_images.reshape(train_images.shape[0], -1))
    
    def visualize_clusters_with_tsne(self, train_images):
        # Reduce dimensionality with t-SNE
        train_images_tsne = self.reduce_dimensionality_with_tsne(train_images)

        # Visualize in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        print("HERE3 ")

        # Scatter plot for each cluster
        for cluster_label in np.unique(self.labels):
            indices = np.where(self.labels == cluster_label)[0]
            ax.scatter(train_images_tsne[indices, 0], train_images_tsne[indices, 1], train_images_tsne[indices, 2],
                        label=str(cluster_label), s=10)
        print("HERE1 ")
        # Add centroids
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2],
                    s=100, c='red', marker='x', label='Centroids')

        ax.set_title('Clusters Visualized with t-SNE in 3D')
        ax.legend(title='Cluster Label')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')
        print("HERE2 ")

        # Save the figure with the current time as the filename
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f'clusters_kmeans_{current_time}.png')  # Save the figure with the current time
        plt.show()

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
     
    def reduce_dimensionality_with_tsne(self, train_images, n_components=3, random_state=42):
        tsne = TSNE(n_components=n_components, random_state=random_state)
        return tsne.fit_transform(train_images.reshape(train_images.shape[0], -1))
    def visualize_clusters_with_tsne(self, train_images):
        # Reduce dimensionality with t-SNE
        train_images_tsne = self.reduce_dimensionality_with_tsne(train_images)

        # Visualize in 3D
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Scatter plot for each cluster
        for cluster_label in np.unique(self.labels):
            indices = np.where(self.labels == cluster_label)[0]
            ax.scatter(train_images_tsne[indices, 0], train_images_tsne[indices, 1], train_images_tsne[indices, 2],
                        label=str(cluster_label), s=10)

        # Add centroids
        ax.scatter(self.centroids[:, 0], self.centroids[:, 1], self.centroids[:, 2],
                    s=100, c='red', marker='x', label='Centroids')

        ax.set_title('Clusters Visualized with t-SNE in 3D')
        ax.legend(title='Cluster Label')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')

        # Save the figure with the current time as the filename
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        plt.savefig(f'clusters_kmeans_cosine_{current_time}.png')  # Save the figure with the current time