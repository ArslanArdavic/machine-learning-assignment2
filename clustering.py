import os
import urllib.request
import gzip
import numpy as np
import matplotlib.pyplot as plt
from   sklearn.decomposition import PCA
import numpy as np
import datetime
import time
import copy

class MNISTLoader:
    def __init__(self, save_path='./mnist_data'):
        self.save_path = save_path
        self.url_base = 'http://yann.lecun.com/exdb/mnist/'
        self.key_file = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }
        self.classes = [2, 3, 8, 9]

    def download_mnist(self):
        os.makedirs(self.save_path, exist_ok=True)
        for v in self.key_file.values():
            filename = os.path.join(self.save_path, v)
            if not os.path.exists(filename):
                urllib.request.urlretrieve(self.url_base + v, filename)

    def load_mnist_images(self, filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        return data.reshape(-1, 28, 28)
    
    def load_mnist_labels(self, filename):
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data

    def load_dataset(self):
        self.download_mnist()
        train_images = self.load_mnist_images(os.path.join(self.save_path, self.key_file['train_images']))
        train_labels = self.load_mnist_labels(os.path.join(self.save_path, self.key_file['train_labels']))
        test_images = self.load_mnist_images(os.path.join(self.save_path, self.key_file['test_images']))
        test_labels = self.load_mnist_labels(os.path.join(self.save_path, self.key_file['test_labels']))
        
        # Filter train and test datasets based on the specified classes
        train_mask = np.isin(train_labels, self.classes)
        test_mask = np.isin(test_labels, self.classes)
        
        filtered_train_images = train_images[train_mask]
        filtered_train_labels = train_labels[train_mask]
        filtered_test_images = test_images[test_mask]
        filtered_test_labels = test_labels[test_mask]
        
        # Merge train and test datasets
        merged_images = np.concatenate((filtered_train_images, filtered_test_images), axis=0)
        merged_labels = np.concatenate((filtered_train_labels, filtered_test_labels), axis=0)
        
        return merged_images, merged_labels
    
class KMeans:
    def __init__(self, experiment_time, n_clusters, max_iter, distance, reduce_with_pca):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.distance = distance
        self.reduce_with_pca = reduce_with_pca

    def fit(self, X):
        self.X = X
        n_samples, n_features = X.shape
        
        # Step 1: Initialize cluster centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        i = 0

        if self.distance == 'euclidean':
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
        elif self.distance == 'cosine':
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
        X_pca = self.X
        if not self.reduce_with_pca:
            # Perform PCA for dimensionality reduction
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(self.X)

        # Plot data points colored by cluster assignments
        plt.scatter(X_pca[:, 0], X_pca[:, 1], c=self.labels, cmap='viridis', s=10)

        # Plot centroids
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red', marker='x', s=100)
        if self.distance == 'euclidean':
            plt.title('Clusters using Euclidian Distance')
        elif self.distance == 'cosine':
            plt.title('Clusters using Cosine Similarity')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        
        plt.savefig(f'experiments/{experiment_time}/{self.distance}{"-reduced" if self.reduce_with_pca else ""}.png')  # Save the figure with the current time

class Experiment:
    def __init__(self, experiment_time, images, labels, n_clusters, max_iter=300, distance='euclidean', reduce_with_pca=False):
        self.kmeans = KMeans(experiment_time, n_clusters, max_iter, distance, reduce_with_pca)
        self.images_flattened = images.reshape(images.shape[0], -1)
        if reduce_with_pca:
            pca = PCA(n_components=2)
            self.images_flattened = pca.fit_transform(self.images_flattened)
        self.labels = labels
        self.distance = distance
        self.reduced = reduce_with_pca
        self.experiment_time = experiment_time
        self.run()

    def calculate_clustering_accuracy(self):
        # Initialize a dictionary to map cluster labels to true labels
        label_map = {}
        for cluster in np.unique(self.cluster_labels):
            # Find the most common true label in the cluster
            cluster_members = self.labels[self.cluster_labels == cluster]
            true_label = np.argmax(np.bincount(cluster_members))
            label_map[cluster] = true_label
        
        # Map cluster labels to true labels
        mapped_labels = [label_map[cluster] for cluster in self.cluster_labels]
        
        # Calculate clustering accuracy
        correct_predictions = np.sum(mapped_labels == self.labels)
        total_samples = len(self.labels)
        acc = correct_predictions / total_samples
        return acc

    def calculate_sse(self):
        sse = 0
        for i, centroid in enumerate(self.cluster_centeroids):
            # Calculate squared Euclidean distance between data points and centroid
            sse += np.sum((self.images_flattened[self.cluster_labels == i] - centroid) ** 2)
        return sse
    
    def run(self): 
        with open(f'experiments/{self.experiment_time}/{self.distance}{"-reduced" if self.reduced else ""}.txt', 'w') as out_file:
            # Perform clustering with KMeans
            print(f"Performing clustering with distance={self.distance}...")
            start_time = time.time()
            self.cluster_labels, self.cluster_centeroids = self.kmeans.fit(self.images_flattened)
            end_time = time.time()
            print("Clustering complete.")
            time_passed = end_time - start_time

            # Evaluate clustering results using clustering accuracy and SSE
            print("Evaluating clustering results...\n")
            
            print("Clustering accuracy using Euclidean distance:", self.calculate_clustering_accuracy(), file=out_file)
            print("Sum of Squared Errors (SSE) using Euclidean distance:", self.calculate_sse(), file=out_file)
            print(f"Time taken: {time_passed}", file=out_file)

        out_file.close()
        self.kmeans.visualize_clusters()

if __name__ == "__main__":
    mnist_loader = MNISTLoader()
    images, labels = mnist_loader.load_dataset()
    experiment_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(f'./experiments/{experiment_time}', exist_ok=True)
    euclidian_all_features  = Experiment(experiment_time=experiment_time, images=copy.deepcopy(images), labels=copy.deepcopy(labels), n_clusters=4, distance='euclidean', reduce_with_pca=False)
    cosine_all_features     = Experiment(experiment_time=experiment_time, images=copy.deepcopy(images), labels=copy.deepcopy(labels), n_clusters=4, distance='cosine', reduce_with_pca=False)
    euclidian_pca           = Experiment(experiment_time=experiment_time, images=copy.deepcopy(images), labels=copy.deepcopy(labels), n_clusters=4, distance='euclidean', reduce_with_pca=True)
    cosine_pca              = Experiment(experiment_time=experiment_time, images=copy.deepcopy(images), labels=copy.deepcopy(labels), n_clusters=4, distance='cosine', reduce_with_pca=True)
    print("All experiments completed.")