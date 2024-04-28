from mnist_loader import MNISTLoader
from kmeans import KMeans, KMeansCosine
from evaluation import calculate_clustering_accuracy, calculate_sse
import time
import matplotlib.pyplot as plt
import numpy as np


# Plot images to visualize the clustering process
def plot_cluster_images(images, cluster_labels, num_clusters):
    fig, axs = plt.subplots(num_clusters, 5, figsize=(10, 10))
    for cluster in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) > 0:  # Check if cluster has data points
            for i in range(min(5, len(cluster_indices))):
                axs[cluster, i].imshow(images[cluster_indices[i]].reshape(28, 28), cmap='gray')
                axs[cluster, i].axis('off')
                axs[cluster, i].set_title(f'Cluster {cluster}')
        else:
            for i in range(5):
                axs[cluster, i].axis('off')
                axs[cluster, i].set_title(f'Cluster {cluster} (No data)')
    plt.tight_layout()
    plt.show()

# Plot the final clustering results
def plot_final_clusters(centroids, cluster_labels, images, num_clusters):
    fig, axs = plt.subplots(1, num_clusters, figsize=(10, 5))
    for i in range(num_clusters):
        axs[i].imshow(centroids[i].reshape(28, 28), cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'Cluster {i} Centroid')
    plt.tight_layout()
    plt.show()

    fig, axs = plt.subplots(num_clusters, 5, figsize=(10, 10))
    for cluster in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        if len(cluster_indices) > 0:  # Check if cluster has data points
            for i in range(min(5, len(cluster_indices))):
                axs[cluster, i].imshow(images[cluster_indices[i]].reshape(28, 28), cmap='gray')
                axs[cluster, i].axis('off')
                axs[cluster, i].set_title(f'Cluster {cluster}')
        else:
            for i in range(5):
                axs[cluster, i].axis('off')
                axs[cluster, i].set_title(f'Cluster {cluster} (No data)')
    plt.tight_layout()
    plt.show()


def main():
    start_time = time.time()  # Record start time
    
    # Load MNIST dataset
    mnist_loader = MNISTLoader()
    train_images, train_labels, test_images, test_labels = mnist_loader.load_dataset()
    
    # Flatten the images
    train_images_flattened = train_images.reshape(train_images.shape[0], -1)
    test_images_flattened = test_images.reshape(test_images.shape[0], -1)
    
    print("MNIST dataset loaded.")
    
    # Perform clustering with KMeans using Euclidean distance
    kmeans = KMeans(n_clusters=4)
    print("Performing clustering with Euclidean distance...")
    cluster_labels_euclidean = kmeans.fit(train_images_flattened)
    print("Clustering complete.")

    # Evaluate clustering results using clustering accuracy and SSE
    print("Evaluating clustering results...")
    clustering_accuracy_euclidean = calculate_clustering_accuracy(cluster_labels_euclidean, train_labels)
    sse_euclidean = calculate_sse(train_images_flattened, cluster_labels_euclidean, kmeans.centroids)
    
    print("Clustering accuracy using Euclidean distance:", clustering_accuracy_euclidean)
    print("Sum of Squared Errors (SSE) using Euclidean distance:", sse_euclidean)

    # Plot clustering images
    plot_cluster_images(train_images, cluster_labels_euclidean, 4)

    # Plot final cluster centroids and images
    plot_final_clusters(kmeans.centroids, cluster_labels_euclidean, train_images, 4)

    # Perform clustering with KMeans using cosine similarity
    kmeans_cosine = KMeansCosine(n_clusters=4)
    print("Performing clustering with cosine similarity...")
    cluster_labels_cosine = kmeans_cosine.fit(train_images_flattened)
    print("Clustering complete.")

    # Evaluate clustering results using clustering accuracy and SSE
    print("Evaluating clustering results...")
    clustering_accuracy_cosine = calculate_clustering_accuracy(cluster_labels_cosine, train_labels)
    sse_cosine = calculate_sse(train_images_flattened, cluster_labels_cosine, kmeans_cosine.centroids)
    
    print("Clustering accuracy using cosine similarity:", clustering_accuracy_cosine)
    print("Sum of Squared Errors (SSE) using cosine similarity:", sse_cosine)

    end_time = time.time()  # Record end time
    execution_time = end_time - start_time  # Calculate execution time
    print("Total execution time: {:.2f} seconds".format(execution_time))

    # Plot clustering images
    plot_cluster_images(train_images, cluster_labels_cosine, 4)

    # Plot final cluster centroids and images
    plot_final_clusters(kmeans_cosine.centroids, cluster_labels_cosine, train_images, 4)

    

if __name__ == "__main__":
    main()
