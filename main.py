from mnist_loader import MNISTLoader
from kmeans import KMeans, KMeansCosine
from evaluation import calculate_clustering_accuracy, calculate_sse
import time
import matplotlib.pyplot as plt

def plot_cluster_images(images, cluster_labels, num_clusters):
    # Plot some sample images from each cluster
    fig, axs = plt.subplots(num_clusters, 5, figsize=(10, 10))
    for cluster in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        for i in range(5):
            axs[cluster, i].imshow(images[cluster_indices[i]].reshape(28, 28), cmap='gray')
            axs[cluster, i].axis('off')
            axs[cluster, i].set_title(f'Cluster {cluster}')
    plt.tight_layout()
    plt.show()

def plot_final_clusters(centroids, cluster_labels, images, num_clusters):
    # Plot cluster centroids
    fig, axs = plt.subplots(1, num_clusters, figsize=(10, 5))
    for i in range(num_clusters):
        axs[i].imshow(centroids[i].reshape(28, 28), cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'Cluster {i} Centroid')
    plt.tight_layout()
    plt.show()

    # Plot some representative images from each cluster
    fig, axs = plt.subplots(num_clusters, 5, figsize=(10, 10))
    for cluster in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        for i in range(5):
            axs[cluster, i].imshow(images[cluster_indices[i]].reshape(28, 28), cmap='gray')
            axs[cluster, i].axis('off')
            axs[cluster, i].set_title(f'Cluster {cluster}')
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
    kmeans = KMeans(n_clusters=10)
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
    plot_cluster_images(train_images, cluster_labels_euclidean, 10)

    # Plot final cluster centroids and images
    plot_final_clusters(kmeans.centroids, cluster_labels_euclidean, train_images, 10)

    end_time = time.time()  # Record end time
    execution_time = end_time - start_time  # Calculate execution time
    print("Total execution time: {:.2f} seconds".format(execution_time))

if __name__ == "__main__":
    main()
