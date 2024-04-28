# main.py

from mnist_loader import MNISTLoader
from kmeans import KMeans, KMeansCosine
from evaluation import calculate_clustering_accuracy, calculate_sse

def main():
    # Load MNIST dataset
    mnist_loader = MNISTLoader()
    train_images, train_labels, test_images, test_labels = mnist_loader.load_dataset()
    
    # Flatten the images
    train_images_flattened = train_images.reshape(train_images.shape[0], -1)
    test_images_flattened = test_images.reshape(test_images.shape[0], -1)
    
    # Perform clustering with KMeans using Euclidean distance
    kmeans = KMeans(n_clusters=10)
    cluster_labels_euclidean = kmeans.fit(train_images_flattened)
    
    # Evaluate clustering results using clustering accuracy and SSE
    clustering_accuracy_euclidean = calculate_clustering_accuracy(cluster_labels_euclidean, train_labels)
    sse_euclidean = calculate_sse(train_images_flattened, cluster_labels_euclidean, kmeans.centroids)
    
    print("Clustering accuracy using Euclidean distance:", clustering_accuracy_euclidean)
    print("Sum of Squared Errors (SSE) using Euclidean distance:", sse_euclidean)
    
    # Perform clustering with KMeansCosine using cosine similarity
    kmeans_cosine = KMeansCosine(n_clusters=10)
    cluster_labels_cosine = kmeans_cosine.fit(train_images_flattened)
    
    # Evaluate clustering results using clustering accuracy and SSE
    clustering_accuracy_cosine = calculate_clustering_accuracy(cluster_labels_cosine, train_labels)
    sse_cosine = calculate_sse(train_images_flattened, cluster_labels_cosine, kmeans_cosine.centroids)
    
    print("Clustering accuracy using cosine similarity:", clustering_accuracy_cosine)
    print("Sum of Squared Errors (SSE) using cosine similarity:", sse_cosine)

if __name__ == "__main__":
    main()
