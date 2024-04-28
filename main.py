from mnist_loader import MNISTLoader
from kmeans import KMeans, KMeansCosine
from evaluation import calculate_clustering_accuracy, calculate_sse
import time

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
    

    # Perform clustering with KMeansCosine using cosine similarity
    print("Performing clustering with cosine similarity...")
    kmeans_cosine = KMeansCosine(n_clusters=10)
    cluster_labels_cosine = kmeans_cosine.fit(train_images_flattened)
    
    # Evaluate clustering results using clustering accuracy and SSE
    clustering_accuracy_cosine = calculate_clustering_accuracy(cluster_labels_cosine, train_labels)
    sse_cosine = calculate_sse(train_images_flattened, cluster_labels_cosine, kmeans_cosine.centroids)
    
    print("Clustering accuracy using cosine similarity:", clustering_accuracy_cosine)
    print("Sum of Squared Errors (SSE) using cosine similarity:", sse_cosine)

    end_time = time.time()  # Record end time
    execution_time = end_time - start_time  # Calculate execution time
    print("Total execution time: {:.2f} seconds".format(execution_time))

if __name__ == "__main__":
    main()
