from mnist_loader import MNISTLoader
from kmeans import KMeans, KMeansCosine
from evaluation import calculate_clustering_accuracy, calculate_sse
import time
import numpy as np
import datetime

MAX_ITER = 50

def main():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Load MNIST dataset
    mnist_loader = MNISTLoader()
    train_images, train_labels, test_images, test_labels = mnist_loader.load_dataset()
    
    # Flatten the images
    train_images_flattened = train_images.reshape(train_images.shape[0], -1)
    test_images_flattened = test_images.reshape(test_images.shape[0], -1)
    
    # Print statements for Question 1
    with open(f'q1-{current_time}.txt', 'w') as q1_file:
        print("MNIST dataset loaded.", file=q1_file)
    
        # Perform clustering with KMeans using Euclidean distance
        kmeans = KMeans(n_clusters=4, max_iter=MAX_ITER )
        print("Performing clustering with Euclidean distance...", file=q1_file)
        cluster_labels_euclidean , cluser_centeroids_euclidian = kmeans.fit(train_images_flattened)
        print("Clustering complete.", file=q1_file)
    
        # Evaluate clustering results using clustering accuracy and SSE
        print("Evaluating clustering results...", file=q1_file)
        clustering_accuracy_euclidean = calculate_clustering_accuracy(cluster_labels_euclidean, train_labels)
        sse_euclidean = calculate_sse(train_images_flattened, cluster_labels_euclidean, kmeans.centroids)
        
        print("Clustering accuracy using Euclidean distance:", clustering_accuracy_euclidean, file=q1_file)
        print("Sum of Squared Errors (SSE) using Euclidean distance:", sse_euclidean, file=q1_file)
        
        # Visualize the clusters
        kmeans.visualize_clusters()
    
    

    # Print statements for Question 2
    with open(f'q2-{current_time}.txt', 'w') as q2_file:
        # Perform clustering with KMeans using cosine similarity
        kmeans_cosine = KMeansCosine(n_clusters=4, max_iter=MAX_ITER)
        print("Performing clustering with cosine similarity...", file=q2_file)
        cluster_labels_cosine = kmeans_cosine.fit(train_images_flattened)
        print("Clustering complete.", file=q2_file)

        # Evaluate clustering results using clustering accuracy and SSE
        print("Evaluating clustering results...", file=q2_file)
        clustering_accuracy_cosine = calculate_clustering_accuracy(cluster_labels_cosine, train_labels)
        sse_cosine = calculate_sse(train_images_flattened, cluster_labels_cosine, kmeans_cosine.centroids)
        
        print("Clustering accuracy using cosine similarity:", clustering_accuracy_cosine, file=q2_file)
        print("Sum of Squared Errors (SSE) using cosine similarity:", sse_cosine, file=q2_file)
        
        # Visualize the clusters
        kmeans_cosine.visualize_clusters()
    

if __name__ == "__main__":
    main()
