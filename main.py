from mnist_loader import MNISTLoader
from kmeans import KMeans, KMeansCosine
from evaluation import calculate_clustering_accuracy, calculate_sse
from sklearn.decomposition import PCA
import time
import numpy as np
import datetime

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE

MAX_ITER = 300

def main(reduce=False):
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Load MNIST dataset
    mnist_loader = MNISTLoader()
    train_images, train_labels = mnist_loader.load_dataset()
    
    # Flatten the images
    train_images_flattened = train_images.reshape(train_images.shape[0], -1)

    if reduce:
        pca = PCA(n_components=2)
        train_images_flattened = pca.fit_transform(train_images_flattened)
    
    # Print statements for Question 2
    with open(f'experiments/q2-{current_time}.txt', 'w') as q1_file:
        print("MNIST dataset loaded.")
        
        # Perform clustering with KMeans using Euclidean distance
        kmeans = KMeans(n_clusters=4, max_iter=MAX_ITER )
        print("Performing clustering with Euclidean distance...")
        cluster_labels_euclidean , cluser_centeroids_euclidian = kmeans.fit(train_images_flattened)
        print("Clustering complete.")
    
        # Evaluate clustering results using clustering accuracy and SSE
        print("Evaluating clustering results...")
        clustering_accuracy_euclidean = calculate_clustering_accuracy(cluster_labels_euclidean, train_labels)
        sse_euclidean = calculate_sse(train_images_flattened, cluster_labels_euclidean, kmeans.centroids)
        
        print("Clustering accuracy using Euclidean distance:", clustering_accuracy_euclidean, file=q1_file)
        print("Sum of Squared Errors (SSE) using Euclidean distance:", sse_euclidean, file=q1_file)
        
        # Visualize the clusters
        kmeans.visualize_clusters(reduce)

        #print("Visualizing clusters with t-SNE...")
        #kmeans.visualize_clusters_with_tsne(train_images)


    
    
    # Print statements for Question 3
    with open(f'experiments/q3-{current_time}.txt', 'w') as q2_file:
        # Perform clustering with KMeans using cosine similarity
        kmeans_cosine = KMeansCosine(n_clusters=4, max_iter=MAX_ITER)
        print("Performing clustering with cosine similarity...")
        cluster_labels_cosine, cluster_centeroids_cosine = kmeans_cosine.fit(train_images_flattened)
        print("Clustering complete.")

        # Evaluate clustering results using clustering accuracy and SSE
        print("Evaluating clustering results...")
        clustering_accuracy_cosine = calculate_clustering_accuracy(cluster_labels_cosine, train_labels)
        sse_cosine = calculate_sse(train_images_flattened, cluster_labels_cosine, kmeans_cosine.centroids)
        
        print("Clustering accuracy using cosine similarity:", clustering_accuracy_cosine, file=q2_file)
        print("Sum of Squared Errors (SSE) using cosine similarity:", sse_cosine, file=q2_file)
        
        # Visualize the clusters
        kmeans_cosine.visualize_clusters()
      
    

if __name__ == "__main__":
    main()
