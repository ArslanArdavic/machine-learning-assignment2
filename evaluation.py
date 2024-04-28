import numpy as np

def calculate_clustering_accuracy(cluster_labels, true_labels):
    # Initialize a dictionary to map cluster labels to true labels
    label_map = {}
    for cluster in np.unique(cluster_labels):
        # Find the most common true label in the cluster
        cluster_members = true_labels[cluster_labels == cluster]
        true_label = np.argmax(np.bincount(cluster_members))
        label_map[cluster] = true_label
    
    # Map cluster labels to true labels
    mapped_labels = [label_map[cluster] for cluster in cluster_labels]
    
    # Calculate clustering accuracy
    correct_predictions = np.sum(mapped_labels == true_labels)
    total_samples = len(true_labels)
    acc = correct_predictions / total_samples
    return acc

def calculate_sse(X, cluster_labels, centroids):
    sse = 0
    for i, centroid in enumerate(centroids):
        # Calculate squared Euclidean distance between data points and centroid
        sse += np.sum((X[cluster_labels == i] - centroid) ** 2)
    return sse
