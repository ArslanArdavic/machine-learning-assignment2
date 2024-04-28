import numpy as np
from sklearn.metrics import accuracy_score

def calculate_clustering_accuracy(cluster_labels, true_labels):
    # Initialize a dictionary to map cluster labels to true labels
    label_map = {}
    for cluster in np.unique(cluster_labels):
        # Find the most common true label in the cluster
        true_label = np.argmax(np.bincount(true_labels[cluster_labels == cluster]))
        label_map[cluster] = true_label
    
    # Map cluster labels to true labels
    mapped_labels = [label_map[cluster] for cluster in cluster_labels]
    
    # Calculate clustering accuracy
    acc = accuracy_score(true_labels, mapped_labels)
    return acc

def calculate_sse(X, cluster_labels, centroids):
    sse = 0
    for i, centroid in enumerate(centroids):
        # Calculate squared Euclidean distance between data points and centroid
        sse += np.sum((X[cluster_labels == i] - centroid) ** 2)
    return sse
