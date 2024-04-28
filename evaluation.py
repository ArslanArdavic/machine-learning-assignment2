# clustering_accuracy.pyx

import numpy as np
cimport numpy as np

cpdef double calculate_clustering_accuracy(np.ndarray[np.int64_t, ndim=1] cluster_labels,
                                          np.ndarray[np.int64_t, ndim=1] true_labels):
    cdef int n_samples = cluster_labels.shape[0]
    cdef int n_clusters = len(np.unique(cluster_labels))
    cdef np.ndarray[np.int64_t, ndim=1] label_map = np.zeros(n_clusters, dtype=np.int64)
    cdef np.ndarray[np.int64_t, ndim=1] mapped_labels = np.zeros(n_samples, dtype=np.int64)
    
    for i in range(n_clusters):
        cluster = np.where(cluster_labels == i)[0]
        cluster_members = true_labels[cluster]
        label_map[i] = np.argmax(np.bincount(cluster_members))
    
    for i in range(n_samples):
        mapped_labels[i] = label_map[cluster_labels[i]]
    
    cdef int correct_predictions = np.sum(mapped_labels == true_labels)
    cdef int total_samples = n_samples
    cdef double acc = correct_predictions / total_samples
    return acc

cpdef double calculate_sse(np.ndarray[np.float64_t, ndim=2] X,
                            np.ndarray[np.int64_t, ndim=1] cluster_labels,
                            np.ndarray[np.float64_t, ndim=2] centroids):
    cdef int n_clusters = centroids.shape[0]
    cdef int n_features = X.shape[1]
    cdef double sse = 0.0
    
    for i in range(n_clusters):
        cluster_indices = np.where(cluster_labels == i)[0]
        cluster_data = X[cluster_indices, :]
        cluster_centroid = centroids[i, :]
        for j in range(cluster_data.shape[0]):
            sse += np.sum((cluster_data[j, :] - cluster_centroid) ** 2)
    
    return sse
