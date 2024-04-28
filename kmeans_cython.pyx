# kmeans_cython.pyx

import numpy as np
cimport numpy as np

cdef extern from "math.h":
    double sqrt(double)

cpdef double euclidean_distance(np.ndarray[np.float64_t, ndim=2] X1, np.ndarray[np.float64_t, ndim=2] X2) nogil:
    cdef double dist_sq = 0
    cdef double diff
    cdef int i, j
    for i in range(X1.shape[0]):
        for j in range(X1.shape[1]):
            diff = X1[i, j] - X2[i, j]
            dist_sq += diff * diff
    return sqrt(dist_sq)

cpdef class KMeans:
    cdef int n_clusters
    cdef int max_iter
    cdef np.ndarray[np.float64_t, ndim=2] centroids

    def __init__(self, int n_clusters, int max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, np.ndarray[np.float64_t, ndim=2] X):
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        
        # Step 1: Initialize cluster centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        cdef np.ndarray[np.int64_t, ndim=1] labels
        cdef np.ndarray[np.float64_t, ndim=2] distances
        cdef np.ndarray[np.float64_t, ndim=2] new_centroids
        cdef int i
        
        for _ in range(self.max_iter):
            # Step 2: Assign each data point to the nearest cluster centroid
            distances = np.zeros((n_samples, self.n_clusters))
            for i in range(n_samples):
                for j in range(self.n_clusters):
                    distances[i, j] = euclidean_distance(X[i], self.centroids[j])
            labels = np.argmin(distances, axis=1)
            
            # Step 3: Update cluster centroids based on the mean of the assigned data points
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            
            self.centroids = new_centroids
        
        return labels

cpdef class KMeansCosine:
    cdef int n_clusters
    cdef int max_iter
    cdef np.ndarray[np.float64_t, ndim=2] centroids

    def __init__(self, int n_clusters, int max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, np.ndarray[np.float64_t, ndim=2] X):
        cdef int n_samples = X.shape[0]
        cdef int n_features = X.shape[1]
        
        # Step 1: Initialize cluster centroids randomly
        self.centroids = X[np.random.choice(n_samples, self.n_clusters, replace=False)]
        
        cdef np.ndarray[np.int64_t, ndim=1] labels
        cdef np.ndarray[np.float64_t, ndim=2] similarities
        cdef np.ndarray[np.float64_t, ndim=2] new_centroids
        cdef int i
        
        for _ in range(self.max_iter):
            # Step 2: Assign each data point to the nearest cluster centroid
            similarities = np.dot(X, self.centroids.T) / (np.linalg.norm(X, axis=1, keepdims=True) * np.linalg.norm(self.centroids, axis=1))
            labels = np.argmax(similarities, axis=1)
            
            # Step 3: Update cluster centroids based on the mean of the assigned data points
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Check for convergence
            if np.all(self.centroids == new_centroids):
                break
            
            self.centroids = new_centroids
        
        return labels
