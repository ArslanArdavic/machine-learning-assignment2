# kmeans_cython.pyx

import numpy as np
cimport numpy as np

cdef class KMeans:
    cdef int n_clusters
    cdef int max_iter
    cdef np.ndarray[np.float64_t, ndim=2] centroids

    def __init__(self, int n_clusters, int max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    cpdef fit(self, np.ndarray[np.float64_t, ndim=2] X):
        # Cython implementation of the k-means algorithm

cdef class KMeansCosine:
    cdef int n_clusters
    cdef int max_iter
    cdef np.ndarray[np.float64_t, ndim=2] centroids

    def __init__(self, int n_clusters, int max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    cpdef fit(self, np.ndarray[np.float64_t, ndim=2] X):
        # Cython implementation of the k-means algorithm

# evaluation_cython.pyx

import numpy as np
cimport numpy as np

cpdef double calculate_clustering_accuracy(np.ndarray[np.int64_t, ndim=1] cluster_labels,
                                          np.ndarray[np.int64_t, ndim=1] true_labels):
    # Cython implementation of clustering accuracy calculation

cpdef double calculate_sse(np.ndarray[np.float64_t, ndim=2] X,
                            np.ndarray[np.int64_t, ndim=1] cluster_labels,
                            np.ndarray[np.float64_t, ndim=2] centroids):
    # Cython implementation of SSE calculation
