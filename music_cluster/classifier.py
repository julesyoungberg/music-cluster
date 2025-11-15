"""Track classification service for music-cluster."""

import numpy as np
from typing import Dict, List, Optional, Tuple


class TrackClassifier:
    """Classify tracks to nearest cluster."""
    
    def __init__(self, centroids: np.ndarray, cluster_ids: List[int]):
        """Initialize classifier with cluster centroids.
        
        Args:
            centroids: Array of cluster centroids (n_clusters, n_features)
            cluster_ids: List of cluster IDs corresponding to centroids
        """
        self.centroids = centroids
        self.cluster_ids = cluster_ids
    
    def classify(self, feature_vector: np.ndarray,
                threshold: Optional[float] = None) -> Tuple[int, float]:
        """Classify a track to the nearest cluster.
        
        Args:
            feature_vector: Feature vector for the track
            threshold: Optional max distance threshold
            
        Returns:
            Tuple of (cluster_id, distance). Returns (-1, distance) if
            distance exceeds threshold.
        """
        # Compute distances to all centroids
        distances = np.linalg.norm(self.centroids - feature_vector, axis=1)
        
        # Find nearest cluster
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        # Check threshold
        if threshold is not None and min_distance > threshold:
            return -1, min_distance
        
        cluster_id = self.cluster_ids[min_idx]
        return cluster_id, min_distance
    
    def classify_batch(self, feature_matrix: np.ndarray,
                      threshold: Optional[float] = None) -> List[Tuple[int, float]]:
        """Classify multiple tracks.
        
        Args:
            feature_matrix: Feature matrix (n_tracks, n_features)
            threshold: Optional max distance threshold
            
        Returns:
            List of (cluster_id, distance) tuples
        """
        results = []
        
        for feature_vector in feature_matrix:
            result = self.classify(feature_vector, threshold)
            results.append(result)
        
        return results
    
    def get_cluster_distances(self, feature_vector: np.ndarray) -> Dict[int, float]:
        """Get distances to all clusters.
        
        Args:
            feature_vector: Feature vector for the track
            
        Returns:
            Dictionary mapping cluster_id -> distance
        """
        distances = np.linalg.norm(self.centroids - feature_vector, axis=1)
        
        return {
            cluster_id: float(dist)
            for cluster_id, dist in zip(self.cluster_ids, distances)
        }
    
    def get_top_k_clusters(self, feature_vector: np.ndarray,
                          k: int = 3) -> List[Tuple[int, float]]:
        """Get top k nearest clusters.
        
        Args:
            feature_vector: Feature vector for the track
            k: Number of top clusters to return
            
        Returns:
            List of (cluster_id, distance) tuples, sorted by distance
        """
        distances = np.linalg.norm(self.centroids - feature_vector, axis=1)
        
        # Get indices of k smallest distances
        top_k_indices = np.argsort(distances)[:k]
        
        results = [
            (self.cluster_ids[idx], float(distances[idx]))
            for idx in top_k_indices
        ]
        
        return results
