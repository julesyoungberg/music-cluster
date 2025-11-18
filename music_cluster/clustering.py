"""Clustering engine for music-cluster."""

import json
import math
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False


# Granularity multipliers
GRANULARITY_MULTIPLIERS = {
    "fewer": 0.5,
    "less": 0.75,
    "normal": 1.0,
    "more": 1.5,
    "finer": 2.0
}


class ClusterEngine:
    """Multi-algorithm clustering engine with auto-detection."""
    
    def __init__(self, min_clusters: int = 5, max_clusters: int = 100,
                 detection_method: str = "silhouette", algorithm: str = "kmeans"):
        """Initialize clustering engine.
        
        Args:
            min_clusters: Minimum number of clusters to test
            max_clusters: Maximum number of clusters to test
            detection_method: Method for auto-detection (silhouette, elbow, calinski)
            algorithm: Clustering algorithm (kmeans, hdbscan, hierarchical, spectral)
        """
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        self.detection_method = detection_method
        self.algorithm = algorithm
    
    def find_optimal_k(self, features: np.ndarray, show_progress: bool = True) -> Tuple[int, Dict]:
        """Find optimal number of clusters using the specified method.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            show_progress: Whether to show progress bar
            
        Returns:
            Tuple of (optimal_k, metrics_dict)
        """
        n_samples = features.shape[0]
        
        # Determine search range based on dataset size
        min_k = max(self.min_clusters, 2)
        max_k = min(self.max_clusters, n_samples // 2, math.floor(math.sqrt(n_samples)))
        
        if max_k < min_k:
            max_k = min_k
        
        # Test different k values
        k_range = range(min_k, max_k + 1)
        scores = []
        inertias = []
        
        iterator = tqdm(k_range, desc="Finding optimal k") if show_progress else k_range
        
        for k in iterator:
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10,
                          max_iter=300, random_state=42)
            labels = kmeans.fit_predict(features)
            
            # Compute metrics
            if k < n_samples:
                if self.detection_method == "silhouette":
                    score = silhouette_score(features, labels)
                elif self.detection_method == "calinski":
                    score = calinski_harabasz_score(features, labels)
                else:  # elbow
                    score = -kmeans.inertia_  # Negative so higher is better
                
                scores.append(score)
                inertias.append(kmeans.inertia_)
        
        # Find optimal k
        if not scores:
            optimal_k = min_k
        else:
            if self.detection_method == "elbow":
                # Find elbow point using rate of change
                optimal_k = self._find_elbow_point(list(k_range), inertias)
            else:
                # Find k with highest score
                optimal_idx = np.argmax(scores)
                optimal_k = list(k_range)[optimal_idx]
        
        # Collect metrics
        metrics = {
            "k_range": list(k_range),
            "scores": scores,
            "inertias": inertias,
            "optimal_k": optimal_k,
            "method": self.detection_method
        }
        
        return optimal_k, metrics
    
    def _find_elbow_point(self, k_values: List[int], inertias: List[float]) -> int:
        """Find elbow point in inertia curve.
        
        Args:
            k_values: List of k values tested
            inertias: List of inertia values
            
        Returns:
            K value at elbow point
        """
        if len(inertias) < 3:
            return k_values[0]
        
        # Compute rate of change
        rates = []
        for i in range(1, len(inertias)):
            rate = (inertias[i-1] - inertias[i]) / inertias[i-1] if inertias[i-1] != 0 else 0
            rates.append(rate)
        
        # Find point where rate of decrease slows significantly
        if len(rates) < 2:
            return k_values[0]
        
        # Find maximum rate change
        rate_changes = []
        for i in range(1, len(rates)):
            change = abs(rates[i] - rates[i-1])
            rate_changes.append(change)
        
        if rate_changes:
            elbow_idx = np.argmax(rate_changes) + 1
            return k_values[min(elbow_idx, len(k_values) - 1)]
        
        return k_values[len(k_values) // 2]
    
    def apply_granularity(self, optimal_k: int, granularity: str) -> int:
        """Apply granularity multiplier to optimal k.
        
        Args:
            optimal_k: Optimal number of clusters
            granularity: Granularity level (fewer/less/normal/more/finer)
            
        Returns:
            Adjusted k value
        """
        if granularity not in GRANULARITY_MULTIPLIERS:
            return optimal_k
        
        multiplier = GRANULARITY_MULTIPLIERS[granularity]
        adjusted_k = int(round(optimal_k * multiplier))
        
        # Ensure k is at least 2
        return max(2, adjusted_k)
    
    def cluster(self, features: np.ndarray, n_clusters: int,
                show_progress: bool = True) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Perform K-means clustering.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            n_clusters: Number of clusters
            show_progress: Whether to show progress
            
        Returns:
            Tuple of (labels, centroids, metrics)
        """
        if show_progress:
            print(f"Running K-means with {n_clusters} clusters...")
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10,
                       max_iter=300, random_state=42)
        labels = kmeans.fit_predict(features)
        centroids = kmeans.cluster_centers_
        
        # Compute quality metrics
        metrics = self._compute_clustering_metrics(features, labels, kmeans.inertia_)
        
        return labels, centroids, metrics
    
    def _compute_clustering_metrics(self, features: np.ndarray, labels: np.ndarray,
                                   inertia: float) -> Dict:
        """Compute clustering quality metrics.
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            inertia: K-means inertia
            
        Returns:
            Dictionary of metrics
        """
        metrics = {
            "inertia": inertia,
            "n_clusters": len(np.unique(labels))
        }
        
        # Only compute if we have enough samples
        if len(np.unique(labels)) > 1 and len(features) > len(np.unique(labels)):
            try:
                metrics["silhouette_score"] = silhouette_score(features, labels)
            except:
                metrics["silhouette_score"] = None
            
            try:
                metrics["davies_bouldin_score"] = davies_bouldin_score(features, labels)
            except:
                metrics["davies_bouldin_score"] = None
            
            try:
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(features, labels)
            except:
                metrics["calinski_harabasz_score"] = None
        
        # Cluster size distribution
        unique, counts = np.unique(labels, return_counts=True)
        metrics["cluster_sizes"] = dict(zip([int(x) for x in unique], 
                                           [int(x) for x in counts]))
        metrics["min_cluster_size"] = int(np.min(counts))
        metrics["max_cluster_size"] = int(np.max(counts))
        metrics["mean_cluster_size"] = float(np.mean(counts))
        
        return metrics
    
    def find_representative_tracks(self, features: np.ndarray, labels: np.ndarray,
                                  centroids: np.ndarray, track_ids: List[int]) -> Dict[int, int]:
        """Find representative track for each cluster.
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            centroids: Cluster centroids
            track_ids: List of track IDs corresponding to features
            
        Returns:
            Dictionary mapping cluster_index -> track_id
        """
        representatives = {}
        
        for cluster_idx in range(len(centroids)):
            # Find all tracks in this cluster
            cluster_mask = labels == cluster_idx
            cluster_features = features[cluster_mask]
            cluster_track_ids = [track_ids[i] for i in range(len(track_ids)) if cluster_mask[i]]
            
            if len(cluster_features) == 0:
                continue
            
            # Find track closest to centroid
            centroid = centroids[cluster_idx]
            distances = np.linalg.norm(cluster_features - centroid, axis=1)
            min_idx = np.argmin(distances)
            
            representatives[cluster_idx] = cluster_track_ids[min_idx]
        
        return representatives
    
    def compute_distances_to_centroids(self, features: np.ndarray, labels: np.ndarray,
                                      centroids: np.ndarray) -> np.ndarray:
        """Compute distance of each point to its cluster centroid.
        
        Args:
            features: Feature matrix
            labels: Cluster labels
            centroids: Cluster centroids
            
        Returns:
            Array of distances
        """
        distances = np.zeros(len(features))
        
        for i, (feature, label) in enumerate(zip(features, labels)):
            centroid = centroids[label]
            distances[i] = np.linalg.norm(feature - centroid)
        
        return distances
