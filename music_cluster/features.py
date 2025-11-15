"""Feature vector utilities for music-cluster."""

import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple


class FeatureNormalizer:
    """Normalizer for feature vectors using StandardScaler."""
    
    def __init__(self):
        """Initialize feature normalizer."""
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, features: np.ndarray) -> None:
        """Fit the normalizer on feature data.
        
        Args:
            features: Feature matrix (n_samples, n_features)
        """
        self.scaler.fit(features)
        self.is_fitted = True
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features using fitted normalizer.
        
        Args:
            features: Feature matrix or vector
            
        Returns:
            Normalized features
        """
        if not self.is_fitted:
            raise ValueError("Normalizer must be fitted before transform")
        
        # Handle single feature vector
        if features.ndim == 1:
            features = features.reshape(1, -1)
            return self.scaler.transform(features)[0]
        
        return self.scaler.transform(features)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            features: Feature matrix
            
        Returns:
            Normalized features
        """
        self.fit(features)
        return self.transform(features)


def aggregate_features(frame_features: np.ndarray) -> np.ndarray:
    """Aggregate frame-level features into track-level features.
    
    Computes mean and standard deviation across frames.
    
    Args:
        frame_features: Array of shape (n_frames, n_features)
        
    Returns:
        Aggregated feature vector [means, stds]
    """
    if len(frame_features) == 0:
        raise ValueError("No frame features provided")
    
    # Compute statistics across frames
    means = np.mean(frame_features, axis=0)
    stds = np.std(frame_features, axis=0)
    
    # Concatenate mean and std
    aggregated = np.concatenate([means, stds])
    
    # Replace NaN/Inf with 0
    aggregated = np.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0)
    
    return aggregated


def compute_feature_statistics(features: np.ndarray) -> dict:
    """Compute statistics about a feature matrix.
    
    Args:
        features: Feature matrix (n_samples, n_features)
        
    Returns:
        Dictionary with statistics
    """
    return {
        "n_samples": features.shape[0],
        "n_features": features.shape[1] if features.ndim > 1 else 1,
        "mean": np.mean(features),
        "std": np.std(features),
        "min": np.min(features),
        "max": np.max(features),
        "has_nan": np.any(np.isnan(features)),
        "has_inf": np.any(np.isinf(features))
    }


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (0-1)
    """
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Compute Euclidean distance between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Euclidean distance
    """
    return np.linalg.norm(vec1 - vec2)
