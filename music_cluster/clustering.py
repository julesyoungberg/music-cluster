"""Unsupervised clustering, used only to *propose* groups.

In the directed workflow the DJ defines the groups. Clustering has one job
left: when someone points at a folder of a few thousand unsorted tracks, break
it into candidate piles so they have something concrete to accept, rename,
merge, or throw away. Nothing here writes a group — that is always a human
decision.
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


try:
    # scikit-learn ships HDBSCAN from 1.3 onwards; the standalone package is
    # only a fallback for older installs.
    from sklearn.cluster import HDBSCAN as _HDBSCAN

    HDBSCAN_AVAILABLE = True
except ImportError:  # pragma: no cover
    try:
        from hdbscan import HDBSCAN as _HDBSCAN

        HDBSCAN_AVAILABLE = True
    except ImportError:
        _HDBSCAN = None
        HDBSCAN_AVAILABLE = False


ALGORITHMS = ("hdbscan", "kmeans", "hierarchical")

# How many candidate piles to aim for, relative to the automatic estimate.
GRANULARITY_MULTIPLIERS = {
    "coarse": 0.5,
    "broad": 0.75,
    "normal": 1.0,
    "fine": 1.5,
    "finest": 2.0,
}

NOISE_LABEL = -1


class ClusterEngine:
    """Wraps the clustering backends behind one call."""

    def __init__(
        self,
        algorithm: str = "hdbscan",
        min_group_size: int = 8,
        max_candidates: int = 30,
        detection_method: str = "silhouette",
    ):
        if algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm {algorithm!r}; choose from {ALGORITHMS}")
        self.algorithm = algorithm
        self.min_group_size = max(2, min_group_size)
        self.max_candidates = max(2, max_candidates)
        self.detection_method = detection_method

    def cluster(
        self,
        features: np.ndarray,
        n_clusters: Optional[int] = None,
        granularity: str = "normal",
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Cluster projected features.

        Returns ``(labels, centroids, metrics)``. Labels of ``-1`` are noise
        (HDBSCAN only) and have no centroid.
        """
        features = np.asarray(features, dtype=float)
        if len(features) < self.min_group_size:
            raise ValueError(
                f"Need at least {self.min_group_size} tracks to look for groups, got {len(features)}"
            )

        if self.algorithm == "hdbscan":
            labels = self._hdbscan(features)
        else:
            if n_clusters is None:
                n_clusters = self.apply_granularity(self.estimate_k(features), granularity)
            n_clusters = int(np.clip(n_clusters, 2, min(self.max_candidates, len(features) // 2)))
            labels = self._partition(features, n_clusters)

        centroids = self._centroids(features, labels)
        return labels, centroids, self._metrics(features, labels)

    def _hdbscan(self, features: np.ndarray) -> np.ndarray:
        if not HDBSCAN_AVAILABLE:
            raise ImportError(
                "HDBSCAN needs scikit-learn 1.3 or newer. Upgrade scikit-learn, or "
                "run discovery with --algorithm kmeans."
            )
        return _HDBSCAN(
            min_cluster_size=self.min_group_size,
            min_samples=max(2, self.min_group_size // 2),
            metric="euclidean",
            cluster_selection_method="eom",
        ).fit_predict(features)

    def _partition(self, features: np.ndarray, n_clusters: int) -> np.ndarray:
        if self.algorithm == "hierarchical":
            return AgglomerativeClustering(n_clusters=n_clusters, linkage="ward").fit_predict(
                features
            )
        return KMeans(
            n_clusters=n_clusters, init="k-means++", n_init=10, random_state=42
        ).fit_predict(features)

    def estimate_k(self, features: np.ndarray) -> int:
        """Pick a cluster count by scoring a small range of candidates."""
        n_samples = len(features)
        min_k = 2
        max_k = int(min(self.max_candidates, max(3, math.floor(math.sqrt(n_samples / 2)))))
        if max_k <= min_k:
            return min_k

        # Sample large libraries: the shape of the score curve is what matters,
        # and scoring every k on 20k tracks is needlessly slow.
        sample = features
        if n_samples > 4000:
            rng = np.random.default_rng(42)
            sample = features[rng.choice(n_samples, 4000, replace=False)]

        best_k, best_score = min_k, -np.inf
        for k in range(min_k, max_k + 1):
            labels = KMeans(n_clusters=k, n_init=5, random_state=42).fit_predict(sample)
            if len(np.unique(labels)) < 2:
                continue
            if self.detection_method == "calinski":
                score = calinski_harabasz_score(sample, labels)
            else:
                score = silhouette_score(sample, labels)
            if score > best_score:
                best_k, best_score = k, score
        return best_k

    @staticmethod
    def apply_granularity(k: int, granularity: str) -> int:
        multiplier = GRANULARITY_MULTIPLIERS.get(granularity, 1.0)
        return max(2, round(k * multiplier))

    @staticmethod
    def _centroids(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
        valid = sorted(int(label) for label in np.unique(labels) if label != NOISE_LABEL)
        if not valid:
            return np.empty((0, features.shape[1]))
        return np.vstack([features[labels == label].mean(axis=0) for label in valid])

    def _metrics(self, features: np.ndarray, labels: np.ndarray) -> Dict:
        valid_mask = labels != NOISE_LABEL
        unique = np.unique(labels[valid_mask])
        metrics: Dict = {
            "n_candidates": len(unique),
            "n_noise": int(np.sum(~valid_mask)),
            "noise_percentage": float(100 * np.sum(~valid_mask) / max(len(labels), 1)),
            "algorithm": self.algorithm,
        }

        if len(unique) > 1 and valid_mask.sum() > len(unique):
            subset, sublabels = features[valid_mask], labels[valid_mask]
            for name, func in (
                ("silhouette_score", silhouette_score),
                ("davies_bouldin_score", davies_bouldin_score),
                ("calinski_harabasz_score", calinski_harabasz_score),
            ):
                try:
                    metrics[name] = float(func(subset, sublabels))
                except Exception:
                    metrics[name] = None

        sizes = {int(label): int(np.sum(labels == label)) for label in unique}
        metrics["candidate_sizes"] = sizes
        return metrics


def select_exemplars(features: np.ndarray, member_indices: List[int], count: int = 5) -> List[int]:
    """Pick tracks that show what a candidate pile contains.

    The medoid comes first (the most typical track), then greedily the members
    furthest from everything already picked. A DJ scanning five tracks learns
    more from the spread of a pile than from its five most average tracks.
    """
    if not member_indices:
        return []
    if len(member_indices) <= count:
        return list(member_indices)

    points = features[member_indices]
    centroid = points.mean(axis=0)
    medoid_local = int(np.argmin(np.linalg.norm(points - centroid, axis=1)))

    chosen = [medoid_local]
    distances = np.linalg.norm(points - points[medoid_local], axis=1)
    while len(chosen) < count:
        nxt = int(np.argmax(distances))
        if nxt in chosen:
            break
        chosen.append(nxt)
        distances = np.minimum(distances, np.linalg.norm(points - points[nxt], axis=1))

    return [member_indices[i] for i in chosen]
