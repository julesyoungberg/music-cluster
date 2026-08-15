"""Directed sorting: score a track against the groups a DJ already keeps.

Unlike unsupervised clustering, the groups here are given. A group may be a
900-track genre folder or eight hand-picked seed tracks, so the scoring has to
be fair across wildly different group sizes and spreads:

* distance to a group is a blend of its k nearest references (handles broad,
  multi-modal folders) and its centroid (stable for tiny groups);
* each group's distance is divided by that group's own internal spread, so a
  sprawling folder does not swallow everything and a tight one is not ignored;
* the resulting scores yield a confidence and a top-1/top-2 margin, which is
  what decides whether a track is auto-filed or sent to the human queue.
"""

import pickle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .embedding import EmbeddingSpace


# Outcome of scoring one track.
STATUS_AUTO = "auto"
STATUS_REVIEW = "review"
STATUS_UNMATCHED = "unmatched"

MODEL_FORMAT = 2


@dataclass
class SorterConfig:
    """Tunable knobs for scoring and decisions."""

    projection: str = "auto"
    pca_variance: float = 0.95
    max_components: int = 40
    discriminant_weight: float = 0.7
    neighbors: int = 5
    knn_weight: float = 0.7
    scale_normalization: bool = True
    temperature: float = 0.5
    auto_accept_confidence: float = 0.6
    min_margin: float = 0.08
    novelty_factor: float = 3.0
    feature_weights: Dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SorterConfig":
        data = data or {}
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> Dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass
class GroupProfile:
    """Everything the sorter needs to know about one group."""

    group_id: int
    name: str
    vectors: np.ndarray  # projected reference vectors
    centroid: np.ndarray
    scale: float  # typical internal distance, used to normalise
    size: int


@dataclass
class Suggestion:
    """Ranked suggestion for one track."""

    track_id: Optional[int]
    ranked: List[Dict[str, Any]]
    status: str
    confidence: float
    margin: float
    distance: float

    @property
    def group_id(self) -> Optional[int]:
        return self.ranked[0]["group_id"] if self.ranked else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "suggested_group_id": self.group_id,
            "confidence": self.confidence,
            "margin": self.margin,
            "distance": self.distance,
            "status": self.status,
            "ranked": self.ranked,
        }


class GroupSorter:
    """Fits reference groups and scores new tracks against them."""

    def __init__(self, config: Optional[SorterConfig] = None):
        self.config = config or SorterConfig()
        self.embedding: Optional[EmbeddingSpace] = None
        self.profiles: List[GroupProfile] = []
        self._global_scale: float = 1.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        group_features: Dict[int, np.ndarray],
        group_names: Optional[Dict[int, str]] = None,
    ) -> "GroupSorter":
        """Fit on ``{group_id: (n_tracks, n_features)}``.

        Groups with no reference tracks are skipped; at least two usable groups
        are required for a meaningful decision.
        """
        group_names = group_names or {}
        usable = {
            gid: np.asarray(matrix, dtype=float)
            for gid, matrix in group_features.items()
            if matrix is not None and len(matrix) > 0
        }
        if len(usable) < 2:
            raise ValueError(
                "Need at least two groups with reference tracks to fit a sorter "
                f"(got {len(usable)})"
            )

        matrices = list(usable.values())
        all_features = np.vstack(matrices)
        labels = np.concatenate(
            [np.full(len(matrix), gid) for gid, matrix in usable.items()]
        )

        self.embedding = EmbeddingSpace(
            projection=self.config.projection,
            pca_variance=self.config.pca_variance,
            max_components=self.config.max_components,
            discriminant_weight=self.config.discriminant_weight,
            feature_weights=self.config.feature_weights,
        ).fit(all_features, labels)

        projected_all = self.embedding.transform(all_features)
        self._global_scale = float(_robust_scale(projected_all)) or 1.0

        self.profiles = []
        cursor = 0
        for gid, matrix in usable.items():
            count = len(matrix)
            vectors = projected_all[cursor : cursor + count]
            cursor += count
            self.profiles.append(
                GroupProfile(
                    group_id=gid,
                    name=group_names.get(gid, str(gid)),
                    vectors=vectors,
                    centroid=vectors.mean(axis=0),
                    scale=self._group_scale(vectors),
                    size=count,
                )
            )
        return self

    def _group_scale(self, vectors: np.ndarray) -> float:
        """Typical distance from a group's members to its centre.

        Used to make distances comparable between a tight seed group and a
        broad folder. Falls back to the global scale when a group is too small
        to estimate its own spread.
        """
        if not self.config.scale_normalization or len(vectors) < 3:
            return self._global_scale
        distances = np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)
        scale = float(np.median(distances))
        if not np.isfinite(scale) or scale <= 1e-9:
            return self._global_scale
        # Keep any single group from dominating through an extreme spread.
        return float(np.clip(scale, 0.25 * self._global_scale, 4.0 * self._global_scale))

    @property
    def is_fitted(self) -> bool:
        return self.embedding is not None and len(self.profiles) >= 2

    @property
    def group_ids(self) -> List[int]:
        return [profile.group_id for profile in self.profiles]

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def suggest(
        self,
        feature_vector: np.ndarray,
        track_id: Optional[int] = None,
        exclude_index: Optional[Tuple[int, int]] = None,
        top_k: int = 5,
    ) -> Suggestion:
        """Rank groups for a single raw feature vector."""
        if not self.is_fitted:
            raise RuntimeError("GroupSorter must be fitted before use")

        point = self.embedding.transform(np.asarray(feature_vector, dtype=float).reshape(1, -1))[0]
        return self._suggest_projected(point, track_id, exclude_index, top_k)

    def suggest_batch(
        self,
        feature_matrix: np.ndarray,
        track_ids: Optional[Sequence[int]] = None,
        top_k: int = 5,
    ) -> List[Suggestion]:
        """Rank groups for many tracks at once."""
        if not self.is_fitted:
            raise RuntimeError("GroupSorter must be fitted before use")

        feature_matrix = np.asarray(feature_matrix, dtype=float)
        if feature_matrix.ndim == 1:
            feature_matrix = feature_matrix.reshape(1, -1)
        projected = self.embedding.transform(feature_matrix)

        ids = list(track_ids) if track_ids is not None else [None] * len(projected)
        return [
            self._suggest_projected(point, track_id, None, top_k)
            for point, track_id in zip(projected, ids)
        ]

    def _suggest_projected(
        self,
        point: np.ndarray,
        track_id: Optional[int],
        exclude_index: Optional[Tuple[int, int]],
        top_k: int,
    ) -> Suggestion:
        scored: List[Tuple[GroupProfile, float]] = []
        for profile in self.profiles:
            skip = exclude_index[1] if exclude_index and exclude_index[0] == profile.group_id else None
            distance = self._distance_to_group(point, profile, skip)
            if distance is not None:
                scored.append((profile, distance))

        if not scored:
            return Suggestion(track_id, [], STATUS_UNMATCHED, 0.0, 0.0, float("inf"))

        scored.sort(key=lambda item: item[1])
        distances = np.array([distance for _, distance in scored])

        # Softmax over negative normalised distance gives a comparable
        # confidence regardless of the absolute scale of the space.
        temperature = max(self.config.temperature, 1e-3)
        weights = np.exp(-(distances - distances[0]) / temperature)
        probabilities = weights / weights.sum()

        ranked = [
            {
                "group_id": profile.group_id,
                "group_name": profile.name,
                "distance": float(distance),
                "score": float(probability),
            }
            for (profile, distance), probability in zip(scored, probabilities)
        ][:top_k]

        confidence = float(probabilities[0])
        margin = float(probabilities[0] - probabilities[1]) if len(probabilities) > 1 else 1.0
        best_distance = float(distances[0])

        status = self._decide(confidence, margin, best_distance)
        return Suggestion(track_id, ranked, status, confidence, margin, best_distance)

    def _distance_to_group(
        self, point: np.ndarray, profile: GroupProfile, skip_index: Optional[int]
    ) -> Optional[float]:
        """Scale-normalised distance from a point to a group."""
        vectors = profile.vectors
        if skip_index is not None:
            mask = np.ones(len(vectors), dtype=bool)
            mask[skip_index] = False
            vectors = vectors[mask]
            if len(vectors) == 0:
                return None
            centroid = vectors.mean(axis=0)
        else:
            centroid = profile.centroid

        member_distances = np.linalg.norm(vectors - point, axis=1)
        k = max(1, min(self.config.neighbors, len(member_distances)))
        knn_distance = float(np.mean(np.partition(member_distances, k - 1)[:k]))
        centroid_distance = float(np.linalg.norm(centroid - point))

        weight = float(np.clip(self.config.knn_weight, 0.0, 1.0))
        blended = weight * knn_distance + (1.0 - weight) * centroid_distance
        return blended / max(profile.scale, 1e-9)

    def _decide(self, confidence: float, margin: float, distance: float) -> str:
        if distance > self.config.novelty_factor:
            # Nothing in the collection is close enough for the ranking to mean
            # anything — this is new territory, not a weak match.
            return STATUS_UNMATCHED
        if confidence >= self.config.auto_accept_confidence and margin >= self.config.min_margin:
            return STATUS_AUTO
        return STATUS_REVIEW

    # ------------------------------------------------------------------
    # Self-evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Dict[str, Any]:
        """Leave-one-out check of how separable the reference groups are.

        Each reference track is re-scored with itself removed from its own
        group. The result answers the question a DJ actually has: "are my
        folders distinct enough for this to work, and which two do I keep
        mixing up?"

        The embedding is fitted once on all references rather than per fold, so
        treat the accuracy as a mild over-estimate.
        """
        if not self.is_fitted:
            raise RuntimeError("GroupSorter must be fitted before evaluation")

        group_ids = self.group_ids
        index_of = {gid: i for i, gid in enumerate(group_ids)}
        confusion = np.zeros((len(group_ids), len(group_ids)), dtype=int)

        per_group: Dict[int, Dict[str, Any]] = {}
        total = 0
        correct = 0
        confidences: List[float] = []

        for profile in self.profiles:
            hits = 0
            for i, point in enumerate(profile.vectors):
                suggestion = self._suggest_projected(
                    point, None, (profile.group_id, i), top_k=len(group_ids)
                )
                predicted = suggestion.group_id
                total += 1
                confidences.append(suggestion.confidence)
                if predicted is not None:
                    confusion[index_of[profile.group_id], index_of[predicted]] += 1
                if predicted == profile.group_id:
                    hits += 1
                    correct += 1

            per_group[profile.group_id] = {
                "group_id": profile.group_id,
                "name": profile.name,
                "size": profile.size,
                "correct": hits,
                "accuracy": hits / profile.size if profile.size else 0.0,
            }

        confusable = self._confusable_pairs(confusion, group_ids)

        return {
            "accuracy": correct / total if total else 0.0,
            "n_tracks": total,
            "n_groups": len(group_ids),
            "mean_confidence": float(np.mean(confidences)) if confidences else 0.0,
            "per_group": list(per_group.values()),
            "group_ids": group_ids,
            "group_names": [profile.name for profile in self.profiles],
            "confusion": confusion.tolist(),
            "confusable_pairs": confusable,
            "embedding": self.embedding.describe() if self.embedding else {},
        }

    def _confusable_pairs(
        self, confusion: np.ndarray, group_ids: List[int], limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Group pairs that swallow each other's tracks most often."""
        names = {profile.group_id: profile.name for profile in self.profiles}
        sizes = {profile.group_id: profile.size for profile in self.profiles}
        pairs = []
        for i, source in enumerate(group_ids):
            for j, target in enumerate(group_ids):
                if i == j or confusion[i, j] == 0:
                    continue
                pairs.append(
                    {
                        "from_group_id": source,
                        "from_name": names[source],
                        "to_group_id": target,
                        "to_name": names[target],
                        "count": int(confusion[i, j]),
                        "rate": float(confusion[i, j] / max(sizes[source], 1)),
                    }
                )
        pairs.sort(key=lambda pair: (pair["rate"], pair["count"]), reverse=True)
        return pairs[:limit]

    def nearest_references(
        self, feature_vector: np.ndarray, group_id: int, k: int = 3
    ) -> List[Tuple[int, float]]:
        """Indices of the group members closest to a track, for "why this group?".

        Returns ``(member_index, distance)`` pairs; the caller maps indices back
        to track IDs using the order the group was fitted in.
        """
        profile = next((p for p in self.profiles if p.group_id == group_id), None)
        if profile is None:
            return []
        point = self.embedding.transform(np.asarray(feature_vector, dtype=float).reshape(1, -1))[0]
        distances = np.linalg.norm(profile.vectors - point, axis=1)
        order = np.argsort(distances)[:k]
        return [(int(i), float(distances[i])) for i in order]

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def dumps(self) -> bytes:
        """Serialise the fitted sorter for storage in the database."""
        if not self.is_fitted:
            raise RuntimeError("Cannot serialise an unfitted sorter")
        return pickle.dumps(
            {
                "format": MODEL_FORMAT,
                "config": self.config.to_dict(),
                "embedding": self.embedding,
                "global_scale": self._global_scale,
                "profiles": [
                    {
                        "group_id": profile.group_id,
                        "name": profile.name,
                        "vectors": profile.vectors,
                        "centroid": profile.centroid,
                        "scale": profile.scale,
                        "size": profile.size,
                    }
                    for profile in self.profiles
                ],
            },
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    @classmethod
    def loads(cls, payload: bytes) -> "GroupSorter":
        data = pickle.loads(payload)
        if data.get("format") != MODEL_FORMAT:
            raise ValueError("Stored sorter was written by an incompatible version; refit it")

        sorter = cls(SorterConfig.from_dict(data["config"]))
        sorter.embedding = data["embedding"]
        sorter._global_scale = data.get("global_scale", 1.0)
        sorter.profiles = [GroupProfile(**profile) for profile in data["profiles"]]
        return sorter


def _robust_scale(vectors: np.ndarray) -> float:
    """Median distance from the overall centre — a stable unit for the space."""
    if len(vectors) < 2:
        return 1.0
    distances = np.linalg.norm(vectors - vectors.mean(axis=0), axis=1)
    scale = float(np.median(distances))
    return scale if np.isfinite(scale) and scale > 1e-9 else 1.0
