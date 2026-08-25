"""The metric space that sorting decisions are made in.

Raw feature vectors are not a good place to measure "is this track like the
stuff in my Deep House folder?": the dimensions have wildly different scales,
many are redundant, and none of them know which distinctions the DJ actually
cares about. This module fits a transform from raw vectors to a space where
plain Euclidean distance means something:

1. standardise every dimension,
2. apply optional per-family weights (timbre / rhythm / harmony / dynamics),
3. reduce with PCA, and — when the reference groups provide labels — rotate
   with LDA so the directions that separate *the DJ's own groups* dominate.

Step 3 is what makes the directed workflow better than generic clustering: the
existing folders supply supervision, so the space is tuned to the boundaries
the DJ has already drawn.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

from . import profiles
from .features import FeatureLayout, family_weight_vector, layout_for_dim


PROJECTIONS = ("auto", "none", "pca", "lda", "nca")


@dataclass
class EmbeddingSpace:
    """A fitted, serialisable feature transform."""

    projection: str = "auto"
    pca_variance: float = 0.95
    max_components: int = 40
    # How much of the space is given over to the directions that separate the
    # existing groups, versus general audio structure. See _SupervisedBlend.
    discriminant_weight: float = 0.7
    feature_weights: Dict[str, float] = field(default_factory=dict)
    # Which feature layout these vectors follow, so per-family weights land on
    # the right dimensions. Only consulted while fitting.
    profile: str = profiles.MUSIC

    scaler: Optional[StandardScaler] = None
    weight_vector: Optional[np.ndarray] = None
    reducer: Optional[Any] = None
    resolved_projection: str = "none"
    input_dim: int = 0
    output_dim: int = 0

    # ------------------------------------------------------------------

    def fit(
        self, features: np.ndarray, labels: Union[Sequence[Any], np.ndarray, None] = None
    ) -> "EmbeddingSpace":
        """Fit on reference vectors, using group labels for supervision if given."""
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[0] == 0:
            raise ValueError("Need a 2-D feature matrix with at least one row")

        self.input_dim = features.shape[1]
        layout: FeatureLayout = layout_for_dim(self.input_dim, self.profile)

        self.scaler = StandardScaler().fit(features)
        scaled = self.scaler.transform(features)

        if self.feature_weights:
            self.weight_vector = family_weight_vector(layout, self.feature_weights)
            scaled = scaled * self.weight_vector
        else:
            self.weight_vector = None

        labels_array = np.asarray(labels) if labels is not None else None
        self.resolved_projection = self._resolve_projection(scaled, labels_array)
        self.reducer = self._fit_reducer(scaled, labels_array, self.resolved_projection)

        self.output_dim = self.transform(features[:1]).shape[1]
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Project raw feature vectors into the fitted space."""
        if self.scaler is None:
            raise RuntimeError("EmbeddingSpace must be fitted before transform")

        features = np.asarray(features, dtype=float)
        single = features.ndim == 1
        if single:
            features = features.reshape(1, -1)
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected {self.input_dim}-dimensional features, got {features.shape[1]}"
            )

        scaled = self.scaler.transform(features)
        if self.weight_vector is not None:
            scaled = scaled * self.weight_vector
        if self.reducer is not None:
            scaled = self.reducer.transform(scaled)

        return scaled[0].reshape(1, -1) if single else scaled

    # ------------------------------------------------------------------

    def _resolve_projection(self, scaled: np.ndarray, labels: Optional[np.ndarray]) -> str:
        requested = (self.projection or "auto").lower()
        if requested not in PROJECTIONS:
            raise ValueError(f"Unknown projection {requested!r}; choose from {PROJECTIONS}")

        if requested != "auto":
            if requested in ("lda", "nca") and not self._supervision_viable(scaled, labels):
                # Not enough labelled examples for supervised reduction; PCA is
                # the honest fallback rather than a badly overfitted rotation.
                return "pca"
            return requested

        if self._supervision_viable(scaled, labels):
            return "lda"
        return "pca" if scaled.shape[0] > 2 else "none"

    @staticmethod
    def _supervision_viable(scaled: np.ndarray, labels: Optional[np.ndarray]) -> bool:
        if labels is None or len(labels) != scaled.shape[0]:
            return False
        unique, counts = np.unique(labels, return_counts=True)
        # LDA needs at least two classes, and enough members per class that the
        # within-class scatter is not pure noise.
        return len(unique) >= 2 and counts.min() >= 3 and scaled.shape[0] >= 3 * len(unique)

    def _fit_reducer(self, scaled: np.ndarray, labels: Optional[np.ndarray], projection: str):
        if projection == "none":
            return None

        if projection == "pca":
            return self._fit_pca(scaled)

        # Supervised projections are numerically happier on a PCA-whitened
        # basis, so they run as a two-stage pipeline.
        pca = self._fit_pca(scaled)
        reduced = pca.transform(scaled) if pca is not None else scaled

        # _resolve_projection only returns a supervised projection when
        # _supervision_viable passed, which requires labels.
        if labels is None:
            raise ValueError(f"The {projection!r} projection needs group labels")

        supervised = None
        if projection == "lda":
            n_classes = len(np.unique(labels))
            n_components = max(1, min(n_classes - 1, reduced.shape[1]))
            # 'eigen' + shrinkage stays stable when groups are small, which is
            # the normal case for hand-picked seed groups.
            supervised = LinearDiscriminantAnalysis(
                n_components=n_components, solver="eigen", shrinkage="auto"
            )
        elif projection == "nca":
            try:
                from sklearn.neighbors import NeighborhoodComponentsAnalysis
            except ImportError:  # pragma: no cover
                return pca
            supervised = NeighborhoodComponentsAnalysis(
                n_components=max(2, min(reduced.shape[1], 16)),
                init="pca",
                max_iter=100,
                random_state=42,
            )

        if supervised is None:
            return None

        try:
            supervised.fit(reduced, labels)
        except Exception:
            # Degenerate scatter (near-duplicate references, collinear groups);
            # the unsupervised reduction is still perfectly usable.
            return pca

        return _SupervisedBlend(pca, supervised, reduced, self.discriminant_weight)

    def _fit_pca(self, scaled: np.ndarray) -> Optional[PCA]:
        n_samples, n_features = scaled.shape
        max_components = min(self.max_components, n_features, max(1, n_samples - 1))
        if max_components < 2:
            return None
        pca = PCA(n_components=max_components, random_state=42).fit(scaled)

        # Trim to the components needed for the requested explained variance.
        cumulative = np.cumsum(pca.explained_variance_ratio_)
        keep = int(np.searchsorted(cumulative, self.pca_variance) + 1)
        keep = max(2, min(keep, max_components))
        if keep < max_components:
            pca = PCA(n_components=keep, random_state=42).fit(scaled)
        return pca

    # ------------------------------------------------------------------

    def describe(self) -> Dict[str, Any]:
        """Summary suitable for showing in the UI or CLI."""
        return {
            "projection": self.resolved_projection,
            "requested_projection": self.projection,
            "profile": self.profile,
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "feature_weights": dict(self.feature_weights),
        }


class _SupervisedBlend:
    """PCA components plus supervised discriminant directions, side by side.

    A pure LDA projection has only ``n_groups - 1`` dimensions. That is ideal
    for telling the existing groups apart and terrible at noticing that a track
    resembles none of them — everything gets squashed onto the few axes that
    separate what is already known. Keeping the general PCA directions
    alongside the discriminant ones preserves enough of the original geometry
    for novelty detection and within-group structure, while the weight keeps
    the discriminant axes in charge of the decision.

    Each block is divided by its own typical magnitude first, so the weight
    means what it says regardless of the two transforms' natural scales.
    """

    def __init__(self, pca: Any, supervised: Any, reference: np.ndarray, weight: float):
        self.pca = pca
        self.supervised = supervised
        self.weight = float(np.clip(weight, 0.0, 1.0))
        self.general_scale = _block_scale(reference)
        self.discriminant_scale = _block_scale(supervised.transform(reference))

    def transform(self, X: np.ndarray) -> np.ndarray:
        reduced = self.pca.transform(X) if self.pca is not None else X
        general = reduced / self.general_scale * (1.0 - self.weight)
        discriminant = self.supervised.transform(reduced) / self.discriminant_scale * self.weight
        return np.hstack([general, discriminant])


def _block_scale(block: np.ndarray) -> float:
    """Typical per-row magnitude of a transform's output."""
    if block.size == 0:
        return 1.0
    scale = float(np.sqrt(np.mean(np.sum(np.square(block), axis=1))))
    return scale if np.isfinite(scale) and scale > 1e-9 else 1.0
