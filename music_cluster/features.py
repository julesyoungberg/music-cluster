"""Feature vector layout and descriptor helpers.

The extractor produces a single fixed-length vector per track. Every other
module (sorting, discovery, labelling) needs to be able to point at individual
parts of that vector, so the layout is declared here once and derived from the
extractor settings rather than hard-coded at each call site.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


# Frame-level blocks, in the exact order the extractor concatenates them.
# (name, width) where width may depend on the number of MFCC coefficients.
def _frame_blocks(n_mfcc: int) -> List[Tuple[str, int]]:
    return [
        ("mfcc", n_mfcc),
        ("spectral_centroid", 1),
        ("spectral_rolloff", 1),
        ("spectral_contrast", 7),
        ("zcr", 1),
        ("chroma", 12),
        ("rms", 1),
    ]


# Track-level blocks appended after the aggregated frame features.
RHYTHM_BLOCK = [("tempo", 1), ("onset_mean", 1), ("onset_std", 1), ("onset_max", 1)]
HIGHLEVEL_BLOCK = [
    ("energy", 1),
    ("dynamic_range", 1),
    ("spectral_bandwidth_mean", 1),
    ("spectral_bandwidth_std", 1),
    ("spectral_flatness_mean", 1),
    ("spectral_flatness_std", 1),
]

# Semantic families used for configurable feature weighting. Every span in the
# layout belongs to exactly one family.
FAMILIES = ("timbre", "rhythm", "harmony", "dynamics")

_FAMILY_OF = {
    "mfcc": "timbre",
    "spectral_centroid": "timbre",
    "spectral_rolloff": "timbre",
    "spectral_contrast": "timbre",
    "zcr": "timbre",
    "chroma": "harmony",
    "rms": "dynamics",
    "tempo": "rhythm",
    "onset_mean": "rhythm",
    "onset_std": "rhythm",
    "onset_max": "rhythm",
    "energy": "dynamics",
    "dynamic_range": "dynamics",
    "spectral_bandwidth_mean": "timbre",
    "spectral_bandwidth_std": "timbre",
    "spectral_flatness_mean": "timbre",
    "spectral_flatness_std": "timbre",
}


def _family_of_span(span_name: str) -> str:
    """Resolve a span name (``mfcc_mean``, ``tempo``, ...) to its family."""
    if span_name in _FAMILY_OF:
        return _FAMILY_OF[span_name]
    for suffix in ("_mean", "_std"):
        if span_name.endswith(suffix):
            base = span_name[: -len(suffix)]
            if base in _FAMILY_OF:
                return _FAMILY_OF[base]
    raise KeyError(f"No feature family registered for span {span_name!r}")


@dataclass(frozen=True)
class FeatureLayout:
    """Index map for a feature vector produced with a given MFCC count."""

    n_mfcc: int
    spans: Dict[str, Tuple[int, int]]
    dim: int

    def span(self, name: str) -> Tuple[int, int]:
        if name not in self.spans:
            raise KeyError(f"Unknown feature block: {name}")
        return self.spans[name]

    def slice(self, vector: np.ndarray, name: str) -> np.ndarray:
        start, end = self.span(name)
        return vector[..., start:end]

    def scalar(self, vector: np.ndarray, name: str) -> float:
        start, end = self.span(name)
        if end - start != 1:
            raise ValueError(f"{name} is not a scalar block")
        return float(np.asarray(vector)[..., start])

    def family_indices(self, family: str) -> np.ndarray:
        """Indices of every element belonging to a semantic family."""
        idx: List[int] = []
        for name, (start, end) in self.spans.items():
            if _family_of_span(name) == family:
                idx.extend(range(start, end))
        return np.array(sorted(idx), dtype=int)


def build_layout(n_mfcc: int = 20) -> FeatureLayout:
    """Build the index map for a feature vector.

    The vector is ``[frame means, frame stds, rhythm, high-level]`` where the
    frame blocks appear in :func:`_frame_blocks` order.
    """
    spans: Dict[str, Tuple[int, int]] = {}
    cursor = 0

    frame = _frame_blocks(n_mfcc)
    frame_width = sum(width for _, width in frame)

    for suffix in ("mean", "std"):
        offset = 0
        for name, width in frame:
            start = cursor + offset
            spans[f"{name}_{suffix}"] = (start, start + width)
            offset += width
        cursor += frame_width

    for name, width in RHYTHM_BLOCK + HIGHLEVEL_BLOCK:
        spans[name] = (cursor, cursor + width)
        cursor += width

    return FeatureLayout(n_mfcc=n_mfcc, spans=spans, dim=cursor)


DEFAULT_LAYOUT = build_layout(20)


def layout_for_dim(dim: int) -> FeatureLayout:
    """Best-effort layout recovery for a stored vector of unknown MFCC count."""
    if dim == DEFAULT_LAYOUT.dim:
        return DEFAULT_LAYOUT
    fixed = sum(w for _, w in RHYTHM_BLOCK + HIGHLEVEL_BLOCK)
    other_frame = sum(w for name, w in _frame_blocks(0) if name != "mfcc")
    # dim = 2 * (n_mfcc + other_frame) + fixed
    remainder = dim - fixed - 2 * other_frame
    if remainder > 0 and remainder % 2 == 0:
        return build_layout(remainder // 2)
    return DEFAULT_LAYOUT


def family_weight_vector(layout: FeatureLayout, weights: Dict[str, float]) -> np.ndarray:
    """Expand per-family weights into a per-dimension multiplier vector."""
    multiplier = np.ones(layout.dim, dtype=float)
    for family in FAMILIES:
        weight = float(weights.get(family, 1.0))
        idx = layout.family_indices(family)
        if len(idx):
            multiplier[idx] = weight
    return multiplier


def describe_vector(vector: np.ndarray, layout: Optional[FeatureLayout] = None) -> Dict[str, float]:
    """Turn a raw (unscaled) feature vector into human-readable descriptors.

    Values are the physical quantities the extractor measured, so they can be
    shown to a user or used for label suggestions.
    """
    vector = np.asarray(vector, dtype=float)
    layout = layout or layout_for_dim(len(vector))

    mfcc_mean = layout.slice(vector, "mfcc_mean")
    chroma_mean = layout.slice(vector, "chroma_mean")
    contrast_mean = layout.slice(vector, "spectral_contrast_mean")

    tempo = layout.scalar(vector, "tempo")
    brightness = layout.scalar(vector, "spectral_centroid_mean")
    rolloff = layout.scalar(vector, "spectral_rolloff_mean")

    return {
        "bpm": tempo,
        "brightness_hz": brightness,
        "rolloff_hz": rolloff,
        # MFCC 0 tracks overall loudness/spectral energy; 1-4 carry the low-end
        # balance that separates bass-forward material from thinner mixes.
        "loudness_coeff": float(mfcc_mean[0]) if len(mfcc_mean) else 0.0,
        "bass_weight": float(np.mean(mfcc_mean[1:5])) if len(mfcc_mean) >= 5 else 0.0,
        "timbral_complexity": float(np.std(mfcc_mean[5:])) if len(mfcc_mean) > 5 else 0.0,
        "spectral_contrast": float(np.mean(contrast_mean)) if len(contrast_mean) else 0.0,
        "harmonic_strength": float(np.mean(chroma_mean)) if len(chroma_mean) else 0.0,
        "harmonic_variance": float(np.std(chroma_mean)) if len(chroma_mean) else 0.0,
        "noisiness": layout.scalar(vector, "zcr_mean"),
        "energy": layout.scalar(vector, "energy"),
        "rms": layout.scalar(vector, "rms_mean"),
        "dynamic_range": layout.scalar(vector, "dynamic_range"),
        "onset_strength": layout.scalar(vector, "onset_mean"),
        "flatness": layout.scalar(vector, "spectral_flatness_mean"),
    }


def aggregate_features(frame_features: np.ndarray) -> np.ndarray:
    """Aggregate frame-level features into a track-level vector (mean, std)."""
    if len(frame_features) == 0:
        raise ValueError("No frame features provided")

    means = np.mean(frame_features, axis=0)
    stds = np.std(frame_features, axis=0)
    aggregated = np.concatenate([means, stds])
    return np.nan_to_num(aggregated, nan=0.0, posinf=0.0, neginf=0.0)


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Cosine similarity between two vectors, 0 when either is degenerate."""
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def euclidean_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(vec1 - vec2))
