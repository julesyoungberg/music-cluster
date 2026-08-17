"""Feature vector layout and descriptor helpers.

The extractor produces a single fixed-length vector per track. Every other
module (sorting, discovery, labelling) needs to be able to point at individual
parts of that vector, so the layout is declared here once and derived from the
extractor settings rather than hard-coded at each call site.

The layout also depends on the collection's audio profile (see
:mod:`music_cluster.profiles`): the ``sample`` profile appends a block of
one-shot descriptors — envelope shape, spectral balance, pitch — that would be
meaningless averaged over a five-minute track.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from . import profiles


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

# Number of log-spaced energy bands measured for samples. Eight is enough to
# separate a sub from a kick from a snare from a hat by spectral balance alone.
N_BANDS = 8

# Band edges in Hz, chosen to line up with how a producer talks about a drum
# kit: sub, low, low-mid, mid, upper-mid, presence, brilliance, air.
BAND_EDGES = (20.0, 60.0, 120.0, 250.0, 500.0, 1000.0, 2500.0, 6000.0, 22050.0)

BAND_LABELS = (
    "sub",
    "low",
    "low_mid",
    "mid",
    "upper_mid",
    "presence",
    "brilliance",
    "air",
)

# Appended for the `sample` profile only. A one-shot is a single event, so it
# can be described by the shape of that event rather than by statistics over
# many bars.
SAMPLE_BLOCK = [
    # Envelope: the shape that tells a clap from a snare and a pluck from a pad.
    ("attack_time", 1),
    ("attack_slope", 1),
    ("decay_time", 1),
    ("sustain_ratio", 1),
    ("release_time", 1),
    ("temporal_centroid", 1),
    ("effective_duration", 1),
    ("log_duration", 1),
    ("crest_factor", 1),
    # Is this one hit, or a bar of them?
    ("onset_count", 1),
    ("percussive_ratio", 1),
    # Pitch: a bass or a chord has one, a hat does not.
    ("f0_hz", 1),
    ("f0_confidence", 1),
    ("pitch_stability", 1),
    ("harmonic_support", 1),
    ("chroma_peaks", 1),
    # Where the energy sits, and where it moves to as the sample decays.
    ("band_energy", N_BANDS),
    ("brightness_attack", 1),
    ("brightness_decay", 1),
    ("brightness_slope", 1),
    ("flux_mean", 1),
    ("flux_std", 1),
]

# Semantic families used for configurable feature weighting. Every span in the
# layout belongs to exactly one family. `envelope` only appears in the sample
# profile; weighting it on a music layout is a harmless no-op.
FAMILIES = ("timbre", "rhythm", "harmony", "dynamics", "envelope")

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
    # Sample block.
    "attack_time": "envelope",
    "attack_slope": "envelope",
    "decay_time": "envelope",
    "sustain_ratio": "envelope",
    "release_time": "envelope",
    "temporal_centroid": "envelope",
    "effective_duration": "envelope",
    "log_duration": "envelope",
    "crest_factor": "envelope",
    "onset_count": "rhythm",
    "percussive_ratio": "rhythm",
    "f0_hz": "harmony",
    "f0_confidence": "harmony",
    "pitch_stability": "harmony",
    "harmonic_support": "harmony",
    "chroma_peaks": "harmony",
    "band_energy": "timbre",
    "brightness_attack": "timbre",
    "brightness_decay": "timbre",
    "brightness_slope": "timbre",
    "flux_mean": "timbre",
    "flux_std": "timbre",
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
    profile: str = profiles.MUSIC

    def has(self, name: str) -> bool:
        return name in self.spans

    @property
    def is_sample(self) -> bool:
        return self.profile == profiles.SAMPLE

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


def _tail_blocks(profile: str) -> List[Tuple[str, int]]:
    """Track-level blocks for a profile, in extractor order."""
    blocks = RHYTHM_BLOCK + HIGHLEVEL_BLOCK
    if profile == profiles.SAMPLE:
        blocks = blocks + SAMPLE_BLOCK
    return blocks


def build_layout(n_mfcc: int = 20, profile: str = profiles.MUSIC) -> FeatureLayout:
    """Build the index map for a feature vector.

    The vector is ``[frame means, frame stds, rhythm, high-level]``, plus the
    sample block for the ``sample`` profile, where the frame blocks appear in
    :func:`_frame_blocks` order.
    """
    profile = profiles.normalize(profile)
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

    for name, width in _tail_blocks(profile):
        spans[name] = (cursor, cursor + width)
        cursor += width

    return FeatureLayout(n_mfcc=n_mfcc, spans=spans, dim=cursor, profile=profile)


DEFAULT_LAYOUT = build_layout(20)

#: Default vector width per profile, which is how a stored vector's profile is
#: recognised when nothing recorded it.
DEFAULT_DIMS = {name: build_layout(20, name).dim for name in profiles.names()}


def _solve_n_mfcc(dim: int, profile: str) -> Optional[int]:
    """Recover the MFCC count a vector of this width was built with."""
    fixed = sum(width for _, width in _tail_blocks(profile))
    other_frame = sum(width for name, width in _frame_blocks(0) if name != "mfcc")
    # dim = 2 * (n_mfcc + other_frame) + fixed
    remainder = dim - fixed - 2 * other_frame
    if remainder > 0 and remainder % 2 == 0:
        return remainder // 2
    return None


def layout_for_dim(dim: int, profile: Optional[str] = None) -> FeatureLayout:
    """Best-effort layout recovery for a stored vector.

    Pass the profile when it is known — the caller nearly always knows it from
    the collection. Without it the width is matched against each profile's
    default, which separates the two shipped profiles cleanly; a vector built
    with a non-default MFCC count is assumed to be music, the older format.
    """
    if profile is not None:
        profile = profiles.normalize(profile)
        if dim == DEFAULT_DIMS.get(profile):
            return build_layout(20, profile)
        n_mfcc = _solve_n_mfcc(dim, profile)
        return build_layout(n_mfcc, profile) if n_mfcc else build_layout(20, profile)

    for name, default_dim in DEFAULT_DIMS.items():
        if dim == default_dim:
            return build_layout(20, name)

    n_mfcc = _solve_n_mfcc(dim, profiles.MUSIC)
    return build_layout(n_mfcc, profiles.MUSIC) if n_mfcc else DEFAULT_LAYOUT


def family_weight_vector(layout: FeatureLayout, weights: Dict[str, float]) -> np.ndarray:
    """Expand per-family weights into a per-dimension multiplier vector."""
    multiplier = np.ones(layout.dim, dtype=float)
    for family in FAMILIES:
        weight = float(weights.get(family, 1.0))
        idx = layout.family_indices(family)
        if len(idx):
            multiplier[idx] = weight
    return multiplier


def describe_vector(vector: np.ndarray, layout: Optional[FeatureLayout] = None) -> Dict[str, Any]:
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

    described = {
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

    if layout.has("attack_time"):
        described.update(describe_sample_vector(vector, layout))
    return described


def describe_sample_vector(
    vector: np.ndarray, layout: Optional[FeatureLayout] = None
) -> Dict[str, Any]:
    """One-shot descriptors in physical units, for labelling and display.

    Only meaningful for vectors extracted under the ``sample`` profile; raises
    KeyError for a music layout, which has none of these blocks.
    """
    vector = np.asarray(vector, dtype=float)
    layout = layout or layout_for_dim(len(vector), profiles.SAMPLE)

    bands = np.asarray(layout.slice(vector, "band_energy"), dtype=float)
    band_map = {label: float(value) for label, value in zip(BAND_LABELS, bands)}
    duration = float(np.expm1(layout.scalar(vector, "log_duration")))

    return {
        "attack_time": layout.scalar(vector, "attack_time"),
        "attack_slope": layout.scalar(vector, "attack_slope"),
        "decay_time": layout.scalar(vector, "decay_time"),
        "sustain_ratio": layout.scalar(vector, "sustain_ratio"),
        "release_time": layout.scalar(vector, "release_time"),
        "temporal_centroid": layout.scalar(vector, "temporal_centroid"),
        "effective_duration": layout.scalar(vector, "effective_duration"),
        # Stored as log1p so that 50 ms and 5 s are comparable magnitudes to a
        # distance metric; reported here as the seconds a human expects.
        "duration": duration,
        "crest_factor": layout.scalar(vector, "crest_factor"),
        "onset_count": layout.scalar(vector, "onset_count"),
        "onset_rate": layout.scalar(vector, "onset_count") / max(duration, 1e-3),
        "percussive_ratio": layout.scalar(vector, "percussive_ratio"),
        "f0_hz": layout.scalar(vector, "f0_hz"),
        "f0_confidence": layout.scalar(vector, "f0_confidence"),
        "pitch_stability": layout.scalar(vector, "pitch_stability"),
        "harmonic_support": layout.scalar(vector, "harmonic_support"),
        "chroma_peaks": layout.scalar(vector, "chroma_peaks"),
        "brightness_attack": layout.scalar(vector, "brightness_attack"),
        "brightness_decay": layout.scalar(vector, "brightness_decay"),
        "brightness_slope": layout.scalar(vector, "brightness_slope"),
        "flux_mean": layout.scalar(vector, "flux_mean"),
        "flux_std": layout.scalar(vector, "flux_std"),
        "bands": band_map,
        # Handy roll-ups: the three questions asked most often about a one-shot.
        "low_energy": band_map["sub"] + band_map["low"],
        "mid_energy": band_map["low_mid"] + band_map["mid"] + band_map["upper_mid"],
        "high_energy": band_map["presence"] + band_map["brilliance"] + band_map["air"],
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
