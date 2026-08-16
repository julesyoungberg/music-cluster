"""Label suggestions for discovered groups.

The DJ names their own groups — that is the point of the directed workflow —
but a discovery run can throw twenty unnamed candidates at them, and staring at
"Candidate 14" is no way to decide anything. These suggestions exist to give
each candidate a handle: what the tags say, how fast it is, and how it sounds.

Signals are used in order of trustworthiness: existing genre tags first
(the DJ's own vocabulary), then tempo, then audio descriptors.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from .features import describe_vector


# Tempo bands, chosen to line up with how dance music is actually filed.
TEMPO_BANDS = [
    (0, 90, "Downtempo"),
    (90, 105, "Slow"),
    (105, 118, "House"),
    (118, 126, "House"),
    (126, 134, "Techno"),
    (134, 145, "Fast Techno"),
    (145, 155, "Hard"),
    (155, 175, "Drum & Bass"),
    (175, 400, "Very Fast"),
]

_GENRE_CLEAN = re.compile(r"\s*[/|,;]\s*")


def suggest_label(
    centroid: np.ndarray,
    member_tracks: List[Dict[str, Any]],
    fallback: str = "Untitled group",
) -> str:
    """Suggest a short label for a group from its centroid and members."""
    parts: List[str] = []

    genre = dominant_genre(member_tracks)
    descriptors = audio_descriptors(centroid)
    bpm_label = bpm_range_label(centroid, member_tracks)

    if genre:
        parts.append(genre)
    elif descriptors:
        parts.append(descriptors[0])
        descriptors = descriptors[1:]

    if not parts:
        parts.append(tempo_band(_representative_bpm(centroid, member_tracks)))

    if descriptors:
        parts.insert(0, descriptors[0])

    label = " ".join(dict.fromkeys(parts)).strip()
    if bpm_label:
        label = f"{label} {bpm_label}".strip()
    return label or fallback


def dominant_genre(tracks: List[Dict[str, Any]], min_share: float = 0.35) -> Optional[str]:
    """The genre tag shared by a meaningful share of a group's tracks."""
    values: List[str] = []
    for track in tracks:
        raw = (track.get("genre") or "").strip()
        if not raw:
            continue
        for piece in _GENRE_CLEAN.split(raw):
            piece = piece.strip()
            if piece:
                values.append(piece.title())

    if not values:
        return None
    genre, count = Counter(values).most_common(1)[0]
    tagged = sum(1 for track in tracks if (track.get("genre") or "").strip())
    return genre if tagged and count / tagged >= min_share else None


def bpm_range_label(
    centroid: np.ndarray, tracks: List[Dict[str, Any]], spread_limit: int = 30
) -> Optional[str]:
    """A BPM range label, omitted when the group is too tempo-diverse to matter."""
    bpms = [
        float(track["tag_bpm"])
        for track in tracks
        if track.get("tag_bpm") and 40 <= float(track["tag_bpm"]) <= 250
    ]
    if len(bpms) < max(2, len(tracks) // 4):
        detected = _representative_bpm(centroid, tracks)
        return f"{round(detected)} BPM" if detected else None

    low, high = int(np.percentile(bpms, 10)), int(np.percentile(bpms, 90))
    if high - low > spread_limit:
        return None
    if high - low <= 2:
        return f"{low} BPM"
    return f"{low}-{high} BPM"


def tempo_band(bpm: Optional[float]) -> str:
    """Coarse name for a tempo, used when nothing else identifies a group."""
    if not bpm:
        return "Mixed"
    for low, high, name in TEMPO_BANDS:
        if low <= bpm < high:
            return name
    return "Mixed"


def audio_descriptors(centroid: np.ndarray, limit: int = 2) -> List[str]:
    """Adjectives derived from the centroid's position in feature space.

    Thresholds are deliberately loose: these are hints for a human who is about
    to rename the group anyway, not classifications.
    """
    stats = describe_vector(np.asarray(centroid, dtype=float))
    descriptors: List[tuple] = []

    brightness = stats["brightness_hz"]
    if brightness:
        if brightness < 1400:
            descriptors.append((abs(brightness - 1400), "Dark"))
        elif brightness > 2800:
            descriptors.append((brightness - 2800, "Bright"))

    complexity = stats["timbral_complexity"]
    if complexity:
        if complexity < 8:
            descriptors.append((8 - complexity, "Minimal"))
        elif complexity > 20:
            descriptors.append((complexity - 20, "Busy"))

    if stats["bass_weight"] > 20:
        descriptors.append((stats["bass_weight"] - 20, "Deep"))

    if stats["onset_strength"] > 3.0:
        descriptors.append((stats["onset_strength"] - 3.0, "Percussive"))

    if stats["harmonic_strength"] > 0.45:
        descriptors.append((stats["harmonic_strength"], "Melodic"))

    if stats["flatness"] > 0.02:
        descriptors.append((stats["flatness"] * 100, "Noisy"))

    descriptors.sort(key=lambda item: item[0], reverse=True)
    return [name for _, name in descriptors[:limit]]


def group_stats(centroid: np.ndarray, tracks: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summary shown next to a candidate group during review."""
    stats = describe_vector(np.asarray(centroid, dtype=float))
    bpms = [
        float(track["tag_bpm"])
        for track in tracks
        if track.get("tag_bpm") and 40 <= float(track["tag_bpm"]) <= 250
    ]
    genres = Counter(
        (track.get("genre") or "").strip().title()
        for track in tracks
        if (track.get("genre") or "").strip()
    )
    artists = Counter(
        (track.get("artist") or "").strip()
        for track in tracks
        if (track.get("artist") or "").strip()
    )

    return {
        "detected_bpm": round(stats["bpm"], 1),
        "tag_bpm_median": round(float(np.median(bpms)), 1) if bpms else None,
        "tag_bpm_range": [round(min(bpms), 1), round(max(bpms), 1)] if bpms else None,
        "brightness_hz": round(stats["brightness_hz"], 1),
        "energy": round(stats["energy"], 5),
        "descriptors": audio_descriptors(centroid, limit=3),
        "top_genres": [{"name": name, "count": count} for name, count in genres.most_common(3)],
        "top_artists": [{"name": name, "count": count} for name, count in artists.most_common(3)],
    }


def _representative_bpm(centroid: np.ndarray, tracks: List[Dict[str, Any]]) -> Optional[float]:
    """Prefer tag BPM (DJ software writes it accurately) over detected tempo."""
    bpms = [
        float(track["tag_bpm"])
        for track in tracks
        if track.get("tag_bpm") and 40 <= float(track["tag_bpm"]) <= 250
    ]
    if bpms:
        return float(np.median(bpms))
    detected = describe_vector(np.asarray(centroid, dtype=float))["bpm"]
    return detected if 40 <= detected <= 250 else None
