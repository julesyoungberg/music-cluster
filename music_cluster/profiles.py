"""Audio profiles: what kind of material a collection is made of.

A full-length track and a 300-millisecond kick are both audio, and almost
nothing else about them is the same. Analysing a track means taking a
representative excerpt from the middle and measuring how it behaves over
ninety seconds. Analysing a one-shot means measuring the whole thing from the
first sample, because the *whole thing* is one event — and what tells a kick
from a clap is its envelope and spectral balance, not its tempo.

So the pipeline is parameterised by a profile:

``music``
    Full-length tracks. The original behaviour, unchanged.

``sample``
    One-shots and loops: kicks, snares, claps, hats, basses, chords, stabs,
    pads, FX. Analyses the file from its start at a much finer time
    resolution, and measures an extra block of features — attack and decay,
    spectral balance across eight bands, pitch and harmonicity — that only
    mean something for a single sonic event.

A profile is fixed per collection, because everything downstream (the feature
layout, the fitted embedding, the stored vectors) has to agree on it. Sorting
samples and sorting tracks are two collections, not two modes of one.
"""

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple


MUSIC = "music"
SAMPLE = "sample"

DEFAULT_PROFILE = MUSIC


@dataclass(frozen=True)
class Profile:
    """One kind of audio material, and how the pipeline should treat it."""

    name: str
    label: str
    description: str
    # Overrides layered over the matching config section for this profile.
    feature_extraction: Dict[str, Any] = field(default_factory=dict)
    sorting: Dict[str, Any] = field(default_factory=dict)
    discovery: Dict[str, Any] = field(default_factory=dict)

    def overrides(self, section: str) -> Dict[str, Any]:
        return copy.deepcopy(getattr(self, section, {}) or {})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "feature_extraction": copy.deepcopy(self.feature_extraction),
            "sorting": copy.deepcopy(self.sorting),
            "discovery": copy.deepcopy(self.discovery),
        }


PROFILES: Dict[str, Profile] = {
    MUSIC: Profile(
        name=MUSIC,
        label="Music",
        description="Full-length tracks, mixes and DJ edits.",
    ),
    SAMPLE: Profile(
        name=SAMPLE,
        label="Samples",
        description="One-shots and loops: kicks, snares, claps, hats, basses, chords, FX.",
        feature_extraction={
            # Frame size and hop are independent, and a one-shot wants both
            # ends: a 2048-point window resolves the low notes an 808 is made
            # of, while a 256-sample hop (5.8 ms) still gives a 60 ms tick
            # enough frames to have a measurable shape.
            "frame_size": 2048,
            "hop_size": 256,
            # Never excerpt: the attack is the most identifying part of a
            # sample, and it is at the very start of the file.
            "excerpt_seconds": 0,
            # ...but a "sample" folder always contains a few five-minute
            # construction kits by mistake, and there is nothing to learn from
            # minutes four and five of one.
            "max_seconds": 30,
        },
        sorting={
            # Envelope is what separates a clap from a snare when their spectra
            # are nearly identical; tempo means little for a single hit.
            "feature_weights": {
                "timbre": 1.0,
                "rhythm": 0.6,
                "harmony": 0.9,
                "dynamics": 1.0,
                "envelope": 1.3,
            },
            # Sample folders are big and multi-modal — a "Kicks" folder holds
            # 808s, acoustic kicks and layered kicks — so lean further on
            # nearest neighbours than on the centroid.
            "neighbors": 7,
            "knn_weight": 0.8,
        },
        discovery={
            # Sample packs are organised in smaller piles than record crates.
            "min_group_size": 5,
            "exemplars_per_candidate": 6,
        },
    ),
}


#: Config sections that accept per-profile overrides.
PROFILED_SECTIONS: Tuple[str, ...] = ("feature_extraction", "sorting", "discovery")


def names() -> List[str]:
    """Every known profile name, in presentation order."""
    return [MUSIC, SAMPLE]


def normalize(name: Any) -> str:
    """Coerce a profile name, falling back to the default for empty values.

    Raises ValueError for a name that is present but unknown — a typo in a
    config file or an API call should be reported, not silently treated as
    music.
    """
    if name is None or name == "":
        return DEFAULT_PROFILE
    text = str(name).strip().lower()
    if text not in PROFILES:
        raise ValueError(f"Unknown audio profile {name!r}; choose from {', '.join(names())}")
    return text


def get(name: Any = None) -> Profile:
    """The Profile for a name (or the default)."""
    return PROFILES[normalize(name)]


def is_sample(name: Any) -> bool:
    return normalize(name) == SAMPLE


def describe_all() -> List[Dict[str, Any]]:
    """Every profile as a payload for the API and UI."""
    return [PROFILES[name].to_dict() for name in names()]
