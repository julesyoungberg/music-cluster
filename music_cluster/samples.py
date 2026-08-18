"""Recognising what a one-shot is: a kick, a snare, a clap, a hat, a bass, a chord.

The rest of music-cluster deliberately refuses to invent categories — the DJ's
folders are the categories. Nothing here changes that. What this module does is
*suggest*, so that a discovery run over an unsorted sample pack hands back
"Kicks" and "Closed Hats" instead of "Candidate 7" and "Candidate 12". Every
suggestion is still renamed, kept or thrown away by a human.

Two kinds of evidence are combined:

**The filename.** Sample libraries are named by people who mean it. ``BD_808_01
.wav`` in a folder called ``Kicks`` is a kick, and no amount of spectral
analysis is going to know better. This is the stronger signal and is weighted
as such.

**The audio.** Where the energy sits, how long the event lasts, how it starts
and stops, and whether it has a note. This is what carries the badly-named
half of every pack — ``Sample 04.wav``, ``Untitled-3.wav`` — and what corrects
a filename that lies.

The rules are thresholds on measured quantities, not a trained model. They are
meant to be read, argued with and edited: every one of them is a claim about
what a kick sounds like, in the open, in one place.
"""

import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .features import FeatureLayout, describe_sample_vector, describe_vector


@dataclass(frozen=True)
class Category:
    """One kind of sample, as a producer would name the folder."""

    key: str
    label: str  # singular, for one file
    plural: str  # for a group of them
    keywords: Tuple[str, ...] = ()
    # Short tokens that only mean this in a sample library (``BD``, ``CH``).
    codes: Tuple[str, ...] = ()


CATEGORIES: Tuple[Category, ...] = (
    Category(
        "kick",
        "Kick",
        "Kicks",
        ("kick", "kicks", "bassdrum", "bassdrums", "boom"),
        ("bd", "kd", "kck"),
    ),
    Category("snare", "Snare", "Snares", ("snare", "snares"), ("sd", "sn", "snr")),
    Category("rim", "Rimshot", "Rimshots", ("rim", "rimshot", "sidestick"), ("rs",)),
    Category("clap", "Clap", "Claps", ("clap", "claps", "handclap"), ("cp", "clp")),
    Category("snap", "Snap", "Snaps", ("snap", "snaps", "fingersnap"), ()),
    Category(
        "hat_closed",
        "Closed Hat",
        "Closed Hats",
        ("closedhat", "closedhihat", "hihat", "hat", "hats", "hihats"),
        ("ch", "chh", "hh"),
    ),
    Category(
        "hat_open",
        "Open Hat",
        "Open Hats",
        ("openhat", "openhihat", "open"),
        ("oh", "ohh"),
    ),
    Category(
        "cymbal",
        "Cymbal",
        "Cymbals",
        ("cymbal", "cymbals", "crash", "ride", "splash", "china"),
        ("cy", "cr", "rd"),
    ),
    Category("tom", "Tom", "Toms", ("tom", "toms", "floortom"), ("tm",)),
    Category(
        "percussion",
        "Percussion",
        "Percussion",
        (
            "perc",
            "percussion",
            "shaker",
            "tambourine",
            "tamb",
            "conga",
            "bongo",
            "cowbell",
            "clave",
            "woodblock",
            "triangle",
            "agogo",
            "cabasa",
            "guiro",
        ),
        ("pc",),
    ),
    # "808" is deliberately not a keyword here: on its own it names a machine,
    # not a category, and it appears just as often on that machine's kicks
    # (`BD_808_01`) as on its basses.
    Category("sub", "Sub", "Subs", ("sub", "subbass"), ()),
    Category(
        "bass",
        "Bass",
        "Basses",
        ("bass", "basses", "bassline", "reese", "wobble"),
        ("bs",),
    ),
    Category(
        "chord",
        "Chord",
        "Chords",
        ("chord", "chords", "harmony"),
        (),
    ),
    Category("stab", "Stab", "Stabs", ("stab", "stabs", "hit", "orchhit"), ()),
    Category("pluck", "Pluck", "Plucks", ("pluck", "plucks", "arp", "arps", "guitar"), ()),
    Category("lead", "Lead", "Leads", ("lead", "leads", "melody", "synth", "saw"), ()),
    Category("pad", "Pad", "Pads", ("pad", "pads", "atmos", "atmosphere", "drone", "string"), ()),
    Category("key", "Key", "Keys", ("key", "keys", "piano", "rhodes", "organ", "epiano"), ()),
    Category("vocal", "Vocal", "Vocals", ("vocal", "vocals", "vox", "voice", "acapella", "adlib"), ()),
    Category("riser", "Riser", "Risers", ("riser", "risers", "uplifter", "sweep", "rise"), ()),
    Category(
        "downlifter",
        "Downlifter",
        "Downlifters",
        ("downlifter", "downsweep", "fall", "drop"),
        (),
    ),
    Category("impact", "Impact", "Impacts", ("impact", "impacts", "boom", "slam", "hitfx"), ()),
    Category("fx", "FX", "FX", ("fx", "sfx", "effect", "effects", "noise", "texture", "foley"), ()),
    Category("drum_loop", "Drum Loop", "Drum Loops", ("drumloop", "beatloop", "breakbeat", "break"), ()),
    Category("music_loop", "Loop", "Loops", ("loop", "loops", "riff", "groove", "phrase"), ()),
)

BY_KEY: Dict[str, Category] = {category.key: category for category in CATEGORIES}

UNKNOWN = "unknown"
UNKNOWN_LABEL = "Unsorted"

# Filename evidence outweighs audio evidence, but never silences it: a file in
# a folder called "Kicks" that measures like a hat is worth a second look.
NAME_WEIGHT = 1.6
AUDIO_WEIGHT = 1.0

# Below this the guess is not worth showing; the sample is reported as unknown
# rather than forced into the least-bad category.
MIN_SCORE = 0.25


# ----------------------------------------------------------------------
# Filename evidence
# ----------------------------------------------------------------------

# Split on separators and at camelCase boundaries, so `OpenHat`, `open_hat`
# and `open-hat-01` all tokenise the same way.
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_SPLIT = re.compile(r"[^a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens from a filename or folder name."""
    spaced = _CAMEL.sub(" ", str(text))
    return [token for token in _SPLIT.split(spaced.lower()) if token]


def name_scores(filepath: str, folder_depth: int = 2) -> Dict[str, float]:
    """Category scores from a path's filename and its enclosing folders.

    The filename counts for more than the folder it sits in: a pack laid out
    as ``Drums/Kicks/`` and one laid out as ``Vol 3/WAV/`` are equally common,
    and only one of them is telling the truth about its contents.
    """
    path = Path(filepath)
    scores: Dict[str, float] = {}

    sources = [(tokenize(path.stem), 1.0)]
    for depth, parent in enumerate(path.parents[:folder_depth]):
        if parent.name:
            sources.append((tokenize(parent.name), 0.6 / (depth + 1)))

    for tokens, weight in sources:
        if not tokens:
            continue
        joined = "".join(tokens)
        for category in CATEGORIES:
            hit = 0.0
            if any(token in category.keywords for token in tokens):
                hit = 1.0
            elif any(keyword in joined for keyword in category.keywords if len(keyword) >= 5):
                # Runs-together names like `closedhat01` or `subbass`.
                hit = 0.85
            elif any(token in category.codes for token in tokens):
                # Two-letter codes are only trustworthy as whole tokens —
                # `BD_01` is a kick, but `bd` inside `bdaywav` is nothing.
                hit = 0.8
            if hit:
                scores[category.key] = max(scores.get(category.key, 0.0), hit * weight)

    return _disambiguate_names(scores)


def _disambiguate_names(scores: Dict[str, float]) -> Dict[str, float]:
    """Resolve keyword overlaps between categories that share vocabulary.

    ``open_hat`` matches both hat categories, ``sub_bass`` matches both bass
    categories, and ``drum_loop`` matches both loop categories. In each pair
    the more specific name wins outright rather than the two splitting the
    vote and losing to something else.
    """
    for specific, general in (
        ("hat_open", "hat_closed"),
        ("sub", "bass"),
        ("drum_loop", "music_loop"),
        ("impact", "kick"),
    ):
        if scores.get(specific, 0.0) >= scores.get(general, 0.0) > 0.0:
            scores[general] = scores[general] * 0.4
    return scores


# ----------------------------------------------------------------------
# Audio evidence
# ----------------------------------------------------------------------


def _ramp(value: float, low: float, high: float) -> float:
    """0 below ``low``, 1 above ``high``, linear between."""
    if high <= low:
        return 1.0 if value >= high else 0.0
    return float(np.clip((value - low) / (high - low), 0.0, 1.0))


def _window(value: float, low: float, high: float, softness: float = 0.25) -> float:
    """1 inside ``[low, high]``, tapering to 0 outside it."""
    if low <= value <= high:
        return 1.0
    span = max((high - low) * softness, 1e-6)
    distance = (low - value) if value < low else (value - high)
    return float(np.clip(1.0 - distance / span, 0.0, 1.0))


def _mean(values: Iterable[float]) -> float:
    values = [float(v) for v in values]
    return float(np.mean(values)) if values else 0.0


def audio_scores(descriptors: Dict[str, Any]) -> Dict[str, float]:
    """Category scores from measured audio, each rule a claim you can read.

    Scores are averages of independent tests, so a category needs to satisfy
    most of what defines it rather than one lucky threshold.
    """
    d = descriptors
    duration = float(d.get("duration") or 0.0)
    low = float(d.get("low_energy") or 0.0)
    mid = float(d.get("mid_energy") or 0.0)
    high = float(d.get("high_energy") or 0.0)
    bands = d.get("bands") or {}
    sub_band = float(bands.get("sub", 0.0))

    # Silence, or a file too short to have measurable shape. Many rules below
    # are partly made of absences — "not bright", "not sustained", "no pitch" —
    # which an empty measurement satisfies perfectly, so without this a
    # zero-length file confidently reads as whatever category asks for least.
    if duration <= 0.0 or (low + mid + high) <= 0.0:
        return {}

    attack = float(d.get("attack_time") or 0.0)
    sustain = float(d.get("sustain_ratio") or 0.0)
    centroid = float(d.get("temporal_centroid") or 0.0)
    hits = float(d.get("onset_count") or 1.0)
    percussive = float(d.get("percussive_ratio") or 0.5)
    f0 = float(d.get("f0_hz") or 0.0)
    support = float(d.get("harmonic_support") or 0.0)
    peaks = float(d.get("chroma_peaks") or 0.0)
    slope = float(d.get("brightness_slope") or 0.0)
    brightness = float(d.get("brightness_attack") or 0.0)

    # "Does this have a note in it" is the question that splits a sample
    # library in half, so it is worth naming once and reusing.
    #
    # Both halves are required, and multiplying is what requires them. Noise
    # with a resonance — which is every snare ever made — tracks a steady
    # pitch in every frame and so scores full confidence; what it does not
    # have is energy sitting on that pitch's harmonics.
    confidence = float(d.get("f0_confidence") or 0.0)
    tonal = support * (0.5 + 0.5 * confidence)
    unpitched = 1.0 - _ramp(tonal, 0.15, 0.45)
    # Melodic material lives above the kick drum. This is what keeps a
    # sine-swept 808 — genuinely pitched, genuinely harmonic — out of the
    # chord and pluck piles.
    not_low = 1.0 - _ramp(low, 0.4, 0.7)
    single_hit = 1.0 - _ramp(hits, 2.0, 5.0)
    looped = _ramp(hits, 3.0, 6.0) * _ramp(duration, 0.8, 1.6)

    scores: Dict[str, float] = {}

    scores["kick"] = single_hit * _mean(
        [
            _window(duration, 0.05, 1.0),
            _ramp(low, 0.35, 0.6),
            1.0 - _ramp(high, 0.15, 0.4),
            _window(f0, 20.0, 130.0) if f0 else 0.5,
            1.0 - _ramp(centroid, 0.35, 0.6),
        ]
    )

    scores["sub"] = single_hit * _mean(
        [
            _ramp(sub_band, 0.3, 0.6),
            _ramp(low, 0.7, 0.9),
            _ramp(duration, 0.2, 0.6),
            _ramp(tonal, 0.3, 0.6),
            _window(f0, 20.0, 80.0) if f0 else 0.3,
            _ramp(sustain, 0.3, 0.55),
        ]
    )

    scores["bass"] = single_hit * _mean(
        [
            _ramp(tonal, 0.25, 0.6),
            _window(f0, 30.0, 220.0) if f0 else 0.2,
            _ramp(low + 0.5 * mid, 0.4, 0.75),
            _ramp(duration, 0.2, 0.6),
            _ramp(sustain, 0.3, 0.6),
            1.0 - _ramp(high, 0.2, 0.5),
        ]
    )

    scores["snare"] = single_hit * _mean(
        [
            _window(duration, 0.05, 0.8),
            _ramp(high, 0.25, 0.6),
            # A snare has a body as well as a crack. This is the whole
            # difference between a snare and a hat, both of which are
            # otherwise short, bright, unpitched and percussive.
            _ramp(mid, 0.12, 0.35),
            unpitched,
            _ramp(percussive, 0.3, 0.7),
            1.0 - _ramp(low, 0.2, 0.45),
        ]
    )

    # A clap is a snare's spectrum with a smeared front: several transients
    # inside 30 ms, which reads as a slower attack and a noisier body.
    scores["clap"] = single_hit * _mean(
        [
            _window(duration, 0.08, 0.9),
            _ramp(high, 0.5, 0.8),
            unpitched,
            _ramp(attack, 0.008, 0.03),
            _ramp(percussive, 0.4, 0.8),
            1.0 - _ramp(low, 0.15, 0.35),
            # A clap's noise is band-limited around 2-4 kHz; a hat's runs to
            # the top of the spectrum, which is the measurable difference
            # between two sounds that are otherwise both bursts of noise.
            1.0 - _ramp(brightness, 5000.0, 9000.0),
        ]
    )

    scores["hat_closed"] = single_hit * _mean(
        [
            1.0 - _ramp(duration, 0.15, 0.4),
            _ramp(high, 0.6, 0.85),
            _ramp(brightness, 3500.0, 7000.0),
            1.0 - _ramp(mid, 0.12, 0.35),
            unpitched,
            _ramp(percussive, 0.4, 0.8),
        ]
    )

    scores["hat_open"] = single_hit * _mean(
        [
            _window(duration, 0.3, 1.6),
            _ramp(high, 0.6, 0.85),
            _ramp(brightness, 3500.0, 7000.0),
            unpitched,
        ]
    )

    scores["cymbal"] = single_hit * _mean(
        [
            _ramp(duration, 1.0, 2.5),
            _ramp(high, 0.5, 0.8),
            unpitched,
            _ramp(centroid, 0.25, 0.5),
        ]
    )

    scores["tom"] = single_hit * _mean(
        [
            _window(duration, 0.15, 1.2),
            _window(f0, 70.0, 400.0) if f0 else 0.2,
            _window(support, 0.2, 0.8),
            _ramp(low + mid, 0.6, 0.9),
            1.0 - _ramp(high, 0.2, 0.45),
            _ramp(percussive, 0.3, 0.7),
        ]
    )

    scores["percussion"] = single_hit * _mean(
        [
            _window(duration, 0.05, 1.0),
            _ramp(mid, 0.3, 0.7),
            _ramp(percussive, 0.4, 0.8),
            unpitched,
        ]
    )

    scores["chord"] = single_hit * _mean(
        [
            _ramp(peaks, 2.0, 3.0),
            _ramp(tonal, 0.25, 0.55),
            _ramp(duration, 0.3, 0.9),
            _window(f0, 80.0, 900.0) if f0 else 0.2,
            1.0 - _ramp(percussive, 0.4, 0.7),
            not_low,
            # A chord sample is struck. The same notes swelling in over half a
            # second is a pad, and that is the only thing separating them.
            1.0 - _ramp(attack, 0.12, 0.35),
        ]
    )

    scores["stab"] = single_hit * _mean(
        [
            _ramp(tonal, 0.25, 0.55),
            _window(duration, 0.1, 0.9),
            1.0 - _ramp(attack, 0.02, 0.08),
            1.0 - _ramp(sustain, 0.5, 0.8),
            _ramp(peaks, 1.0, 2.0),
            not_low,
        ]
    )

    scores["pluck"] = single_hit * _mean(
        [
            _ramp(tonal, 0.25, 0.55),
            _window(duration, 0.1, 1.2),
            1.0 - _ramp(attack, 0.02, 0.08),
            1.0 - _ramp(peaks, 2.0, 3.0),
            1.0 - _ramp(centroid, 0.35, 0.6),
            not_low,
        ]
    )

    scores["pad"] = _mean(
        [
            _ramp(duration, 1.2, 2.5),
            _ramp(attack, 0.08, 0.3),
            _ramp(sustain, 0.55, 0.8),
            _ramp(tonal, 0.2, 0.5),
            1.0 - _ramp(percussive, 0.3, 0.6),
        ]
    )

    scores["lead"] = single_hit * _mean(
        [
            _ramp(tonal, 0.3, 0.6),
            _ramp(duration, 0.4, 1.2),
            _ramp(sustain, 0.45, 0.75),
            1.0 - _ramp(peaks, 2.0, 3.0),
            _window(f0, 150.0, 1200.0) if f0 else 0.2,
            1.0 - _ramp(attack, 0.08, 0.3),
            not_low,
        ]
    )

    # A riser is defined by going somewhere: brightness climbing through a
    # long, late-weighted, non-percussive sound.
    scores["riser"] = _mean(
        [
            _ramp(duration, 1.0, 3.0),
            _ramp(slope, 200.0, 2000.0),
            _ramp(centroid, 0.45, 0.65),
            1.0 - _ramp(percussive, 0.4, 0.7),
        ]
    )

    scores["downlifter"] = _mean(
        [
            _ramp(duration, 1.0, 3.0),
            _ramp(-slope, 200.0, 2000.0),
            _ramp(centroid, 0.45, 0.65),
            1.0 - _ramp(percussive, 0.4, 0.7),
        ]
    )

    scores["impact"] = single_hit * _mean(
        [
            _ramp(duration, 0.5, 2.0),
            _ramp(low, 0.3, 0.6),
            unpitched,
            1.0 - _ramp(centroid, 0.3, 0.5),
        ]
    )

    scores["drum_loop"] = looped * _mean(
        [
            _ramp(percussive, 0.35, 0.7),
            unpitched,
        ]
    )

    scores["music_loop"] = looped * _mean(
        [
            _ramp(tonal, 0.2, 0.5),
            1.0 - _ramp(percussive, 0.4, 0.75),
        ]
    )

    return scores


# ----------------------------------------------------------------------
# Combining the two
# ----------------------------------------------------------------------


@dataclass
class Guess:
    """What a sample probably is, and why."""

    key: str
    label: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    evidence: List[str] = field(default_factory=list)

    @property
    def known(self) -> bool:
        return self.key != UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        ranked = sorted(self.scores.items(), key=lambda item: item[1], reverse=True)[:3]
        return {
            "category": self.key,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "alternatives": [
                {"category": key, "label": label_for(key), "score": round(score, 3)}
                for key, score in ranked
                if score > 0
            ],
        }


def label_for(key: str, plural: bool = False) -> str:
    category = BY_KEY.get(key)
    if not category:
        return UNKNOWN_LABEL
    return category.plural if plural else category.label


def classify(
    descriptors: Optional[Dict[str, Any]] = None, filepath: Optional[str] = None
) -> Guess:
    """Guess what one sample is, from its audio, its filename, or both."""
    audio = audio_scores(descriptors) if descriptors else {}
    named = name_scores(filepath) if filepath else {}

    # Only the sources that actually said something get to divide the total.
    # A well-measured kick called `Sample 04.wav` should not be penalised for
    # the half of the evidence that does not exist.
    divisor = 0.0
    if any(score > 0 for score in audio.values()):
        divisor += AUDIO_WEIGHT
    if any(score > 0 for score in named.values()):
        divisor += NAME_WEIGHT
    if divisor <= 0:
        return Guess(UNKNOWN, UNKNOWN_LABEL, 0.0, {}, [])

    scores = {
        key: (AUDIO_WEIGHT * audio.get(key, 0.0) + NAME_WEIGHT * named.get(key, 0.0)) / divisor
        for key in set(audio) | set(named)
    }

    ranked = sorted(scores.values(), reverse=True)
    best_key = max(scores, key=lambda key: scores[key])
    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else 0.0

    # Confidence is how well the winner fits, discounted by how close behind
    # the runner-up is. A sample that scores 0.9 for both "snare" and "clap"
    # is not a confident snare, and saying so is the point.
    confidence = float(best * (0.5 + 0.5 * (best - runner_up) / best)) if best > 0 else 0.0

    if best < MIN_SCORE:
        return Guess(UNKNOWN, UNKNOWN_LABEL, confidence, scores, [])

    evidence: List[str] = []
    if named.get(best_key):
        evidence.append("named like one")
    if audio.get(best_key, 0.0) >= 0.5:
        evidence.append(_audio_evidence(descriptors or {}))

    return Guess(best_key, label_for(best_key), confidence, scores, [e for e in evidence if e])


def _audio_evidence(descriptors: Dict[str, Any]) -> str:
    """A one-line reading of the measurements, for a human deciding."""
    duration = float(descriptors.get("duration") or 0.0)
    low = float(descriptors.get("low_energy") or 0.0)
    high = float(descriptors.get("high_energy") or 0.0)
    support = float(descriptors.get("harmonic_support") or 0.0)
    hits = float(descriptors.get("onset_count") or 1.0)

    parts = [f"{duration * 1000:.0f} ms" if duration < 1 else f"{duration:.1f} s"]
    if low > 0.5:
        parts.append("low-heavy")
    elif high > 0.5:
        parts.append("high-heavy")
    if support >= 0.3:
        parts.append("pitched")
    else:
        parts.append("unpitched")
    if hits >= 4:
        parts.append(f"{hits:.0f} hits")
    return ", ".join(parts)


def classify_vector(
    vector: np.ndarray,
    filepath: Optional[str] = None,
    layout: Optional[FeatureLayout] = None,
) -> Guess:
    """Classify from a stored feature vector extracted under the sample profile."""
    return classify(describe_sample_vector(vector, layout), filepath)


# ----------------------------------------------------------------------
# Groups of samples
# ----------------------------------------------------------------------


def summarize(guesses: Sequence[Guess]) -> Dict[str, Any]:
    """What a pile of samples mostly is, weighted by how sure each guess was."""
    weighted: Counter = Counter()
    for guess in guesses:
        if guess.known:
            weighted[guess.key] += guess.confidence

    counts = Counter(guess.key for guess in guesses if guess.known)
    total = len(guesses)
    if not weighted:
        return {
            "category": UNKNOWN,
            "label": UNKNOWN_LABEL,
            "share": 0.0,
            "breakdown": [],
            "classified": 0,
            "total": total,
        }

    best_key = weighted.most_common(1)[0][0]
    return {
        "category": best_key,
        "label": label_for(best_key),
        "plural": label_for(best_key, plural=True),
        "share": float(counts[best_key] / total) if total else 0.0,
        "breakdown": [
            {"category": key, "label": label_for(key), "count": count}
            for key, count in counts.most_common(6)
        ],
        "classified": int(sum(counts.values())),
        "total": total,
    }


def classify_many(
    vectors: Optional[np.ndarray] = None,
    filepaths: Optional[Sequence[str]] = None,
    layout: Optional[FeatureLayout] = None,
) -> List[Guess]:
    """Classify a batch, from vectors, filenames, or both aligned by position."""
    if vectors is None and not filepaths:
        return []

    if vectors is None:
        return [classify(None, path) for path in filepaths or []]

    matrix = np.asarray(vectors, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    paths = list(filepaths or [])
    return [
        classify_vector(row, paths[index] if index < len(paths) else None, layout)
        for index, row in enumerate(matrix)
    ]


def suggest_group_name(
    vectors: Optional[np.ndarray],
    tracks: Sequence[Dict[str, Any]],
    fallback: str = "Samples",
    layout: Optional[FeatureLayout] = None,
) -> str:
    """A name for a pile of samples: what most of them are.

    Classifying every member and taking the weighted majority beats
    classifying the pile's centroid — the average of a kick and a hat is not a
    sample, and the rules above would read that average as something it is not.
    """
    filepaths = [str(track.get("filepath") or track.get("filename") or "") for track in tracks]
    guesses = classify_many(vectors, filepaths, layout)
    if not guesses:
        return fallback

    summary = summarize(guesses)
    if summary["category"] == UNKNOWN:
        return fallback

    name = summary.get("plural") or summary["label"]
    # A pile that is only half kicks is still best named "Kicks", but one that
    # is a quarter of everything should not claim to be any of them.
    return name if summary["share"] >= 0.35 else fallback


def group_summary(
    vectors: Optional[np.ndarray],
    tracks: Sequence[Dict[str, Any]],
    centroid: Optional[np.ndarray] = None,
    layout: Optional[FeatureLayout] = None,
) -> Dict[str, Any]:
    """Stats shown beside a candidate group of samples during review."""
    filepaths = [str(track.get("filepath") or track.get("filename") or "") for track in tracks]
    guesses = classify_many(vectors, filepaths, layout)
    summary = summarize(guesses)

    stats: Dict[str, Any] = {
        "kind": "sample",
        "category": summary["category"],
        "category_label": summary["label"],
        "category_share": round(summary["share"], 3),
        "breakdown": summary["breakdown"],
    }

    if centroid is not None:
        described = describe_sample_vector(np.asarray(centroid, dtype=float), layout)
        stats.update(
            {
                "median_duration": round(described["duration"], 3),
                "attack_ms": round(described["attack_time"] * 1000, 1),
                "brightness_hz": round(described["brightness_attack"], 1),
                "low_energy": round(described["low_energy"], 3),
                "high_energy": round(described["high_energy"], 3),
                "pitched": bool(described["harmonic_support"] >= 0.3),
                "f0_hz": round(described["f0_hz"], 1) if described["f0_hz"] else None,
                "onset_count": round(described["onset_count"], 1),
            }
        )
    return stats


def describe_for_display(vector: np.ndarray, layout: Optional[FeatureLayout] = None) -> Dict[str, Any]:
    """Everything worth showing about one sample, audio-side."""
    described = describe_vector(np.asarray(vector, dtype=float), layout)
    return {
        "duration": round(described.get("duration", 0.0), 3),
        "attack_ms": round(described.get("attack_time", 0.0) * 1000, 1),
        "decay_ms": round(described.get("decay_time", 0.0) * 1000, 1),
        "sustain_ratio": round(described.get("sustain_ratio", 0.0), 3),
        "brightness_hz": round(described.get("brightness_attack", 0.0), 1),
        "low_energy": round(described.get("low_energy", 0.0), 3),
        "mid_energy": round(described.get("mid_energy", 0.0), 3),
        "high_energy": round(described.get("high_energy", 0.0), 3),
        "f0_hz": round(described.get("f0_hz", 0.0), 1) or None,
        "pitched": bool(described.get("harmonic_support", 0.0) >= 0.3),
        "percussive_ratio": round(described.get("percussive_ratio", 0.0), 3),
        "onset_count": round(described.get("onset_count", 0.0), 1),
    }


def taxonomy() -> List[Dict[str, str]]:
    """The category list, for the API and the CLI."""
    return [
        {"key": category.key, "label": category.label, "plural": category.plural}
        for category in CATEGORIES
    ]
