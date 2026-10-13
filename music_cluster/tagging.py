"""Auto-tagging: genre, mood, energy, danceability, key and beat type.

Sorting answers "which of *my* folders does this belong in?". Tags answer a
different question — "what is this?" — in the words a DJ already uses to
talk about a record: a half-time 172, a swung 2-step, a dark rolling techno
track at energy 8, a baile funk edit. They sit alongside your groups, never
instead of them, and nothing here moves a file.

Every tag is a *rule over measured quantities*, in the same spirit as
:mod:`music_cluster.samples`. There is no trained model. That is a
limitation — a rule cannot hear that a track is "deep" the way a person can —
but it means every tag can be explained, argued with and edited in one
place, and a tag that is wrong is wrong for a reason you can read.

What each tag is built from:

**Beat type** — where the kick, snare and hats land in the bar, measured by
:mod:`music_cluster.rhythm`. A four-to-the-floor has a kick on every beat; a
half-time groove has one snare per bar, on three; a 2-step drops the kick on
three and syncopates the second one; breaks are busy with ghost snares;
baile funk's low end plays the 3+3+2 tamborzão; dembow puts that 3+3+2 on
the snare over a straight kick.

**Genre** — tempo and beat type first (a 2-step at 132 is garage, at 172
it is drum & bass), then timbre and energy to split neighbours like house
and tech house. The file's own genre tag, when it names one of these, is
counted as evidence.

**Energy** (1-10) — loudness, onset density, brightness and tempo.

**Danceability** (0-1) — how clear and steady the pulse is, whether there
is a kick to follow, and whether the tempo is one people dance at.

**Mood** — mode (major or minor), brightness, energy and groove, read as
one or two of: dark, euphoric, melancholic, aggressive, chill, hypnotic,
groovy.

**Key** — Krumhansl-Schmuckler key finding on the chroma, reported with its
Camelot code for harmonic mixing.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import rhythm


logger = logging.getLogger(__name__)

#: Bumped whenever a rule or a measurement changes, so stale tags can be found
#: and re-run without re-running everything.
TAGGER_VERSION = "1"


# ----------------------------------------------------------------------
# Vocabulary
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class Term:
    key: str
    label: str
    description: str = ""


BEATS: Tuple[Term, ...] = (
    Term("four_on_floor", "Four to the floor", "A kick on every beat: house, techno, trance."),
    Term("half_time", "Half-time", "One snare per bar, on three: dubstep, halftime DnB, trap."),
    Term("two_step", "2-step", "Kick on one, skipped on three, backbeat snare: garage, DnB."),
    Term("breakbeat", "Breaks", "Syncopated kicks and busy ghost snares: breaks, jungle."),
    Term("baile_funk", "Baile funk", "Low end in the 3+3+2 tamborzão: funk carioca."),
    Term("dembow", "Dembow", "Straight kick, 3+3+2 snare: reggaeton, dembow."),
    Term("backbeat", "Backbeat", "Snare on two and four at a slow tempo: hip hop, boom bap."),
    Term("beatless", "Beatless", "No steady pulse to count: ambient, intros, beatless edits."),
)

GENRES: Tuple[Term, ...] = (
    Term("deep_house", "Deep House"),
    Term("house", "House"),
    Term("tech_house", "Tech House"),
    Term("techno", "Techno"),
    Term("hard_techno", "Hard Techno"),
    Term("trance", "Trance"),
    Term("uk_garage", "UK Garage"),
    Term("breaks", "Breaks"),
    Term("drum_and_bass", "Drum & Bass"),
    Term("jungle", "Jungle"),
    Term("dubstep", "Dubstep"),
    Term("trap", "Trap"),
    Term("baile_funk", "Baile Funk"),
    Term("reggaeton", "Reggaeton"),
    Term("hip_hop", "Hip Hop"),
    Term("downtempo", "Downtempo"),
    Term("ambient", "Ambient"),
)

MOODS: Tuple[Term, ...] = (
    Term("dark", "Dark", "Minor, low and heavy."),
    Term("euphoric", "Euphoric", "Major, bright and driving."),
    Term("melancholic", "Melancholic", "Minor and tonal, without the push."),
    Term("aggressive", "Aggressive", "Loud, bright, noisy and fast."),
    Term("chill", "Chill", "Low energy and unhurried."),
    Term("hypnotic", "Hypnotic", "A locked, repetitive groove with little harmony."),
    Term("groovy", "Groovy", "Swung or syncopated, in the pocket."),
)

ENERGY_LABELS = ((0.0, "Low"), (0.4, "Medium"), (0.7, "High"))

_BY_KIND = {"beat": BEATS, "genre": GENRES, "mood": MOODS}


def label_for(kind: str, key: Optional[str]) -> Optional[str]:
    """The display label for a tag key, or None for an unknown one."""
    if not key:
        return None
    for term in _BY_KIND.get(kind, ()):
        if term.key == key:
            return term.label
    return None


def vocabulary() -> Dict[str, List[Dict[str, str]]]:
    """Every tag the tagger can emit, for the API, CLI and UI filters."""
    return {
        kind: [{"key": t.key, "label": t.label, "description": t.description} for t in terms]
        for kind, terms in (("beats", BEATS), ("genres", GENRES), ("moods", MOODS))
    }


# ----------------------------------------------------------------------
# Measuring
# ----------------------------------------------------------------------

#: Tagging measures at this rate whatever the file's own rate is.
TAGGING_SAMPLE_RATE = 22050

PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Krumhansl-Kessler key profiles.
_MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Camelot wheel numbers, indexed by pitch class of the tonic.
_CAMELOT_MAJOR = (8, 3, 10, 5, 12, 7, 2, 9, 4, 11, 6, 1)
_CAMELOT_MINOR = (5, 12, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10)


def estimate_key(chroma: np.ndarray) -> Dict[str, Any]:
    """Key, mode and Camelot code from a 12-bin mean chroma vector."""
    chroma = np.asarray(chroma, dtype=float)
    if chroma.size != 12 or float(np.std(chroma)) < 1e-9:
        return {"key": None, "mode": None, "camelot": None, "key_strength": 0.0}

    best: Tuple[float, int, str] = (-2.0, 0, "major")
    scores = []
    for tonic in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            r = float(np.corrcoef(chroma, np.roll(profile, tonic))[0, 1])
            scores.append(r)
            if r > best[0]:
                best = (r, tonic, mode)

    strength, tonic, mode = best
    number = (_CAMELOT_MAJOR if mode == "major" else _CAMELOT_MINOR)[tonic]
    # How far ahead of the next-best key the winner is: 0 for noise that fits
    # every key equally badly, rising towards 1 for an unambiguous key.
    ranked = sorted(scores, reverse=True)
    clarity = float(np.clip((ranked[0] - ranked[1]) * 5.0 + max(ranked[0], 0.0) * 0.5, 0.0, 1.0))
    return {
        "key": f"{PITCH_CLASSES[tonic]} {mode}",
        "mode": mode,
        "camelot": f"{number}{'B' if mode == 'major' else 'A'}",
        "key_strength": round(clarity, 4),
        "key_correlation": round(strength, 4),
    }


def describe_audio(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Everything the tag rules read, measured from one mono signal.

    Returned as a plain dict so it can be stored, shown, and fed back into
    :func:`tag` without audio — which is how the rules are tested.
    """
    import librosa

    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return {}
    # Nothing a tag reads lives above 11 kHz, and halving the rate halves the
    # cost of every transform below — which matters, because this runs on
    # every track as it is analysed.
    if sr > TAGGING_SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=TAGGING_SAMPLE_RATE)
        sr = TAGGING_SAMPLE_RATE

    hop = max(256, round(sr * 0.023))
    n_fft = 2048 if sr <= 24000 else 4096
    magnitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    power = np.square(magnitude)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    frame_rms = librosa.feature.rms(S=magnitude, frame_length=n_fft)[0]
    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-6))
    # Loudness of the sounding part of the track: a breakdown or a silent
    # tail should not make a banger read as quiet.
    loud = frame_db[frame_db >= np.percentile(frame_db, 20)] if frame_db.size else frame_db
    total_power = float(power.sum()) or 1.0

    centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    flatness = librosa.feature.spectral_flatness(S=magnitude)[0]
    # Tuning is fixed at concert pitch: estimating it costs more than every
    # other measurement here combined, and a key finder that folds to twelve
    # pitch classes does not need it.
    chroma = librosa.feature.chroma_stft(S=power, sr=sr, tuning=0.0)

    # Nothing to describe: silence, or a file that decoded to near-silence.
    if float(np.max(np.abs(y))) < 1e-3:
        return {}

    described: Dict[str, Any] = {
        "duration": round(y.size / float(sr), 3),
        "loudness_db": round(float(np.mean(loud)) if loud.size else -80.0, 3),
        "loudness_range_db": round(
            float(np.percentile(frame_db, 95) - np.percentile(frame_db, 10))
            if frame_db.size
            else 0.0,
            3,
        ),
        "brightness_hz": round(float(np.mean(centroid)) if centroid.size else 0.0, 2),
        "flatness": round(float(np.mean(flatness)) if flatness.size else 0.0, 5),
        "low_share": round(float(power[freqs < 150.0].sum()) / total_power, 5),
        "high_share": round(float(power[freqs >= 6000.0].sum()) / total_power, 5),
        # How much the harmony moves: frame-to-frame chroma change.
        "harmonic_change": round(
            float(np.mean(np.abs(np.diff(chroma, axis=1)))) if chroma.shape[1] > 1 else 0.0, 5
        ),
        "rhythm": rhythm.analyze(y, sr).to_dict(),
    }
    mean_chroma = chroma.mean(axis=1)
    key = estimate_key(mean_chroma)
    # Noise and drums fit some key by chance. A real key needs some pitch
    # classes to stand clearly above the others.
    contrast = float(np.std(mean_chroma) / max(float(np.mean(mean_chroma)), 1e-9))
    key["key_strength"] = round(float(key["key_strength"]) * _ramp(contrast, 0.15, 0.35), 4)
    key["chroma_contrast"] = round(contrast, 4)
    described.update(key)
    return described


# ----------------------------------------------------------------------
# Rules
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


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def _hits(row: np.ndarray, threshold: float = 0.35) -> np.ndarray:
    return row >= threshold


@dataclass
class _Groove:
    """One bar on the sixteenth grid, rotated so a given beat is beat one."""

    kick: np.ndarray
    snare: np.ndarray
    hat: np.ndarray

    def rotated(self, beats: int) -> "_Groove":
        shift = -4 * beats
        return _Groove(
            np.roll(self.kick, shift), np.roll(self.snare, shift), np.roll(self.hat, shift)
        )


QUARTERS = [0, 4, 8, 12]
OFF_QUARTERS = [i for i in range(16) if i not in QUARTERS]
# 3+3+2, twice per bar: the tresillo that tamborzão and dembow are built on.
TRESILLO = [0, 3, 6, 8, 11, 14]
TRESILLO_OFFBEATS = [3, 6, 11, 14]


def _beat_rules(g: _Groove, bpm: float) -> Dict[str, float]:
    """Beat-type scores for one reading of the bar. Each rule is readable."""
    k, s = g.kick, g.snare
    kick_on_quarters = float(np.min(k[QUARTERS]))
    kick_off_quarters = float(np.mean(k[OFF_QUARTERS]))
    backbeat = float(np.mean(s[[4, 12]]))
    ghost_snares = int(np.count_nonzero(_hits(s[[i for i in range(16) if i not in (4, 12)]])))
    syncopated_kicks = int(np.count_nonzero(_hits(k[OFF_QUARTERS])))
    snare_hits = int(np.count_nonzero(_hits(s)))

    scores: Dict[str, float] = {}

    scores["four_on_floor"] = _mean(
        [
            _ramp(kick_on_quarters, 0.3, 0.6),
            1.0 - _ramp(kick_off_quarters, 0.12, 0.35),
            # Whatever the snare does, it does not sit on three alone...
            1.0 - _ramp(float(s[8]) - backbeat, 0.2, 0.5),
            # ...or answer the kick on the odd sixteenths, which is what a
            # dembow's 3+3+2 snare does. Odd sixteenths stay odd whichever
            # beat is taken as one, so this holds under every rotation.
            1.0 - _ramp(float(np.mean(np.sort(s[1::2])[-2:])), 0.35, 0.65),
        ]
    )

    scores["half_time"] = _mean(
        [
            _ramp(float(s[8]), 0.45, 0.75),
            1.0 - _ramp(backbeat, 0.2, 0.45),
            _ramp(float(k[0]), 0.4, 0.7),
            1.0 - _ramp(kick_on_quarters, 0.3, 0.6),
            _window(bpm, 125.0, 180.0, softness=0.2),
        ]
    )

    scores["two_step"] = _mean(
        [
            _ramp(backbeat, 0.45, 0.75),
            _ramp(float(k[0]), 0.4, 0.7),
            # The whole point of a 2-step: no kick on three...
            1.0 - _ramp(float(k[8]), 0.2, 0.45),
            # ...and a second kick pushed off the beat instead.
            _ramp(float(np.max(k[[6, 7, 9, 10, 11]])), 0.35, 0.6),
            # Sparse: a snare with ghosts all over it is a break.
            1.0 - _ramp(float(ghost_snares), 1.0, 3.0),
            _window(bpm, 125.0, 180.0, softness=0.2),
        ]
    )

    scores["breakbeat"] = _mean(
        [
            _ramp(float(np.max(s[[4, 12]])), 0.4, 0.7),
            _ramp(float(ghost_snares), 1.0, 3.0),
            _ramp(float(syncopated_kicks), 1.0, 2.0),
            1.0 - _ramp(kick_on_quarters, 0.3, 0.6),
            _window(bpm, 110.0, 180.0, softness=0.2),
        ]
    )

    tresillo_low = float(np.mean(k[TRESILLO]))
    scores["baile_funk"] = _mean(
        [
            _ramp(tresillo_low, 0.45, 0.75),
            _ramp(float(np.mean(k[TRESILLO_OFFBEATS])), 0.4, 0.7),
            # The low end skips the quarters the tresillo does not land on.
            1.0 - _ramp(float(np.mean(k[[4, 12]])), 0.25, 0.5),
            _window(bpm, 120.0, 155.0, softness=0.2),
        ]
    )

    scores["dembow"] = _mean(
        [
            _ramp(kick_on_quarters, 0.3, 0.6),
            _ramp(float(np.mean(s[TRESILLO_OFFBEATS])), 0.45, 0.75),
            1.0 - _ramp(backbeat, 0.25, 0.5),
            _window(bpm, 85.0, 120.0, softness=0.2),
        ]
    )

    scores["backbeat"] = _mean(
        [
            _ramp(backbeat, 0.45, 0.75),
            _ramp(float(k[0]), 0.4, 0.7),
            1.0 - _ramp(float(snare_hits), 3.0, 5.0),
            _window(bpm, 60.0, 110.0, softness=0.2),
        ]
    )

    return scores


# Reading the bar from a beat other than the one the analysis chose costs a
# little: the chosen downbeat is usually right, and ties should go to it.
_ROTATION_PENALTY = 0.93


def beat_scores(profile: Dict[str, Any]) -> Dict[str, float]:
    """Scores for every beat type, from a stored rhythm profile.

    Where beat one falls is the least certain part of the measurement — a
    four-to-the-floor gives no clue, and a dembow's kick is the same on every
    beat — so every rule is tried from all four beats and scored on its best
    reading.
    """
    measured = rhythm.RhythmProfile.from_dict(profile)
    clarity = measured.pulse_clarity
    # Is there anything drum-like at all? Without this, the gentle swell of a
    # pad is folded into bars and read as a very tidy four-to-the-floor.
    drums = _ramp(measured.transient_strength, 0.01, 0.03)
    beatless = max(
        1.0 - drums,
        _mean([1.0 - _ramp(clarity, 0.1, 0.35), 1.0 - _ramp(measured.onset_rate, 1.0, 2.5)]),
    )
    if not measured.has_pattern or drums <= 0.0:
        return {"beatless": max(beatless, 0.5)}

    groove = _Groove(
        rhythm.sixteenths(measured.pattern["kick"]),
        rhythm.sixteenths(measured.pattern["snare"]),
        rhythm.sixteenths(measured.pattern["hat"]),
    )
    scores: Dict[str, float] = {}
    for turn in range(4):
        penalty = 1.0 if turn == 0 else _ROTATION_PENALTY
        for key, value in _beat_rules(groove.rotated(turn), measured.bpm).items():
            scores[key] = max(scores.get(key, 0.0), value * penalty)

    # A pattern only means something if there is a pulse carrying it.
    pulse = _ramp(clarity, 0.1, 0.3)
    scores = {key: value * (0.4 + 0.6 * pulse) * drums for key, value in scores.items()}
    scores["beatless"] = beatless
    return scores


def energy_score(d: Dict[str, Any]) -> float:
    """0-1: how hard a track pushes. Loudness, density, brightness, tempo."""
    groove = d.get("rhythm") or {}
    return _mean(
        [
            _ramp(float(d.get("loudness_db", -60.0)), -26.0, -9.0),
            _ramp(float(groove.get("onset_rate", 0.0)), 1.5, 6.0),
            _ramp(float(d.get("brightness_hz", 0.0)), 900.0, 3200.0),
            _ramp(float(groove.get("bpm", 0.0)), 85.0, 160.0),
            _ramp(float(d.get("high_share", 0.0)), 0.01, 0.08),
        ]
    )


def danceability_score(d: Dict[str, Any], beat: Optional[str] = None) -> float:
    """0-1: how easy this is to move to."""
    groove = d.get("rhythm") or {}
    bpm = float(groove.get("bpm", 0.0))
    kick = float((groove.get("band_activity") or {}).get("kick", 0.0))
    return _mean(
        [
            _ramp(float(groove.get("pulse_clarity", 0.0)), 0.15, 0.6),
            _ramp(float(groove.get("beat_regularity", 0.0)), 0.4, 0.85),
            # Half-time 172s are danced at 86; the window covers both.
            max(_window(bpm, 100.0, 135.0, 0.5), _window(bpm / 2.0, 80.0, 95.0, 0.3)),
            _ramp(kick + float(d.get("low_share", 0.0)), 0.2, 0.6),
        ]
    ) * (
        0.0
        if beat in (None, "beatless")
        else _ramp(float(groove.get("transient_strength", 0.0)), 0.01, 0.03)
    )


def mood_scores(d: Dict[str, Any], energy: float, beat: Optional[str] = None) -> Dict[str, float]:
    """Scores for every mood. A track usually earns one or two."""
    groove = d.get("rhythm") or {}
    minor = 1.0 if d.get("mode") == "minor" else 0.0
    tonal = _ramp(float(d.get("key_strength", 0.0)), 0.1, 0.5)
    brightness = float(d.get("brightness_hz", 0.0))
    bright = _ramp(brightness, 1500.0, 3200.0)
    low = _ramp(float(d.get("low_share", 0.0)), 0.25, 0.55)
    flat = _ramp(float(d.get("flatness", 0.0)), 0.02, 0.12)
    swing = float(groove.get("swing", 0.0))
    bpm = float(groove.get("bpm", 0.0))
    harmonic_change = float(d.get("harmonic_change", 0.0))

    # Mode only counts as far as there is a key to have a mode: a drum loop
    # "in D minor" is a chroma accident, not a mood.
    minor_evidence = minor * tonal
    major_evidence = (1.0 - minor) * tonal
    has_beat = 0.0 if beat in (None, "beatless") else 1.0

    return {
        "dark": _mean([0.5 * minor_evidence + 0.5 * (1.0 - bright), low, _ramp(energy, 0.3, 0.6)]),
        "euphoric": _mean([major_evidence, bright, _ramp(energy, 0.5, 0.8)]),
        "melancholic": minor_evidence * _mean([1.0 - _ramp(energy, 0.4, 0.7), 1.0 - low, 1.0]),
        "aggressive": _mean([_ramp(energy, 0.65, 0.9), bright, flat, _ramp(bpm, 125.0, 150.0)]),
        "chill": _mean([1.0 - _ramp(energy, 0.25, 0.5), 1.0 - _ramp(bpm, 95.0, 125.0), 1.0 - flat]),
        "hypnotic": has_beat
        * _mean(
            [
                _ramp(float(groove.get("beat_regularity", 0.0)), 0.6, 0.9),
                1.0 - _ramp(harmonic_change, 0.03, 0.08),
                1.0 - tonal,
                1.0 if beat == "four_on_floor" else 0.4,
            ]
        ),
        # Groove needs something off the grid — swing, or a beat built on
        # syncopation. A metronomic four-to-the-floor can be many things, but
        # it is not this.
        "groovy": has_beat
        * max(
            _ramp(swing, 0.25, 0.6),
            1.0 if beat in ("two_step", "breakbeat", "baile_funk", "dembow") else 0.0,
        )
        * _mean(
            [
                _ramp(float(groove.get("pulse_clarity", 0.0)), 0.2, 0.5),
                _window(bpm, 95.0, 135.0, 0.4),
            ]
        ),
    }


# Keywords in a file's own genre tag that vote for one of ours.
_GENRE_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "deep_house": ("deep house",),
    "tech_house": ("tech house",),
    "house": ("house",),
    "hard_techno": ("hard techno", "hardgroove"),
    "techno": ("techno",),
    "trance": ("trance",),
    "uk_garage": ("garage", "ukg", "2-step", "2 step", "speed garage"),
    "breaks": ("breaks", "breakbeat", "nu skool"),
    "drum_and_bass": ("drum & bass", "drum and bass", "drum n bass", "dnb", "d&b", "liquid"),
    "jungle": ("jungle",),
    "dubstep": ("dubstep", "riddim", "140"),
    "trap": ("trap",),
    "baile_funk": ("baile", "funk carioca", "brazilian funk", "funk mandelao", "mandelão"),
    "reggaeton": ("reggaeton", "dembow", "perreo", "latin urban"),
    "hip_hop": ("hip hop", "hip-hop", "rap", "boom bap"),
    "downtempo": ("downtempo", "chillout", "trip hop", "lo-fi", "lofi"),
    "ambient": ("ambient", "drone"),
}


def genre_from_tag(tag: Optional[str]) -> Optional[str]:
    """Map a free-text genre tag onto our vocabulary, most specific first."""
    if not tag:
        return None
    text = str(tag).lower()
    for key, words in _GENRE_KEYWORDS.items():
        if any(word in text for word in words):
            return key
    return None


def genre_scores(
    d: Dict[str, Any],
    beat: Optional[str],
    energy: float,
    moods: Dict[str, float],
    tagged: Optional[str] = None,
) -> Dict[str, float]:
    """Scores for every genre: tempo and beat first, then character."""
    groove = d.get("rhythm") or {}
    bpm = float(groove.get("bpm", 0.0))
    bright = _ramp(float(d.get("brightness_hz", 0.0)), 1500.0, 3200.0)
    hat_busy = _ramp(float((groove.get("band_activity") or {}).get("hat", 0.0)), 0.25, 0.45)
    low = _ramp(float(d.get("low_share", 0.0)), 0.25, 0.55)
    swing = _ramp(float(groove.get("swing", 0.0)), 0.2, 0.5)

    def is_(name: str) -> float:
        return 1.0 if beat == name else 0.0

    def rule(gate: float, tempo: float, character: Sequence[float] = ()) -> float:
        # Beat type and tempo are gates, not votes: a 95 BPM track is not
        # house however housey it sounds. Character only ranks what is left.
        return gate * tempo * (_mean(character) if len(character) else 1.0)

    four = is_("four_on_floor")
    key_strength = _ramp(float(d.get("key_strength", 0.0)), 0.2, 0.5)
    scores = {
        "deep_house": rule(
            four,
            _window(bpm, 115, 124),
            [1.0 - _ramp(energy, 0.45, 0.7), 0.5 + 0.5 * moods["dark"], 1.0 - bright],
        ),
        "house": rule(four, _window(bpm, 118, 128), [_ramp(energy, 0.3, 0.55), 0.7 + 0.3 * swing]),
        "tech_house": rule(
            four, _window(bpm, 123, 130), [_ramp(energy, 0.45, 0.7), 1.0 - bright, low]
        ),
        "techno": rule(
            four, _window(bpm, 126, 142), [_ramp(energy, 0.5, 0.75), 0.5 + 0.5 * moods["hypnotic"]]
        ),
        "hard_techno": rule(four, _window(bpm, 142, 165), [_ramp(energy, 0.6, 0.85)]),
        "trance": rule(
            four,
            _window(bpm, 130, 142),
            [moods["euphoric"], key_strength, _ramp(energy, 0.5, 0.75)],
        ),
        "uk_garage": rule(is_("two_step"), _window(bpm, 126, 140), [0.6 + 0.4 * swing]),
        "breaks": rule(is_("breakbeat"), _window(bpm, 115, 145)),
        "drum_and_bass": rule(
            max(is_("two_step"), is_("half_time"), 0.8 * is_("breakbeat")),
            _window(bpm, 160, 180),
        ),
        "jungle": rule(is_("breakbeat"), _window(bpm, 155, 180)),
        "dubstep": rule(is_("half_time"), _window(bpm, 135, 150), [1.0 - 0.5 * hat_busy, low]),
        "trap": rule(is_("half_time"), _window(bpm, 130, 160), [hat_busy, low]),
        "baile_funk": rule(is_("baile_funk"), _window(bpm, 120, 155)),
        "reggaeton": rule(is_("dembow"), _window(bpm, 85, 110)),
        "hip_hop": rule(is_("backbeat"), _window(bpm, 70, 100)),
        "downtempo": rule(
            max(is_("backbeat"), 0.6 * is_("beatless")),
            _window(bpm, 60, 105),
            [1.0 - _ramp(energy, 0.35, 0.6)],
        ),
        "ambient": rule(is_("beatless"), 1.0, [1.0 - _ramp(energy, 0.2, 0.45)]),
    }

    # The file's own tag is the DJ's (or the label's) vocabulary. It is not
    # trusted blindly — mistagged files are everywhere — but it tips a close
    # call and rescues a track whose beat reading failed.
    from_tag = genre_from_tag(tagged)
    if from_tag:
        scores[from_tag] = min(1.0, 0.4 + 0.6 * scores.get(from_tag, 0.0) + 0.25)
    return scores


# ----------------------------------------------------------------------
# Putting it together
# ----------------------------------------------------------------------

# Below these a tag is not worth showing; the track is reported without one
# rather than forced into the least-bad option.
MIN_BEAT_SCORE = 0.45
MIN_GENRE_SCORE = 0.45
MIN_MOOD_SCORE = 0.55
MIN_KEY_STRENGTH = 0.15
MAX_MOODS = 2
OPPOSED_MOODS = {
    frozenset(pair)
    for pair in (
        ("dark", "euphoric"),
        ("melancholic", "euphoric"),
        ("chill", "aggressive"),
        ("chill", "euphoric"),
    )
}


@dataclass
class TrackTags:
    """Every tag for one track, plus the scores and measurements behind them."""

    genre: Optional[str] = None
    beat: Optional[str] = None
    moods: List[str] = field(default_factory=list)
    energy: float = 0.0
    danceability: float = 0.0
    bpm: Optional[float] = None
    key: Optional[str] = None
    camelot: Optional[str] = None
    swung: bool = False
    confidence: Dict[str, float] = field(default_factory=dict)
    scores: Dict[str, Dict[str, float]] = field(default_factory=dict)
    descriptors: Dict[str, Any] = field(default_factory=dict)
    version: str = TAGGER_VERSION

    @property
    def energy_level(self) -> int:
        """Energy on the 1-10 scale DJs already use for it."""
        return int(np.clip(round(1 + 9 * self.energy), 1, 10))

    @property
    def energy_label(self) -> str:
        label = ENERGY_LABELS[0][1]
        for threshold, name in ENERGY_LABELS:
            if self.energy >= threshold:
                label = name
        return label

    def labels(self) -> List[str]:
        """Short human-readable tags, most informative first."""
        out: List[str] = []
        if self.genre:
            out.append(label_for("genre", self.genre) or self.genre)
        if self.beat:
            beat = label_for("beat", self.beat) or self.beat
            out.append(f"Swung {beat.lower()}" if self.swung and self.beat != "beatless" else beat)
        out.extend(label_for("mood", mood) or mood for mood in self.moods)
        out.append(f"Energy {self.energy_level}")
        return out

    def to_dict(self, include_descriptors: bool = True) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "genre": self.genre,
            "genre_label": label_for("genre", self.genre),
            "beat": self.beat,
            "beat_label": label_for("beat", self.beat),
            "moods": list(self.moods),
            "mood_labels": [label_for("mood", mood) for mood in self.moods],
            "energy": round(self.energy, 3),
            "energy_level": self.energy_level,
            "energy_label": self.energy_label,
            "danceability": round(self.danceability, 3),
            "bpm": round(self.bpm, 1) if self.bpm else None,
            "key": self.key,
            "camelot": self.camelot,
            "swung": self.swung,
            "confidence": {k: round(v, 3) for k, v in self.confidence.items()},
            "labels": self.labels(),
            "version": self.version,
        }
        if include_descriptors:
            payload["scores"] = {
                kind: {k: round(v, 3) for k, v in sorted(s.items(), key=lambda i: -i[1])}
                for kind, s in self.scores.items()
            }
            payload["descriptors"] = self.descriptors
        return payload


def _pick(scores: Dict[str, float], minimum: float) -> Tuple[Optional[str], float]:
    """The winner and a confidence discounted by how close the runner-up is."""
    if not scores:
        return None, 0.0
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    best_key, best = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0.0
    confidence = float(best * (0.5 + 0.5 * (best - runner_up) / best)) if best > 0 else 0.0
    if best < minimum:
        return None, confidence
    return best_key, confidence


def tag(descriptors: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> TrackTags:
    """Every tag for one track, from :func:`describe_audio` output.

    Args:
        metadata: The track's own tags (``genre``, ``tag_bpm``); used as
            evidence, never copied through unexamined.
    """
    metadata = metadata or {}
    d = descriptors or {}
    if not d:
        return TrackTags()

    beats = beat_scores(d.get("rhythm") or {})
    beat, beat_confidence = _pick(beats, MIN_BEAT_SCORE)

    if beat == "beatless":
        # Whatever tempo and swing the tracker reported for a pad are noise;
        # nothing downstream should read them.
        d = {
            **d,
            "rhythm": {
                **(d.get("rhythm") or {}),
                "bpm": 0.0,
                "onset_rate": 0.0,
                "swing": 0.0,
                "pulse_clarity": 0.0,
            },
        }

    energy = energy_score(d)
    danceability = danceability_score(d, beat)
    moods = mood_scores(d, energy, beat)
    mood_keys: List[str] = []
    for key, value in sorted(moods.items(), key=lambda item: -item[1]):
        if value < MIN_MOOD_SCORE or len(mood_keys) >= MAX_MOODS:
            break
        # "Dark and euphoric" describes nothing; the stronger reading stands.
        if any(frozenset((key, kept)) in OPPOSED_MOODS for kept in mood_keys):
            continue
        mood_keys.append(key)

    genres = genre_scores(d, beat, energy, moods, metadata.get("genre"))
    genre, genre_confidence = _pick(genres, MIN_GENRE_SCORE)

    groove = d.get("rhythm") or {}
    bpm = float(groove.get("bpm") or 0.0) or None
    has_key = float(d.get("key_strength") or 0.0) >= MIN_KEY_STRENGTH
    return TrackTags(
        genre=genre,
        beat=beat,
        moods=mood_keys,
        energy=energy,
        danceability=danceability,
        bpm=bpm,
        key=d.get("key") if has_key else None,
        camelot=d.get("camelot") if has_key else None,
        swung=float(groove.get("swing", 0.0)) >= 0.35,
        confidence={
            "beat": beat_confidence,
            "genre": genre_confidence,
            "key": float(d.get("key_strength") or 0.0),
        },
        scores={"beat": beats, "genre": genres, "mood": moods},
        descriptors=descriptors,
    )


def tag_audio(
    y: np.ndarray, sr: int, metadata: Optional[Dict[str, Any]] = None
) -> Optional[TrackTags]:
    """Measure and tag a decoded signal. None if it could not be measured."""
    try:
        described = describe_audio(y, sr)
    except Exception as exc:
        logger.warning("Tagging failed: %s", exc)
        return None
    return tag(described, metadata) if described else None
