"""Auto-tag suites: synthetic audio with known answers, and annotated datasets.

The synthetic suites render audio whose beat type, tempo and key are right by
construction, then roughen it (a drummer's timing, a detuned kit, noise) to
find where each rule stops holding. They need no download, so they run in CI.

The dataset suites read GiantSteps annotations against the cached analysis
from :mod:`.analysis`.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..library import resolve_worker_count
from . import metrics
from .datasets import Item
from .drums import SAMPLE_RATE, render


# Tempos each groove is actually played at.
GROOVE_TEMPOS: Dict[str, Tuple[float, ...]] = {
    "four_on_floor": (122.0, 128.0, 140.0),
    "half_time": (140.0, 172.0),
    "two_step": (130.0, 136.0),
    "breakbeat": (128.0, 170.0),
    "baile_funk": (130.0,),
    "dembow": (95.0,),
    "backbeat": (85.0, 95.0),
}

# How each case is roughened. "clean" is what the integration tests assert on.
VARIANTS: Dict[str, Dict[str, Any]] = {
    "clean": {},
    "humanized": {"humanize_ms": 8.0, "velocity_jitter": 0.15},
    "noisy": {"noise": 0.15},
    "deep_kit_with_pad": {"tune": 0.8, "pad": 110.0},
    "rough": {"humanize_ms": 12.0, "velocity_jitter": 0.2, "noise": 0.2, "tune": 1.2},
}
QUICK_VARIANTS = ("clean", "rough")
# Each case is rendered with several kit seeds: the seed only changes the noise
# inside the snare and hat samples, so a tag that flips with it is fragile.
SEEDS = (7, 8, 9)


def beat_cases(quick: bool = False) -> List[Dict[str, Any]]:
    variants = QUICK_VARIANTS if quick else tuple(VARIANTS)
    seeds = SEEDS[:1] if quick else SEEDS
    cases: List[Dict[str, Any]] = []
    for groove, tempos in GROOVE_TEMPOS.items():
        for bpm in tempos:
            for variant in variants:
                for seed in seeds:
                    # Two-step only reads as itself when it swings, as in the tests.
                    swing = 1.0 if groove == "two_step" else 0.0
                    cases.append(
                        {
                            "groove": groove,
                            "bpm": bpm,
                            "variant": variant,
                            "swing": swing,
                            "seed": seed,
                        }
                    )
    return cases


def _tag_beat_case(case: Dict[str, Any]) -> Dict[str, Any]:
    from .. import tagging

    audio = render(
        case["groove"],
        case["bpm"],
        swing=case["swing"],
        seed=case["seed"],
        **VARIANTS[case["variant"]],
    )
    tags = tagging.tag_audio(audio, SAMPLE_RATE)
    return {
        **case,
        "beat_out": tags.beat if tags else None,
        "bpm_out": (tags.bpm or 0.0) if tags else 0.0,
    }


def run_beats(quick: bool = False, workers: int = -1) -> Dict[str, Any]:
    cases = beat_cases(quick)
    with ThreadPoolExecutor(max_workers=resolve_worker_count(workers)) as pool:
        results = list(pool.map(_tag_beat_case, cases))

    truth = [r["groove"] for r in results]
    predicted = [r["beat_out"] for r in results]
    tempo = metrics.tempo_accuracy([r["bpm"] for r in results], [r["bpm_out"] for r in results])
    by_variant = {}
    for variant in sorted({r["variant"] for r in results}):
        rows = [r for r in results if r["variant"] == variant]
        by_variant[variant] = {
            "beat_accuracy": metrics.accuracy(
                [r["groove"] for r in rows], [r["beat_out"] for r in rows]
            ),
            "tempo_accuracy1": metrics.tempo_accuracy(
                [r["bpm"] for r in rows], [r["bpm_out"] for r in rows]
            )["accuracy1"],
            "cases": len(rows),
        }
    clean = by_variant.get("clean", {}).get("beat_accuracy", 0.0)
    return {
        "headline": {
            "beat_accuracy": metrics.accuracy(truth, predicted),
            "beat_accuracy_clean": clean,
            "beat_macro_f1": metrics.macro_f1(truth, predicted),
            "tempo_accuracy1": tempo["accuracy1"],
            "tempo_accuracy2": tempo["accuracy2"],
        },
        "by_variant": by_variant,
        "per_beat": metrics.per_class(truth, predicted),
        "confusions": metrics.confusion_pairs(truth, predicted, limit=8),
        "failures": [
            {k: r[k] for k in ("groove", "bpm", "variant", "seed", "beat_out", "bpm_out")}
            for r in results
            if r["beat_out"] != r["groove"]
        ][:30],
        "cases": len(results),
    }


# ----------------------------------------------------------------------
# Synthetic keys
# ----------------------------------------------------------------------

PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
# Scale degrees (semitones above the tonic) of each chord's root, and its quality.
# The "loop" progressions are the four-chord loops dance music leans on, and
# share every pitch with their relative key: only the weight on the tonic says
# which one it is. The "cadence" progressions keep returning home, which is
# how a theory book would make the key unambiguous.
PROGRESSIONS = {
    ("major", "loop"): ((0, "maj"), (7, "maj"), (9, "min"), (5, "maj")),  # I-V-vi-IV
    ("minor", "loop"): ((0, "min"), (8, "maj"), (3, "maj"), (10, "maj")),  # i-VI-III-VII
    ("major", "cadence"): ((0, "maj"), (5, "maj"), (0, "maj"), (7, "maj")),  # I-IV-I-V
    ("minor", "cadence"): ((0, "min"), (5, "min"), (0, "min"), (7, "min")),  # i-iv-i-v
}


def render_progression(
    tonic: int,
    mode: str,
    style: str = "loop",
    seconds_per_chord: float = 2.0,
    loops: int = 2,
    drums: bool = False,
) -> np.ndarray:
    """A four-chord progression in a known key: bass note plus a triad of soft saw-like tones."""
    sr = SAMPLE_RATE
    n = int(seconds_per_chord * sr)
    t = np.arange(n) / sr
    envelope = np.minimum(1.0, t * 20.0) * np.exp(-t * 0.6)
    pieces = []
    for _ in range(loops):
        for degree, quality in PROGRESSIONS[(mode, style)]:
            root = 48 + (tonic + degree) % 12  # MIDI note in the octave below middle C
            third = 4 if quality == "maj" else 3
            notes = (root - 12, root, root + third, root + 7)
            chord = np.zeros(n)
            for note in notes:
                freq = 440.0 * 2 ** ((note - 69) / 12.0)
                for partial in range(1, 6):
                    chord += np.sin(2 * np.pi * freq * partial * t) / partial**1.5
            pieces.append(chord * envelope)
    audio = np.concatenate(pieces)
    if drums:
        beat = render("four_on_floor", 124.0, seconds=audio.size / sr + 0.1)[: audio.size]
        audio = audio / (np.max(np.abs(audio)) or 1.0) + 0.6 * beat
    return (audio / (np.max(np.abs(audio)) or 1.0) * 0.9).astype(np.float32)


def _key_case(case: Dict[str, Any]) -> Dict[str, Any]:
    from .. import tagging

    audio = render_progression(case["tonic"], case["mode"], case["style"], drums=case["drums"])
    described = tagging.describe_audio(audio, SAMPLE_RATE)
    return {**case, "key_out": described.get("key")}


def run_keys(workers: int = -1) -> Dict[str, Any]:
    cases = [
        {
            "tonic": tonic,
            "mode": mode,
            "style": style,
            "drums": drums,
            "key": f"{PITCH_NAMES[tonic]} {mode}",
        }
        for tonic in range(12)
        for mode in ("major", "minor")
        for style in ("loop", "cadence")
        for drums in (False, True)
    ]
    with ThreadPoolExecutor(max_workers=resolve_worker_count(workers)) as pool:
        results = list(pool.map(_key_case, cases))
    overall = metrics.key_accuracy([r["key"] for r in results], [r["key_out"] for r in results])

    def subset(**match: Any) -> Dict[str, float]:
        rows = [r for r in results if all(r[k] == v for k, v in match.items())]
        return metrics.key_accuracy([r["key"] for r in rows], [r["key_out"] for r in rows])

    with_drums = subset(drums=True)
    return {
        "headline": {
            "key_weighted": overall["weighted"],
            "key_correct": overall["correct"],
            "key_correct_with_drums": with_drums["correct"],
            "key_correct_minor_loop": subset(mode="minor", style="loop")["correct"],
            "key_correct_cadence": subset(style="cadence")["correct"],
        },
        "breakdown": overall,
        "failures": [
            {k: r[k] for k in ("key", "style", "drums", "key_out")}
            for r in results
            if r["key"] != r["key_out"]
        ],
        "cases": len(results),
    }


# ----------------------------------------------------------------------
# GiantSteps
# ----------------------------------------------------------------------


def _by_genre(
    items: Sequence[Item], scores: Sequence[Optional[float]]
) -> Dict[str, Dict[str, float]]:
    out: Dict[str, List[float]] = {}
    for item, score in zip(items, scores, strict=True):
        if score is None:
            continue
        out.setdefault(item.meta.get("genre") or "unknown", []).append(score)
    return {
        genre: {"score": float(np.mean(values)), "count": len(values)}
        for genre, values in sorted(out.items(), key=lambda kv: -len(kv[1]))
    }


def score_keys(
    items: Sequence[Item], records: Sequence[Optional[Dict[str, Any]]]
) -> Dict[str, Any]:
    pairs = [(i, r) for i, r in zip(items, records, strict=True) if r is not None]
    reference = [i.label for i, _ in pairs]
    ours = metrics.key_accuracy(reference, [r.get("key_raw") for _, r in pairs])
    # Only the keys confident enough to become a tag.
    tagged = [(i.label, r.get("key")) for i, r in pairs if r.get("key")]
    ours_tagged = metrics.key_accuracy([a for a, _ in tagged], [b for _, b in tagged])

    # Baseline: the commonest annotated key for every track.
    parsed = [metrics.parse_key(k) for k in reference]
    counts: Dict[Tuple[int, str], int] = {}
    for p in parsed:
        if p:
            counts[p] = counts.get(p, 0) + 1
    common = max(counts, key=lambda k: counts[k]) if counts else (0, "minor")
    common_text = f"{PITCH_NAMES[common[0]]} {common[1]}"
    methods: Dict[str, Any] = {
        "ours_all": ours,
        "ours_tagged_only": {
            **ours_tagged,
            "share_tagged": len(tagged) / len(pairs) if pairs else 0.0,
        },
        "most_common_key": metrics.key_accuracy(reference, [common_text] * len(reference)),
    }
    if any("essentia_key" in r for _, r in pairs):
        methods["essentia_edma"] = metrics.key_accuracy(
            reference, [r.get("essentia_key") for _, r in pairs]
        )

    per_item = [
        metrics.key_score(p, metrics.parse_key(r.get("key_raw"))) if p else None
        for p, (_, r) in zip(parsed, pairs, strict=True)
    ]
    return {
        "headline": {
            "key_weighted": ours["weighted"],
            "key_correct": ours["correct"],
            "key_tagged_correct": ours_tagged["correct"],
            "key_share_tagged": methods["ours_tagged_only"]["share_tagged"],
        },
        "methods": methods,
        "by_genre": _by_genre([i for i, _ in pairs], per_item),
        "analysed": len(pairs),
        "missing": len(items) - len(pairs),
    }


# Beatport genres whose beat type is not in doubt, for a beat-type score until
# real beat labels exist. Drum & bass accepts any break-based groove.
_FOUR_ON_FLOOR_GENRES = {
    "techno",
    "house",
    "tech-house",
    "deep-house",
    "progressive-house",
    "minimal",
    "trance",
    "psy-trance",
    "hard-dance",
}
_EXPECTED_BEATS = {
    **{genre: {"four_on_floor"} for genre in _FOUR_ON_FLOOR_GENRES},
    "dubstep": {"half_time"},
    "drum-and-bass": {"breakbeat", "two_step", "half_time"},
}


def _beat_label_accuracy(genres: Sequence[str], beats: Sequence[str]) -> Optional[float]:
    hits = [
        beat in _EXPECTED_BEATS[genre]
        for genre, beat in zip(genres, beats, strict=True)
        if genre in _EXPECTED_BEATS
    ]
    return float(np.mean(hits)) if hits else None


def score_tempo(
    items: Sequence[Item], records: Sequence[Optional[Dict[str, Any]]]
) -> Dict[str, Any]:
    pairs = []
    for item, record in zip(items, records, strict=True):
        try:
            annotated = float(item.label or 0)
        except ValueError:
            continue
        if record is not None and annotated > 0:
            pairs.append((item, annotated, record))
    reference = [ref for _, ref, _ in pairs]
    methods = {
        "ours": metrics.tempo_accuracy(reference, [r.get("bpm") or 0.0 for _, _, r in pairs]),
        "librosa_beat_track": metrics.tempo_accuracy(
            reference, [r.get("librosa_bpm") or 0.0 for _, _, r in pairs]
        ),
    }
    if any("essentia_bpm" in r for _, _, r in pairs):
        methods["essentia_rhythm2013"] = metrics.tempo_accuracy(
            reference, [r.get("essentia_bpm") or 0.0 for _, _, r in pairs]
        )
    per_item = [float(abs((r.get("bpm") or 0.0) - ref) <= 0.04 * ref) for _, ref, r in pairs]
    beats: Dict[str, Dict[str, int]] = {}
    for item, _ref, record in pairs:
        genre = item.meta.get("genre") or "unknown"
        beat = record.get("beat") or "none"
        beats.setdefault(genre, {})
        beats[genre][beat] = beats[genre].get(beat, 0) + 1
    beat_accuracy = _beat_label_accuracy(
        [item.meta.get("genre") or "unknown" for item, _, _ in pairs],
        [record.get("beat") or "none" for _, _, record in pairs],
    )
    return {
        "headline": {
            "tempo_accuracy1": methods["ours"]["accuracy1"],
            "tempo_accuracy2": methods["ours"]["accuracy2"],
            "librosa_accuracy1": methods["librosa_beat_track"]["accuracy1"],
            # Share of tracks from unambiguous genres (techno, house, trance
            # families -> four_on_floor; dubstep -> half_time; drum & bass ->
            # a break) whose beat type matches.
            "beat_genre_accuracy": beat_accuracy,
        },
        "methods": methods,
        "by_genre": _by_genre([i for i, _, _ in pairs], per_item),
        # Beat type per Beatport genre: a proxy check until beat-type labels exist.
        "beat_by_genre": beats,
        "analysed": len(pairs),
        "missing": len(items) - len(pairs),
    }
