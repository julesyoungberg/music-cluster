"""Scoring functions for the evaluation harness.

Plain numpy over plain lists, so each one can be unit-tested with a handful of
hand-made numbers and read without a framework in the way.
"""

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


# ----------------------------------------------------------------------
# Classification
# ----------------------------------------------------------------------


def accuracy(truth: Sequence, predicted: Sequence) -> float:
    if not truth:
        return 0.0
    return float(np.mean([t == p for t, p in zip(truth, predicted, strict=True)]))


def top_k_accuracy(truth: Sequence, ranked: Sequence[Sequence], k: int) -> float:
    """Share of items whose true label is among the first ``k`` ranked guesses."""
    if not truth:
        return 0.0
    return float(np.mean([t in list(r)[:k] for t, r in zip(truth, ranked, strict=True)]))


def macro_f1(truth: Sequence, predicted: Sequence) -> float:
    """F1 averaged over the true classes, so an 8-track group counts as much as a big one."""
    labels = sorted(set(truth), key=str)
    if not labels:
        return 0.0
    scores = []
    for label in labels:
        tp = sum(1 for t, p in zip(truth, predicted, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, predicted, strict=True) if t != label and p == label)
        fn = sum(1 for t, p in zip(truth, predicted, strict=True) if t == label and p != label)
        denominator = 2 * tp + fp + fn
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def balanced_accuracy(truth: Sequence, predicted: Sequence) -> float:
    """Recall averaged over the true classes."""
    labels = sorted(set(truth), key=str)
    if not labels:
        return 0.0
    recalls = []
    for label in labels:
        members = [p for t, p in zip(truth, predicted, strict=True) if t == label]
        recalls.append(sum(1 for p in members if p == label) / len(members))
    return float(np.mean(recalls))


def per_class(truth: Sequence, predicted: Sequence) -> Dict[str, Dict[str, float]]:
    """Precision, recall and support for every label that appears in the truth."""
    out: Dict[str, Dict[str, float]] = {}
    for label in sorted(set(truth), key=str):
        tp = sum(1 for t, p in zip(truth, predicted, strict=True) if t == label and p == label)
        fp = sum(1 for t, p in zip(truth, predicted, strict=True) if t != label and p == label)
        support = sum(1 for t in truth if t == label)
        out[str(label)] = {
            "precision": tp / (tp + fp) if tp + fp else 0.0,
            "recall": tp / support if support else 0.0,
            "support": support,
        }
    return out


def confusion_pairs(
    truth: Sequence, predicted: Sequence, limit: int = 5
) -> List[Tuple[str, str, int]]:
    """The most common (true, predicted) mistakes, most frequent first."""
    counts: Dict[Tuple[str, str], int] = {}
    for t, p in zip(truth, predicted, strict=True):
        if t != p:
            pair = (str(t), str(p))
            counts[pair] = counts.get(pair, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: -item[1])[:limit]
    return [(t, p, n) for (t, p), n in ranked]


# ----------------------------------------------------------------------
# Calibration and selective prediction
# ----------------------------------------------------------------------


def reliability(
    confidence: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> List[Dict[str, float]]:
    """Mean confidence vs observed accuracy, per equal-width confidence bin."""
    conf = np.asarray(confidence, dtype=float)
    hit = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    rows = []
    for i in range(bins):
        low, high = edges[i], edges[i + 1]
        mask = (conf >= low) & ((conf < high) if i < bins - 1 else (conf <= high))
        if not mask.any():
            continue
        rows.append(
            {
                "low": float(low),
                "high": float(high),
                "count": int(mask.sum()),
                "confidence": float(conf[mask].mean()),
                "accuracy": float(hit[mask].mean()),
            }
        )
    return rows


def expected_calibration_error(
    confidence: Sequence[float], correct: Sequence[bool], bins: int = 10
) -> float:
    """Average gap between how sure the scores say they are and how often they are right.

    0 means a confidence of 0.8 is right 80% of the time; 0.2 means the bar on
    the review screen is off by twenty points on average.
    """
    total = len(confidence)
    if not total:
        return 0.0
    return float(
        sum(
            row["count"] * abs(row["confidence"] - row["accuracy"])
            for row in reliability(confidence, correct, bins)
        )
        / total
    )


def brier(confidence: Sequence[float], correct: Sequence[bool]) -> float:
    """Mean squared error of the top-1 confidence as a probability of being right."""
    if not confidence:
        return 0.0
    conf = np.asarray(confidence, dtype=float)
    hit = np.asarray(correct, dtype=float)
    return float(np.mean((conf - hit) ** 2))


def risk_coverage(
    confidence: Sequence[float], correct: Sequence[bool], points: int = 20
) -> Dict[str, object]:
    """Precision of the most confident decisions as more of them are accepted.

    This is the auto-file question directly: if only the surest X% of tracks
    are filed without asking, how many of those land in the right folder?
    ``aurc`` is the area under the error-vs-coverage curve (lower is better).
    """
    conf = np.asarray(confidence, dtype=float)
    hit = np.asarray(correct, dtype=float)
    if conf.size == 0:
        return {"curve": [], "aurc": 0.0}
    order = np.argsort(-conf, kind="stable")
    cumulative = np.cumsum(hit[order]) / np.arange(1, conf.size + 1)
    errors = 1.0 - cumulative
    curve = []
    for coverage in np.linspace(1.0 / points, 1.0, points):
        n = max(1, round(coverage * conf.size))
        curve.append(
            {
                "coverage": float(n / conf.size),
                "precision": float(cumulative[n - 1]),
                "threshold": float(conf[order][n - 1]),
            }
        )
    return {"curve": curve, "aurc": float(np.mean(errors))}


def auroc(positive_scores: Sequence[float], negative_scores: Sequence[float]) -> Optional[float]:
    """Probability that a random positive scores above a random negative (ties count half)."""
    pos = np.asarray(positive_scores, dtype=float)
    neg = np.asarray(negative_scores, dtype=float)
    if pos.size == 0 or neg.size == 0:
        return None
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if pos.size == 0 or neg.size == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (pos.size * neg.size))


# ----------------------------------------------------------------------
# Tempo
# ----------------------------------------------------------------------


def tempo_accuracy(reference: Sequence[float], estimate: Sequence[float]) -> Dict[str, float]:
    """MIREX-style tempo accuracy.

    ``accuracy1``: within 4% of the annotation. ``accuracy2``: also counts the
    octave errors (x2, x1/2, x3, x1/3). The gap between them is the share of
    tracks whose tempo was right but counted at the wrong speed.
    """
    pairs = [(float(r), float(e)) for r, e in zip(reference, estimate, strict=True) if r and r > 0]
    if not pairs:
        return {"accuracy1": 0.0, "accuracy2": 0.0, "count": 0}

    def within(ref: float, est: float) -> bool:
        return abs(est - ref) <= 0.04 * ref

    acc1 = [within(r, e) for r, e in pairs]
    acc2 = [any(within(r * f, e) for f in (1.0, 2.0, 0.5, 3.0, 1.0 / 3.0)) for r, e in pairs]
    return {
        "accuracy1": float(np.mean(acc1)),
        "accuracy2": float(np.mean(acc2)),
        "count": len(pairs),
    }


# ----------------------------------------------------------------------
# Key
# ----------------------------------------------------------------------

_PITCH = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def parse_key(text: Optional[str]) -> Optional[Tuple[int, str]]:
    """``"Eb minor"``, ``"D# min"``, ``"f#m"``, ``"C major"`` → (pitch class, mode).

    None when the text is not a single key (annotations sometimes say "-" or
    list two).
    """
    if not text:
        return None
    cleaned = text.strip().lower().replace("♯", "#").replace("♭", "b")
    match = re.fullmatch(r"([a-g])\s*([#b]?)\s*(major|minor|maj|min|m)?", cleaned)
    if not match:
        return None
    letter, accidental, mode = match.groups()
    pitch = _PITCH[letter] + (1 if accidental == "#" else -1 if accidental == "b" else 0)
    is_minor = mode in ("minor", "min", "m")
    return pitch % 12, "minor" if is_minor else "major"


def key_score(reference: Tuple[int, str], estimate: Optional[Tuple[int, str]]) -> float:
    """MIREX weighted key score for one track.

    1 for the right key, 0.5 a fifth above in the same mode, 0.3 the relative
    key, 0.2 the parallel key, 0 otherwise. For mixing in key the first three
    are compatible on the Camelot wheel, which is why they earn partial credit.
    """
    if estimate is None:
        return 0.0
    (ref_pc, ref_mode), (est_pc, est_mode) = reference, estimate
    if ref_pc == est_pc and ref_mode == est_mode:
        return 1.0
    if ref_mode == est_mode and (est_pc - ref_pc) % 12 == 7:
        return 0.5
    if ref_mode == "major" and est_mode == "minor" and (est_pc - ref_pc) % 12 == 9:
        return 0.3
    if ref_mode == "minor" and est_mode == "major" and (est_pc - ref_pc) % 12 == 3:
        return 0.3
    if ref_pc == est_pc and ref_mode != est_mode:
        return 0.2
    return 0.0


def key_accuracy(
    reference: Sequence[Optional[str]], estimate: Sequence[Optional[str]]
) -> Dict[str, float]:
    """Weighted score plus the breakdown by kind of error."""
    kinds = {1.0: "correct", 0.5: "fifth", 0.3: "relative", 0.2: "parallel", 0.0: "other"}
    counts = dict.fromkeys(kinds.values(), 0)
    scores = []
    for ref_text, est_text in zip(reference, estimate, strict=True):
        ref = parse_key(ref_text)
        if ref is None:
            continue
        score = key_score(ref, parse_key(est_text))
        scores.append(score)
        counts[kinds[score]] += 1
    n = len(scores)
    result: Dict[str, float] = {
        "weighted": float(np.mean(scores)) if n else 0.0,
        "count": n,
    }
    for kind, count in counts.items():
        result[kind] = count / n if n else 0.0
    return result
