"""Simulate a DJ's library and measure how well new music is sorted into it.

A classification benchmark asks "which of ten balanced genres is this?". The
sorter's job is different: a handful of user-chosen folders of very different
sizes, a few new tracks, some of which belong in no folder at all, and a
decision about which tracks are safe to file without asking. So each trial
builds a fresh imaginary library from a labelled dataset:

1. Draw a random set of labels to be the user's folders, and one more label
   whose tracks belong nowhere.
2. Split every label's tracks by artist, so no artist is on both sides.
3. Give each folder a random number of references, log-uniform between a few
   and a few hundred, the way real crates vary.
4. Fit the sorter on the references, score the held-out tracks, and score the
   same split with simple baselines.

Hundreds of trials give a distribution rather than one lucky number.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..features import layout_for_dim
from ..sorter import STATUS_AUTO, STATUS_UNMATCHED, GroupSorter, SorterConfig
from . import metrics


NOWHERE = "__nowhere__"


class NotEnoughData(ValueError):
    """The data cannot support the suite (a reason to skip it, not a bug)."""


# Thresholds swept to find the auto-file operating point.
CONFIDENCE_GRID = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
MARGIN_GRID = (0.0, 0.04, 0.08, 0.15, 0.25, 0.4)
TARGET_PRECISIONS = (0.9, 0.95)


@dataclass
class SimulationSettings:
    trials: int = 50
    min_groups: int = 4
    max_groups: int = 10
    min_refs: int = 8
    max_refs: int = 300
    queries_per_group: int = 15
    novel_queries: int = 15
    query_share: float = 0.3
    seed: int = 0


@dataclass
class _Outcomes:
    """Every query's result for one method, pooled across trials."""

    truth: List[str] = field(default_factory=list)
    top1: List[Optional[str]] = field(default_factory=list)
    ranked: List[List[str]] = field(default_factory=list)
    confidence: List[float] = field(default_factory=list)
    margin: List[float] = field(default_factory=list)
    distance: List[float] = field(default_factory=list)
    status: List[str] = field(default_factory=list)
    ref_count: List[int] = field(default_factory=list)


def _split_by_artist(
    indices: np.ndarray, artists: Sequence[Optional[str]], share: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Reference and query indices with no artist on both sides."""
    keyed = [artists[i] or f"__track{i}" for i in indices]
    unique = sorted(set(keyed))
    rng.shuffle(unique)
    n_query = max(1, round(len(unique) * share)) if len(unique) > 1 else 0
    query_artists = set(unique[:n_query])
    mask = np.array([k in query_artists for k in keyed], dtype=bool)
    return indices[~mask], indices[mask]


def find_tempo_column(width: int) -> Optional[int]:
    """The tempo's index in an extractor vector of this width, if it has one."""
    try:
        layout = layout_for_dim(width)
        return layout.span("tempo")[0] if layout.has("tempo") else None
    except Exception:
        return None


def _log_uniform(rng: np.random.Generator, low: int, high: int) -> int:
    if high <= low:
        return int(high)
    return round(math.exp(rng.uniform(math.log(low), math.log(high))))


def _baselines(
    ref_x: np.ndarray, ref_y: List[str], query_x: np.ndarray, tempo_column: Optional[int]
) -> Dict[str, Tuple[List[str], List[List[str]], List[float]]]:
    """(top-1, ranking, confidence) per query for each baseline method."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = sorted(set(ref_y))
    counts = {label: ref_y.count(label) for label in labels}
    by_size = sorted(labels, key=lambda label: -counts[label])
    n = len(query_x)
    out: Dict[str, Tuple[List[str], List[List[str]], List[float]]] = {
        # Always the biggest folder: what any method must beat.
        "majority": (
            [by_size[0]] * n,
            [by_size] * n,
            [counts[by_size[0]] / len(ref_y)] * n,
        )
    }

    if tempo_column is not None:
        ref_tempo = ref_x[:, tempo_column]
        centres = {label: float(np.median(ref_tempo[np.array(ref_y) == label])) for label in labels}
        tops: List[str] = []
        ranks: List[List[str]] = []
        for bpm in query_x[:, tempo_column]:
            gaps = {label: abs(centres[label] - float(bpm)) for label in labels}
            nearest = sorted(labels, key=gaps.__getitem__)
            tops.append(nearest[0])
            ranks.append(nearest)
        # Tempo-only: how far a single number gets. If the sorter barely beats
        # this, the folders are really tempo bands.
        out["tempo_only"] = (tops, ranks, [1.0 / len(labels)] * n)

    for name, model in (
        ("logistic", LogisticRegression(max_iter=2000, C=0.5)),
        ("knn", KNeighborsClassifier(n_neighbors=min(5, len(ref_y)))),
    ):
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(ref_x, ref_y)
        probabilities = pipeline.predict_proba(query_x)
        classes = [str(c) for c in pipeline.classes_]
        ranking = np.argsort(-probabilities, axis=1)
        out[name] = (
            [classes[int(row[0])] for row in ranking],
            [[classes[int(j)] for j in row] for row in ranking],
            [float(probabilities[i, int(row[0])]) for i, row in enumerate(ranking)],
        )
    return out


def _trial(
    x: np.ndarray,
    y: np.ndarray,
    artists: Sequence[Optional[str]],
    settings: SimulationSettings,
    sorter_config: SorterConfig,
    rng: np.random.Generator,
    tempo_column: Optional[int],
    pooled: Dict[str, _Outcomes],
) -> bool:
    labels = sorted(set(y.tolist()))
    if len(labels) < 3:
        raise NotEnoughData("Need at least three labels to simulate a library")
    # One label is always held back as "belongs nowhere"; with few labels the
    # folder count shrinks rather than failing.
    most = min(settings.max_groups, len(labels) - 1)
    fewest = min(settings.min_groups, most)
    n_groups = int(rng.integers(fewest, most + 1))
    chosen = list(rng.choice(labels, size=n_groups + 1, replace=False))
    folders, nowhere = chosen[:-1], chosen[-1]

    ref_idx: Dict[str, np.ndarray] = {}
    query_idx: List[Tuple[int, str]] = []
    for label in folders:
        members = np.flatnonzero(y == label)
        refs, queries = _split_by_artist(members, artists, settings.query_share, rng)
        if len(refs) < 2 or len(queries) == 0:
            return False
        take = min(len(refs), _log_uniform(rng, settings.min_refs, settings.max_refs))
        ref_idx[label] = rng.choice(refs, size=take, replace=False)
        for q in rng.choice(
            queries, size=min(len(queries), settings.queries_per_group), replace=False
        ):
            query_idx.append((int(q), label))

    novel = np.flatnonzero(y == nowhere)
    for q in rng.choice(novel, size=min(len(novel), settings.novel_queries), replace=False):
        query_idx.append((int(q), NOWHERE))

    ids = {label: i for i, label in enumerate(folders)}
    names = {i: label for label, i in ids.items()}
    sorter = GroupSorter(sorter_config).fit(
        {ids[label]: x[idx] for label, idx in ref_idx.items()}, names
    )
    query_x = x[[q for q, _ in query_idx]]
    truths = [t for _, t in query_idx]
    ref_sizes = {label: len(idx) for label, idx in ref_idx.items()}

    suggestions = sorter.suggest_batch(query_x, top_k=len(folders))
    out = pooled.setdefault("sorter", _Outcomes())
    for truth, suggestion in zip(truths, suggestions, strict=True):
        ranked = [names[r["group_id"]] for r in suggestion.ranked]
        out.truth.append(truth)
        out.top1.append(ranked[0] if ranked else None)
        out.ranked.append(ranked)
        out.confidence.append(suggestion.confidence)
        out.margin.append(suggestion.margin)
        out.distance.append(suggestion.distance)
        out.status.append(suggestion.status)
        out.ref_count.append(ref_sizes.get(truth, 0))

    ref_x = np.vstack([x[idx] for idx in ref_idx.values()])
    ref_y = [label for label, idx in ref_idx.items() for _ in idx]
    for method, (tops, ranks, confs) in _baselines(ref_x, ref_y, query_x, tempo_column).items():
        out = pooled.setdefault(method, _Outcomes())
        out.truth.extend(truths)
        out.top1.extend(tops)
        out.ranked.extend(ranks)
        out.confidence.extend(confs)
        out.ref_count.extend(ref_sizes.get(t, 0) for t in truths)
    return True


def _known(outcomes: _Outcomes) -> List[int]:
    return [i for i, t in enumerate(outcomes.truth) if t != NOWHERE]


def _summarise_method(outcomes: _Outcomes, scored_confidence: bool = True) -> Dict[str, Any]:
    known = _known(outcomes)
    truth = [outcomes.truth[i] for i in known]
    top1 = [outcomes.top1[i] for i in known]
    ranked = [outcomes.ranked[i] for i in known]
    confidence = [outcomes.confidence[i] for i in known]
    correct = [t == p for t, p in zip(truth, top1, strict=True)]

    by_size: Dict[str, Dict[str, float]] = {}
    for name, low, high in (
        ("small (<20)", 0, 20),
        ("medium (20-99)", 20, 100),
        ("large (100+)", 100, 10**9),
    ):
        members = [i for i in known if low <= outcomes.ref_count[i] < high]
        if members:
            by_size[name] = {
                "recall": float(np.mean([outcomes.truth[i] == outcomes.top1[i] for i in members])),
                "queries": len(members),
            }

    return {
        "top1": metrics.accuracy(truth, top1),
        "top3": metrics.top_k_accuracy(truth, ranked, 3),
        "macro_f1": metrics.macro_f1(truth, top1),
        "balanced_accuracy": metrics.balanced_accuracy(truth, top1),
        # A constant confidence (majority, tempo-only) has no calibration or
        # ranking to judge; a number here would only mislead.
        "ece": metrics.expected_calibration_error(confidence, correct)
        if scored_confidence
        else None,
        "brier": metrics.brier(confidence, correct) if scored_confidence else None,
        "aurc": metrics.risk_coverage(confidence, correct)["aurc"] if scored_confidence else None,
        "recall_by_folder_size": by_size,
        "queries": len(known),
    }


def _decisions(
    outcomes: _Outcomes, min_confidence: float, min_margin: float, novelty: float
) -> Dict[str, float]:
    """What auto-filing would do at given thresholds, counting novel tracks as errors."""
    filed = correct = 0
    for truth, top1, conf, margin, dist in zip(
        outcomes.truth,
        outcomes.top1,
        outcomes.confidence,
        outcomes.margin,
        outcomes.distance,
        strict=True,
    ):
        if dist > novelty or conf < min_confidence or margin < min_margin:
            continue
        filed += 1
        correct += int(truth == top1)
    total = len(outcomes.truth)
    return {
        "precision": correct / filed if filed else 1.0,
        "coverage": filed / total if total else 0.0,
    }


def _sorter_details(outcomes: _Outcomes, config: SorterConfig) -> Dict[str, Any]:
    known = _known(outcomes)
    novel = [i for i, t in enumerate(outcomes.truth) if t == NOWHERE]
    auto = [i for i in known if outcomes.status[i] == STATUS_AUTO]
    confidence = [outcomes.confidence[i] for i in known]
    correct = [outcomes.truth[i] == outcomes.top1[i] for i in known]

    sweep = []
    for c in CONFIDENCE_GRID:
        for m in MARGIN_GRID:
            row = _decisions(outcomes, c, m, config.novelty_factor)
            sweep.append({"confidence": c, "margin": m, **row})
    operating_points = {}
    for target in TARGET_PRECISIONS:
        feasible = [row for row in sweep if row["precision"] >= target and row["coverage"] > 0]
        best = max(feasible, key=lambda row: row["coverage"]) if feasible else None
        operating_points[f"precision_{int(target * 100)}"] = best

    return {
        "current_thresholds": {
            "auto_accept_confidence": config.auto_accept_confidence,
            "min_margin": config.min_margin,
            "novelty_factor": config.novelty_factor,
            **_decisions(
                outcomes, config.auto_accept_confidence, config.min_margin, config.novelty_factor
            ),
        },
        "auto_rate_known": len(auto) / len(known) if known else 0.0,
        "auto_precision_known": (
            float(np.mean([outcomes.truth[i] == outcomes.top1[i] for i in auto])) if auto else 0.0
        ),
        "nowhere_flagged_unmatched": (
            float(np.mean([outcomes.status[i] == STATUS_UNMATCHED for i in novel]))
            if novel
            else None
        ),
        "nowhere_auto_filed": (
            float(np.mean([outcomes.status[i] == STATUS_AUTO for i in novel])) if novel else None
        ),
        "known_flagged_unmatched": (
            float(np.mean([outcomes.status[i] == STATUS_UNMATCHED for i in known]))
            if known
            else 0.0
        ),
        # Does distance separate "belongs nowhere" from "belongs somewhere"?
        "novelty_auroc": metrics.auroc(
            [outcomes.distance[i] for i in novel], [outcomes.distance[i] for i in known]
        ),
        "reliability": metrics.reliability(confidence, correct),
        "risk_coverage": metrics.risk_coverage(confidence, correct, points=10)["curve"],
        "threshold_sweep": sweep,
        "operating_points": operating_points,
        "confusions": metrics.confusion_pairs(
            [outcomes.truth[i] for i in known], [outcomes.top1[i] for i in known]
        ),
    }


def simulate(
    x: np.ndarray,
    y: Sequence[str],
    artists: Optional[Sequence[Optional[str]]] = None,
    settings: Optional[SimulationSettings] = None,
    sorter_config: Optional[SorterConfig] = None,
    tempo_column: Optional[int] = None,
    infer_tempo: bool = True,
) -> Dict[str, Any]:
    """Run the trials and summarise every method.

    Args:
        x: One raw feature vector per track (the extractor's output).
        y: The track's label (genre, folder name).
        artists: Artist per track, for artist-separated splits. None splits by
            track, which overstates accuracy when artists repeat.
        tempo_column: Where the tempo is, for the tempo-only baseline.
        infer_tempo: Find the tempo from the feature layout when no column is
            given. Turn off for vectors that are not (only) extractor output,
            such as a pretrained embedding, whose width says nothing.
    """
    settings = settings or SimulationSettings()
    sorter_config = sorter_config or SorterConfig()
    x = np.asarray(x, dtype=float)
    labels = np.asarray(list(y), dtype=object)
    artist_list: Sequence[Optional[str]] = (
        list(artists) if artists is not None else [None] * len(labels)
    )

    if tempo_column is None and infer_tempo:
        tempo_column = find_tempo_column(x.shape[1])

    rng = np.random.default_rng(settings.seed)
    pooled: Dict[str, _Outcomes] = {}
    completed = attempts = 0
    while completed < settings.trials and attempts < settings.trials * 5:
        attempts += 1
        if _trial(x, labels, artist_list, settings, sorter_config, rng, tempo_column, pooled):
            completed += 1

    if "sorter" not in pooled:
        raise NotEnoughData("No trial could be built; too few tracks or artists per label")

    unscored = {"majority", "tempo_only"}
    methods = {
        name: _summarise_method(outcomes, scored_confidence=name not in unscored)
        for name, outcomes in pooled.items()
    }
    details = _sorter_details(pooled["sorter"], sorter_config)
    best = details["operating_points"].get("precision_95")
    return {
        "headline": {
            "sorter_top1": methods["sorter"]["top1"],
            "sorter_top3": methods["sorter"]["top3"],
            "sorter_macro_f1": methods["sorter"]["macro_f1"],
            "sorter_ece": methods["sorter"]["ece"],
            "logistic_top1": methods.get("logistic", {}).get("top1"),
            "auto_precision_current": details["current_thresholds"]["precision"],
            "auto_coverage_current": details["current_thresholds"]["coverage"],
            "coverage_at_95_precision": best["coverage"] if best else 0.0,
            "novelty_auroc": details["novelty_auroc"],
            "nowhere_auto_filed": details["nowhere_auto_filed"],
        },
        "lower_is_better": ["sorter_ece", "nowhere_auto_filed"],
        "settings": settings.__dict__,
        "trials": completed,
        "methods": methods,
        "sorter": details,
    }
