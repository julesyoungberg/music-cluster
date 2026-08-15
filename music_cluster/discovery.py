"""Assisted discovery of groups from an unsorted pile.

The starting point for a DJ who has one huge folder and no structure yet.
Clustering proposes candidate piles with representative tracks; the human
listens, renames, merges, and promotes the ones that are real. Nothing becomes
a group without that step — the machine narrows the choice, it does not make it.

The second entry point, :func:`expand_seeds`, goes the other way: the DJ picks
a couple of tracks that capture an idea and asks what else in the pile sounds
like them.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from .clustering import NOISE_LABEL, ClusterEngine, select_exemplars
from .config import Config
from .database import Database
from .embedding import EmbeddingSpace
from .groups import create_group
from .library import analyze_files, collect_files
from .naming import group_stats, suggest_label


def run_discovery(
    db: Database,
    config: Config,
    paths: Optional[Sequence[str]] = None,
    track_ids: Optional[Sequence[int]] = None,
    name: Optional[str] = None,
    algorithm: Optional[str] = None,
    granularity: Optional[str] = None,
    target_groups: Optional[int] = None,
    min_group_size: Optional[int] = None,
    exemplars: Optional[int] = None,
    recursive: bool = True,
    workers: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    use_llm: Optional[bool] = None,
) -> Dict[str, Any]:
    """Propose candidate groups from a folder or an explicit track list."""
    settings = config.section("discovery")
    algorithm = algorithm or settings.get("algorithm", "hdbscan")
    granularity = granularity or settings.get("granularity", "normal")
    min_group_size = min_group_size or settings.get("min_group_size", 8)
    max_candidates = settings.get("max_candidates", 30)
    exemplar_count = exemplars or settings.get("exemplars_per_candidate", 5)

    if track_ids is None:
        if not paths:
            raise ValueError("Provide either paths or track_ids to discover from")
        files = collect_files(paths, recursive=recursive)
        if not files:
            raise ValueError("No audio files found in the given path(s)")
        analysis = analyze_files(db, config, files, workers=workers, progress=progress)
        track_ids = analysis.track_ids

    matrix, found_ids = db.get_feature_matrix(list(track_ids))
    if len(found_ids) < min_group_size:
        raise ValueError(
            f"Need at least {min_group_size} analysed tracks to look for groups, "
            f"got {len(found_ids)}"
        )

    # No labels exist yet, so the space is unsupervised: standardise and reduce.
    space = EmbeddingSpace(
        projection="pca",
        pca_variance=config.get("sorting", "pca_variance", default=0.95),
        max_components=config.get("sorting", "max_components", default=40),
        feature_weights=config.get("sorting", "feature_weights", default={}),
    ).fit(matrix)
    projected = space.transform(matrix)

    engine = ClusterEngine(
        algorithm=algorithm,
        min_group_size=min_group_size,
        max_candidates=max_candidates,
        detection_method=settings.get("detection_method", "silhouette"),
    )
    labels, centroids, metrics = engine.cluster(
        projected, n_clusters=target_groups, granularity=granularity
    )

    valid_labels = sorted(int(label) for label in np.unique(labels) if label != NOISE_LABEL)
    if not valid_labels:
        raise ValueError(
            "No candidate groups found. Try a smaller --min-size, or the kmeans algorithm "
            "with an explicit --groups count."
        )

    source_path = (
        os.path.abspath(os.path.expanduser(paths[0])) if paths and len(paths) == 1 else None
    )
    run_id = db.create_discovery_run(
        name=name or _default_run_name(paths),
        source_path=source_path,
        algorithm=algorithm,
        params={
            "granularity": granularity,
            "min_group_size": min_group_size,
            "target_groups": target_groups,
            "exemplars": exemplar_count,
            "embedding": space.describe(),
        },
        metrics=metrics,
    )

    candidates = _build_candidates(
        db,
        run_id,
        matrix,
        projected,
        found_ids,
        labels,
        valid_labels,
        exemplar_count,
    )

    if use_llm is None:
        use_llm = config.get("labeling", "llm_enabled", default=False)
    if use_llm:
        _apply_llm_names(db, config, candidates)

    return {
        "run": db.get_discovery_run(run_id),
        "candidates": db.list_discovery_candidates(run_id),
        "unassigned": int(np.sum(labels == NOISE_LABEL)),
    }


def _build_candidates(
    db: Database,
    run_id: int,
    raw_matrix: np.ndarray,
    projected: np.ndarray,
    track_ids: List[int],
    labels: np.ndarray,
    valid_labels: List[int],
    exemplar_count: int,
) -> List[Dict[str, Any]]:
    """Persist one candidate per cluster, with exemplars and a suggested name."""
    candidates: List[Dict[str, Any]] = []

    for index, label in enumerate(valid_labels):
        member_positions = [i for i, value in enumerate(labels) if value == label]
        member_ids = [track_ids[i] for i in member_positions]
        tracks = db.get_tracks(member_ids)
        member_tracks = [tracks[tid] for tid in member_ids if tid in tracks]

        # Descriptors need physical units, so the centroid is taken in raw
        # feature space; distances and exemplars use the projected space.
        raw_centroid = raw_matrix[member_positions].mean(axis=0)
        projected_centroid = projected[member_positions].mean(axis=0)

        exemplar_positions = select_exemplars(projected, member_positions, exemplar_count)
        exemplar_ids = [track_ids[i] for i in exemplar_positions]

        stats = group_stats(raw_centroid, member_tracks)
        suggested = suggest_label(raw_centroid, member_tracks, fallback=f"Candidate {index + 1}")

        candidate_id = db.add_discovery_candidate(
            run_id=run_id,
            candidate_index=index,
            suggested_name=suggested,
            size=len(member_ids),
            centroid=raw_centroid,
            exemplars=exemplar_ids,
            stats=stats,
        )

        distances = np.linalg.norm(projected[member_positions] - projected_centroid, axis=1)
        db.add_discovery_members(
            candidate_id, [(tid, float(d)) for tid, d in zip(member_ids, distances)]
        )

        candidates.append(
            {
                "id": candidate_id,
                "candidate_index": index,
                "size": len(member_ids),
                "stats": stats,
                "suggested_name": suggested,
                "exemplar_tracks": [tracks[tid] for tid in exemplar_ids if tid in tracks],
            }
        )

    return candidates


def _apply_llm_names(db: Database, config: Config, candidates: List[Dict[str, Any]]) -> None:
    """Overlay LLM-suggested names, keeping heuristic ones on any failure."""
    from .llm import suggest_names

    names = suggest_names(candidates, config)
    for candidate in candidates:
        suggested = names.get(candidate["candidate_index"])
        if suggested:
            db.update_discovery_candidate(candidate["id"], suggested_name=suggested)


def promote_candidate(
    db: Database,
    config: Config,
    collection: Dict,
    candidate_id: int,
    name: Optional[str] = None,
    destination_path: Optional[str] = None,
    seeds_only: bool = False,
    target_group_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Turn a reviewed candidate into a real group (or merge it into one).

    Args:
        seeds_only: Add only the exemplar tracks as references. Useful when the
            pile is broadly right but not yet trustworthy enough to define the
            group wholesale.
        target_group_id: Merge into an existing group instead of creating one.
    """
    candidate = db.get_discovery_candidate(candidate_id)
    if not candidate:
        raise LookupError(f"No discovery candidate with ID {candidate_id}")

    if target_group_id is not None:
        group = db.get_group(group_id=target_group_id)
        if not group or group["collection_id"] != collection["id"]:
            raise LookupError("Target group is not part of this collection")
    else:
        group = create_group(
            db,
            collection["id"],
            name=name or candidate["suggested_name"] or f"Candidate {candidate['candidate_index']}",
            destination_path=destination_path,
            source_kind="discovered",
        )

    track_ids = (
        candidate["exemplars"] if seeds_only else db.get_discovery_member_ids(candidate_id)
    )
    existing = set(db.get_group_member_ids(group["id"]))
    fresh = [track_id for track_id in track_ids if track_id not in existing]
    if fresh:
        db.add_group_members(group["id"], fresh, role="seed" if seeds_only else "reference")
        db.touch_collection(collection["id"])

    db.update_discovery_candidate(
        candidate_id,
        status="promoted",
        promoted_group_id=group["id"],
        suggested_name=name or candidate["suggested_name"],
    )

    return {"group": db.get_group(group_id=group["id"]), "added": len(fresh)}


def reject_candidate(db: Database, candidate_id: int) -> None:
    """Mark a candidate as not worth keeping."""
    db.update_discovery_candidate(candidate_id, status="rejected")


def expand_seeds(
    db: Database,
    config: Config,
    seed_track_ids: Sequence[int],
    pool_track_ids: Optional[Sequence[int]] = None,
    limit: int = 50,
    max_distance: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Find pool tracks that sound like a handful of hand-picked seeds.

    This is the manual half of discovery made less manual: the DJ picks two or
    three tracks that capture an idea, and gets back the nearest candidates to
    audition.
    """
    if not seed_track_ids:
        raise ValueError("Provide at least one seed track")

    if pool_track_ids is None:
        _, pool_track_ids = db.get_all_features()
    pool_track_ids = [tid for tid in pool_track_ids if tid not in set(seed_track_ids)]
    if not pool_track_ids:
        return []

    seed_matrix, seed_ids = db.get_feature_matrix(list(seed_track_ids))
    pool_matrix, pool_ids = db.get_feature_matrix(pool_track_ids)
    if len(seed_ids) == 0 or len(pool_ids) == 0:
        return []

    # Fit the space on everything so scaling reflects the whole pool, not just
    # the handful of seeds.
    space = EmbeddingSpace(
        projection="pca",
        pca_variance=config.get("sorting", "pca_variance", default=0.95),
        max_components=config.get("sorting", "max_components", default=40),
        feature_weights=config.get("sorting", "feature_weights", default={}),
    ).fit(np.vstack([seed_matrix, pool_matrix]))

    seeds = space.transform(seed_matrix)
    pool = space.transform(pool_matrix)

    # Distance to the closest seed, so a set of seeds spanning two moods still
    # finds matches for both rather than the midpoint between them.
    distances = np.min(
        np.linalg.norm(pool[:, None, :] - seeds[None, :, :], axis=2), axis=1
    )
    scale = float(np.median(distances)) or 1.0
    order = np.argsort(distances)[:limit]

    tracks = db.get_tracks([pool_ids[i] for i in order])
    results = []
    for i in order:
        normalized = float(distances[i] / scale)
        if max_distance is not None and normalized > max_distance:
            continue
        track = tracks.get(pool_ids[i])
        if track:
            results.append(
                {
                    "track": track,
                    "distance": float(distances[i]),
                    "relative_distance": normalized,
                }
            )
    return results


def _default_run_name(paths: Optional[Sequence[str]]) -> str:
    if paths and len(paths) == 1:
        return os.path.basename(os.path.normpath(os.path.expanduser(paths[0]))) or "Discovery"
    return "Discovery"
