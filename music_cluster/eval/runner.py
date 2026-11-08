"""Run named suites and collect their results."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import analysis, datasets, pretrained, sorting_sim, tag_suites


Echo = Callable[[str], None]

SYNTHETIC_SUITES = ("beats-synthetic", "keys-synthetic")
DATASET_SUITES = (
    "sorting-fma",
    "sorting-fma-sub",
    "key-giantsteps",
    "tempo-giantsteps",
)
ALL_SUITES = SYNTHETIC_SUITES + DATASET_SUITES


def _sorting(
    items: Sequence[datasets.Item],
    cache_name: str,
    settings: sorting_sim.SimulationSettings,
    workers: int,
    root: Optional[str],
    echo: Echo,
    embedding: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """The suite's result, plus one result per feature set when an embedding is asked for."""
    echo(f"  analysing {len(items)} files (cached after the first run)")
    records = analysis.analyze(
        items, cache_name, tags=False, workers=workers, root=root, progress=_progress(echo)
    )
    keep = [
        (item, record["features"])
        for item, record in zip(items, records, strict=True)
        if record and record.get("features")
    ]
    if not keep:
        raise sorting_sim.NotEnoughData("No files could be analysed")
    x = np.array([features for _, features in keep], dtype=float)
    labels = [item.label or "" for item, _ in keep]
    artists = [item.artist for item, _ in keep]
    echo(f"  simulating {settings.trials} libraries over {len(set(labels))} labels")
    result = sorting_sim.simulate(x, labels, artists, settings)
    result["tracks"] = len(keep)
    result["labels"] = sorted(set(labels))
    out = {"": result}
    if embedding:
        out.update(
            _with_embedding(
                keep, x, labels, artists, cache_name, settings, workers, root, echo, embedding
            )
        )
    return out


def _with_embedding(
    keep: List[Tuple[datasets.Item, Any]],
    x: np.ndarray,
    labels: List[str],
    artists: List[Optional[str]],
    cache_name: str,
    settings: sorting_sim.SimulationSettings,
    workers: int,
    root: Optional[str],
    echo: Echo,
    model: str,
) -> Dict[str, Dict[str, Any]]:
    """Score the embedding alone and appended to the hand features.

    All three feature sets run on the same tracks with the same seed, so every
    trial draws the same folders, references and queries: the comparison is
    paired, and the difference is the features alone.
    """
    about = pretrained.spec(model)
    echo(f"  embedding {len(keep)} files with {about.repo} (cached after the first run)")
    vectors = pretrained.embed(
        [item for item, _ in keep], model, cache_name, root, workers, _progress(echo)
    )
    rows = [i for i, v in enumerate(vectors) if v is not None]
    if not rows:
        raise sorting_sim.NotEnoughData(f"No files could be embedded with {model}")
    if len(rows) < len(keep):
        echo(f"  {len(keep) - len(rows)} files could not be embedded and are left out")
    hand = x[rows]
    learned = np.vstack([vectors[i] for i in rows])
    sub_labels = [labels[i] for i in rows]
    sub_artists = [artists[i] for i in rows]
    tempo = sorting_sim.find_tempo_column(hand.shape[1])
    variants = {
        f"+{model}:features": (hand, tempo),
        f"+{model}:embedding": (learned, None),
        f"+{model}:both": (np.hstack([hand, learned]), tempo),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for suffix, (matrix, column) in variants.items():
        echo(f"  simulating {suffix[1:]} ({matrix.shape[1]} dimensions)")
        result = sorting_sim.simulate(
            matrix, sub_labels, sub_artists, settings, tempo_column=column, infer_tempo=False
        )
        result["tracks"] = len(rows)
        result["dimensions"] = int(matrix.shape[1])
        result["model"] = {"repo": about.repo, "licence": about.licence, "about": about.about}
        out[suffix] = result
    return out


def _store(
    results: Dict[str, Dict[str, Any]], suite: str, found: Dict[str, Dict[str, Any]]
) -> None:
    for suffix, result in found.items():
        results[suite + suffix] = result


def _progress(echo: Echo) -> Callable[[int, int], None]:
    def report(done: int, total: int) -> None:
        if done == total or done % max(1, total // 10) == 0:
            echo(f"    {done}/{total}")

    return report


def run(
    suites: Sequence[str],
    root: Optional[str] = None,
    quick: bool = False,
    trials: int = 50,
    workers: int = -1,
    limit: Optional[int] = None,
    folders: Optional[str] = None,
    seed: int = 0,
    echo: Echo = print,
    embedding: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    """Run each suite; a suite whose data is missing is reported, not fatal.

    Raises:
        ValueError: An unknown suite name. Checked before anything runs.
    """
    results: Dict[str, Dict[str, Any]] = {}
    settings = sorting_sim.SimulationSettings(trials=trials, seed=seed)

    def trim(items: List[datasets.Item]) -> List[datasets.Item]:
        if not limit or len(items) <= limit:
            return items
        rng = np.random.default_rng(seed)
        picked = sorted(rng.choice(len(items), size=limit, replace=False))
        return [items[i] for i in picked]

    if embedding:
        pretrained.spec(embedding)
        pretrained.require()
    wanted = list(suites)
    if folders and "sorting-folders" not in wanted:
        wanted.append("sorting-folders")
    unknown = [s for s in wanted if s not in (*ALL_SUITES, "sorting-folders")]
    if unknown:
        raise ValueError(f"Unknown suite {unknown[0]!r}; choose from {', '.join(ALL_SUITES)}")

    for suite in wanted:
        echo(f"{suite}:")
        try:
            if suite == "beats-synthetic":
                results[suite] = tag_suites.run_beats(quick=quick, workers=workers)
            elif suite == "keys-synthetic":
                results[suite] = tag_suites.run_keys(workers=workers)
            elif suite == "sorting-fma":
                items = trim(datasets.fma(root, subset="small", level="top"))
                _store(
                    results,
                    suite,
                    _sorting(items, "fma-small", settings, workers, root, echo, embedding),
                )
            elif suite == "sorting-fma-sub":
                # Sub-genres need the bigger subset to have enough tracks each.
                items = trim(datasets.fma(root, subset="medium", level="sub"))
                counts: Dict[str, int] = {}
                for item in items:
                    counts[item.label or ""] = counts.get(item.label or "", 0) + 1
                items = [item for item in items if counts[item.label or ""] >= 40]
                _store(
                    results,
                    suite,
                    _sorting(items, "fma-medium", settings, workers, root, echo, embedding),
                )
            elif suite == "sorting-folders":
                if not folders:
                    raise sorting_sim.NotEnoughData("Pass --folders PARENT to score a folder tree")
                items = trim(datasets.folder_tree(folders))
                _store(
                    results,
                    suite,
                    _sorting(items, "folders", settings, workers, root, echo, embedding),
                )
            elif suite == "key-giantsteps":
                items = trim(datasets.giantsteps_key(root))
                echo(f"  analysing {len(items)} previews")
                records = analysis.analyze(
                    items,
                    "giantsteps-key",
                    tags=True,
                    workers=workers,
                    root=root,
                    progress=_progress(echo),
                )
                results[suite] = tag_suites.score_keys(items, records)
            elif suite == "tempo-giantsteps":
                items = trim(datasets.giantsteps_tempo(root))
                echo(f"  analysing {len(items)} previews")
                records = analysis.analyze(
                    items,
                    "giantsteps-tempo",
                    tags=True,
                    workers=workers,
                    root=root,
                    progress=_progress(echo),
                )
                results[suite] = tag_suites.score_tempo(items, records)
            else:
                raise ValueError(f"Unknown suite {suite!r}; choose from {', '.join(ALL_SUITES)}")
        except FileNotFoundError as exc:
            echo(f"  skipped: data not found ({exc}). Run `cratewise eval download` first.")
            results[suite] = {"skipped": f"data not found: {exc}"}
        except sorting_sim.NotEnoughData as exc:
            # Only "the data can't support this" is a skip. Any other error is
            # a bug and propagates, so a broken suite can never pass the
            # regression gate by being reported as skipped.
            echo(f"  skipped: {exc}")
            results[suite] = {"skipped": str(exc)}
    return results
