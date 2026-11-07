"""Run named suites and collect their results."""

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from . import analysis, datasets, sorting_sim, tag_suites


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
) -> Dict[str, Any]:
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
        raise ValueError("No files could be analysed")
    x = np.array([features for _, features in keep], dtype=float)
    labels = [item.label or "" for item, _ in keep]
    artists = [item.artist for item, _ in keep]
    echo(f"  simulating {settings.trials} libraries over {len(set(labels))} labels")
    result = sorting_sim.simulate(x, labels, artists, settings)
    result["tracks"] = len(keep)
    result["labels"] = sorted(set(labels))
    return result


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
) -> Dict[str, Dict[str, Any]]:
    """Run each suite; a suite whose data is missing is reported, not fatal."""
    results: Dict[str, Dict[str, Any]] = {}
    settings = sorting_sim.SimulationSettings(trials=trials, seed=seed)

    def trim(items: List[datasets.Item]) -> List[datasets.Item]:
        if not limit or len(items) <= limit:
            return items
        rng = np.random.default_rng(seed)
        picked = sorted(rng.choice(len(items), size=limit, replace=False))
        return [items[i] for i in picked]

    wanted = list(suites)
    if folders and "sorting-folders" not in wanted:
        wanted.append("sorting-folders")

    for suite in wanted:
        echo(f"{suite}:")
        try:
            if suite == "beats-synthetic":
                results[suite] = tag_suites.run_beats(quick=quick, workers=workers)
            elif suite == "keys-synthetic":
                results[suite] = tag_suites.run_keys(workers=workers)
            elif suite == "sorting-fma":
                items = trim(datasets.fma(root, subset="small", level="top"))
                results[suite] = _sorting(items, "fma-small", settings, workers, root, echo)
            elif suite == "sorting-fma-sub":
                # Sub-genres need the bigger subset to have enough tracks each.
                items = trim(datasets.fma(root, subset="medium", level="sub"))
                counts: Dict[str, int] = {}
                for item in items:
                    counts[item.label or ""] = counts.get(item.label or "", 0) + 1
                items = [item for item in items if counts[item.label or ""] >= 40]
                results[suite] = _sorting(items, "fma-medium", settings, workers, root, echo)
            elif suite == "sorting-folders":
                if not folders:
                    raise ValueError("Pass --folders PARENT to score a folder tree")
                items = trim(datasets.folder_tree(folders))
                results[suite] = _sorting(items, "folders", settings, workers, root, echo)
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
            echo(f"  skipped: data not found ({exc}). Run `music-cluster eval download` first.")
            results[suite] = {"skipped": f"data not found: {exc}"}
        except ValueError as exc:
            echo(f"  skipped: {exc}")
            results[suite] = {"skipped": str(exc)}
    return results
