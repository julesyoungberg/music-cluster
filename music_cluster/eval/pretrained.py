"""Pretrained audio embeddings, measured as an extra feature set.

The sorter's 96 hand-made features look like the ceiling on every sorting
suite: on FMA the best classifier on them gets about 45%, and the confusions
on a real DJ library (deep vs future dub, club vs garage) are about texture
and style, not tempo or key. A network trained on a lot of music might
separate those. This module computes such an embedding for the eval datasets
so `cratewise eval run --embedding NAME` can score it with the same
simulated libraries as the hand features: the embedding on its own, and
appended to the hand features.

The models themselves live in `music_cluster/pretrained.py`, which the app
uses too. What this module adds is a per-file cache for the eval datasets,
keyed by the model, so a second run is instant.
"""

import json
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..library import resolve_worker_count
from ..pretrained import (  # noqa: F401  (check is used by the runner)
    BATCH,
    WINDOWS,
    Embedder,
    MissingDependency,
    _loaded,
    _pool,
    check,
    spec,
    windows,
)
from .analysis import _stamp
from .datasets import Item, data_dir


Progress = Callable[[int, int], None]


def cache_path(cache_name: str, model: str, root: Optional[str] = None) -> Path:
    return data_dir(root) / ".cache" / f"{cache_name}.{model}.jsonl"


def embed(
    items: Sequence[Item],
    model_name: str,
    cache_name: str,
    root: Optional[str] = None,
    workers: int = -1,
    progress: Optional[Progress] = None,
    embedder: Optional[Embedder] = None,
) -> List[Optional[np.ndarray]]:
    """One embedding per item, in order; None where a file could not be decoded.

    Args:
        embedder: Use this instead of loading the named model (for tests).
    """
    model = spec(model_name)
    path = cache_path(cache_name, model.name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    signature = f"{model.repo}|{model.window}|{WINDOWS}"

    cached: Dict[str, np.ndarray] = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("signature") == signature:
                    cached[entry["key"]] = np.asarray(entry["vector"], dtype=float)

    keys: List[str] = []
    todo: List[Tuple[Item, str]] = []
    for item in items:
        try:
            key = f"{item.path}|{_stamp(item.path)}"
        except OSError:
            key = f"{item.path}|missing"
        keys.append(key)
        if key not in cached and not key.endswith("|missing"):
            todo.append((item, key))

    if todo:
        run = embedder or _loaded(model.name)
        total = len(todo)
        done = 0
        batches = [todo[start : start + BATCH] for start in range(0, total, BATCH)]
        with (
            open(path, "a", encoding="utf-8") as out,
            ThreadPoolExecutor(max_workers=resolve_worker_count(workers)) as pool,
        ):

            def decode(batch: List[Tuple[Item, str]]) -> List["Future[Optional[List[np.ndarray]]]"]:
                return [pool.submit(windows, item.path, model) for item, _ in batch]

            # The next batch decodes on the threads while the model works on this one.
            pending = decode(batches[0])
            for index, batch in enumerate(batches):
                decoded = [future.result() for future in pending]
                if index + 1 < len(batches):
                    pending = decode(batches[index + 1])
                clips = [clip for found in decoded if found for clip in found]
                rows = np.asarray(run(clips), dtype=float) if clips else np.zeros((0, 0))
                if rows.shape[0] != len(clips):
                    # A wrong count would pool one track's windows into another's vector.
                    raise RuntimeError(
                        f"{model.repo} returned {rows.shape[0]} rows for {len(clips)} clips"
                    )
                cursor = 0
                for (_, key), found in zip(batch, decoded, strict=True):
                    done += 1
                    if progress:
                        progress(done, total)
                    if not found:
                        continue
                    vector = _pool(rows[cursor : cursor + len(found)])
                    cursor += len(found)
                    cached[key] = vector
                    out.write(
                        json.dumps(
                            {
                                "key": key,
                                "signature": signature,
                                "vector": [round(float(v), 6) for v in vector],
                            }
                        )
                        + "\n"
                    )
        # The model is only needed while embedding; free its memory for the simulations.
        _loaded.cache_clear()

    return [cached.get(key) for key in keys]
