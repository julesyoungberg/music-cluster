"""Analyse dataset audio once and keep the results.

Decoding is the expensive part of every suite, so each file is decoded once
and everything the suites need is measured from that one excerpt: the
96-dimension feature vector the sorter uses, the auto-tags, and the reference
measurements the baselines need. Results go to a JSON-lines cache keyed by
path, size, modification time and analysis settings, so a second run of any
suite starts instantly and a changed setting re-analyses only what it affects.

Tags are computed with no file metadata, so a genre tag in the file can never
leak into a genre score.
"""

import hashlib
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from ..config import Config
from ..library import resolve_worker_count
from .datasets import Item, data_dir


logger = logging.getLogger(__name__)

Progress = Callable[[int, int], None]


def default_config() -> Config:
    """Library defaults, ignoring the user's own config file, so runs are comparable."""
    return Config(os.devnull)


def _signature(extractor_config: Dict[str, Any], tags: bool, version: str) -> str:
    payload = json.dumps({"x": extractor_config, "tags": tags, "v": version}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _stamp(path: str) -> str:
    stat = os.stat(path)
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def _essentia_reference(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Essentia's own key and tempo, when Essentia is installed (an optional baseline)."""
    try:
        import essentia.standard as es  # type: ignore[import-not-found]
    except Exception:
        return {}
    out: Dict[str, Any] = {}
    audio = np.asarray(y, dtype=np.float32)
    if sr != 44100:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=44100).astype(np.float32)
    try:
        bpm = es.RhythmExtractor2013(method="multifeature")(audio)[0]
        out["essentia_bpm"] = float(bpm)
    except Exception as exc:
        logger.debug("Essentia tempo failed: %s", exc)
    try:
        key, scale, _strength = es.KeyExtractor(profileType="edma")(audio)
        out["essentia_key"] = f"{key} {scale}"
    except Exception as exc:
        logger.debug("Essentia key failed: %s", exc)
    return out


def analyze_one(path: str, extractor_config: Dict[str, Any], tags: bool) -> Dict[str, Any]:
    """Worker: one decode, every measurement. Never raises."""
    import librosa

    from .. import tagging
    from ..extractor import FeatureExtractor

    try:
        extractor = FeatureExtractor(**extractor_config)
        audio = extractor.load(path)
        if audio is None:
            return {"error": "could not decode"}
        y, sr = audio
        features = extractor.extract_audio(y, sr, filepath=path)
        record: Dict[str, Any] = {
            "features": None if features is None else [float(v) for v in features],
        }
        if tags:
            tagged = tagging.tag_audio(y, sr)
            if tagged is not None:
                record.update(
                    {
                        "bpm": tagged.bpm,
                        "key": tagged.key,
                        "key_strength": tagged.confidence.get("key"),
                        "beat": tagged.beat,
                        "genre": tagged.genre,
                        "moods": tagged.moods,
                        "energy": tagged.energy,
                        "danceability": tagged.danceability,
                    }
                )
                # The detected key even when it is too weak to tag, so key
                # accuracy is measured on every track, not just the confident.
                descriptors = tagged.descriptors or {}
                record["key_raw"] = descriptors.get("key")
            # Baseline: the stock beat tracker on the same audio, without the
            # drum-pattern octave correction.
            tempo, _beats = librosa.beat.beat_track(y=y, sr=sr)
            record["librosa_bpm"] = float(np.atleast_1d(np.asarray(tempo, dtype=float))[0])
            record.update(_essentia_reference(y, sr))
        return record
    except Exception as exc:
        return {"error": str(exc)}


def cache_path(name: str, root: Optional[str] = None) -> Path:
    return data_dir(root).parent / "cache" / f"{name}.jsonl"


def analyze(
    items: Sequence[Item],
    cache_name: str,
    tags: bool = False,
    workers: int = 0,
    root: Optional[str] = None,
    progress: Optional[Progress] = None,
    config: Optional[Config] = None,
) -> List[Optional[Dict[str, Any]]]:
    """Measurements for every item, in order; None where a file failed to decode."""
    config = config or default_config()
    extractor_config = config.extractor_config("music")
    version = str(config.get("feature_extraction", "analysis_version", default=""))
    signature = _signature(extractor_config, tags, version)

    path = cache_path(cache_name, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    cached: Dict[str, Dict[str, Any]] = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("signature") == signature:
                    cached[f"{entry['path']}|{entry['stamp']}"] = entry["record"]

    keys = []
    todo = []
    for item in items:
        try:
            key = f"{item.path}|{_stamp(item.path)}"
        except OSError:
            key = f"{item.path}|missing"
        keys.append(key)
        if key not in cached and not key.endswith("|missing"):
            todo.append((item.path, key))

    if todo:
        done = 0
        total = len(todo)
        with (
            open(path, "a", encoding="utf-8") as out,
            ThreadPoolExecutor(max_workers=resolve_worker_count(workers)) as pool,
        ):
            futures = {
                pool.submit(analyze_one, file, extractor_config, tags): (file, key)
                for file, key in todo
            }
            for future in as_completed(futures):
                file, key = futures[future]
                record = future.result()
                cached[key] = record
                out.write(
                    json.dumps(
                        {
                            "path": file,
                            "stamp": key.rsplit("|", 1)[1],
                            "signature": signature,
                            "record": record,
                        }
                    )
                    + "\n"
                )
                done += 1
                if progress:
                    progress(done, total)

    results: List[Optional[Dict[str, Any]]] = []
    for key in keys:
        found = cached.get(key)
        results.append(None if found is None or "error" in found else found)
    return results
