"""Library ingestion: turn files on disk into analysed tracks in the database.

Shared by the CLI and the API so both take exactly the same path through
feature extraction and metadata reading.
"""

import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

from .config import Config
from .database import Database
from .extractor import FeatureExtractor
from .metadata import read_metadata
from .utils import compute_file_checksum, find_audio_files, get_file_info


ProgressCallback = Callable[[int, int, str], None]


@dataclass
class AnalysisResult:
    """Outcome of an ingestion run."""

    analyzed: int = 0
    skipped: int = 0
    failed: int = 0
    track_ids: List[int] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.analyzed + self.skipped + self.failed

    def to_dict(self) -> Dict:
        return {
            "analyzed": self.analyzed,
            "skipped": self.skipped,
            "failed": self.failed,
            "total": self.total,
            "track_ids": self.track_ids,
            "errors": self.errors[:50],
        }


def resolve_worker_count(workers: int) -> int:
    """Translate a configured worker count into an actual thread count."""
    if workers == -1:
        return max(1, mp.cpu_count())
    return max(1, workers)


def collect_files(
    paths: Sequence[str], recursive: bool = True, extensions: Optional[Set[str]] = None
) -> List[str]:
    """Expand paths (files or directories) into a de-duplicated file list."""
    found: List[str] = []
    seen: Set[str] = set()
    for path in paths:
        for filepath in find_audio_files(path, recursive=recursive, extensions=extensions):
            if filepath not in seen:
                seen.add(filepath)
                found.append(filepath)
    return found


def _extract_one(filepath: str, extractor_config: Dict) -> Dict:
    """Worker: extract features and metadata for one file."""
    try:
        extractor = FeatureExtractor(**extractor_config)
        features = extractor.extract(filepath)
        if features is None:
            return {"filepath": filepath, "error": "Could not decode audio"}

        return {
            "filepath": filepath,
            "features": features,
            "duration": extractor.get_audio_duration(filepath),
            "file_info": get_file_info(filepath),
            "checksum": compute_file_checksum(filepath),
            "metadata": read_metadata(filepath),
        }
    except Exception as exc:
        return {"filepath": filepath, "error": str(exc)}


def analyze_files(
    db: Database,
    config: Config,
    filepaths: Sequence[str],
    update: bool = False,
    workers: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
) -> AnalysisResult:
    """Analyse the given files, skipping ones already in the database.

    Tracks that are already analysed still have their IDs returned, so callers
    can use this as "make sure these files are usable" rather than only as an
    import step.
    """
    result = AnalysisResult()
    extractor_config = config.extractor_config()
    analysis_version = config.get("feature_extraction", "analysis_version", default="2.0.0")

    pending: List[str] = []
    for filepath in filepaths:
        existing = db.get_track_by_filepath(filepath)
        if existing and not update and db.get_features(existing["id"]) is not None:
            result.skipped += 1
            result.track_ids.append(existing["id"])
        else:
            pending.append(filepath)

    if not pending:
        if progress:
            progress(0, 0, "")
        return result

    worker_count = resolve_worker_count(
        workers if workers is not None else config.get("performance", "num_workers", default=-1)
    )
    done = 0
    total = len(pending)

    def handle(payload: Dict) -> None:
        nonlocal done
        done += 1
        if "error" in payload:
            result.failed += 1
            result.errors.append({"filepath": payload["filepath"], "error": payload["error"]})
        else:
            track_id = db.upsert_track(
                filepath=payload["filepath"],
                filename=payload["file_info"]["filename"],
                duration=payload["duration"],
                file_size=payload["file_info"]["file_size"],
                checksum=payload["checksum"],
                analysis_version=analysis_version,
                metadata=payload["metadata"],
            )
            db.add_features(track_id, payload["features"])
            result.analyzed += 1
            result.track_ids.append(track_id)
        if progress:
            progress(done, total, payload["filepath"])

    if worker_count == 1:
        for filepath in pending:
            handle(_extract_one(filepath, extractor_config))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_extract_one, filepath, extractor_config): filepath
                for filepath in pending
            }
            for future in as_completed(futures):
                try:
                    handle(future.result())
                except Exception as exc:
                    result.failed += 1
                    result.errors.append({"filepath": futures[future], "error": str(exc)})
                    done += 1
                    if progress:
                        progress(done, total, futures[future])

    return result


def analyze_paths(
    db: Database,
    config: Config,
    paths: Sequence[str],
    recursive: bool = True,
    update: bool = False,
    extensions: Optional[Set[str]] = None,
    workers: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
) -> AnalysisResult:
    """Analyse every audio file under the given paths."""
    files = collect_files(paths, recursive=recursive, extensions=extensions)
    return analyze_files(db, config, files, update=update, workers=workers, progress=progress)


def prune_missing(db: Database) -> List[int]:
    """Remove tracks whose files no longer exist. Returns the removed IDs."""
    removed: List[int] = []
    for track in db.list_tracks(limit=10**9):
        if not os.path.exists(track["filepath"]):
            db.delete_track(track["id"])
            removed.append(track["id"])
    return removed
