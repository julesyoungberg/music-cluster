"""Library ingestion: turn files on disk into analysed tracks in the database.

Shared by the CLI and the API so both take exactly the same path through
feature extraction and metadata reading.
"""

import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set

from . import profiles, tagging
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
    tagged: int = 0
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
            "tagged": self.tagged,
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


def _extract_one(filepath: str, extractor_config: Dict, tag: bool = False) -> Dict:
    """Worker: extract features and metadata (and optionally tags) for one file."""
    try:
        extractor = FeatureExtractor(**extractor_config)
        audio = extractor.load(filepath)
        features = extractor.extract_audio(*audio, filepath=filepath) if audio else None
        if features is None:
            return {"filepath": filepath, "error": "Could not decode audio"}

        metadata = read_metadata(filepath)
        payload = {
            "filepath": filepath,
            "features": features,
            "duration": extractor.get_audio_duration(filepath),
            "file_info": get_file_info(filepath),
            "checksum": compute_file_checksum(filepath),
            "metadata": metadata,
        }
        if tag and audio is not None:
            # Same decoded excerpt the features came from: tagging costs
            # measurement time here, never a second read of the file.
            tags = tagging.tag_audio(audio[0], audio[1], metadata)
            if tags is not None:
                payload["tags"] = tags.to_dict()
        return payload
    except Exception as exc:
        return {"filepath": filepath, "error": str(exc)}


def analyze_files(
    db: Database,
    config: Config,
    filepaths: Sequence[str],
    update: bool = False,
    workers: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
    profile: Optional[str] = None,
) -> AnalysisResult:
    """Analyse the given files, skipping ones already in the database.

    Tracks that are already analysed still have their IDs returned, so callers
    can use this as "make sure these files are usable" rather than only as an
    import step.

    Args:
        profile: Which audio profile to analyse under. A file already analysed
            as music is still re-analysed for the sample profile — the two
            produce different vectors and both are kept.
    """
    result = AnalysisResult()
    profile = profiles.normalize(profile)
    extractor_config = config.extractor_config(profile)
    # Tags describe songs — a beat type or a mood means nothing for a kick —
    # so only music is tagged.
    tag = profile == profiles.MUSIC and bool(config.get("tagging", "on_analyze", default=True))
    analysis_version = config.get("feature_extraction", "analysis_version", default="2.0.0")

    pending: List[str] = []
    for filepath in filepaths:
        existing = db.get_track_by_filepath(filepath)
        if existing and not update and db.get_features(existing["id"], profile) is not None:
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
            db.add_features(track_id, payload["features"], profile)
            if payload.get("tags"):
                db.save_tags(track_id, payload["tags"])
                result.tagged += 1
            result.analyzed += 1
            result.track_ids.append(track_id)
        if progress:
            progress(done, total, payload["filepath"])

    if worker_count == 1:
        for filepath in pending:
            handle(_extract_one(filepath, extractor_config, tag))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_extract_one, filepath, extractor_config, tag): filepath
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
    profile: Optional[str] = None,
) -> AnalysisResult:
    """Analyse every audio file under the given paths."""
    files = collect_files(paths, recursive=recursive, extensions=extensions)
    return analyze_files(
        db,
        config,
        files,
        update=update,
        workers=workers,
        progress=progress,
        profile=profile,
    )


@dataclass
class TaggingResult:
    """Outcome of a tagging run."""

    tagged: int = 0
    skipped: int = 0
    failed: int = 0
    track_ids: List[int] = field(default_factory=list)
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "tagged": self.tagged,
            "skipped": self.skipped,
            "failed": self.failed,
            "track_ids": self.track_ids,
            "errors": self.errors[:50],
        }


def _tag_one(filepath: str, extractor_config: Dict, metadata: Dict) -> Dict:
    """Worker: decode one file and tag it."""
    try:
        audio = FeatureExtractor(**extractor_config).load(filepath)
        if audio is None:
            return {"filepath": filepath, "error": "Could not decode audio"}
        tags = tagging.tag_audio(audio[0], audio[1], metadata)
        if tags is None:
            return {"filepath": filepath, "error": "Could not measure audio"}
        return {"filepath": filepath, "tags": tags.to_dict()}
    except Exception as exc:
        return {"filepath": filepath, "error": str(exc)}


def tag_tracks(
    db: Database,
    config: Config,
    track_ids: Optional[Sequence[int]] = None,
    update: bool = False,
    workers: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
) -> TaggingResult:
    """Tag tracks already in the library.

    Analysis tags new music as it goes; this is for everything analysed
    before tagging existed, for tags left behind by an older tagger version,
    and (with ``update``) for re-tagging on demand. Decodes each file again,
    since the library keeps feature vectors, not audio.

    Args:
        track_ids: Which tracks. None means every track analysed as music.
        update: Re-tag tracks that already have current tags.
    """
    result = TaggingResult()
    if track_ids is None:
        candidates = (
            [row["id"] for row in db.list_tracks(limit=10**9, profile=profiles.MUSIC)]
            if update
            else db.tracks_needing_tags(None, tagging.TAGGER_VERSION)
        )
    else:
        candidates = (
            list(track_ids) if update else db.tracks_needing_tags(track_ids, tagging.TAGGER_VERSION)
        )
        result.skipped = len(track_ids) - len(candidates)

    tracks = db.get_tracks(candidates)
    pending = [tracks[tid] for tid in candidates if tid in tracks]
    if not pending:
        if progress:
            progress(0, 0, "")
        return result

    extractor_config = config.extractor_config(profiles.MUSIC)
    worker_count = resolve_worker_count(
        workers if workers is not None else config.get("performance", "num_workers", default=-1)
    )
    by_path = {track["filepath"]: track for track in pending}
    done = 0
    total = len(pending)

    def handle(payload: Dict) -> None:
        nonlocal done
        done += 1
        track = by_path[payload["filepath"]]
        if "error" in payload:
            result.failed += 1
            result.errors.append({"filepath": payload["filepath"], "error": payload["error"]})
        else:
            db.save_tags(track["id"], payload["tags"])
            result.tagged += 1
            result.track_ids.append(track["id"])
        if progress:
            progress(done, total, payload["filepath"])

    def job(track: Dict) -> Dict:
        if not os.path.isfile(track["filepath"]):
            return {"filepath": track["filepath"], "error": "File not found"}
        return _tag_one(track["filepath"], extractor_config, track)

    if worker_count == 1:
        for track in pending:
            handle(job(track))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for future in as_completed([executor.submit(job, track) for track in pending]):
                handle(future.result())

    return result


def prune_missing(db: Database) -> List[int]:
    """Remove tracks whose files no longer exist. Returns the removed IDs."""
    removed: List[int] = []
    for track in db.list_tracks(limit=10**9):
        if not os.path.exists(track["filepath"]):
            db.delete_track(track["id"])
            removed.append(track["id"])
    return removed
