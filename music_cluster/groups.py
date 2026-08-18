"""Collections and reference groups.

A *collection* is one sorting scheme — typically "my DJ folders". A *group* is
one destination inside it: a genre folder on disk, a playlist, or a handful of
hand-picked seed tracks. Groups are what new music gets sorted into, and their
member tracks are the references that define what each group sounds like.
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from urllib.parse import unquote, urlparse

import numpy as np

from . import profiles
from .config import Config
from .database import Database
from .library import AnalysisResult, analyze_files, collect_files
from .sorter import GroupSorter, SorterConfig
from .utils import DEFAULT_AUDIO_EXTENSIONS


PLAYLIST_EXTENSIONS = {".m3u", ".m3u8", ".pls", ".txt"}


@dataclass
class ImportReport:
    """What happened when reference tracks were added to a group."""

    group_id: int
    group_name: str
    added: int = 0
    already_present: int = 0
    analysis: Optional[AnalysisResult] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "group_name": self.group_name,
            "added": self.added,
            "already_present": self.already_present,
            "analysis": self.analysis.to_dict() if self.analysis else None,
        }


# ----------------------------------------------------------------------
# Collections
# ----------------------------------------------------------------------


def ensure_collection(
    db: Database,
    name: str,
    description: Optional[str] = None,
    profile: Optional[str] = None,
) -> Dict:
    """Fetch a collection by name, creating it if necessary."""
    collection = db.get_collection(name=name)
    if collection:
        return collection
    db.create_collection(name=name, description=description, profile=profile)
    return db.get_collection(name=name)


def profile_of(collection: Optional[Dict]) -> str:
    """The audio profile a collection sorts, defaulting to music.

    Fixed when the collection is created: the stored feature vectors, the
    fitted embedding and the group references all assume one profile, so
    switching would invalidate every one of them. Sorting samples as well as
    tracks means a second collection, not a setting.
    """
    return profiles.normalize((collection or {}).get("profile"))


def resolve_collection(
    db: Database, identifier: Optional[Any] = None, required: bool = True
) -> Optional[Dict]:
    """Resolve a collection from an ID, a name, or the most recent one."""
    if identifier is None:
        collection = db.get_collection()
    elif isinstance(identifier, int) or (isinstance(identifier, str) and identifier.isdigit()):
        collection = db.get_collection(collection_id=int(identifier))
    else:
        collection = db.get_collection(name=str(identifier))

    if collection is None and required:
        raise LookupError(
            f"No collection found for {identifier!r}. Create one with "
            "`music-cluster collection create <name>`."
        )
    return collection


def sorter_config_for(collection: Dict, config: Config) -> SorterConfig:
    """Per-collection sorting settings over the profile's, over the global."""
    merged = config.section("sorting", profile_of(collection))
    merged.update(collection.get("settings", {}).get("sorting", {}) or {})
    return SorterConfig.from_dict(merged)


# ----------------------------------------------------------------------
# Groups
# ----------------------------------------------------------------------


def resolve_group(db: Database, collection_id: int, identifier: Any) -> Dict:
    """Resolve a group within a collection from an ID or a name."""
    if isinstance(identifier, int) or (isinstance(identifier, str) and str(identifier).isdigit()):
        group = db.get_group(group_id=int(identifier))
        if group and group["collection_id"] == collection_id:
            return group
    group = db.get_group(collection_id=collection_id, name=str(identifier))
    if not group:
        raise LookupError(f"No group named {identifier!r} in this collection")
    return group


def create_group(
    db: Database,
    collection_id: int,
    name: str,
    description: Optional[str] = None,
    destination_path: Optional[str] = None,
    source_kind: str = "manual",
    source_path: Optional[str] = None,
    color: Optional[str] = None,
) -> Dict:
    """Create a group, or return the existing one with the same name."""
    existing = db.get_group(collection_id=collection_id, name=name)
    if existing:
        return existing

    group_id = db.create_group(
        collection_id=collection_id,
        name=name,
        description=description,
        color=color,
        destination_path=os.path.abspath(os.path.expanduser(destination_path))
        if destination_path
        else None,
        source_kind=source_kind,
        source_path=os.path.abspath(os.path.expanduser(source_path)) if source_path else None,
    )
    db.touch_collection(collection_id)
    return db.get_group(group_id=group_id)


def add_tracks_to_group(
    db: Database,
    group: Dict,
    filepaths: Sequence[str],
    config: Config,
    role: str = "reference",
    update: bool = False,
    workers: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> ImportReport:
    """Analyse files if needed and register them as references for a group."""
    report = ImportReport(group_id=group["id"], group_name=group["name"])

    collection = db.get_collection(collection_id=group["collection_id"])
    analysis = analyze_files(
        db,
        config,
        filepaths,
        update=update,
        workers=workers,
        progress=progress,
        profile=profile_of(collection),
    )
    report.analysis = analysis

    existing = set(db.get_group_member_ids(group["id"]))
    new_ids = [track_id for track_id in analysis.track_ids if track_id not in existing]
    report.already_present = len(analysis.track_ids) - len(new_ids)

    if new_ids:
        db.add_group_members(group["id"], new_ids, role=role)
        report.added = len(new_ids)
        db.touch_collection(group["collection_id"])

    return report


def import_folder_as_group(
    db: Database,
    collection_id: int,
    folder: str,
    config: Config,
    name: Optional[str] = None,
    recursive: bool = True,
    set_destination: bool = True,
    workers: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> ImportReport:
    """Import an existing genre folder as a reference group.

    The folder doubles as the group's destination, so new music sorted into
    this group lands where the DJ already keeps that genre.
    """
    folder = os.path.abspath(os.path.expanduser(folder))
    if not os.path.isdir(folder):
        raise ValueError(f"Not a directory: {folder}")

    group = create_group(
        db,
        collection_id,
        name=name or Path(folder).name,
        destination_path=folder if set_destination else None,
        source_kind="folder",
        source_path=folder,
    )
    files = collect_files([folder], recursive=recursive)
    if not files:
        raise ValueError(f"No audio files found in {folder}")

    return add_tracks_to_group(db, group, files, config, workers=workers, progress=progress)


def import_folders_as_groups(
    db: Database,
    collection_id: int,
    parent: str,
    config: Config,
    recursive: bool = True,
    min_tracks: int = 1,
    workers: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> List[ImportReport]:
    """Import every immediate subfolder of ``parent`` as its own group.

    This is the one-shot path for a DJ whose library is already a folder of
    genre folders.
    """
    parent = os.path.abspath(os.path.expanduser(parent))
    if not os.path.isdir(parent):
        raise ValueError(f"Not a directory: {parent}")

    reports: List[ImportReport] = []
    for entry in sorted(os.scandir(parent), key=lambda e: e.name.lower()):
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        files = collect_files([entry.path], recursive=recursive)
        if len(files) < min_tracks:
            continue
        group = create_group(
            db,
            collection_id,
            name=entry.name,
            destination_path=entry.path,
            source_kind="folder",
            source_path=entry.path,
        )
        reports.append(
            add_tracks_to_group(db, group, files, config, workers=workers, progress=progress)
        )
    return reports


def import_playlist_as_group(
    db: Database,
    collection_id: int,
    playlist_path: str,
    config: Config,
    name: Optional[str] = None,
    destination_path: Optional[str] = None,
    workers: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
) -> ImportReport:
    """Import an M3U/PLS playlist as a reference group."""
    playlist_path = os.path.abspath(os.path.expanduser(playlist_path))
    entries = read_playlist(playlist_path)
    if not entries:
        raise ValueError(f"No playable entries found in {playlist_path}")

    group = create_group(
        db,
        collection_id,
        name=name or Path(playlist_path).stem,
        destination_path=destination_path,
        source_kind="playlist",
        source_path=playlist_path,
    )
    return add_tracks_to_group(db, group, entries, config, workers=workers, progress=progress)


def read_playlist(playlist_path: str) -> List[str]:
    """Read playable file paths out of an M3U/M3U8/PLS/plain-text playlist."""
    base = os.path.dirname(playlist_path)
    entries: List[str] = []

    with open(playlist_path, "r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            # PLS lines look like `File1=/path/to/track.mp3`.
            match = re.match(r"^File\d+\s*=\s*(.+)$", line, re.IGNORECASE)
            if match:
                line = match.group(1).strip()

            if line.startswith("file://"):
                line = unquote(urlparse(line).path)
            elif "://" in line:
                continue  # remote stream, nothing to analyse

            path = line if os.path.isabs(line) else os.path.normpath(os.path.join(base, line))
            if os.path.isfile(path) and Path(path).suffix.lower() in DEFAULT_AUDIO_EXTENSIONS:
                entries.append(os.path.abspath(path))

    return entries


def looks_like_playlist(path: str) -> bool:
    return Path(path).suffix.lower() in PLAYLIST_EXTENSIONS


# ----------------------------------------------------------------------
# Fitting
# ----------------------------------------------------------------------


def load_group_features(
    db: Database, collection_id: int, profile: Optional[str] = None
) -> Tuple[Dict[int, np.ndarray], Dict[int, str], Dict[int, List[int]], List[Dict]]:
    """Load reference features for every group in a collection.

    Returns ``(features_by_group, names_by_group, track_ids_by_group, problems)``
    where *problems* lists groups that are empty or missing features.
    """
    profile = profiles.normalize(profile)
    groups = db.list_groups(collection_id)
    features: Dict[int, np.ndarray] = {}
    names: Dict[int, str] = {}
    track_ids: Dict[int, List[int]] = {}
    problems: List[Dict] = []

    for group in groups:
        member_ids = db.get_group_member_ids(group["id"])
        names[group["id"]] = group["name"]
        if not member_ids:
            problems.append(
                {"group_id": group["id"], "name": group["name"], "issue": "no reference tracks"}
            )
            continue

        matrix, found_ids = db.get_feature_matrix(member_ids, profile)
        if len(found_ids) == 0:
            problems.append(
                {
                    "group_id": group["id"],
                    "name": group["name"],
                    "issue": f"no tracks analysed for the {profile} profile",
                }
            )
            continue
        if len(found_ids) < len(member_ids):
            problems.append(
                {
                    "group_id": group["id"],
                    "name": group["name"],
                    "issue": f"{len(member_ids) - len(found_ids)} member(s) not analysed",
                }
            )

        features[group["id"]] = matrix
        track_ids[group["id"]] = found_ids

    return features, names, track_ids, problems


def fit_collection(
    db: Database, collection: Dict, config: Config, evaluate: bool = True
) -> Dict[str, Any]:
    """Fit and persist the sorter for a collection.

    Returns the evaluation report, which doubles as feedback on whether the
    DJ's folders are actually distinguishable from each other.
    """
    features, names, track_ids, problems = load_group_features(
        db, collection["id"], profile_of(collection)
    )
    if len(features) < 2:
        raise ValueError(
            "At least two groups with analysed reference tracks are required. "
            + (f"Problems: {problems}" if problems else "")
        )

    sorter = GroupSorter(
        sorter_config_for(collection, config), profile_of(collection)
    ).fit(features, names)
    metrics = sorter.evaluate() if evaluate else {}
    metrics["problems"] = problems
    metrics["profile"] = profile_of(collection)
    metrics["group_track_ids"] = {str(gid): ids for gid, ids in track_ids.items()}

    db.save_model(
        collection_id=collection["id"],
        payload=sorter.dumps(),
        params=sorter.config.to_dict(),
        metrics=metrics,
        n_groups=len(features),
        n_tracks=sum(len(matrix) for matrix in features.values()),
    )
    return metrics


def load_sorter(db: Database, collection: Dict, config: Config, autofit: bool = True):
    """Load the fitted sorter for a collection, fitting on demand if stale.

    Returns ``(sorter, model_row)``.
    """
    model = db.load_model(collection["id"])
    stale = model is None or (
        collection.get("updated_at") and model.get("fitted_at")
        and str(model["fitted_at"]) < str(collection["updated_at"])
    )

    if stale:
        if not autofit:
            raise LookupError(
                "No up-to-date sorter for this collection. Run `music-cluster fit` first."
            )
        fit_collection(db, collection, config)
        model = db.load_model(collection["id"])

    return GroupSorter.loads(model["payload"]), model
