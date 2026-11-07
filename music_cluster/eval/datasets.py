"""Labelled audio the harness can score against.

Each loader reads a dataset already on disk (see :mod:`.download`) and yields
:class:`Item` records: a file, its label, and — where the dataset knows it —
the artist, so splits can keep one artist's tracks on one side.
"""

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..utils import DEFAULT_AUDIO_EXTENSIONS, parse_extensions


DEFAULT_DATA_DIR = "~/.music-cluster-eval/data"


@dataclass
class Item:
    path: str
    label: Optional[str] = None
    artist: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


def data_dir(path: Optional[str] = None) -> Path:
    return Path(
        os.path.expanduser(path or os.environ.get("MUSIC_CLUSTER_EVAL_DATA") or DEFAULT_DATA_DIR)
    )


# ----------------------------------------------------------------------
# FMA — Free Music Archive (Defferrard et al., 2017), CC-licensed audio
# ----------------------------------------------------------------------


def _read_fma_tracks(path: Path) -> List[Dict[str, str]]:
    """tracks.csv has a three-row header (group, column, index); flatten it."""
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        groups = next(reader)
        columns = next(reader)
        next(reader)  # "track_id" row
        names = []
        current = ""
        for group, column in zip(groups, columns, strict=False):
            current = group or current
            names.append(f"{current}.{column}" if column else "track_id")
        names[0] = "track_id"
        return [dict(zip(names, row, strict=False)) for row in reader]


def _read_fma_genres(path: Path) -> Dict[str, Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["genre_id"]: row for row in csv.DictReader(handle)}


def fma(root: Optional[str] = None, subset: str = "small", level: str = "top") -> List[Item]:
    """FMA tracks with their genre.

    Args:
        subset: ``small`` (8 balanced top genres, 8,000 clips) or ``medium``
            (16 unbalanced top genres, 25,000 clips). Both are 30 s excerpts.
        level: ``top`` labels each track with its top-level genre (Electronic,
            Hip-Hop, …). ``sub`` keeps tracks tagged with exactly one
            sub-genre (Techno, House, Dubstep, …) and labels them with it — the
            granularity a DJ's folders actually have.
    """
    base = data_dir(root) / "fma"
    tracks = _read_fma_tracks(base / "fma_metadata" / "tracks.csv")
    genres = _read_fma_genres(base / "fma_metadata" / "genres.csv")
    allowed = {"small"} if subset == "small" else {"small", "medium"}
    audio = base / f"fma_{subset}"

    items = []
    for row in tracks:
        if row.get("set.subset") not in allowed:
            continue
        if level == "top":
            label = row.get("track.genre_top") or None
        else:
            ids = [
                g.strip() for g in row.get("track.genres", "").strip("[]").split(",") if g.strip()
            ]
            subs = [genres[g]["title"] for g in ids if g in genres and genres[g]["parent"] != "0"]
            label = subs[0] if len(subs) == 1 else None
        if not label:
            continue
        tid = f"{int(row['track_id']):06d}"
        path = audio / tid[:3] / f"{tid}.mp3"
        if not path.exists():
            continue
        items.append(
            Item(
                path=str(path),
                label=label,
                artist=row.get("artist.id") or None,
                meta={"track_id": tid, "split": row.get("set.split")},
            )
        )
    return items


# ----------------------------------------------------------------------
# GiantSteps — Beatport two-minute previews of electronic music
# (Knees et al., ISMIR 2015), key and tempo annotations
# ----------------------------------------------------------------------


def _giantsteps(repo: Path, kind: str, suffix: str) -> List[Item]:
    annotations = None
    for candidate in (repo / "annotations_v2" / kind, repo / "annotations" / kind):
        if candidate.is_dir():
            annotations = candidate
            break
    if annotations is None:
        raise FileNotFoundError(f"No {kind} annotations under {repo}")

    genre_dir = repo / "annotations" / "genre"
    items = []
    for annotation in sorted(annotations.glob(f"*{suffix}")):
        stem = annotation.name[: -len(suffix)]
        audio = repo / "audio" / f"{stem}.mp3"
        if not audio.exists():
            continue
        label = annotation.read_text(encoding="utf-8", errors="replace").strip().splitlines()
        genre_file = genre_dir / f"{stem}.genre"
        genre = (
            genre_file.read_text(encoding="utf-8", errors="replace").strip()
            if genre_file.exists()
            else None
        )
        items.append(
            Item(
                path=str(audio),
                label=label[0].strip() if label else None,
                meta={"id": stem, "genre": genre},
            )
        )
    return items


def giantsteps_key(root: Optional[str] = None) -> List[Item]:
    """604 previews; label is the annotated key, e.g. ``"Eb minor"``."""
    return _giantsteps(data_dir(root) / "giantsteps-key-dataset", "key", ".key")


def giantsteps_tempo(root: Optional[str] = None) -> List[Item]:
    """664 previews; label is the annotated BPM as text (v2 corrections when present)."""
    return _giantsteps(data_dir(root) / "giantsteps-tempo-dataset", "tempo", ".bpm")


# ----------------------------------------------------------------------
# A folder tree: each subfolder is a label — the shape of a DJ's own library
# ----------------------------------------------------------------------


def folder_tree(
    parent: str, extensions: Optional[str] = None, read_artists: bool = True
) -> List[Item]:
    """Every audio file under each immediate subfolder, labelled with that subfolder's name.

    Reads only; nothing is moved or written.
    """
    allowed = parse_extensions(extensions) if extensions else set(DEFAULT_AUDIO_EXTENSIONS)
    items = []
    root = Path(os.path.expanduser(parent))
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        for dirpath, _dirs, files in os.walk(folder):
            for name in sorted(files):
                if Path(name).suffix.lower() not in allowed:
                    continue
                path = os.path.join(dirpath, name)
                artist = None
                if read_artists:
                    try:
                        from ..metadata import read_metadata

                        artist = read_metadata(path).get("artist") or None
                    except Exception:
                        artist = None
                items.append(Item(path=path, label=folder.name, artist=artist))
    return items
