"""Writing groups as Serato DJ crates.

Serato keeps its crates as small binary files in ``_Serato_/Subcrates`` (on a
Mac's start-up disk, ``~/Music/_Serato_``). Each collection becomes a crate
under one parent crate named ``Cratewise``, and each group a crate inside its
collection's, so everything cratewise writes sits in one place in Serato's
crate list and never mixes with the user's own crates.

This writes into the DJ app's own folder, so it follows the filing rules: a dry
run first, a refusal to replace any crate cratewise did not write, and an undo
(``remove_crates``) that deletes exactly the files it wrote. The files it wrote
are listed in ``_Serato_/cratewise-crates.json``, each with the collection it
belongs to and a fingerprint of its track list. A crate whose tracks the user
changed inside Serato no longer matches its fingerprint: it is theirs from then
on, so it is never replaced or deleted.

A crate file holds tagged records: a four-letter tag, a big-endian 32-bit
length, then the payload (text is UTF-16BE). The parts written here are the
version header, a few display columns, and one ``otrk``/``ptrk`` record per
track holding its path relative to the root of the drive the ``_Serato_``
folder is on. A track on another drive belongs to that drive's own
``_Serato_`` folder, so it is left out and counted.
"""

import hashlib
import json
import os
import struct
from pathlib import PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .database import Database
from .djexport import ExportConflict, _absolute, collect_playlists
from .organizer import _safe_filename


PARENT_CRATE = "Cratewise"
MANIFEST = "cratewise-crates.json"
FOLDER_NAME = "_Serato_"
_VERSION = "1.0/Serato ScratchLive Crate"
_COLUMNS = ("song", "artist", "bpm", "key")

# One manifest entry: the collection a crate belongs to (None for the shared
# parent crate) and the fingerprint of the track list cratewise wrote.
Manifest = Dict[str, Dict[str, Any]]


class CratesEdited(ExportConflict):
    """Crates cratewise wrote whose tracks were changed in Serato since."""

    def __init__(self, paths: Sequence[str]):
        super().__init__(paths)
        listed = ", ".join(self.paths[:5]) + (" ..." if len(self.paths) > 5 else "")
        self.args = (
            f"Not replacing {len(self.paths)} crate(s) whose tracks were changed in Serato: "
            f"{listed}. Rename or delete them in Serato first.",
        )


def write_crates(
    db: Database, collection: Dict[str, Any], serato_dir: str, dry_run: bool = False
) -> Dict[str, Any]:
    """Write one crate per group into ``serato_dir`` (the ``_Serato_`` folder).

    Returns ``{"files": [...], "removed": [...], "left_out": n}``: the crate
    files written, this collection's earlier crates deleted because their group
    is gone or renamed, and how many tracks were left out because they are on a
    different drive (on a dry run, what would be). Raises ``ExportConflict``
    before touching anything if a crate of the same name exists that cratewise
    did not write, or whose tracks were changed in Serato.
    """
    serato_dir = check_serato_dir(serato_dir)
    subcrates = os.path.join(serato_dir, "Subcrates")
    playlists = collect_playlists(db, db.list_groups(collection["id"]))
    if not playlists:
        return {"files": [], "removed": [], "left_out": 0}

    crates: Dict[str, Tuple[Optional[int], List[str]]] = {
        _crate_path(subcrates, []): (None, []),
        _crate_path(subcrates, [collection["name"]]): (collection["id"], []),
    }
    off_drive = set()
    taken = {os.path.basename(path).lower() for path in crates}
    for name, tracks in playlists:
        paths = []
        for track in tracks:
            path = serato_track_path(track["filepath"], serato_dir)
            if path is None:
                off_drive.add(track["filepath"])
            else:
                paths.append(path)
        crate = _unique(_crate_path(subcrates, [collection["name"], name]), taken)
        crates[crate] = (collection["id"], paths)

    manifest = _read_manifest(serato_dir)
    foreign = [
        path
        for path, (owner, _) in crates.items()
        if os.path.lexists(path)
        and (path not in manifest or manifest[path].get("collection") != owner)
    ]
    if foreign:
        raise ExportConflict(foreign)
    stale = [
        path
        for path, entry in manifest.items()
        if entry.get("collection") == collection["id"] and path not in crates
    ]
    edited = [path for path in list(crates) + stale if path in manifest and _edited(path, manifest)]
    if edited:
        raise CratesEdited(edited)
    stale = [path for path in stale if os.path.lexists(path)]
    if dry_run:
        return {"files": list(crates), "removed": stale, "left_out": len(off_drive)}

    os.makedirs(subcrates, exist_ok=True)
    for path in stale:
        del manifest[path]
    for path, (owner, track_paths) in crates.items():
        manifest[path] = {"collection": owner, "tracks": _fingerprint(track_paths)}
    # Record first, so a crash mid-write still leaves every file removable.
    _write_manifest(serato_dir, manifest)
    for path in stale:
        os.remove(path)
    for path, (_, track_paths) in crates.items():
        with open(path, "wb") as handle:
            handle.write(build_crate(track_paths))
    return {"files": list(crates), "removed": stale, "left_out": len(off_drive)}


def remove_crates(serato_dir: str, dry_run: bool = False) -> Dict[str, List[str]]:
    """Delete every crate cratewise wrote into ``serato_dir``.

    Returns ``{"files": [...], "kept": [...]}``: the crates deleted, and those
    kept because their tracks were changed in Serato (they are the user's now).
    """
    serato_dir = check_serato_dir(serato_dir)
    manifest = _read_manifest(serato_dir)
    present = [path for path in manifest if os.path.lexists(path)]
    kept = [path for path in present if _edited(path, manifest)]
    doomed = [path for path in present if path not in kept]
    if dry_run:
        return {"files": doomed, "kept": kept}
    for path in doomed:
        os.remove(path)
    manifest_path = os.path.join(serato_dir, MANIFEST)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
    return {"files": doomed, "kept": kept}


def check_serato_dir(serato_dir: str) -> str:
    """The absolute ``_Serato_`` folder, or ValueError for any other folder.

    Serato only reads crates from a folder named ``_Serato_``, so any other
    name is a mistake that would scatter crate files where nothing reads them.
    """
    path = os.path.abspath(os.path.expanduser(serato_dir.strip())) if serato_dir.strip() else ""
    if os.path.basename(path).lower() != FOLDER_NAME.lower():
        raise ValueError(f"Choose Serato's {FOLDER_NAME} folder, not {serato_dir!r}.")
    return path


def build_crate(track_paths: Sequence[str]) -> bytes:
    """The bytes of a ``.crate`` file listing ``track_paths`` in order."""
    parts = [_record(b"vrsn", _text(_VERSION))]
    parts.append(_record(b"osrt", _record(b"tvcn", _text("song")) + _record(b"brev", b"\x00")))
    for column in _COLUMNS:
        parts.append(
            _record(b"ovct", _record(b"tvcn", _text(column)) + _record(b"tvcw", _text("0")))
        )
    for path in track_paths:
        parts.append(_record(b"otrk", _record(b"ptrk", _text(path))))
    return b"".join(parts)


def read_crate(data: bytes) -> List[str]:
    """Track paths listed in a ``.crate`` file's bytes (the inverse of ``build_crate``)."""
    paths = []
    for tag, payload in _records(data):
        if tag == b"otrk":
            for inner, value in _records(payload):
                if inner == b"ptrk":
                    paths.append(value.decode("utf-16-be"))
    return paths


def serato_track_path(filepath: str, serato_dir: str) -> Optional[str]:
    """A track's path as a crate stores it, or None if it is on another drive.

    Serato stores paths relative to the root of the drive its ``_Serato_``
    folder is on, with ``/`` separators: ``/Users/dj/a.mp3`` is
    ``Users/dj/a.mp3`` in ``~/Music/_Serato_``, and ``/Volumes/USB/a.mp3`` is
    ``a.mp3`` in ``/Volumes/USB/_Serato_``.
    """
    track_root, track_rest = _split_drive(filepath)
    serato_root, _ = _split_drive(serato_dir)
    if track_root.lower() != serato_root.lower():
        return None
    return track_rest


def _split_drive(path: str) -> Tuple[str, str]:
    """``(drive root, path below it)``: ``C:``, ``/Volumes/USB`` or ``/``."""
    path = _absolute(path)
    if (len(path) > 1 and path[1] == ":") or path.startswith("\\\\"):
        windows = PureWindowsPath(path)
        return windows.drive, "/".join(windows.parts[1:])
    if path.startswith("/Volumes/"):
        parts = path.split("/", 3)
        return "/Volumes/" + parts[2], parts[3] if len(parts) > 3 else ""
    return "/", path.lstrip("/")


def _crate_path(subcrates: str, names: Sequence[str]) -> str:
    parts = [PARENT_CRATE] + [_crate_name(name) for name in names]
    return os.path.join(subcrates, "%%".join(parts) + ".crate")


def _crate_name(name: str) -> str:
    # Serato nests crates by "%%" in the filename, so a name must not contain it.
    cleaned = _safe_filename(name)
    while "%%" in cleaned:
        cleaned = cleaned.replace("%%", "%")
    return cleaned


def _unique(path: str, taken: set) -> str:
    """``path``, or with a counter if its name is taken (ignoring case, as macOS does)."""
    stem, ext = os.path.splitext(path)
    candidate, counter = path, 2
    while os.path.basename(candidate).lower() in taken:
        candidate = f"{stem} ({counter}){ext}"
        counter += 1
    taken.add(os.path.basename(candidate).lower())
    return candidate


def _fingerprint(track_paths: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(track_paths).encode("utf-8")).hexdigest()


def _edited(path: str, manifest: Manifest) -> bool:
    """Whether the crate's tracks differ from what cratewise wrote.

    Compares tracks rather than bytes, because Serato rewrites a crate's
    column settings on its own.
    """
    try:
        with open(path, "rb") as handle:
            tracks = read_crate(handle.read())
    except OSError:
        return False
    return _fingerprint(tracks) != manifest[path]["tracks"]


def _read_manifest(serato_dir: str) -> Manifest:
    try:
        with open(os.path.join(serato_dir, MANIFEST), encoding="utf-8") as handle:
            entries = json.load(handle).get("crates", {})
    except (OSError, ValueError, AttributeError):
        return {}
    if not isinstance(entries, dict):
        return {}
    subcrates = os.path.join(serato_dir, "Subcrates")
    return {
        os.path.join(subcrates, os.path.basename(name)): entry
        for name, entry in entries.items()
        if isinstance(entry, dict) and "tracks" in entry
    }


def _write_manifest(serato_dir: str, manifest: Manifest) -> None:
    entries = {os.path.basename(path): entry for path, entry in sorted(manifest.items())}
    with open(os.path.join(serato_dir, MANIFEST), "w", encoding="utf-8") as handle:
        json.dump({"crates": entries}, handle, indent=2)


def _record(tag: bytes, payload: bytes) -> bytes:
    return tag + struct.pack(">I", len(payload)) + payload


def _records(data: bytes) -> List[Tuple[bytes, bytes]]:
    out = []
    offset = 0
    while offset + 8 <= len(data):
        tag = data[offset : offset + 4]
        (length,) = struct.unpack(">I", data[offset + 4 : offset + 8])
        out.append((tag, data[offset + 8 : offset + 8 + length]))
        offset += 8 + length
    return out


def _text(value: str) -> bytes:
    return value.encode("utf-16-be")
