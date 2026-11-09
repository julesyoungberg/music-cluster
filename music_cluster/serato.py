"""Writing groups as Serato DJ crates.

Serato keeps its crates as small binary files in ``_Serato_/Subcrates`` (on a
Mac's start-up disk, ``~/Music/_Serato_``). Each group becomes a subcrate under
one parent crate named ``Cratewise``, so everything cratewise writes sits in
one place in Serato's crate list and never mixes with the user's own crates.

This writes into the DJ app's own folder, so it follows the filing rules: a dry
run first, a refusal to replace any crate cratewise did not write, and an undo
(``remove_crates``) that deletes exactly the files it wrote. The files it wrote
are listed in ``_Serato_/cratewise-crates.json``.

A crate file holds tagged records: a four-letter tag, a big-endian 32-bit
length, then the payload (text is UTF-16BE). The parts written here are the
version header, a few display columns, and one ``otrk``/``ptrk`` record per
track holding its path relative to the root of the drive the ``_Serato_``
folder is on. A track on another drive belongs to that drive's own
``_Serato_`` folder, so it is left out and counted.
"""

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
_VERSION = "1.0/Serato ScratchLive Crate"
_COLUMNS = ("song", "artist", "bpm", "key")


def write_crates(
    db: Database, collection: Dict[str, Any], serato_dir: str, dry_run: bool = False
) -> Dict[str, Any]:
    """Write one crate per group into ``serato_dir`` (the ``_Serato_`` folder).

    Returns ``{"files": [...], "left_out": n}``: the crate files written (or
    that would be, on a dry run) and how many tracks were left out because
    they are on a different drive. Raises ``ExportConflict`` before writing
    anything if a crate of the same name exists that cratewise did not write.
    """
    serato_dir = os.path.abspath(os.path.expanduser(serato_dir))
    subcrates = os.path.join(serato_dir, "Subcrates")
    playlists = collect_playlists(db, db.list_groups(collection["id"]))
    if not playlists:
        return {"files": [], "left_out": 0}

    crates: Dict[str, List[str]] = {_crate_path(subcrates, []): []}
    left_out = 0
    for name, tracks in playlists:
        paths = [serato_track_path(track["filepath"], serato_dir) for track in tracks]
        left_out += sum(1 for path in paths if path is None)
        crates[_crate_path(subcrates, [name])] = [path for path in paths if path is not None]

    ours = set(_read_manifest(serato_dir))
    foreign = [path for path in crates if os.path.lexists(path) and path not in ours]
    if foreign:
        raise ExportConflict(foreign)
    if dry_run:
        return {"files": list(crates), "left_out": left_out}

    os.makedirs(subcrates, exist_ok=True)
    # Record first, so a crash mid-write still leaves every file removable.
    _write_manifest(serato_dir, sorted(ours | set(crates)))
    for path, track_paths in crates.items():
        with open(path, "wb") as handle:
            handle.write(build_crate(track_paths))
    return {"files": list(crates), "left_out": left_out}


def remove_crates(serato_dir: str, dry_run: bool = False) -> List[str]:
    """Delete every crate cratewise wrote into ``serato_dir``; return their paths."""
    serato_dir = os.path.abspath(os.path.expanduser(serato_dir))
    written = [path for path in _read_manifest(serato_dir) if os.path.lexists(path)]
    if dry_run:
        return written
    for path in written:
        os.remove(path)
    manifest = os.path.join(serato_dir, MANIFEST)
    if os.path.exists(manifest):
        os.remove(manifest)
    return written


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
    # Serato nests crates by "%%" in the filename, so a name must not contain it.
    parts = [PARENT_CRATE] + [_safe_filename(name).replace("%%", "%") for name in names]
    return os.path.join(subcrates, "%%".join(parts) + ".crate")


def _read_manifest(serato_dir: str) -> List[str]:
    try:
        with open(os.path.join(serato_dir, MANIFEST), encoding="utf-8") as handle:
            names = json.load(handle).get("crates", [])
    except (OSError, ValueError):
        return []
    subcrates = os.path.join(serato_dir, "Subcrates")
    return [os.path.join(subcrates, os.path.basename(name)) for name in names]


def _write_manifest(serato_dir: str, paths: Sequence[str]) -> None:
    with open(os.path.join(serato_dir, MANIFEST), "w", encoding="utf-8") as handle:
        json.dump({"crates": [os.path.basename(path) for path in paths]}, handle, indent=2)


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
