"""Exporting groups to DJ software.

Each group becomes a playlist in a file the DJ app imports:

``m3u8`` / ``m3u``  one playlist file per group (Serato, Engine DJ, VirtualDJ, and
                    anything else that reads M3U)
``rekordbox``       one ``rekordbox.xml`` holding every group as a playlist, with
                    BPM, key and genre (Rekordbox: Preferences > Advanced >
                    rekordbox xml)
``traktor``         one ``.nml`` collection file holding every group as a playlist,
                    with BPM, key and genre (Traktor: Import Playlist)

Exporting only writes these new files. It never touches the music itself, so
re-running an export is how a DJ app is kept in sync with the groups. Every
export file is marked as music-cluster's, and an export refuses to replace a
file of the same name that music-cluster did not write.

Metadata the user wrote into a file (title, artist, genre, BPM, key) wins over
what music-cluster measured: the measurement only fills the gaps.
"""

import functools
import os
import re
import sys
import uuid
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote
from xml.etree import ElementTree as ET

from . import __version__
from .database import Database
from .organizer import EXPORT_MARKER, _safe_filename, write_playlist


FORMATS = ("m3u8", "m3u", "rekordbox", "traktor")

_SHARP_TO_RB = {"C#": "Db", "D#": "Eb", "F#": "F#", "G#": "Ab", "A#": "Bb"}

# Traktor's MUSICAL_KEY: 0-11 are C..B major, 12-23 are C..B minor.
_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_SHORT_KEY = re.compile(r"^([A-G])([#b]?)(m?)$")

# Characters XML 1.0 allows; anything else (control characters, lone
# surrogates from undecodable filenames) makes the file unreadable.
_NOT_XML = re.compile("[^\t\n\r\x20-\ud7ff\ue000-\ufffd\U00010000-\U0010ffff]")

Playlist = Tuple[str, List[Dict[str, Any]]]


class ExportConflict(FileExistsError):
    """The export would replace files music-cluster did not write."""

    def __init__(self, paths: Sequence[str]):
        self.paths = list(paths)
        listed = ", ".join(self.paths[:5]) + (" ..." if len(self.paths) > 5 else "")
        super().__init__(
            f"Not replacing {len(self.paths)} file(s) that music-cluster did not write: "
            f"{listed}. Choose another folder, or move those files first."
        )


def export_collection(
    db: Database,
    collection: Dict[str, Any],
    output_dir: str,
    fmt: str = "m3u8",
    relative_paths: bool = False,
) -> List[str]:
    """Write a collection's groups to ``output_dir`` in a DJ app's format.

    Returns the paths written. Groups with no tracks are left out.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unknown export format {fmt!r}; choose from {FORMATS}")
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    playlists = collect_playlists(db, db.list_groups(collection["id"]))
    if not playlists:
        return []

    if fmt in ("m3u8", "m3u"):
        targets = _playlist_paths(output_dir, [name for name, _ in playlists], fmt)
        _refuse_foreign_files(targets)
        for path, (name, tracks) in zip(targets, playlists, strict=True):
            write_playlist(path, tracks, relative_paths=relative_paths, header=name)
        return targets

    stem = _safe_filename(f"music-cluster - {collection['name']}")
    if fmt == "rekordbox":
        path = os.path.join(output_dir, f"{stem}.xml")
        tree = build_rekordbox_xml(playlists, folder_name=collection["name"])
    else:
        path = os.path.join(output_dir, f"{stem}.nml")
        tree = build_traktor_nml(playlists, folder_name=collection["name"])

    _refuse_foreign_files([path])
    root = tree.getroot()
    assert root is not None
    root.insert(0, ET.Comment(f" {EXPORT_MARKER} "))
    for element in tree.iter():
        for name, value in element.attrib.items():
            element.set(name, _NOT_XML.sub("", value))
    os.makedirs(output_dir, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return [path]


def collect_playlists(db: Database, groups: Sequence[Dict[str, Any]]) -> List[Playlist]:
    """``[(group name, [track, ...]), ...]`` with export metadata merged in.

    Groups with no tracks are left out.
    """
    members = {group["id"]: db.get_group_members(group["id"], limit=10**6) for group in groups}
    track_ids = {track["id"] for tracks in members.values() for track in tracks}
    tags = db.get_tags_many(sorted(track_ids))
    return [
        (
            group["name"],
            [track_metadata(track, tags.get(track["id"])) for track in members[group["id"]]],
        )
        for group in groups
        if members[group["id"]]
    ]


def unique_tracks(playlists: Sequence[Playlist]) -> List[Dict[str, Any]]:
    """Every track across the playlists, once each, in first-seen order."""
    tracks: Dict[int, Dict[str, Any]] = {}
    for _, members in playlists:
        for track in members:
            tracks.setdefault(track["id"], track)
    return list(tracks.values())


def _playlist_paths(output_dir: str, names: Sequence[str], fmt: str) -> List[str]:
    """One file per group; groups whose names clean up the same get a counter."""
    paths: List[str] = []
    for name in names:
        stem = _safe_filename(name)
        candidate = os.path.join(output_dir, f"{stem}.{fmt}")
        counter = 2
        while candidate in paths:
            candidate = os.path.join(output_dir, f"{stem} ({counter}).{fmt}")
            counter += 1
        paths.append(candidate)
    return paths


def _refuse_foreign_files(paths: Sequence[str]) -> None:
    """Raise if any path exists and is not an earlier music-cluster export."""
    foreign = [path for path in paths if os.path.lexists(path) and not _is_our_export(path)]
    if foreign:
        raise ExportConflict(foreign)


def _is_our_export(path: str) -> bool:
    try:
        with open(path, "rb") as handle:
            return EXPORT_MARKER.encode() in handle.read(4096)
    except OSError:
        return False


def track_metadata(track: Dict[str, Any], tags: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """What a DJ app should show for a track: the file's own tags, then ours."""
    tags = tags or {}
    filepath = track["filepath"]
    return {
        "id": track["id"],
        "filepath": filepath,
        "title": track.get("title") or Path(filepath).stem,
        "artist": track.get("artist") or "",
        "album": track.get("album") or "",
        "genre": track.get("genre") or tags.get("genre_label") or "",
        "year": track.get("year"),
        "comment": track.get("comment") or "",
        "duration": float(track.get("duration") or 0.0),
        "size": track.get("file_size"),
        "bpm": track.get("tag_bpm") or tags.get("bpm"),
        "key": short_key(track.get("tag_key")) or short_key(tags.get("key")),
    }


def short_key(key: Optional[str]) -> Optional[str]:
    """``"A minor"`` -> ``"Am"``, ``"C# major"`` -> ``"Db"``; anything else as given.

    Rekordbox and Traktor both read keys written this way. A key that is
    already short (the file's own tag, or Camelot like ``8A``) passes through.
    """
    if not key:
        return None
    parts = key.strip().split()
    if len(parts) != 2 or parts[1].lower() not in ("major", "minor"):
        return key.strip()
    tonic = _SHARP_TO_RB.get(parts[0], parts[0])
    return tonic + ("m" if parts[1].lower() == "minor" else "")


# ----------------------------------------------------------------------
# Rekordbox
# ----------------------------------------------------------------------


def build_rekordbox_xml(playlists: Sequence[Playlist], folder_name: str) -> ET.ElementTree:
    """A rekordbox.xml with every playlist inside one folder named ``folder_name``."""
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="music-cluster", Version=__version__, Company="")

    tracks = unique_tracks(playlists)
    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(tracks)))
    for track in tracks:
        attrs = {
            "TrackID": str(track["id"]),
            "Name": track["title"],
            "Artist": track["artist"],
            "Album": track["album"],
            "Genre": track["genre"],
            "Kind": _kind(track["filepath"]),
            "TotalTime": str(round(track["duration"])),
            "Comments": track["comment"],
            "Location": rekordbox_location(track["filepath"]),
        }
        if track.get("size"):
            attrs["Size"] = str(track["size"])
        if track.get("year"):
            attrs["Year"] = str(track["year"])
        if track.get("bpm"):
            attrs["AverageBpm"] = f"{float(track['bpm']):.2f}"
        if track.get("key"):
            attrs["Tonality"] = track["key"]
        ET.SubElement(collection, "TRACK", attrs)

    playlists_el = ET.SubElement(root, "PLAYLISTS")
    top = ET.SubElement(playlists_el, "NODE", Type="0", Name="ROOT", Count="1")
    folder = ET.SubElement(top, "NODE", Type="0", Name=folder_name, Count=str(len(playlists)))
    for name, members in playlists:
        node = ET.SubElement(
            folder, "NODE", Name=name, Type="1", KeyType="0", Entries=str(len(members))
        )
        for track in members:
            ET.SubElement(node, "TRACK", Key=str(track["id"]))
    return ET.ElementTree(root)


def rekordbox_location(filepath: str) -> str:
    """A file URL the way Rekordbox writes them: ``file://localhost/...``."""
    path = _absolute(filepath)
    if _is_windows_path(path):
        posix = PureWindowsPath(path).as_posix()  # C:/Users/... or //server/share/...
        if posix.startswith("//"):
            return "file:" + quote(posix, safe="/")
        return "file://localhost/" + quote(posix, safe="/:")
    # Bytes, so a filename that isn't valid UTF-8 still round-trips exactly.
    return "file://localhost" + quote(os.fsencode(path), safe="/")


# ----------------------------------------------------------------------
# Traktor
# ----------------------------------------------------------------------


def build_traktor_nml(playlists: Sequence[Playlist], folder_name: str) -> ET.ElementTree:
    """A Traktor collection file with every playlist inside one folder."""
    root = ET.Element("NML", VERSION="19")
    ET.SubElement(root, "HEAD", COMPANY="www.native-instruments.com", PROGRAM="Traktor")
    ET.SubElement(root, "MUSICFOLDERS")

    tracks = unique_tracks(playlists)
    keys: Dict[int, str] = {}
    collection = ET.SubElement(root, "COLLECTION", ENTRIES=str(len(tracks)))
    for track in tracks:
        volume, directory, filename = traktor_location(track["filepath"])
        keys[track["id"]] = f"{volume}{directory}{filename}"
        entry = ET.SubElement(collection, "ENTRY", TITLE=track["title"], ARTIST=track["artist"])
        ET.SubElement(
            entry, "LOCATION", DIR=directory, FILE=filename, VOLUME=volume, VOLUMEID=volume
        )
        if track["album"]:
            ET.SubElement(entry, "ALBUM", TITLE=track["album"])
        info = {"PLAYTIME": str(round(track["duration"]))}
        if track["genre"]:
            info["GENRE"] = track["genre"]
        if track["comment"]:
            info["COMMENT"] = track["comment"]
        if track.get("key"):
            info["KEY"] = track["key"]
        if track.get("year"):
            info["RELEASE_DATE"] = f"{track['year']}/1/1"
        ET.SubElement(entry, "INFO", info)
        musical_key = traktor_musical_key(track.get("key"))
        if musical_key is not None:
            ET.SubElement(entry, "MUSICAL_KEY", VALUE=str(musical_key))
        if track.get("bpm"):
            ET.SubElement(
                entry, "TEMPO", BPM=f"{float(track['bpm']):.6f}", BPM_QUALITY="100.000000"
            )

    ET.SubElement(root, "SETS", ENTRIES="0")
    playlists_el = ET.SubElement(root, "PLAYLISTS")
    top = ET.SubElement(playlists_el, "NODE", TYPE="FOLDER", NAME="$ROOT")
    top_sub = ET.SubElement(top, "SUBNODES", COUNT="1")
    folder = ET.SubElement(top_sub, "NODE", TYPE="FOLDER", NAME=folder_name)
    folder_sub = ET.SubElement(folder, "SUBNODES", COUNT=str(len(playlists)))
    for name, members in playlists:
        node = ET.SubElement(folder_sub, "NODE", TYPE="PLAYLIST", NAME=name)
        playlist = ET.SubElement(
            node, "PLAYLIST", ENTRIES=str(len(members)), TYPE="LIST", UUID=uuid.uuid4().hex
        )
        for track in members:
            item = ET.SubElement(playlist, "ENTRY")
            ET.SubElement(item, "PRIMARYKEY", TYPE="TRACK", KEY=keys[track["id"]])
    return ET.ElementTree(root)


def traktor_musical_key(key: Optional[str]) -> Optional[int]:
    """``"Am"`` -> 21, ``"Db"`` -> 1: the number Traktor's Key column reads."""
    match = _SHORT_KEY.match(key or "")
    if not match:
        return None
    letter, accidental, minor = match.groups()
    semitone = (_SEMITONES[letter] + {"#": 1, "b": -1, "": 0}[accidental]) % 12
    return semitone + (12 if minor else 0)


def traktor_location(filepath: str) -> Tuple[str, str, str]:
    """``(volume, dir, file)`` in Traktor's ``/:``-separated form.

    ``/Users/dj/Music/a.mp3`` on the Mac's start-up disk becomes
    ``("Macintosh HD", "/:Users/:dj/:Music/:", "a.mp3")``.
    """
    path = _absolute(filepath)
    if _is_windows_path(path):
        windows = PureWindowsPath(path)
        volume = windows.drive  # "C:"
        parts = list(windows.parts[1:-1])
        filename = windows.name
    else:
        volume, rest = _posix_volume(path)
        parts = [part for part in rest.split("/") if part][:-1]
        filename = os.path.basename(path)
    directory = "/:" + "".join(f"{part}/:" for part in parts)
    return volume, directory, filename


def _posix_volume(path: str) -> Tuple[str, str]:
    """Split a POSIX path into the volume Traktor expects and the rest."""
    if path.startswith("/Volumes/"):
        parts = path.split("/", 3)
        return parts[2], "/" + (parts[3] if len(parts) > 3 else "")
    return _startup_volume_name(), path


@functools.lru_cache(maxsize=1)
def _startup_volume_name() -> str:
    """The Mac start-up disk's name (``/Volumes`` holds a link to ``/`` under it)."""
    if sys.platform != "darwin":
        return ""
    try:
        for name in os.listdir("/Volumes"):
            candidate = os.path.join("/Volumes", name)
            if os.path.islink(candidate) and os.path.realpath(candidate) == "/":
                return name
    except OSError:
        pass
    return "Macintosh HD"


def _absolute(path: str) -> str:
    """The path as given if already absolute in either convention, else made absolute.

    The library stores absolute paths, so this only matters for relative ones;
    keeping ``/Users/...`` as it is also means a Mac path is written the same
    way whichever OS runs the export.
    """
    if path.startswith("/") or _is_windows_path(path):
        return path
    return os.path.abspath(path)


def _is_windows_path(path: str) -> bool:
    """A drive-letter (``C:\\...``) or network-share (``\\\\server\\share``) path."""
    return (len(path) > 1 and path[1] == ":") or path.startswith("\\\\")


def _kind(filepath: str) -> str:
    extension = Path(filepath).suffix.lstrip(".").upper()
    return f"{extension} File" if extension else "Audio File"
