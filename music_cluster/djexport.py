"""Exporting groups to DJ software.

Each group becomes a playlist in a file the DJ app imports:

``m3u8`` / ``m3u``  one playlist file per group (Serato, Engine DJ, VirtualDJ, and
                    anything else that reads M3U)
``rekordbox``       one ``rekordbox.xml`` holding every group as a playlist, with
                    BPM, key and genre (Rekordbox: Preferences > Advanced >
                    rekordbox xml)
``traktor``         one ``.nml`` collection file holding every group as a playlist,
                    with BPM, key and genre (Traktor: Import Collection)

Exporting only writes these new files. It never touches the music itself, so
re-running an export is how a DJ app is kept in sync with the groups.

Metadata the user wrote into a file (title, artist, genre, BPM, key) wins over
what music-cluster measured: the measurement only fills the gaps.
"""

import os
import sys
import uuid
from pathlib import Path, PureWindowsPath
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote
from xml.etree import ElementTree as ET

from .database import Database
from .organizer import _safe_filename, export_groups_as_playlists


FORMATS = ("m3u8", "m3u", "rekordbox", "traktor")

_SHARP_TO_RB = {"C#": "Db", "D#": "Eb", "F#": "F#", "G#": "Ab", "A#": "Bb"}


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
    groups = db.list_groups(collection["id"])

    if fmt in ("m3u8", "m3u"):
        return export_groups_as_playlists(
            db, groups, output_dir, playlist_format=fmt, relative_paths=relative_paths
        )

    playlists = collect_playlists(db, groups)
    stem = _safe_filename(f"music-cluster - {collection['name']}")
    if fmt == "rekordbox":
        path = os.path.join(output_dir, f"{stem}.xml")
        tree = build_rekordbox_xml(playlists, folder_name=collection["name"])
    else:
        path = os.path.join(output_dir, f"{stem}.nml")
        tree = build_traktor_nml(playlists, folder_name=collection["name"])

    os.makedirs(output_dir, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return [path]


def collect_playlists(
    db: Database, groups: Sequence[Dict[str, Any]]
) -> List[Tuple[str, List[Dict[str, Any]]]]:
    """``[(group name, [track, ...]), ...]`` with export metadata merged in."""
    playlists: List[Tuple[str, List[Dict[str, Any]]]] = []
    for group in groups:
        members = db.get_group_members(group["id"], limit=10**6)
        if not members:
            continue
        tags = db.get_tags_many([member["id"] for member in members])
        playlists.append(
            (group["name"], [track_metadata(member, tags.get(member["id"])) for member in members])
        )
    return playlists


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


def build_rekordbox_xml(
    playlists: Sequence[Tuple[str, Sequence[Dict[str, Any]]]], folder_name: str
) -> ET.ElementTree:
    """A rekordbox.xml with every playlist inside one folder named ``folder_name``."""
    root = ET.Element("DJ_PLAYLISTS", Version="1.0.0")
    ET.SubElement(root, "PRODUCT", Name="music-cluster", Version="2.0.0", Company="")

    tracks: Dict[int, Dict[str, Any]] = {}
    for _, members in playlists:
        for track in members:
            tracks.setdefault(track["id"], track)

    collection = ET.SubElement(root, "COLLECTION", Entries=str(len(tracks)))
    for track in tracks.values():
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
    path = os.path.abspath(filepath)
    if _is_windows_path(path):
        posix = PureWindowsPath(path).as_posix()  # C:/Users/...
        return "file://localhost/" + quote(posix, safe="/:")
    return "file://localhost" + quote(path, safe="/")


# ----------------------------------------------------------------------
# Traktor
# ----------------------------------------------------------------------


def build_traktor_nml(
    playlists: Sequence[Tuple[str, Sequence[Dict[str, Any]]]], folder_name: str
) -> ET.ElementTree:
    """A Traktor collection file with every playlist inside one folder."""
    root = ET.Element("NML", VERSION="19")
    ET.SubElement(root, "HEAD", COMPANY="www.native-instruments.com", PROGRAM="Traktor")
    ET.SubElement(root, "MUSICFOLDERS")

    tracks: Dict[int, Dict[str, Any]] = {}
    for _, members in playlists:
        for track in members:
            tracks.setdefault(track["id"], track)

    keys: Dict[int, str] = {}
    collection = ET.SubElement(root, "COLLECTION", ENTRIES=str(len(tracks)))
    for track in tracks.values():
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


def traktor_location(filepath: str) -> Tuple[str, str, str]:
    """``(volume, dir, file)`` in Traktor's ``/:``-separated form.

    ``/Users/dj/Music/a.mp3`` on the Mac's start-up disk becomes
    ``("Macintosh HD", "/:Users/:dj/:Music/:", "a.mp3")``.
    """
    path = os.path.abspath(filepath)
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


def _is_windows_path(path: str) -> bool:
    return len(path) > 1 and path[1] == ":"


def _kind(filepath: str) -> str:
    extension = Path(filepath).suffix.lstrip(".").upper()
    return f"{extension} File" if extension else "Audio File"
