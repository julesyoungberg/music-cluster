"""Tag metadata reading.

Audio features describe how a track sounds; tags describe how the DJ already
thinks about it. Both are useful for sorting, so tags are read at analysis time
and stored alongside the feature vector.
"""

import re
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from mutagen import File as MutagenFile

    MUTAGEN_AVAILABLE = True
except ImportError:  # pragma: no cover - mutagen is a hard dependency in practice
    MUTAGEN_AVAILABLE = False


# Tag keys vary by container; check each candidate in order.
_TAG_KEYS = {
    "title": ("title", "TIT2", "\xa9nam"),
    "artist": ("artist", "TPE1", "\xa9ART", "albumartist", "TPE2", "aART"),
    "album": ("album", "TALB", "\xa9alb"),
    "genre": ("genre", "TCON", "\xa9gen"),
    "year": ("date", "TDRC", "year", "\xa9day", "originaldate"),
    "tag_bpm": ("bpm", "TBPM", "tmpo"),
    "tag_key": ("initialkey", "TKEY", "key", "----:com.apple.iTunes:initialkey"),
    "comment": ("comment", "COMM", "\xa9cmt", "COMM::eng"),
}

_FILENAME_SPLIT = re.compile(r"\s+-\s+|\s+–\s+|\s+_\s+")


def _first_value(tags: Any, keys) -> Optional[str]:
    for key in keys:
        try:
            value = tags.get(key)
        except Exception:
            value = None
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            if not value:
                continue
            value = value[0]
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        text = str(value).strip()
        if text:
            return text
    return None


def _parse_year(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    match = re.search(r"(19|20)\d{2}", raw)
    return int(match.group(0)) if match else None


def _parse_bpm(raw: Optional[str]) -> Optional[float]:
    if not raw:
        return None
    match = re.search(r"\d+(\.\d+)?", raw)
    if not match:
        return None
    value = float(match.group(0))
    return value if 20.0 <= value <= 300.0 else None


def read_metadata(filepath: str) -> Dict[str, Any]:
    """Read tag metadata, falling back to the filename for artist/title.

    Never raises: a track with unreadable tags is still a usable track.
    """
    result: Dict[str, Any] = {
        "title": None,
        "artist": None,
        "album": None,
        "genre": None,
        "year": None,
        "tag_bpm": None,
        "tag_key": None,
        "comment": None,
    }

    if MUTAGEN_AVAILABLE:
        try:
            audio = MutagenFile(filepath, easy=True)
            if audio is None:
                audio = MutagenFile(filepath)
            if audio is not None and audio.tags is not None:
                tags = audio.tags
                for field, keys in _TAG_KEYS.items():
                    result[field] = _first_value(tags, keys)
        except Exception:
            pass

    result["year"] = _parse_year(result.get("year"))
    result["tag_bpm"] = _parse_bpm(result.get("tag_bpm"))

    if not result["title"] or not result["artist"]:
        artist, title = _from_filename(filepath)
        result["artist"] = result["artist"] or artist
        result["title"] = result["title"] or title

    return result


def _from_filename(filepath: str) -> tuple[Optional[str], Optional[str]]:
    """Guess ``artist - title`` from a filename stem."""
    stem = Path(filepath).stem.strip()
    if not stem:
        return None, None
    parts = [p.strip() for p in _FILENAME_SPLIT.split(stem) if p.strip()]
    if len(parts) >= 2:
        return parts[0], " - ".join(parts[1:])
    return None, stem


def display_name(track: Dict[str, Any]) -> str:
    """Human-facing label for a track row."""
    artist = track.get("artist")
    title = track.get("title")
    if artist and title:
        return f"{artist} - {title}"
    if title:
        return str(title)
    return str(track.get("filename") or track.get("filepath") or "Unknown track")
