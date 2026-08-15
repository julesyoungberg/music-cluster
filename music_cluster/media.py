"""Reading audio for playback and preview in the UI.

Auditioning is not a nicety in this workflow — a DJ cannot confirm that a track
belongs in a folder without hearing it — so streaming, waveforms and artwork are
first-class.
"""

import base64
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MEDIA_TYPES = {
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".aiff": "audio/aiff",
    ".aif": "audio/aiff",
    ".m4a": "audio/mp4",
    ".aac": "audio/aac",
    ".ogg": "audio/ogg",
    ".opus": "audio/opus",
    ".wma": "audio/x-ms-wma",
    ".wv": "audio/x-wavpack",
    ".ape": "audio/x-ape",
}


def media_type_for(filepath: str) -> str:
    return MEDIA_TYPES.get(Path(filepath).suffix.lower(), "application/octet-stream")


def parse_range_header(header: Optional[str], file_size: int) -> Optional[Tuple[int, int]]:
    """Parse a single-range ``Range: bytes=start-end`` header.

    Returns an inclusive ``(start, end)`` pair, or None when the header is
    absent or not a form worth honouring.
    """
    if not header or not header.startswith("bytes="):
        return None

    spec = header[len("bytes=") :].split(",")[0].strip()
    start_text, _, end_text = spec.partition("-")

    try:
        if not start_text:
            # Suffix range: the last N bytes.
            length = int(end_text)
            if length <= 0:
                return None
            return max(0, file_size - length), file_size - 1
        start = int(start_text)
        end = int(end_text) if end_text else file_size - 1
    except ValueError:
        return None

    if start >= file_size or start > end:
        return None
    return start, min(end, file_size - 1)


def compute_waveform(filepath: str, samples: int = 240, max_seconds: float = 420.0) -> Dict[str, Any]:
    """Peak envelope for drawing a waveform.

    Raises ValueError when the file cannot be decoded.
    """
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover
        raise ValueError("librosa is required for waveform generation") from exc

    y, sr = librosa.load(filepath, sr=11025, mono=True, duration=max_seconds)
    if len(y) == 0:
        raise ValueError("Empty audio file")

    magnitudes = np.abs(y)
    samples = max(16, min(samples, 2000))

    if len(magnitudes) >= samples:
        # Trim to a whole number of windows so the reshape is exact, then take
        # the peak of each window.
        window = len(magnitudes) // samples
        trimmed = magnitudes[: window * samples].reshape(samples, window)
        peaks = trimmed.max(axis=1)
    else:
        peaks = np.pad(magnitudes, (0, samples - len(magnitudes)))

    ceiling = float(peaks.max())
    if ceiling > 0:
        peaks = peaks / ceiling

    return {
        "peaks": [round(float(value), 4) for value in peaks],
        "duration": float(len(y) / sr),
        "samples": int(samples),
    }


def extract_artwork(filepath: str) -> Optional[Dict[str, str]]:
    """Embedded cover art as a data URL, or None when the file has none."""
    try:
        from mutagen import File as MutagenFile
        from mutagen.flac import FLAC
        from mutagen.mp4 import MP4
        from mutagen.oggopus import OggOpus
        from mutagen.oggvorbis import OggVorbis
    except ImportError:  # pragma: no cover
        return None

    try:
        audio = MutagenFile(filepath)
    except Exception as exc:
        logger.debug("Could not open %s for artwork: %s", filepath, exc)
        return None
    if audio is None:
        return None

    data: Optional[bytes] = None
    mime = "image/jpeg"

    try:
        if isinstance(audio, FLAC) and audio.pictures:
            picture = audio.pictures[0]
            data, mime = picture.data, picture.mime or mime
        elif isinstance(audio, MP4):
            covers = audio.tags.get("covr") if audio.tags else None
            if covers:
                data = bytes(covers[0])
                mime = "image/png" if data[:8] == b"\x89PNG\r\n\x1a\n" else "image/jpeg"
        elif isinstance(audio, (OggVorbis, OggOpus)):
            encoded = audio.get("metadata_block_picture")
            if encoded:
                from mutagen.flac import Picture

                picture = Picture(base64.b64decode(encoded[0]))
                data, mime = picture.data, picture.mime or mime
        else:
            # ID3-tagged formats (MP3, AIFF, WAV with ID3 chunks).
            frames = getattr(audio, "tags", None)
            if frames is not None:
                for key in list(frames.keys()):
                    if str(key).startswith("APIC"):
                        frame = frames[key]
                        data = getattr(frame, "data", None)
                        mime = getattr(frame, "mime", mime) or mime
                        break
    except Exception as exc:
        logger.debug("Artwork extraction failed for %s: %s", filepath, exc)
        return None

    if not data:
        return None

    return {
        "artwork": f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}",
        "mime_type": mime,
    }
