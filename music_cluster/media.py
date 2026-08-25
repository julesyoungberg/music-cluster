"""Reading audio for playback and preview in the UI.

Auditioning is not a nicety in this workflow — a DJ cannot confirm that a track
belongs in a folder without hearing it — so streaming, waveforms and artwork are
first-class.
"""

import base64
import hashlib
import json
import logging
import os
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


MIN_WAVEFORM_SAMPLES = 16
MAX_WAVEFORM_SAMPLES = 2000

# Bumped whenever the envelope maths changes, so cached waveforms from an older
# version are recomputed rather than served.
WAVEFORM_VERSION = 3

# Below this length a file is one sonic event rather than an arrangement, and
# its envelope is drawn as measured. See _shape_envelope.
ONE_SHOT_SECONDS = 2.0

# How much the envelope leans on window RMS rather than window peak. Peak alone
# tracks the loudest transient in a window, which on a limited master is the
# same value everywhere — the classic "every bar is the same height" waveform.
RMS_WEIGHT = 0.5

# The envelope is stretched between a low and a high percentile of itself. Using
# percentiles rather than min/max keeps one stray transient from squashing the
# rest of the track.
HIGH_PERCENTILE = 99.0
LOW_PERCENTILE = 5.0

# ...but the floor never rises above this share of the ceiling, so genuinely
# steady material stays flat instead of having its noise stretched into a shape
# that is not in the audio.
MAX_FLOOR_RATIO = 0.6

# How much of the drawn height comes from that stretched envelope rather than
# the plain one. Keeping some of the plain envelope means a quiet section still
# reads as quiet-but-present instead of dropping out to nothing.
EXPANSION_MIX = 0.7


def _shape_envelope(peak: np.ndarray, rms: np.ndarray, duration: float = 0.0) -> np.ndarray:
    """Blend peak and RMS windows into a 0-1 envelope with visible structure."""
    envelope = RMS_WEIGHT * rms + (1.0 - RMS_WEIGHT) * peak

    high = float(np.percentile(envelope, HIGH_PERCENTILE))
    if high <= 0.0:
        # Digital silence, or so close to it that there is nothing to draw.
        return np.zeros_like(envelope)

    plain = np.clip(envelope / high, 0.0, 1.0)

    if 0.0 < duration <= ONE_SHOT_SECONDS:
        # A one-shot's decay *is* its shape — it is what tells a closed hat
        # from an open one at a glance. The percentile stretch below exists to
        # pull structure out of a limited master where every bar is the same
        # height; applied to a 300 ms kick it lifts the tail off the floor and
        # draws a sound that keeps ringing when it does not.
        return plain

    low = min(float(np.percentile(envelope, LOW_PERCENTILE)), MAX_FLOOR_RATIO * high)
    stretched = np.clip((envelope - low) / (high - low), 0.0, 1.0)
    return EXPANSION_MIX * stretched + (1.0 - EXPANSION_MIX) * plain


def _resize_envelope(values: np.ndarray, samples: int) -> np.ndarray:
    """Stretch a short envelope up to ``samples`` points.

    Files shorter than the requested resolution used to be zero-padded, which
    drew a sliver of audio against a long empty tail. Interpolating keeps the
    shape spread across the whole width, which is what the duration says.
    """
    if len(values) == samples:
        return values
    if len(values) == 1:
        return np.repeat(values, samples)
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, samples)
    return np.interp(target, source, values)


def _windows_streaming(
    filepath: str, samples: int
) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Per-window peak and RMS read block by block, without decoding it all.

    Returns None when soundfile cannot handle the format, so the caller can fall
    back to librosa. Memory stays bounded by one window, which matters for the
    long DJ sets this app is pointed at.
    """
    try:
        import soundfile as sf
    except ImportError:  # pragma: no cover - soundfile is a hard dependency
        return None

    try:
        info = sf.info(filepath)
    except Exception as exc:
        logger.debug("soundfile cannot read %s: %s", filepath, exc)
        return None

    frames, rate = int(info.frames), int(info.samplerate)
    if frames <= 0 or rate <= 0:
        return None

    duration = frames / rate
    window = max(1, frames // samples)

    peaks: List[float] = []
    rms: List[float] = []
    try:
        for block in sf.blocks(
            filepath, blocksize=window, dtype="float32", always_2d=True, fill_value=0.0
        ):
            if len(peaks) >= samples or block.size == 0:
                break
            mono = block.mean(axis=1).astype(np.float64)
            peaks.append(float(np.abs(mono).max()))
            rms.append(float(np.sqrt(np.mean(np.square(mono)))))
    except Exception as exc:
        logger.debug("Streaming decode of %s failed: %s", filepath, exc)
        return None

    if not peaks:
        return None
    return np.asarray(peaks), np.asarray(rms), duration


def _windows_decoded(filepath: str, samples: int) -> Tuple[np.ndarray, np.ndarray, float]:
    """Per-window peak and RMS via a full librosa decode.

    Used for formats libsndfile will not open (m4a, opus, wma...). The decode
    rate drops for very long files so memory stays reasonable; the envelope is
    unaffected because each window still covers thousands of samples.
    """
    try:
        import librosa
    except ImportError as exc:  # pragma: no cover
        raise ValueError("librosa is required for waveform generation") from exc

    try:
        reported = float(librosa.get_duration(path=filepath))
    except Exception:
        reported = 0.0

    rate = 11025
    if reported > 5400:  # 90 minutes
        rate = 2756
    elif reported > 1800:  # 30 minutes
        rate = 5512

    try:
        y, sr = librosa.load(filepath, sr=rate, mono=True)
    except Exception as exc:
        raise ValueError(f"Could not decode audio: {exc}") from exc

    if len(y) == 0:
        raise ValueError("Empty audio file")

    duration = len(y) / float(sr)
    y = y.astype(np.float64)

    if len(y) < samples:
        return np.abs(y), np.abs(y), duration

    window = len(y) // samples
    blocks = y[: window * samples].reshape(samples, window)
    return np.abs(blocks).max(axis=1), np.sqrt(np.square(blocks).mean(axis=1)), duration


def compute_waveform(filepath: str, samples: int = 240) -> Dict[str, Any]:
    """Envelope for drawing a waveform, covering the whole track.

    The result is normalised per track: 1.0 is that track's own loud passages,
    0.0 its quiet ones. Raises ValueError when the file cannot be decoded.
    """
    samples = max(MIN_WAVEFORM_SAMPLES, min(int(samples), MAX_WAVEFORM_SAMPLES))

    windows = _windows_streaming(filepath, samples)
    if windows is None:
        windows = _windows_decoded(filepath, samples)

    peak, rms, duration = windows
    if len(peak) == 0:
        raise ValueError("Empty audio file")

    values = _resize_envelope(_shape_envelope(peak, rms, duration), samples)

    return {
        "peaks": [round(float(value), 3) for value in values],
        "duration": round(float(duration), 3),
        "samples": int(samples),
        "version": WAVEFORM_VERSION,
    }


MEMORY_CACHE_SIZE = 64

_memory_cache: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_memory_lock = threading.Lock()


def waveform_cache_key(filepath: str, samples: int) -> str:
    """Identity of a waveform: the file as it is now, at this resolution.

    Includes mtime and size so replacing a file (a re-rip, a re-tag) invalidates
    the cached envelope without anyone having to clear anything.
    """
    try:
        stat = os.stat(filepath)
        stamp = f"{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        stamp = "missing"
    digest = hashlib.sha1(
        f"{WAVEFORM_VERSION}|{filepath}|{stamp}|{samples}".encode("utf-8", "replace")
    ).hexdigest()
    return digest


def cached_waveform(
    filepath: str, samples: int = 240, cache_dir: Optional[str] = None
) -> Tuple[Dict[str, Any], str]:
    """A waveform plus its cache key, computing it only when needed.

    Decoding a long track takes real time, and the UI asks for the same
    waveforms over and over as you move through a review queue, so results are
    kept in memory and on disk.
    """
    samples = max(MIN_WAVEFORM_SAMPLES, min(int(samples), MAX_WAVEFORM_SAMPLES))
    key = waveform_cache_key(filepath, samples)

    with _memory_lock:
        payload = _memory_cache.get(key)
        if payload is not None:
            _memory_cache.move_to_end(key)
            return payload, key

    cache_file = os.path.join(cache_dir, f"{key}.json") if cache_dir else None
    if cache_file and os.path.isfile(cache_file):
        try:
            with open(cache_file) as handle:
                payload = json.load(handle)
            if isinstance(payload, dict) and payload.get("peaks"):
                _remember(key, payload)
                return payload, key
        except Exception as exc:
            logger.debug("Ignoring unreadable waveform cache %s: %s", cache_file, exc)

    payload = compute_waveform(filepath, samples=samples)
    _remember(key, payload)

    if cache_file and cache_dir:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            # Write then rename so a crash mid-write cannot leave a torn file
            # that would be read back as a broken waveform.
            temporary = f"{cache_file}.{os.getpid()}.tmp"
            with open(temporary, "w") as handle:
                json.dump(payload, handle)
            os.replace(temporary, cache_file)
        except Exception as exc:
            logger.debug("Could not cache waveform to %s: %s", cache_file, exc)

    return payload, key


def _remember(key: str, payload: Dict[str, Any]) -> None:
    with _memory_lock:
        _memory_cache[key] = payload
        _memory_cache.move_to_end(key)
        while len(_memory_cache) > MEMORY_CACHE_SIZE:
            _memory_cache.popitem(last=False)


def clear_waveform_memory_cache() -> None:
    """Drop in-process cached waveforms (used by tests)."""
    with _memory_lock:
        _memory_cache.clear()


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
