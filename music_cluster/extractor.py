"""Audio feature extraction.

Produces the fixed-length vector described by :mod:`music_cluster.features`.
The block order here and the layout there must stay in sync.
"""

import logging
import warnings
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from .features import aggregate_features, build_layout


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("audioread.ffdec").setLevel(logging.ERROR)
logging.getLogger("audioread").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """Extract a track-level feature vector from an audio file."""

    def __init__(
        self,
        sample_rate: int = 44100,
        frame_size: int = 2048,
        hop_size: int = 1024,
        n_mfcc: int = 20,
        excerpt_seconds: Optional[float] = 90,
    ):
        """
        Args:
            excerpt_seconds: Analyse only this many seconds from the middle of
                the track. Intros and outros are unrepresentative of a DJ tool's
                subject matter, and excerpting is several times faster. Falsy
                values analyse the whole file.
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.n_mfcc = n_mfcc
        self.excerpt_seconds = excerpt_seconds
        self.layout = build_layout(n_mfcc)

    def extract(self, filepath: str) -> Optional[np.ndarray]:
        """Extract features, returning None when the file cannot be decoded."""
        try:
            y, sr = self._load(filepath)
            if y is None or len(y) == 0:
                return None
            return self._extract_all_features(y, sr)
        except Exception as exc:
            logger.warning("Feature extraction failed for %s: %s", filepath, exc)
            return None

    def _load(self, filepath: str):
        """Load audio, taking a centred excerpt when configured."""
        if not self.excerpt_seconds:
            return librosa.load(filepath, sr=self.sample_rate, mono=True)

        duration = self.get_audio_duration(filepath)
        if duration and duration > self.excerpt_seconds:
            offset = max(0.0, (duration - self.excerpt_seconds) / 2.0)
            return librosa.load(
                filepath,
                sr=self.sample_rate,
                mono=True,
                offset=offset,
                duration=self.excerpt_seconds,
            )
        return librosa.load(filepath, sr=self.sample_rate, mono=True)

    def _extract_all_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        # Frame-level blocks, in the order declared by features._frame_blocks.
        frame_features = [
            librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.frame_size, hop_length=self.hop_size
            ).T,
            librosa.feature.spectral_centroid(
                y=y, sr=sr, n_fft=self.frame_size, hop_length=self.hop_size
            ).T,
            librosa.feature.spectral_rolloff(
                y=y, sr=sr, n_fft=self.frame_size, hop_length=self.hop_size
            ).T,
            librosa.feature.spectral_contrast(
                y=y, sr=sr, n_fft=self.frame_size, hop_length=self.hop_size
            ).T,
            librosa.feature.zero_crossing_rate(
                y=y, frame_length=self.frame_size, hop_length=self.hop_size
            ).T,
            librosa.feature.chroma_stft(
                y=y, sr=sr, n_fft=self.frame_size, hop_length=self.hop_size
            ).T,
            librosa.feature.rms(y=y, frame_length=self.frame_size, hop_length=self.hop_size).T,
        ]

        aggregated = aggregate_features(np.concatenate(frame_features, axis=1))
        vector = np.concatenate(
            [
                aggregated,
                self._extract_rhythmic_features(y, sr),
                self._extract_highlevel_features(y, sr),
            ]
        )
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        if len(vector) != self.layout.dim:
            raise ValueError(
                f"Extracted {len(vector)} features but layout expects {self.layout.dim}"
            )
        return vector

    def _extract_rhythmic_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            tempo_value = float(tempo) if np.isscalar(tempo) else float(np.atleast_1d(tempo)[0])
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            return np.array(
                [
                    tempo_value,
                    float(np.mean(onset_env)),
                    float(np.std(onset_env)),
                    float(np.max(onset_env)),
                ]
            )
        except Exception:
            return np.zeros(4)

    def _extract_highlevel_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        energy = float(np.sum(y**2) / len(y))
        dynamic_range = float(np.max(np.abs(y)) - np.min(np.abs(y)))
        spec_bw = librosa.feature.spectral_bandwidth(
            y=y, sr=sr, n_fft=self.frame_size, hop_length=self.hop_size
        )
        flatness = librosa.feature.spectral_flatness(
            y=y, n_fft=self.frame_size, hop_length=self.hop_size
        )
        return np.array(
            [
                energy,
                dynamic_range,
                float(np.mean(spec_bw)),
                float(np.std(spec_bw)),
                float(np.mean(flatness)),
                float(np.std(flatness)),
            ]
        )

    def get_audio_duration(self, filepath: str) -> Optional[float]:
        """Duration in seconds, or None if it cannot be determined."""
        try:
            return float(sf.info(filepath).duration)
        except Exception:
            try:
                return float(librosa.get_duration(path=filepath))
            except Exception:
                return None
