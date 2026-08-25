"""Audio feature extraction.

Produces the fixed-length vector described by :mod:`music_cluster.features`.
The block order here and the layout there must stay in sync.

Two profiles are supported. ``music`` analyses a representative excerpt from
the middle of a track. ``sample`` analyses a one-shot or loop from its first
sample — a kick's attack is the most identifying thing about it, and it is
gone 20 ms in — and measures an extra block of descriptors that only mean
something for a single sonic event.
"""

import itertools
import logging
import warnings
from typing import Optional

import librosa
import numpy as np
import soundfile as sf

from . import profiles
from .features import BAND_EDGES, N_BANDS, aggregate_features, build_layout


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("audioread.ffdec").setLevel(logging.ERROR)
logging.getLogger("audioread").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)


# Below this length a sample is treated as a single hit, and tempo estimation
# is skipped rather than invented.
SAMPLE_TEMPO_MIN_SECONDS = 2.0

# Two envelope peaks closer than this are one event with an internal texture,
# not two hits — a clap is three transients inside 30 ms.
MIN_HIT_SPACING_SECONDS = 0.05

# Past this many hits a file is "a loop", and the exact number says nothing.
MAX_COUNTED_HITS = 64

# Harmonic/percussive separation runs on its own spectrogram: coarser in time
# and bounded in length, because it is the most expensive step by far.
HPSS_HOP = 1024
HPSS_MAX_FRAMES = 256  # ~6 seconds at 44.1 kHz
MIN_HPSS_FRAMES = 16  # enough shape to decompose, even for a 70 ms tick

# Counting pitch classes is only worth its cost when there is a pitch to
# count. Below both of these, the sample is percussive or noisy and has none.
PITCHED_MIN_CONFIDENCE = 0.25
PITCHED_MIN_SUPPORT = 0.25

# Constant-Q settings for counting pitch classes. Six octaves from C1 (32.7 Hz)
# reaches past 2 kHz, which covers every note anyone puts in a sample folder.
CHROMA_HOP = 2048
CHROMA_OCTAVES = 6
CHROMA_FMIN = 32.703  # C1


def _first_time_at_or_above(envelope: np.ndarray, times: np.ndarray, threshold: float) -> float:
    """When the envelope first reaches a level, or its last time if never."""
    hits = np.flatnonzero(envelope >= threshold)
    index = int(hits[0]) if len(hits) else len(envelope) - 1
    return float(times[min(index, len(times) - 1)])


def _first_time_at_or_below(
    envelope: np.ndarray, times: np.ndarray, threshold: float, fallback: float
) -> float:
    """When the envelope first drops to a level, or ``fallback`` if it never does."""
    hits = np.flatnonzero(envelope <= threshold)
    if not len(hits):
        return fallback
    return float(times[min(int(hits[0]), len(times) - 1)])


def _percussive_ratio(y: np.ndarray, sr: int, n_fft: int) -> float:
    """Share of the energy that is percussive rather than harmonic.

    Near 1 for a hat, near 0 for a held chord. Returns a neutral 0.5 when the
    decomposition cannot run — a two-frame sample carries no evidence either
    way, and claiming it does would skew the space.

    Deliberately measured on its own coarser spectrogram: the median filtering
    that separates horizontal (harmonic) from vertical (percussive) structure
    is by far the most expensive thing done to a sample, and it costs the same
    per frame whether or not those frames are 6 ms apart. At 23 ms the split
    is if anything cleaner — percussive energy stays vertical while harmonic
    ridges get longer relative to the kernel — and several times cheaper.
    """
    # The coarse hop is a cost control for long files. A 70 ms hat produces one
    # frame at that hop and no measurement at all, so short files — which are
    # cheap either way — get a hop fine enough to have a shape to decompose.
    hop = int(min(HPSS_HOP, max(64, len(y) // MIN_HPSS_FRAMES)))
    try:
        magnitude = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    except Exception:
        return 0.5

    frames = magnitude.shape[1] if magnitude.ndim == 2 else 0
    if frames < 3:
        return 0.5
    # Long loops are all texture by the second bar; the ratio is established
    # well before then, and the cost is linear in frames.
    if frames > HPSS_MAX_FRAMES:
        magnitude = magnitude[:, :HPSS_MAX_FRAMES]
        frames = HPSS_MAX_FRAMES

    # The median filter's time kernel has to fit inside the spectrogram, which
    # for a 60 ms tick is a handful of frames.
    time_kernel = int(min(31, frames if frames % 2 else frames - 1))
    try:
        harmonic, percussive = librosa.decompose.hpss(
            magnitude, kernel_size=(31, max(3, time_kernel))
        )
    except Exception:
        return 0.5

    harmonic_energy = float(np.sum(np.square(harmonic)))
    percussive_energy = float(np.sum(np.square(percussive)))
    total = harmonic_energy + percussive_energy
    return float(percussive_energy / total) if total > 0 else 0.5


def _harmonic_support(spectrum: np.ndarray, freqs: np.ndarray, f0: float) -> float:
    """How much energy sits on the harmonics of ``f0``.

    High for a plucked or bowed note, low for noise that merely has a spectral
    peak — which is how a tuned 808 is told from a snare whose body rings.
    """
    if f0 <= 0 or spectrum.size < 2 or freqs.size < 2:
        return 0.0

    total = float(spectrum.sum())
    if total <= 0:
        return 0.0

    # A quarter-tone either side, so a slightly detuned partial still counts —
    # but never narrower than the analysis itself can resolve, which at 20 Hz
    # per bin is much wider than a quarter-tone down in the bass.
    bin_width = float(freqs[1] - freqs[0])
    ratio = 2.0 ** (0.5 / 12.0)

    support = 0.0
    for harmonic in (1, 2, 3, 4):
        centre = f0 * harmonic
        if centre >= freqs[-1]:
            break
        tolerance = max(centre * (ratio - 1.0), 1.5 * bin_width)
        window = (freqs >= centre - tolerance) & (freqs <= centre + tolerance)
        if np.any(window):
            support += float(spectrum[window].sum())

    return float(min(support / total, 1.0))


def _chroma_peak_count(y: np.ndarray, sr: int) -> float:
    """Roughly how many distinct pitch classes are sounding.

    One for a bass note or a lead, three or more for a chord — which is the
    difference between two of the piles a producer actually keeps.

    Constant-Q rather than the STFT chroma used elsewhere: at the short frame
    a sample needs, an FFT bin is tens of Hz wide and the resulting chroma is
    too smeared to count anything, reporting a triad and a noise burst alike.
    """
    if len(y) < 512:
        return 0.0
    try:
        chroma = librosa.feature.chroma_cqt(
            y=y,
            sr=sr,
            hop_length=CHROMA_HOP,
            n_octaves=CHROMA_OCTAVES,
            fmin=CHROMA_FMIN,
        ).mean(axis=1)
    except Exception:
        return 0.0
    if chroma.size == 0:
        return 0.0

    # Subtracting the median removes the broadband floor that noise spreads
    # evenly across all twelve bins, leaving only genuine peaks standing.
    whitened = np.clip(chroma - np.median(chroma), 0.0, None)
    ceiling = float(whitened.max())
    if ceiling <= 0:
        return 0.0
    return float(np.count_nonzero(whitened >= 0.5 * ceiling))


def _onset_count(envelope: np.ndarray, frame_seconds: float) -> float:
    """How many hits are in this file: one shot, or a bar of them?

    Peak-picked from the amplitude envelope rather than from an onset-strength
    curve, because at a sample's hop length the usual detector fires several
    times down the tail of a single decaying kick.

    A count rather than a rate: a 60 ms tick and a two-bar loop both come out
    at roughly ten hits per second, and it is the *count* that says which is
    which. How long the file is, is already recorded separately.
    """
    if envelope.size == 0 or frame_seconds <= 0:
        return 0.0

    peak = float(envelope.max())
    if peak <= 0:
        return 0.0

    # Two hits closer together than this are one hit with a texture — the
    # several micro-transients inside a clap, or a flammed snare.
    spacing = max(1, round(MIN_HIT_SPACING_SECONDS / frame_seconds))
    try:
        from scipy.signal import find_peaks

        # Prominence is what keeps the ripple down the tail of a sustained
        # note from being counted as a second hit: a real onset rises clear of
        # the level it started from, a ripple does not.
        found, _ = find_peaks(envelope, height=0.35 * peak, prominence=0.2 * peak, distance=spacing)
        count = len(found)
    except Exception:
        count = 1

    # A construction-kit loop can carry hundreds; past a couple of bars the
    # exact number stops meaning anything and only distorts distances.
    return float(min(max(count, 1), MAX_COUNTED_HITS))


class FeatureExtractor:
    """Extract a track-level feature vector from an audio file."""

    def __init__(
        self,
        sample_rate: int = 44100,
        frame_size: int = 2048,
        hop_size: int = 1024,
        n_mfcc: int = 20,
        excerpt_seconds: Optional[float] = 90,
        profile: str = profiles.MUSIC,
        max_seconds: Optional[float] = None,
    ):
        """
        Args:
            excerpt_seconds: Analyse only this many seconds from the middle of
                the track. Intros and outros are unrepresentative of a DJ tool's
                subject matter, and excerpting is several times faster. Falsy
                values analyse the whole file.
            profile: ``music`` or ``sample``. Selects the feature layout and
                how audio is read off disk.
            max_seconds: Hard cap on how much audio is read, counted from the
                start. Used by the sample profile, where excerpting from the
                middle would throw away the attack but a stray five-minute
                construction kit still should not be decoded in full.
        """
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.n_mfcc = n_mfcc
        self.excerpt_seconds = excerpt_seconds
        self.profile = profiles.normalize(profile)
        self.max_seconds = max_seconds
        self.layout = build_layout(n_mfcc, self.profile)

    @property
    def is_sample(self) -> bool:
        return self.profile == profiles.SAMPLE

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
        """Load audio: a centred excerpt for tracks, the head for samples."""
        if self.is_sample:
            return librosa.load(
                filepath,
                sr=self.sample_rate,
                mono=True,
                duration=self.max_seconds or None,
            )

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
        blocks = [
            aggregated,
            self._extract_rhythmic_features(y, sr),
            self._extract_highlevel_features(y, sr),
        ]
        if self.is_sample:
            blocks.append(self._extract_sample_features(y, sr))

        vector = np.concatenate(blocks)
        vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        if len(vector) != self.layout.dim:
            raise ValueError(
                f"Extracted {len(vector)} features but layout expects {self.layout.dim}"
            )
        return vector

    def _extract_rhythmic_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        try:
            if self.is_sample and len(y) / float(sr) < SAMPLE_TEMPO_MIN_SECONDS:
                # A single hit has no tempo. Beat tracking still returns one —
                # essentially its prior — and a fabricated 117 BPM sitting in
                # every kick's vector is worse than an honest zero.
                tempo_value = 0.0
            else:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo_value = float(np.atleast_1d(np.asarray(tempo, dtype=float))[0])
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

    # ------------------------------------------------------------------
    # Sample profile
    # ------------------------------------------------------------------

    def _extract_sample_features(self, y: np.ndarray, sr: int) -> np.ndarray:
        """The one-shot block: envelope, event shape, pitch, spectral balance.

        Everything here is derived from one STFT and one amplitude envelope,
        because a sample library is tens of thousands of tiny files and the
        per-file overhead is what decides whether analysing it is a coffee
        break or an afternoon.
        """
        duration = len(y) / float(sr)
        magnitude = np.abs(librosa.stft(y, n_fft=self.frame_size, hop_length=self.hop_size))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.frame_size)
        envelope = librosa.feature.rms(y=y, frame_length=self.frame_size, hop_length=self.hop_size)[
            0
        ]
        times = librosa.frames_to_time(np.arange(len(envelope)), sr=sr, hop_length=self.hop_size)

        frame_seconds = float(self.hop_size) / float(sr)

        return np.concatenate(
            [
                self._envelope_features(y, envelope, times, duration),
                np.array(
                    [
                        _onset_count(envelope, frame_seconds),
                        _percussive_ratio(y, sr, self.frame_size),
                    ]
                ),
                self._pitch_features(y, magnitude, freqs, sr),
                self._balance_features(magnitude, freqs, envelope, sr),
            ]
        )

    def _envelope_features(
        self, y: np.ndarray, envelope: np.ndarray, times: np.ndarray, duration: float
    ) -> np.ndarray:
        """Attack, decay, sustain, release — the shape of the event.

        This is what separates a clap from a snare, or a pluck from a pad,
        when their spectra are near enough identical.
        """
        peak = float(envelope.max()) if len(envelope) else 0.0
        log_duration = float(np.log1p(duration))
        if peak <= 0.0 or len(envelope) < 2:
            return np.array([0.0] * 7 + [log_duration, 0.0])

        peak_index = int(np.argmax(envelope))
        peak_time = float(times[peak_index])
        frame_seconds = float(times[1] - times[0]) if len(times) > 1 else duration

        rise = envelope[: peak_index + 1]
        start = _first_time_at_or_above(rise, times, 0.1 * peak)
        knee = _first_time_at_or_above(rise, times, 0.9 * peak)
        attack_time = max(knee - start, 0.0)
        # Compressed: a 2 ms transient and a 2 s swell are three orders of
        # magnitude apart, and untamed the difference dominates every distance.
        attack_slope = float(np.log1p((0.8 * peak) / max(attack_time, frame_seconds)))

        fall = envelope[peak_index:]
        fall_times = times[peak_index:]
        decay_time = _first_time_at_or_below(fall, fall_times, 0.5 * peak, duration) - peak_time
        release_time = _first_time_at_or_below(fall, fall_times, 0.05 * peak, duration) - peak_time

        sounding = envelope >= 0.1 * peak
        tail = envelope[peak_index:][sounding[peak_index:]]
        sustain_ratio = float(np.mean(tail) / peak) if len(tail) else 0.0
        effective_duration = float(np.count_nonzero(sounding) * frame_seconds)

        total = float(envelope.sum())
        temporal_centroid = (
            float(np.sum(times * envelope) / total / max(duration, 1e-6)) if total > 0 else 0.0
        )

        rms = float(np.sqrt(np.mean(np.square(y))))
        crest_factor = float(np.max(np.abs(y)) / rms) if rms > 1e-9 else 0.0

        return np.array(
            [
                attack_time,
                attack_slope,
                max(decay_time, 0.0),
                sustain_ratio,
                max(release_time, 0.0),
                float(np.clip(temporal_centroid, 0.0, 1.0)),
                effective_duration,
                log_duration,
                crest_factor,
            ]
        )

    def _pitch_features(
        self, y: np.ndarray, magnitude: np.ndarray, freqs: np.ndarray, sr: int
    ) -> np.ndarray:
        """Does this sample have a note, is it steady, and is it one note?

        A bass, a stab and a chord differ from a snare here more than anywhere
        else, and a chord differs from a bass by how many notes are in it.
        """
        blank = np.zeros(5)
        if magnitude.size == 0:
            return blank

        try:
            pitches, magnitudes = librosa.piptrack(
                S=magnitude, sr=sr, fmin=30.0, fmax=4000.0, threshold=0.1
            )
        except Exception:
            return blank

        best = np.argmax(magnitudes, axis=0)
        frame_index = np.arange(magnitudes.shape[1])
        strength = magnitudes[best, frame_index]
        tracked = pitches[best, frame_index]

        # A frame only votes if its pitch peak stands out within the file;
        # every frame has *some* maximum, including the silence after a hit.
        ceiling = float(strength.max()) if strength.size else 0.0
        salient = (tracked > 0) & (strength >= 0.1 * ceiling) & (strength > 0)
        if not np.any(salient):
            return blank

        voiced = tracked[salient]
        f0 = float(np.median(voiced))
        semitones = 12.0 * np.log2(np.maximum(voiced, 1e-6) / max(f0, 1e-6))

        # "Is there a note here" is not how many frames were loud enough — it
        # is how many of them agreed on the same note. Noise is salient in
        # every frame and settles on a different pitch in each one.
        confidence = float(np.count_nonzero(np.abs(semitones) <= 1.0) / len(semitones))
        # Median absolute deviation, so one octave-jumped frame does not
        # report a rock-steady bass note as unstable.
        deviation = float(np.median(np.abs(semitones)))
        stability = float(1.0 / (1.0 + deviation))

        support = _harmonic_support(magnitude.mean(axis=1), freqs, f0)

        # Skip the constant-Q transform for material that has no note to
        # resolve. It is the single most expensive measurement here, and most
        # of any sample folder is drums. Either kind of evidence is enough on
        # its own: a pad's slow beating partials confuse frame-to-frame pitch
        # tracking while sitting squarely on their harmonic series.
        pitched = confidence >= PITCHED_MIN_CONFIDENCE or support >= PITCHED_MIN_SUPPORT
        peaks = _chroma_peak_count(y, sr) if pitched else 0.0

        return np.array([f0, confidence, stability, support, peaks])

    def _balance_features(
        self, magnitude: np.ndarray, freqs: np.ndarray, envelope: np.ndarray, sr: int
    ) -> np.ndarray:
        """Where the energy sits, and where it moves as the sample decays.

        The band split alone tells a sub from a kick from a snare from a hat;
        the brightness slope catches a filtered decay, which is most of what
        makes a stab a stab.
        """
        power = np.square(magnitude)
        totals = np.array(
            [
                float(power[(freqs >= low) & (freqs < high)].sum())
                for low, high in itertools.pairwise(BAND_EDGES)
            ]
        )
        grand_total = float(totals.sum())
        bands = totals / grand_total if grand_total > 0 else np.full(N_BANDS, 1.0 / N_BANDS)

        centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
        peak = float(envelope.max()) if len(envelope) else 0.0
        sounding = np.flatnonzero(envelope >= 0.1 * peak) if peak > 0 else np.arange(len(centroid))
        sounding = sounding[sounding < len(centroid)]
        if len(sounding) == 0:
            sounding = np.arange(len(centroid))

        head = sounding[: max(1, len(sounding) // 4)]
        tail = sounding[len(sounding) // 2 :]
        brightness_attack = float(np.mean(centroid[head])) if len(head) else 0.0
        brightness_decay = float(np.mean(centroid[tail])) if len(tail) else brightness_attack

        # Loudness-invariant: normalise each frame before differencing, so a
        # quiet sample is not read as having a softer transient.
        norms = np.linalg.norm(magnitude, axis=0)
        normalised = magnitude / np.maximum(norms, 1e-9)
        flux = np.sqrt(np.sum(np.square(np.maximum(np.diff(normalised, axis=1), 0.0)), axis=0))

        return np.concatenate(
            [
                bands,
                [
                    brightness_attack,
                    brightness_decay,
                    brightness_decay - brightness_attack,
                    float(np.mean(flux)) if flux.size else 0.0,
                    float(np.std(flux)) if flux.size else 0.0,
                ],
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
