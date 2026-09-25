"""What a track's drums are doing: the measurements behind beat-type tags.

The feature vector knows a track's tempo and how busy its onsets are, which is
enough to tell 124 from 174 but not a four-to-the-floor from a 2-step at the
same tempo. What separates those is *where in the bar* the kick and the snare
land, and this module measures exactly that.

The method is deliberately plain:

1. Split the spectrum into three bands that line up with a drum kit — kick
   (below 150 Hz), snare and clap (roughly 1-5 kHz), hats and shakers (above
   6 kHz) — and take an onset-strength curve for each.
2. Track the beat, then sample every band at twelve points inside every beat.
   Twelve divides into both sixteenths (every 3) and triplets (every 4), so
   swing and shuffles land on grid points instead of between them.
3. Fold the beats into four-beat bars and average, giving one 48-step bar per
   band: the track's drum pattern, as a producer would draw it in a sequencer.
4. Choose which beat is "the one" by where the kick sits and the snare does not.

Everything returned is a plain number or list, so the rules that turn it into
tags (:mod:`music_cluster.tagging`) can be tested without any audio at all.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np


logger = logging.getLogger(__name__)

# Twelve grid points per beat: sixteenths are every 3rd point, eighth-note
# triplets every 4th. A bar of 4/4 is therefore 48 steps.
STEPS_PER_BEAT = 12
BEATS_PER_BAR = 4
STEPS_PER_BAR = STEPS_PER_BEAT * BEATS_PER_BAR
SIXTEENTH = STEPS_PER_BEAT // 4

# Band edges in Hz. The snare band skips the 150 Hz - 1 kHz region on purpose:
# that is where kick harmonics and basslines live, and letting them in makes
# every kick look like a snare too.
KICK_BAND = (20.0, 150.0)
SNARE_BAND = (1000.0, 5000.0)
HAT_BAND = (6000.0, 16000.0)

BANDS = ("kick", "snare", "hat")

# About 11.6 ms per analysis frame whatever the sample rate, so a sixteenth at
# 175 BPM (86 ms) still spans several frames.
FRAME_SECONDS = 0.0116

# A tracked tempo below this is checked for being a half-time reading of a
# faster track, and the share of busy sixteenths that says it is.
DOUBLE_BELOW_BPM = 100.0
DOUBLE_TIME_BUSYNESS = 0.7
DOUBLE_TIME_ONSETS_PER_BEAT = 2.8
# ...and a tracked tempo at or above this is checked for being a double-time
# reading of a slow one.
HALVE_ABOVE_BPM = 150.0

# Below this, the chosen reading's bars disagree enough that the tracker is
# probably locked on to a cross-rhythm, and related tempi are tried.
RETRACK_BELOW_CONSISTENCY = 0.85

# Fewer beats than this and there is no pattern to average, just an accident.
MIN_BEATS = 8


@dataclass
class RhythmProfile:
    """The measured groove of one track."""

    bpm: float = 0.0
    # How clearly the onset curve repeats at the beat period, 0-1.
    pulse_clarity: float = 0.0
    # How evenly the tracked beats are spaced, 0-1 (1 = a drum machine).
    beat_regularity: float = 0.0
    # Onsets per second across the whole spectrum.
    onset_rate: float = 0.0
    # How sharply energy rises relative to its level: near zero for pads and
    # drones, 0.05 and up for anything with drums in it.
    transient_strength: float = 0.0
    # Share of offbeat sixteenths pushed late towards the triplet, 0-1.
    swing: float = 0.0
    # How much of the spectrum's onset energy each band carries, 0-1 each.
    band_activity: Dict[str, float] = field(default_factory=dict)
    # One averaged 48-step bar per band, each scaled to 0-1.
    pattern: Dict[str, List[float]] = field(default_factory=dict)
    bars: int = 0
    # True when the beat tracker's tempo was doubled because the drums said
    # the track was being counted at half speed.
    double_time: bool = False

    @property
    def has_pattern(self) -> bool:
        return self.bars > 0 and all(len(self.pattern.get(band, [])) for band in BANDS)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bpm": round(self.bpm, 2),
            "pulse_clarity": round(self.pulse_clarity, 4),
            "beat_regularity": round(self.beat_regularity, 4),
            "onset_rate": round(self.onset_rate, 3),
            "transient_strength": round(self.transient_strength, 5),
            "swing": round(self.swing, 4),
            "band_activity": {k: round(v, 4) for k, v in self.band_activity.items()},
            "pattern": {k: [round(x, 3) for x in v] for k, v in self.pattern.items()},
            "bars": self.bars,
            "double_time": self.double_time,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "RhythmProfile":
        data = data or {}
        return cls(
            bpm=float(data.get("bpm") or 0.0),
            pulse_clarity=float(data.get("pulse_clarity") or 0.0),
            beat_regularity=float(data.get("beat_regularity") or 0.0),
            onset_rate=float(data.get("onset_rate") or 0.0),
            transient_strength=float(data.get("transient_strength") or 0.0),
            swing=float(data.get("swing") or 0.0),
            band_activity=dict(data.get("band_activity") or {}),
            pattern={k: list(v) for k, v in (data.get("pattern") or {}).items()},
            bars=int(data.get("bars") or 0),
            double_time=bool(data.get("double_time")),
        )


def sixteenths(pattern: List[float]) -> np.ndarray:
    """A 48-step pattern read on the sixteenth grid (16 values)."""
    values = np.asarray(pattern, dtype=float)
    if values.size != STEPS_PER_BAR:
        return np.zeros(16)
    return values[::SIXTEENTH]


def _band_onsets(
    power_mel: np.ndarray, mel_freqs: np.ndarray, band: Tuple[float, float]
) -> np.ndarray:
    """Onset strength of one band: positive change in its log energy."""
    rows = (mel_freqs >= band[0]) & (mel_freqs < band[1])
    if not np.any(rows):
        return np.zeros(power_mel.shape[1])
    # Amplitude rather than log energy. A log curve turns the faint click
    # every kick leaks into the top of the spectrum into a rise as big as a
    # real hat, and then every band shows the kick pattern.
    energy = np.sqrt(power_mel[rows].sum(axis=0))
    flux = np.maximum(np.diff(energy, prepend=energy[:1]), 0.0)
    return flux


def _tracking_envelope(full: np.ndarray, kick: np.ndarray) -> np.ndarray:
    """What the beat tracker listens to: every onset, with the kick leaned on.

    Syncopated snares — a dembow's 3+3+2, a breakbeat's ghost notes — form
    periodicities of their own that a tracker happily locks on to, reporting
    a 95 BPM reggaeton beat as 126. In nearly everything a DJ plays, the kick
    is what the crowd counts, so it gets an equal say.
    """

    def unit(curve: np.ndarray) -> np.ndarray:
        peak = float(np.percentile(curve, 99)) if curve.size else 0.0
        return np.clip(curve / peak, 0.0, 1.5) if peak > 1e-12 else curve

    size = min(full.size, kick.size)
    return unit(full[:size]) + unit(kick[:size])


def _track(
    curve: np.ndarray, sr: int, hop: int, start_bpm: float = 120.0, tightness: float = 100.0
) -> Tuple[float, np.ndarray]:
    """Tempo and beat frames for an onset curve.

    Tempo is estimated on the curve at half resolution — 23 ms, librosa's own
    default — because the autocorrelation behind it is the costliest step in
    tagging and gains nothing from finer frames. The beats themselves are then
    placed on the full-resolution curve, where the pattern is read.
    """
    usable = curve[: curve.size - curve.size % 2]
    coarse = usable.reshape(-1, 2).max(axis=1) if usable.size else usable
    if coarse.size < 4:
        return 0.0, np.zeros(0)
    tempo = librosa.feature.tempo(
        onset_envelope=coarse, sr=sr, hop_length=hop * 2, start_bpm=start_bpm
    )
    bpm = float(np.atleast_1d(np.asarray(tempo, dtype=float))[0]) if np.size(tempo) else 0.0
    if bpm <= 0:
        return 0.0, np.zeros(0)
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=curve,
        sr=sr,
        hop_length=hop,
        trim=False,
        units="frames",
        bpm=bpm,
        tightness=tightness,
    )
    bpm = float(np.atleast_1d(np.asarray(tempo, dtype=float))[0]) if np.size(tempo) else 0.0
    return bpm, np.asarray(beats, dtype=float)


def _grid(envelope: np.ndarray, beats: np.ndarray) -> np.ndarray:
    """Sample an envelope at STEPS_PER_BEAT points inside every beat.

    Each point takes the maximum over a grid step's width centred on it,
    rather than one interpolated frame, so a hit a few milliseconds off the
    grid — which is every hit a human played — still lands on it.
    """
    from scipy.ndimage import maximum_filter1d

    if beats.size < 2 or envelope.size == 0:
        return np.zeros((0, STEPS_PER_BEAT))
    intervals = np.diff(beats)
    step = float(np.median(intervals)) / STEPS_PER_BEAT
    widened = maximum_filter1d(envelope, size=int(np.ceil(step)) + 1, mode="nearest")
    offsets = np.arange(STEPS_PER_BEAT) / STEPS_PER_BEAT
    centres = beats[:-1, None] + offsets[None, :] * intervals[:, None]
    index = np.clip(np.round(centres).astype(int), 0, envelope.size - 1)
    return widened[index]


def _align(beats: np.ndarray, envelope: np.ndarray, reach: int = 3) -> np.ndarray:
    """Nudge the beat grid onto the hits by a constant few frames.

    A beat tracker reports where its own onset curve peaks, which can sit a
    frame or two away from where each band's energy actually rises. Half a
    sixteenth at 175 BPM is only four frames, so that small lag is enough to
    move every downbeat into the previous beat. One shift for the whole
    track, chosen to put the most onset energy exactly on the beats.
    """
    best_shift, best_total = 0, -np.inf
    limit = len(envelope) - 1
    for shift in range(-reach, reach + 1):
        frames = np.clip(np.round(beats + shift).astype(int), 0, limit)
        total = float(envelope[frames].sum())
        if total > best_total + 1e-9:
            best_shift, best_total = shift, total
    return beats + best_shift


def _normalise(pattern: np.ndarray) -> np.ndarray:
    """Scale a pattern to 0-1 above its own floor.

    Onset strength never quite reaches zero in a mixed track, so the floor is
    the median step rather than zero — what is left is what stands out.
    """
    floor = float(np.median(pattern))
    ceiling = float(pattern.max())
    if ceiling - floor <= 1e-9:
        return np.zeros_like(pattern)
    return np.clip((pattern - floor) / (ceiling - floor), 0.0, 1.0)


def _choose_downbeat(per_beat: Dict[str, np.ndarray]) -> int:
    """Which of the first four tracked beats is beat one of the bar.

    Beat one is where the kick is and the snare is not; the backbeat (or the
    half-time snare on three) is where the snare is. Every common pattern
    agrees on that much, and for the ones that are symmetric under a
    two-beat shift — a four-to-the-floor with claps on two and four — either
    answer reads the same.
    """
    kick = per_beat["kick"][:, 0]
    snare = per_beat["snare"][:, 0]
    best_phase, best_score = 0, -np.inf
    for phase in range(BEATS_PER_BAR):
        k = kick[phase::BEATS_PER_BAR].mean() if len(kick[phase::BEATS_PER_BAR]) else 0.0
        s = snare[phase::BEATS_PER_BAR].mean() if len(snare[phase::BEATS_PER_BAR]) else 0.0
        score = k - s
        if score > best_score + 1e-9:
            best_phase, best_score = phase, score
    return best_phase


def _fold(per_beat: np.ndarray, phase: int) -> Tuple[np.ndarray, np.ndarray]:
    """Whole bars starting at ``phase``: (their median bar, every bar)."""
    usable = per_beat[phase:]
    bars = len(usable) // BEATS_PER_BAR
    if bars == 0:
        return np.zeros(STEPS_PER_BAR), np.zeros((0, STEPS_PER_BAR))
    stacked = usable[: bars * BEATS_PER_BAR].reshape(bars, STEPS_PER_BAR)
    # Median rather than mean: one fill or one breakdown bar should not
    # rewrite the groove of the other thirty.
    return np.median(stacked, axis=0), stacked


def _swing(hat: np.ndarray, beats: np.ndarray) -> float:
    """How late the offbeat sixteenths sit: 0 = straight, 1 = triplet shuffle.

    Measured from individual hat onsets rather than the folded grid, because
    the whole effect is a few tens of milliseconds — a 2-step at 132 BPM is
    fully swung when its offbeat hats land 38 ms late — and folding averages
    exactly that detail away. Each hat onset is placed as a fraction of its
    beat; straight offbeat sixteenths sit at 0.25 and 0.75, swung ones drift
    towards the triplet at 0.33 and 0.83.
    """
    if hat.size == 0 or beats.size < 2:
        return 0.0
    try:
        from scipy.signal import find_peaks

        peak = float(np.percentile(hat, 99.5))
        if peak <= 0:
            return 0.0
        spacing = max(1, int(np.median(np.diff(beats)) / 8))
        found, _ = find_peaks(hat, height=0.3 * peak, distance=spacing)
    except Exception:
        return 0.0

    offsets: List[float] = []
    positions = np.searchsorted(beats, found, side="right") - 1
    for frame, index in zip(found, positions, strict=True):
        if index < 0 or index + 1 >= beats.size:
            continue
        start, end = beats[index], beats[index + 1]
        fraction = (frame - start) / max(end - start, 1e-9)
        for straight in (0.25, 0.75):
            if straight - 0.08 <= fraction <= straight + 0.17:
                offsets.append((fraction - straight) / (1.0 / 12.0))

    # A handful of offbeat hats is a fill, not a feel.
    if len(offsets) < 8:
        return 0.0
    return float(np.clip(np.median(offsets), 0.0, 1.0))


def _pulse_clarity(onsets: np.ndarray, period_frames: float) -> float:
    """Autocorrelation of the onset curve at one beat, relative to lag zero."""
    if onsets.size < 4 or period_frames < 1:
        return 0.0
    centred = onsets - onsets.mean()
    denom = float(np.dot(centred, centred))
    if denom <= 1e-12:
        return 0.0
    lag = round(period_frames)
    best = 0.0
    # A beat tracker's period is within a frame or two of the true one.
    for offset in (-1, 0, 1):
        shift = lag + offset
        if 0 < shift < centred.size:
            best = max(best, float(np.dot(centred[:-shift], centred[shift:]) / denom))
    return float(np.clip(best, 0.0, 1.0))


def analyze(y: np.ndarray, sr: int) -> RhythmProfile:
    """Measure the groove of a mono signal. Never raises."""
    try:
        return _analyze(np.asarray(y, dtype=float), int(sr))
    except Exception as exc:  # pragma: no cover - defensive, librosa edge cases
        logger.warning("Rhythm analysis failed: %s", exc)
        return RhythmProfile()


def _analyze(y: np.ndarray, sr: int) -> RhythmProfile:
    if y.size < sr * 2:
        return RhythmProfile()

    hop = max(64, round(sr * FRAME_SECONDS))
    n_fft = 2048 if sr <= 24000 else 4096
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=96, fmin=20.0, fmax=min(16000.0, sr / 2)
    )
    mel_freqs = librosa.mel_frequencies(n_mels=96, fmin=20.0, fmax=min(16000.0, sr / 2))

    envelopes = {
        "kick": _band_onsets(mel, mel_freqs, KICK_BAND),
        "snare": _band_onsets(mel, mel_freqs, SNARE_BAND),
        "hat": _band_onsets(mel, mel_freqs, HAT_BAND),
    }
    full = librosa.onset.onset_strength(S=librosa.power_to_db(mel), sr=sr, hop_length=hop)

    band_totals = {band: float(env.sum()) for band, env in envelopes.items()}
    grand = sum(band_totals.values())
    band_activity = {
        band: (total / grand if grand > 0 else 0.0) for band, total in band_totals.items()
    }

    duration = y.size / float(sr)
    peaks = librosa.onset.onset_detect(onset_envelope=full, sr=sr, hop_length=hop, units="frames")
    onset_rate = float(len(peaks) / duration) if duration > 0 else 0.0

    # Onset curves are scale-free, so a slowly breathing pad produces one just
    # as peaky as a drum loop once normalised. What a pad does not have is
    # energy that *jumps*: its rises are tiny next to its level.
    amplitude = np.sqrt(mel.sum(axis=0))
    rises = np.maximum(np.diff(amplitude, prepend=amplitude[:1]), 0.0)
    transient_strength = float(rises.sum() / amplitude.sum()) if amplitude.sum() > 0 else 0.0

    profile = RhythmProfile(
        onset_rate=onset_rate,
        band_activity=band_activity,
        transient_strength=transient_strength,
    )
    combined = envelopes["kick"] + envelopes["snare"] + envelopes["hat"]

    # Two listeners: one hearing every onset, one leaning on the kick. Each
    # is right where the other is fooled, so both are asked and the reading
    # whose bars agree with each other best is kept.
    candidates = []
    for curve in (full, _tracking_envelope(full, envelopes["kick"])):
        bpm, beats = _track(curve, sr, hop)
        if bpm <= 0 or beats.size < MIN_BEATS:
            continue
        reading = _resolve_octave(_align(beats, combined), bpm, envelopes, combined, onset_rate)
        if reading is not None:
            candidates.append(reading)

    if not candidates:
        return profile

    best = max(candidates, key=lambda reading: reading.consistency)
    if best.consistency < RETRACK_BELOW_CONSISTENCY:
        # Bars that do not agree with each other mean the tracker locked on
        # to a cross-rhythm — two-thirds or three-quarters of the real tempo
        # is the usual mistake, especially with little audio to settle on.
        # Ask again with the tracker's prior centred on each related tempo.
        for ratio in (1.5, 2.0 / 3.0, 4.0 / 3.0, 0.75):
            bpm, beats = _track(full, sr, hop, start_bpm=best.bpm * ratio, tightness=400)
            if bpm <= 0 or beats.size < MIN_BEATS:
                continue
            reading = _resolve_octave(_align(beats, combined), bpm, envelopes, combined, onset_rate)
            if reading is not None and reading.consistency > best.consistency:
                best = reading
    intervals = np.diff(best.beats)
    median_interval = float(np.median(intervals))
    profile.bpm = best.bpm
    profile.double_time = best.double_time
    profile.beat_regularity = float(
        1.0 / (1.0 + np.std(intervals) / max(median_interval, 1e-9) * 10.0)
    )
    # At whichever of the beat or the bar-half the onsets repeat most clearly:
    # a 2-step's full-band curve repeats every two beats, not every one.
    profile.pulse_clarity = max(
        _pulse_clarity(full, median_interval), _pulse_clarity(full, 2 * median_interval)
    )
    profile.pattern = best.pattern
    profile.bars = best.bars
    profile.swing = best.swing
    return profile


@dataclass
class _Reading:
    """The drums folded onto one candidate beat grid."""

    beats: np.ndarray
    bpm: float
    pattern: Dict[str, List[float]]
    bars: int
    swing: float
    # Mean correlation of each bar with the median bar. The right tempo makes
    # every bar look like the last; a wrong one smears them.
    consistency: float
    double_time: bool = False


def _resolve_octave(
    beats: np.ndarray,
    bpm: float,
    envelopes: Dict[str, np.ndarray],
    combined: np.ndarray,
    onset_rate: float,
) -> Optional["_Reading"]:
    """Read a tracked grid, re-counting it at double or half speed if needed."""
    tracked = _read(beats, bpm, envelopes)
    if bpm < DOUBLE_BELOW_BPM:
        slow, fast = tracked, _read(_align(_double(beats), combined), bpm * 2.0, envelopes)
    elif bpm >= HALVE_ABOVE_BPM:
        slow, fast = _read(_halve(beats, envelopes["kick"]), bpm / 2.0, envelopes), tracked
    else:
        return tracked if tracked.bars else None

    if not slow.bars or not fast.bars:
        return tracked if tracked.bars else None
    onsets_per_beat = onset_rate * 60.0 / slow.bpm
    if _reads_as_double_time(slow.pattern, onsets_per_beat):
        fast.double_time = fast is not tracked
        # Judged on the slow grid, where both readings cover the same bars.
        fast.consistency = max(fast.consistency, slow.consistency)
        return fast
    slow.consistency = max(fast.consistency, slow.consistency)
    return slow


def _double(beats: np.ndarray) -> np.ndarray:
    """The same beat grid at twice the tempo: a new beat halfway between each."""
    mids = (beats[:-1] + beats[1:]) / 2.0
    merged = np.empty(beats.size + mids.size)
    merged[0::2] = beats
    merged[1::2] = mids
    return merged


def _halve(beats: np.ndarray, kick: np.ndarray) -> np.ndarray:
    """Every other beat: the grid at half the tempo, on the kick's side."""
    limit = len(kick) - 1
    even, odd = beats[0::2], beats[1::2]
    weight = [float(kick[np.clip(np.round(b).astype(int), 0, limit)].sum()) for b in (even, odd)]
    return even if weight[0] >= weight[1] else odd


def _read(beats: np.ndarray, bpm: float, envelopes: Dict[str, np.ndarray]) -> _Reading:
    """Fold the bands onto a beat grid."""
    per_beat = {band: _grid(env, beats) for band, env in envelopes.items()}
    phase = _choose_downbeat(per_beat)
    folded: Dict[str, List[float]] = {}
    stacks = []
    bars = 0
    for band in BANDS:
        bar, stack = _fold(per_beat[band], phase)
        bars = len(stack)
        folded[band] = _normalise(bar).tolist()
        # Each band scaled by its own peak so a loud kick does not decide
        # alone whether the bars agree.
        peak = float(stack.max()) if stack.size else 0.0
        stacks.append(stack / peak if peak > 0 else stack)
    return _Reading(
        beats=beats,
        bpm=bpm,
        pattern=folded,
        bars=bars,
        swing=_swing(envelopes["hat"], beats),
        consistency=_consistency(np.concatenate(stacks, axis=1)) if bars else 0.0,
    )


def _consistency(stack: np.ndarray) -> float:
    """Mean correlation between each bar and the median bar."""
    if stack.shape[0] < 2:
        return 0.0
    reference = np.median(stack, axis=0)
    scores = []
    for row in stack:
        if np.std(row) < 1e-9 or np.std(reference) < 1e-9:
            continue
        scores.append(float(np.corrcoef(row, reference)[0, 1]))
    return float(np.mean(scores)) if scores else 0.0


def _busy_sixteenths(pattern: Dict[str, List[float]], threshold: float = 0.3) -> float:
    """Share of the bar's sixteenths where the snare or hat band is playing."""
    snare = sixteenths(pattern.get("snare", []))
    hat = sixteenths(pattern.get("hat", []))
    return float(np.mean(np.maximum(snare, hat) >= threshold))


def _reads_as_double_time(slow: Dict[str, List[float]], onsets_per_beat: float) -> bool:
    """Is a slow reading really a dance track counted at half its tempo?

    Beat trackers routinely hear 170 as 85, 132 as 66 — and, the other way
    round, an 80 BPM boom-bap beat as 160. Both readings describe the same
    audio, so the question is which one a DJ would count in. At half tempo a
    dance track's eighth-note hats and ghost notes fill nearly every
    sixteenth of the bar and land several onsets on every beat; a genuinely
    slow groove leaves most sixteenths empty.
    """
    return (
        _busy_sixteenths(slow) >= DOUBLE_TIME_BUSYNESS
        or onsets_per_beat >= DOUBLE_TIME_ONSETS_PER_BEAT
    )
