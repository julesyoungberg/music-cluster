"""A tiny drum machine for tests: known grooves, rendered to audio.

Beat-type tagging is only worth anything if a four-to-the-floor reads as one
and a 2-step does not, so the tests need grooves whose ground truth is known
by construction. These are drawn on a sixteenth-note grid exactly as they
would be in a sequencer, then rendered with synthesised kit pieces.

Importable without pytest so the patterns can also be auditioned by hand.
"""

from typing import Dict, List, Optional

import numpy as np


SAMPLE_RATE = 22050

# Sixteenth-note steps, 0-15, for one bar of each groove. "x" rows are hits.
GROOVES: Dict[str, Dict[str, List[int]]] = {
    "four_on_floor": {
        "kick": [0, 4, 8, 12],
        "snare": [4, 12],
        "hat": [2, 6, 10, 14],
    },
    "half_time": {
        "kick": [0, 10],
        "snare": [8],
        "hat": [0, 2, 4, 6, 8, 10, 12, 14],
    },
    "two_step": {
        "kick": [0, 10],
        "snare": [4, 12],
        "hat": [2, 6, 7, 10, 14, 15],
    },
    # After the Amen: kicks that dodge the grid, a busy snare full of ghosts.
    "breakbeat": {
        "kick": [0, 2, 10, 11],
        "snare": [4, 7, 9, 12, 15],
        "hat": [0, 2, 4, 6, 8, 10, 12, 14],
    },
    # Tamborzão: the low end plays 3+3+2 twice, not four on the floor.
    "baile_funk": {
        "kick": [0, 3, 6, 8, 11, 14],
        "snare": [4, 12],
        "hat": [0, 2, 4, 6, 8, 10, 12, 14],
    },
    # Dembow: kick on every quarter, snare answering 3+3+2 around it.
    "dembow": {
        "kick": [0, 4, 8, 12],
        "snare": [3, 6, 11, 14],
        "hat": [0, 2, 4, 6, 8, 10, 12, 14],
    },
    "backbeat": {
        "kick": [0, 7, 8],
        "snare": [4, 12],
        "hat": [0, 2, 4, 6, 8, 10, 12, 14],
    },
}


def _kick(sr: int) -> np.ndarray:
    t = np.arange(int(0.25 * sr)) / sr
    freq = 45.0 + 110.0 * np.exp(-t * 30.0)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    return np.sin(phase) * np.exp(-t * 14.0)


def _snare(sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(0.18 * sr)) / sr
    noise = rng.normal(0, 1, t.size)
    # A crude high-pass: the difference of noise is brighter than noise.
    bright = np.diff(noise, prepend=0.0)
    body = 0.4 * np.sin(2 * np.pi * 190.0 * t)
    return (0.5 * bright + body) * np.exp(-t * 22.0)


def _hat(sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(int(0.05 * sr)) / sr
    noise = rng.normal(0, 1, t.size)
    very_bright = np.diff(np.diff(noise, prepend=0.0), prepend=0.0)
    return 0.25 * very_bright * np.exp(-t * 80.0)


def render(
    groove: str,
    bpm: float,
    seconds: float = 24.0,
    sr: int = SAMPLE_RATE,
    swing: float = 0.0,
    seed: int = 7,
    pad: Optional[float] = None,
) -> np.ndarray:
    """Render a named groove as mono float audio.

    Args:
        swing: 0 for straight sixteenths; 1 pushes every offbeat sixteenth all
            the way to the triplet position.
        pad: Frequency of a quiet sustained chord under the drums, so the
            harmonic descriptors have something to read. None for drums only.
    """
    rng = np.random.default_rng(seed)
    pattern = GROOVES[groove]
    sixteenth = 60.0 / bpm / 4.0
    bar = sixteenth * 16
    out = np.zeros(int(seconds * sr) + sr)
    kit = {"kick": _kick(sr), "snare": _snare(sr, rng), "hat": _hat(sr, rng)}
    gain = {"kick": 0.9, "snare": 0.6, "hat": 0.5}

    start = 0.0
    while start < seconds:
        for piece, steps in pattern.items():
            for step in steps:
                offset = step * sixteenth
                if step % 2 == 1:
                    offset += swing * sixteenth / 3.0
                index = int((start + offset) * sr)
                sound = kit[piece] * gain[piece]
                end = min(index + sound.size, out.size)
                if index < out.size:
                    out[index:end] += sound[: end - index]
        start += bar

    out = out[: int(seconds * sr)]
    if pad is not None:
        t = np.arange(out.size) / sr
        chord = sum(np.sin(2 * np.pi * pad * ratio * t) for ratio in (1.0, 1.2, 1.5))
        out = out + 0.08 * chord
    peak = float(np.max(np.abs(out))) or 1.0
    return (out / peak * 0.9).astype(np.float32)
