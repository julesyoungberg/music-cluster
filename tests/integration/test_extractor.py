"""The extractor must produce exactly the vector the layout describes."""

import numpy as np
import pytest

from music_cluster.extractor import FeatureExtractor
from music_cluster.features import build_layout, describe_vector
from music_cluster.metadata import read_metadata

soundfile = pytest.importorskip("soundfile")


SAMPLE_RATE = 22050


def write_tone(path, seconds=6.0, freq=220.0, sample_rate=SAMPLE_RATE, pulse_hz=2.0):
    """A pulsed tone: enough structure for tempo and spectral features to exist."""
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    envelope = 0.5 * (1 + np.sign(np.sin(2 * np.pi * pulse_hz * t)))
    signal = envelope * np.sin(2 * np.pi * freq * t)
    soundfile.write(str(path), signal.astype(np.float32), sample_rate)
    return str(path)


def test_extracted_vector_matches_the_layout(temp_dir):
    path = write_tone(temp_dir / "tone.wav")
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE, n_mfcc=20, excerpt_seconds=0)

    vector = extractor.extract(path)
    assert vector is not None
    assert len(vector) == build_layout(20).dim
    assert np.all(np.isfinite(vector))


@pytest.mark.parametrize("n_mfcc", [13, 20])
def test_layout_follows_the_mfcc_setting(temp_dir, n_mfcc):
    path = write_tone(temp_dir / f"tone{n_mfcc}.wav")
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE, n_mfcc=n_mfcc, excerpt_seconds=0)
    vector = extractor.extract(path)
    assert len(vector) == build_layout(n_mfcc).dim


def test_brighter_audio_reads_as_brighter(temp_dir):
    """Descriptors must track the physical property they name."""
    extractor = FeatureExtractor(sample_rate=SAMPLE_RATE, excerpt_seconds=0)

    low = describe_vector(extractor.extract(write_tone(temp_dir / "low.wav", freq=150.0)))
    high = describe_vector(extractor.extract(write_tone(temp_dir / "high.wav", freq=4000.0)))

    assert high["brightness_hz"] > low["brightness_hz"]


def test_excerpting_analyses_a_shorter_span(temp_dir):
    """Excerpting should change what is measured on a long, varying track."""
    path = temp_dir / "long.wav"
    t = np.linspace(0, 40.0, int(SAMPLE_RATE * 40), endpoint=False)
    # Quiet first half, loud second half: an excerpt from the middle differs
    # from the whole file.
    signal = np.sin(2 * np.pi * 220 * t) * np.where(t < 20, 0.05, 0.9)
    soundfile.write(str(path), signal.astype(np.float32), SAMPLE_RATE)

    whole = FeatureExtractor(sample_rate=SAMPLE_RATE, excerpt_seconds=0).extract(str(path))
    excerpt = FeatureExtractor(sample_rate=SAMPLE_RATE, excerpt_seconds=8).extract(str(path))

    assert whole is not None and excerpt is not None
    assert not np.allclose(whole, excerpt)


def test_undecodable_file_returns_none(temp_dir):
    broken = temp_dir / "broken.wav"
    broken.write_bytes(b"not audio at all")
    assert FeatureExtractor().extract(str(broken)) is None


def test_metadata_falls_back_to_the_filename(temp_dir):
    path = temp_dir / "Aphex Twin - Windowlicker.wav"
    write_tone(path, seconds=1.0)

    metadata = read_metadata(str(path))
    assert metadata["artist"] == "Aphex Twin"
    assert metadata["title"] == "Windowlicker"
