"""Auto-tagging from real audio: rendered grooves whose ground truth is known.

The drum machine in ``tests/drums.py`` draws each groove on a sixteenth grid
exactly as a producer would, so "is this a 2-step?" has a right answer. These
tests run the whole measurement — onset bands, beat tracking, tempo octave,
folding into bars — and assert the tag a DJ would give.
"""

import numpy as np
import pytest

from music_cluster import rhythm, tagging
from music_cluster.library import analyze_paths, tag_tracks
from tests.drums import SAMPLE_RATE, render


soundfile = pytest.importorskip("soundfile")

pytestmark = pytest.mark.slow


@pytest.mark.parametrize(
    "groove, bpm, swing, beat, genre",
    [
        ("four_on_floor", 124, 0.0, "four_on_floor", "house"),
        ("half_time", 140, 0.0, "half_time", "dubstep"),
        # Tracked at 86 by the beat tracker; the drums say it is 172.
        ("half_time", 172, 0.0, "half_time", "drum_and_bass"),
        ("two_step", 132, 1.0, "two_step", "uk_garage"),
        ("breakbeat", 170, 0.0, "breakbeat", "jungle"),
        ("baile_funk", 130, 0.0, "baile_funk", "baile_funk"),
        ("dembow", 95, 0.0, "dembow", "reggaeton"),
        ("backbeat", 90, 0.0, "backbeat", "hip_hop"),
    ],
)
def test_rendered_grooves_are_tagged_as_drawn(groove, bpm, swing, beat, genre):
    tags = tagging.tag_audio(render(groove, bpm, swing=swing), SAMPLE_RATE)
    assert tags is not None
    assert tags.beat == beat
    assert tags.genre == genre
    assert tags.bpm == pytest.approx(bpm, rel=0.03)


def test_half_speed_tempo_readings_are_doubled():
    measured = rhythm.analyze(render("two_step", 132), SAMPLE_RATE)
    assert measured.bpm == pytest.approx(132, rel=0.03)
    assert measured.double_time


def test_a_slow_backbeat_is_not_doubled():
    measured = rhythm.analyze(render("backbeat", 80), SAMPLE_RATE)
    assert measured.bpm == pytest.approx(80, rel=0.03)


def test_swing_is_measured():
    straight = rhythm.analyze(render("two_step", 136, swing=0.0), SAMPLE_RATE)
    swung = rhythm.analyze(render("two_step", 136, swing=1.0), SAMPLE_RATE)
    assert straight.swing < 0.2
    assert swung.swing > 0.5


def test_a_pad_is_beatless():
    t = np.arange(SAMPLE_RATE * 20) / SAMPLE_RATE
    pad = sum(np.sin(2 * np.pi * 220 * ratio * t) for ratio in (1.0, 1.2, 1.5))
    swell = pad * 0.2 * (0.8 + 0.2 * np.sin(2 * np.pi * 0.1 * t))
    tags = tagging.tag_audio(swell, SAMPLE_RATE)
    assert tags.beat == "beatless"
    assert tags.danceability == 0.0
    assert tags.key == "A minor"


def test_silence_is_not_tagged():
    assert tagging.tag_audio(np.zeros(SAMPLE_RATE * 5), SAMPLE_RATE) is None


def _write(path, groove, bpm, **kwargs):
    soundfile.write(str(path), render(groove, bpm, seconds=16.0, **kwargs), SAMPLE_RATE)
    return str(path)


def test_analysis_tags_music_as_it_goes(db, config, temp_dir):
    config.set(["feature_extraction", "sample_rate"], SAMPLE_RATE)
    folder = temp_dir / "new"
    folder.mkdir()
    _write(folder / "roller.wav", "half_time", 172)
    _write(folder / "garage.wav", "two_step", 132, swing=1.0)

    result = analyze_paths(db, config, [str(folder)], workers=1)
    assert result.analyzed == 2 and result.tagged == 2

    by_name = {db.get_track(tid)["filename"]: db.get_tags(tid) for tid in result.track_ids}
    assert by_name["roller.wav"]["beat"] == "half_time"
    assert by_name["garage.wav"]["beat"] == "two_step"
    assert db.tracks_needing_tags(None, tagging.TAGGER_VERSION) == []


def test_samples_are_not_tagged(db, config, temp_dir):
    config.set(["feature_extraction", "sample_rate"], SAMPLE_RATE)
    path = _write(temp_dir / "loop.wav", "four_on_floor", 124)
    result = analyze_paths(db, config, [path], workers=1, profile="sample")
    assert result.tagged == 0
    assert db.get_tags(result.track_ids[0]) is None


def test_tracks_analysed_before_tagging_can_be_tagged_later(db, config, temp_dir):
    config.set(["feature_extraction", "sample_rate"], SAMPLE_RATE)
    config.set(["tagging", "on_analyze"], False)
    path = _write(temp_dir / "house.wav", "four_on_floor", 124)
    analysed = analyze_paths(db, config, [path], workers=1)
    assert analysed.tagged == 0

    result = tag_tracks(db, config, workers=1)
    assert result.tagged == 1
    assert db.get_tags(analysed.track_ids[0])["beat"] == "four_on_floor"

    # Already current: nothing to do unless asked to redo it.
    assert tag_tracks(db, config, workers=1).tagged == 0
    assert tag_tracks(db, config, workers=1, update=True).tagged == 1
