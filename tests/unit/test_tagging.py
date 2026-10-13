"""Tag rules, read from measurements rather than audio.

Every rule in :mod:`music_cluster.tagging` is a claim about what a groove or
a mood measures like. These tests state those claims with hand-built
measurements, so a rule that drifts shows up here without decoding a file.
"""

from typing import Dict, List, Optional

import numpy as np
import pytest

from music_cluster import rhythm, tagging


def bar(steps: List[int], strength: float = 1.0) -> List[float]:
    """A 48-step band pattern with hits on the given sixteenths."""
    pattern = np.zeros(rhythm.STEPS_PER_BAR)
    for step in steps:
        pattern[step * rhythm.SIXTEENTH] = strength
    return pattern.tolist()


def groove(
    kick: List[int],
    snare: List[int],
    hat: Optional[List[int]] = None,
    bpm: float = 124.0,
    swing: float = 0.0,
    clarity: float = 0.8,
) -> Dict:
    return {
        "bpm": bpm,
        "pulse_clarity": clarity,
        "beat_regularity": 0.9,
        "onset_rate": 5.0,
        "transient_strength": 0.08,
        "swing": swing,
        "band_activity": {"kick": 0.6, "snare": 0.2, "hat": 0.2},
        "pattern": {
            "kick": bar(kick),
            "snare": bar(snare),
            "hat": bar(hat if hat is not None else [2, 6, 10, 14]),
        },
        "bars": 16,
    }


def descriptors(rhythm_profile: Dict, **overrides) -> Dict:
    base = {
        "duration": 90.0,
        "loudness_db": -10.0,
        "loudness_range_db": 8.0,
        "brightness_hz": 2000.0,
        "flatness": 0.03,
        "low_share": 0.4,
        "high_share": 0.04,
        "harmonic_change": 0.05,
        "rhythm": rhythm_profile,
        "key": "A minor",
        "mode": "minor",
        "camelot": "8A",
        "key_strength": 0.5,
    }
    base.update(overrides)
    return base


def best_beat(profile: Dict) -> str:
    scores = tagging.beat_scores(profile)
    return max(scores, key=lambda key: scores[key])


# ----------------------------------------------------------------------
# Beat type
# ----------------------------------------------------------------------


GROOVES = {
    "four_on_floor": {"kick": [0, 4, 8, 12], "snare": [4, 12], "bpm": 124},
    "half_time": {"kick": [0, 10], "snare": [8], "bpm": 172},
    "two_step": {"kick": [0, 10], "snare": [4, 12], "bpm": 132},
    "breakbeat": {"kick": [0, 2, 10, 11], "snare": [4, 7, 9, 12, 15], "bpm": 170},
    "baile_funk": {"kick": [0, 3, 6, 8, 11, 14], "snare": [4, 12], "bpm": 130},
    "dembow": {"kick": [0, 4, 8, 12], "snare": [3, 6, 11, 14], "bpm": 95},
    "backbeat": {"kick": [0, 7, 8], "snare": [4, 12], "bpm": 88},
}


@pytest.mark.parametrize("expected", sorted(GROOVES))
def test_each_groove_reads_as_itself(expected):
    assert best_beat(groove(**GROOVES[expected])) == expected


@pytest.mark.parametrize("expected", sorted(GROOVES))
@pytest.mark.parametrize("turn", [1, 2, 3])
def test_beat_type_survives_a_wrong_downbeat(expected, turn):
    """Where beat one falls is the least certain measurement; rules must cope."""
    spec = GROOVES[expected]
    shift = 4 * turn
    rotated = dict(
        spec,
        kick=[(step + shift) % 16 for step in spec["kick"]],
        snare=[(step + shift) % 16 for step in spec["snare"]],
    )
    assert best_beat(groove(**rotated)) == expected


def test_a_two_step_is_not_a_four_to_the_floor_at_the_same_tempo():
    """The whole reason beat type needs a pattern rather than a tempo."""
    four = tagging.beat_scores(groove(kick=[0, 4, 8, 12], snare=[4, 12], bpm=132))
    two = tagging.beat_scores(groove(kick=[0, 10], snare=[4, 12], bpm=132))
    assert four["four_on_floor"] > four["two_step"]
    assert two["two_step"] > two["four_on_floor"]


def test_half_time_needs_a_dance_tempo():
    """One snare per bar at 172 is half-time; at 60 it is just slow."""
    fast = tagging.beat_scores(groove(kick=[0, 10], snare=[8], bpm=172))
    slow = tagging.beat_scores(groove(kick=[0, 10], snare=[8], bpm=60))
    assert fast["half_time"] > slow["half_time"]


def test_no_drums_is_beatless_whatever_the_pattern_says():
    """A pad's swell folds into a tidy bar; transients are what it lacks."""
    profile = groove(kick=[0, 4, 8, 12], snare=[4, 12])
    profile["transient_strength"] = 0.001
    assert best_beat(profile) == "beatless"


def test_no_pattern_is_beatless():
    assert tagging.beat_scores({}) == {"beatless": pytest.approx(1.0)}


# ----------------------------------------------------------------------
# Genre
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "spec, expected",
    [
        ({"kick": [0, 4, 8, 12], "snare": [4, 12], "bpm": 122}, "house"),
        ({"kick": [0, 4, 8, 12], "snare": [4, 12], "bpm": 134}, "techno"),
        ({"kick": [0, 10], "snare": [4, 12], "bpm": 132, "swing": 0.6}, "uk_garage"),
        ({"kick": [0, 10], "snare": [4, 12], "bpm": 174}, "drum_and_bass"),
        ({"kick": [0, 10], "snare": [8], "bpm": 174}, "drum_and_bass"),
        ({"kick": [0, 10], "snare": [8], "bpm": 140}, "dubstep"),
        ({"kick": [0, 3, 6, 8, 11, 14], "snare": [4, 12], "bpm": 130}, "baile_funk"),
        ({"kick": [0, 4, 8, 12], "snare": [3, 6, 11, 14], "bpm": 95}, "reggaeton"),
        ({"kick": [0, 7, 8], "snare": [4, 12], "bpm": 88}, "hip_hop"),
    ],
)
def test_tempo_and_beat_pick_the_genre(spec, expected):
    assert tagging.tag(descriptors(groove(**spec))).genre == expected


def test_same_beat_different_tempo_different_genre():
    """A 2-step at 132 is garage; at 174 it is drum & bass."""
    garage = tagging.tag(descriptors(groove(kick=[0, 10], snare=[4, 12], bpm=132, swing=0.5)))
    dnb = tagging.tag(descriptors(groove(kick=[0, 10], snare=[4, 12], bpm=174)))
    assert garage.beat == dnb.beat == "two_step"
    assert (garage.genre, dnb.genre) == ("uk_garage", "drum_and_bass")


def test_a_four_to_the_floor_at_95_is_not_house():
    """Tempo is a gate, not one vote among several."""
    tags = tagging.tag(descriptors(groove(kick=[0, 4, 8, 12], snare=[4, 12], bpm=95)))
    assert tags.genre not in {"house", "deep_house", "tech_house", "techno"}


def test_the_files_genre_tag_tips_a_close_call():
    profile = descriptors(groove(kick=[0, 4, 8, 12], snare=[4, 12], bpm=127))
    assert tagging.tag(profile, {"genre": "Tech House"}).genre == "tech_house"


@pytest.mark.parametrize(
    "text, expected",
    [
        ("Deep House", "deep_house"),
        ("Tech House", "tech_house"),
        ("Drum & Bass / Liquid", "drum_and_bass"),
        ("UKG", "uk_garage"),
        ("Funk Carioca", "baile_funk"),
        ("Polka", None),
        (None, None),
    ],
)
def test_genre_tags_map_onto_the_vocabulary(text, expected):
    assert tagging.genre_from_tag(text) == expected


# ----------------------------------------------------------------------
# Energy, danceability, mood
# ----------------------------------------------------------------------


def test_louder_denser_brighter_is_more_energetic():
    calm = descriptors(
        dict(groove(kick=[0, 8], snare=[4, 12], bpm=90), onset_rate=1.5),
        loudness_db=-24.0,
        brightness_hz=900.0,
        high_share=0.005,
    )
    hard = descriptors(
        dict(groove(kick=[0, 4, 8, 12], snare=[4, 12], bpm=145), onset_rate=7.0),
        loudness_db=-7.0,
        brightness_hz=3500.0,
        high_share=0.1,
    )
    assert tagging.tag(hard).energy_level > tagging.tag(calm).energy_level
    assert tagging.tag(hard).energy_level >= 8
    assert tagging.tag(calm).energy_level <= 3


def test_energy_level_spans_one_to_ten():
    assert tagging.TrackTags(energy=0.0).energy_level == 1
    assert tagging.TrackTags(energy=1.0).energy_level == 10


def test_a_steady_kick_at_124_is_danceable_and_a_pad_is_not():
    dance = tagging.tag(descriptors(groove(kick=[0, 4, 8, 12], snare=[4, 12], bpm=124)))
    pad_profile = groove(kick=[0, 4, 8, 12], snare=[4, 12], bpm=124)
    pad_profile["transient_strength"] = 0.001
    pad = tagging.tag(descriptors(pad_profile))
    assert dance.danceability > 0.8
    assert pad.danceability == 0.0


def test_moods_follow_mode_and_brightness():
    dark = tagging.tag(
        descriptors(
            groove(kick=[0, 4, 8, 12], snare=[4, 12], bpm=130),
            mode="minor",
            brightness_hz=1000.0,
            low_share=0.6,
        )
    )
    bright = tagging.tag(
        descriptors(
            groove(kick=[0, 4, 8, 12], snare=[4, 12], bpm=138),
            mode="major",
            key="C major",
            brightness_hz=3500.0,
            low_share=0.2,
        )
    )
    assert "dark" in dark.moods
    assert "euphoric" in bright.moods
    assert "euphoric" not in dark.moods


def test_a_straight_beat_is_not_groovy_and_a_swung_one_is():
    straight = tagging.tag(descriptors(groove(kick=[0, 4, 8, 12], snare=[4, 12], bpm=124)))
    swung = tagging.tag(descriptors(groove(kick=[0, 4, 8, 12], snare=[4, 12], bpm=124, swing=0.7)))
    assert "groovy" not in straight.moods
    assert "groovy" in swung.moods
    assert swung.swung and not straight.swung


def test_a_key_is_only_reported_when_there_is_one():
    keyed = tagging.tag(descriptors(groove(kick=[0, 4, 8, 12], snare=[4, 12])))
    keyless = tagging.tag(descriptors(groove(kick=[0, 4, 8, 12], snare=[4, 12]), key_strength=0.02))
    assert keyed.camelot == "8A"
    assert keyless.key is None and keyless.camelot is None


# ----------------------------------------------------------------------
# Key finding
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "notes, key, camelot",
    [
        ([9, 0, 4], "A minor", "8A"),
        ([0, 4, 7], "C major", "8B"),
        ([7, 11, 2], "G major", "9B"),
        ([2, 5, 9], "D minor", "7A"),
    ],
)
def test_triads_find_their_key(notes, key, camelot):
    chroma = np.full(12, 0.1)
    chroma[notes] = 1.0
    chroma[notes[0]] = 1.5  # the tonic sounds most
    found = tagging.estimate_key(chroma)
    assert (found["key"], found["camelot"]) == (key, camelot)


def test_flat_chroma_has_no_key():
    assert tagging.estimate_key(np.ones(12))["key"] is None


# ----------------------------------------------------------------------
# Output
# ----------------------------------------------------------------------


def test_tags_serialise_with_labels():
    tags = tagging.tag(descriptors(groove(kick=[0, 10], snare=[8], bpm=172)))
    payload = tags.to_dict()
    assert payload["beat"] == "half_time"
    assert payload["beat_label"] == "Half-time"
    assert payload["genre_label"] == "Drum & Bass"
    assert 1 <= payload["energy_level"] <= 10
    assert "Half-time" in payload["labels"]
    assert "descriptors" in payload
    assert "descriptors" not in tags.to_dict(include_descriptors=False)


def test_empty_measurements_give_empty_tags():
    tags = tagging.tag({})
    assert tags.genre is None and tags.beat is None and tags.moods == []


def test_vocabulary_lists_every_kind():
    vocabulary = tagging.vocabulary()
    assert {"beats", "genres", "moods"} <= set(vocabulary)
    beat_keys = {entry["key"] for entry in vocabulary["beats"]}
    assert {"four_on_floor", "half_time", "two_step", "breakbeat", "baile_funk"} <= beat_keys


def test_rhythm_profiles_round_trip():
    profile = rhythm.RhythmProfile.from_dict(groove(kick=[0, 10], snare=[8], bpm=172))
    again = rhythm.RhythmProfile.from_dict(profile.to_dict())
    assert again.bpm == pytest.approx(172.0)
    assert again.has_pattern
    assert rhythm.sixteenths(again.pattern["snare"])[8] == pytest.approx(1.0)


def test_opposed_moods_are_never_both_reported():
    """Measurements can score both; the tags should not say both."""
    tags = tagging.tag(
        descriptors(
            groove(kick=[0, 10], snare=[8], bpm=172),
            mode="minor",
            brightness_hz=3500.0,
            low_share=0.6,
            flatness=0.02,
        )
    )
    for pair in tagging.OPPOSED_MOODS:
        assert not pair <= set(tags.moods)
