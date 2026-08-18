"""The sample classifier: filename evidence, audio evidence, and their sum.

The audio rules are checked against descriptor dictionaries rather than real
files, because what is being tested is the reasoning — "low-heavy, short and
unpitched is a kick" — not the measurement, which the extractor tests cover.
"""

import pytest

from music_cluster import samples


def descriptors(**overrides):
    """A neutral one-shot, to be pushed towards one category at a time."""
    base = {
        "duration": 0.4,
        "attack_time": 0.005,
        "attack_slope": 5.0,
        "decay_time": 0.1,
        "sustain_ratio": 0.3,
        "release_time": 0.3,
        "temporal_centroid": 0.25,
        "effective_duration": 0.3,
        "crest_factor": 5.0,
        "onset_count": 1.0,
        "percussive_ratio": 0.5,
        "f0_hz": 0.0,
        "f0_confidence": 0.0,
        "pitch_stability": 0.0,
        "harmonic_support": 0.0,
        "chroma_peaks": 0.0,
        "brightness_attack": 2000.0,
        "brightness_decay": 2000.0,
        "brightness_slope": 0.0,
        "flux_mean": 0.1,
        "flux_std": 0.05,
        "low_energy": 0.33,
        "mid_energy": 0.34,
        "high_energy": 0.33,
        "bands": dict.fromkeys(
            ("sub", "low", "low_mid", "mid", "upper_mid", "presence", "brilliance", "air"),
            0.125,
        ),
    }
    base.update(overrides)
    return base


KICK = descriptors(
    duration=0.35,
    low_energy=0.75,
    mid_energy=0.24,
    high_energy=0.01,
    f0_hz=55.0,
    f0_confidence=0.6,
    harmonic_support=0.8,
    bands={**descriptors()["bands"], "sub": 0.4, "low": 0.35},
)

HAT = descriptors(
    duration=0.08,
    low_energy=0.0,
    mid_energy=0.03,
    high_energy=0.97,
    brightness_attack=9000.0,
    percussive_ratio=0.95,
)

SNARE = descriptors(
    duration=0.3,
    low_energy=0.05,
    mid_energy=0.35,
    high_energy=0.6,
    brightness_attack=3000.0,
    percussive_ratio=0.8,
)

BASS = descriptors(
    duration=1.2,
    sustain_ratio=0.65,
    low_energy=0.8,
    mid_energy=0.19,
    high_energy=0.01,
    f0_hz=60.0,
    f0_confidence=0.95,
    harmonic_support=0.9,
    chroma_peaks=1.0,
    percussive_ratio=0.05,
)

CHORD = descriptors(
    duration=1.5,
    sustain_ratio=0.6,
    low_energy=0.05,
    mid_energy=0.85,
    high_energy=0.10,
    f0_hz=220.0,
    f0_confidence=0.9,
    harmonic_support=0.6,
    chroma_peaks=3.0,
    percussive_ratio=0.05,
    temporal_centroid=0.42,
)

PAD = descriptors(
    duration=3.0,
    attack_time=0.5,
    sustain_ratio=0.9,
    low_energy=0.1,
    mid_energy=0.8,
    high_energy=0.1,
    f0_hz=200.0,
    f0_confidence=0.7,
    harmonic_support=0.5,
    chroma_peaks=3.0,
    percussive_ratio=0.02,
    temporal_centroid=0.5,
)

DRUM_LOOP = descriptors(
    duration=2.0,
    onset_count=8.0,
    percussive_ratio=0.8,
    low_energy=0.4,
    mid_energy=0.3,
    high_energy=0.3,
)


@pytest.mark.parametrize(
    "expected, measured",
    [
        ("kick", KICK),
        ("hat_closed", HAT),
        ("snare", SNARE),
        ("bass", BASS),
        ("chord", CHORD),
        ("pad", PAD),
        ("drum_loop", DRUM_LOOP),
    ],
)
def test_audio_alone_identifies_the_obvious_cases(expected, measured):
    guess = samples.classify(measured)
    assert guess.key == expected, f"scored {sorted(guess.scores.items(), key=lambda i: -i[1])[:3]}"


def test_filenames_are_read_the_way_sample_packs_are_named():
    for path, expected in [
        ("/packs/BD_808_01.wav", "kick"),
        ("/packs/Kick 04.wav", "kick"),
        ("/packs/SD_acoustic.wav", "snare"),
        ("/packs/CP_room.wav", "clap"),
        ("/packs/closedhat_02.wav", "hat_closed"),
        ("/packs/OpenHat-3.wav", "hat_open"),
        ("/packs/sub_bass_C.wav", "sub"),
        ("/packs/Chord_Am7.wav", "chord"),
        ("/packs/vox_adlib.wav", "vocal"),
        ("/packs/riser_long.wav", "riser"),
    ]:
        scores = samples.name_scores(path)
        assert scores, f"no filename evidence in {path}"
        assert max(scores, key=lambda key: scores[key]) == expected, path


def test_the_enclosing_folder_is_weaker_evidence_than_the_filename():
    """A pack laid out well should not be overruled by one laid out badly."""
    both = samples.name_scores("/packs/Snares/clap_01.wav")
    assert both["clap"] > both["snare"]

    folder_only = samples.name_scores("/packs/Kicks/01.wav")
    assert folder_only.get("kick", 0) > 0


def test_a_specific_name_beats_the_general_one_it_contains():
    """`open_hat` matches both hat categories; the specific one must win."""
    scores = samples.name_scores("/packs/open_hat_01.wav")
    assert scores["hat_open"] > scores["hat_closed"]

    scores = samples.name_scores("/packs/sub_bass_01.wav")
    assert scores["sub"] > scores["bass"]


def test_audio_corrects_nothing_when_the_filename_is_uninformative():
    """A meaningless name must not drag the audio's verdict down below the floor."""
    named = samples.classify(KICK, "/packs/Sample 04.wav")
    unnamed = samples.classify(KICK)
    assert named.key == unnamed.key == "kick"
    assert named.confidence == pytest.approx(unnamed.confidence)


def test_agreeing_evidence_is_more_confident_than_either_alone():
    audio_only = samples.classify(KICK)
    both = samples.classify(KICK, "/packs/kick_01.wav")
    assert both.confidence > audio_only.confidence
    assert "named like one" in both.evidence


def test_a_close_second_place_lowers_confidence():
    """Two categories fitting equally well is not a confident answer."""
    ambiguous = samples.classify(descriptors())
    assert ambiguous.confidence < 0.6


def test_material_that_fits_nothing_is_reported_as_unknown():
    silence = descriptors(
        duration=0.0,
        low_energy=0.0,
        mid_energy=0.0,
        high_energy=0.0,
        brightness_attack=0.0,
        percussive_ratio=0.0,
        onset_count=0.0,
    )
    guess = samples.classify(silence)
    assert not guess.known
    assert guess.label == samples.UNKNOWN_LABEL


def test_no_evidence_at_all_is_unknown():
    assert not samples.classify().known


def test_a_group_is_named_for_what_most_of_it_is():
    guesses = [samples.classify(KICK) for _ in range(7)] + [samples.classify(HAT)]
    summary = samples.summarize(guesses)
    assert summary["category"] == "kick"
    assert summary["plural"] == "Kicks"
    assert summary["share"] == pytest.approx(7 / 8)


def test_a_group_of_everything_keeps_its_fallback_name():
    """A pile that is a quarter of each thing should not claim to be any of them."""
    tracks = [{"filepath": f"/packs/{name}.wav"} for name in ("kick", "snare", "hat", "bass")]
    assert samples.suggest_group_name(None, tracks, fallback="Candidate 3") == "Candidate 3"


def test_a_consistent_group_is_named_from_its_members():
    tracks = [{"filepath": f"/packs/kick_{i:02d}.wav"} for i in range(6)]
    assert samples.suggest_group_name(None, tracks, fallback="Candidate 1") == "Kicks"


def test_tokenizer_splits_the_ways_packs_are_written():
    assert samples.tokenize("OpenHat") == ["open", "hat"]
    assert samples.tokenize("open_hat-01") == ["open", "hat", "01"]
    assert samples.tokenize("BD 808") == ["bd", "808"]


def test_taxonomy_is_complete_and_unique():
    keys = [entry["key"] for entry in samples.taxonomy()]
    assert len(keys) == len(set(keys))
    for expected in ("kick", "snare", "clap", "hat_closed", "bass", "chord"):
        assert expected in keys
