"""The sample profile, from real audio through to a fitted, sorting collection.

Synthetic one-shots stand in for a drum library: a swept sine for a kick,
filtered noise for a hat, a harmonic stack for a bass. They are crude, but each
one is unambiguous about the property being asserted — a kick is low, a hat is
short and bright, a bass has a note — which is exactly what the descriptors
claim to measure.
"""

import numpy as np
import pytest

from music_cluster import groups as groups_mod
from music_cluster import samples, sorting
from music_cluster.extractor import FeatureExtractor
from music_cluster.features import build_layout, describe_sample_vector
from music_cluster.library import analyze_paths


soundfile = pytest.importorskip("soundfile")


SAMPLE_RATE = 22050
LAYOUT = build_layout(20, "sample")


def _write(path, signal, sample_rate=SAMPLE_RATE):
    peak = float(np.max(np.abs(signal))) or 1.0
    soundfile.write(str(path), (signal / peak * 0.9).astype(np.float32), sample_rate)
    return str(path)


def _time(samples_count, sample_rate=SAMPLE_RATE):
    return np.arange(samples_count) / sample_rate


def kick(path, seconds=0.4, rng=None):
    """A pitch-swept sine with a click: low, short, decaying."""
    t = _time(int(seconds * SAMPLE_RATE))
    sweep = np.sin(2 * np.pi * np.cumsum(45 + 120 * np.exp(-t * 30)) / SAMPLE_RATE)
    click = (rng or np.random.default_rng(0)).normal(0, 0.05, len(t)) * np.exp(-t * 400)
    return _write(path, sweep * np.exp(-t * 12) + click)


def hat(path, seconds=0.07, rng=None):
    """A very short burst of bright noise."""
    t = _time(int(seconds * SAMPLE_RATE))
    noise = (rng or np.random.default_rng(1)).normal(0, 1, len(t)) * np.exp(-t * 70)
    return _write(path, np.diff(np.concatenate([[0.0], noise])))


def snare(path, seconds=0.3, rng=None):
    """Noise over a tuned body: mid and high, unpitched overall."""
    t = _time(int(seconds * SAMPLE_RATE))
    noise = (rng or np.random.default_rng(2)).normal(0, 1, len(t)) * 0.75
    body = np.sin(2 * np.pi * 190 * t) * 0.5
    return _write(path, (noise + body) * np.exp(-t * 20))


def bass(path, seconds=1.1, rng=None):
    """A sustained harmonic stack on a low fundamental."""
    t = _time(int(seconds * SAMPLE_RATE))
    tone = sum(np.sin(2 * np.pi * 55 * k * t) / k for k in (1, 2, 3, 4))
    return _write(path, tone * np.minimum(1.0, np.exp(-t * 0.9)))


def chord(path, seconds=1.6, rng=None):
    """Three notes at once, struck and decaying."""
    t = _time(int(seconds * SAMPLE_RATE))
    tone = sum(np.sin(2 * np.pi * f * t) for f in (196.0, 246.9, 293.7)) / 3
    return _write(path, tone * np.exp(-t * 0.6))


def pad(path, seconds=2.6, rng=None):
    """The same notes, swelling in slowly."""
    t = _time(int(seconds * SAMPLE_RATE))
    tone = sum(np.sin(2 * np.pi * f * t + f) for f in (174.6, 220.0, 261.6)) / 3
    return _write(path, tone * np.minimum(1.0, t * 1.2))


BUILDERS = {
    "kick": kick,
    "hat": hat,
    "snare": snare,
    "bass": bass,
    "chord": chord,
    "pad": pad,
}


@pytest.fixture
def extractor():
    return FeatureExtractor(
        sample_rate=SAMPLE_RATE,
        frame_size=2048,
        hop_size=256,
        excerpt_seconds=0,
        profile="sample",
        max_seconds=15,
    )


@pytest.fixture
def described(temp_dir, extractor):
    """Descriptors for one of each kind of one-shot."""
    out = {}
    for name, build in BUILDERS.items():
        vector = extractor.extract(build(temp_dir / f"{name}.wav"))
        assert vector is not None, name
        out[name] = describe_sample_vector(vector, LAYOUT)
    return out


def test_the_sample_vector_matches_the_sample_layout(temp_dir, extractor):
    vector = extractor.extract(kick(temp_dir / "kick.wav"))
    assert len(vector) == LAYOUT.dim
    assert np.all(np.isfinite(vector))


def test_a_music_extractor_on_the_same_file_produces_the_music_vector(temp_dir):
    path = kick(temp_dir / "kick.wav")
    music = FeatureExtractor(sample_rate=SAMPLE_RATE, excerpt_seconds=0).extract(path)
    assert len(music) == build_layout(20, "music").dim


def test_duration_survives_the_round_trip(described):
    assert described["hat"]["duration"] == pytest.approx(0.07, abs=0.02)
    assert described["kick"]["duration"] == pytest.approx(0.4, abs=0.05)
    assert described["pad"]["duration"] == pytest.approx(2.6, abs=0.1)


def test_energy_lands_in_the_band_the_sound_occupies(described):
    assert described["kick"]["low_energy"] > 0.5
    assert described["hat"]["high_energy"] > 0.8
    assert described["bass"]["low_energy"] > 0.5
    assert described["chord"]["mid_energy"] > 0.5


def test_pitched_material_is_told_from_noise(described):
    for pitched in ("bass", "chord"):
        assert described[pitched]["harmonic_support"] > 0.25, pitched
    # A snare tracks a steady pitch from its body tone but has no harmonics on
    # it, which is precisely the case harmonic_support exists to catch.
    assert described["snare"]["harmonic_support"] < 0.2


def test_a_chord_reports_more_pitch_classes_than_a_bass_note(described):
    assert described["chord"]["chroma_peaks"] > described["bass"]["chroma_peaks"]


def test_a_swell_has_a_slower_attack_than_a_hit(described):
    assert described["pad"]["attack_time"] > described["chord"]["attack_time"]
    assert described["pad"]["sustain_ratio"] > described["kick"]["sustain_ratio"]


def test_percussive_material_is_told_from_sustained(described):
    assert described["hat"]["percussive_ratio"] > 0.5
    assert described["chord"]["percussive_ratio"] < 0.3


def test_a_one_shot_is_counted_as_one_hit(described):
    for name in BUILDERS:
        assert described[name]["onset_count"] == 1, name


def test_a_loop_is_counted_as_many_hits(temp_dir, extractor):
    """The measurement that separates a one-shot library from a loop library."""
    seconds, hits = 2.0, 8
    signal = np.zeros(int(seconds * SAMPLE_RATE))
    hit = _time(int(0.2 * SAMPLE_RATE))
    for index in range(hits):
        start = int(index * (seconds / hits) * SAMPLE_RATE)
        signal[start : start + len(hit)] += np.sin(2 * np.pi * 60 * hit) * np.exp(-hit * 20)

    vector = extractor.extract(_write(temp_dir / "loop.wav", signal))
    assert describe_sample_vector(vector, LAYOUT)["onset_count"] == pytest.approx(hits, abs=1)


def test_tempo_is_not_invented_for_a_one_shot(temp_dir, extractor):
    """A 400 ms kick has no BPM, and a fabricated one would poison the space."""
    vector = extractor.extract(kick(temp_dir / "kick.wav"))
    assert LAYOUT.scalar(vector, "tempo") == 0.0


def test_the_classifier_agrees_with_the_audio(described):
    for name, expected in (
        ("kick", "kick"),
        ("hat", "hat_closed"),
        ("bass", "bass"),
        ("chord", "chord"),
        ("pad", "pad"),
    ):
        guess = samples.classify(described[name])
        assert guess.key == expected, f"{name} read as {guess.key}"


# ----------------------------------------------------------------------
# The whole workflow
# ----------------------------------------------------------------------


def build_pack(root, per_category=4):
    """A folder of category folders, as a sample pack ships."""
    rng = np.random.default_rng(11)
    for name, build in BUILDERS.items():
        folder = root / name
        folder.mkdir(parents=True, exist_ok=True)
        for index in range(per_category):
            build(folder / f"{name}_{index:02d}.wav", rng=rng)
    return str(root)


@pytest.fixture
def pack(temp_dir):
    return build_pack(temp_dir / "pack")


def test_a_sample_collection_learns_its_folders(db, config, pack):
    collection = groups_mod.ensure_collection(db, "Drums", profile="sample")
    groups_mod.import_folders_as_groups(db, collection["id"], pack, config)

    collection = db.get_collection(collection_id=collection["id"])
    metrics = groups_mod.fit_collection(db, collection, config)

    assert metrics["profile"] == "sample"
    assert metrics["n_groups"] == len(BUILDERS)
    assert metrics["embedding"]["profile"] == "sample"
    assert metrics["embedding"]["input_dim"] == LAYOUT.dim
    # These categories are not subtle; anything low here means the space is
    # not carrying what the descriptors measured.
    assert metrics["accuracy"] > 0.8


def test_new_one_shots_are_sorted_into_the_right_folders(db, config, temp_dir, pack):
    collection = groups_mod.ensure_collection(db, "Drums", profile="sample")
    groups_mod.import_folders_as_groups(db, collection["id"], pack, config)
    collection = db.get_collection(collection_id=collection["id"])
    groups_mod.fit_collection(db, collection, config)

    # Deliberately uninformative names: this must be decided on the audio.
    incoming = temp_dir / "incoming"
    incoming.mkdir()
    rng = np.random.default_rng(77)
    expected = {}
    for index, (name, build) in enumerate(BUILDERS.items()):
        path = build(incoming / f"unnamed_{index:02d}.wav", rng=rng)
        expected[path.rsplit("/", 1)[-1]] = name

    outcome = sorting.create_session(db, config, collection, [str(incoming)])
    items = db.list_sort_items(outcome["session"]["id"], limit=100)

    assert len(items) == len(BUILDERS)
    correct = sum(1 for item in items if item["suggested_group_name"] == expected[item["filename"]])
    assert correct >= len(BUILDERS) - 1, [
        (item["filename"], item["suggested_group_name"]) for item in items
    ]


def test_analysis_under_one_profile_does_not_satisfy_the_other(db, config, temp_dir):
    path = kick(temp_dir / "kick.wav")

    analyze_paths(db, config, [str(temp_dir)], profile="sample")
    track = db.get_track_by_filepath(path)

    assert db.get_features(track["id"], "sample") is not None
    assert db.get_features(track["id"], "music") is None

    analyze_paths(db, config, [str(temp_dir)], profile="music")
    assert len(db.get_features(track["id"], "music")) == build_layout(20, "music").dim
    assert len(db.get_features(track["id"], "sample")) == LAYOUT.dim


def test_fitting_reports_groups_that_lack_features_for_this_profile(db, config, temp_dir):
    """A music-analysed folder in a sample collection must say so, not crash."""
    kick(temp_dir / "kick.wav")
    analyze_paths(db, config, [str(temp_dir)], profile="music")

    collection = groups_mod.ensure_collection(db, "Drums", profile="sample")
    group = groups_mod.create_group(db, collection["id"], "Kicks")
    track = db.get_track_by_filepath(str(temp_dir / "kick.wav"))
    db.add_group_members(group["id"], [track["id"]])

    _, _, _, problems = groups_mod.load_group_features(db, collection["id"], "sample")
    assert any("sample" in problem["issue"] for problem in problems)
