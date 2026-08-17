"""Profiles select a feature layout and a set of defaults, and nothing else.

The point of these tests is that the two profiles stay genuinely separate:
different widths, different families, no silent cross-contamination when a
vector of one kind meets code expecting the other.
"""

import pytest

from music_cluster import profiles
from music_cluster.config import Config
from music_cluster.features import DEFAULT_DIMS, build_layout, layout_for_dim


def test_known_profiles():
    assert profiles.names() == ["music", "sample"]
    assert profiles.normalize(None) == "music"
    assert profiles.normalize("SAMPLE") == "sample"
    assert profiles.is_sample("sample")
    assert not profiles.is_sample("music")


def test_unknown_profile_is_rejected_rather_than_defaulted():
    """A typo in a config file must not quietly become "music"."""
    with pytest.raises(ValueError, match="Unknown audio profile"):
        profiles.normalize("drums")


def test_sample_layout_is_wider_and_contains_the_music_blocks():
    music = build_layout(20, "music")
    sample = build_layout(20, "sample")

    assert sample.dim > music.dim
    assert sample.is_sample and not music.is_sample
    # Every music span keeps its position, so the shared blocks are extracted
    # by the same code in both profiles.
    for name, span in music.spans.items():
        assert sample.spans[name] == span

    for block in ("attack_time", "band_energy", "f0_hz", "chroma_peaks"):
        assert sample.has(block)
        assert not music.has(block)


def test_envelope_family_only_exists_for_samples():
    assert len(build_layout(20, "sample").family_indices("envelope")) > 0
    assert len(build_layout(20, "music").family_indices("envelope")) == 0


def test_every_span_belongs_to_a_family():
    """A span with no family would silently escape feature weighting."""
    for profile in profiles.names():
        layout = build_layout(20, profile)
        covered = set()
        for family in ("timbre", "rhythm", "harmony", "dynamics", "envelope"):
            covered.update(layout.family_indices(family).tolist())
        assert covered == set(range(layout.dim))


def test_layout_is_recovered_from_a_vector_width():
    for profile in profiles.names():
        assert layout_for_dim(DEFAULT_DIMS[profile]).profile == profile
        assert layout_for_dim(DEFAULT_DIMS[profile], profile).profile == profile


def test_layout_honours_an_explicit_profile_for_unusual_widths():
    narrow = build_layout(13, "sample")
    recovered = layout_for_dim(narrow.dim, "sample")
    assert recovered.profile == "sample"
    assert recovered.n_mfcc == 13
    assert recovered.dim == narrow.dim


def test_config_layers_profile_defaults_over_the_global_section(temp_dir):
    config = Config(str(temp_dir / "config.yaml"))

    music = config.section("feature_extraction", "music")
    sample = config.section("feature_extraction", "sample")

    assert music["excerpt_seconds"] == 90
    # Samples are never excerpted: the attack is at the start of the file.
    assert sample["excerpt_seconds"] == 0
    assert sample["hop_size"] < music["hop_size"]
    assert sample["max_seconds"]


def test_user_overrides_beat_profile_defaults(temp_dir):
    config = Config(str(temp_dir / "config.yaml"))
    config.set(["feature_extraction", "profiles", "sample", "hop_size"], 128)

    assert config.section("feature_extraction", "sample")["hop_size"] == 128
    # ...and leave the other profile alone.
    assert config.section("feature_extraction", "music")["hop_size"] == 1024


def test_extractor_config_carries_the_profile(temp_dir):
    config = Config(str(temp_dir / "config.yaml"))
    assert config.extractor_config("sample")["profile"] == "sample"
    assert config.extractor_config()["profile"] == "music"


def test_sample_sorting_weights_the_envelope(temp_dir):
    config = Config(str(temp_dir / "config.yaml"))
    weights = config.section("sorting", "sample")["feature_weights"]
    assert weights["envelope"] > weights["rhythm"]
