"""Label suggestions for candidate groups."""

import numpy as np
import pytest

from music_cluster import naming
from music_cluster.features import build_layout


LAYOUT = build_layout(20)


def vector_with(**spans) -> np.ndarray:
    """A feature vector with named spans set and everything else at zero."""
    vector = np.zeros(LAYOUT.dim)
    for name, value in spans.items():
        start, end = LAYOUT.spans[name]
        vector[start:end] = value
    return vector


def track(**fields) -> dict:
    base = {"artist": None, "title": None, "genre": None, "tag_bpm": None}
    base.update(fields)
    return base


class TestDominantGenre:
    def test_reports_the_genre_most_of_the_group_carries(self):
        tracks = [track(genre="House")] * 8 + [track(genre="Techno")] * 2
        assert naming.dominant_genre(tracks) == "House"

    def test_title_cases_so_tagging_style_does_not_show_through(self):
        assert naming.dominant_genre([track(genre="deep house")] * 4) == "Deep House"

    def test_splits_multi_genre_tags(self):
        tracks = [track(genre="House / Deep House")] * 4 + [track(genre="House, Tech")] * 4
        assert naming.dominant_genre(tracks) == "House"

    def test_returns_nothing_when_no_genre_is_dominant(self):
        tracks = [track(genre=name) for name in ("House", "Techno", "Jungle", "Ambient")]
        assert naming.dominant_genre(tracks) is None

    def test_returns_nothing_when_nothing_is_tagged(self):
        assert naming.dominant_genre([track(), track()]) is None

    def test_the_share_is_measured_against_tagged_tracks_only(self):
        # Two tagged tracks out of twenty untagged ones still describe the
        # group's only available vocabulary.
        tracks = [track(genre="Jungle")] * 2 + [track()] * 18
        assert naming.dominant_genre(tracks) == "Jungle"

    def test_an_empty_group_has_no_genre(self):
        assert naming.dominant_genre([]) is None


class TestTempoBand:
    @pytest.mark.parametrize(
        ("bpm", "expected"),
        [
            (75, "Downtempo"),
            (100, "Slow"),
            (124, "House"),
            (130, "Techno"),
            (140, "Fast Techno"),
            (150, "Hard"),
            (172, "Drum & Bass"),
            (200, "Very Fast"),
        ],
    )
    def test_names_the_band_a_tempo_falls_in(self, bpm, expected):
        assert naming.tempo_band(bpm) == expected

    def test_an_unknown_tempo_is_mixed(self):
        assert naming.tempo_band(None) == "Mixed"
        assert naming.tempo_band(0) == "Mixed"
        assert naming.tempo_band(500) == "Mixed"


class TestBpmRangeLabel:
    def test_reports_a_range_when_the_group_is_tempo_coherent(self):
        tracks = [track(tag_bpm=bpm) for bpm in (124, 125, 126, 127, 128, 126, 125, 124)]
        label = naming.bpm_range_label(vector_with(tempo=126.0), tracks)
        # The 10th and 90th percentiles, so a single outlier cannot widen it.
        assert label == "124-127 BPM"

    def test_collapses_to_a_single_number_when_the_spread_is_tiny(self):
        tracks = [track(tag_bpm=128) for _ in range(8)]
        assert naming.bpm_range_label(vector_with(tempo=128.0), tracks) == "128 BPM"

    def test_omits_the_label_when_the_group_spans_too_many_tempos(self):
        # A "90-175 BPM" label says nothing useful about a crate.
        tracks = [track(tag_bpm=bpm) for bpm in (90, 100, 120, 140, 160, 175, 95, 130)]
        assert naming.bpm_range_label(vector_with(tempo=120.0), tracks) is None

    def test_falls_back_to_the_detected_tempo_when_few_tracks_are_tagged(self):
        tracks = [track() for _ in range(10)]
        assert naming.bpm_range_label(vector_with(tempo=174.0), tracks) == "174 BPM"

    def test_ignores_implausible_tag_values(self):
        tracks = [track(tag_bpm=9999) for _ in range(8)]
        assert naming.bpm_range_label(vector_with(tempo=128.0), tracks) == "128 BPM"


class TestAudioDescriptors:
    def test_a_dark_centroid_reads_as_dark(self):
        assert "Dark" in naming.audio_descriptors(vector_with(spectral_centroid_mean=600.0))

    def test_a_bright_centroid_reads_as_bright(self):
        assert "Bright" in naming.audio_descriptors(vector_with(spectral_centroid_mean=6000.0))

    def test_strong_onsets_read_as_percussive(self):
        assert "Percussive" in naming.audio_descriptors(vector_with(onset_mean=9.0), limit=4)

    def test_a_neutral_vector_yields_no_descriptors(self):
        assert naming.audio_descriptors(vector_with(spectral_centroid_mean=2000.0)) == []

    def test_never_returns_more_than_the_limit(self):
        loud = vector_with(
            spectral_centroid_mean=6000.0,
            onset_mean=9.0,
            spectral_flatness_mean=0.5,
        )
        assert len(naming.audio_descriptors(loud, limit=2)) <= 2
        assert len(naming.audio_descriptors(loud, limit=1)) <= 1


class TestSuggestLabel:
    def test_prefers_the_djs_own_genre_vocabulary(self):
        tracks = [track(genre="Deep House", tag_bpm=122) for _ in range(10)]
        label = naming.suggest_label(vector_with(tempo=122.0), tracks)
        assert "Deep House" in label

    def test_falls_back_to_tempo_and_character_without_tags(self):
        tracks = [track() for _ in range(10)]
        label = naming.suggest_label(vector_with(tempo=172.0), tracks)
        assert label and label != "Untitled group"

    def test_uses_the_fallback_only_when_nothing_at_all_is_known(self):
        label = naming.suggest_label(np.zeros(LAYOUT.dim), [], fallback="Nameless")
        assert label

    def test_does_not_repeat_a_word_within_one_label(self):
        tracks = [track(genre="Minimal") for _ in range(10)]
        label = naming.suggest_label(vector_with(spectral_centroid_mean=600.0), tracks)
        words = [word for word in label.split() if not word.endswith("BPM")]
        assert len(words) == len(set(words))


class TestGroupStats:
    def test_summarises_tempo_genres_and_artists(self):
        tracks = [
            track(artist="Artist A", genre="House", tag_bpm=124),
            track(artist="Artist A", genre="House", tag_bpm=126),
            track(artist="Artist B", genre="Techno", tag_bpm=128),
        ]

        stats = naming.group_stats(vector_with(tempo=126.0, spectral_centroid_mean=2000.0), tracks)

        assert stats["detected_bpm"] == 126.0
        assert stats["tag_bpm_median"] == 126.0
        assert stats["tag_bpm_range"] == [124.0, 128.0]
        assert stats["top_genres"][0] == {"name": "House", "count": 2}
        assert stats["top_artists"][0] == {"name": "Artist A", "count": 2}

    def test_reports_no_tag_tempo_when_nothing_is_tagged(self):
        stats = naming.group_stats(vector_with(tempo=140.0), [track(), track()])
        assert stats["tag_bpm_median"] is None
        assert stats["tag_bpm_range"] is None
        assert stats["detected_bpm"] == 140.0

    def test_lists_at_most_three_genres_and_artists(self):
        tracks = [track(artist=f"A{i}", genre=f"G{i}") for i in range(10)]
        stats = naming.group_stats(vector_with(tempo=120.0), tracks)
        assert len(stats["top_genres"]) == 3
        assert len(stats["top_artists"]) == 3

    def test_is_json_serialisable(self):
        import json

        stats = naming.group_stats(vector_with(tempo=120.0), [track(artist="A", tag_bpm=120)])
        # These go over the wire to the UI, so numpy scalars must already have
        # been converted to plain Python numbers.
        json.dumps(stats)
