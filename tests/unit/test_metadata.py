"""Tag reading, and the filename fallback for files that have no tags."""

import pytest

from music_cluster import metadata


class FakeTags:
    """Stands in for a mutagen tag container, which is dict-like but not a dict."""

    def __init__(self, values):
        self._values = values

    def get(self, key):
        return self._values.get(key)


class TestFirstValue:
    def test_takes_the_first_key_that_has_a_value(self):
        tags = FakeTags({"TPE1": "From ID3", "artist": "From easy tags"})
        assert metadata._first_value(tags, ("artist", "TPE1")) == "From easy tags"

    def test_skips_missing_and_empty_values(self):
        tags = FakeTags({"artist": "", "TPE1": "   ", "\xa9ART": "Real artist"})
        assert metadata._first_value(tags, ("artist", "TPE1", "\xa9ART")) == "Real artist"

    def test_unwraps_the_list_mutagen_usually_returns(self):
        tags = FakeTags({"artist": ["Boards of Canada", "Someone else"]})
        assert metadata._first_value(tags, ("artist",)) == "Boards of Canada"

    def test_decodes_bytes(self):
        tags = FakeTags({"comment": b"caf\xc3\xa9"})
        assert metadata._first_value(tags, ("comment",)) == "café"

    def test_an_empty_list_is_not_a_value(self):
        tags = FakeTags({"artist": [], "TPE1": "Fallback"})
        assert metadata._first_value(tags, ("artist", "TPE1")) == "Fallback"

    def test_a_tag_container_that_raises_is_not_fatal(self):
        class Exploding:
            def get(self, key):
                raise RuntimeError("corrupt tag frame")

        assert metadata._first_value(Exploding(), ("artist",)) is None


class TestParseYear:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1998", 1998),
            ("2021-06-04", 2021),
            ("Released 2003 on vinyl", 2003),
            (None, None),
            ("", None),
            ("no year here", None),
            ("1899", None),
        ],
    )
    def test_pulls_a_plausible_year_out_of_free_text(self, raw, expected):
        assert metadata._parse_year(raw) == expected


class TestParseBpm:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("128", 128.0),
            ("124.5", 124.5),
            ("128 BPM", 128.0),
            (None, None),
            ("", None),
            ("not a tempo", None),
        ],
    )
    def test_reads_a_tempo(self, raw, expected):
        assert metadata._parse_bpm(raw) == expected

    @pytest.mark.parametrize("raw", ["5", "19", "301", "9999"])
    def test_rejects_tempos_outside_the_range_music_actually_uses(self, raw):
        # DJ software occasionally writes garbage into the BPM field; a value of
        # 9999 would distort every tempo-based label and group summary.
        assert metadata._parse_bpm(raw) is None


class TestFromFilename:
    def test_splits_artist_and_title_on_a_hyphen(self):
        assert metadata._from_filename("/m/Floating Points - Nespole.mp3") == (
            "Floating Points",
            "Nespole",
        )

    def test_splits_on_an_en_dash_too(self):
        artist, title = metadata._from_filename("/m/Jamie xx – Gosh.flac")  # noqa: RUF001
        assert (artist, title) == ("Jamie xx", "Gosh")

    def test_keeps_later_hyphens_in_the_title(self):
        artist, title = metadata._from_filename("/m/Artist - Track - Remix.mp3")
        assert artist == "Artist"
        assert title == "Track - Remix"

    def test_a_name_with_no_separator_is_all_title(self):
        assert metadata._from_filename("/m/Untitled.wav") == (None, "Untitled")

    def test_a_hyphen_without_spaces_is_part_of_the_name(self):
        # "Sun-Ra" is one word, not an artist and a title.
        assert metadata._from_filename("/m/Sun-Ra.mp3") == (None, "Sun-Ra")


class TestReadMetadata:
    def test_a_file_with_no_tags_falls_back_to_the_filename(self, temp_dir):
        path = temp_dir / "4hero - Mr Kirks Nightmare.mp3"
        path.write_bytes(b"not really an mp3")

        result = metadata.read_metadata(str(path))

        assert result["artist"] == "4hero"
        assert result["title"] == "Mr Kirks Nightmare"

    def test_an_unreadable_file_still_returns_the_full_shape(self, temp_dir):
        path = temp_dir / "broken.mp3"
        path.write_bytes(b"\x00\x01\x02")

        result = metadata.read_metadata(str(path))

        # Callers index this dict directly, so every key must exist even when
        # nothing could be read.
        assert set(result) == {
            "title",
            "artist",
            "album",
            "genre",
            "year",
            "tag_bpm",
            "tag_key",
            "comment",
        }

    def test_a_missing_file_does_not_raise(self, temp_dir):
        result = metadata.read_metadata(str(temp_dir / "gone.mp3"))
        assert result["title"] == "gone"


class TestDisplayName:
    def test_prefers_artist_and_title(self):
        assert (
            metadata.display_name({"artist": "Burial", "title": "Archangel", "filename": "x.mp3"})
            == "Burial - Archangel"
        )

    def test_falls_back_through_title_then_filename_then_filepath(self):
        assert metadata.display_name({"title": "Archangel"}) == "Archangel"
        assert metadata.display_name({"filename": "x.mp3"}) == "x.mp3"
        assert metadata.display_name({"filepath": "/m/x.mp3"}) == "/m/x.mp3"
        assert metadata.display_name({}) == "Unknown track"

    def test_an_artist_with_no_title_is_not_shown_alone(self):
        # "Burial - " would be a worse label than the filename.
        assert metadata.display_name({"artist": "Burial", "filename": "x.mp3"}) == "x.mp3"
