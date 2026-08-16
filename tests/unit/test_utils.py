"""File discovery and the small helpers everything else leans on."""

import os

import pytest

from music_cluster import utils


def touch(directory, name: str, content: bytes = b"x") -> str:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return str(path)


class TestFindAudioFiles:
    def test_finds_audio_and_ignores_everything_else(self, temp_dir):
        touch(temp_dir, "a.mp3")
        touch(temp_dir, "b.flac")
        touch(temp_dir, "cover.jpg")
        touch(temp_dir, "notes.txt")

        found = utils.find_audio_files(str(temp_dir))

        assert [os.path.basename(path) for path in found] == ["a.mp3", "b.flac"]

    def test_recurses_by_default(self, temp_dir):
        touch(temp_dir, "top.mp3")
        touch(temp_dir, "House/deep/inner.wav")

        assert len(utils.find_audio_files(str(temp_dir))) == 2
        assert len(utils.find_audio_files(str(temp_dir), recursive=False)) == 1

    def test_extension_matching_is_case_insensitive(self, temp_dir):
        touch(temp_dir, "SHOUTY.MP3")
        touch(temp_dir, "quiet.AiFf")

        assert len(utils.find_audio_files(str(temp_dir))) == 2

    def test_accepts_extensions_with_or_without_a_leading_dot(self, temp_dir):
        touch(temp_dir, "a.mp3")
        touch(temp_dir, "b.flac")

        with_dot = utils.find_audio_files(str(temp_dir), extensions={".mp3"})
        without_dot = utils.find_audio_files(str(temp_dir), extensions={"mp3"})

        assert with_dot == without_dot
        assert len(with_dot) == 1

    def test_results_are_sorted_so_runs_are_reproducible(self, temp_dir):
        for name in ("c.mp3", "a.mp3", "b.mp3"):
            touch(temp_dir, name)

        found = utils.find_audio_files(str(temp_dir))
        assert found == sorted(found)

    def test_a_single_audio_file_resolves_to_itself(self, temp_dir):
        path = touch(temp_dir, "one.mp3")
        assert utils.find_audio_files(path) == [path]

    def test_a_single_non_audio_file_resolves_to_nothing(self, temp_dir):
        path = touch(temp_dir, "notes.txt")
        assert utils.find_audio_files(path) == []

    def test_a_missing_directory_is_an_error_rather_than_an_empty_result(self, temp_dir):
        # Silently returning [] would let a typo in a path look like an empty
        # folder, and the user would never learn their path was wrong.
        with pytest.raises(ValueError, match="does not exist"):
            utils.find_audio_files(str(temp_dir / "nope"))

    def test_returns_absolute_paths(self, temp_dir):
        touch(temp_dir, "a.mp3")
        assert all(os.path.isabs(path) for path in utils.find_audio_files(str(temp_dir)))


class TestChecksum:
    def test_identical_content_hashes_identically(self, temp_dir):
        first = touch(temp_dir, "a.mp3", b"same bytes")
        second = touch(temp_dir, "b.mp3", b"same bytes")

        assert utils.compute_file_checksum(first) == utils.compute_file_checksum(second)

    def test_different_content_hashes_differently(self, temp_dir):
        first = touch(temp_dir, "a.mp3", b"one")
        second = touch(temp_dir, "b.mp3", b"two")

        assert utils.compute_file_checksum(first) != utils.compute_file_checksum(second)

    def test_chunk_size_does_not_change_the_result(self, temp_dir):
        path = touch(temp_dir, "big.mp3", b"abcdefghij" * 500)

        assert utils.compute_file_checksum(path, chunk_size=7) == utils.compute_file_checksum(
            path, chunk_size=4096
        )

    def test_an_empty_file_still_hashes(self, temp_dir):
        path = touch(temp_dir, "empty.mp3", b"")
        assert len(utils.compute_file_checksum(path)) == 32


class TestGetFileInfo:
    def test_reports_name_size_and_absolute_path(self, temp_dir):
        path = touch(temp_dir, "track.mp3", b"12345")

        info = utils.get_file_info(path)

        assert info["filename"] == "track.mp3"
        assert info["file_size"] == 5
        assert os.path.isabs(info["filepath"])
        assert info["modified_time"] > 0


class TestFormatDuration:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (0, "0:00"),
            (9, "0:09"),
            (65, "1:05"),
            (225, "3:45"),
            (3599, "59:59"),
            (3600, "1:00:00"),
            (5025, "1:23:45"),
        ],
    )
    def test_formats_the_way_a_player_would(self, seconds, expected):
        assert utils.format_duration(seconds) == expected

    def test_truncates_rather_than_rounding_up(self, seconds=59.9):
        assert utils.format_duration(seconds) == "0:59"


class TestParseExtensions:
    def test_adds_dots_and_lowercases(self):
        assert utils.parse_extensions("MP3, flac,.WAV") == {".mp3", ".flac", ".wav"}

    def test_ignores_empty_entries(self):
        assert utils.parse_extensions("mp3,,  ,flac") == {".mp3", ".flac"}

    def test_an_empty_string_yields_no_extensions(self):
        assert utils.parse_extensions("") == set()


class TestEnsureDirectory:
    def test_creates_nested_directories(self, temp_dir):
        target = temp_dir / "a" / "b" / "c"
        utils.ensure_directory(str(target))
        assert target.is_dir()

    def test_is_a_no_op_when_the_directory_exists(self, temp_dir):
        utils.ensure_directory(str(temp_dir))
        utils.ensure_directory(str(temp_dir))
        assert temp_dir.is_dir()


class TestGetRelativePath:
    def test_relativises_below_the_base(self, temp_dir):
        result = utils.get_relative_path(str(temp_dir / "House" / "a.mp3"), str(temp_dir))
        assert result == os.path.join("House", "a.mp3")

    def test_walks_up_when_the_path_is_outside_the_base(self, temp_dir):
        result = utils.get_relative_path(str(temp_dir / "a.mp3"), str(temp_dir / "House"))
        assert result.startswith("..")


def test_default_extensions_cover_the_formats_the_readme_promises():
    # FORMATS.md and the README both advertise this list; a format dropped from
    # here would silently stop being imported.
    for extension in (".mp3", ".flac", ".wav", ".aiff", ".m4a", ".ogg", ".opus", ".aac", ".wma"):
        assert extension in utils.DEFAULT_AUDIO_EXTENSIONS
