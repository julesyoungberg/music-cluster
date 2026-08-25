"""Ingestion: turning files on disk into analysed tracks.

Feature extraction itself is exercised in tests/integration/test_extractor.py
against real audio; here it is stubbed so the ingestion logic — skipping,
re-analysing, error collection, pruning — can be tested exhaustively.
"""

import numpy as np
import pytest

from music_cluster import library
from music_cluster.library import AnalysisResult
from tests.conftest import FEATURE_DIM


@pytest.fixture
def audio_tree(temp_dir):
    """Three audio files and one non-audio file, in nested folders."""
    (temp_dir / "House").mkdir()
    (temp_dir / "Techno").mkdir()
    paths = [
        temp_dir / "House" / "a.mp3",
        temp_dir / "House" / "b.flac",
        temp_dir / "Techno" / "c.wav",
    ]
    for path in paths:
        path.write_bytes(b"pretend audio")
    (temp_dir / "House" / "cover.jpg").write_bytes(b"not audio")
    return temp_dir, [str(path) for path in paths]


@pytest.fixture
def fake_extraction(monkeypatch):
    """Stub extraction. Returns the list of files it was asked about."""
    seen: list = []
    failures: set = set()

    def extract_one(filepath, extractor_config):
        seen.append(filepath)
        if filepath in failures:
            return {"filepath": filepath, "error": "Could not decode audio"}
        return {
            "filepath": filepath,
            "features": np.arange(FEATURE_DIM, dtype=float),
            "duration": 300.0,
            "file_info": {"filename": filepath.split("/")[-1], "file_size": 1234},
            "checksum": "abc123",
            "metadata": {"artist": "Someone", "title": "A track"},
        }

    monkeypatch.setattr(library, "_extract_one", extract_one)
    return seen, failures


class TestResolveWorkerCount:
    def test_minus_one_means_every_cpu(self):
        import multiprocessing as mp

        assert library.resolve_worker_count(-1) == max(1, mp.cpu_count())

    def test_never_returns_fewer_than_one(self):
        assert library.resolve_worker_count(0) == 1
        assert library.resolve_worker_count(-99) == 1

    def test_an_explicit_count_is_used_as_given(self):
        assert library.resolve_worker_count(4) == 4


class TestCollectFiles:
    def test_expands_directories_into_audio_files(self, audio_tree):
        root, paths = audio_tree
        assert library.collect_files([str(root)]) == sorted(paths)

    def test_de_duplicates_overlapping_paths(self, audio_tree):
        root, paths = audio_tree
        collected = library.collect_files([str(root), str(root / "House"), paths[0]])
        assert sorted(collected) == sorted(paths)

    def test_preserves_first_seen_order(self, audio_tree):
        root, paths = audio_tree
        collected = library.collect_files([str(root / "Techno"), str(root / "House")])
        assert collected[0].endswith("c.wav")

    def test_honours_an_extension_filter(self, audio_tree):
        root, _ = audio_tree
        assert len(library.collect_files([str(root)], extensions={".mp3"})) == 1

    def test_can_be_told_not_to_recurse(self, audio_tree):
        root, _ = audio_tree
        assert library.collect_files([str(root)], recursive=False) == []


class TestAnalyzeFiles:
    def test_analyses_every_file_and_stores_features(self, db, config, audio_tree, fake_extraction):
        _, paths = audio_tree

        result = library.analyze_files(db, config, paths, workers=1)

        assert result.analyzed == 3
        assert result.failed == 0
        assert len(result.track_ids) == 3
        for track_id in result.track_ids:
            assert db.get_features(track_id) is not None

    def test_stores_the_metadata_that_came_with_the_file(
        self, db, config, audio_tree, fake_extraction
    ):
        _, paths = audio_tree

        result = library.analyze_files(db, config, paths[:1], workers=1)

        track = db.get_track(result.track_ids[0])
        assert track["artist"] == "Someone"
        assert track["duration"] == 300.0
        assert track["checksum"] == "abc123"

    def test_skips_files_already_analysed(self, db, config, audio_tree, fake_extraction):
        _, paths = audio_tree
        seen, _ = fake_extraction

        library.analyze_files(db, config, paths, workers=1)
        seen.clear()
        second = library.analyze_files(db, config, paths, workers=1)

        assert second.skipped == 3
        assert second.analyzed == 0
        assert seen == []
        # A skipped track is still a usable track, so callers get its ID back.
        assert len(second.track_ids) == 3

    def test_update_forces_a_re_analysis(self, db, config, audio_tree, fake_extraction):
        _, paths = audio_tree
        seen, _ = fake_extraction

        library.analyze_files(db, config, paths, workers=1)
        seen.clear()
        second = library.analyze_files(db, config, paths, update=True, workers=1)

        assert second.analyzed == 3
        assert len(seen) == 3

    def test_a_failing_file_does_not_stop_the_others(self, db, config, audio_tree, fake_extraction):
        _, paths = audio_tree
        _, failures = fake_extraction
        failures.add(paths[1])

        result = library.analyze_files(db, config, paths, workers=1)

        assert result.analyzed == 2
        assert result.failed == 1
        assert result.errors[0]["filepath"] == paths[1]
        assert "decode" in result.errors[0]["error"]

    def test_reports_progress_for_every_file(self, db, config, audio_tree, fake_extraction):
        _, paths = audio_tree
        reported = []

        library.analyze_files(
            db,
            config,
            paths,
            workers=1,
            progress=lambda done, total, path: reported.append((done, total)),
        )

        assert reported == [(1, 3), (2, 3), (3, 3)]

    def test_an_empty_list_is_a_no_op(self, db, config, fake_extraction):
        result = library.analyze_files(db, config, [], workers=1)
        assert result.total == 0
        assert result.track_ids == []

    def test_parallel_and_serial_runs_agree(self, db, config, audio_tree, fake_extraction):
        _, paths = audio_tree

        parallel = library.analyze_files(db, config, paths, workers=4)

        assert parallel.analyzed == 3
        assert sorted(parallel.track_ids) == sorted(
            db.get_track_by_filepath(path)["id"] for path in paths
        )


class TestAnalyzePaths:
    def test_walks_a_folder_tree(self, db, config, audio_tree, fake_extraction):
        root, paths = audio_tree

        result = library.analyze_paths(db, config, [str(root)], workers=1)

        assert result.analyzed == len(paths)

    def test_ignores_files_that_are_not_audio(self, db, config, audio_tree, fake_extraction):
        root, _ = audio_tree
        seen, _ = fake_extraction

        library.analyze_paths(db, config, [str(root)], workers=1)

        assert not any(path.endswith(".jpg") for path in seen)


class TestAnalysisResult:
    def test_total_counts_every_outcome(self):
        result = AnalysisResult(analyzed=2, skipped=3, failed=1)
        assert result.total == 6

    def test_serialises_for_the_api(self):
        result = AnalysisResult(analyzed=1, track_ids=[7])
        payload = result.to_dict()

        assert payload["analyzed"] == 1
        assert payload["track_ids"] == [7]
        assert payload["total"] == 1

    def test_caps_the_error_list_so_a_bad_folder_cannot_flood_a_response(self):
        result = AnalysisResult(
            failed=200, errors=[{"filepath": str(i), "error": "boom"} for i in range(200)]
        )
        assert len(result.to_dict()["errors"]) == 50


class TestPruneMissing:
    def test_removes_tracks_whose_files_are_gone(self, db, config, audio_tree, fake_extraction):
        root, paths = audio_tree
        library.analyze_files(db, config, paths, workers=1)
        import os

        os.remove(paths[0])

        removed = library.prune_missing(db)

        assert len(removed) == 1
        assert db.get_track_by_filepath(paths[0]) is None
        assert db.get_track_by_filepath(paths[1]) is not None

    def test_removes_nothing_when_every_file_is_present(
        self, db, config, audio_tree, fake_extraction
    ):
        _, paths = audio_tree
        library.analyze_files(db, config, paths, workers=1)

        assert library.prune_missing(db) == []
