"""Collections, groups, playlist import, and fitting.

Extraction is stubbed: what matters here is which tracks end up in which group
and when the collection is considered stale.
"""

import os

import numpy as np
import pytest

from music_cluster import groups as groups_mod
from music_cluster import library
from tests.conftest import FEATURE_DIM, make_centres, make_vector


@pytest.fixture
def fake_extraction(monkeypatch, rng):
    """Stub extraction that gives each folder its own region of feature space."""
    centres: dict = {}

    def extract_one(filepath, extractor_config):
        folder = os.path.dirname(filepath)
        if folder not in centres:
            centres[folder] = rng.normal(0, 1, FEATURE_DIM) * 8.0
        return {
            "filepath": filepath,
            "features": make_vector(rng, centres[folder]),
            "duration": 300.0,
            "file_info": {"filename": os.path.basename(filepath), "file_size": 1000},
            "checksum": filepath,
            "metadata": {"artist": "A", "title": os.path.basename(filepath)},
        }

    monkeypatch.setattr(library, "_extract_one", extract_one)
    return centres


def make_tree(root, layout):
    """Create ``{folder: [filename, ...]}`` under ``root``."""
    for folder, names in layout.items():
        directory = root / folder
        directory.mkdir(parents=True, exist_ok=True)
        for name in names:
            (directory / name).write_bytes(b"pretend audio")
    return root


class TestCollections:
    def test_ensure_creates_once_and_then_reuses(self, db):
        first = groups_mod.ensure_collection(db, "Crates")
        second = groups_mod.ensure_collection(db, "Crates")

        assert first["id"] == second["id"]
        assert len(db.list_collections()) == 1

    def test_find_resolves_by_id_name_or_most_recent(self, db):
        crates = groups_mod.ensure_collection(db, "Crates")
        archive = groups_mod.ensure_collection(db, "Archive")

        assert groups_mod.find_collection(db, crates["id"])["name"] == "Crates"
        assert groups_mod.find_collection(db, str(crates["id"]))["name"] == "Crates"
        assert groups_mod.find_collection(db, "Archive")["id"] == archive["id"]
        assert groups_mod.find_collection(db) is not None

    def test_find_returns_none_for_something_that_does_not_exist(self, db):
        groups_mod.ensure_collection(db, "Crates")
        assert groups_mod.find_collection(db, "Nope") is None

    def test_resolve_raises_with_advice_on_how_to_fix_it(self, db):
        with pytest.raises(LookupError, match="collection create"):
            groups_mod.resolve_collection(db, "Nope")

    def test_sorter_config_layers_collection_settings_over_the_global_ones(self, db, config):
        collection = groups_mod.ensure_collection(db, "Crates")
        collection = dict(collection, settings={"sorting": {"neighbors": 11}})

        sorter_config = groups_mod.sorter_config_for(collection, config)

        assert sorter_config.neighbors == 11
        assert sorter_config.knn_weight == config.get("sorting", "knn_weight")


class TestGroups:
    def test_create_is_idempotent_on_name(self, db):
        collection = groups_mod.ensure_collection(db, "Crates")

        first = groups_mod.create_group(db, collection["id"], "House")
        second = groups_mod.create_group(db, collection["id"], "House")

        assert first["id"] == second["id"]

    def test_paths_are_stored_absolute_and_expanded(self, db, temp_dir):
        collection = groups_mod.ensure_collection(db, "Crates")

        group = groups_mod.create_group(
            db, collection["id"], "House", destination_path=str(temp_dir / "out")
        )

        assert os.path.isabs(group["destination_path"])

    def test_resolve_accepts_an_id_or_a_name(self, db):
        collection = groups_mod.ensure_collection(db, "Crates")
        group = groups_mod.create_group(db, collection["id"], "House")

        assert groups_mod.resolve_group(db, collection["id"], group["id"])["name"] == "House"
        assert groups_mod.resolve_group(db, collection["id"], "House")["id"] == group["id"]

    def test_resolve_will_not_reach_into_another_collection(self, db):
        crates = groups_mod.ensure_collection(db, "Crates")
        archive = groups_mod.ensure_collection(db, "Archive")
        elsewhere = groups_mod.create_group(db, archive["id"], "House")

        with pytest.raises(LookupError):
            groups_mod.resolve_group(db, crates["id"], elsewhere["id"])


class TestImport:
    def test_a_folder_becomes_a_group_that_files_back_into_itself(
        self, db, config, temp_dir, fake_extraction
    ):
        make_tree(temp_dir, {"Deep House": ["a.mp3", "b.mp3", "c.mp3"]})
        collection = groups_mod.ensure_collection(db, "Crates")

        report = groups_mod.import_folder_as_group(
            db, collection["id"], str(temp_dir / "Deep House"), config, workers=1
        )

        group = db.get_group(group_id=report.group_id)
        assert group["name"] == "Deep House"
        assert group["destination_path"] == str(temp_dir / "Deep House")
        assert report.added == 3

    def test_importing_the_same_folder_twice_adds_nothing_new(
        self, db, config, temp_dir, fake_extraction
    ):
        make_tree(temp_dir, {"House": ["a.mp3", "b.mp3"]})
        collection = groups_mod.ensure_collection(db, "Crates")

        groups_mod.import_folder_as_group(
            db, collection["id"], str(temp_dir / "House"), config, workers=1
        )
        second = groups_mod.import_folder_as_group(
            db, collection["id"], str(temp_dir / "House"), config, workers=1
        )

        assert second.added == 0
        assert second.already_present == 2

    def test_an_empty_folder_is_an_error_rather_than_an_empty_group(
        self, db, config, temp_dir, fake_extraction
    ):
        (temp_dir / "Empty").mkdir()
        collection = groups_mod.ensure_collection(db, "Crates")

        with pytest.raises(ValueError, match="No audio files"):
            groups_mod.import_folder_as_group(db, collection["id"], str(temp_dir / "Empty"), config)

    def test_a_file_is_not_a_folder(self, db, config, temp_dir):
        path = temp_dir / "a.mp3"
        path.write_bytes(b"x")
        collection = groups_mod.ensure_collection(db, "Crates")

        with pytest.raises(ValueError, match="Not a directory"):
            groups_mod.import_folder_as_group(db, collection["id"], str(path), config)

    def test_a_tree_becomes_one_group_per_subfolder(self, db, config, temp_dir, fake_extraction):
        make_tree(
            temp_dir,
            {
                "Deep House": ["a.mp3", "b.mp3"],
                "Techno": ["c.mp3", "d.mp3"],
                "Jungle": ["e.mp3"],
            },
        )
        collection = groups_mod.ensure_collection(db, "Crates")

        reports = groups_mod.import_folders_as_groups(
            db, collection["id"], str(temp_dir), config, workers=1
        )

        assert {report.group_name for report in reports} == {"Deep House", "Techno", "Jungle"}

    def test_a_minimum_track_count_skips_thin_folders(self, db, config, temp_dir, fake_extraction):
        make_tree(temp_dir, {"Big": ["a.mp3", "b.mp3", "c.mp3"], "Small": ["d.mp3"]})
        collection = groups_mod.ensure_collection(db, "Crates")

        reports = groups_mod.import_folders_as_groups(
            db, collection["id"], str(temp_dir), config, min_tracks=2, workers=1
        )

        assert [report.group_name for report in reports] == ["Big"]

    def test_hidden_folders_are_left_alone(self, db, config, temp_dir, fake_extraction):
        make_tree(temp_dir, {"House": ["a.mp3"], ".Trash": ["b.mp3"]})
        collection = groups_mod.ensure_collection(db, "Crates")

        reports = groups_mod.import_folders_as_groups(
            db, collection["id"], str(temp_dir), config, workers=1
        )

        assert [report.group_name for report in reports] == ["House"]

    def test_import_report_serialises_for_the_api(self, db, config, temp_dir, fake_extraction):
        make_tree(temp_dir, {"House": ["a.mp3"]})
        collection = groups_mod.ensure_collection(db, "Crates")

        payload = groups_mod.import_folder_as_group(
            db, collection["id"], str(temp_dir / "House"), config, workers=1
        ).to_dict()

        assert payload["group_name"] == "House"
        assert payload["analysis"]["analyzed"] == 1


class TestReadPlaylist:
    def test_reads_an_m3u_with_comments_and_relative_paths(self, temp_dir):
        make_tree(temp_dir, {"music": ["a.mp3", "b.mp3"]})
        playlist = temp_dir / "set.m3u"
        playlist.write_text("#EXTM3U\n#EXTINF:300,A\nmusic/a.mp3\nmusic/b.mp3\n")

        entries = groups_mod.read_playlist(str(playlist))

        assert [os.path.basename(entry) for entry in entries] == ["a.mp3", "b.mp3"]
        assert all(os.path.isabs(entry) for entry in entries)

    def test_reads_pls_style_assignments(self, temp_dir):
        make_tree(temp_dir, {"music": ["a.mp3"]})
        playlist = temp_dir / "set.pls"
        playlist.write_text("[playlist]\nFile1=music/a.mp3\n")

        assert len(groups_mod.read_playlist(str(playlist))) == 1

    def test_decodes_file_urls(self, temp_dir):
        make_tree(temp_dir, {"music": ["a b.mp3"]})
        playlist = temp_dir / "set.m3u8"
        playlist.write_text(f"file://{temp_dir}/music/a%20b.mp3\n")

        assert [os.path.basename(e) for e in groups_mod.read_playlist(str(playlist))] == ["a b.mp3"]

    def test_skips_remote_streams(self, temp_dir):
        playlist = temp_dir / "set.m3u"
        playlist.write_text("http://example.com/stream.mp3\n")

        assert groups_mod.read_playlist(str(playlist)) == []

    def test_skips_entries_whose_files_are_missing(self, temp_dir):
        playlist = temp_dir / "set.m3u"
        playlist.write_text("music/gone.mp3\n")

        assert groups_mod.read_playlist(str(playlist)) == []

    def test_a_playlist_becomes_a_group(self, db, config, temp_dir, fake_extraction):
        make_tree(temp_dir, {"music": ["a.mp3", "b.mp3"]})
        playlist = temp_dir / "Peak Time.m3u"
        playlist.write_text("music/a.mp3\nmusic/b.mp3\n")
        collection = groups_mod.ensure_collection(db, "Crates")

        report = groups_mod.import_playlist_as_group(
            db, collection["id"], str(playlist), config, workers=1
        )

        assert report.group_name == "Peak Time"
        assert report.added == 2

    def test_a_playlist_with_nothing_playable_is_an_error(self, db, config, temp_dir):
        playlist = temp_dir / "set.m3u"
        playlist.write_text("#EXTM3U\nhttp://example.com/stream\n")
        collection = groups_mod.ensure_collection(db, "Crates")

        with pytest.raises(ValueError, match="No playable entries"):
            groups_mod.import_playlist_as_group(db, collection["id"], str(playlist), config)

    @pytest.mark.parametrize("name", ["set.m3u", "set.m3u8", "set.pls", "set.txt"])
    def test_recognises_playlist_extensions(self, name):
        assert groups_mod.looks_like_playlist(name)

    def test_does_not_mistake_audio_for_a_playlist(self):
        assert not groups_mod.looks_like_playlist("track.mp3")


class TestLoadGroupFeatures:
    def test_loads_a_matrix_per_group(self, db, populated):
        collection_id, membership, _ = populated

        features, names, track_ids, problems = groups_mod.load_group_features(db, collection_id)

        assert set(features) == set(membership)
        assert all(matrix.shape == (12, FEATURE_DIM) for matrix in features.values())
        assert set(names) == set(membership)
        assert problems == []

    def test_reports_a_group_with_no_members_as_a_problem(self, db, populated):
        collection_id, _, _ = populated
        empty_id = db.create_group(collection_id, "Empty")

        features, _, _, problems = groups_mod.load_group_features(db, collection_id)

        assert empty_id not in features
        assert any(problem["group_id"] == empty_id for problem in problems)

    def test_reports_members_that_were_never_analysed(self, db, populated):
        collection_id, membership, _ = populated
        group_id = next(iter(membership))
        unanalysed = db.upsert_track(filepath="/fake/new.mp3", filename="new.mp3")
        db.add_group_members(group_id, [unanalysed])

        _, _, _, problems = groups_mod.load_group_features(db, collection_id)

        assert any("not analysed" in problem["issue"] for problem in problems)


class TestFitting:
    def test_fitting_saves_a_model_and_reports_accuracy(self, db, config, populated):
        collection_id, _, _ = populated
        collection = db.get_collection(collection_id=collection_id)

        metrics = groups_mod.fit_collection(db, collection, config)

        assert metrics["accuracy"] > 0.8
        assert db.load_model(collection_id) is not None

    def test_fitting_needs_at_least_two_groups(self, db, config):
        collection_id = db.create_collection("Thin")
        group_id = db.create_group(collection_id, "Only")
        track_id = db.upsert_track(filepath="/fake/a.mp3", filename="a.mp3")
        db.add_features(track_id, np.zeros(FEATURE_DIM))
        db.add_group_members(group_id, [track_id])

        with pytest.raises(ValueError, match="At least two groups"):
            groups_mod.fit_collection(db, db.get_collection(collection_id=collection_id), config)

    def test_load_sorter_fits_on_demand(self, db, config, populated):
        collection_id, _, _ = populated
        collection = db.get_collection(collection_id=collection_id)

        sorter, model = groups_mod.load_sorter(db, collection, config)

        assert sorter.is_fitted
        assert model is not None

    def test_load_sorter_can_refuse_to_fit(self, db, config, populated):
        collection_id, _, _ = populated
        collection = db.get_collection(collection_id=collection_id)

        with pytest.raises(LookupError, match="music-cluster fit"):
            groups_mod.load_sorter(db, collection, config, autofit=False)

    def test_a_changed_collection_makes_the_model_stale(self, db, config, populated, rng):
        collection_id, membership, centres = populated
        collection = db.get_collection(collection_id=collection_id)
        groups_mod.fit_collection(db, collection, config)

        new_group = db.create_group(collection_id, "Added later")
        track_ids = []
        for index in range(10):
            track_id = db.upsert_track(filepath=f"/fake/new/{index}.mp3", filename=f"{index}.mp3")
            db.add_features(track_id, make_vector(rng, make_centres(rng, 1, separation=20.0)[0]))
            track_ids.append(track_id)
        db.add_group_members(new_group, track_ids)
        db.touch_collection(collection_id)

        sorter, _ = groups_mod.load_sorter(
            db, db.get_collection(collection_id=collection_id), config
        )

        # A stale sorter would keep suggesting only the original groups.
        assert new_group in sorter.group_ids
