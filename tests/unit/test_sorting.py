"""Sort sessions: scoring, decisions, commit, and learning."""

import os

import numpy as np
import pytest

from music_cluster import groups as groups_mod
from music_cluster import library, sorting
from tests.conftest import FEATURE_DIM, make_vector


@pytest.fixture
def collection(db, config, populated):
    collection_id, membership, centres = populated
    return db.get_collection(collection_id=collection_id), membership, centres


@pytest.fixture
def incoming(temp_dir, monkeypatch, rng, collection):
    """A folder of new music that sounds like the first group."""
    _, membership, centres = collection
    folder = temp_dir / "NewMusic"
    folder.mkdir()
    for index in range(6):
        (folder / f"new{index}.mp3").write_bytes(b"pretend audio")

    target_centre = centres[0]

    def extract_one(filepath, extractor_config):
        return {
            "filepath": filepath,
            "features": make_vector(rng, target_centre),
            "duration": 300.0,
            "file_info": {"filename": os.path.basename(filepath), "file_size": 1000},
            "checksum": filepath,
            "metadata": {"artist": "New", "title": os.path.basename(filepath)},
        }

    monkeypatch.setattr(library, "_extract_one", extract_one)
    return str(folder), next(iter(membership))


class TestCreateSession:
    def test_scores_every_incoming_track(self, db, config, collection, incoming):
        collection_row, _, _ = collection
        folder, _ = incoming

        result = sorting.create_session(db, config, collection_row, [folder], workers=1)

        assert result["analysis"]["analyzed"] == 6
        assert sum(result["summary"].values()) == 6

    def test_names_the_session_after_the_folder_by_default(self, db, config, collection, incoming):
        collection_row, _, _ = collection
        folder, _ = incoming

        result = sorting.create_session(db, config, collection_row, [folder], workers=1)

        assert result["session"]["name"] == "NewMusic"

    def test_an_explicit_name_wins(self, db, config, collection, incoming):
        collection_row, _, _ = collection
        folder, _ = incoming

        result = sorting.create_session(
            db, config, collection_row, [folder], name="Friday buys", workers=1
        )

        assert result["session"]["name"] == "Friday buys"

    def test_a_folder_with_no_audio_is_an_error(self, db, config, collection, temp_dir):
        collection_row, _, _ = collection
        empty = temp_dir / "Empty"
        empty.mkdir()

        with pytest.raises(ValueError, match="No audio files"):
            sorting.create_session(db, config, collection_row, [str(empty)])

    def test_suggestions_point_at_the_group_the_tracks_resemble(
        self, db, config, collection, incoming
    ):
        collection_row, _, _ = collection
        folder, expected_group = incoming

        result = sorting.create_session(db, config, collection_row, [folder], workers=1)
        items = db.list_sort_items(result["session"]["id"], limit=100)

        assert all(item["suggested_group_id"] == expected_group for item in items)

    def test_items_wait_for_a_human_by_default(self, db, config, collection, incoming):
        collection_row, _, _ = collection
        folder, _ = incoming

        result = sorting.create_session(db, config, collection_row, [folder], workers=1)

        # Nothing is filed until someone confirms it — that is the whole promise
        # of the directed workflow.
        assert result["summary"].get(sorting.STATUS_ACCEPTED, 0) == 0
        assert result["summary"][sorting.STATUS_PENDING] == 6

    def test_auto_accept_pre_confirms_confident_suggestions(self, db, config, collection, incoming):
        collection_row, _, _ = collection
        folder, expected_group = incoming

        result = sorting.create_session(
            db, config, collection_row, [folder], auto_accept=True, workers=1
        )

        assert result["summary"].get(sorting.STATUS_ACCEPTED, 0) > 0
        accepted = [
            item
            for item in db.list_sort_items(result["session"]["id"], limit=100)
            if item["status"] == sorting.STATUS_ACCEPTED
        ]
        assert all(item["final_group_id"] == expected_group for item in accepted)


class TestScoreSession:
    def test_rescoring_preserves_human_decisions(self, db, config, collection, incoming):
        collection_row, membership, _ = collection
        folder, _ = incoming
        session_id = sorting.create_session(db, config, collection_row, [folder], workers=1)[
            "session"
        ]["id"]

        other_group = list(membership)[1]
        item = db.list_sort_items(session_id, limit=1)[0]
        sorting.decide(db, item["id"], group_id=other_group)

        sorting.score_session(db, config, session_id)

        refreshed = db.get_sort_item(item["id"])
        assert refreshed["status"] == sorting.STATUS_ACCEPTED
        assert refreshed["final_group_id"] == other_group

    def test_rescoring_an_unknown_session_is_an_error(self, db, config):
        with pytest.raises(LookupError, match="No sort session"):
            sorting.score_session(db, config, 999)


class TestDecide:
    def test_accepting_records_the_group(self, db, config, collection, incoming):
        collection_row, membership, _ = collection
        folder, _ = incoming
        session_id = sorting.create_session(db, config, collection_row, [folder], workers=1)[
            "session"
        ]["id"]
        item = db.list_sort_items(session_id, limit=1)[0]

        decided = sorting.decide(db, item["id"], group_id=list(membership)[1])

        assert decided["status"] == sorting.STATUS_ACCEPTED
        assert decided["final_group_id"] == list(membership)[1]

    def test_skipping_clears_the_group(self, db, config, collection, incoming):
        collection_row, _, _ = collection
        folder, _ = incoming
        session_id = sorting.create_session(db, config, collection_row, [folder], workers=1)[
            "session"
        ]["id"]
        item = db.list_sort_items(session_id, limit=1)[0]

        decided = sorting.decide(db, item["id"], skip=True)

        assert decided["status"] == sorting.STATUS_SKIPPED
        assert decided["final_group_id"] is None

    def test_deciding_nothing_returns_the_item_to_the_queue(self, db, config, collection, incoming):
        collection_row, membership, _ = collection
        folder, _ = incoming
        session_id = sorting.create_session(db, config, collection_row, [folder], workers=1)[
            "session"
        ]["id"]
        item = db.list_sort_items(session_id, limit=1)[0]

        sorting.decide(db, item["id"], group_id=list(membership)[1])
        reverted = sorting.decide(db, item["id"])

        assert reverted["status"] == sorting.STATUS_PENDING
        assert reverted["final_group_id"] is None

    def test_an_unknown_item_is_an_error(self, db):
        with pytest.raises(LookupError, match="No sort item"):
            sorting.decide(db, 12345, skip=True)


class TestAcceptAllConfident:
    def test_accepts_the_confident_ones(self, db, config, collection, incoming):
        collection_row, _, _ = collection
        folder, expected_group = incoming
        session_id = sorting.create_session(db, config, collection_row, [folder], workers=1)[
            "session"
        ]["id"]

        accepted = sorting.accept_all_confident(db, config, session_id)

        assert accepted > 0
        assert all(
            item["final_group_id"] == expected_group
            for item in db.list_sort_items(session_id, status=sorting.STATUS_ACCEPTED, limit=100)
        )

    def test_a_threshold_no_score_can_reach_accepts_nothing(self, db, config, collection, incoming):
        collection_row, _, _ = collection
        folder, _ = incoming
        session_id = sorting.create_session(db, config, collection_row, [folder], workers=1)[
            "session"
        ]["id"]
        # Confidence is a probability, so nothing can clear a threshold above 1.
        config.set(["sorting", "auto_accept_confidence"], 1.5)

        assert sorting.accept_all_confident(db, config, session_id) == 0
        assert db.sort_session_summary(session_id)[sorting.STATUS_PENDING] == 6

    def test_a_collections_own_threshold_overrides_the_global_one(
        self, db, config, collection, incoming
    ):
        collection_row, _, _ = collection
        folder, _ = incoming
        session_id = sorting.create_session(db, config, collection_row, [folder], workers=1)[
            "session"
        ]["id"]
        db.update_collection(
            collection_row["id"], settings={"sorting": {"auto_accept_confidence": 1.5}}
        )

        assert sorting.accept_all_confident(db, config, session_id) == 0

    def test_an_unknown_session_is_an_error(self, db, config):
        with pytest.raises(LookupError, match="No sort session"):
            sorting.accept_all_confident(db, config, 999)


class TestCommit:
    def _accepted_session(self, db, config, collection_row, folder):
        session_id = sorting.create_session(db, config, collection_row, [folder], workers=1)[
            "session"
        ]["id"]
        sorting.accept_all_confident(db, config, session_id)
        return session_id

    def test_planning_lists_what_would_be_written_without_writing_it(
        self, db, config, collection, incoming, temp_dir
    ):
        collection_row, _, _ = collection
        folder, _ = incoming
        session_id = self._accepted_session(db, config, collection_row, folder)
        playlists = temp_dir / "playlists"

        plan = sorting.plan_session_commit(
            db, config, session_id, mode="playlist", playlist_dir=str(playlists)
        )

        assert plan.mode == "playlist"
        assert len(plan.playlists) == 1
        assert not playlists.exists()

    def test_committing_writes_the_playlist(self, db, config, collection, incoming, temp_dir):
        collection_row, _, _ = collection
        folder, _ = incoming
        session_id = self._accepted_session(db, config, collection_row, folder)
        playlists = temp_dir / "playlists"

        result = sorting.commit_session(
            db, config, session_id, mode="playlist", playlist_dir=str(playlists)
        )

        written = result["result"]["playlists"]
        assert len(written) == 1
        assert os.path.isfile(written[0])

    def test_committing_folds_the_tracks_back_into_their_group(
        self, db, config, collection, incoming, temp_dir
    ):
        collection_row, _, _ = collection
        folder, expected_group = incoming
        before = len(db.get_group_member_ids(expected_group))
        session_id = self._accepted_session(db, config, collection_row, folder)

        result = sorting.commit_session(
            db, config, session_id, mode="playlist", playlist_dir=str(temp_dir / "p")
        )

        assert result["learned"] > 0
        assert len(db.get_group_member_ids(expected_group)) > before

    def test_learning_can_be_turned_off(self, db, config, collection, incoming, temp_dir):
        collection_row, _, _ = collection
        folder, expected_group = incoming
        before = len(db.get_group_member_ids(expected_group))
        session_id = self._accepted_session(db, config, collection_row, folder)

        result = sorting.commit_session(
            db, config, session_id, mode="playlist", playlist_dir=str(temp_dir / "p"), learn=False
        )

        assert result["learned"] == 0
        assert len(db.get_group_member_ids(expected_group)) == before

    def test_learning_invalidates_the_fitted_sorter(
        self, db, config, collection, incoming, temp_dir
    ):
        collection_row, _, _ = collection
        folder, _ = incoming
        groups_mod.fit_collection(db, collection_row, config)
        fitted_at = db.load_model(collection_row["id"])["fitted_at"]
        session_id = self._accepted_session(db, config, collection_row, folder)

        sorting.commit_session(
            db, config, session_id, mode="playlist", playlist_dir=str(temp_dir / "p")
        )
        refreshed = db.get_collection(collection_id=collection_row["id"])

        # References changed, so the saved model no longer describes the groups.
        assert str(refreshed["updated_at"]) >= str(fitted_at)

    def test_committing_an_unknown_session_is_an_error(self, db, config):
        with pytest.raises(LookupError, match="No sort session"):
            sorting.commit_session(db, config, 999)


class TestPendingAssignments:
    def test_lists_only_accepted_and_unapplied_items(self, db, config, collection, incoming):
        collection_row, _, _ = collection
        folder, _ = incoming
        session_id = sorting.create_session(db, config, collection_row, [folder], workers=1)[
            "session"
        ]["id"]

        assert sorting.pending_assignments(db, session_id) == []

        sorting.accept_all_confident(db, config, session_id)
        assignments = sorting.pending_assignments(db, session_id)

        assert len(assignments) == 6
        assert all("filepath" in assignment for assignment in assignments)
        assert all(assignment["display"] for assignment in assignments)


class TestSessionOverview:
    def test_reports_the_session_its_groups_and_counts(self, db, config, collection, incoming):
        collection_row, membership, _ = collection
        folder, _ = incoming
        session_id = sorting.create_session(db, config, collection_row, [folder], workers=1)[
            "session"
        ]["id"]

        overview = sorting.session_overview(db, session_id)

        assert overview["session"]["id"] == session_id
        assert len(overview["groups"]) == len(membership)
        assert overview["total"] == 6

    def test_an_unknown_session_is_an_error(self, db):
        with pytest.raises(LookupError, match="No sort session"):
            sorting.session_overview(db, 999)


def test_a_track_far_from_every_group_is_reported_as_unmatched(
    db, config, collection, temp_dir, monkeypatch, rng
):
    collection_row, _, _ = collection
    folder = temp_dir / "Weird"
    folder.mkdir()
    (folder / "outlier.mp3").write_bytes(b"pretend audio")

    def extract_one(filepath, extractor_config):
        return {
            "filepath": filepath,
            # Far outside every group's region of the space.
            "features": np.full(FEATURE_DIM, 500.0),
            "duration": 300.0,
            "file_info": {"filename": "outlier.mp3", "file_size": 1000},
            "checksum": filepath,
            "metadata": {},
        }

    monkeypatch.setattr(library, "_extract_one", extract_one)

    result = sorting.create_session(db, config, collection_row, [str(folder)], workers=1)

    # Forcing an outlier into the least-bad group would be worse than saying so.
    from music_cluster.sorter import STATUS_UNMATCHED

    assert result["summary"].get(STATUS_UNMATCHED) == 1
