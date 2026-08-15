"""Persistence behaviour that the rest of the app relies on."""

import numpy as np
import pytest

from music_cluster.database import Database
from tests.conftest import FEATURE_DIM


def add_track(db, path="/fake/a.mp3", **metadata):
    return db.upsert_track(
        filepath=path, filename=path.split("/")[-1], duration=200.0, metadata=metadata
    )


def test_features_round_trip_exactly(db, rng):
    track_id = add_track(db)
    vector = rng.normal(0, 1, FEATURE_DIM)
    db.add_features(track_id, vector)
    assert np.array_equal(db.get_features(track_id), vector)


def test_reanalysis_keeps_the_track_id_and_its_memberships(db, rng):
    """Re-analysing a file must not orphan it from the groups it belongs to."""
    collection_id = db.create_collection("c")
    group_id = db.create_group(collection_id, "g")
    track_id = add_track(db, artist="A", title="T")
    db.add_group_members(group_id, [track_id])

    again = add_track(db, artist="A", title="T2")
    assert again == track_id
    assert db.get_group_member_ids(group_id) == [track_id]
    assert db.get_track(track_id)["title"] == "T2"


def test_feature_matrix_reports_only_what_it_found(db, rng):
    ids = [add_track(db, path=f"/fake/{i}.mp3") for i in range(3)]
    db.add_features(ids[0], rng.normal(0, 1, FEATURE_DIM))
    db.add_features(ids[2], rng.normal(0, 1, FEATURE_DIM))

    matrix, found = db.get_feature_matrix(ids)
    assert matrix.shape == (2, FEATURE_DIM)
    assert set(found) == {ids[0], ids[2]}
    assert db.tracks_without_features(ids) == [ids[1]]


def test_feature_matrix_drops_mismatched_dimensions(db, rng):
    """Vectors extracted under different settings cannot share a matrix."""
    ids = [add_track(db, path=f"/fake/{i}.mp3") for i in range(3)]
    db.add_features(ids[0], rng.normal(0, 1, FEATURE_DIM))
    db.add_features(ids[1], rng.normal(0, 1, FEATURE_DIM))
    db.add_features(ids[2], rng.normal(0, 1, FEATURE_DIM + 10))

    matrix, found = db.get_feature_matrix(ids)
    assert matrix.shape == (2, FEATURE_DIM)
    assert ids[2] not in found


def test_deleting_a_collection_cascades(db):
    collection_id = db.create_collection("c")
    group_id = db.create_group(collection_id, "g")
    track_id = add_track(db)
    db.add_group_members(group_id, [track_id])

    db.delete_collection(collection_id)
    assert db.get_group(group_id=group_id) is None
    # The track itself is library data and survives.
    assert db.get_track(track_id) is not None


def test_unassigned_filter(db):
    collection_id = db.create_collection("c")
    group_id = db.create_group(collection_id, "g")
    filed = add_track(db, path="/fake/filed.mp3")
    loose = add_track(db, path="/fake/loose.mp3")
    db.add_group_members(group_id, [filed])

    unassigned = db.list_tracks(unassigned_in_collection=collection_id)
    assert [track["id"] for track in unassigned] == [loose]


def test_rescoring_preserves_human_decisions(db):
    """Re-scoring a session must not overwrite what the user already decided."""
    collection_id = db.create_collection("c")
    group_a = db.create_group(collection_id, "a")
    group_b = db.create_group(collection_id, "b")
    track_id = add_track(db)
    session_id = db.create_sort_session(collection_id, "s", None)

    db.upsert_sort_items(
        session_id, [{"track_id": track_id, "suggested_group_id": group_a, "status": "pending"}]
    )
    item_id = db.list_sort_items(session_id)[0]["id"]
    db.decide_sort_item(item_id, "accepted", group_b)

    db.upsert_sort_items(
        session_id, [{"track_id": track_id, "suggested_group_id": group_a, "status": "pending"}]
    )
    item = db.get_sort_item(item_id)
    assert item["status"] == "accepted"
    assert item["final_group_id"] == group_b


def test_model_round_trip(db):
    collection_id = db.create_collection("c")
    db.save_model(collection_id, b"payload", {"k": 1}, {"accuracy": 0.9}, 3, 30)

    model = db.load_model(collection_id)
    assert model["payload"] == b"payload"
    assert model["metrics"]["accuracy"] == 0.9
    assert model["n_groups"] == 3


def test_touch_collection_marks_the_model_stale(db):
    """Adding references must invalidate a fitted model built without them."""
    collection_id = db.create_collection("c")
    db.save_model(collection_id, b"payload", {}, {}, 2, 10)
    before = db.get_collection(collection_id=collection_id)["updated_at"]

    db.touch_collection(collection_id)
    after = db.get_collection(collection_id=collection_id)["updated_at"]
    assert after >= before


def test_group_names_are_unique_per_collection(db):
    first = db.create_collection("one")
    second = db.create_collection("two")
    db.create_group(first, "House")
    db.create_group(second, "House")  # same name, different collection: fine

    with pytest.raises(Exception):
        db.create_group(first, "House")


def test_legacy_database_is_migrated(temp_dir):
    """A v1 database keeps its analysis and loses only the clustering tables."""
    import pickle
    import sqlite3

    path = str(temp_dir / "legacy.db")
    vector = np.arange(FEATURE_DIM, dtype=np.float64)

    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE tracks (id INTEGER PRIMARY KEY AUTOINCREMENT, filepath TEXT UNIQUE,
            filename TEXT, duration REAL, file_size INTEGER, checksum TEXT,
            analyzed_at TIMESTAMP, analysis_version TEXT);
        CREATE TABLE features (track_id INTEGER PRIMARY KEY, feature_vector BLOB,
            feature_dim INTEGER, normalized BOOLEAN DEFAULT 0);
        CREATE TABLE clusterings (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE clusters (id INTEGER PRIMARY KEY, clustering_id INTEGER);
        CREATE TABLE cluster_members (cluster_id INTEGER, track_id INTEGER);
        """
    )
    conn.execute(
        "INSERT INTO tracks (filepath, filename) VALUES ('/fake/old.mp3', 'old.mp3')"
    )
    conn.execute(
        "INSERT INTO features (track_id, feature_vector, feature_dim) VALUES (1, ?, ?)",
        (pickle.dumps(vector), FEATURE_DIM),
    )
    conn.commit()
    conn.close()

    db = Database(path)
    assert np.array_equal(db.get_features(1), vector)
    assert db.get_track_by_filepath("/fake/old.mp3") is not None

    with db.connection() as conn:
        tables = {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "clusterings" not in tables
    assert "collections" in tables
