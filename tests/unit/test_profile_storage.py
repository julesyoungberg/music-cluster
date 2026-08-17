"""Features of different profiles must never be mixed, and v2 databases must open.

A music vector and a sample vector have different widths and different
meanings. Handing a sorter a matrix containing both would either crash on the
shape or, worse, not crash.
"""

import sqlite3

import numpy as np
import pytest

from music_cluster.database import SCHEMA_VERSION, Database
from music_cluster.features import build_layout


MUSIC_DIM = build_layout(20, "music").dim
SAMPLE_DIM = build_layout(20, "sample").dim


def test_one_track_can_hold_a_vector_per_profile(db, rng):
    track_id = db.upsert_track("/a/kick.wav", "kick.wav")
    music = rng.normal(0, 1, MUSIC_DIM)
    sample = rng.normal(0, 1, SAMPLE_DIM)

    db.add_features(track_id, music, "music")
    db.add_features(track_id, sample, "sample")

    assert len(db.get_features(track_id, "music")) == MUSIC_DIM
    assert len(db.get_features(track_id, "sample")) == SAMPLE_DIM
    np.testing.assert_allclose(db.get_features(track_id, "sample"), sample)


def test_features_default_to_music(db, rng):
    track_id = db.upsert_track("/a/track.mp3", "track.mp3")
    db.add_features(track_id, rng.normal(0, 1, MUSIC_DIM))
    assert db.get_features(track_id) is not None
    assert db.get_features(track_id, "sample") is None


def test_a_feature_matrix_holds_one_profile_only(db, rng):
    ids = []
    for index in range(4):
        track_id = db.upsert_track(f"/a/{index}.wav", f"{index}.wav")
        db.add_features(track_id, rng.normal(0, 1, MUSIC_DIM), "music")
        ids.append(track_id)
    # Only two of them were also analysed as samples.
    for track_id in ids[:2]:
        db.add_features(track_id, rng.normal(0, 1, SAMPLE_DIM), "sample")

    music_matrix, music_ids = db.get_feature_matrix(ids, "music")
    sample_matrix, sample_ids = db.get_feature_matrix(ids, "sample")

    assert music_matrix.shape == (4, MUSIC_DIM)
    assert sample_matrix.shape == (2, SAMPLE_DIM)
    assert set(sample_ids) == set(ids[:2])
    assert set(music_ids) == set(ids)


def test_counts_are_reported_per_profile(db, rng):
    for index in range(3):
        track_id = db.upsert_track(f"/a/{index}.wav", f"{index}.wav")
        db.add_features(track_id, rng.normal(0, 1, MUSIC_DIM), "music")
        if index == 0:
            db.add_features(track_id, rng.normal(0, 1, SAMPLE_DIM), "sample")

    assert db.count_features() == 4
    assert db.count_features("music") == 3
    assert db.count_features("sample") == 1
    assert db.count_features_by_profile() == {"music": 3, "sample": 1}
    assert db.count_tracks() == 3
    assert db.count_tracks("sample") == 1


def test_tracks_can_be_listed_by_profile(db, rng):
    for index in range(3):
        track_id = db.upsert_track(f"/a/{index}.wav", f"{index}.wav")
        db.add_features(track_id, rng.normal(0, 1, SAMPLE_DIM if index else MUSIC_DIM),
                        "sample" if index else "music")

    assert len(db.list_tracks(profile="sample")) == 2
    assert len(db.list_tracks(profile="music")) == 1
    assert len(db.list_tracks()) == 3


def test_missing_features_are_reported_per_profile(db, rng):
    analysed = db.upsert_track("/a/one.wav", "one.wav")
    bare = db.upsert_track("/a/two.wav", "two.wav")
    db.add_features(analysed, rng.normal(0, 1, SAMPLE_DIM), "sample")

    assert db.tracks_without_features([analysed, bare], "sample") == [bare]
    # Nothing has been analysed as music at all.
    assert db.tracks_without_features([analysed, bare], "music") == [analysed, bare]


def test_collections_carry_a_profile(db):
    music_id = db.create_collection("Records")
    sample_id = db.create_collection("Drums", profile="sample")

    assert db.get_collection(collection_id=music_id)["profile"] == "music"
    assert db.get_collection(collection_id=sample_id)["profile"] == "sample"


def test_an_unknown_collection_profile_is_rejected(db):
    with pytest.raises(ValueError, match="Unknown audio profile"):
        db.create_collection("Nonsense", profile="drums")


def test_a_collections_profile_cannot_be_changed_by_update(db):
    """It is fixed by construction: every stored vector already assumes it."""
    collection_id = db.create_collection("Drums", profile="sample")
    db.update_collection(collection_id, profile="music", name="Drums renamed")

    collection = db.get_collection(collection_id=collection_id)
    assert collection["profile"] == "sample"
    assert collection["name"] == "Drums renamed"


def test_a_v2_database_opens_and_keeps_its_vectors(temp_dir, rng):
    """Analysis is expensive; upgrading must not throw a library away."""
    path = str(temp_dir / "v2.db")
    vector = rng.normal(0, 1, MUSIC_DIM)

    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filepath TEXT UNIQUE NOT NULL,
            filename TEXT NOT NULL,
            duration REAL, file_size INTEGER, checksum TEXT,
            analyzed_at TIMESTAMP, analysis_version TEXT,
            title TEXT, artist TEXT, album TEXT, genre TEXT,
            year INTEGER, tag_bpm REAL, tag_key TEXT, comment TEXT
        );
        CREATE TABLE features (
            track_id INTEGER PRIMARY KEY,
            feature_vector BLOB NOT NULL,
            feature_dim INTEGER
        );
        CREATE TABLE collections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE NOT NULL, description TEXT, settings TEXT,
            created_at TIMESTAMP, updated_at TIMESTAMP
        );
        CREATE TABLE discovery_runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT, source_path TEXT, algorithm TEXT,
            params TEXT, metrics TEXT, status TEXT, created_at TIMESTAMP
        );
        CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO schema_meta VALUES ('version', '2');
        INSERT INTO tracks (filepath, filename) VALUES ('/a/one.mp3', 'one.mp3');
        INSERT INTO collections (name, settings) VALUES ('Old', '{}');
        """
    )
    conn.execute(
        "INSERT INTO features (track_id, feature_vector, feature_dim) VALUES (1, ?, ?)",
        (np.asarray(vector, dtype=np.float64).tobytes(), MUSIC_DIM),
    )
    conn.commit()
    conn.close()

    db = Database(path)

    # Everything that existed was music, and still is.
    np.testing.assert_allclose(db.get_features(1, "music"), vector)
    assert db.get_collection(name="Old")["profile"] == "music"
    assert db.count_features_by_profile() == {"music": 1}

    # ...and the upgraded table now accepts a second profile for that track.
    db.add_features(1, np.zeros(SAMPLE_DIM), "sample")
    assert db.count_features() == 2

    with db.connection() as conn:
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()[0]
    assert int(version) == SCHEMA_VERSION


def test_opening_a_current_database_twice_is_a_no_op(temp_dir, rng):
    path = str(temp_dir / "twice.db")
    first = Database(path)
    track_id = first.upsert_track("/a/one.wav", "one.wav")
    first.add_features(track_id, rng.normal(0, 1, SAMPLE_DIM), "sample")

    second = Database(path)
    assert second.count_features_by_profile() == {"sample": 1}
