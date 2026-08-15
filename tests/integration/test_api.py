"""API surface: the routes the desktop UI depends on."""

import numpy as np
import pytest

from tests.conftest import FEATURE_DIM, make_centres, make_vector

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(temp_dir, monkeypatch):
    """A client bound to a throwaway database and config."""
    monkeypatch.setenv("MUSIC_CLUSTER_DB", str(temp_dir / "library.db"))
    monkeypatch.setenv("MUSIC_CLUSTER_CONFIG", str(temp_dir / "config.yaml"))

    from music_cluster.api import app

    return TestClient(app)


@pytest.fixture
def seeded(client, rng):
    """A collection with two fitted groups."""
    from music_cluster.config import Config
    from music_cluster.database import Database

    db = Database(Config().get_db_path())
    collection_id = db.create_collection("Test")
    centres = make_centres(rng, 2, separation=6.0)

    group_ids = []
    for index, centre in enumerate(centres):
        group_id = db.create_group(collection_id, f"Group {index}")
        for position in range(12):
            track_id = db.upsert_track(
                filepath=f"/fake/{index}/{position}.mp3",
                filename=f"{position}.mp3",
                metadata={"artist": f"A{index}", "title": f"T{position}"},
            )
            db.add_features(track_id, make_vector(rng, centre))
            db.add_group_members(group_id, [track_id])
        group_ids.append(group_id)

    return db, collection_id, group_ids


def wait_for(client, task_id, attempts=200):
    for _ in range(attempts):
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] != "running":
            return task
        import time

        time.sleep(0.05)
    raise AssertionError("task did not finish")


def test_info_reports_an_empty_library(client):
    response = client.get("/api/info")
    assert response.status_code == 200
    assert response.json()["track_count"] == 0


def test_collection_lifecycle(client):
    created = client.post("/api/collections", json={"name": "Crates"}).json()
    assert created["name"] == "Crates"

    duplicate = client.post("/api/collections", json={"name": "Crates"})
    assert duplicate.status_code == 409

    detail = client.get(f"/api/collections/{created['id']}").json()
    assert detail["groups"] == []

    assert client.delete(f"/api/collections/{created['id']}").status_code == 200
    assert client.get(f"/api/collections/{created['id']}").status_code == 404


def test_fit_and_check(client, seeded):
    _, collection_id, _ = seeded

    task_id = client.post(f"/api/collections/{collection_id}/fit").json()["task_id"]
    task = wait_for(client, task_id)
    assert task["status"] == "completed"
    assert task["result"]["accuracy"] > 0.9

    metrics = client.get(f"/api/collections/{collection_id}/check").json()["metrics"]
    assert metrics["n_groups"] == 2
    assert len(metrics["confusion"]) == 2


def test_check_before_fitting_is_a_404(client, seeded):
    _, collection_id, _ = seeded
    assert client.get(f"/api/collections/{collection_id}/check").status_code == 404


def test_group_membership_endpoints(client, seeded):
    db, collection_id, group_ids = seeded
    loose = db.upsert_track(filepath="/fake/loose.mp3", filename="loose.mp3")

    response = client.post(f"/api/groups/{group_ids[0]}/members", json={"track_ids": [loose]})
    assert response.json()["added"] == 1

    detail = client.get(f"/api/groups/{group_ids[0]}").json()
    assert detail["size"] == 13

    removed = client.delete(f"/api/groups/{group_ids[0]}/members/{loose}")
    assert removed.json()["removed"] is True


def test_unassigned_filter(client, seeded):
    db, collection_id, _ = seeded
    db.upsert_track(filepath="/fake/loose.mp3", filename="loose.mp3")

    response = client.get(f"/api/tracks?unassigned_in={collection_id}").json()
    assert [track["filename"] for track in response["tracks"]] == ["loose.mp3"]


def test_dry_run_commit_writes_nothing(client, seeded, rng, temp_dir):
    db, collection_id, group_ids = seeded
    client_task = client.post(f"/api/collections/{collection_id}/fit").json()["task_id"]
    wait_for(client, client_task)

    inbox = temp_dir / "inbox"
    inbox.mkdir()
    path = inbox / "new.mp3"
    path.write_bytes(b"audio")
    track_id = db.upsert_track(filepath=str(path), filename="new.mp3")
    db.add_features(track_id, make_vector(rng, np.zeros(FEATURE_DIM)))

    session_id = db.create_sort_session(collection_id, "s", str(inbox))
    db.upsert_sort_items(session_id, [{"track_id": track_id, "suggested_group_id": group_ids[0]}])
    item_id = db.list_sort_items(session_id)[0]["id"]
    client.put(f"/api/items/{item_id}", json={"group_id": group_ids[0]})

    response = client.post(
        f"/api/sessions/{session_id}/commit",
        json={"dry_run": True, "mode": "playlist", "playlist_dir": str(temp_dir / "pl")},
    ).json()

    assert response["dry_run"] is True
    assert response["plan"]["playlist_count"] == 1
    assert not (temp_dir / "pl").exists()


def test_audio_range_requests_are_supported(client, seeded, temp_dir):
    db, _, _ = seeded
    path = temp_dir / "clip.wav"
    path.write_bytes(b"0123456789" * 100)
    track_id = db.upsert_track(filepath=str(path), filename="clip.wav")

    response = client.get(f"/api/tracks/{track_id}/audio", headers={"Range": "bytes=10-19"})
    assert response.status_code == 206
    assert response.content == b"0123456789"
    assert response.headers["content-range"] == "bytes 10-19/1000"


def test_missing_resources_are_404s(client):
    assert client.get("/api/tracks/999").status_code == 404
    assert client.get("/api/groups/999").status_code == 404
    assert client.get("/api/sessions/999").status_code == 404
    assert client.get("/api/discovery/999").status_code == 404


def test_config_updates_persist(client):
    updated = client.put(
        "/api/config", json={"values": {"sorting.auto_accept_confidence": 0.85}}
    ).json()
    assert updated["sorting"]["auto_accept_confidence"] == 0.85
    assert client.get("/api/config").json()["sorting"]["auto_accept_confidence"] == 0.85
