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


def test_audio_can_be_revalidated_rather_than_refetched(client, seeded, temp_dir):
    """Scrubbing issues a burst of range requests; re-downloading them is slow."""
    db, _, _ = seeded
    path = temp_dir / "clip.wav"
    path.write_bytes(b"0123456789" * 100)
    track_id = db.upsert_track(filepath=str(path), filename="clip.wav")

    response = client.get(f"/api/tracks/{track_id}/audio")
    assert response.headers["cache-control"] == "no-cache"
    assert response.headers["etag"]


class TestWaveformEndpoint:
    @pytest.fixture
    def track_id(self, seeded, temp_dir):
        pytest.importorskip("soundfile")
        import soundfile as sf

        db, _, _ = seeded
        path = temp_dir / "wave.wav"
        rate = 22050
        t = np.arange(rate * 4) / rate
        audio = np.sin(2 * np.pi * 220 * t) * np.concatenate(
            [np.ones(rate * 2), np.full(rate * 2, 0.2)]
        )
        sf.write(str(path), audio.astype(np.float32), rate)
        return db.upsert_track(filepath=str(path), filename="wave.wav")

    def test_returns_an_envelope_and_the_full_duration(self, client, track_id):
        payload = client.get(f"/api/tracks/{track_id}/waveform?samples=100").json()

        assert len(payload["peaks"]) == 100
        assert payload["duration"] == pytest.approx(4, abs=0.1)
        assert max(payload["peaks"]) <= 1.0
        assert np.mean(payload["peaks"][:40]) > np.mean(payload["peaks"][60:])

    def test_repeat_requests_revalidate(self, client, track_id):
        first = client.get(f"/api/tracks/{track_id}/waveform?samples=100")
        etag = first.headers["etag"]
        assert first.headers["cache-control"] == "private, max-age=86400"

        again = client.get(
            f"/api/tracks/{track_id}/waveform?samples=100", headers={"If-None-Match": etag}
        )
        assert again.status_code == 304
        assert again.content == b""

    def test_missing_track_and_missing_file_are_404s(self, client, seeded, temp_dir):
        db, _, _ = seeded
        assert client.get("/api/tracks/9999/waveform").status_code == 404

        gone = db.upsert_track(filepath=str(temp_dir / "gone.wav"), filename="gone.wav")
        assert client.get(f"/api/tracks/{gone}/waveform").status_code == 404

    def test_undecodable_audio_is_a_422(self, client, seeded, temp_dir):
        db, _, _ = seeded
        path = temp_dir / "broken.wav"
        path.write_bytes(b"not really a wav file")
        track_id = db.upsert_track(filepath=str(path), filename="broken.wav")

        assert client.get(f"/api/tracks/{track_id}/waveform").status_code == 422

    def test_resolution_is_validated(self, client, track_id):
        assert client.get(f"/api/tracks/{track_id}/waveform?samples=0").status_code == 422
        assert client.get(f"/api/tracks/{track_id}/waveform?samples=99999").status_code == 422


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


# ----------------------------------------------------------------------
# Audio profiles
# ----------------------------------------------------------------------


def test_profiles_are_advertised(client):
    body = client.get("/api/profiles").json()
    assert [profile["name"] for profile in body["profiles"]] == ["music", "sample"]
    categories = {entry["key"] for entry in body["sample_categories"]}
    assert {"kick", "snare", "clap", "hat_closed", "bass", "chord"} <= categories


def test_a_collection_can_be_created_for_samples(client):
    created = client.post("/api/collections", json={"name": "Drums", "profile": "sample"}).json()
    assert created["profile"] == "sample"
    assert client.get(f"/api/collections/{created['id']}").json()["collection"]["profile"] == "sample"


def test_a_collection_defaults_to_music(client):
    assert client.post("/api/collections", json={"name": "Records"}).json()["profile"] == "music"


def test_an_unknown_profile_is_a_client_error(client):
    response = client.post("/api/collections", json={"name": "Nope", "profile": "drums"})
    assert response.status_code == 400
    assert "Unknown audio profile" in response.json()["detail"]


def test_tracks_can_be_filtered_by_profile(client, rng):
    from music_cluster.config import Config
    from music_cluster.database import Database
    from music_cluster.features import build_layout

    db = Database(Config().get_db_path())
    music_id = db.upsert_track("/fake/track.mp3", "track.mp3")
    db.add_features(music_id, rng.normal(0, 1, FEATURE_DIM), "music")
    sample_id = db.upsert_track("/fake/kick.wav", "kick.wav")
    db.add_features(sample_id, rng.normal(0, 1, build_layout(20, "sample").dim), "sample")

    everything = client.get("/api/tracks").json()
    assert everything["total"] == 2

    only_samples = client.get("/api/tracks?profile=sample").json()
    assert only_samples["total"] == 1
    assert only_samples["tracks"][0]["filename"] == "kick.wav"

    assert client.get("/api/tracks?profile=drums").status_code == 400


def test_info_reports_what_was_analysed_under_each_profile(client, rng):
    from music_cluster.config import Config
    from music_cluster.database import Database

    db = Database(Config().get_db_path())
    track_id = db.upsert_track("/fake/one.mp3", "one.mp3")
    db.add_features(track_id, rng.normal(0, 1, FEATURE_DIM), "music")

    body = client.get("/api/info").json()
    assert body["analyzed_by_profile"] == {"music": 1}
    assert [profile["name"] for profile in body["profiles"]] == ["music", "sample"]


def test_the_sample_view_of_a_music_only_track_is_a_404(client, rng):
    from music_cluster.config import Config
    from music_cluster.database import Database

    db = Database(Config().get_db_path())
    track_id = db.upsert_track("/fake/one.mp3", "one.mp3")
    db.add_features(track_id, rng.normal(0, 1, FEATURE_DIM), "music")

    response = client.get(f"/api/tracks/{track_id}/sample")
    assert response.status_code == 404
    assert "not been analysed as a sample" in response.json()["detail"]


def test_a_sample_track_reports_its_category_and_measurements(client, rng):
    from music_cluster.config import Config
    from music_cluster.database import Database
    from music_cluster.extractor import FeatureExtractor
    from tests.integration.test_sample_profile import SAMPLE_RATE, kick

    pytest.importorskip("soundfile")
    config = Config()
    db = Database(config.get_db_path())

    import tempfile
    from pathlib import Path

    folder = Path(tempfile.mkdtemp())
    path = kick(folder / "kick_01.wav")
    extractor = FeatureExtractor(
        sample_rate=SAMPLE_RATE, frame_size=2048, hop_size=256,
        excerpt_seconds=0, profile="sample",
    )
    track_id = db.upsert_track(path, "kick_01.wav")
    db.add_features(track_id, extractor.extract(path), "sample")

    body = client.get(f"/api/tracks/{track_id}/sample").json()
    assert body["guess"]["category"] == "kick"
    assert body["descriptors"]["duration"] > 0
    assert body["descriptors"]["low_energy"] > body["descriptors"]["high_energy"]
