"""The HTTP API, end to end, against real audio.

This is the path the desktop UI takes: import folders, fit, sort a batch of new
buys, decide, commit. Background tasks are polled the way the UI polls them.
"""

import time

import pytest

from tests.e2e.conftest import genre_folder, new_music


pytestmark = pytest.mark.e2e

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(workspace):
    from music_cluster.api import app

    return TestClient(app)


def finish(client, response, attempts=600):
    """Poll a background task the way the UI does, and return the finished task."""
    assert response.status_code == 200, response.text
    task_id = response.json()["task_id"]
    for _ in range(attempts):
        task = client.get(f"/api/tasks/{task_id}").json()
        if task["status"] != "running":
            assert task["status"] == "completed", task
            return task
        time.sleep(0.05)
    raise AssertionError(f"task {task_id} never finished")


@pytest.fixture
def collection(client):
    response = client.post("/api/collections", json={"name": "Crates"})
    assert response.status_code == 200, response.text
    return response.json()


@pytest.fixture
def fitted(client, collection, music):
    """Three imported genre folders, fitted and ready to sort against."""
    finish(
        client,
        client.post(
            f"/api/collections/{collection['id']}/import-tree",
            json={"path": str(music / "DJ"), "workers": 1},
        ),
    )
    finish(client, client.post(f"/api/collections/{collection['id']}/fit"))
    return collection


class TestInfo:
    def test_root_identifies_the_api(self, client):
        body = client.get("/").json()
        assert "Music Cluster" in body["name"] or "music" in str(body).lower()

    def test_info_reports_an_empty_library_before_anything_is_imported(self, client):
        body = client.get("/api/info").json()

        assert body["track_count"] == 0
        assert body["collections"] == []

    def test_info_reflects_what_has_been_imported(self, client, fitted):
        body = client.get("/api/info").json()

        assert body["track_count"] == 24
        assert body["analyzed_count"] == 24
        assert [c["name"] for c in body["collections"]] == ["Crates"]


class TestBrowse:
    def test_lists_subdirectories_and_counts_audio(self, client, music):
        body = client.get("/api/browse", params={"path": str(music / "DJ")}).json()

        assert {entry["name"] for entry in body["directories"]} == {
            "Deep House",
            "Techno",
            "Drum & Bass",
        }

    def test_reports_how_many_tracks_a_folder_holds(self, client, music):
        body = client.get("/api/browse", params={"path": genre_folder(music, "Techno")}).json()

        assert body["audio_file_count"] == 8

    def test_a_path_that_is_not_a_directory_is_rejected(self, client, music):
        response = client.get("/api/browse", params={"path": str(music / "nope")})

        assert response.status_code == 400
        assert "Not a directory" in response.json()["detail"]


class TestImportAndFit:
    def test_importing_a_tree_creates_one_group_per_folder(self, client, collection, music):
        task = finish(
            client,
            client.post(
                f"/api/collections/{collection['id']}/import-tree",
                json={"path": str(music / "DJ"), "workers": 1},
            ),
        )

        assert len(task["result"]["groups"]) == 3
        groups = client.get(f"/api/collections/{collection['id']}/groups").json()["groups"]
        assert {group["name"] for group in groups} == {"Deep House", "Techno", "Drum & Bass"}

    def test_importing_one_folder_creates_one_group(self, client, collection, music):
        finish(
            client,
            client.post(
                f"/api/collections/{collection['id']}/import-folder",
                json={"path": genre_folder(music, "Techno"), "workers": 1},
            ),
        )

        groups = client.get(f"/api/collections/{collection['id']}/groups").json()["groups"]
        assert [group["name"] for group in groups] == ["Techno"]
        assert groups[0]["size"] == 8

    def test_fitting_reports_per_group_accuracy(self, client, fitted):
        body = client.get(f"/api/collections/{fitted['id']}/check").json()

        assert body["metrics"]["accuracy"] > 0.8
        assert len(body["metrics"]["per_group"]) == 3

    def test_a_group_detail_lists_its_members(self, client, fitted):
        groups = client.get(f"/api/collections/{fitted['id']}/groups").json()["groups"]
        body = client.get(f"/api/groups/{groups[0]['id']}").json()

        assert body["size"] == 8
        assert len(body["members"]) == 8
        assert body["members"][0]["artist"]

    def test_a_group_can_be_renamed_and_redirected(self, client, fitted, workspace):
        groups = client.get(f"/api/collections/{fitted['id']}/groups").json()["groups"]
        target = str(workspace / "elsewhere")

        response = client.put(
            f"/api/groups/{groups[0]['id']}",
            json={"name": "Renamed", "destination_path": target},
        )

        assert response.status_code == 200
        assert response.json()["name"] == "Renamed"
        assert response.json()["destination_path"] == target

    def test_a_member_can_be_removed_from_a_group(self, client, fitted):
        groups = client.get(f"/api/collections/{fitted['id']}/groups").json()["groups"]
        detail = client.get(f"/api/groups/{groups[0]['id']}").json()
        track_id = detail["members"][0]["id"]

        response = client.delete(f"/api/groups/{groups[0]['id']}/members/{track_id}")

        assert response.status_code == 200
        assert client.get(f"/api/groups/{groups[0]['id']}").json()["size"] == 7

    def test_tracks_can_be_added_to_a_group_by_filepath(self, client, fitted, music):
        groups = client.get(f"/api/collections/{fitted['id']}/groups").json()["groups"]
        newcomer = sorted(str(p) for p in (music / "NewMusic").glob("*.flac"))[0]

        response = client.post(
            f"/api/groups/{groups[0]['id']}/members",
            json={"filepaths": [newcomer]},
        )

        assert response.status_code == 200
        assert response.json()["added"] == 1

    def test_deleting_a_group_leaves_the_others(self, client, fitted):
        groups = client.get(f"/api/collections/{fitted['id']}/groups").json()["groups"]

        client.delete(f"/api/groups/{groups[0]['id']}")

        remaining = client.get(f"/api/collections/{fitted['id']}/groups").json()["groups"]
        assert len(remaining) == 2

    def test_exporting_writes_one_playlist_per_group(self, client, fitted, workspace):
        output = workspace / "exported"

        response = client.post(
            f"/api/collections/{fitted['id']}/export", params={"output": str(output)}
        )

        assert response.status_code == 200
        assert len(list(output.glob("*.m3u8"))) == 3


class TestTracks:
    def test_lists_tracks_with_their_metadata(self, client, fitted):
        body = client.get("/api/tracks", params={"limit": 5}).json()

        assert len(body["tracks"]) == 5
        assert body["tracks"][0]["artist"]
        assert body["tracks"][0]["display"]

    def test_search_narrows_the_list(self, client, fitted):
        body = client.get("/api/tracks", params={"query": "Techno Artist"}).json()

        assert body["tracks"]
        assert all("Techno" in track["artist"] for track in body["tracks"])

    def test_a_track_detail_includes_its_group_memberships(self, client, fitted):
        track_id = client.get("/api/tracks", params={"limit": 1}).json()["tracks"][0]["id"]

        body = client.get(f"/api/tracks/{track_id}").json()

        assert body["id"] == track_id
        assert body["groups"]
        assert body["exists"] is True

    def test_an_unknown_track_is_a_404(self, client, fitted):
        assert client.get("/api/tracks/999999").status_code == 404


class TestAudioAndWaveform:
    def test_audio_streams_with_a_content_type_a_browser_can_play(self, client, fitted):
        track_id = client.get("/api/tracks", params={"limit": 1}).json()["tracks"][0]["id"]

        response = client.get(f"/api/tracks/{track_id}/audio")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("audio/")
        assert len(response.content) > 0

    def test_a_range_request_returns_partial_content(self, client, fitted):
        track_id = client.get("/api/tracks", params={"limit": 1}).json()["tracks"][0]["id"]

        response = client.get(f"/api/tracks/{track_id}/audio", headers={"Range": "bytes=0-1023"})

        # Seeking in the player bar depends on this.
        assert response.status_code == 206
        assert response.headers["content-range"].startswith("bytes 0-1023/")
        assert len(response.content) == 1024

    def test_a_waveform_comes_back_as_peaks(self, client, fitted):
        track_id = client.get("/api/tracks", params={"limit": 1}).json()["tracks"][0]["id"]

        body = client.get(f"/api/tracks/{track_id}/waveform", params={"samples": 120}).json()

        assert len(body["peaks"]) == 120
        assert max(body["peaks"]) > 0
        assert body["duration"] > 0

    def test_the_second_request_for_a_waveform_is_served_from_cache(self, client, fitted):
        track_id = client.get("/api/tracks", params={"limit": 1}).json()["tracks"][0]["id"]

        first = client.get(f"/api/tracks/{track_id}/waveform", params={"samples": 120}).json()
        second = client.get(f"/api/tracks/{track_id}/waveform", params={"samples": 120}).json()

        assert first["peaks"] == second["peaks"]

    def test_a_waveform_for_an_unknown_track_is_a_404(self, client, fitted):
        assert client.get("/api/tracks/999999/waveform").status_code == 404

    def test_artwork_is_no_content_when_the_file_carries_none(self, client, fitted):
        track_id = client.get("/api/tracks", params={"limit": 1}).json()["tracks"][0]["id"]

        # 204 rather than 404: the track exists, it just has no cover art, and
        # the UI shows a placeholder rather than an error for that.
        assert client.get(f"/api/tracks/{track_id}/artwork").status_code == 204


class TestSortSession:
    @pytest.fixture
    def session(self, client, fitted, music):
        response = client.post(
            "/api/sessions",
            json={
                "collection_id": fitted["id"],
                "paths": [new_music(music)],
                "name": "Friday buys",
                "workers": 1,
            },
        )
        return finish(client, response)["result"]["session"]

    def test_creating_a_session_scores_every_incoming_track(self, client, session):
        body = client.get(f"/api/sessions/{session['id']}").json()

        assert body["total"] == 6
        assert body["session"]["name"] == "Friday buys"

    def test_items_carry_a_ranked_list_of_groups(self, client, session):
        items = client.get(f"/api/sessions/{session['id']}/items").json()["items"]

        assert len(items) == 6
        assert items[0]["ranked"]
        assert items[0]["ranked"][0]["group_name"]
        assert 0 <= items[0]["confidence"] <= 1

    def test_a_track_is_suggested_the_group_it_sounds_like(self, client, session, fitted):
        items = client.get(f"/api/sessions/{session['id']}/items").json()["items"]

        # Each incoming track was synthesised from one of the three families.
        for item in items:
            expected = item["title"].replace("New ", "").rsplit(" ", 1)[0]
            assert item["ranked"][0]["group_name"] == expected

    def test_deciding_an_item_records_the_group(self, client, session, fitted):
        items = client.get(f"/api/sessions/{session['id']}/items").json()["items"]
        groups = client.get(f"/api/collections/{fitted['id']}/groups").json()["groups"]

        response = client.put(f"/api/items/{items[0]['id']}", json={"group_id": groups[1]["id"]})

        assert response.status_code == 200
        assert response.json()["final_group_id"] == groups[1]["id"]

    def test_an_item_can_be_skipped(self, client, session):
        items = client.get(f"/api/sessions/{session['id']}/items").json()["items"]

        response = client.put(f"/api/items/{items[0]['id']}", json={"skip": True})

        assert response.json()["status"] == "skipped"

    def test_items_can_be_decided_in_bulk(self, client, session, fitted):
        items = client.get(f"/api/sessions/{session['id']}/items").json()["items"]
        groups = client.get(f"/api/collections/{fitted['id']}/groups").json()["groups"]

        response = client.put(
            f"/api/sessions/{session['id']}/items",
            json={"item_ids": [item["id"] for item in items[:3]], "group_id": groups[0]["id"]},
        )

        assert response.json()["updated"] == 3

    def test_accept_confident_files_the_easy_ones(self, client, session):
        response = client.post(f"/api/sessions/{session['id']}/accept-confident")

        assert response.status_code == 200
        assert response.json()["accepted"] > 0

    def test_a_dry_run_commit_reports_the_plan_without_writing(self, client, session, workspace):
        client.post(f"/api/sessions/{session['id']}/accept-confident")
        playlists = workspace / "playlists"

        body = client.post(
            f"/api/sessions/{session['id']}/commit",
            json={"mode": "playlist", "playlist_dir": str(playlists), "dry_run": True},
        ).json()

        assert body["dry_run"] is True
        assert body["plan"]["playlists"]
        assert not playlists.exists()

    def test_committing_writes_the_playlists(self, client, session, workspace):
        client.post(f"/api/sessions/{session['id']}/accept-confident")
        playlists = workspace / "playlists"

        body = client.post(
            f"/api/sessions/{session['id']}/commit",
            json={"mode": "playlist", "playlist_dir": str(playlists)},
        ).json()

        assert body["result"]["playlists"]
        assert list(playlists.glob("*.m3u8"))

    def test_committing_in_copy_mode_writes_files_and_can_be_undone(self, client, session, music):
        client.post(f"/api/sessions/{session['id']}/accept-confident")

        committed = client.post(
            f"/api/sessions/{session['id']}/commit", json={"mode": "copy"}
        ).json()
        assert committed["result"]["performed_count"] > 0

        undone = client.post(f"/api/sessions/{session['id']}/undo").json()
        assert undone["undone"] == committed["result"]["performed_count"]

    def test_committing_folds_tracks_back_into_their_groups(self, client, session, fitted):
        before = sum(
            group["size"]
            for group in client.get(f"/api/collections/{fitted['id']}/groups").json()["groups"]
        )
        client.post(f"/api/sessions/{session['id']}/accept-confident")

        body = client.post(
            f"/api/sessions/{session['id']}/commit", json={"mode": "playlist"}
        ).json()

        assert body["learned"] > 0
        after = sum(
            group["size"]
            for group in client.get(f"/api/collections/{fitted['id']}/groups").json()["groups"]
        )
        assert after > before

    def test_rescoring_a_session_refreshes_its_suggestions(self, client, session):
        finish(client, client.post(f"/api/sessions/{session['id']}/rescore"))

        items = client.get(f"/api/sessions/{session['id']}/items").json()["items"]
        assert len(items) == 6

    def test_a_session_can_be_deleted(self, client, session):
        assert client.delete(f"/api/sessions/{session['id']}").status_code == 200
        assert client.get(f"/api/sessions/{session['id']}").status_code == 404

    def test_an_unknown_session_is_a_404(self, client, fitted):
        assert client.get("/api/sessions/999999").status_code == 404


class TestDiscovery:
    @pytest.fixture
    def run(self, client, collection, music):
        response = client.post(
            "/api/discovery",
            json={
                "paths": [str(music / "DJ")],
                "name": "The pile",
                "algorithm": "kmeans",
                "target_groups": 3,
                "min_group_size": 4,
                "workers": 1,
            },
        )
        return finish(client, response)["result"]["run"]

    def test_a_run_proposes_candidates_with_names_and_exemplars(self, client, run):
        body = client.get(f"/api/discovery/{run['id']}").json()

        assert len(body["candidates"]) == 3
        for candidate in body["candidates"]:
            assert candidate["suggested_name"]
            assert candidate["exemplar_tracks"]

    def test_candidate_members_can_be_listed_for_audition(self, client, run):
        candidate = client.get(f"/api/discovery/{run['id']}").json()["candidates"][0]

        body = client.get(f"/api/discovery/candidates/{candidate['id']}/members").json()

        assert body["total"] == candidate["size"]
        assert len(body["tracks"]) == candidate["size"]

    def test_promoting_a_candidate_creates_a_group(self, client, run, collection):
        candidate = client.get(f"/api/discovery/{run['id']}").json()["candidates"][0]

        body = client.post(
            f"/api/discovery/candidates/{candidate['id']}/promote",
            json={"collection_id": collection["id"], "name": "Rollers"},
        ).json()

        assert body["group"]["name"] == "Rollers"
        assert body["added"] == candidate["size"]

    def test_rejecting_a_candidate_creates_nothing(self, client, run, collection):
        candidate = client.get(f"/api/discovery/{run['id']}").json()["candidates"][0]

        client.post(f"/api/discovery/candidates/{candidate['id']}/reject")

        groups = client.get(f"/api/collections/{collection['id']}/groups").json()["groups"]
        assert groups == []

    def test_a_run_can_be_deleted(self, client, run):
        assert client.delete(f"/api/discovery/{run['id']}").status_code == 200
        assert client.get(f"/api/discovery/{run['id']}").status_code == 404


class TestSimilar:
    def test_finds_tracks_that_sound_like_a_seed(self, client, fitted):
        tracks = client.get("/api/tracks", params={"query": "Techno Artist"}).json()["tracks"]

        body = client.post(
            "/api/similar", json={"seed_track_ids": [tracks[0]["id"]], "limit": 5}
        ).json()

        assert len(body["results"]) == 5
        # The nearest neighbours of a Techno track should be Techno tracks.
        assert sum(1 for r in body["results"] if "Techno" in r["track"]["artist"]) >= 4

    def test_needs_a_seed(self, client, fitted):
        assert client.post("/api/similar", json={"seed_track_ids": []}).status_code == 400


class TestConfig:
    def test_reads_the_current_configuration(self, client):
        body = client.get("/api/config").json()
        assert "sorting" in body

    def test_updating_a_setting_persists_it(self, client):
        client.put("/api/config", json={"values": {"sorting.neighbors": 9}})

        assert client.get("/api/config").json()["sorting"]["neighbors"] == 9


class TestPrune:
    def test_removes_tracks_whose_files_are_gone(self, client, fitted, music):
        import os

        os.remove(next(iter((music / "DJ" / "Techno").glob("*.flac"))))

        body = client.post("/api/prune").json()

        assert body["removed"] == 1
        assert client.get("/api/info").json()["track_count"] == 23


class TestErrors:
    def test_analysing_a_path_that_does_not_exist_fails_the_task(self, client, workspace):
        response = client.post(
            "/api/analyze", json={"paths": [str(workspace / "nope")], "workers": 1}
        )
        task_id = response.json()["task_id"]

        for _ in range(200):
            task = client.get(f"/api/tasks/{task_id}").json()
            if task["status"] != "running":
                break
            time.sleep(0.05)

        assert task["status"] == "failed"
        assert task["error"]

    def test_an_unknown_task_is_a_404(self, client):
        assert client.get("/api/tasks/not-a-task").status_code == 404

    def test_fitting_a_collection_with_one_group_fails_with_a_reason(
        self, client, collection, music
    ):
        finish(
            client,
            client.post(
                f"/api/collections/{collection['id']}/import-folder",
                json={"path": genre_folder(music, "Techno"), "workers": 1},
            ),
        )

        response = client.post(f"/api/collections/{collection['id']}/fit")
        task_id = response.json()["task_id"]
        for _ in range(200):
            task = client.get(f"/api/tasks/{task_id}").json()
            if task["status"] != "running":
                break
            time.sleep(0.05)

        assert task["status"] == "failed"
        assert "two groups" in task["error"]

    def test_an_unknown_collection_is_a_404(self, client):
        assert client.get("/api/collections/999999").status_code == 404
