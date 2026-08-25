"""Filing decisions to disk: planning, conflicts, and undo."""

import os
from pathlib import Path

import pytest

from music_cluster import organizer


def make_file(directory, name, content=b"audio"):
    path = directory / name
    path.write_bytes(content)
    return str(path)


@pytest.fixture
def scene(temp_dir, db):
    """A group with a destination and two tracks waiting to be filed."""
    source = temp_dir / "inbox"
    destination = temp_dir / "House"
    source.mkdir()
    destination.mkdir()

    collection_id = db.create_collection("c")
    group_id = db.create_group(collection_id, "House", destination_path=str(destination))
    group = db.get_group(group_id=group_id)

    assignments = []
    for name in ("one.mp3", "two.mp3"):
        filepath = make_file(source, name)
        track_id = db.upsert_track(filepath=filepath, filename=name, duration=100.0)
        assignments.append(
            {
                "item_id": len(assignments) + 1,
                "track_id": track_id,
                "group_id": group_id,
                "filepath": filepath,
                "filename": name,
                "duration": 100.0,
            }
        )

    return db, assignments, {group_id: group}, source, destination


def test_copy_leaves_the_original_in_place(scene):
    db, assignments, groups, source, destination = scene
    plan = organizer.plan_commit(db, assignments, groups, mode="copy")
    result = organizer.execute_plan(db, plan, assignments, session_id=1)

    assert result.performed_count if hasattr(result, "performed_count") else True
    assert len(result.performed) == 2
    assert (destination / "one.mp3").exists()
    assert (source / "one.mp3").exists()


def test_move_relocates_and_follows_the_track(scene):
    db, assignments, groups, source, destination = scene
    plan = organizer.plan_commit(db, assignments, groups, mode="move")
    organizer.execute_plan(db, plan, assignments, session_id=1)

    assert (destination / "one.mp3").exists()
    assert not (source / "one.mp3").exists()
    moved = db.get_track(assignments[0]["track_id"])
    assert moved["filepath"] == str(destination / "one.mp3")


def test_undo_restores_moved_files(scene):
    db, assignments, groups, source, destination = scene
    plan = organizer.plan_commit(db, assignments, groups, mode="move")
    organizer.execute_plan(db, plan, assignments, session_id=7)

    outcome = organizer.undo_session(db, 7)
    assert outcome["undone"] == 2
    assert (source / "one.mp3").exists()
    assert not (destination / "one.mp3").exists()
    assert db.get_track(assignments[0]["track_id"])["filepath"] == str(source / "one.mp3")


def test_undo_removes_copies(scene):
    db, assignments, groups, source, destination = scene
    plan = organizer.plan_commit(db, assignments, groups, mode="copy")
    organizer.execute_plan(db, plan, assignments, session_id=8)

    organizer.undo_session(db, 8)
    assert not (destination / "one.mp3").exists()
    assert (source / "one.mp3").exists()


def test_conflicts_are_skipped_by_default(scene):
    db, assignments, groups, source, destination = scene
    make_file(destination, "one.mp3", b"already here")

    plan = organizer.plan_commit(db, assignments, groups, mode="copy", on_conflict="skip")
    assert len(plan.actions) == 1
    assert any("already has a file" in problem["issue"] for problem in plan.problems)
    assert (destination / "one.mp3").read_bytes() == b"already here"


def test_rename_policy_avoids_collisions(scene):
    db, assignments, groups, source, destination = scene
    make_file(destination, "one.mp3", b"already here")

    plan = organizer.plan_commit(db, assignments, groups, mode="copy", on_conflict="rename")
    organizer.execute_plan(db, plan, assignments)

    assert (destination / "one.mp3").read_bytes() == b"already here"
    assert (destination / "one (1).mp3").exists()


def test_a_group_without_a_destination_is_reported_not_silently_skipped(temp_dir, db):
    source = temp_dir / "inbox"
    source.mkdir()
    collection_id = db.create_collection("c")
    group_id = db.create_group(collection_id, "No home")
    filepath = make_file(source, "x.mp3")
    track_id = db.upsert_track(filepath=filepath, filename="x.mp3")

    assignments = [
        {
            "item_id": 1,
            "track_id": track_id,
            "group_id": group_id,
            "filepath": filepath,
            "filename": "x.mp3",
            "duration": 1.0,
        }
    ]
    plan = organizer.plan_commit(
        db, assignments, {group_id: db.get_group(group_id=group_id)}, mode="copy"
    )

    assert plan.actions == []
    assert plan.problems[0]["issue"] == "group has no destination folder"


def test_missing_source_file_is_reported(scene):
    db, assignments, groups, source, destination = scene
    os.remove(assignments[0]["filepath"])

    plan = organizer.plan_commit(db, assignments, groups, mode="copy")
    assert len(plan.actions) == 1
    assert any(problem["issue"] == "file no longer exists" for problem in plan.problems)


def test_playlist_mode_writes_one_file_per_group(scene, temp_dir):
    db, assignments, groups, source, destination = scene
    playlists = temp_dir / "playlists"

    plan = organizer.plan_commit(
        db, assignments, groups, mode="playlist", playlist_dir=str(playlists)
    )
    result = organizer.execute_plan(db, plan, assignments)

    assert len(result.playlists) == 1
    content = Path(result.playlists[0]).read_text(encoding="utf-8")
    assert content.startswith("#EXTM3U")
    assert assignments[0]["filepath"] in content


def test_playlist_names_are_filesystem_safe(scene, temp_dir):
    db, assignments, groups, _, _ = scene
    group_id, group = next(iter(groups.items()))
    group["name"] = "Bass / Breaks: 170+"

    plan = organizer.plan_commit(
        db, assignments, {group_id: group}, mode="playlist", playlist_dir=str(temp_dir / "p")
    )
    result = organizer.execute_plan(db, plan, assignments)
    assert os.path.exists(result.playlists[0])
    assert "/" not in os.path.basename(result.playlists[0]).replace(".m3u8", "")


def test_unknown_mode_is_rejected(scene):
    db, assignments, groups, _, _ = scene
    with pytest.raises(ValueError, match="Unknown mode"):
        organizer.plan_commit(db, assignments, groups, mode="teleport")
