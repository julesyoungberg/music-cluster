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


@pytest.mark.parametrize("mode", ["copy", "move", "symlink"])
def test_undo_puts_back_a_file_that_overwrite_replaced(scene, mode):
    db, assignments, groups, source, destination = scene
    make_file(destination, "one.mp3", b"the user's own file")

    plan = organizer.plan_commit(db, assignments, groups, mode=mode, on_conflict="overwrite")
    result = organizer.execute_plan(db, plan, assignments, session_id=9)
    assert not result.failures
    assert (destination / "one.mp3").read_bytes() == b"audio"

    outcome = organizer.undo_session(db, 9)

    assert not outcome["failures"]
    assert (destination / "one.mp3").read_bytes() == b"the user's own file"
    assert not (destination / "one.mp3").is_symlink()
    assert (source / "one.mp3").read_bytes() == b"audio"
    assert not (destination / "two.mp3").exists()
    backups = Path(db.db_path).parent / "backups"
    assert list(backups.iterdir()) == []


def test_overwrite_keeps_the_replaced_file_outside_the_destination(scene):
    db, assignments, groups, source, destination = scene
    make_file(destination, "one.mp3", b"the user's own file")

    plan = organizer.plan_commit(db, assignments, groups, mode="copy", on_conflict="overwrite")
    organizer.execute_plan(db, plan, assignments, session_id=10)

    backups = [action["backup_path"] for action in db.list_file_actions(10)]
    kept = [path for path in backups if path]
    assert len(kept) == 1
    assert Path(kept[0]).read_bytes() == b"the user's own file"
    assert Path(kept[0]).parent != destination
    assert sorted(os.listdir(destination)) == ["one.mp3", "two.mp3"]


def test_overwrite_says_the_replaced_file_is_kept_for_undo(scene):
    db, assignments, groups, source, destination = scene
    make_file(destination, "one.mp3", b"the user's own file")

    plan = organizer.plan_commit(db, assignments, groups, mode="copy", on_conflict="overwrite")

    notes = [action.note for action in plan.actions if action.note]
    assert notes == ["overwrites existing file, kept for undo"]


def test_undoing_an_older_session_first_is_refused_so_the_replaced_file_survives(scene):
    db, assignments, groups, _, destination = scene
    make_file(destination, "one.mp3", b"the user's own file")
    for session_id in (1, 2):
        plan = organizer.plan_commit(db, assignments, groups, mode="copy", on_conflict="overwrite")
        organizer.execute_plan(db, plan, assignments, session_id=session_id)

    refused = organizer.undo_session(db, 1)

    assert refused["undone"] == 0
    assert all("session 2" in failure["error"] for failure in refused["failures"])
    assert organizer.undo_session(db, 2)["failures"] == []
    assert organizer.undo_session(db, 1)["failures"] == []
    assert (destination / "one.mp3").read_bytes() == b"the user's own file"


def test_a_move_that_fails_after_placing_the_file_puts_both_files_back(scene, monkeypatch):
    db, assignments, groups, source, destination = scene
    make_file(destination, "one.mp3", b"the user's own file")

    def fail(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db, "update_track_path", fail)
    plan = organizer.plan_commit(db, assignments, groups, mode="move", on_conflict="overwrite")
    result = organizer.execute_plan(db, plan, assignments, session_id=3)

    assert len(result.failures) == 2
    assert not any("backup_path" in failure for failure in result.failures)
    assert (source / "one.mp3").read_bytes() == b"audio"
    assert (source / "two.mp3").read_bytes() == b"audio"
    assert (destination / "one.mp3").read_bytes() == b"the user's own file"
    assert os.listdir(destination) == ["one.mp3"]


def test_a_copy_that_fails_partway_leaves_the_replaced_file_in_place(scene, monkeypatch):
    db, assignments, groups, _, destination = scene
    make_file(destination, "one.mp3", b"the user's own file")

    def partial_copy(source, dest):
        Path(dest).write_bytes(b"aud")
        raise OSError("No space left on device")

    monkeypatch.setattr(organizer.shutil, "copy2", partial_copy)
    plan = organizer.plan_commit(db, assignments, groups, mode="copy", on_conflict="overwrite")
    result = organizer.execute_plan(db, plan, assignments, session_id=4)

    assert len(result.failures) == 2
    assert (destination / "one.mp3").read_bytes() == b"the user's own file"
    assert os.listdir(destination) == ["one.mp3"]


def test_a_backup_that_fails_leaves_the_file_where_it_was(scene, monkeypatch):
    db, assignments, groups, _, destination = scene
    make_file(destination, "one.mp3", b"the user's own file")

    def partial_move(source, dest):
        Path(dest).write_bytes(b"the u")  # a copy to another drive that ran out of space
        raise OSError("No space left on device")

    monkeypatch.setattr(organizer.shutil, "move", partial_move)
    plan = organizer.plan_commit(db, assignments, groups, mode="copy", on_conflict="overwrite")
    result = organizer.execute_plan(db, plan, assignments, session_id=5)

    assert [failure["dest_path"] for failure in result.failures] == [str(destination / "one.mp3")]
    assert "backup_path" not in result.failures[0]
    assert (destination / "one.mp3").read_bytes() == b"the user's own file"
    backups = Path(db.db_path).parent / "backups"
    assert not any(path.is_file() for path in backups.rglob("*"))


def test_overwrite_never_replaces_a_file_the_same_commit_filed(scene, temp_dir):
    db, assignments, groups, _, _ = scene
    elsewhere = temp_dir / "other"
    elsewhere.mkdir()
    twin = dict(assignments[0], item_id=3, filepath=make_file(elsewhere, "one.mp3"))
    twin["track_id"] = db.upsert_track(filepath=twin["filepath"], filename="one.mp3")

    plan = organizer.plan_commit(
        db, [*assignments, twin], groups, mode="move", on_conflict="overwrite"
    )

    assert [action.source_path for action in plan.actions] == [
        assignments[0]["filepath"],
        assignments[1]["filepath"],
    ]
    assert [problem["track_id"] for problem in plan.problems] == [twin["track_id"]]


def test_overwrite_never_replaces_a_folder(scene):
    db, assignments, groups, _, destination = scene
    (destination / "one.mp3").mkdir()

    plan = organizer.plan_commit(db, assignments, groups, mode="copy", on_conflict="overwrite")

    assert [action.source_path for action in plan.actions] == [assignments[1]["filepath"]]
    assert plan.problems[0]["issue"] == "a folder has this name"
