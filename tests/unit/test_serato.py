"""Serato crates: the file format, drive-relative paths, and the write/remove safety rules."""

import os
import struct

import pytest

from music_cluster import serato
from music_cluster.djexport import ExportConflict


@pytest.fixture
def crates(temp_dir, db):
    """Two groups on the same drive as the Serato folder, plus an empty group."""
    collection_id = db.create_collection("Crates")
    house = db.create_group(collection_id, "House")
    techno = db.create_group(collection_id, "Techno")
    db.create_group(collection_id, "Empty")
    for group_id, name in ((house, "a.mp3"), (techno, "b.mp3")):
        path = temp_dir / "music" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b"audio")
        db.add_group_members(group_id, [db.upsert_track(filepath=str(path), filename=name)])
    return db, db.get_collection(collection_id=collection_id), temp_dir / "_Serato_"


def test_a_crate_is_tagged_records_with_utf16_text():
    data = serato.build_crate(["Users/dj/Mü sic/a.mp3"])

    assert data[:4] == b"vrsn"
    (length,) = struct.unpack(">I", data[4:8])
    assert data[8 : 8 + length].decode("utf-16-be") == "1.0/Serato ScratchLive Crate"
    assert serato.read_crate(data) == ["Users/dj/Mü sic/a.mp3"]


@pytest.mark.parametrize(
    "track, serato_dir, expected",
    [
        ("/Users/dj/a.mp3", "/Users/dj/Music/_Serato_", "Users/dj/a.mp3"),
        ("/Volumes/USB/House/a.mp3", "/Volumes/USB/_Serato_", "House/a.mp3"),
        ("/Volumes/USB/a.mp3", "/Users/dj/Music/_Serato_", None),
        ("/Users/dj/a.mp3", "/Volumes/USB/_Serato_", None),
        ("C:\\Music\\a.mp3", "C:\\Users\\dj\\Music\\_Serato_", "Music/a.mp3"),
        ("D:\\a.mp3", "C:\\Users\\dj\\Music\\_Serato_", None),
    ],
)
def test_tracks_are_stored_relative_to_the_serato_folders_drive(track, serato_dir, expected):
    assert serato.serato_track_path(track, serato_dir) == expected


def test_a_dry_run_lists_the_crates_and_writes_nothing(crates):
    db, collection, serato_dir = crates

    planned = serato.write_crates(db, collection, str(serato_dir), dry_run=True)

    names = sorted(os.path.basename(path) for path in planned["files"])
    assert names == [
        "Cratewise%%Crates%%House.crate",
        "Cratewise%%Crates%%Techno.crate",
        "Cratewise%%Crates.crate",
        "Cratewise.crate",
    ]
    assert not serato_dir.exists()


def test_writing_then_removing_leaves_the_users_own_crates_alone(crates):
    db, collection, serato_dir = crates
    subcrates = serato_dir / "Subcrates"
    subcrates.mkdir(parents=True)
    (subcrates / "My Sets.crate").write_bytes(b"theirs")

    written = serato.write_crates(db, collection, str(serato_dir))
    house = subcrates / "Cratewise%%Crates%%House.crate"
    assert serato.read_crate(house.read_bytes())[0].endswith("music/a.mp3")
    assert serato.write_crates(db, collection, str(serato_dir)) == written  # refresh is fine

    preview = serato.remove_crates(str(serato_dir), dry_run=True)
    assert sorted(preview["files"]) == sorted(written["files"])
    assert house.exists()

    removed = serato.remove_crates(str(serato_dir))

    assert removed == preview
    assert sorted(p.name for p in subcrates.iterdir()) == ["My Sets.crate"]
    assert not (serato_dir / serato.MANIFEST).exists()


def test_a_crate_cratewise_did_not_write_is_never_replaced(crates):
    db, collection, serato_dir = crates
    subcrates = serato_dir / "Subcrates"
    subcrates.mkdir(parents=True)
    (subcrates / "Cratewise%%Crates%%House.crate").write_bytes(b"made in Serato")

    with pytest.raises(ExportConflict):
        serato.write_crates(db, collection, str(serato_dir))

    assert (subcrates / "Cratewise%%Crates%%House.crate").read_bytes() == b"made in Serato"
    assert sorted(p.name for p in subcrates.iterdir()) == ["Cratewise%%Crates%%House.crate"]


def test_tracks_on_another_drive_are_counted_not_written(crates):
    db, collection, _ = crates

    house = db.list_groups(collection["id"])[0]
    techno_track = db.get_group_members(db.list_groups(collection["id"])[1]["id"])[0]
    db.add_group_members(house["id"], [techno_track["id"]])  # one track in two crates

    planned = serato.write_crates(db, collection, "/Volumes/USB/_Serato_", dry_run=True)

    assert planned["left_out"] == 2  # tracks, not crate entries


def test_groups_whose_names_clean_up_the_same_get_separate_crates(crates):
    db, collection, serato_dir = crates
    for name in ("A/B", "A_B", "house"):
        group_id = db.create_group(collection["id"], name)
        track = db.get_group_members(db.list_groups(collection["id"])[0]["id"])[0]
        db.add_group_members(group_id, [track["id"]])

    planned = serato.write_crates(db, collection, str(serato_dir), dry_run=True)

    names = {os.path.basename(path).lower() for path in planned["files"]}
    assert len(names) == len(planned["files"]) == 7
    assert "cratewise%%crates%%house (2).crate" in names


def test_a_name_never_adds_a_level_of_nesting():
    assert serato._crate_name("a%%%b") == "a%b"
    assert serato._crate_name("a%%%%b") == "a%b"


def test_writing_again_removes_the_crate_of_a_renamed_group(crates):
    db, collection, serato_dir = crates
    serato.write_crates(db, collection, str(serato_dir))
    house = db.list_groups(collection["id"])[0]
    db.update_group(house["id"], name="Deep House")

    planned = serato.write_crates(db, collection, str(serato_dir), dry_run=True)
    assert [os.path.basename(p) for p in planned["removed"]] == ["Cratewise%%Crates%%House.crate"]
    serato.write_crates(db, collection, str(serato_dir))

    names = sorted(p.name for p in (serato_dir / "Subcrates").iterdir())
    assert "Cratewise%%Crates%%House.crate" not in names
    assert "Cratewise%%Crates%%Deep House.crate" in names


def test_each_collection_gets_its_own_crates(crates):
    db, collection, serato_dir = crates
    other_id = db.create_collection("Other")
    group_id = db.create_group(other_id, "House")
    track = db.get_group_members(db.list_groups(collection["id"])[0]["id"])[0]
    db.add_group_members(group_id, [track["id"]])

    serato.write_crates(db, collection, str(serato_dir))
    serato.write_crates(db, db.get_collection(collection_id=other_id), str(serato_dir))

    names = {p.name for p in (serato_dir / "Subcrates").iterdir()}
    assert {"Cratewise%%Crates%%House.crate", "Cratewise%%Other%%House.crate"} <= names
    serato.write_crates(db, collection, str(serato_dir))  # refreshing one keeps the other's
    assert (serato_dir / "Subcrates" / "Cratewise%%Other%%House.crate").exists()


def test_a_crate_whose_tracks_were_changed_in_serato_is_never_replaced_or_removed(crates):
    db, collection, serato_dir = crates
    serato.write_crates(db, collection, str(serato_dir))
    house = serato_dir / "Subcrates" / "Cratewise%%Crates%%House.crate"
    edited = serato.build_crate([*serato.read_crate(house.read_bytes()), "Users/dj/added.mp3"])
    house.write_bytes(edited)

    with pytest.raises(serato.CratesEdited):
        serato.write_crates(db, collection, str(serato_dir))
    removed = serato.remove_crates(str(serato_dir))

    assert removed["kept"] == [str(house)]
    assert house.read_bytes() == edited


@pytest.mark.parametrize("folder", ["", "  ", "~/Music/Serato", "/tmp"])
def test_only_a_folder_named_serato_is_accepted(folder):
    with pytest.raises(ValueError):
        serato.check_serato_dir(folder)
