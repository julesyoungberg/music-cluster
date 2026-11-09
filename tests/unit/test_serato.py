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
    assert names == ["Cratewise%%House.crate", "Cratewise%%Techno.crate", "Cratewise.crate"]
    assert not serato_dir.exists()


def test_writing_then_removing_leaves_the_users_own_crates_alone(crates):
    db, collection, serato_dir = crates
    subcrates = serato_dir / "Subcrates"
    subcrates.mkdir(parents=True)
    (subcrates / "My Sets.crate").write_bytes(b"theirs")

    written = serato.write_crates(db, collection, str(serato_dir))
    house = subcrates / "Cratewise%%House.crate"
    assert serato.read_crate(house.read_bytes())[0].endswith("music/a.mp3")
    assert serato.write_crates(db, collection, str(serato_dir)) == written  # refresh is fine

    removed = serato.remove_crates(str(serato_dir))

    assert sorted(removed) == sorted(written["files"])
    assert sorted(p.name for p in subcrates.iterdir()) == ["My Sets.crate"]
    assert not (serato_dir / serato.MANIFEST).exists()


def test_a_crate_cratewise_did_not_write_is_never_replaced(crates):
    db, collection, serato_dir = crates
    subcrates = serato_dir / "Subcrates"
    subcrates.mkdir(parents=True)
    (subcrates / "Cratewise%%House.crate").write_bytes(b"made in Serato")

    with pytest.raises(ExportConflict):
        serato.write_crates(db, collection, str(serato_dir))

    assert (subcrates / "Cratewise%%House.crate").read_bytes() == b"made in Serato"
    assert sorted(p.name for p in subcrates.iterdir()) == ["Cratewise%%House.crate"]


def test_tracks_on_another_drive_are_counted_not_written(crates):
    db, collection, _ = crates

    planned = serato.write_crates(db, collection, "/Volumes/USB/_Serato_", dry_run=True)

    assert planned["left_out"] == 2
