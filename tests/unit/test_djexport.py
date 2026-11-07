"""Exporting groups to DJ software: metadata, locations and file structure."""

import os
from xml.etree import ElementTree as ET

import pytest

from music_cluster import djexport


@pytest.fixture
def crates(temp_dir, db):
    """Two groups sharing one track, and an empty group."""
    collection_id = db.create_collection("Crates")
    house = db.create_group(collection_id, "House")
    techno = db.create_group(collection_id, "Techno")
    db.create_group(collection_id, "Empty")

    def track(name, **meta):
        path = temp_dir / name
        path.write_bytes(b"audio")
        return db.upsert_track(filepath=str(path), filename=name, duration=300.4, metadata=meta)

    tagged = track("one.mp3", title="One", artist="DJ A", genre="Deep House", tag_bpm=122.0)
    measured = track("two track.flac")
    shared = track("three.wav", title="Three")
    db.save_tags(measured, {"genre_label": "Techno", "bpm": 131.04, "key": "A minor"})
    db.add_group_members(house, [tagged, shared])
    db.add_group_members(techno, [measured, shared])
    return db, db.get_collection(collection_id=collection_id), (tagged, measured, shared), temp_dir


@pytest.mark.parametrize(
    "key, expected",
    [
        ("A minor", "Am"),
        ("C major", "C"),
        ("C# minor", "Dbm"),
        ("F# major", "F#"),
        ("8A", "8A"),
        ("", None),
        (None, None),
    ],
)
def test_keys_are_written_the_way_dj_apps_read_them(key, expected):
    assert djexport.short_key(key) == expected


def test_the_files_own_tags_win_over_measured_ones():
    track = {"id": 1, "filepath": "/m/a.mp3", "genre": "Garage", "tag_bpm": 130.0, "tag_key": "Fm"}
    tags = {"genre_label": "House", "bpm": 129.0, "key": "A minor"}

    meta = djexport.track_metadata(track, tags)

    assert (meta["genre"], meta["bpm"], meta["key"]) == ("Garage", 130.0, "Fm")


def test_measured_tags_fill_the_gaps():
    track = {"id": 1, "filepath": "/m/a.mp3"}
    meta = djexport.track_metadata(track, {"genre_label": "House", "bpm": 124.0, "key": "G major"})

    assert (meta["title"], meta["genre"], meta["bpm"], meta["key"]) == ("a", "House", 124.0, "G")


def test_rekordbox_locations_are_url_encoded_file_urls():
    assert (
        djexport.rekordbox_location("/Users/dj/My Music/a&b.mp3")
        == "file://localhost/Users/dj/My%20Music/a%26b.mp3"
    )


@pytest.mark.parametrize(
    "path, expected",
    [
        ("C:\\Music\\a b.mp3", "file://localhost/C:/Music/a%20b.mp3"),
        ("\\\\NAS\\Music\\a.mp3", "file://NAS/Music/a.mp3"),
    ],
)
def test_rekordbox_locations_for_windows_drives_and_shares(path, expected):
    assert djexport.rekordbox_location(path) == expected


def test_traktor_locations_split_off_external_volumes():
    assert djexport.traktor_location("/Volumes/USB/House/a.mp3") == (
        "USB",
        "/:House/:",
        "a.mp3",
    )


def test_rekordbox_export_has_one_playlist_per_group_pointing_at_the_collection(crates):
    db, collection, (tagged, measured, shared), temp_dir = crates
    out = djexport.export_collection(db, collection, str(temp_dir / "out"), "rekordbox")

    root = ET.parse(out[0]).getroot()
    tracks = {t.get("TrackID"): t for t in root.find("COLLECTION")}
    assert len(tracks) == 3  # the shared track appears once
    assert tracks[str(measured)].get("AverageBpm") == "131.04"
    assert tracks[str(measured)].get("Tonality") == "Am"
    assert tracks[str(tagged)].get("Genre") == "Deep House"

    folder = root.find("PLAYLISTS/NODE/NODE")
    assert folder.get("Name") == "Crates"
    playlists = {node.get("Name"): [t.get("Key") for t in node] for node in folder}
    assert set(playlists) == {"House", "Techno"}  # the empty group is left out
    assert all(key in tracks for keys in playlists.values() for key in keys)


def test_traktor_export_playlist_entries_match_collection_entries(crates):
    db, collection, _, temp_dir = crates
    out = djexport.export_collection(db, collection, str(temp_dir / "out"), "traktor")

    root = ET.parse(out[0]).getroot()
    locations = {
        loc.get("VOLUME") + loc.get("DIR") + loc.get("FILE") for loc in root.iter("LOCATION")
    }
    keys = {pk.get("KEY") for pk in root.iter("PRIMARYKEY")}
    assert len(locations) == 3
    assert keys == locations
    names = [node.get("NAME") for node in root.iter("NODE") if node.get("TYPE") == "PLAYLIST"]
    assert sorted(names) == ["House", "Techno"]


def test_unknown_formats_are_refused(crates):
    db, collection, _, temp_dir = crates
    with pytest.raises(ValueError):
        djexport.export_collection(db, collection, str(temp_dir / "never"), "itunes")


@pytest.mark.parametrize(
    "key, expected", [("C", 0), ("Db", 1), ("F#", 6), ("Am", 21), ("8A", None)]
)
def test_traktor_reads_keys_as_numbers(key, expected):
    assert djexport.traktor_musical_key(key) == expected


def test_control_characters_in_tags_do_not_break_the_xml(crates):
    db, collection, (tagged, _, _), temp_dir = crates
    db.upsert_track(
        filepath=str(temp_dir / "one.mp3"),
        filename="one.mp3",
        metadata={"title": "One\x01", "comment": "bad\x0bbyte"},
    )

    for fmt in ("rekordbox", "traktor"):
        [written] = djexport.export_collection(db, collection, str(temp_dir / fmt), fmt)
        ET.parse(written)  # raises if the file is not well-formed


def test_an_export_refreshes_its_own_files_but_leaves_the_users_alone(crates):
    db, collection, _, temp_dir = crates
    out = temp_dir / "out"
    first = djexport.export_collection(db, collection, str(out), "m3u8")
    assert djexport.export_collection(db, collection, str(out), "m3u8") == first

    (out / "House.m3u8").write_text("#EXTM3U\n/my/own/track.mp3\n")
    with pytest.raises(djexport.ExportConflict):
        djexport.export_collection(db, collection, str(out), "m3u8")
    assert (out / "House.m3u8").read_text() == "#EXTM3U\n/my/own/track.mp3\n"


def test_groups_whose_names_clash_on_disk_get_separate_playlists(temp_dir, db):
    collection_id = db.create_collection("c")
    path = temp_dir / "a.mp3"
    path.write_bytes(b"audio")
    track = db.upsert_track(filepath=str(path), filename="a.mp3")
    for name in ("A/B", "A:B"):
        db.add_group_members(db.create_group(collection_id, name), [track])

    written = djexport.export_collection(
        db, db.get_collection(collection_id=collection_id), str(temp_dir / "out"), "m3u"
    )

    assert sorted(os.path.basename(p) for p in written) == ["A_B (2).m3u", "A_B.m3u"]


def test_a_collection_with_only_empty_groups_writes_nothing(temp_dir, db):
    collection_id = db.create_collection("c")
    db.create_group(collection_id, "Empty")
    collection = db.get_collection(collection_id=collection_id)

    assert djexport.export_collection(db, collection, str(temp_dir / "out"), "rekordbox") == []
    assert not (temp_dir / "out").exists()
