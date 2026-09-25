"""Tags are stored per track, filtered on, counted, and found stale."""

import pytest

from music_cluster import tagging


def payload(genre="house", beat="four_on_floor", moods=("dark",), energy=0.6, **extra):
    data = {
        "genre": genre,
        "beat": beat,
        "moods": list(moods),
        "energy": energy,
        "danceability": 0.8,
        "bpm": 124.0,
        "key": "A minor",
        "camelot": "8A",
        "swung": False,
        "version": tagging.TAGGER_VERSION,
        "scores": {"beat": {beat: 0.9}},
        "descriptors": {"loudness_db": -9.0},
    }
    data.update(extra)
    return data


@pytest.fixture
def tagged(db, rng):
    """Four music tracks, three tagged differently and one untagged."""
    ids = []
    for index in range(4):
        track_id = db.upsert_track(f"/music/{index}.flac", f"{index}.flac")
        db.add_features(track_id, rng.normal(0, 1, 96), "music")
        ids.append(track_id)
    db.save_tags(ids[0], payload())
    db.save_tags(
        ids[1],
        payload(genre="drum_and_bass", beat="half_time", moods=("dark", "aggressive"), energy=0.9),
    )
    db.save_tags(
        ids[2],
        payload(
            genre="uk_garage",
            beat="two_step",
            moods=("groovy",),
            energy=0.5,
            swung=True,
            camelot="5A",
        ),
    )
    return ids


def test_tags_round_trip(db, tagged):
    stored = db.get_tags(tagged[1])
    assert stored["genre"] == "drum_and_bass"
    assert stored["moods"] == ["dark", "aggressive"]
    assert stored["descriptors"] == {"loudness_db": -9.0}
    assert stored["tagged_at"]
    assert db.get_tags(tagged[3]) is None


def test_saving_again_replaces(db, tagged):
    db.save_tags(tagged[0], payload(genre="techno"))
    assert db.get_tags(tagged[0])["genre"] == "techno"
    assert db.tag_counts()["tagged"] == 3


@pytest.mark.parametrize(
    "filters, expected",
    [
        ({"beat": "half_time"}, [1]),
        ({"genre": "uk_garage"}, [2]),
        ({"mood": "dark"}, [0, 1]),
        ({"mood": "aggressive"}, [1]),
        ({"min_energy": 0.55}, [0, 1]),
        ({"max_energy": 0.55}, [2]),
        ({"camelot": "5a"}, [2]),
        ({"swung": True}, [2]),
        ({"tagged": False}, [3]),
        ({"tagged": True, "mood": "dark", "min_energy": 0.8}, [1]),
    ],
)
def test_tracks_filter_by_tag(db, tagged, filters, expected):
    found = sorted(track["id"] for track in db.list_tracks(tags=filters))
    assert found == sorted(tagged[i] for i in expected)
    assert db.count_matching_tracks(tags=filters) == len(expected)


def test_a_mood_filter_matches_whole_words_only(db, tagged):
    """`dark` must not match a hypothetical `darkwave` by substring."""
    db.save_tags(tagged[3], payload(moods=("darkwave",)))
    found = {track["id"] for track in db.list_tracks(tags={"mood": "dark"})}
    assert tagged[3] not in found


def test_counts_break_down_the_library(db, tagged):
    counts = db.tag_counts()
    assert counts["tagged"] == 3
    assert counts["beats"] == {"four_on_floor": 1, "half_time": 1, "two_step": 1}
    assert counts["moods"]["dark"] == 2


def test_untagged_and_stale_tracks_need_tagging(db, tagged):
    assert db.tracks_needing_tags(None, tagging.TAGGER_VERSION) == [tagged[3]]
    db.save_tags(tagged[0], payload(version="0-old"))
    assert db.tracks_needing_tags(None, tagging.TAGGER_VERSION) == [tagged[0], tagged[3]]
    assert db.tracks_needing_tags(tagged[:2], tagging.TAGGER_VERSION) == [tagged[0]]


def test_samples_are_never_waiting_for_tags(db, rng):
    """A beat type means nothing for a kick; one-shots are not tagged."""
    kick = db.upsert_track("/samples/kick.wav", "kick.wav")
    db.add_features(kick, rng.normal(0, 1, 125), "sample")
    assert db.tracks_needing_tags(None, None) == []


def test_deleting_a_track_deletes_its_tags(db, tagged):
    db.delete_track(tagged[0])
    assert db.get_tags(tagged[0]) is None
    assert db.tag_counts()["tagged"] == 2
