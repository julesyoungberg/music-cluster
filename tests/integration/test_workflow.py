"""The whole directed-sorting workflow, end to end.

Groups are built from reference tracks, the sorter is fitted, a batch of new
music is scored, decisions are recorded, and the result is written to disk.
Synthetic feature vectors stand in for audio; the extractor has its own test.
"""

import os

import numpy as np
import pytest

from music_cluster import groups as groups_mod
from music_cluster import organizer, sorting
from music_cluster.discovery import promote_candidate, run_discovery
from tests.conftest import FEATURE_DIM, make_centres, make_vector


def seed_group(db, collection_id, name, centre, rng, count, destination=None):
    group_id = db.create_group(collection_id, name, destination_path=destination)
    for index in range(count):
        track_id = db.upsert_track(
            filepath=f"/library/{name}/{index}.mp3",
            filename=f"{index}.mp3",
            duration=300.0,
            metadata={"artist": name, "title": f"Track {index}", "genre": name},
        )
        db.add_features(track_id, make_vector(rng, centre))
        db.add_group_members(group_id, [track_id])
    return group_id


def incoming(db, centre, rng, count, directory):
    """Create files on disk with pre-computed features, as if just analysed."""
    track_ids = []
    for index in range(count):
        path = directory / f"new_{index}.mp3"
        path.write_bytes(b"audio")
        track_id = db.upsert_track(
            filepath=str(path), filename=path.name, duration=300.0,
            metadata={"artist": "New", "title": f"New {index}"},
        )
        db.add_features(track_id, make_vector(rng, centre))
        track_ids.append(track_id)
    return track_ids


@pytest.fixture
def library(db, config, rng, temp_dir):
    collection_id = db.create_collection("DJ folders")
    centres = make_centres(rng, 3, separation=5.0)

    destinations = []
    group_ids = []
    for index, centre in enumerate(centres):
        destination = temp_dir / f"dest{index}"
        destination.mkdir()
        destinations.append(destination)
        group_ids.append(
            seed_group(db, collection_id, f"Group {index}", centre, rng, 15, str(destination))
        )

    collection = db.get_collection(collection_id=collection_id)
    return collection, group_ids, centres, destinations


def test_fit_reports_separable_groups(db, config, library):
    collection, group_ids, _, _ = library
    metrics = groups_mod.fit_collection(db, collection, config)

    assert metrics["accuracy"] > 0.9
    assert metrics["n_groups"] == 3
    assert db.load_model(collection["id"]) is not None


def test_sort_then_commit_files_the_music(db, config, library, rng, temp_dir):
    collection, group_ids, centres, destinations = library
    groups_mod.fit_collection(db, collection, config)

    inbox = temp_dir / "inbox"
    inbox.mkdir()
    track_ids = incoming(db, centres[1], rng, 5, inbox)

    session_id = db.create_sort_session(collection["id"], "batch", str(inbox))
    sorting.score_session(db, config, session_id, track_ids=track_ids)

    items = db.list_sort_items(session_id)
    assert len(items) == 5
    assert all(item["suggested_group_id"] == group_ids[1] for item in items)

    # The human confirms every suggestion.
    for item in items:
        sorting.decide(db, item["id"], group_id=item["suggested_group_id"])

    outcome = sorting.commit_session(db, config, session_id, mode="copy", learn=False)
    assert outcome["result"]["performed_count"] == 5
    assert len(os.listdir(destinations[1])) == 5


def test_nothing_is_written_until_a_human_accepts(db, config, library, rng, temp_dir):
    """Suggestions alone must not move a single file."""
    collection, _, centres, destinations = library
    groups_mod.fit_collection(db, collection, config)

    inbox = temp_dir / "inbox"
    inbox.mkdir()
    track_ids = incoming(db, centres[0], rng, 4, inbox)

    session_id = db.create_sort_session(collection["id"], "batch", str(inbox))
    sorting.score_session(db, config, session_id, track_ids=track_ids)

    plan = sorting.plan_session_commit(db, config, session_id, mode="copy")
    assert plan.actions == []
    assert all(os.listdir(destination) == [] for destination in destinations)


def test_auto_accept_only_takes_confident_suggestions(db, config, library, rng, temp_dir):
    collection, group_ids, centres, _ = library
    groups_mod.fit_collection(db, collection, config)

    inbox = temp_dir / "inbox"
    inbox.mkdir()
    clear = incoming(db, centres[2], rng, 3, inbox)

    # A track from nowhere near any group.
    odd_path = inbox / "odd.mp3"
    odd_path.write_bytes(b"audio")
    odd_id = db.upsert_track(filepath=str(odd_path), filename="odd.mp3")
    db.add_features(odd_id, np.full(FEATURE_DIM, 400.0))

    session_id = db.create_sort_session(collection["id"], "batch", str(inbox))
    sorting.score_session(db, config, session_id, track_ids=clear + [odd_id], auto_accept=True)

    summary = db.sort_session_summary(session_id)
    assert summary.get("accepted") == 3
    assert summary.get("unmatched") == 1


def test_committing_teaches_the_groups(db, config, library, rng, temp_dir):
    """With learning on, filed tracks become references for next time."""
    collection, group_ids, centres, _ = library
    groups_mod.fit_collection(db, collection, config)

    inbox = temp_dir / "inbox"
    inbox.mkdir()
    track_ids = incoming(db, centres[0], rng, 3, inbox)

    session_id = db.create_sort_session(collection["id"], "batch", str(inbox))
    sorting.score_session(db, config, session_id, track_ids=track_ids)
    for item in db.list_sort_items(session_id):
        sorting.decide(db, item["id"], group_id=group_ids[0])

    before = len(db.get_group_member_ids(group_ids[0]))
    outcome = sorting.commit_session(db, config, session_id, mode="copy", learn=True)

    assert outcome["learned"] == 3
    assert len(db.get_group_member_ids(group_ids[0])) == before + 3


def test_undo_reverses_a_move_commit(db, config, library, rng, temp_dir):
    collection, group_ids, centres, destinations = library
    groups_mod.fit_collection(db, collection, config)

    inbox = temp_dir / "inbox"
    inbox.mkdir()
    track_ids = incoming(db, centres[1], rng, 3, inbox)

    session_id = db.create_sort_session(collection["id"], "batch", str(inbox))
    sorting.score_session(db, config, session_id, track_ids=track_ids)
    for item in db.list_sort_items(session_id):
        sorting.decide(db, item["id"], group_id=group_ids[1])

    sorting.commit_session(db, config, session_id, mode="move", learn=False)
    assert len(os.listdir(destinations[1])) == 3
    assert len(os.listdir(inbox)) == 0

    organizer.undo_session(db, session_id)
    assert len(os.listdir(destinations[1])) == 0
    assert len(os.listdir(inbox)) == 3


def test_rescoring_picks_up_a_new_group(db, config, library, rng, temp_dir):
    """A track that matched nothing should find a home once one exists."""
    collection, _, centres, _ = library
    groups_mod.fit_collection(db, collection, config)

    outlier_centre = np.full(FEATURE_DIM, 60.0)
    inbox = temp_dir / "inbox"
    inbox.mkdir()
    track_ids = incoming(db, outlier_centre, rng, 3, inbox)

    session_id = db.create_sort_session(collection["id"], "batch", str(inbox))
    sorting.score_session(db, config, session_id, track_ids=track_ids)
    assert db.sort_session_summary(session_id).get("unmatched") == 3

    new_group = seed_group(db, collection["id"], "Outliers", outlier_centre, rng, 10)
    collection = db.get_collection(collection_id=collection["id"])
    groups_mod.fit_collection(db, collection, config)

    sorting.score_session(db, config, session_id)
    items = db.list_sort_items(session_id)
    assert all(item["suggested_group_id"] == new_group for item in items)


def test_discovery_proposes_groups_that_can_be_promoted(db, config, rng, temp_dir):
    """The unsorted-pile path: cluster, review, promote — with a human in between."""
    centres = make_centres(rng, 3, separation=6.0)
    track_ids = []
    for index, centre in enumerate(centres):
        for position in range(12):
            track_id = db.upsert_track(
                filepath=f"/pile/{index}_{position}.mp3",
                filename=f"{index}_{position}.mp3",
                metadata={"genre": ["House", "Techno", "Jungle"][index], "tag_bpm": 120 + index * 20},
            )
            db.add_features(track_id, make_vector(rng, centre))
            track_ids.append(track_id)

    outcome = run_discovery(
        db, config, track_ids=track_ids, algorithm="kmeans", target_groups=3, min_group_size=4
    )
    candidates = outcome["candidates"]
    assert len(candidates) == 3
    assert all(candidate["suggested_name"] for candidate in candidates)
    assert all(candidate["exemplars"] for candidate in candidates)

    # Nothing exists as a group until it is promoted.
    collection_id = db.create_collection("From the pile")
    collection = db.get_collection(collection_id=collection_id)
    assert db.list_groups(collection_id) == []

    result = promote_candidate(db, config, collection, candidates[0]["id"], name="Kept")
    assert result["added"] == candidates[0]["size"]
    assert [group["name"] for group in db.list_groups(collection_id)] == ["Kept"]


def test_promoting_with_seeds_only_adds_the_exemplars(db, config, rng):
    centres = make_centres(rng, 2, separation=6.0)
    track_ids = []
    for index, centre in enumerate(centres):
        for position in range(12):
            track_id = db.upsert_track(
                filepath=f"/pile/{index}_{position}.mp3", filename=f"{index}_{position}.mp3"
            )
            db.add_features(track_id, make_vector(rng, centre))
            track_ids.append(track_id)

    outcome = run_discovery(
        db, config, track_ids=track_ids, algorithm="kmeans", target_groups=2, min_group_size=4
    )
    candidate = outcome["candidates"][0]

    collection = db.get_collection(collection_id=db.create_collection("c"))
    result = promote_candidate(db, config, collection, candidate["id"], seeds_only=True)

    assert result["added"] == len(candidate["exemplars"])
    assert result["added"] < candidate["size"]
