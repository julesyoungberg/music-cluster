"""Assisted discovery: proposing candidate piles, and promoting the real ones."""

import pytest

from music_cluster import discovery
from tests.conftest import FEATURE_DIM, make_centres, make_vector


@pytest.fixture
def pile(db, rng):
    """An unsorted pile of 90 tracks in three well-separated clumps."""
    centres = make_centres(rng, 3, separation=12.0)
    track_ids = []
    genres = ["House", "Techno", "Drum & Bass"]
    for index, centre in enumerate(centres):
        for position in range(30):
            track_id = db.upsert_track(
                filepath=f"/pile/{index}/{position}.mp3",
                filename=f"{position}.mp3",
                metadata={
                    "artist": f"Artist {index}",
                    "title": f"Track {position}",
                    "genre": genres[index],
                    "tag_bpm": [124.0, 130.0, 172.0][index],
                },
            )
            db.add_features(track_id, make_vector(rng, centre))
            track_ids.append(track_id)
    return track_ids


class TestRunDiscovery:
    def test_proposes_candidates_from_a_track_list(self, db, config, pile):
        result = discovery.run_discovery(
            db, config, track_ids=pile, algorithm="kmeans", target_groups=3, min_group_size=5
        )

        assert len(result["candidates"]) == 3
        assert sum(candidate["size"] for candidate in result["candidates"]) == 90

    def test_candidates_carry_a_suggested_name_and_statistics(self, db, config, pile):
        result = discovery.run_discovery(
            db, config, track_ids=pile, algorithm="kmeans", target_groups=3, min_group_size=5
        )

        for candidate in result["candidates"]:
            assert candidate["suggested_name"]
            assert candidate["stats"]["descriptors"] is not None
            assert candidate["exemplars"]

    def test_exemplars_come_from_the_candidates_own_members(self, db, config, pile):
        result = discovery.run_discovery(
            db, config, track_ids=pile, algorithm="kmeans", target_groups=3, min_group_size=5
        )

        for candidate in result["candidates"]:
            members = set(db.get_discovery_member_ids(candidate["id"]))
            assert set(candidate["exemplars"]) <= members

    def test_the_suggested_name_uses_the_genre_tags_when_they_agree(self, db, config, pile):
        result = discovery.run_discovery(
            db, config, track_ids=pile, algorithm="kmeans", target_groups=3, min_group_size=5
        )

        names = " ".join(candidate["suggested_name"] for candidate in result["candidates"])
        assert "House" in names or "Techno" in names

    def test_nothing_becomes_a_group_without_a_human(self, db, config, pile):
        collection_id = db.create_collection("Crates")
        discovery.run_discovery(
            db, config, track_ids=pile, algorithm="kmeans", target_groups=3, min_group_size=5
        )

        # The whole promise of the directed workflow: discovery proposes, the
        # user disposes.
        assert db.list_groups(collection_id) == []

    def test_needs_something_to_discover_from(self, db, config):
        with pytest.raises(ValueError, match="paths or track_ids"):
            discovery.run_discovery(db, config)

    def test_needs_enough_analysed_tracks(self, db, config, pile):
        with pytest.raises(ValueError, match="at least"):
            discovery.run_discovery(db, config, track_ids=pile[:3], min_group_size=8)

    def test_records_the_run_with_its_parameters(self, db, config, pile):
        result = discovery.run_discovery(
            db,
            config,
            track_ids=pile,
            name="Friday pile",
            algorithm="kmeans",
            target_groups=3,
            min_group_size=5,
        )

        run = result["run"]
        assert run["name"] == "Friday pile"
        assert run["algorithm"] == "kmeans"
        assert run["params"]["target_groups"] == 3
        assert run["metrics"]["n_candidates"] == 3

    def test_granularity_changes_how_many_piles_are_proposed(self, db, config, pile):
        coarse = discovery.run_discovery(
            db, config, track_ids=pile, algorithm="kmeans", granularity="coarse", min_group_size=5
        )
        fine = discovery.run_discovery(
            db, config, track_ids=pile, algorithm="kmeans", granularity="finest", min_group_size=5
        )

        assert len(fine["candidates"]) >= len(coarse["candidates"])

    def test_an_llm_failure_leaves_the_heuristic_names_in_place(
        self, db, config, pile, monkeypatch
    ):
        monkeypatch.setattr("music_cluster.llm.suggest_names", lambda *a, **k: {})

        result = discovery.run_discovery(
            db,
            config,
            track_ids=pile,
            algorithm="kmeans",
            target_groups=3,
            min_group_size=5,
            use_llm=True,
        )

        assert all(candidate["suggested_name"] for candidate in result["candidates"])

    def test_llm_names_replace_the_heuristic_ones_when_available(
        self, db, config, pile, monkeypatch
    ):
        monkeypatch.setattr(
            "music_cluster.llm.suggest_names",
            lambda candidates, cfg: {
                c["candidate_index"]: f"LLM {c['candidate_index']}" for c in candidates
            },
        )

        result = discovery.run_discovery(
            db,
            config,
            track_ids=pile,
            algorithm="kmeans",
            target_groups=3,
            min_group_size=5,
            use_llm=True,
        )

        assert {c["suggested_name"] for c in result["candidates"]} == {"LLM 0", "LLM 1", "LLM 2"}


class TestPromoteCandidate:
    @pytest.fixture
    def discovered(self, db, config, pile):
        collection_id = db.create_collection("Crates")
        result = discovery.run_discovery(
            db, config, track_ids=pile, algorithm="kmeans", target_groups=3, min_group_size=5
        )
        return db.get_collection(collection_id=collection_id), result["candidates"]

    def test_creates_a_group_from_the_whole_pile(self, db, config, discovered):
        collection, candidates = discovered

        promoted = discovery.promote_candidate(db, config, collection, candidates[0]["id"])

        assert promoted["added"] == candidates[0]["size"]
        assert promoted["group"]["source_kind"] == "discovered"

    def test_an_explicit_name_overrides_the_suggestion(self, db, config, discovered):
        collection, candidates = discovered

        promoted = discovery.promote_candidate(
            db, config, collection, candidates[0]["id"], name="Rollers"
        )

        assert promoted["group"]["name"] == "Rollers"

    def test_seeds_only_adds_the_exemplars_alone(self, db, config, discovered):
        collection, candidates = discovered

        promoted = discovery.promote_candidate(
            db, config, collection, candidates[0]["id"], seeds_only=True
        )

        assert promoted["added"] == len(candidates[0]["exemplars"])

    def test_can_merge_into_an_existing_group(self, db, config, discovered):
        collection, candidates = discovered
        first = discovery.promote_candidate(db, config, collection, candidates[0]["id"])
        before = len(db.get_group_member_ids(first["group"]["id"]))

        merged = discovery.promote_candidate(
            db, config, collection, candidates[1]["id"], target_group_id=first["group"]["id"]
        )

        assert merged["group"]["id"] == first["group"]["id"]
        assert len(db.get_group_member_ids(first["group"]["id"])) > before

    def test_will_not_merge_into_another_collections_group(self, db, config, discovered):
        collection, candidates = discovered
        other_id = db.create_collection("Elsewhere")
        stranger = db.create_group(other_id, "Not mine")

        with pytest.raises(LookupError, match="not part of this collection"):
            discovery.promote_candidate(
                db, config, collection, candidates[0]["id"], target_group_id=stranger
            )

    def test_promoting_marks_the_candidate_so_it_is_not_offered_twice(self, db, config, discovered):
        collection, candidates = discovered

        discovery.promote_candidate(db, config, collection, candidates[0]["id"])

        assert db.get_discovery_candidate(candidates[0]["id"])["status"] == "promoted"

    def test_an_unknown_candidate_is_an_error(self, db, config, discovered):
        collection, _ = discovered

        with pytest.raises(LookupError, match="No discovery candidate"):
            discovery.promote_candidate(db, config, collection, 99999)

    def test_rejecting_marks_the_candidate_without_creating_anything(self, db, config, discovered):
        collection, candidates = discovered

        discovery.reject_candidate(db, candidates[0]["id"])

        assert db.get_discovery_candidate(candidates[0]["id"])["status"] == "rejected"
        assert db.list_groups(collection["id"]) == []


class TestExpandSeeds:
    def test_finds_the_tracks_nearest_the_seeds(self, db, config, pile):
        seeds = pile[:2]

        results = discovery.expand_seeds(db, config, seeds, limit=10)

        assert len(results) == 10
        # The first clump is tracks 0-29; near neighbours should come from it.
        assert sum(1 for r in results if r["track"]["id"] in pile[:30]) >= 8

    def test_results_are_ordered_by_distance(self, db, config, pile):
        results = discovery.expand_seeds(db, config, pile[:2], limit=20)
        distances = [result["distance"] for result in results]
        assert distances == sorted(distances)

    def test_seeds_are_never_returned_as_their_own_matches(self, db, config, pile):
        seeds = pile[:3]
        results = discovery.expand_seeds(db, config, seeds, limit=50)
        assert not any(result["track"]["id"] in seeds for result in results)

    def test_a_distance_cutoff_drops_the_far_ones(self, db, config, pile):
        wide = discovery.expand_seeds(db, config, pile[:2], limit=90)
        narrow = discovery.expand_seeds(db, config, pile[:2], limit=90, max_distance=0.5)

        assert len(narrow) < len(wide)
        assert all(result["relative_distance"] <= 0.5 for result in narrow)

    def test_seeds_spanning_two_sounds_find_matches_for_both(self, db, config, pile):
        # Distance to the *nearest* seed, not to their midpoint: otherwise a
        # pair of seeds from different genres matches neither.
        seeds = [pile[0], pile[60]]

        results = discovery.expand_seeds(db, config, seeds, limit=20)
        matched = {result["track"]["id"] for result in results}

        assert matched & set(pile[:30])
        assert matched & set(pile[60:])

    def test_needs_at_least_one_seed(self, db, config):
        with pytest.raises(ValueError, match="at least one seed"):
            discovery.expand_seeds(db, config, [])

    def test_an_empty_pool_returns_nothing(self, db, config, pile):
        assert discovery.expand_seeds(db, config, pile[:1], pool_track_ids=[]) == []

    def test_a_pool_can_be_restricted_to_specific_tracks(self, db, config, pile):
        pool = pile[60:]

        results = discovery.expand_seeds(db, config, pile[:1], pool_track_ids=pool, limit=50)

        assert {result["track"]["id"] for result in results} <= set(pool)


def test_a_pile_with_no_structure_reports_that_rather_than_inventing_groups(db, config, rng):
    """HDBSCAN calling everything noise must be an error, not an empty run."""
    track_ids = []
    for index in range(40):
        track_id = db.upsert_track(filepath=f"/flat/{index}.mp3", filename=f"{index}.mp3")
        # A single tight blob: there are no candidate piles to find in it.
        db.add_features(track_id, rng.normal(0, 0.001, FEATURE_DIM))
        track_ids.append(track_id)

    with pytest.raises(ValueError, match="No candidate groups found"):
        discovery.run_discovery(
            db, config, track_ids=track_ids, algorithm="hdbscan", min_group_size=30
        )
