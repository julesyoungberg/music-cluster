"""The CLI, end to end, against real audio on a real database.

These follow the two workflows the README documents. Nothing is stubbed: the
commands decode audio, fit a model, and write files to disk.
"""

import json
import os
from pathlib import Path

import pytest

from music_cluster.cli import DEFAULT_COLLECTION
from tests.e2e.conftest import genre_folder, new_music, run_cli


pytestmark = pytest.mark.e2e


@pytest.fixture
def imported(cli_runner, music, workspace):
    """A collection whose groups are the three genre folders, fitted."""
    run_cli(cli_runner, "init")
    run_cli(cli_runner, "collection", "create", "Crates")
    run_cli(cli_runner, "groups", "import-tree", str(music / "DJ"), "--collection", "Crates")
    run_cli(cli_runner, "fit", "--collection", "Crates")
    return workspace


class TestInit:
    def test_creates_the_database_and_config(self, cli_runner, workspace):
        result = run_cli(cli_runner, "init")

        assert (workspace / "library.db").is_file()
        assert (workspace / "config.yaml").is_file()
        assert "music-cluster" in result.output

    def test_info_reports_an_empty_library(self, cli_runner, workspace):
        run_cli(cli_runner, "init")
        result = run_cli(cli_runner, "info")

        assert "0" in result.output


class TestFoldersIAlreadyHave:
    """`groups import-tree` → `fit` → `check`, the first README workflow."""

    def test_each_subfolder_becomes_a_group(self, cli_runner, music, workspace):
        run_cli(cli_runner, "init")
        run_cli(cli_runner, "collection", "create", "Crates")

        result = run_cli(
            cli_runner, "groups", "import-tree", str(music / "DJ"), "--collection", "Crates"
        )

        assert "Deep House" in result.output
        assert "Techno" in result.output
        assert "Drum & Bass" in result.output

    def test_a_group_files_back_into_its_own_folder(self, cli_runner, imported, music):
        result = run_cli(cli_runner, "groups", "show", "Techno", "--collection", "Crates")

        assert str(music / "DJ" / "Techno") in result.output

    def test_groups_list_shows_every_group_with_its_size(self, cli_runner, imported):
        result = run_cli(cli_runner, "groups", "list", "--collection", "Crates")

        assert "Deep House" in result.output
        assert "8" in result.output

    def test_fitting_learns_the_folders(self, cli_runner, music, workspace):
        run_cli(cli_runner, "init")
        run_cli(cli_runner, "collection", "create", "Crates")
        run_cli(cli_runner, "groups", "import-tree", str(music / "DJ"), "--collection", "Crates")

        result = run_cli(cli_runner, "fit", "--collection", "Crates")

        assert "accuracy" in result.output.lower()

    def test_check_reports_per_group_accuracy(self, cli_runner, imported):
        result = run_cli(cli_runner, "check", "--collection", "Crates")

        assert "Accuracy" in result.output
        for name in ("Deep House", "Techno", "Drum & Bass"):
            assert name in result.output

    def test_the_synthetic_folders_are_distinguishable(self, cli_runner, imported, workspace):
        """The three tone families must be separable, or the rest proves nothing."""
        from music_cluster.config import Config
        from music_cluster.database import Database

        db = Database(Config().get_db_path())
        collection = db.get_collection(name="Crates")

        metrics = db.load_model(collection["id"])["metrics"]

        assert metrics["accuracy"] > 0.8

    def test_check_before_fitting_says_so(self, cli_runner, music, workspace):
        run_cli(cli_runner, "init")
        run_cli(cli_runner, "collection", "create", "Crates")
        run_cli(cli_runner, "groups", "import-tree", str(music / "DJ"), "--collection", "Crates")

        # import-tree touches the collection, so nothing is fitted yet.
        result = run_cli(cli_runner, "check", "--collection", "Crates", expect_success=False)

        assert result.exit_code != 0
        assert "fit" in result.output


class TestSortAndApply:
    def test_sorting_new_music_produces_a_review_queue(self, cli_runner, imported, music):
        result = run_cli(
            cli_runner,
            "sort",
            new_music(music),
            "--collection",
            "Crates",
            "--no-review",
        )

        assert "to review" in result.output
        assert "music-cluster review" in result.output

    def test_reviewing_files_each_track_into_its_top_suggestion(self, cli_runner, imported, music):
        run_cli(cli_runner, "sort", new_music(music), "--collection", "Crates", "--no-review")

        # Six tracks, each answered with "1" — take the top suggestion.
        result = run_cli(cli_runner, "review", "1", input_text="1\n" * 6)

        assert "Review complete: 6 filed" in result.output

    def test_review_can_skip_a_track(self, cli_runner, imported, music):
        run_cli(cli_runner, "sort", new_music(music), "--collection", "Crates", "--no-review")

        result = run_cli(cli_runner, "review", "1", input_text="s\n" + "1\n" * 5)

        assert "5 filed" in result.output
        assert "1 skipped" in result.output

    def test_review_can_be_quit_and_resumed(self, cli_runner, imported, music):
        run_cli(cli_runner, "sort", new_music(music), "--collection", "Crates", "--no-review")

        stopped = run_cli(cli_runner, "review", "1", input_text="1\nq\n")
        assert "Resume with" in stopped.output

        resumed = run_cli(cli_runner, "review", "1", input_text="1\n" * 5)
        assert "Review complete" in resumed.output

    def test_auto_accept_skips_the_queue_for_confident_tracks(self, cli_runner, imported, music):
        result = run_cli(
            cli_runner,
            "sort",
            new_music(music),
            "--collection",
            "Crates",
            "--auto-accept",
            "--no-review",
        )

        assert "auto-accepted" in result.output

    def test_a_dry_run_changes_nothing(self, cli_runner, imported, music, workspace):
        run_cli(
            cli_runner,
            "sort",
            new_music(music),
            "--collection",
            "Crates",
            "--auto-accept",
            "--no-review",
        )
        playlists = workspace / "playlists"

        result = run_cli(
            cli_runner,
            "apply",
            "1",
            "--mode",
            "playlist",
            "--playlist-dir",
            str(playlists),
            "--dry-run",
        )

        assert "Dry run" in result.output
        assert not playlists.exists()

    def test_applying_in_playlist_mode_writes_m3u_files(
        self, cli_runner, imported, music, workspace
    ):
        run_cli(
            cli_runner,
            "sort",
            new_music(music),
            "--collection",
            "Crates",
            "--auto-accept",
            "--no-review",
        )
        playlists = workspace / "playlists"

        run_cli(
            cli_runner,
            "apply",
            "1",
            "--mode",
            "playlist",
            "--playlist-dir",
            str(playlists),
            "--yes",
        )

        written = list(playlists.glob("*.m3u8"))
        assert written
        assert "#EXTM3U" in written[0].read_text()

    def test_applying_in_copy_mode_lands_tracks_in_their_group_folder(
        self, cli_runner, imported, music
    ):
        run_cli(
            cli_runner,
            "sort",
            new_music(music),
            "--collection",
            "Crates",
            "--auto-accept",
            "--no-review",
        )
        before = {
            name: len(list((music / "DJ" / name).glob("*.flac")))
            for name in ("Deep House", "Techno", "Drum & Bass")
        }

        run_cli(cli_runner, "apply", "1", "--mode", "copy", "--yes")

        after = {
            name: len(list((music / "DJ" / name).glob("*.flac")))
            for name in ("Deep House", "Techno", "Drum & Bass")
        }
        assert sum(after.values()) > sum(before.values())
        # The originals stay put in copy mode.
        assert len(list(Path(new_music(music)).glob("*.flac"))) == 6

    def test_move_mode_can_be_undone(self, cli_runner, imported, music):
        run_cli(
            cli_runner,
            "sort",
            new_music(music),
            "--collection",
            "Crates",
            "--auto-accept",
            "--no-review",
        )
        originals = sorted(p.name for p in Path(new_music(music)).glob("*.flac"))

        run_cli(cli_runner, "apply", "1", "--mode", "move", "--yes")
        assert list(Path(new_music(music)).glob("*.flac")) == []

        run_cli(cli_runner, "undo", "1", "--yes")

        assert sorted(p.name for p in Path(new_music(music)).glob("*.flac")) == originals

    def test_committing_folds_the_tracks_back_into_their_groups(
        self, cli_runner, imported, music, workspace
    ):
        run_cli(
            cli_runner,
            "sort",
            new_music(music),
            "--collection",
            "Crates",
            "--auto-accept",
            "--no-review",
        )

        result = run_cli(
            cli_runner,
            "apply",
            "1",
            "--mode",
            "playlist",
            "--playlist-dir",
            str(workspace / "p"),
            "--yes",
        )

        assert "new references" in result.output

    def test_learning_can_be_declined(self, cli_runner, imported, music, workspace):
        run_cli(
            cli_runner,
            "sort",
            new_music(music),
            "--collection",
            "Crates",
            "--auto-accept",
            "--no-review",
        )

        result = run_cli(
            cli_runner,
            "apply",
            "1",
            "--mode",
            "playlist",
            "--playlist-dir",
            str(workspace / "p"),
            "--yes",
            "--no-learn",
        )

        assert "new references" not in result.output

    def test_applying_with_nothing_accepted_says_so(self, cli_runner, imported, music):
        run_cli(cli_runner, "sort", new_music(music), "--collection", "Crates", "--no-review")

        result = run_cli(cli_runner, "apply", "1")

        assert "Nothing to apply" in result.output

    def test_sessions_lists_the_run(self, cli_runner, imported, music):
        run_cli(cli_runner, "sort", new_music(music), "--collection", "Crates", "--no-review")

        result = run_cli(cli_runner, "sessions")

        assert "NewMusic" in result.output
        assert "pending" in result.output

    def test_sorting_an_unknown_collection_fails_with_advice(self, cli_runner, imported, music):
        result = run_cli(
            cli_runner,
            "sort",
            new_music(music),
            "--collection",
            "Nope",
            "--no-review",
            expect_success=False,
        )

        assert result.exit_code != 0
        assert "collection create" in result.output


class TestOneBigUnsortedPile:
    """`discover` → `candidates` → `fit`, the second README workflow.

    `candidates` promotes into the most recent collection, which for a fresh
    library is the one `init` created — so these tests use it rather than
    creating a second one.
    """

    @pytest.fixture
    def discovered(self, cli_runner, music, workspace):
        run_cli(cli_runner, "init")
        return run_cli(
            cli_runner,
            "discover",
            str(music / "DJ"),
            "--algorithm",
            "kmeans",
            "--groups",
            "3",
            "--min-size",
            "4",
        )

    def test_discovery_proposes_candidate_piles(self, discovered):
        assert "3 candidate group(s)" in discovered.output
        assert "music-cluster candidates" in discovered.output

    def test_candidates_are_named_from_the_genre_tags(self, discovered):
        # The synthetic library is tagged, so the suggestions should use the
        # DJ's own vocabulary rather than a tempo band.
        assert any(genre in discovered.output for genre in ("Deep House", "Techno", "Drum & Bass"))

    def test_a_candidate_only_becomes_a_group_when_accepted(self, cli_runner, discovered):
        # Accept the first candidate, discard the other two.
        result = run_cli(cli_runner, "candidates", "1", input_text="y\nRollers\n\nn\nn\n")

        assert "Created 'Rollers'" in result.output

        listed = run_cli(cli_runner, "groups", "list", "--collection", DEFAULT_COLLECTION)
        assert "Rollers" in listed.output
        # The two discarded candidates left nothing behind.
        assert listed.output.count("Rollers") == 1

    def test_seeds_only_promotes_just_the_exemplars(self, cli_runner, discovered):
        result = run_cli(cli_runner, "candidates", "1", input_text="s\nSeeded\n\nq\n")

        assert "Created 'Seeded'" in result.output
        # Exemplars only: far fewer than the 8 tracks in the pile.
        shown = run_cli(cli_runner, "groups", "show", "Seeded", "--collection", DEFAULT_COLLECTION)
        assert "8 tracks" not in shown.output

    def test_quitting_the_review_promotes_nothing(self, cli_runner, discovered):
        run_cli(cli_runner, "candidates", "1", input_text="q\n")

        listed = run_cli(cli_runner, "groups", "list", "--collection", DEFAULT_COLLECTION)
        assert "No groups" in listed.output

    def test_promoted_candidates_can_then_be_fitted_and_sorted_against(
        self, cli_runner, discovered, music
    ):
        run_cli(
            cli_runner,
            "candidates",
            "1",
            input_text="y\nOne\n\ny\nTwo\n\ny\nThree\n\n",
        )

        run_cli(cli_runner, "fit", "--collection", DEFAULT_COLLECTION)
        result = run_cli(
            cli_runner,
            "sort",
            new_music(music),
            "--collection",
            DEFAULT_COLLECTION,
            "--no-review",
        )

        assert "to review" in result.output


class TestLibraryCommands:
    def test_analyze_imports_without_sorting(self, cli_runner, music, workspace):
        run_cli(cli_runner, "init")

        result = run_cli(cli_runner, "analyze", genre_folder(music, "Techno"))

        assert "8" in result.output
        assert "0" in run_cli(cli_runner, "info").output

    def test_search_finds_tracks_and_shows_their_groups(self, cli_runner, imported):
        result = run_cli(cli_runner, "search", "Techno")

        assert "Techno" in result.output

    def test_search_with_no_matches_says_so(self, cli_runner, imported):
        assert "No matches" in run_cli(cli_runner, "search", "zzzznothing").output

    def test_similar_finds_neighbours_of_a_named_track(self, cli_runner, imported):
        result = run_cli(cli_runner, "similar", "Techno Artist", "--limit", "5")

        assert "similar to" in result.output.lower()

    def test_similar_needs_a_track_that_exists(self, cli_runner, imported):
        result = run_cli(cli_runner, "similar", "zzzznothing", expect_success=False)

        assert result.exit_code != 0

    def test_prune_removes_tracks_whose_files_are_gone(self, cli_runner, imported, music):
        removed = next(iter((music / "DJ" / "Techno").glob("*.flac")))
        os.remove(removed)

        result = run_cli(cli_runner, "prune")

        assert "1" in result.output

    def test_groups_export_writes_one_playlist_per_group(self, cli_runner, imported, workspace):
        output = workspace / "exported"

        run_cli(cli_runner, "groups", "export", "--collection", "Crates", "--output", str(output))

        assert len(list(output.glob("*.m3u8"))) == 3

    def test_a_group_can_be_renamed_and_pointed_somewhere_else(
        self, cli_runner, imported, workspace
    ):
        destination = workspace / "elsewhere"

        run_cli(cli_runner, "groups", "rename", "Techno", "Tech House", "--collection", "Crates")
        run_cli(
            cli_runner,
            "groups",
            "set-destination",
            "Tech House",
            str(destination),
            "--collection",
            "Crates",
        )

        shown = run_cli(cli_runner, "groups", "show", "Tech House", "--collection", "Crates")
        assert str(destination) in shown.output

    def test_a_group_can_be_removed(self, cli_runner, imported):
        run_cli(cli_runner, "groups", "remove", "Techno", "--collection", "Crates", "--yes")

        assert (
            "Techno" not in run_cli(cli_runner, "groups", "list", "--collection", "Crates").output
        )

    def test_a_group_can_be_built_from_a_single_folder(self, cli_runner, music, workspace):
        run_cli(cli_runner, "init")
        run_cli(cli_runner, "collection", "create", "Crates")

        run_cli(
            cli_runner,
            "groups",
            "add",
            "My House",
            "--collection",
            "Crates",
            "--folder",
            genre_folder(music, "Deep House"),
        )

        assert "My House" in run_cli(cli_runner, "groups", "list", "--collection", "Crates").output

    def test_a_group_can_be_built_from_named_tracks(self, cli_runner, music, workspace):
        run_cli(cli_runner, "init")
        run_cli(cli_runner, "collection", "create", "Crates")
        tracks = sorted((music / "DJ" / "Techno").glob("*.flac"))[:3]

        args = ["groups", "add", "Seeds", "--collection", "Crates"]
        for path in tracks:
            args += ["--track", str(path)]
        run_cli(cli_runner, *args)

        assert "Seeds" in run_cli(cli_runner, "groups", "list", "--collection", "Crates").output


class TestCollections:
    def test_collections_are_listed_after_creation(self, cli_runner, workspace):
        run_cli(cli_runner, "init")
        run_cli(cli_runner, "collection", "create", "Crates")
        run_cli(cli_runner, "collection", "create", "Archive")

        result = run_cli(cli_runner, "collection", "list")

        assert "Crates" in result.output
        assert "Archive" in result.output

    def test_a_collection_can_be_deleted(self, cli_runner, workspace):
        run_cli(cli_runner, "init")
        run_cli(cli_runner, "collection", "create", "Crates")

        run_cli(cli_runner, "collection", "delete", "Crates", "--yes")

        assert "Crates" not in run_cli(cli_runner, "collection", "list").output

    def test_groups_stay_inside_their_own_collection(self, cli_runner, music, workspace):
        run_cli(cli_runner, "init")
        run_cli(cli_runner, "collection", "create", "Crates")
        run_cli(cli_runner, "collection", "create", "Archive")
        run_cli(
            cli_runner,
            "groups",
            "add",
            "House",
            "--collection",
            "Crates",
            "--folder",
            genre_folder(music, "Deep House"),
        )

        assert (
            "House" not in run_cli(cli_runner, "groups", "list", "--collection", "Archive").output
        )


class TestConfigCommand:
    def test_setting_a_key_persists_it(self, cli_runner, workspace):
        run_cli(cli_runner, "init")

        run_cli(cli_runner, "config", "--set", "sorting.neighbors=8")

        assert "neighbors: 8" in run_cli(cli_runner, "config").output

    def test_values_are_coerced_to_the_right_type(self, cli_runner, workspace):
        run_cli(cli_runner, "init")

        run_cli(
            cli_runner,
            "config",
            "--set",
            "sorting.learn_on_commit=false",
            "--set",
            "sorting.knn_weight=0.4",
        )

        printed = json.loads(run_cli(cli_runner, "config", "--json").output)
        assert printed["sorting"]["learn_on_commit"] is False
        assert printed["sorting"]["knn_weight"] == 0.4

    def test_a_malformed_assignment_is_rejected(self, cli_runner, workspace):
        run_cli(cli_runner, "init")

        result = run_cli(cli_runner, "config", "--set", "nonsense", expect_success=False)

        assert result.exit_code != 0
        assert "KEY=VALUE" in result.output
