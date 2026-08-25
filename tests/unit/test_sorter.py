"""Scoring behaviour — the part that decides where a track goes."""

import numpy as np
import pytest

from music_cluster.embedding import EmbeddingSpace
from music_cluster.sorter import (
    STATUS_AUTO,
    STATUS_UNMATCHED,
    GroupSorter,
    SorterConfig,
)
from tests.conftest import FEATURE_DIM, make_centres, make_vector


def build_sorter(rng, sizes=(20, 20, 20), separation=4.0, config=None):
    """Fit a sorter on synthetic groups; returns (sorter, centres, group_ids)."""
    centres = make_centres(rng, len(sizes), separation=separation)
    features = {}
    names = {}
    for index, (centre, size) in enumerate(zip(centres, sizes, strict=True)):
        gid = index + 1
        features[gid] = np.vstack([make_vector(rng, centre) for _ in range(size)])
        names[gid] = f"Group {index}"
    sorter = GroupSorter(config or SorterConfig()).fit(features, names)
    return sorter, centres, list(features.keys())


def test_separated_groups_are_sorted_correctly(rng):
    sorter, centres, group_ids = build_sorter(rng)
    for expected_gid, centre in zip(group_ids, centres, strict=True):
        suggestion = sorter.suggest(make_vector(rng, centre))
        assert suggestion.group_id == expected_gid
        assert suggestion.status == STATUS_AUTO


def test_distant_track_is_unmatched(rng):
    sorter, _, _ = build_sorter(rng)
    suggestion = sorter.suggest(np.full(FEATURE_DIM, 500.0))
    assert suggestion.status == STATUS_UNMATCHED


def test_ranked_list_is_ordered_and_normalised(rng):
    sorter, centres, _ = build_sorter(rng)
    suggestion = sorter.suggest(make_vector(rng, centres[0]))

    distances = [entry["distance"] for entry in suggestion.ranked]
    assert distances == sorted(distances)
    assert sum(entry["score"] for entry in suggestion.ranked) == pytest.approx(1.0, abs=1e-6)
    assert suggestion.confidence == pytest.approx(suggestion.ranked[0]["score"])


def test_a_small_group_survives_beside_a_large_one(rng):
    """A tiny seed group must not be drowned out by a huge folder.

    Without per-group scale normalisation the large group wins on sheer
    coverage, which is exactly the failure mode a DJ with one 900-track folder
    and one 8-track crate would hit.
    """
    centres = make_centres(rng, 2, separation=3.0)
    features = {
        1: np.vstack([make_vector(rng, centres[0], spread=1.4) for _ in range(300)]),
        2: np.vstack([make_vector(rng, centres[1], spread=0.6) for _ in range(6)]),
    }
    sorter = GroupSorter().fit(features, {1: "Big folder", 2: "Small crate"})

    hits = sum(
        sorter.suggest(make_vector(rng, centres[1], spread=0.6)).group_id == 2 for _ in range(20)
    )
    assert hits >= 18


def test_overlapping_groups_lower_confidence(rng):
    """Ambiguity should show up as low confidence, not a confident wrong answer."""
    separated, clear_centres, _ = build_sorter(rng, separation=6.0)
    overlapping, murky_centres, _ = build_sorter(rng, separation=0.35)

    clear = np.mean(
        [separated.suggest(make_vector(rng, clear_centres[0])).confidence for _ in range(20)]
    )
    murky = np.mean(
        [overlapping.suggest(make_vector(rng, murky_centres[0])).confidence for _ in range(20)]
    )
    assert murky < clear


def test_evaluate_reports_perfect_separation(rng):
    sorter, _, _ = build_sorter(rng, separation=6.0)
    report = sorter.evaluate()

    assert report["accuracy"] > 0.95
    assert report["n_groups"] == 3
    assert report["n_tracks"] == 60
    assert len(report["confusion"]) == 3
    # Every reference track is accounted for in the confusion matrix.
    assert sum(sum(row) for row in report["confusion"]) == 60


def test_evaluate_flags_confusable_pairs(rng):
    sorter, _, _ = build_sorter(rng, separation=0.2)
    report = sorter.evaluate()

    assert report["accuracy"] < 0.95
    assert report["confusable_pairs"], "overlapping groups should be reported"
    assert report["confusable_pairs"][0]["count"] > 0


def test_round_trip_through_serialisation(rng):
    sorter, centres, _ = build_sorter(rng)
    probe = make_vector(rng, centres[1])
    before = sorter.suggest(probe)

    restored = GroupSorter.loads(sorter.dumps())
    after = restored.suggest(probe)

    assert after.group_id == before.group_id
    assert after.confidence == pytest.approx(before.confidence)
    assert after.distance == pytest.approx(before.distance)


def test_fit_requires_two_usable_groups(rng):
    with pytest.raises(ValueError, match="at least two groups"):
        GroupSorter().fit({1: np.zeros((5, FEATURE_DIM)), 2: np.empty((0, FEATURE_DIM))})


def test_thresholds_move_the_decision(rng):
    """Raising the bar sends more tracks to the human queue."""
    lenient, centres, _ = build_sorter(
        rng, separation=0.5, config=SorterConfig(auto_accept_confidence=0.35, min_margin=0.0)
    )
    strict, strict_centres, _ = build_sorter(
        rng, separation=0.5, config=SorterConfig(auto_accept_confidence=0.98, min_margin=0.9)
    )

    lenient_auto = sum(
        lenient.suggest(make_vector(rng, centres[0])).status == STATUS_AUTO for _ in range(15)
    )
    strict_auto = sum(
        strict.suggest(make_vector(rng, strict_centres[0])).status == STATUS_AUTO for _ in range(15)
    )
    assert strict_auto < lenient_auto


class TestEmbedding:
    def test_supervision_is_used_when_groups_are_labelled(self, rng):
        centres = make_centres(rng, 3)
        features = np.vstack([make_vector(rng, centre) for centre in centres for _ in range(15)])
        labels = np.repeat([1, 2, 3], 15)

        space = EmbeddingSpace(projection="auto").fit(features, labels)
        assert space.resolved_projection == "lda"

    def test_falls_back_to_pca_without_labels(self, rng):
        features = rng.normal(0, 1, (40, FEATURE_DIM))
        space = EmbeddingSpace(projection="auto").fit(features)
        assert space.resolved_projection == "pca"

    def test_falls_back_when_groups_are_too_small_for_supervision(self, rng):
        features = rng.normal(0, 1, (4, FEATURE_DIM))
        labels = np.array([1, 1, 2, 2])
        space = EmbeddingSpace(projection="lda").fit(features, labels)
        assert space.resolved_projection == "pca"

    def test_blend_keeps_more_than_the_discriminant_axes(self, rng):
        """LDA alone gives n_groups-1 dims; the blend must keep general structure."""
        centres = make_centres(rng, 3)
        features = np.vstack([make_vector(rng, centre) for centre in centres for _ in range(15)])
        labels = np.repeat([1, 2, 3], 15)

        space = EmbeddingSpace(projection="lda").fit(features, labels)
        assert space.output_dim > 2

    def test_transform_rejects_wrong_dimensionality(self, rng):
        space = EmbeddingSpace(projection="pca").fit(rng.normal(0, 1, (20, FEATURE_DIM)))
        with pytest.raises(ValueError, match="dimensional"):
            space.transform(np.zeros(FEATURE_DIM + 1))

    def test_feature_weights_change_the_geometry(self, rng):
        features = rng.normal(0, 1, (30, FEATURE_DIM))
        plain = EmbeddingSpace(projection="none").fit(features)
        weighted = EmbeddingSpace(projection="none", feature_weights={"rhythm": 5.0}).fit(features)

        assert not np.allclose(plain.transform(features), weighted.transform(features))
