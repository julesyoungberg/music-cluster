"""Clustering, which only ever proposes candidate piles."""

import numpy as np
import pytest

from music_cluster.clustering import (
    ALGORITHMS,
    GRANULARITY_MULTIPLIERS,
    NOISE_LABEL,
    ClusterEngine,
    select_exemplars,
)


def blobs(rng, groups=3, per_group=30, dim=8, separation=15.0):
    """Well-separated blobs — any working algorithm should recover these."""
    centres = [rng.normal(0, 1, dim) * separation for _ in range(groups)]
    return np.vstack([centre + rng.normal(0, 1, (per_group, dim)) for centre in centres])


class TestConstruction:
    def test_rejects_an_unknown_algorithm(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            ClusterEngine(algorithm="magic")

    def test_clamps_nonsensical_sizes(self):
        engine = ClusterEngine(algorithm="kmeans", min_group_size=0, max_candidates=1)
        assert engine.min_group_size == 2
        assert engine.max_candidates == 2


class TestCluster:
    @pytest.mark.parametrize("algorithm", ALGORITHMS)
    def test_recovers_separated_blobs(self, rng, algorithm):
        features = blobs(rng)

        labels, centroids, metrics = ClusterEngine(algorithm=algorithm, min_group_size=5).cluster(
            features, n_clusters=3
        )

        assigned = [label for label in np.unique(labels) if label != NOISE_LABEL]
        assert len(assigned) >= 2
        assert len(centroids) == len(assigned)
        assert metrics["algorithm"] == algorithm
        assert metrics["n_candidates"] == len(assigned)

    def test_refuses_to_cluster_fewer_tracks_than_a_group_needs(self, rng):
        with pytest.raises(ValueError, match="at least"):
            ClusterEngine(algorithm="kmeans", min_group_size=10).cluster(blobs(rng, per_group=2))

    def test_an_explicit_cluster_count_is_honoured(self, rng):
        labels, centroids, _ = ClusterEngine(algorithm="kmeans", min_group_size=3).cluster(
            blobs(rng, groups=4, per_group=25), n_clusters=4
        )
        assert len(np.unique(labels)) == 4
        assert len(centroids) == 4

    def test_the_cluster_count_is_clamped_to_what_the_data_can_support(self, rng):
        # Asking for 50 groups from 30 tracks would produce singleton piles that
        # tell the user nothing.
        features = blobs(rng, groups=2, per_group=15)
        labels, _, _ = ClusterEngine(
            algorithm="kmeans", min_group_size=3, max_candidates=6
        ).cluster(features, n_clusters=50)
        assert len(np.unique(labels)) <= 6

    def test_centroids_sit_at_the_middle_of_their_members(self, rng):
        features = blobs(rng, groups=3, per_group=20)
        labels, centroids, _ = ClusterEngine(algorithm="kmeans", min_group_size=3).cluster(
            features, n_clusters=3
        )

        for position, label in enumerate(sorted(np.unique(labels))):
            expected = features[labels == label].mean(axis=0)
            assert np.allclose(centroids[position], expected)

    def test_metrics_report_sizes_and_quality(self, rng):
        _, _, metrics = ClusterEngine(algorithm="kmeans", min_group_size=3).cluster(
            blobs(rng), n_clusters=3
        )

        assert sum(metrics["candidate_sizes"].values()) == 90
        assert metrics["silhouette_score"] > 0.5
        assert metrics["n_noise"] == 0

    def test_hdbscan_marks_outliers_as_noise_rather_than_forcing_them_into_a_pile(self, rng):
        tight = blobs(rng, groups=2, per_group=40, separation=30.0)
        strays = rng.normal(0, 1, (5, tight.shape[1])) * 200
        features = np.vstack([tight, strays])

        labels, centroids, metrics = ClusterEngine(algorithm="hdbscan", min_group_size=8).cluster(
            features
        )

        assert metrics["n_noise"] == int(np.sum(labels == NOISE_LABEL))
        # Noise has no centroid: a pile of outliers is not a group.
        assert len(centroids) == len([label for label in np.unique(labels) if label != NOISE_LABEL])


class TestEstimateK:
    def test_lands_near_the_true_number_of_blobs(self, rng):
        engine = ClusterEngine(algorithm="kmeans", min_group_size=5)
        assert 2 <= engine.estimate_k(blobs(rng, groups=3, per_group=40)) <= 5

    def test_never_returns_fewer_than_two(self, rng):
        engine = ClusterEngine(algorithm="kmeans", min_group_size=2)
        assert engine.estimate_k(rng.normal(0, 1, (6, 4))) >= 2

    def test_the_calinski_method_also_produces_a_usable_count(self, rng):
        engine = ClusterEngine(algorithm="kmeans", min_group_size=5, detection_method="calinski")
        assert engine.estimate_k(blobs(rng, groups=3, per_group=40)) >= 2


class TestGranularity:
    @pytest.mark.parametrize("granularity", sorted(GRANULARITY_MULTIPLIERS))
    def test_scales_the_estimate(self, granularity):
        scaled = ClusterEngine.apply_granularity(10, granularity)
        assert scaled == max(2, round(10 * GRANULARITY_MULTIPLIERS[granularity]))

    def test_finer_never_yields_fewer_groups_than_coarser(self):
        order = ["coarse", "broad", "normal", "fine", "finest"]
        counts = [ClusterEngine.apply_granularity(12, name) for name in order]
        assert counts == sorted(counts)

    def test_an_unknown_granularity_leaves_the_estimate_alone(self):
        assert ClusterEngine.apply_granularity(9, "whatever") == 9

    def test_never_proposes_fewer_than_two_groups(self):
        assert ClusterEngine.apply_granularity(1, "coarse") == 2


class TestSelectExemplars:
    def test_returns_every_member_of_a_small_pile(self):
        features = np.random.default_rng(0).normal(0, 1, (10, 4))
        assert select_exemplars(features, [1, 2, 3], count=5) == [1, 2, 3]

    def test_an_empty_pile_has_no_exemplars(self):
        assert select_exemplars(np.zeros((5, 3)), [], count=5) == []

    def test_returns_exactly_the_requested_count(self, rng):
        features = rng.normal(0, 1, (50, 6))
        chosen = select_exemplars(features, list(range(50)), count=5)
        assert len(chosen) == 5
        assert len(set(chosen)) == 5

    def test_starts_from_the_most_typical_track(self):
        # Members clustered around the origin plus one far outlier: the medoid
        # must be a central member, not the outlier.
        features = np.vstack([np.zeros((8, 3)) + 0.01, np.array([[50.0, 50.0, 50.0]])])
        chosen = select_exemplars(features, list(range(9)), count=2)
        assert chosen[0] != 8

    def test_covers_the_spread_rather_than_the_average(self):
        # Two tight clumps far apart: five exemplars should show both, not five
        # near-identical tracks from one of them.
        left = np.zeros((20, 3))
        right = np.zeros((20, 3)) + 100.0
        features = np.vstack([left, right])

        chosen = select_exemplars(features, list(range(40)), count=5)

        assert any(index < 20 for index in chosen)
        assert any(index >= 20 for index in chosen)

    def test_indices_refer_to_the_original_feature_matrix(self, rng):
        features = rng.normal(0, 1, (30, 4))
        members = [3, 7, 11, 15, 19, 23, 27]
        assert set(select_exemplars(features, members, count=4)) <= set(members)
