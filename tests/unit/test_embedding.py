"""The metric space that sorting decisions are measured in."""

import numpy as np
import pytest

from music_cluster.embedding import PROJECTIONS, EmbeddingSpace
from music_cluster.features import build_layout


LAYOUT = build_layout(20)


def labelled(rng, groups=3, per_group=20, separation=8.0):
    """Feature vectors with group labels, well enough separated for LDA."""
    centres = [rng.normal(0, 1, LAYOUT.dim) * separation for _ in range(groups)]
    features = np.vstack([centre + rng.normal(0, 1, (per_group, LAYOUT.dim)) for centre in centres])
    labels = np.repeat(np.arange(groups), per_group)
    return features, labels


class TestFit:
    def test_rejects_something_that_is_not_a_feature_matrix(self):
        with pytest.raises(ValueError, match="2-D feature matrix"):
            EmbeddingSpace().fit(np.zeros((0, LAYOUT.dim)))
        with pytest.raises(ValueError, match="2-D feature matrix"):
            EmbeddingSpace().fit(np.zeros(LAYOUT.dim))

    def test_records_the_dimensions_it_was_fitted_on(self, rng):
        features, labels = labelled(rng)
        space = EmbeddingSpace().fit(features, labels)

        assert space.input_dim == LAYOUT.dim
        assert space.output_dim == space.transform(features).shape[1]

    def test_auto_picks_a_supervised_projection_when_labels_support_it(self, rng):
        features, labels = labelled(rng)
        assert EmbeddingSpace(projection="auto").fit(features, labels).resolved_projection == "lda"

    def test_auto_falls_back_to_pca_without_labels(self, rng):
        features, _ = labelled(rng)
        assert EmbeddingSpace(projection="auto").fit(features).resolved_projection == "pca"

    def test_a_requested_supervised_projection_degrades_to_pca_when_unsupportable(self, rng):
        # Two tracks per group is not enough to estimate within-class scatter;
        # a badly overfitted rotation is worse than no rotation.
        features, labels = labelled(rng, groups=3, per_group=2)
        space = EmbeddingSpace(projection="lda").fit(features, labels)
        assert space.resolved_projection == "pca"

    def test_projection_none_leaves_the_dimensionality_alone(self, rng):
        features, labels = labelled(rng)
        space = EmbeddingSpace(projection="none").fit(features, labels)

        assert space.reducer is None
        assert space.transform(features).shape[1] == LAYOUT.dim

    def test_rejects_an_unknown_projection(self, rng):
        features, _ = labelled(rng)
        with pytest.raises(ValueError, match="Unknown projection"):
            EmbeddingSpace(projection="tsne").fit(features)

    @pytest.mark.parametrize("projection", PROJECTIONS)
    def test_every_advertised_projection_produces_a_usable_space(self, rng, projection):
        features, labels = labelled(rng)
        space = EmbeddingSpace(projection=projection).fit(features, labels)

        projected = space.transform(features)
        assert projected.shape[0] == len(features)
        assert np.all(np.isfinite(projected))


class TestTransform:
    def test_refuses_to_transform_before_being_fitted(self):
        with pytest.raises(RuntimeError, match="must be fitted"):
            EmbeddingSpace().transform(np.zeros((1, LAYOUT.dim)))

    def test_rejects_a_vector_of_the_wrong_width(self, rng):
        features, labels = labelled(rng)
        space = EmbeddingSpace().fit(features, labels)

        with pytest.raises(ValueError, match="dimensional features"):
            space.transform(np.zeros((1, LAYOUT.dim + 5)))

    def test_a_single_vector_comes_back_as_one_row(self, rng):
        features, labels = labelled(rng)
        space = EmbeddingSpace().fit(features, labels)

        single = space.transform(features[0])
        assert single.shape == (1, space.output_dim)
        assert np.allclose(single[0], space.transform(features[:1])[0])

    def test_the_same_input_always_projects_to_the_same_point(self, rng):
        features, labels = labelled(rng)
        space = EmbeddingSpace().fit(features, labels)

        assert np.allclose(space.transform(features), space.transform(features))


class TestSupervision:
    def test_supervised_projection_separates_the_labelled_groups(self, rng):
        features, labels = labelled(rng, separation=4.0)
        space = EmbeddingSpace(projection="lda", discriminant_weight=0.9).fit(features, labels)
        projected = space.transform(features)

        centroids = np.vstack([projected[labels == g].mean(axis=0) for g in np.unique(labels)])
        within = np.mean(
            [
                np.linalg.norm(projected[labels == g] - centroids[g], axis=1).mean()
                for g in np.unique(labels)
            ]
        )
        between = np.mean(
            [
                np.linalg.norm(centroids[a] - centroids[b])
                for a in range(len(centroids))
                for b in range(a + 1, len(centroids))
            ]
        )

        assert between > within

    def test_a_blend_keeps_more_dimensions_than_pure_lda_would(self, rng):
        # Pure LDA on three groups gives two axes, which leaves no room to tell
        # "far from everything" from "between two groups".
        features, labels = labelled(rng, groups=3)
        space = EmbeddingSpace(projection="lda", discriminant_weight=0.7).fit(features, labels)

        assert space.output_dim > 2

    def test_the_discriminant_weight_changes_the_geometry(self, rng):
        features, labels = labelled(rng)

        low = EmbeddingSpace(projection="lda", discriminant_weight=0.1).fit(features, labels)
        high = EmbeddingSpace(projection="lda", discriminant_weight=0.9).fit(features, labels)

        assert not np.allclose(low.transform(features), high.transform(features))


class TestFeatureWeights:
    def test_weighting_a_family_changes_the_projection(self, rng):
        features, labels = labelled(rng)

        plain = EmbeddingSpace(projection="pca").fit(features, labels)
        rhythmic = EmbeddingSpace(projection="pca", feature_weights={"rhythm": 4.0}).fit(
            features, labels
        )

        plain_points = plain.transform(features)
        rhythmic_points = rhythmic.transform(features)

        assert plain_points.shape != rhythmic_points.shape or not np.allclose(
            plain_points, rhythmic_points
        )

    def test_all_ones_is_the_same_as_no_weighting(self, rng):
        features, labels = labelled(rng)

        plain = EmbeddingSpace(projection="pca").fit(features, labels)
        ones = EmbeddingSpace(
            projection="pca",
            feature_weights=dict.fromkeys(("timbre", "rhythm", "harmony", "dynamics"), 1.0),
        ).fit(features, labels)

        assert np.allclose(plain.transform(features), ones.transform(features))


class TestDescribe:
    def test_reports_what_was_asked_for_and_what_was_used(self, rng):
        features, labels = labelled(rng)
        described = EmbeddingSpace(projection="auto").fit(features, labels).describe()

        assert described["requested_projection"] == "auto"
        assert described["projection"] == "lda"
        assert described["input_dim"] == LAYOUT.dim
        assert described["output_dim"] > 0

    def test_is_json_serialisable(self, rng):
        import json

        features, labels = labelled(rng)
        json.dumps(EmbeddingSpace().fit(features, labels).describe())
