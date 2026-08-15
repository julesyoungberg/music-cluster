"""The feature layout is the contract between the extractor and everything else."""

import numpy as np
import pytest

from music_cluster.features import (
    FAMILIES,
    build_layout,
    describe_vector,
    family_weight_vector,
    layout_for_dim,
)


def test_default_layout_dimensions():
    layout = build_layout(20)
    # 43 frame features x (mean, std) + 4 rhythm + 6 high-level
    assert layout.dim == 96


@pytest.mark.parametrize("n_mfcc", [13, 20, 40])
def test_layout_scales_with_mfcc_count(n_mfcc):
    layout = build_layout(n_mfcc)
    assert layout.dim == 2 * (n_mfcc + 23) + 10
    assert layout.span("mfcc_mean") == (0, n_mfcc)


def test_spans_are_contiguous_and_complete():
    layout = build_layout(20)
    spans = sorted(layout.spans.values())
    assert spans[0][0] == 0
    assert spans[-1][1] == layout.dim
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end == start, "layout has a gap or an overlap"


def test_every_index_belongs_to_exactly_one_family():
    layout = build_layout(20)
    seen = np.zeros(layout.dim, dtype=int)
    for family in FAMILIES:
        seen[layout.family_indices(family)] += 1
    assert np.all(seen == 1)


def test_layout_recovered_from_dimension():
    for n_mfcc in (13, 20, 40):
        layout = build_layout(n_mfcc)
        assert layout_for_dim(layout.dim).n_mfcc == n_mfcc


def test_family_weights_only_touch_their_own_family():
    layout = build_layout(20)
    weights = family_weight_vector(layout, {"rhythm": 3.0})
    rhythm = layout.family_indices("rhythm")
    assert np.all(weights[rhythm] == 3.0)
    others = np.setdiff1d(np.arange(layout.dim), rhythm)
    assert np.all(weights[others] == 1.0)


def test_describe_vector_reads_the_right_slots():
    """Descriptors must read the physical quantity they claim to."""
    layout = build_layout(20)
    vector = np.zeros(layout.dim)
    vector[layout.span("tempo")[0]] = 128.0
    vector[layout.span("spectral_centroid_mean")[0]] = 2500.0
    vector[layout.span("energy")[0]] = 0.42

    described = describe_vector(vector)
    assert described["bpm"] == 128.0
    assert described["brightness_hz"] == 2500.0
    assert described["energy"] == pytest.approx(0.42)
