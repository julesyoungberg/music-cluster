"""Rhythm checks that need only a folded bar, not audio."""

import numpy as np

from music_cluster import rhythm


def _kick(steps):
    bar = np.zeros(rhythm.STEPS_PER_BAR)
    bar[list(steps)] = 1.0
    return {"kick": bar.tolist()}


def test_a_kick_on_every_third_of_the_bar_marks_a_grid_too_fast():
    # A four-to-the-floor read at 4/3 of its tempo: kicks every 16 steps.
    assert rhythm._kick_on_thirds(_kick([0, 16, 32]))


def test_a_kick_on_the_beats_is_not_on_thirds():
    assert not rhythm._kick_on_thirds(_kick([0, 12, 24, 36]))


def test_an_odd_sixteenth_kick_beside_a_triplet_point_is_not_on_thirds():
    # A backbeat's kick on sixteenth 7 is widened onto step 20 by the grid.
    assert not rhythm._kick_on_thirds(_kick([0, 20, 21, 22, 24]))
