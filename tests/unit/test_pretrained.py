"""Batching in pretrained.embed_files: each file gets its own windows' mean."""

import numpy as np
import pytest

from music_cluster import pretrained


def test_each_file_pools_only_its_own_windows(monkeypatch):
    # Window count per file, None for one that would not decode; 11 files span two batches.
    counts = {f"{i}.mp3": (None if i == 3 else 1 + i % 3) for i in range(11)}

    def windows(path, model):
        n = counts[path]
        return None if n is None else [np.full(4, float(path.split(".")[0]))] * n

    monkeypatch.setattr(pretrained, "windows", windows)
    rows = pretrained.embed_files(list(counts), "ast", embedder=lambda clips: np.stack(clips))

    assert rows[3] is None
    for i, row in enumerate(rows):
        if i != 3:
            # Unit-length windows of one constant value pool to 0.5 in every column.
            assert np.allclose(row, 0.5 if i else 0.0)


def test_a_model_returning_the_wrong_row_count_fails(monkeypatch):
    monkeypatch.setattr(pretrained, "windows", lambda path, model: [np.ones(4)])

    with pytest.raises(RuntimeError, match="rows for 2 clips"):
        pretrained.embed_files(["a", "b"], "ast", embedder=lambda clips: np.ones((1, 4)))
