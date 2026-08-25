"""Shared fixtures.

Tests use synthetic feature vectors rather than audio wherever possible:
extraction is slow, and nothing above the extractor cares where the numbers
came from. The one test that does need real audio generates it.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from music_cluster.config import Config
from music_cluster.database import Database
from music_cluster.features import build_layout


LAYOUT = build_layout(20)
FEATURE_DIM = LAYOUT.dim


@pytest.fixture
def temp_dir():
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def db(temp_dir) -> Database:
    return Database(str(temp_dir / "library.db"))


@pytest.fixture
def config(temp_dir) -> Config:
    config = Config(str(temp_dir / "config.yaml"))
    config.set(["database", "path"], str(temp_dir / "library.db"))
    return config


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


def make_vector(rng: np.random.Generator, centre: np.ndarray, spread: float = 1.0) -> np.ndarray:
    """A feature vector near a given centre."""
    return centre + rng.normal(0, spread, FEATURE_DIM)


def make_centres(rng: np.random.Generator, count: int, separation: float = 4.0) -> list:
    """Well-separated centres to act as synthetic genres."""
    return [rng.normal(0, 1, FEATURE_DIM) * separation for _ in range(count)]


@pytest.fixture
def populated(db, rng):
    """A collection with three well-separated groups of 12 tracks each.

    Returns ``(collection_id, {group_id: [track_id, ...]})``.
    """
    collection_id = db.create_collection("Test collection")
    centres = make_centres(rng, 3)
    membership = {}

    for index, centre in enumerate(centres):
        group_id = db.create_group(collection_id, f"Group {index}")
        track_ids = []
        for position in range(12):
            track_id = db.upsert_track(
                filepath=f"/fake/group{index}/track{position}.mp3",
                filename=f"track{position}.mp3",
                duration=300.0,
                metadata={
                    "artist": f"Artist {index}",
                    "title": f"Track {position}",
                    "genre": ["House", "Techno", "Drum & Bass"][index],
                    "tag_bpm": 120.0 + index * 15,
                },
            )
            db.add_features(track_id, make_vector(rng, centre))
            track_ids.append(track_id)
        db.add_group_members(group_id, track_ids)
        membership[group_id] = track_ids

    return collection_id, membership, centres
