"""A real music library on disk, for tests that drive the whole application.

Everything here goes through the actual feature extractor, the actual database,
and the actual CLI or HTTP layer — no stubs. The audio is synthesised rather
than shipped: three families of tones that are as distinguishable to the
extractor as three genre folders are to a DJ, so a sorter fitted on them
genuinely has something to learn.
"""

from pathlib import Path

import numpy as np
import pytest


soundfile = pytest.importorskip("soundfile")
from mutagen.flac import FLAC  # noqa: E402  (import after the availability check)


SAMPLE_RATE = 22050
SECONDS = 4.0

# (folder, carrier frequency, pulse rate, genre tag, tag BPM)
FAMILIES = [
    ("Deep House", 110.0, 2.0, "Deep House", 122),
    ("Techno", 440.0, 4.0, "Techno", 132),
    ("Drum & Bass", 1800.0, 8.0, "Drum & Bass", 174),
]


def write_track(path: Path, freq: float, pulse: float, jitter: float = 0.0) -> str:
    """A pulsed tone with enough structure for every feature family to exist."""
    time = np.linspace(0, SECONDS, int(SAMPLE_RATE * SECONDS), endpoint=False)
    envelope = 0.5 * (1 + np.sign(np.sin(2 * np.pi * pulse * time)))
    carrier = np.sin(2 * np.pi * freq * (1 + jitter) * time)
    # A second partial gives the spectral shape something to vary with.
    overtone = 0.3 * np.sin(2 * np.pi * freq * 2.5 * (1 + jitter) * time)
    signal = envelope * (carrier + overtone) * 0.4

    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(str(path), signal.astype(np.float32), SAMPLE_RATE)
    return str(path)


def tag(path: str, artist: str, title: str, genre: str, bpm: int) -> None:
    audio = FLAC(path)
    audio["artist"] = [artist]
    audio["title"] = [title]
    audio["genre"] = [genre]
    audio["bpm"] = [str(bpm)]
    audio.save()


@pytest.fixture(scope="session")
def library_root(tmp_path_factory) -> Path:
    """Three tagged genre folders of eight tracks each, plus a folder of new buys.

    Session-scoped: synthesising and tagging the audio once keeps the
    end-to-end suite to a few seconds. Tests copy it into their own workspace
    rather than writing to it.
    """
    root = tmp_path_factory.mktemp("music")
    rng = np.random.default_rng(20240501)

    collection = root / "DJ"
    for folder, freq, pulse, genre, bpm in FAMILIES:
        for index in range(8):
            path = collection / folder / f"{folder} Artist - Track {index + 1}.flac"
            write_track(path, freq, pulse, jitter=float(rng.normal(0, 0.01)))
            tag(str(path), f"{folder} Artist", f"Track {index + 1}", genre, bpm)

    # New music to sort: two tracks from each family, in one flat folder.
    incoming = root / "NewMusic"
    for folder, freq, pulse, genre, bpm in FAMILIES:
        for index in range(2):
            path = incoming / f"New {folder} {index + 1}.flac"
            write_track(path, freq, pulse, jitter=float(rng.normal(0, 0.01)))
            tag(str(path), "New Artist", f"New {folder} {index + 1}", genre, bpm)

    return root


@pytest.fixture
def workspace(tmp_path, library_root, monkeypatch) -> Path:
    """An isolated home for one test: its own database, config, and copy of the library."""
    import shutil

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copytree(library_root, workspace / "music")

    monkeypatch.setenv("MUSIC_CLUSTER_DB", str(workspace / "library.db"))
    monkeypatch.setenv("MUSIC_CLUSTER_CONFIG", str(workspace / "config.yaml"))
    monkeypatch.setenv("MUSIC_CLUSTER_CACHE", str(workspace / "cache"))
    # Analysing the whole (short) file keeps features stable for 4-second tracks.
    monkeypatch.setenv("HOME", str(workspace))

    return workspace


@pytest.fixture
def music(workspace) -> Path:
    return workspace / "music"


def genre_folder(music: Path, name: str) -> str:
    return str(music / "DJ" / name)


def new_music(music: Path) -> str:
    return str(music / "NewMusic")


@pytest.fixture
def cli_runner():
    from click.testing import CliRunner

    return CliRunner()


def run_cli(runner, *args, input_text=None, expect_success=True):
    """Invoke the CLI the way a user would, and fail loudly on an unexpected error."""
    from music_cluster.cli import cli

    result = runner.invoke(cli, list(args), input=input_text, catch_exceptions=False)
    if expect_success and result.exit_code != 0:
        raise AssertionError(
            f"`music-cluster {' '.join(args)}` exited {result.exit_code}:\n{result.output}"
        )
    return result
