"""A real music library on disk, for tests that drive the whole application.

Everything here goes through the actual feature extractor, the actual database,
and the actual CLI or HTTP layer — no stubs. The audio itself is described in
:mod:`tests.e2e.fixtures`, which the browser tests seed from too.
"""

import shutil
from pathlib import Path

import pytest


pytest.importorskip("soundfile")

from tests.e2e.fixtures import build_library


@pytest.fixture(scope="session")
def library_root(tmp_path_factory) -> Path:
    """Three tagged genre folders of eight tracks each, plus a folder of new buys.

    Session-scoped: synthesising and tagging the audio once keeps the
    end-to-end suite to about a minute. Tests copy it into their own workspace
    rather than writing to it.
    """
    return build_library(tmp_path_factory.mktemp("music"))


@pytest.fixture
def workspace(tmp_path, library_root, monkeypatch) -> Path:
    """An isolated home for one test: its own database, config, and copy of the library."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    shutil.copytree(library_root, workspace / "music")

    monkeypatch.setenv("MUSIC_CLUSTER_DB", str(workspace / "library.db"))
    monkeypatch.setenv("MUSIC_CLUSTER_CONFIG", str(workspace / "config.yaml"))
    monkeypatch.setenv("MUSIC_CLUSTER_CACHE", str(workspace / "cache"))
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
