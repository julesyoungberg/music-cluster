"""The evaluation harness end to end on its synthetic suites (no downloads)."""

import json

import pytest
from click.testing import CliRunner

from music_cluster.cli import cli


pytestmark = pytest.mark.slow


def test_eval_run_writes_a_report_and_compares_against_a_baseline(temp_dir):
    out = temp_dir / "report.json"
    runner = CliRunner()
    result = runner.invoke(
        cli, ["eval", "run", "--suite", "beats-synthetic", "--quick", "--out", str(out)]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(out.read_text())
    headline = data["suites"]["beats-synthetic"]["headline"]
    assert 0.0 <= headline["beat_accuracy"] <= 1.0
    assert out.with_suffix(".md").exists()

    # Comparing a report against itself never regresses.
    result = runner.invoke(cli, ["eval", "compare", str(out), str(out)])
    assert result.exit_code == 0, result.output


def test_a_suite_without_its_data_is_skipped_not_fatal(temp_dir):
    out = temp_dir / "report.json"
    result = CliRunner().invoke(
        cli,
        [
            "eval",
            "run",
            "--suite",
            "key-giantsteps",
            "--data-dir",
            str(temp_dir),
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "skipped" in json.loads(out.read_text())["suites"]["key-giantsteps"]
