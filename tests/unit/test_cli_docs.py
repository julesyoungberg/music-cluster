"""docs/cli.md is the reference for every command, so it must name each one and each option."""

from pathlib import Path
from typing import Iterator, Tuple

import click

from music_cluster.cli import cli


CLI_DOC = Path(__file__).resolve().parents[2] / "docs" / "cli.md"


def _commands(group: click.Group, prefix: str = "") -> Iterator[Tuple[str, click.Command]]:
    for name, command in group.commands.items():
        full = f"{prefix} {name}".strip()
        if isinstance(command, click.Group):
            yield from _commands(command, full)
        else:
            yield full, command


def test_every_command_and_option_is_in_the_cli_doc():
    doc = CLI_DOC.read_text()
    missing = [
        f"{full} {option}" if option else full
        for full, command in _commands(cli)
        for option in [None] + [o for p in command.params for o in p.opts if o.startswith("--")]
        if (option or f"`{full}`") not in doc
    ]
    assert not missing, f"Not in docs/cli.md: {missing}"
