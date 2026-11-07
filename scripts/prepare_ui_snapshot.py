#!/usr/bin/env python3
"""Give the UI snapshot run a fresh copy of the seeded library.

The snapshot run clicks through flows that change the library (sort sessions,
a playlist commit and its undo, a discovery run). So that every run starts
from the same state, the library is seeded once into ``<workspace>/seed`` and
each run works on a copy of its database in ``<workspace>/run``:

    python scripts/prepare_ui_snapshot.py .ui-snapshot-workspace

Playlists are written to ``<workspace>/run/playlists``, so nothing the run
does lands outside the workspace. The synthesised audio is shared and only
ever read.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from music_cluster.config import Config  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="Directory to prepare")
    parser.add_argument("--reseed", action="store_true", help="Synthesise the library again")
    args = parser.parse_args()

    workspace: Path = args.workspace.resolve()
    seed = workspace / "seed"
    run = workspace / "run"

    command = [sys.executable, str(ROOT / "scripts" / "seed_e2e_library.py"), str(seed)]
    if args.reseed:
        command.append("--force")
    subprocess.run(command, check=True)

    if run.exists():
        # Only clear a folder this script made: it holds the copied database.
        if any(run.iterdir()) and not (run / "library.db").exists():
            print(f"Refusing to clear {run}: it was not made by this script.", file=sys.stderr)
            return 1
        shutil.rmtree(run)
    run.mkdir(parents=True)
    shutil.copy2(seed / "library.db", run / "library.db")

    config = Config(str(seed / "config.yaml"))
    config.config_path = str(run / "config.yaml")
    config.set(["database", "path"], str(run / "library.db"))
    config.set(["organize", "playlist_dir"], str(run / "playlists"))
    config.save()

    print(f"Fresh snapshot library in {run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
