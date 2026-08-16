#!/usr/bin/env python3
"""Seed a throwaway library for the browser end-to-end tests.

Builds the same synthetic music the Python end-to-end suite uses, imports the
genre folders as groups, and fits the sorter — so the UI has real content to
render the moment the API comes up.

    python scripts/seed_e2e_library.py .e2e-workspace

Writes `<workspace>/library.db`, `<workspace>/config.yaml` and
`<workspace>/music/`, and is idempotent: an already-seeded workspace is left
alone unless --force is given.
"""

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tests.e2e.fixtures import build_library, import_and_fit  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workspace", type=Path, help="Directory to seed")
    parser.add_argument("--force", action="store_true", help="Re-seed an existing workspace")
    args = parser.parse_args()

    workspace: Path = args.workspace.resolve()
    database = workspace / "library.db"

    if database.exists() and not args.force:
        print(f"Already seeded: {database}")
        return 0

    if workspace.exists() and args.force:
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    print(f"Synthesising a music library in {workspace / 'music'}...")
    music = build_library(workspace / "music")

    print("Importing the genre folders and fitting the sorter...")
    metrics = import_and_fit(str(database), str(workspace / "config.yaml"), music)

    accuracy = (metrics or {}).get("accuracy")
    print(f"Seeded {database}" + (f" (self-check {accuracy * 100:.0f}%)" if accuracy else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
