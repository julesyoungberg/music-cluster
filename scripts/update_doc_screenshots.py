#!/usr/bin/env python3
"""Copy the screenshots the docs use from a UI snapshot run into docs/images/.

    cd ui && npm run ui:snapshot && cd ..
    python scripts/update_doc_screenshots.py

The docs reference images by these names, so retaking every screenshot after a
UI change is the two commands above. To use a new screen in the docs, add a
`shot` for it in ui/snapshots/screens.spec.ts and its name to SCREENS.
"""

import argparse
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent

# Snapshot screen names used by docs/tutorial.md (and the README).
SCREENS = [
    "overview",
    "groups-new-collection-dialog",
    "groups-import-dialog",
    "groups",
    "group-detail",
    "groups-overlap",
    "sort-new-dialog",
    "sort-review",
    "sort-group-picker",
    "sort-all-decided",
    "sort-commit-preview-playlists",
    "sort-commit-preview-move",
    "sort-committed",
    "sort-undone",
    "discover-new-dialog",
    "discover-run",
    "discover-keep-dialog",
    "library-tag-filter",
    "library-similar",
    "library-analyze-dialog",
    "settings",
    "dark-group-detail",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--source", type=Path, default=ROOT / "ui" / "ui-snapshots")
    parser.add_argument("--dest", type=Path, default=ROOT / "docs" / "images")
    args = parser.parse_args()

    missing = [name for name in SCREENS if not (args.source / f"{name}.png").exists()]
    if missing:
        print(f"Missing from {args.source}: {', '.join(missing)}", file=sys.stderr)
        print("Run `npm run ui:snapshot` in ui/ first.", file=sys.stderr)
        return 1

    args.dest.mkdir(parents=True, exist_ok=True)
    for name in SCREENS:
        shutil.copyfile(args.source / f"{name}.png", args.dest / f"{name}.png")
    print(f"Copied {len(SCREENS)} screenshots to {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
