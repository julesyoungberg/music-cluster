#!/usr/bin/env python3
"""Build the landing page in site/ into a folder any static host can serve.

    python scripts/build_site.py            # writes site/dist/
    python -m http.server -d site/dist      # preview at http://localhost:8000

site/index.html is a template: `${key}` is replaced by that key's value from
site/site.json, so the product name, links and form endpoints are each set in
one place. Screenshots the page uses (`images/<name>.png`) are copied from
docs/images/, the same files the README and tutorial show.

Values in site.json that start with "TODO" still need a real value before the
site goes live; the build lists them.
"""

import argparse
import html
import json
import re
import shutil
import sys
from pathlib import Path
from string import Template
from typing import Dict, List


ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
IMAGES = ROOT / "docs" / "images"

# Files copied to the output as they are.
STATIC_FILES = ["styles.css"]

# Written into every output folder, so a rebuild only ever deletes a folder
# this script made, never one passed to --out by mistake.
MARKER = ".built-by-build_site"

IMAGE_REF = re.compile(r"""\bsrc="images/([\w.-]+)\"""")


def load_values(path: Path) -> Dict[str, str]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict) or not all(isinstance(v, str) for v in values.values()):
        raise ValueError(f"{path} must be a JSON object of strings")
    return values


def placeholders(values: Dict[str, str]) -> List[str]:
    """Keys whose value has not been filled in yet."""
    return sorted(key for key, value in values.items() if value.startswith("TODO"))


def render(template: str, values: Dict[str, str]) -> str:
    escaped = {key: html.escape(value) for key, value in values.items()}
    try:
        return Template(template).substitute(escaped)
    except KeyError as err:
        raise ValueError(
            f"site/index.html uses ${{{err.args[0]}}}, which site.json does not set"
        ) from None


def build(out: Path, site: Path = SITE, images: Path = IMAGES) -> List[str]:
    """Write the site to `out` and return the keys still holding placeholders."""
    values = load_values(site / "site.json")
    page = render((site / "index.html").read_text(encoding="utf-8"), values)

    if out.exists():
        if any(out.iterdir()) and not (out / MARKER).is_file():
            raise ValueError(
                f"{out} is not empty and was not made by this script; not replacing it"
            )
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / MARKER).write_text("", encoding="utf-8")
    (out / "index.html").write_text(page, encoding="utf-8")
    for name in STATIC_FILES:
        shutil.copy2(site / name, out / name)
    for name in sorted(set(IMAGE_REF.findall(page))):
        source = images / name
        if not source.is_file():
            raise FileNotFoundError(f"The page shows images/{name}, but {source} does not exist")
        shutil.copy2(source, out / "images" / name)
    return placeholders(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=SITE / "dist", help="output folder")
    args = parser.parse_args()

    try:
        todo = build(args.out)
    except (ValueError, FileNotFoundError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    print(f"Built {args.out}")
    if todo:
        print("Still placeholders in site/site.json: " + ", ".join(todo))
    return 0


if __name__ == "__main__":
    sys.exit(main())
