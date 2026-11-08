"""The landing page build: every value filled in, every local file it links present."""

import importlib.util
import json
import re
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("build_site", ROOT / "scripts" / "build_site.py")
assert spec and spec.loader
build_site = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build_site)


def test_real_site_builds_with_every_local_reference_present(tmp_path):
    out = tmp_path / "dist"
    build_site.build(out)
    page = (out / "index.html").read_text(encoding="utf-8")

    assert "${" not in page
    local = [
        ref
        for ref in re.findall(r'(?:href|src)="([^"]+)"', page)
        if not ref.startswith(("http", "#", "mailto:", "TODO"))
    ]
    assert local, "expected the page to reference its stylesheet and screenshots"
    for ref in local:
        assert (out / ref).is_file(), f"index.html links {ref}, which the build did not write"


def _site(tmp_path, template, values):
    site = tmp_path / "site"
    site.mkdir()
    (site / "index.html").write_text(template, encoding="utf-8")
    (site / "site.json").write_text(json.dumps(values), encoding="utf-8")
    (site / "styles.css").write_text("", encoding="utf-8")
    return site


def test_values_are_escaped_and_placeholders_reported(tmp_path):
    site = _site(
        tmp_path,
        '<h1>${name}</h1><a href="${donate_url}">x</a>',
        {"name": "Crates & <Co>", "donate_url": "TODO: link"},
    )
    todo = build_site.build(tmp_path / "out", site=site, images=tmp_path)

    page = (tmp_path / "out" / "index.html").read_text(encoding="utf-8")
    assert "<h1>Crates &amp; &lt;Co&gt;</h1>" in page
    assert todo == ["donate_url"]


def test_unknown_key_fails(tmp_path):
    site = _site(tmp_path, "${missing}", {"name": "x"})
    with pytest.raises(ValueError, match="missing"):
        build_site.build(tmp_path / "out", site=site, images=tmp_path)


def test_missing_screenshot_fails(tmp_path):
    site = _site(tmp_path, '<img src="images/nope.png">', {})
    images = tmp_path / "images"
    images.mkdir()
    with pytest.raises(FileNotFoundError, match=r"nope\.png"):
        build_site.build(tmp_path / "out", site=site, images=images)
