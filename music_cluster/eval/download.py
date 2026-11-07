"""Fetch the public datasets the harness scores against.

Everything lands under the eval data directory (``~/.music-cluster-eval/data``
by default, or ``$MUSIC_CLUSTER_EVAL_DATA``), never inside the repository and
never anywhere near the user's library. Downloads resume: a file already
present with the right size or checksum is skipped.
"""

import hashlib
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path
from typing import Callable, Dict, Optional

from .datasets import data_dir


FMA_BASE = "https://os.unil.cloud.switch.ch/fma/"
FMA_FILES = {
    "metadata": "fma_metadata.zip",  # 342 MB
    "small": "fma_small.zip",  # 7.2 GB, 8,000 clips
    "medium": "fma_medium.zip",  # 22 GB, 25,000 clips
}

GIANTSTEPS_REPOS = {
    "key": "https://github.com/GiantSteps/giantsteps-key-dataset.git",
    "tempo": "https://github.com/GiantSteps/giantsteps-tempo-dataset.git",
}
# Where both repositories' own audio_dl.sh scripts fetch the previews from.
GIANTSTEPS_AUDIO = "http://www.cp.jku.at/datasets/giantsteps/backup/"

Echo = Callable[[str], None]


def _fetch(url: str, target: Path, echo: Echo) -> None:
    if target.exists() and target.stat().st_size > 0:
        echo(f"  have {target.name}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    echo(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    partial.rename(target)


def _unzip(archive: Path, into: Path, marker: str, echo: Echo) -> None:
    if (into / marker).exists():
        return
    echo(f"  unpacking {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(into)


def download_fma(subset: str = "small", root: Optional[str] = None, echo: Echo = print) -> Path:
    """Metadata plus one audio subset. ``small`` is enough for every phase-1 suite."""
    if subset not in ("small", "medium"):
        raise ValueError("FMA subset must be 'small' or 'medium'")
    base = data_dir(root) / "fma"
    for key, marker in (("metadata", "fma_metadata"), (subset, f"fma_{subset}")):
        archive = base / FMA_FILES[key]
        _fetch(FMA_BASE + FMA_FILES[key], archive, echo)
        _unzip(archive, base, marker, echo)
    return base


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_giantsteps(kind: str, root: Optional[str] = None, echo: Echo = print) -> Path:
    """Clone the annotation repository, then fetch each preview it lists.

    Mirrors the repository's own ``audio_dl.sh`` (which needs wget and
    md5sum) in Python so it runs the same on macOS, Linux and Windows.
    """
    if kind not in GIANTSTEPS_REPOS:
        raise ValueError("GiantSteps dataset must be 'key' or 'tempo'")
    repo = data_dir(root) / f"giantsteps-{kind}-dataset"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        echo(f"  cloning {GIANTSTEPS_REPOS[kind]}")
        subprocess.run(
            ["git", "clone", "--depth", "1", GIANTSTEPS_REPOS[kind], str(repo)], check=True
        )

    expected: Dict[str, Optional[str]] = {}
    md5_dir = repo / "md5"
    if md5_dir.is_dir():
        for entry in sorted(md5_dir.glob("*.md5")):
            expected[entry.name[: -len(".md5")]] = entry.read_text().split()[0].strip()
    else:
        suffix = ".key" if kind == "key" else ".bpm"
        for entry in sorted((repo / "annotations" / kind).glob(f"*{suffix}")):
            expected[entry.name[: -len(suffix)]] = None

    audio = repo / "audio"
    audio.mkdir(exist_ok=True)
    failed = 0
    for i, (stem, checksum) in enumerate(expected.items(), 1):
        target = audio / f"{stem}.mp3"
        if target.exists() and (checksum is None or _md5(target) == checksum):
            continue
        if i % 50 == 0:
            echo(f"  {i}/{len(expected)}")
        try:
            _fetch(f"{GIANTSTEPS_AUDIO}{stem}.mp3", target, lambda _m: None)
        except Exception as exc:
            failed += 1
            echo(f"  could not fetch {stem}: {exc}")
            target.unlink(missing_ok=True)
    echo(f"  {len(expected) - failed} of {len(expected)} previews present")
    return repo
