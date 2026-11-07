"""Fetch the public datasets the harness scores against.

Everything lands under the eval data directory (``~/.music-cluster-eval/data``
by default, or ``$MUSIC_CLUSTER_EVAL_DATA``), never inside the repository and
never anywhere near the user's library. Re-running a download skips what is
already complete: an archive whose size matches what the server reports, a
preview whose checksum matches the published one, an unpacked archive whose
completion marker exists. An interrupted transfer starts that file again.
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


def _remote_size(url: str) -> Optional[int]:
    try:
        request = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            length = response.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def _fetch(url: str, target: Path, echo: Echo, check_size: bool = True) -> None:
    """Download ``url`` to ``target`` unless a complete copy is already there.

    "Complete" means the size the server reports, so a transfer cut short
    without an error is downloaded again rather than kept.
    """
    expected = _remote_size(url) if check_size else None
    if target.exists() and target.stat().st_size > 0:
        if expected is None or target.stat().st_size == expected:
            echo(f"  have {target.name}")
            return
        echo(f"  {target.name} is incomplete; downloading again")
        target.unlink()
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    echo(f"  fetching {url}")
    with urllib.request.urlopen(url, timeout=60) as response, open(partial, "wb") as out:
        shutil.copyfileobj(response, out, length=1 << 20)
    if expected is not None and partial.stat().st_size != expected:
        partial.unlink()
        raise OSError(
            f"{url} ended early ({partial.stat().st_size if partial.exists() else 0} of {expected} bytes)"
        )
    partial.rename(target)


def _unzip(archive: Path, into: Path, marker: str, echo: Echo) -> None:
    # The marker is written only after extraction finishes, so an interrupted
    # unpack is redone rather than mistaken for a complete one.
    done = into / f".{marker}.unpacked"
    if done.exists():
        return
    echo(f"  unpacking {archive.name}")
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(into)
    done.write_text("ok\n")


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
            words = entry.read_text().split()
            expected[entry.name[: -len(".md5")]] = words[0].strip() if words else None
    else:
        suffix = ".key" if kind == "key" else ".bpm"
        for entry in sorted((repo / "annotations" / kind).glob(f"*{suffix}")):
            expected[entry.name[: -len(suffix)]] = None

    audio = repo / "audio"
    audio.mkdir(exist_ok=True)
    failed = 0
    for i, (stem, checksum) in enumerate(expected.items(), 1):
        target = audio / f"{stem}.mp3"
        if target.exists():
            if checksum is None or _md5(target) == checksum:
                continue
            target.unlink()  # corrupt or truncated: fetch it again
        if i % 50 == 0:
            echo(f"  {i}/{len(expected)}")
        try:
            # Plain HTTP, so the published checksum is what vouches for the file.
            _fetch(f"{GIANTSTEPS_AUDIO}{stem}.mp3", target, lambda _m: None, check_size=False)
            if checksum is not None and _md5(target) != checksum:
                raise OSError("checksum mismatch")
        except Exception as exc:
            failed += 1
            echo(f"  could not fetch {stem}: {exc}")
            target.unlink(missing_ok=True)
    echo(f"  {len(expected) - failed} of {len(expected)} previews present")
    return repo
