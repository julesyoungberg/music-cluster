#!/usr/bin/env python3
"""Freeze the API server into the sidecar the desktop app ships.

The desktop build has to work for someone who does not have Python, so the
whole server (FastAPI, librosa, scikit-learn, torch and all) is frozen into a
folder holding an executable and its libraries, and copied where Tauri bundles
it as a resource:

    src-tauri/sidecar/music-cluster-api[.exe]
    src-tauri/sidecar/_internal/...

A folder rather than one self-extracting file: that unpacks hundreds of MB to
a temp folder on every launch, and leaves it behind whenever the app kills it.

    python scripts/build_sidecar.py
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent
SIDECAR = ROOT / "src-tauri" / "sidecar"
NAME = "music-cluster-api"

# librosa, numba and scikit-learn all resolve parts of themselves at runtime,
# so PyInstaller's static analysis alone leaves the binary broken.
COLLECT_ALL = ["librosa", "soundfile", "soxr", "numba", "llvmlite", "lazy_loader", "sklearn"]
HIDDEN_IMPORTS = [
    "music_cluster.api",
    "music_cluster.server",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.lifespan.on",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "sklearn.utils._vector_sentinel",
    "sklearn.neighbors._partition_nodes",
    "scipy._lib.messagestream",
    "scipy.special._cdflib",
    # transformers imports its model classes lazily, out of PyInstaller's sight.
    "transformers.models.audio_spectrogram_transformer.configuration_audio_spectrogram_transformer",
    "transformers.models.audio_spectrogram_transformer.feature_extraction_audio_spectrogram_transformer",
    "transformers.models.audio_spectrogram_transformer.modeling_audio_spectrogram_transformer",
]
# Nothing in the server draws anything; excluding these saves a lot of weight.
EXCLUDES = [
    "tkinter",
    "matplotlib",
    "IPython",
    "pytest",
    "PyQt5",
    "PySide6",
    "notebook",
    # Needs Intel TBB, which is not installed, so numba never uses it; and the
    # AppImage tool refuses to bundle a library whose dependency is missing.
    "numba.np.ufunc.tbbpool",
]


def build(clean: bool) -> Path:
    try:
        import PyInstaller.__main__ as pyinstaller
    except ImportError as exc:
        raise SystemExit(
            "PyInstaller is not installed. Run: pip install -r requirements-build.txt"
        ) from exc

    work = ROOT / "build" / "sidecar"
    dist = ROOT / "dist" / "sidecar"
    if clean:
        shutil.rmtree(work, ignore_errors=True)
        shutil.rmtree(dist, ignore_errors=True)

    arguments = [
        str(ROOT / "music_cluster" / "server.py"),
        "--name",
        NAME,
        "--onedir",
        "--console",
        "--noconfirm",
        "--distpath",
        str(dist),
        "--workpath",
        str(work),
        "--specpath",
        str(work),
        "--paths",
        str(ROOT),
    ]
    for package in COLLECT_ALL:
        arguments += ["--collect-all", package]
    for module in HIDDEN_IMPORTS:
        arguments += ["--hidden-import", module]
    for module in EXCLUDES:
        arguments += ["--exclude-module", module]

    print("Freezing the API server (this takes a few minutes)...")
    pyinstaller.run(arguments)

    produced = dist / NAME
    executable = f"{NAME}.exe" if os.name == "nt" else NAME
    if not (produced / executable).is_file():
        raise SystemExit(f"PyInstaller did not produce {produced / executable}")

    # Tauri copies resources file by file, following symlinks, so copy them
    # as plain files here and the smoke test runs the same layout that ships.
    links = sum(1 for f in produced.rglob("*") if f.is_symlink())
    print(f"Resolving {links} symlinks into plain copies")
    shutil.rmtree(SIDECAR, ignore_errors=True)
    shutil.copytree(produced, SIDECAR, symlinks=False)
    return SIDECAR / executable


def write_test_tone(path: Path, seconds: int = 12, rate: int = 22050) -> None:
    """A short chord with a pulse, enough for analysis and one AST window."""
    import wave

    import numpy as np

    t = np.arange(seconds * rate) / rate
    chord = sum(np.sin(2 * np.pi * f * t) for f in (220.0, 277.2, 329.6)) / 3
    pulse = (np.sin(2 * np.pi * 2.0 * t) > 0.95).astype(float) * np.sin(2 * np.pi * 60.0 * t)
    samples = np.int16(np.clip(0.5 * chord + 0.4 * pulse, -1, 1) * 32767)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(samples.tobytes())


def smoke_test(binary: Path) -> None:
    """Start the frozen server and make sure it answers before shipping it.

    It runs against a throwaway library set to sort on AST, and analyses one
    generated file, so a frozen torch or transformers that cannot load the
    model fails the build instead of a user's first sort. That downloads the
    model weights (about 350 MB).
    """
    import json
    import socket
    import tempfile
    import time
    import urllib.error
    import urllib.request

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    # A proxy in the environment must not be consulted for loopback.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    base = f"http://127.0.0.1:{port}"

    def call(path: str, body: Optional[dict] = None) -> dict:
        request = urllib.request.Request(
            base + path,
            data=json.dumps(body).encode() if body is not None else None,
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=10) as response:
            return json.load(response)

    workspace = Path(tempfile.mkdtemp(prefix="sidecar-smoke-"))
    (workspace / "config.yaml").write_text("sorting:\n  feature_set: ast\n")
    tone = workspace / "tone.wav"
    write_test_tone(tone)
    env = {
        **os.environ,
        "MUSIC_CLUSTER_DB": str(workspace / "library.db"),
        "MUSIC_CLUSTER_CONFIG": str(workspace / "config.yaml"),
        "MUSIC_CLUSTER_CACHE": str(workspace / "cache"),
    }

    # To a file, not a pipe: the model download's progress bars would fill an
    # unread pipe and stall the server.
    log = workspace / "server.log"
    started = time.time()
    with log.open("w") as output:
        process = subprocess.Popen(
            [str(binary), "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
            stdout=output,
            stderr=subprocess.STDOUT,
            env=env,
        )

    def fail(message: str) -> SystemExit:
        process.kill()
        process.wait()
        return SystemExit(f"{message}\n{log.read_text(errors='replace')}")

    try:
        deadline = time.time() + 180
        last_error: Optional[BaseException] = None
        while True:
            if process.poll() is not None:
                raise fail("The frozen server exited early:")
            if time.time() > deadline:
                raise fail(f"The frozen server never answered /api/info: {last_error}")
            try:
                payload = call("/api/info")
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                time.sleep(1)
        print(
            f"Smoke test: /api/info reported {payload['track_count']} tracks "
            f"after {time.time() - started:.0f} s"
        )

        deadline = time.time() + 900
        try:
            task_id = call("/api/analyze", {"paths": [str(tone)]})["task_id"]
            while True:
                if process.poll() is not None:
                    raise fail("The frozen server died while analysing with AST:")
                task = call(f"/api/tasks/{task_id}")
                if task["status"] != "running":
                    break
                if time.time() > deadline:
                    raise fail("Analysing with AST did not finish in 15 minutes:")
                time.sleep(2)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            # Usually the server crashed in native torch code: its log says why.
            raise fail(f"Lost the frozen server while analysing with AST: {exc}") from exc
        if task["status"] != "completed" or task["result"]["embedded"] != 1:
            raise fail(f"Analysing with AST failed: {json.dumps(task)}")
        print("Smoke test passed: analysed and embedded a track with AST")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        shutil.rmtree(workspace, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-clean", action="store_true", help="Reuse previous build artefacts")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Do not run the sidecar")
    parser.add_argument(
        "--smoke-test",
        type=Path,
        metavar="EXECUTABLE",
        help="Only smoke-test this sidecar executable, such as the copy inside a built bundle",
    )
    args = parser.parse_args()

    if args.smoke_test:
        if not args.smoke_test.is_file():
            raise SystemExit(f"No sidecar executable at {args.smoke_test}")
        smoke_test(args.smoke_test)
        return 0

    binary = build(clean=not args.no_clean)

    size_mb = sum(f.stat().st_size for f in SIDECAR.rglob("*") if f.is_file()) / (1024 * 1024)
    print(f"Built {binary} ({size_mb:.0f} MB in {SIDECAR})")

    if not args.skip_smoke_test:
        smoke_test(binary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
