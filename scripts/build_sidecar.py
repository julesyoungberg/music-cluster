#!/usr/bin/env python3
"""Freeze the API server into the sidecar binary the desktop app ships.

The desktop build has to work for someone who does not have Python, so the
whole server — FastAPI, librosa, scikit-learn and all — is packed into one
executable and dropped where Tauri expects an `externalBin`:

    src-tauri/binaries/music-cluster-api-<rust-target-triple>[.exe]

Tauri appends the target triple itself when it resolves a sidecar, which is
what makes one repository produce a working bundle on three platforms.

    python scripts/build_sidecar.py
    python scripts/build_sidecar.py --target-triple x86_64-apple-darwin
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent.parent
BINARIES = ROOT / "src-tauri" / "binaries"
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
]
# Nothing in the server draws anything; excluding these saves a lot of weight.
EXCLUDES = ["tkinter", "matplotlib", "IPython", "pytest", "PyQt5", "PySide6", "notebook"]


def rust_target_triple() -> str:
    """The triple Tauri will look for, taken from the local Rust toolchain."""
    try:
        output = subprocess.run(["rustc", "-vV"], check=True, capture_output=True, text=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            "Could not ask rustc for the target triple. Install Rust, or pass "
            "--target-triple explicitly."
        ) from exc

    for line in output.splitlines():
        if line.startswith("host:"):
            return line.split(":", 1)[1].strip()
    raise SystemExit("rustc -vV did not report a host triple")


def build(triple: str, clean: bool) -> Path:
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
        "--onefile",
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

    produced = dist / (f"{NAME}.exe" if os.name == "nt" else NAME)
    if not produced.is_file():
        raise SystemExit(f"PyInstaller did not produce {produced}")

    BINARIES.mkdir(parents=True, exist_ok=True)
    suffix = ".exe" if os.name == "nt" else ""
    target = BINARIES / f"{NAME}-{triple}{suffix}"
    shutil.copy2(produced, target)
    target.chmod(0o755)
    return target


def smoke_test(binary: Path) -> None:
    """Start the frozen server and make sure it answers before shipping it."""
    import json
    import socket
    import time
    import urllib.error
    import urllib.request

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    # A proxy in the environment must not be consulted for loopback.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    process = subprocess.Popen(
        [str(binary), "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.time() + 180
        last_error: Optional[BaseException] = None
        while time.time() < deadline:
            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                raise SystemExit(f"The frozen server exited early:\n{output}")
            try:
                with opener.open(f"http://127.0.0.1:{port}/api/info", timeout=2) as response:
                    payload = json.load(response)
                print(f"Smoke test passed: /api/info reported {payload['track_count']} tracks")
                return
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = exc
                time.sleep(1)
        raise SystemExit(f"The frozen server never answered /api/info: {last_error}")
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-triple", default=None, help="Override the Rust target triple")
    parser.add_argument("--no-clean", action="store_true", help="Reuse previous build artefacts")
    parser.add_argument("--skip-smoke-test", action="store_true", help="Do not run the binary")
    args = parser.parse_args()

    triple = args.target_triple or rust_target_triple()
    binary = build(triple, clean=not args.no_clean)

    size_mb = binary.stat().st_size / (1024 * 1024)
    print(f"Built {binary} ({size_mb:.0f} MB)")

    if not args.skip_smoke_test:
        smoke_test(binary)

    return 0


if __name__ == "__main__":
    sys.exit(main())
