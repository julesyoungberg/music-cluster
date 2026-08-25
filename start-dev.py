#!/usr/bin/env python3
"""Run the API and the desktop UI together for development.

python start-dev.py            # API + web UI at http://localhost:1420
python start-dev.py --tauri    # API + the Tauri desktop shell
"""

import argparse
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
UI = ROOT / "ui"

BLUE, GREEN, RED, RESET = "\033[0;34m", "\033[0;32m", "\033[0;31m", "\033[0m"


def fail(message: str) -> "None":
    print(f"{RED}{message}{RESET}", file=sys.stderr)
    sys.exit(1)


def python_executable() -> str:
    """Prefer a project virtualenv, falling back to the current interpreter."""
    for candidate in (ROOT / ".venv" / "bin" / "python", ROOT / "venv" / "bin" / "python"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tauri", action="store_true", help="Run the desktop shell")
    parser.add_argument("--port", type=int, default=8000, help="API port")
    args = parser.parse_args()

    if not shutil.which("npm"):
        fail("npm is required to run the UI. Install Node.js, or run the API alone with uvicorn.")

    if not (UI / "node_modules").exists():
        print(f"{BLUE}Installing UI dependencies...{RESET}")
        subprocess.run(["npm", "install"], cwd=UI, check=True)

    # The Tauri CLI lives at the repository root, because it can only find
    # src-tauri by looking *down* from where it is run.
    if args.tauri and not (ROOT / "node_modules").exists():
        print(f"{BLUE}Installing the desktop shell's dependencies...{RESET}")
        subprocess.run(["npm", "install"], cwd=ROOT, check=True)

    python = python_executable()
    try:
        subprocess.run(
            [python, "-c", "import fastapi, uvicorn"],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        fail(f"{python} cannot import fastapi/uvicorn. Run: pip install -r requirements.txt")

    processes = []
    try:
        print(f"{BLUE}Starting API on port {args.port}...{RESET}")
        processes.append(
            subprocess.Popen(
                [
                    python,
                    "-m",
                    "uvicorn",
                    "music_cluster.api:app",
                    "--reload",
                    "--port",
                    str(args.port),
                ],
                cwd=ROOT,
            )
        )
        time.sleep(2)

        print(f"{BLUE}Starting UI...{RESET}")
        # `tauri dev` starts the Vite server itself (beforeDevCommand), so only
        # the plain web mode launches it here.
        command = ["npm", "run", "tauri:dev"] if args.tauri else ["npm", "run", "dev"]
        processes.append(subprocess.Popen(command, cwd=ROOT if args.tauri else UI))

        print(f"\n{GREEN}Running. Press Ctrl+C to stop.{RESET}")
        if not args.tauri:
            print("  UI:  http://localhost:1420")
        print(f"  API: http://localhost:{args.port}/docs\n")

        while all(process.poll() is None for process in processes):
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{BLUE}Shutting down...{RESET}")
        for process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
