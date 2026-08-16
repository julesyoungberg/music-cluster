"""Entry point for the bundled API server.

The desktop app ships the API as a sidecar binary rather than asking the user
to install Python, so it needs a real ``main()`` to build against — importing
``uvicorn`` by module string works from a checkout but not from a frozen
executable.

    python -m music_cluster.server --port 8000
"""

import argparse
import sys
from typing import List, Optional


DEFAULT_PORT = 8000
DEFAULT_HOST = "127.0.0.1"


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the music-cluster API server")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Interface to bind")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Reload on source changes")
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
    )
    args = parser.parse_args(argv)

    import uvicorn

    if args.reload:
        # Reloading needs an import string so the worker can re-import the app.
        uvicorn.run(
            "music_cluster.api:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
        )
        return 0

    from music_cluster.api import app

    # Announce the address on stdout before serving: the desktop shell reads
    # this line to learn which port it ended up on.
    print(f"music-cluster api listening on http://{args.host}:{args.port}", flush=True)

    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
