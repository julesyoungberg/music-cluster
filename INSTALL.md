# Installation

## Download a build

The quickest route, if you do not want to touch Python: grab the installer for
your platform from the [releases page](https://github.com/julesyoungberg/music-cluster/releases).
Each build bundles the analysis server, so nothing else needs installing except
FFmpeg.

| Platform | File |
| --- | --- |
| macOS (Apple silicon) | `.dmg` from the `macos-aarch64` build |
| macOS (Intel) | `.dmg` from the `macos-x86_64` build |
| Windows | `.msi` |
| Linux | `.AppImage` or `.deb` |

The builds are unsigned, so the first launch needs a right-click → Open on
macOS, or "More info → Run anyway" on Windows.

Everything below is for running from source or working on the project.

## Prerequisites

- Python 3.10 or newer
- FFmpeg, for decoding audio
- Node.js 18+, only if you want the desktop UI

### FFmpeg

```bash
brew install ffmpeg              # macOS
sudo apt-get install ffmpeg      # Debian / Ubuntu
winget install ffmpeg            # Windows
```

Check it with `ffmpeg -version`.

## Install

```bash
git clone https://github.com/julesyoungberg/music-cluster.git
cd music-cluster

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .

music-cluster init
```

`init` creates `~/.music-cluster/library.db` and `~/.music-cluster/config.yaml`.
Both locations can be changed in the config, or overridden per run with the
`MUSIC_CLUSTER_DB` and `MUSIC_CLUSTER_CONFIG` environment variables — useful for
keeping a separate database per DJ setup.

## Desktop UI

```bash
cd ui && npm install && cd ..
python start-dev.py
```

The UI runs at <http://localhost:1420> and the API at <http://localhost:8000>
(interactive docs at `/docs`). For the native shell, `python start-dev.py --tauri`
— that also needs the [Tauri prerequisites](https://tauri.app/start/prerequisites/)
for your platform.

### Building a distributable app

The packaged app has to work for someone who has never installed Python, so the
API is frozen into a single binary and shipped inside the bundle as a Tauri
sidecar. The desktop shell starts it on a free port and shuts it down with the
window.

```bash
pip install -r requirements-build.txt
python scripts/build_sidecar.py      # freezes the API (a few minutes, ~150 MB)
python scripts/make_icons.py         # only needed if the icons are missing

npm install --prefix ui              # the web UI
npm install                          # the Tauri CLI, which must live at the root
npm run tauri:build
```

The Tauri CLI finds `src-tauri` by searching downwards, so it has to be run
from the repository root — that is the only reason there is a `package.json`
there as well as in `ui/`.

Installers land in `src-tauri/target/release/bundle/`. This also needs the
[Tauri prerequisites](https://tauri.app/v1/guides/getting-started/prerequisites)
for your platform; on Linux that means `libwebkit2gtk-4.0-dev`, which is why CI
builds Linux on Ubuntu 22.04.

CI does all of this for you — see [Continuous integration](#continuous-integration).

## Optional extras

```bash
pip install anthropic        # LLM-suggested names for discovered groups
pip install -r requirements-dev.txt   # tests
pip install -r requirements-build.txt # packaging the desktop app
```

## Continuous integration

Four workflows run in GitHub Actions. All of them can be run locally with the
same commands.

| Workflow | What it runs |
| --- | --- |
| `lint.yml` | `ruff format --check`, `ruff check`, `mypy`; prettier, eslint and `svelte-check` for the UI; `cargo fmt` and `clippy` for the desktop shell |
| `test-unit.yml` | `pytest tests/unit` on Python 3.10–3.12 and on all three platforms, plus `npm test` (vitest) |
| `test-e2e.yml` | `pytest tests/e2e tests/integration` against real audio, plus the Playwright browser suite against a real API and a real build |
| `build.yml` | Freezes the API, builds the UI, and bundles installers for macOS (both architectures), Windows and Linux |

`build.yml` runs the full matrix on `main` and on `v*` tags, and a single Linux
build on pull requests that touch packaging. Tagging `v2.1.0` drafts a release
with every installer attached, ready to review and publish.

```bash
# The same checks, locally
ruff format --check . && ruff check . && mypy
pytest tests/unit
pytest tests/e2e tests/integration

cd ui
npm run format:check && npm run lint && npm run check
npm test
npx playwright install --with-deps chromium   # first time only
npx playwright test
```

The browser suite seeds its own throwaway library and starts both servers
itself; `scripts/seed_e2e_library.py` is what builds that fixture.

## Troubleshooting

**`command not found: music-cluster`** — activate the virtualenv, or use
`python -m music_cluster.cli` instead. Reinstall with `pip install -e .` if it
persists.

**Audio files fail to analyse** — almost always missing FFmpeg. Check
`ffmpeg -version`. A handful of failures out of thousands usually means those
specific files are corrupt; `music-cluster analyze` reports each one.

**Analysis is slow** — it is roughly a second per track and runs on every core.
Lower `feature_extraction.excerpt_seconds` (default 90) to trade accuracy for
speed. Analysis happens once per track; sorting afterwards is instant.

**Memory pressure on a very large library** — run with `--workers 2` to cap
parallelism, or analyse in chunks. Fitting holds only the reference tracks in
memory, not the whole library.

**"Need at least two groups with reference tracks"** — a collection needs two
non-empty groups before it can be fitted. `music-cluster groups list` shows
which are empty.

**The UI says it cannot reach the API** — start it with `python start-dev.py`,
or run `uvicorn music_cluster.api:app --port 8000` yourself. If you moved the
API to another port, set `VITE_API_BASE` when running the UI.

## Upgrading from 1.x

The clustering-era database is migrated automatically on first use: analysed
features are converted and kept, and the old clustering tables are dropped. Your
previous clusterings are not carried over — they were anonymous groupings, and
the new workflow is built on groups you name. Import your folders with
`music-cluster groups import-tree` and re-fit; no re-analysis is needed.
