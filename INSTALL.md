# Installation

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

To build a distributable app:

```bash
cd ui && npm run tauri:build
```

## Optional extras

```bash
pip install anthropic        # LLM-suggested names for discovered groups
pip install -r requirements-dev.txt   # tests
```

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
