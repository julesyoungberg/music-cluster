# music-cluster

Sorts new music (and samples) into groups the user already keeps. The user
defines the groups; the tool scores, the user confirms, then it files. It never
invents a category or touches a file the user has not confirmed. README.md
explains the product; INSTALL.md covers setup, CI and troubleshooting.

## Layout

- `music_cluster/` — Python package (CLI + FastAPI server, SQLite storage)
  - `cli.py` — the `music-cluster` command (entry point `music_cluster.cli:main`)
  - `api.py` — REST API behind the UI; `server.py` is the frozen-sidecar entry point
  - `database.py` — all SQLite persistence (`~/.music-cluster/library.db`)
  - `library.py` → `extractor.py` / `features.py` — ingestion and feature extraction
  - `profiles.py` — `music` vs `sample` profiles; a collection's profile is fixed at creation
  - `samples.py` — one-shot classification (kick, snare, hat, bass, chord…)
  - `rhythm.py`, `tagging.py` — beat measurement and auto-tags (genre, mood, energy, key)
  - `groups.py` — collections and reference groups
  - `embedding.py` — LDA+PCA space fitted to the user's groups
  - `sorter.py` — scores a track against groups; `sorting.py` — sort sessions/review queue
  - `organizer.py` — applies decisions to disk (playlist/copy/move/symlink, dry run, undo)
  - `discovery.py`, `clustering.py`, `naming.py`, `llm.py` — proposing candidate groups from an unsorted pile
- `ui/` — SvelteKit + Tailwind front end (vitest unit tests, Playwright e2e in `ui/e2e/`)
- `src-tauri/` — Tauri 1 desktop shell; `scripts/build_sidecar.py` freezes the API into it
- `tests/unit` (synthetic vectors, fast), `tests/integration` and `tests/e2e` (real/synthesised audio, slow)

## Commands

```bash
# Setup (CI runs Python 3.10–3.12; librosa/numba may lag newer Pythons)
uv venv --python 3.12 .venv && source .venv/bin/activate
pip install -r requirements-dev.txt && pip install -e .
cd ui && npm ci && cd ..

# Run
python start-dev.py              # API on :8000 + UI on http://localhost:1420
python start-dev.py --tauri      # desktop shell

# Python checks
ruff format . && ruff check --fix . && mypy
pytest tests/unit                # fast
pytest tests/e2e tests/integration

# UI checks (from ui/)
npm run format:check && npm run lint && npm run check
npm test
npx playwright test              # seeds its own library via scripts/seed_e2e_library.py
```

FFmpeg must be installed for any audio decoding. CI (`.github/workflows/`) runs
exactly the commands above; keep them passing before opening a PR.

## Conventions

- Ruff with line length 100. Type hints use `typing` (`List`, `Optional`, …)
  consistently — do not rewrite to PEP 585/604 syntax (those rules are ignored on
  purpose in `pyproject.toml`).
- mypy is progressive: annotate new code, but missing annotations are not errors.
- Unit tests use synthetic feature vectors (`tests/conftest.py`), not audio.
  Mark tests that decode or synthesise audio `slow` or `e2e`.
- Never move, copy or delete user files without an explicit confirmation step;
  every filing mode has a dry run and an undo, and new modes must too.
- Tags and sample categories are readable rules over measured quantities, not
  trained models — keep them explainable.
- `package-lock.json` (root and `ui/`) and `src-tauri/Cargo.lock` are committed;
  install with `npm ci`.
