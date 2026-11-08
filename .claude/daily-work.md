# Daily project work

budget_minutes: 60

Log of the scheduled "Daily Project Work: Music Cluster" runs. Newest last.

## Log

### 2026-11-08 (first run)

- Minutes: ~50 of 60. Subagents: 0.
- State: main green (lint, unit, e2e, desktop build). Unit tests, ruff and mypy pass locally. No open issues. Open PRs #24 (landing page credit) and #25 (cratewise rename) belong to other threads.
- Shipped: this log (draft PR). No code change: the top remaining items (novelty check at chance, techno read as breakbeat, minor-loop keys) all change measuring or scoring code, and none could be checked from the cloud because dataset hosts are blocked here.
- Started: first dataset evals in GitHub Actions (`eval.yml`, workflow_dispatch), to find out whether Actions can reach the dataset hosts. If they can, future runs can check sorter, key and tempo changes without Jules's Mac.
  - sorting-fma on main: https://github.com/julesyoungberg/music-cluster/actions/runs/42524306085
  - key-giantsteps + tempo-giantsteps on this branch (same code as main): https://github.com/julesyoungberg/music-cluster/actions/runs/42524493788
  - Both were past install and downloading data when this run ended.
- Next: read both runs. If they passed, compare with eval/baselines/datasets.json (tolerance 0.03; Linux vs Mac and no Essentia may shift numbers) and note whether the dataset cache saved (FMA small is 7.2 GB zipped plus the extracted copy; the Actions cache limit is 10 GB). If downloads failed, record why and fall back to Remote Control on the Mac. Then take one of: novelty check, techno four-on-the-floor in rhythm.py.
- Notes for later runs:
  - Don't dispatch a second eval on the same ref while one runs: eval.yml cancels the older run (concurrency group per ref).
  - keys-synthetic `key_correct_minor_loop` is 0 by construction: the i-VI-III-VII loop has the same pitch classes as its relative major with C sounding most, so no mean-chroma method can pass it. Judge key changes on GiantSteps instead.
- Questions for Jules: none new. Still open from earlier threads: product name, license, commit dialog default mode, pretrained embedding (CLAP/OpenL3) for the sorter.
