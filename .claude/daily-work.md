# Daily project work

budget_minutes: 60

Log of the scheduled "Daily Project Work: Music Cluster" runs. Newest last.

## Log

### 2026-11-08 (first run)

- Minutes: ~50 of 60. Subagents: 0.
- State: main green (lint, unit, e2e, desktop build). Unit tests, ruff and mypy pass locally. No open issues. Open PRs #24 (landing page credit) and #25 (cratewise rename) belong to other threads.
- Shipped: this log (PR #26). No code change: the top remaining items (novelty check at chance, techno read as breakbeat, minor-loop keys) all change measuring or scoring code, and none could be checked from the cloud because dataset hosts are blocked here.
- Started: first dataset evals in GitHub Actions (`eval.yml`, workflow_dispatch), to find out whether Actions can reach the dataset hosts. If they can, future runs can check sorter, key and tempo changes without Jules's Mac.
  - sorting-fma on main: https://github.com/julesyoungberg/music-cluster/actions/runs/42524306085
  - key-giantsteps + tempo-giantsteps on this branch (same code as main): https://github.com/julesyoungberg/music-cluster/actions/runs/42524493788
  - Both were past install and downloading data when this run ended.
  - Update 22:25 UTC: the GiantSteps run passed in 67 min (37 download, 30 analysis) and matched the Mac baseline exactly: key weighted 0.589; tempo within 4% (acc1) 0.678, or within 4% allowing double or half tempo (acc2) 0.885; beat_genre_accuracy 0.568. Essentia, the comparison key and tempo detector in the baseline, is not installed on Actions, so those reference rows are missing there. Key, tempo and beat-type changes can now be checked from the cloud.
  - The data cache saved (1.8 GB) under the suite list `key-giantsteps tempo-giantsteps`. Dispatch exactly that list, in that order, to reuse it. A change to the measuring files still restores the data by prefix; it only saves a new cache entry.
- Next: read the sorting-fma run (42524306085): if it passed, compare with eval/baselines/datasets.json (tolerance 0.03; Linux vs Mac may shift numbers) and note whether its cache saved (FMA small is 7.2 GB zipped plus the extracted copy; the Actions cache limit is 10 GB). If its download failed, record why and fall back to Remote Control on the Mac. Then take one of: novelty check, techno four-on-the-floor in rhythm.py.
- Notes for later runs:
  - Don't dispatch a second eval on the same ref while one runs: eval.yml cancels the older run (concurrency group per ref).
  - keys-synthetic `key_correct_minor_loop` is 0: the i-VI-III-VII loop uses the same notes as its relative major, and the relative major's tonic occurs most often, so the averaged-chroma key match in `tagging.py` picks the relative major. Judge key changes on GiantSteps instead.
- Questions for Jules: none new. Still open from earlier threads: product name, license, commit dialog default mode, pretrained embedding (CLAP/OpenL3) for the sorter.

### 2026-11-09

- Minutes: ~55 of 60. Subagents: 1 (log review).
- State: main green. `pytest tests/unit` passes; the synthetic eval on main matches `eval/baselines/synthetic.json` exactly (beat_accuracy 0.882, key_weighted 0.825). No open issues. Open PR #31 (AST embedding eval) belongs to another thread.
- sorting-fma in Actions (follow-up from Nov 8): the PR #31 thread has since run it in Actions (run 42617669952, 2,500 FMA tracks, hand-feature sorter top-1 0.447). Its data cache comes to about 16 GB (FMA small plus the extracted copy and analysis, as that thread measured), over the 10 GB Actions limit, so it gets evicted and every run downloads again. sorting-fma works in CI, just slowly.
- Shipped: this log (PR #32). No code change.
- Tried: whether a techno bassline in the kick band (20-150 Hz) explains techno being misread (21 of 61 GiantSteps techno tracks read as breakbeat, `eval/baselines/datasets.json`). Method: a throwaway script (not kept) added a 55 Hz bass note to `render("four_on_floor", bpm)` from `music_cluster/eval/drums.py`, on the sixteenths the kick leaves free, at 124, 128, 132 and 136 BPM, then ran `tagging.tag_audio`.
  - Bass on the offbeats, or a rolling bass up to about half the kick's level: still four-to-the-floor.
  - A rolling bass as loud as the kick at 128 and 132: read as two_step, not breakbeat; the bass outweighs the kick in the kick band.
  - Added to every groove as an eval variant, it mostly made slow grooves (dembow 95, backbeat 85) read at double time as four-to-the-floor, because every sixteenth then has an onset. Not a realistic case, so the variant was not kept.
  - So the bass-in-kick-band idea does not reproduce the breakbeat misreads. Next step for that bug is to look at the misread tracks themselves: dispatch `eval.yml` with `key-giantsteps tempo-giantsteps` and read the failures list for techno, rather than guess from synthetic audio.
- Next: the GiantSteps techno failures as above; the novelty check (`novelty_auroc` in EVALUATION.md: whether a track that fits no group sits further from every group than tracks that do; 0.5 is chance, and it measures about 0.5); minor-loop keys.
- Budget: 2 runs logged so far, no change to budget_minutes.
- Questions for Jules: none new. Still open: license, commit dialog default mode, shipping AST in the app (needs approval for torch + transformers, about 350 MB). Settled since Nov 8: the product name (Cratewise, CLI renamed in PR #25), and the pretrained embedding question became the AST evaluation in PR #31 (CLAP was tried there and its input is broken).
