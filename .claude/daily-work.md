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

### 2026-11-10

- Minutes: ~60 of 60 active (plus about 70 minutes waiting on the dataset run in Actions). Subagents: 2 (independent reviews of PR #33, one at a time).
- State: main green. No open issues. Open PR #31 (AST embedding eval) belongs to another thread.
- Shipped: PR #33 (https://github.com/julesyoungberg/music-cluster/pull/33). The `tempo-giantsteps` report now lists every track from a genre whose beat type is not in doubt and whose beat type came out wrong (`beat_misses`): genre, measured and annotated BPM, the three strongest beat rules, and the kick and snare bar as 16 digits (one per sixteenth, 0 silent to 9 loudest). The Markdown summary prints the same table, so it shows in the Actions job log.
  - Why the log: the report artifact is stored on a host the cloud session cannot reach (`*.blob.core.windows.net` is refused by the proxy), but job logs are readable through the GitHub tools.
- Dataset run on the PR branch (run 42905140242, then 42916100263 from the cached analysis): headline numbers match `eval/baselines/datasets.json` exactly (key weighted 0.589; tempo acc1 0.678, acc2 0.885; beat_genre_accuracy 0.568). The data cache had been evicted and was downloaded again (36 min); it is saved again under the new analysis code.
- What the techno misses show (27 of 61 techno tracks wrong in the run's table; the Nov 7 note of 21 counted only "breakbeat"):
  - Beat grid half a beat off the kick, so the kick sits on the offbeat eighth (bar like `0080009000800090`), or three sixteenths late: 7 tracks (1816200, 2726350, 2745205, 3443201, 4166612, 4207348, 4494659). `beat_scores` does try the bar shifted by 1 to 3 sixteenths for four-to-the-floor, but that reading gets `_ROTATION_PENALTY` (0.93), and the breakbeat rule scores the same off-grid kicks at up to 1.00, so breakbeat wins.
  - Tempo read about 4/3 too fast (172 to 185 BPM for tracks annotated 127 to 142): 7 tracks (1548088, 3241615, 3298908, 3435022, 4186191, 4358231, 4984238). Trance has the same error on about 10 tracks at 184.57 for 139. With the wrong tempo every bar is misread.
  - Read as half-time with the kick on the beat but sparse: 4 tracks (1791554, 3692859, 5157605, 956606).
  - The rest are mixed: kick and snare bands carrying the same pattern, almost no kick, or half tempo.
- Next:
  1. Offbeat-grid four-to-the-floor (largest fixable group): when the bar shifted by two sixteenths has a kick on every beat, prefer four-to-the-floor over breakbeat instead of letting the penalty hand it to breakbeat. Change goes in `tagging.py`, judge it with a `key-giantsteps tempo-giantsteps` dispatch (beat_genre_accuracy, and check drum & bass and dubstep did not get worse) and `beats-synthetic`.
  2. The 4/3 tempo error in `rhythm.py` (techno and trance): look at why the tracker locks onto dotted eighths.
  3. Novelty check (AUROC about 0.5) and minor-loop keys, as before.
- Budget: last 3 runs used about 50, 55 and 60 of 60; not all hit the cap, so budget_minutes stays 60.
- Questions for Jules: none new. Still open: license, commit dialog default mode, shipping AST in the app (needs approval for torch + transformers, about 350 MB).

### 2026-11-11

- Minutes: ~50 of 60 active (plus about 50 minutes waiting on the dataset run, whose cache had been evicted again). Subagents: 1 (independent review of PR #34).
- State: main green. No open issues. Open PR #31 (AST embedding eval) belongs to another thread.
- Shipped: PR #34 (https://github.com/julesyoungberg/music-cluster/pull/34), item 1 from the Nov 10 list. In `tagging.py`:
  - Breakbeat rule: a kick on every beat at any fixed offset (`steady_kick`) counts against a break, not only a kick on the grid's quarters.
  - Four-to-the-floor dembow guard: odd-sixteenth snares are measured against a typical even one (`_odd_snare_lead`, ramp 0.2 to 0.45), so hats bleeding into the snare band no longer fail it.
- Numbers: GiantSteps run 43102415466: beat_genre_accuracy 0.568 to 0.588 (misses 211 to 201 of 488); techno misses 27 to 24, tech-house 13 to 10; drum & bass 43 to 44 and dubstep 47 unchanged; key and tempo unchanged. Synthetic eval unchanged. Baseline refreshed in the same PR (`eval/baselines/datasets.json`, `.md`).
- Method worth reusing: the misses table in the job log can be re-scored offline by rebuilding each bar from its 16 digits and running `tagging.beat_scores`. It reproduced 180 of 211 logged results and predicted the direction of the change. It can't see tracks that were right and turned wrong, so the dataset run still decides.
- Review findings left open on purpose (answered on PR #34):
  - The shifted four-to-the-floor reading in `beat_scores` wins outright on late half-time and backbeat bars with no kick on any beat (e.g. kick [1,11], snare [9] at 140 reads four_on_floor 0.698). This is in main already. The comment there says "can tie", which understates it. An earlier gate on the kick cost 0.031; try gating on `steady_kick` instead.
  - A dembow over a shaker at 0.6 on every step reads as four-to-the-floor.
- Gotcha: the local `origin/main` ref in a fresh session can be stale (it pointed at abbb64f). Compare against the PR's parent commit, not `origin/main`, unless you fetch first.
- Next:
  1. The 4/3 tempo error in `rhythm.py` (techno and trance read at 172 to 185 for 127 to 142).
  2. Gate the shifted four-to-the-floor reading on `steady_kick` (above), judged by beat_genre_accuracy.
  3. Dubstep: 47 misses, most read as breakbeat. Not looked at yet.
  4. Novelty check (AUROC about 0.5) and minor-loop keys, as before.
- Budget: last 3 runs used about 55, 60 and 50 of 60; not all hit the cap, so budget_minutes stays 60.
- Questions for Jules: none new. Still open: license, commit dialog default mode, shipping AST in the app (needs approval for torch + transformers, about 350 MB).

### 2026-11-12

- Minutes: ~55 of 60 active (plus about 75 minutes waiting on two dataset runs; the data cache had been evicted again). Subagents: 1 (independent review of PR #35).
- State: main green. No open issues. PR #31 (AST embedding eval, another thread) merged during this run.
- Shipped: PR #35 (https://github.com/julesyoungberg/music-cluster/pull/35), item 1 from the Nov 11 list, merged by Jules.
  - Cause of the 4/3 tempo error, reproduced on rendered audio: a riff or delay every three sixteenths gives the tracker a pulse a third faster than the kick. Four-to-the-floor at 139 plus such a riff reads 184.6, the exact number on the GiantSteps trance misses. The bars still agree, so the retrack below 0.85 consistency never ran.
  - Fix in `rhythm.py`: if the folded kick sits on every third of the bar (steps 16 and 32) and off beats two to four, track again at exactly 3/4 of the tempo and keep it if its bars hold together (consistency >= 0.66, `SLOWER_CONSISTENCY_FLOOR`).
  - Review found the first version also fired on a broken kick (sixteenths 0, 5, 11) and a 12/8 half-note-triplet kick at their right tempo; the consistency floor fixed that.
- Numbers (GiantSteps, `key-giantsteps tempo-giantsteps`):
  - First commit, no floor (run 43310234461): tempo acc1 0.678 to 0.697, acc2 0.885 to 0.906, beat_genre_accuracy 0.588 to 0.600.
  - Merged head with the 0.66 floor (run 43318548417): acc1 0.684, acc2 0.891, beat_genre_accuracy 0.594. Key unchanged (0.589). Baseline headline refreshed to these in this log PR.
  - So the floor gave back about half the gain: some real trance and techno re-tracks score under 0.66 (e.g. 28952, 310291, 3211234, 3241615, 3435022, 4077896, 4151958 went back to 172 or 184.57).
- Synthetic: beat_accuracy 0.882 to 0.923, tempo_accuracy1 0.867 to 0.923 (baseline refreshed in #35).
- Gotcha: once a work branch's PR merges, a fresh push to the same branch needs a force push, which the session refuses. Put follow-ups on a new branch.
- Next:
  1. Tune `SLOWER_CONSISTENCY_FLOOR`: real tracks need it lower than 0.66, rendered false cases sit at 0.50-0.61. Try a drop relative to the tracked reading's consistency instead of a fixed floor, or a second guard. Judge with a GiantSteps dispatch; the cache is warm under the current `rhythm.py` hash.
  2. Gate the shifted four-to-the-floor reading on `steady_kick` (from Nov 11).
  3. Dubstep misses (most read as breakbeat), novelty check, minor-loop keys.
- Budget: last 3 runs used about 60, 50 and 55 of 60; not all hit the cap, so budget_minutes stays 60.
- Questions for Jules: none new. Still open: license, commit dialog default mode, shipping AST in the app.
