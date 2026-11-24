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

### 2026-11-13

- Minutes: ~55 of 60 active (plus about 60 minutes waiting on two dataset runs). Subagents: 1 (independent review of PR #41).
- State: main green. No open issues or PRs.
- Tried item 1 from the Nov 12 list (the 3/4 re-track floor) in PR #41 (https://github.com/julesyoungberg/music-cluster/pull/41). Closed without merging: no effect on GiantSteps.
  - Rendered with timing jitter, velocity jitter and noise, a four-to-the-floor under a dotted-eighth riff re-tracks at 3/4 with bar consistency 0.47 to 0.64 over seeds and tempi. The broken kick and 12/8 kick false cases score 0.51 to 0.63. So bar consistency alone can't separate them, whatever the floor or a ratio to the tracked reading.
  - First version: lower the floor to 0.55 and skip the re-track when the tracked grid shows a backbeat. Review found it re-tracked a broken or 12/8 kick under a half-time snare (no backbeat), and an accented riff looked like a backbeat and blocked a correct re-track.
  - Second version: floor 0.66, plus a slower reading at 0.55 to 0.66 wins if it shows a backbeat the tracked grid lacks. Synthetic unchanged; GiantSteps run 43510317294 identical to main on every number (key 0.589, tempo 0.684/0.891, beat_genre_accuracy 0.594). The real misses don't show a backbeat on the slower grid.
  - Not fixed on main either, found while checking the review (its scripts run against main): a 12/8 kick under a half-time snare at 120, 140 and 170, and a broken kick with no snare at 125 and 140, read at the wrong tempo.
- Next:
  1. 3/4 re-track: drop bar consistency as the judge. Look at the GiantSteps tracks that went back to 172 or 184.57 (28952, 310291, 3211234, 3241615, 3435022, 4077896, 4151958) for something the slower reading shows that a broken kick doesn't, e.g. the kick on all four beats of the slower grid. The misses table in the job log has their bars.
  2. Gate the shifted four-to-the-floor reading on `steady_kick` (from Nov 11).
  3. Dubstep misses (most read as breakbeat), novelty check, minor-loop keys.
- Budget: last 3 runs used about 50, 55 and 55 of 60; not all hit the cap, so budget_minutes stays 60.
- Questions for Jules: none new. Still open: license, commit dialog default mode, shipping AST in the app.

### 2026-11-14

- Minutes: ~55 of 60 active (plus about 75 minutes waiting on two dataset runs; the first re-downloaded the data). Subagents: 1 (independent review of PR #43, run twice).
- State: main green. No open issues or PRs.
- Skipped items 1 and 2 from the Nov 13 list after a look: the 3/4 re-track has failed twice on GiantSteps, and the `steady_kick` gate on the late-grid reading is the same kick gate that cost 0.031 in PR #12 (many real techno and trance bars measure no steady kick).
- Shipped: PR #43 (https://github.com/julesyoungberg/music-cluster/pull/43), item 3 (drum & bass and dubstep misses), in `tagging.py`:
  - Cause: the four-to-the-floor rule averages four terms, so a sparse bar with no kick on the beat still scored 0.75. In run 43510317294, 31 of 44 drum & bass misses and 16 of 33 dubstep misses read as four-to-the-floor.
  - Fix: the four terms are scaled by the larger of a steady-kick ramp, a 110-150 bpm window, and the same window on half the tracked tempo.
  - First form (a fifth averaged term) broke slow house, hard techno and dembow over a shaker in review. Gate form without the half-tempo window (run 43642390663): 0.594 to 0.555, because 33 four-to-the-floor tracks are tracked at double (235-287).
- Numbers: GiantSteps run 43647103782 at f60d4aa: beat_genre_accuracy 0.594 to 0.623 (misses 198 to 184; 19 drum & bass and 1 dubstep fixed, 6 four-to-the-floor tracks lost, all read at 83-198 bpm). Key 0.589 and tempo 0.684/0.891 unchanged. Synthetic unchanged. Baseline refreshed in #43.
- Gotcha: the GiantSteps cache restored on the second run (download 7 s, analysis 24 min), so back-to-back runs on one branch are about 25 minutes.
- Next:
  1. Dubstep: 11 still read as four-to-the-floor at about 140, inside the window, and 19 as breakbeat. Tempo can't separate them from techno there; look at the snare (a single snare on three) or the low end.
  2. Drum & bass read as backbeat (14) at 86-89: the backbeat rule's 60-110 window takes half-tempo drum & bass.
  3. 3/4 re-track, novelty check, minor-loop keys, as before.
- Budget: last 3 runs used about 55, 55 and 55 of 60; not all hit the cap, so budget_minutes stays 60.
- Questions for Jules: none new. Still open: license, commit dialog default mode, shipping AST in the app.

### 2026-11-15

- Minutes: ~55 of 60 active (plus waiting on CI and a dataset run). Subagents: 1 (independent review of PR #44).
- State: main green. No open issues or PRs.
- Took item 2 from the Nov 14 list (drum & bass read as backbeat at 86-89). The bar alone can't fix it: tracked at half speed, a drum & bass bar looks just like a hip hop backbeat, so the fix belongs in the octave choice in `rhythm.py` (`_reads_as_double_time`: busy-sixteenth share >= 0.7 or onsets per beat >= 2.8 on the slow reading). The misses table showed no hats or onset counts, so there was nothing to tune against.
- Shipped: PR #44 (https://github.com/julesyoungberg/music-cluster/pull/44): the misses table and `beat_misses` gain the hat bar and onsets per beat on the slow reading. No numbers move. Review found 5 issues (onsets counted against the doubled tempo, docstring claims, a missing test, formatting); all fixed.
- Data run: GiantSteps dispatch 43765955305 on the merged branch head; its job log has the new columns.
- Gotcha: the session's local `origin/main` was stale (b09dedf), so the first branch was cut from it and the PR showed a conflict and got no CI. Run `git fetch origin main` before branching.
- Next:
  1. Read run 43765955305's misses: for the drum & bass rows at 83-89 bpm, compare onsets per beat and the hat bar against the 2.8 and 0.7 thresholds. If they sit just under, a lower threshold may double them; check that the synthetic boom-bap and backbeat bars still read slow (`cratewise eval run --baseline eval/baselines/synthetic.json`). GiantSteps has only 2 hip-hop tracks, so it can't show harm to slow genres on its own.
  2. Dubstep at about 140 read as four-to-the-floor (11) or breakbeat (19).
  3. 3/4 re-track, novelty check, minor-loop keys, as before.
- Budget: last 3 runs used about 55, 55 and 55 of 60; not all hit the cap, so budget_minutes stays 60.
- Questions for Jules: none new. Still open: license, commit dialog default mode, shipping AST in the app.

### 2026-11-16

- Minutes: ~45 of 60 active (plus about 45 minutes waiting on the dataset run; the data cache had been evicted). Subagents: 1 (independent review of PR #45).
- State: main green. No open issues or PRs.
- Item 1 from the Nov 15 list (drum & bass read as backbeat at 83-89): run 43765955305 shows onsets per beat from 0.9 to 2.8 on those rows, spread widely under the 2.8 threshold, not just under it. 8 of the 25 drum & bass misses are annotated at half tempo themselves. A lower threshold can't be judged without hip hop data, so it was left alone.
- Tried item 2 (dubstep read as four-to-the-floor at about 140) in PR #45 (https://github.com/julesyoungberg/music-cluster/pull/45). Closed without merging.
  - Most of the 11 rows have one kick a bar (e.g. `7000000000000000`), which still scored 0.75 from the rule's absence terms. Gate tried: second-loudest kick on a quarter, ramp 0.15-0.35 (review lowered it from 0.3-0.6 for accented downbeats).
  - GiantSteps run 43959295898: beat_genre_accuracy 0.623 to 0.578. 1 dubstep fixed, 23 lost: 20 trance, techno and house tracked at double tempo (235-287) whose half bar measures a near-zero second kick. Exempting double tempo still nets -1.
- Next:
  1. Dubstep: the kick band can't separate it from four-to-the-floor at 140. Try the snare (one hit on three, the half_time rule) or the low end.
  2. 3/4 re-track, novelty check (AUROC about 0.5), minor-loop keys, as before.
  3. Beat-type tuning has given small or no gains for four runs. A run on the novelty check or another roadmap item may be worth more.
- Budget: last 3 runs used about 55, 55 and 45 of 60; not all hit the cap, so budget_minutes stays 60.
- Questions for Jules: none new. Still open: license, commit dialog default mode, shipping AST in the app.

### 2026-11-17

- Minutes: ~35 of 60 active. Subagents: 1 (independent review of PR #47).
- State: main green. No open issues or PRs.
- Skipped beat-type tuning (small or no gains for four runs) and the novelty check (a sorter change needs a sorting-fma dispatch whose 16 GB cache is evicted; too slow to iterate inside the budget without a plan).
- Checked docs against code instead. Shipped PR #47 (https://github.com/julesyoungberg/music-cluster/pull/47):
  - `docs/cli.md` gains `eval run --embedding`, the one undocumented option; `tests/unit/test_cli_docs.py` now fails on any CLI command or option missing from the doc.
  - The synthetic minor loop (i-VI-III-VII) is the major loop of its relative major (I-V-vi-IV) in another order, same chords for the same time. Its mean chroma is the same, so `key_correct_minor_loop` 0/24 can't be fixed by any key profile. Documented in EVALUATION.md ("The minor loop"). Stop listing "minor-loop keys" as a fix target; a real fix needs a cue beyond the mean chroma (bass under the downbeat, phrase start) and must be judged on GiantSteps.
- Next:
  1. Novelty check (AUROC about 0.5 on hand features and AST alike, so the distance rule may be the limit, not the features). Plan it first: list two or three alternative rules (e.g. kNN distance to the nearest reference track instead of group widths) and test them in one sorting-fma dispatch with a warm cache.
  2. Dubstep at about 140 (snare on three or the low end), 3/4 re-track, as before.
- Budget: last 3 runs used about 55, 45 and 35 of 60; not all hit the cap and not all under half, so budget_minutes stays 60.
- Questions for Jules: none new. Still open: license, commit dialog default mode, shipping AST in the app.

### 2026-11-18

- Minutes: ~40 of 60 active (plus waiting on the dataset run). Subagents: 1 (independent review of PR #48).
- State: main green. No open issues or PRs.
- Took item 1 from the Nov 17 list (novelty check), measuring first. Shipped PR #48 (https://github.com/julesyoungberg/music-cluster/pull/48): the sorting suites report `novelty_auroc_by_score`, pooled and averaged per trial, for the current rule (`group_widths`) and four candidates: `library_units`, `one_minus_confidence`, `nearest_track`, `plain_space_group_widths` (the rule in PCA space without the folder-separating part). Sorter unchanged. Review found 5 issues; 3 fixed (per-trial AUROC, since pooled confidence mixes in the folder count; cdist; public `GroupSorter.global_scale`), 2 answered.
- Data run: sorting-fma dispatch 44399750712 on the first commit (pooled numbers only; the FMA cache had been evicted, so it downloads and analyses from scratch). Numbers go on PR #48 when it finishes.
- Next:
  1. Read the novelty lines in run 44399750712's job log (or re-dispatch sorting-fma on main for per-trial numbers; the cache should be warm). If a candidate clearly beats 0.5, switch the sorter's `_decide` to it in one PR, judged on `nowhere_auto_filed`, top-1 and the baseline in `eval/baselines/datasets.json`. If none does, the "nowhere" genre in FMA may simply overlap the folders; say so in EVALUATION.md and stop.
  2. Dubstep at about 140 (snare on three or the low end), 3/4 re-track, as before.
- Budget: last 3 runs used about 45, 35 and 40 of 60; not all under half, so budget_minutes stays 60.
- Questions for Jules: none new. Still open: license, commit dialog default mode, shipping AST in the app.

### 2026-11-19

- Minutes: ~25 of 60 active. Subagents: 1 (independent review of the PR).
- State: main green. No open issues or PRs.
- Item 1 from the Nov 18 list: run 44399750712 put every novelty candidate at 0.50-0.52 (numbers on PR #48). Wrote the result into EVALUATION.md ("Belongs nowhere") and stopped, as planned. Shipped in PR #56 (https://github.com/julesyoungberg/music-cluster/pull/56). Review also caught a #55 bug: analysing a music file as a sample dropped its AST embedding; fixed there.
- Looked at item 2 (dubstep at about 140) in run 43765955305's misses: 45 dubstep rows, losing mostly to breakbeat (snare band busy on most sixteenths) and to four-to-the-floor (one kick a bar). No single bar cue separates them, and the misses table lists no correct reads, so a rule change can't be checked for harm before a GiantSteps run. Not tried.
- Next: beat-type tuning and the novelty check have both stopped paying off. The bigger levers need Jules: shipping AST (top-1 0.646 vs 0.447 on FMA, needs torch + transformers), or a direction for new work.
- Budget: last 3 runs used about 35, 40 and 25 of 60; not all under half, so budget_minutes stays 60.
- Questions for Jules: what to work on next (see Next). Still open: license, commit dialog default mode, shipping AST in the app.

### 2026-11-20

- Minutes: ~15 of 60 active. Subagents: 1 (independent review of the log PR).
- State: main green (all four workflows on c9cea34). No open issues or PRs.
- No reply yet to the Nov 19 question (ship AST in the app, or another direction). No roadmap item is left that doesn't need Jules: beat-type tuning and the novelty check have stopped paying off (see the Nov 16 to Nov 19 entries), and README, docs and code list no unbuilt features. Started no new work rather than invent a feature.
- Housekeeping found: 35 remote branches whose PRs are merged or closed (#1 to #49, e.g. `claude/project-thread-rkela7`, `daily-work-log`, `add-graphify`). Not deleted; deleting remote branches needs Jules's OK.
- Next (all need Jules): ship AST (FMA top-1 0.646 vs 0.447, adds torch + transformers, about 350 MB); or a new direction; plus the standing questions below and the branch cleanup.
- Budget: last 3 runs used about 40, 25 and 15 of 60; not all under half, so budget_minutes stays 60.
- Questions for Jules: what to work on next. Still open: license, commit dialog default mode, shipping AST in the app, deleting the 35 merged branches.

### 2026-11-21

- Minutes: ~45 of 60 active. Subagents: 1 (independent review of PR #55).
- State: main green. No open issues or PRs. Jules picked "Ship AST in the app" on the Nov 20 card.
- Shipped: PR #55 (https://github.com/julesyoungberg/music-cluster/pull/55). Review found 7 issues, all fixed (a global `ast` setting broke sample collections; unembeddable tracks vanished from sessions; stale embeddings after re-analysis). `sorting.feature_set: ast` makes the sorter fit and score on the AST embedding (FMA top-1 0.646 vs 0.447). Opt-in extra `cratewise[ast]` (torch, transformers); embeddings stored in a new `embeddings` table on first fit or sort; changing the setting refits. Music collections only. Not merged: it adds large dependencies, so it waits for Jules (promised in the Nov 20 thread).
- Real-model check: huggingface.co and download.pytorch.org are blocked from the cloud session. Dispatched eval.yml (sorting-fma, limit 300, embedding ast) on the branch to prove the moved model loader still works.
- Next:
  1. Desktop app: bundle torch in the PyInstaller sidecar (size, build time, `COLLECT_ALL` in `scripts/build_sidecar.py`) and add a Settings toggle. Plan first; it is a large build change.
  2. Try AST on Jules's own crates (Mac via Remote Control): the eval number is FMA only.
  3. Embed during `analyze` instead of at the first fit, so the wait comes with the analysis the user already expects.
- Budget: last 3 runs used about 25, 15 and 45 of 60; not all under half, so budget_minutes stays 60.
- Questions for Jules: OK to merge PR #55 (adds torch + transformers as an optional extra)? Still open: deleting the merged branches (script in project files).

### 2026-11-22

- Minutes: ~45 of 60 active. Subagents: 1 (independent review of PR #56: 6 findings, 4 fixed, 2 answered).
- State: main green. No open issues or PRs. PR #55 (AST feature set) merged Nov 21.
- Took item 3 from the Nov 21 list. `analyze` (CLI and API) now embeds music with AST right after measuring it when `sorting.feature_set` is ast, with its own "Embedding" progress, so the first fit or sort no longer waits. Missing torch fails before analysis starts. Other callers of `analyze_files` (discover, tags, group import, sort) don't embed: discover and tags never use it, and fit/sort embed lazily as before through the shared `library.embed_tracks`. Shipped in PR #56 (https://github.com/julesyoungberg/music-cluster/pull/56). Review also caught a #55 bug: analysing a music file as a sample dropped its AST embedding; fixed there.
- graphify isn't installed in the cloud session, so `graphify update .` was not run.
- Next:
  1. Desktop app: bundle torch in the PyInstaller sidecar and add a Settings toggle. Plan first; large build change and size question for Jules.
  2. Try AST on Jules's own crates (Mac via Remote Control): the eval number is FMA only.
- Budget: last 3 runs used about 15, 45 and 45 of 60; not all under half, so budget_minutes stays 60.
- Questions for Jules: is a much larger desktop app (torch alone is several hundred MB) acceptable to ship AST in it, or keep AST CLI-only? Still open: deleting the merged branches (script in project files).

### 2026-11-23

- Minutes: ~30 of 60 active. Subagents: 1 (independent review of the PR).
- State: no open issues or PRs. Main's desktop build had one red job (macos-x86_64 on a4fce40 and cd6010a): the runner couldn't resolve static.rust-lang.org (DNS), not a code fault. Re-ran the failed job.
- No answer yet on the Nov 22 question (AST in the desktop app), so item 1 stays parked. Item 2 (AST on Jules's crates) needs the Mac; not started unattended.
- Shipped the leftover from PR #56: `sort` (CLI and app) shows an "Embedding" progress bar while it embeds new tracks with AST, instead of a silent wait. Same callback as `analyze`.
- Review found the refit path is still silent: switching a collection to AST makes the next `sort` or `fit` embed every reference track (the long wait) with no progress. Left for a follow-up (needs one embed pass over all group tracks in `fit_collection`, so the bar doesn't restart per group).
- Next:
  1. Progress for the refit embedding in `fit_collection` (`fit`, and `sort` when the sorter is stale). No Jules input needed.
  2. Desktop app: bundle torch in the sidecar, if Jules says yes to the larger app.
  3. Try AST on Jules's own crates (Mac via Remote Control).
- Budget: last 3 runs used about 45, 45 and 30 of 60; not all under half, so budget_minutes stays 60.
- Questions for Jules: AST in the desktop app (several hundred MB larger) or CLI only? Still open: deleting the merged branches (script in project files).

### 2026-11-24

- Minutes: ~35 of 60 active. Subagents: 1 (independent review of the PR).
- State: main green (all four workflows on 6344cb0). No open issues or PRs. PR #58 (AST in the desktop app) merged earlier today from its own thread.
- Took item 1 from the Nov 23 list: the refit embedding now has progress. `fit_collection` embeds every reference track in one pass before loading groups, so the bar covers the whole wait instead of restarting per group. Wired into `fit`, `check --refit`, `sort` (a stale sorter refits first) and the API's fit and re-score tasks. The CLI progress bar now starts a fresh bar for a second pass. Review also caught that a reference file that can't be decoded was decoded twice per fit; fixed.
- graphify isn't installed in the cloud session, so `graphify update .` was not run.
- Next:
  1. Try AST on Jules's own crates (Mac via Remote Control): the eval number is FMA only.
  2. Try the AST desktop build on Jules's Mac (size, startup time, Settings switch).
  3. App: the Overview "Fit now" button shows no task progress, and the Groups page panel is titled "Analysing tracks" during a fit. Small UI fix with a browser test and a shot.
  4. Idea, not requested: the onefile sidecar leaves a temp folder behind each time it is killed; a onedir build would fix it.
- Budget: last 3 runs used about 45, 30 and 35 of 60; not all under half, so budget_minutes stays 60.
- Questions for Jules: none new. Still open: deleting the merged branches (script in project files).
