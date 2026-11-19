# Evaluation

How well does music-cluster actually sort, tag and measure? `cratewise eval`
answers with numbers instead of a listening session, so a change can be judged
by what it moves.

```bash
cratewise eval run                         # synthetic suites, ~1 min, no downloads
cratewise eval run --baseline eval/baselines/synthetic.json   # and diff against the committed numbers

cratewise eval download giantsteps-key     # ~600 Beatport previews, about 1 GB
cratewise eval download giantsteps-tempo   # ~660 previews
cratewise eval download fma-small          # 8,000 clips, 7.2 GB (+ the same again while unzipping)
cratewise eval run --all                   # every suite whose data is present

cratewise eval run --folders ~/Music/DJ    # simulate sorting into your own folder tree
```

Datasets live in `~/.music-cluster-eval/data` (or `$MUSIC_CLUSTER_EVAL_DATA`, or
`--data-dir`), and the analysis cache in a `.cache` folder inside it: never in
the repository and never in your library database. Each file is decoded once;
later runs read the cache until the measuring code (`extractor.py`,
`features.py`, `tagging.py`, `rhythm.py`) or the analysis settings change, at
which point it is re-analysed. Files that fail to decode are retried on the
next run. Nothing in the harness moves, copies or writes audio files.

Every run writes a JSON report (`--out`, default `eval-report.json`) and a
Markdown summary beside it. `--baseline OLD.json` adds a comparison table and
exits non-zero if any headline number got worse by more than `--tolerance`
(default 0.02), or if a suite that was run no longer reports a number the
baseline has. Only missing data turns a suite into "skipped"; any other error
fails the run.

## Suites

| Suite | Data | What it measures | Baselines beside it |
| --- | --- | --- | --- |
| `beats-synthetic` | Rendered grooves, 13 groove/tempo pairs × 5 roughness variants × 3 kit seeds | Beat-type accuracy, tempo accuracy | – |
| `keys-synthetic` | Rendered chord progressions in all 24 keys, with and without drums | Key accuracy (MIREX weighted); `key_correct_minor_loop` stays at 0, a known limit, see [The minor loop](#the-minor-loop) | – |
| `sorting-fma` | FMA small, 8 top-level genres, artist ids | Simulated libraries: see below | Majority folder, tempo only, logistic regression, kNN |
| `sorting-fma-sub` | FMA medium, sub-genres with 40+ tracks | Same, at the granularity of a DJ's folders | Same |
| `sorting-folders` | Any folder tree you pass with `--folders` | Same, on your own library | Same |
| `key-giantsteps` | GiantSteps Key (Beatport EDM) | Weighted key score, error kinds, per Beatport genre | Most common key; Essentia `KeyExtractor` (EDM profile) if installed |
| `tempo-giantsteps` | GiantSteps Tempo, v2 annotations | Accuracy1 (±4%) and Accuracy2 (octave errors forgiven), beat type per genre, and each misread beat type with its measured kick, snare and hat bar and onsets per beat (`beat_misses` in the JSON report) | librosa `beat_track`; Essentia `RhythmExtractor2013` if installed |

Essentia is an optional reference, not a dependency: `pip install essentia`
and the key and tempo suites score it on the same audio. Its licence (AGPL,
and its models are non-commercial) keeps it out of the app itself.

### The minor loop

The minor loop (i-VI-III-VII, Am-F-C-G in A minor) uses the same four chords,
each for the same time, as the major loop of its relative major (I-V-vi-IV,
C-G-Am-F in C major). Only the order differs, so the mean chroma the key rule
reads is the same for both, and no key profile can get both right.
`key_correct_minor_loop` is 0 and the loop is read as the relative major.

- It is kept as a reminder of what a mean-chroma key can't see, not as a
  target. A fix would need a cue other than the mean chroma (which chord
  starts a phrase, or the bass note under the downbeat), checked on
  GiantSteps, not tuned to this case.
- The cadence progressions return to the tonic and test the key rule itself.

### Simulated libraries

A genre benchmark asks "which of ten balanced genres is this?". The sorter's
job is different, so each sorting trial builds an imaginary library:

1. Pick 4–10 labels as the user's folders and one more whose tracks belong nowhere.
2. Split every label by artist, so within each folder no artist is on both
   sides. (An artist filed under two labels can still appear in one folder's
   references and another's queries; that makes the task harder, not easier.)
3. Give each folder a log-uniform random number of references, 8 to 300.
4. Fit the sorter on the references; score the held-out tracks with it and
   with each baseline.

Fifty trials by default. The report pools every query across trials:

- `top1`, `top3`, `macro_f1` — is the right folder first, or near the top?
  Macro-F1 weighs an 8-track folder as much as a 300-track one.
- `ece`, `brier`, reliability bins — does a confidence of 0.8 mean right 80%
  of the time? The review screen shows it and auto-filing trusts it.
- `recall_by_folder_size` — do small folders still win their own tracks, or do
  big folders win everything?
- `auto_precision_current`, `auto_coverage_current` — what auto-filing does
  today at `auto_accept_confidence` and `min_margin`, counting a
  "belongs nowhere" track that gets filed as an error.
- `threshold_sweep`, `operating_points` — the thresholds that reach 90% and
  95% auto-file precision, and how much of the batch they cover. They are
  picked and reported on the same queries, so treat the coverage as slightly
  optimistic; the grid is coarse (8 × 6 settings), which keeps that small.
- `novelty_auroc` — does distance separate "belongs nowhere" from "belongs
  somewhere"? 0.5 is chance.
- `novelty_auroc_by_score` — the same for other candidate scores, to choose
  the sorter's rule: `group_widths` (the sorter's own), `library_units`,
  `one_minus_confidence`, `nearest_track` (one nearest reference track) and
  `plain_space_group_widths` (the sorter's rule in plain PCA space, without
  the part fitted to tell the folders apart). The report prints them.
  `novelty_auroc_by_score_per_trial` averages each over the simulated
  libraries instead of pooling them; pooled, a score whose range shifts with
  the folder count (confidence) looks worse than it is.
- `nowhere_auto_filed` — share of "belongs nowhere" tracks that clear the
  auto-accept thresholds. Lower is better.

### Belongs nowhere

On FMA small with the hand features, no "belongs nowhere" score is more than
seed noise above chance (run 44399750712, 50 trials, pooled):

| Score | Novelty AUROC |
| --- | --- |
| `group_widths` (the sorter's rule) | 0.503 |
| `library_units` | 0.520 |
| `one_minus_confidence` | 0.497 |
| `nearest_track` | 0.502 |
| `plain_space_group_widths` | 0.512 |

- Seed noise is about 0.02, so the sorter's rule stays as it is. A held-back
  FMA genre sits as close to the folders as their own unseen tracks do.
- With the AST embedding alone (the `+ast:embedding` result on 2,500 FMA tracks,
  the run in PR #31), the pooled `novelty_auroc` is 0.52 as well, so a better
  space alone has not fixed it either.

### Pretrained embeddings

`--embedding ast` checks whether a network trained on lots of audio
describes music better than the 96 hand features do. It needs a sorting
suite. On each sorting suite it embeds every track and adds three results,
all on the same tracks with the same trials, so the comparison is paired:

- `SUITE+ast:features`: the hand features, the reference.
- `SUITE+ast:embedding`: the embedding alone.
- `SUITE+ast:both`: the embedding appended to the hand features.

Each also reports `probe_top1`: logistic regression on every track, in five
folds split by artist. It shows how much the features say about the label
when references are plentiful, apart from how the sorter uses them.

A track's embedding is the mean of three windows from the middle of the file.
The model needs `pip install torch transformers` (eval.yml pins the major
versions), and it downloads its weights on first use. The app doesn't use
it; this only measures whether it should. If embedding fails, the
hand-feature result is still reported and the embedding result is marked
skipped or failed. These results are not gated.

| Name | Model | Licence |
| --- | --- | --- |
| `ast` | `MIT/ast-finetuned-audioset-10-10-0.4593` | BSD-3-Clause |

AST takes about 4 seconds a track on a CI runner's CPU, so use the `limit`
input for FMA.

## Ground rules

- Splits are by artist, within each folder, wherever the dataset knows the artist.
- Tags are computed without file metadata, so a genre tag in the file cannot
  leak into a genre score.
- Reports commit, data does not. `eval/baselines/` holds the numbers each
  suite is compared against; refresh one deliberately, in the PR that earns it.
  `synthetic.json` is deterministic. `datasets.json` was measured on a Mac with
  Essentia installed; at 50 trials the sorting numbers move about 0.02 between
  seeds, so compare dataset runs with `--tolerance 0.03`:
  `cratewise eval run --suite sorting-fma --baseline eval/baselines/datasets.json --tolerance 0.03`.
  Its headline also carries the baselines' own scores (`logistic_top1`,
  `librosa_accuracy1`), so a library upgrade can move those without our code
  changing.

## CI

`.github/workflows/eval.yml` runs the synthetic suites against
`eval/baselines/synthetic.json` on every pull request that touches
`music_cluster/`, and weekly. The dataset suites run from the
Actions tab (`workflow_dispatch`) with the suites to run as input; the
downloaded data is cached between runs. Those runs report their numbers but
are not gated: compare them against `datasets.json` by hand, as above.

## Adding a suite

Write a function that returns a dict with a `headline` of numbers (and
`lower_is_better` naming any where smaller is better), register it in
`music_cluster/eval/runner.py`, and add a loader to
`music_cluster/eval/datasets.py` if it needs new data. Unit-test the metric on
hand-made numbers in `tests/unit/test_eval.py`.
