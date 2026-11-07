# Evaluation

How well does music-cluster actually sort, tag and measure? `music-cluster eval`
answers with numbers instead of a listening session, so a change can be judged
by what it moves.

```bash
music-cluster eval run                         # synthetic suites, ~1 min, no downloads
music-cluster eval run --baseline eval/baselines/synthetic.json   # and diff against the committed numbers

music-cluster eval download giantsteps-key     # ~600 Beatport previews, about 1 GB
music-cluster eval download giantsteps-tempo   # ~660 previews
music-cluster eval download fma-small          # 8,000 clips, 7.2 GB (+ the same again while unzipping)
music-cluster eval run --all                   # every suite whose data is present

music-cluster eval run --folders ~/Music/DJ    # simulate sorting into your own folder tree
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
| `keys-synthetic` | Rendered chord progressions in all 24 keys, with and without drums | Key accuracy (MIREX weighted) | – |
| `sorting-fma` | FMA small, 8 top-level genres, artist ids | Simulated libraries: see below | Majority folder, tempo only, logistic regression, kNN |
| `sorting-fma-sub` | FMA medium, sub-genres with 40+ tracks | Same, at the granularity of a DJ's folders | Same |
| `sorting-folders` | Any folder tree you pass with `--folders` | Same, on your own library | Same |
| `key-giantsteps` | GiantSteps Key (Beatport EDM) | Weighted key score, error kinds, per Beatport genre | Most common key; Essentia `KeyExtractor` (EDM profile) if installed |
| `tempo-giantsteps` | GiantSteps Tempo, v2 annotations | Accuracy1 (±4%) and Accuracy2 (octave errors forgiven), beat type per genre | librosa `beat_track`; Essentia `RhythmExtractor2013` if installed |

Essentia is an optional reference, not a dependency: `pip install essentia`
and the key and tempo suites score it on the same audio. Its licence (AGPL,
and its models are non-commercial) keeps it out of the app itself.

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
- `recall_by_folder_size` — does scale normalisation really stop big folders
  winning everything?
- `auto_precision_current`, `auto_coverage_current` — what auto-filing does
  today at `auto_accept_confidence` and `min_margin`, counting a
  "belongs nowhere" track that gets filed as an error.
- `threshold_sweep`, `operating_points` — the thresholds that reach 90% and
  95% auto-file precision, and how much of the batch they cover. They are
  picked and reported on the same queries, so treat the coverage as slightly
  optimistic; the grid is coarse (8 × 6 settings), which keeps that small.
- `novelty_auroc` — does distance separate "belongs nowhere" from "belongs
  somewhere"? 0.5 is chance.

## Ground rules

- Splits are by artist, within each folder, wherever the dataset knows the artist.
- Tags are computed without file metadata, so a genre tag in the file cannot
  leak into a genre score.
- Reports commit, data does not. `eval/baselines/` holds the numbers each
  suite is compared against; refresh one deliberately, in the PR that earns it.

## CI

`.github/workflows/eval.yml` runs the synthetic suites against
`eval/baselines/synthetic.json` on every pull request that touches
`music_cluster/`, and weekly. The dataset suites run from the
Actions tab (`workflow_dispatch`) with the suites to run as input; the
downloaded data is cached between runs.

## Adding a suite

Write a function that returns a dict with a `headline` of numbers (and
`lower_is_better` naming any where smaller is better), register it in
`music_cluster/eval/runner.py`, and add a loader to
`music_cluster/eval/datasets.py` if it needs new data. Unit-test the metric on
hand-made numbers in `tests/unit/test_eval.py`.
