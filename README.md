# music-cluster

[![Lint](https://github.com/julesyoungberg/music-cluster/actions/workflows/lint.yml/badge.svg)](https://github.com/julesyoungberg/music-cluster/actions/workflows/lint.yml)
[![Unit tests](https://github.com/julesyoungberg/music-cluster/actions/workflows/test-unit.yml/badge.svg)](https://github.com/julesyoungberg/music-cluster/actions/workflows/test-unit.yml)
[![End-to-end tests](https://github.com/julesyoungberg/music-cluster/actions/workflows/test-e2e.yml/badge.svg)](https://github.com/julesyoungberg/music-cluster/actions/workflows/test-e2e.yml)
[![Build desktop app](https://github.com/julesyoungberg/music-cluster/actions/workflows/build.yml/badge.svg)](https://github.com/julesyoungberg/music-cluster/actions/workflows/build.yml)

Sort new music into the folders you already keep.

You buy fifteen tracks, and now you have to decide which crate each one goes in.
music-cluster learns what your existing genre folders sound like and, for every
new track, tells you where it thinks it belongs and how sure it is. You listen,
you decide, and it files the results.

**You define the groups.** The tool never invents a category and never moves a
file you have not confirmed.

## The two workflows

**I already have folders.** Point it at the parent of your genre folders. Each
subfolder becomes a group, with itself as the destination for new music. Then
sort a folder of new buys against them.

```bash
music-cluster init
music-cluster groups import-tree ~/Music/DJ    # each subfolder becomes a group
music-cluster fit                              # learn what each one contains
music-cluster check                            # are they actually distinguishable?

music-cluster sort ~/Downloads/NewMusic        # score, then review interactively
music-cluster apply 1 --mode copy              # write it to disk
```

**I have one big unsorted pile.** Discovery breaks it into candidate piles, each
with a handful of representative tracks to audition. You keep the ones that are
real, name them, and throw away the rest — then sort against them like any other
groups.

```bash
music-cluster discover ~/Music/Unsorted        # propose candidate groups
music-cluster candidates 1                     # audition, name, keep or discard
music-cluster fit
```

Nothing becomes a group until you say so.

## Desktop app

The same workflow with audio previews and waveforms, which is what you actually
want when deciding whether a track is deep house or tech house.

```bash
python start-dev.py        # API + UI at http://localhost:1420
python start-dev.py --tauri
```

The review screen plays each track, shows the ranked groups with confidence
bars, and takes `1`–`5` to file, `s` to skip, `g` to pick any group. Committing
shows exactly what would be written before anything is.

## How the sorting works

Your folders are the training data. That is the whole idea, and it is what makes
this different from clustering a library and hoping the results mean something.

1. **Features.** Every track gets a 96-dimension vector: MFCCs and spectral
   shape (timbre), tempo and onset statistics (rhythm), chroma (harmony), RMS
   and dynamics. Tags — artist, title, genre, BPM, key — are read at the same
   time.

2. **A space tuned to your boundaries.** Reference tracks carry group labels, so
   the feature space is fitted with LDA, which finds the directions that
   separate *your* groups, blended with PCA components that preserve general
   audio structure. Pure LDA would flatten everything onto `n_groups − 1` axes
   and lose the ability to notice a track that resembles nothing you own.

3. **Fair comparison across group sizes.** A track's distance to a group blends
   its *k* nearest references (handles a folder containing several distinct
   sounds) with the group centroid (stable for a group of eight), then divides
   by that group's own internal spread. Without that last step a sprawling
   900-track folder wins everything and an 8-track crate never wins anything.

4. **A decision, with a confidence.** Scores across groups yield a confidence
   and a top-1/top-2 margin. Clear the thresholds and a track can be auto-filed;
   otherwise it goes to your queue. Too far from everything and it is reported
   as matching nothing rather than forced into the least-bad group.

5. **Learning.** Tracks you file become references for that group, so the sorter
   follows your taste as it drifts. Turn it off if you would rather curate your
   reference sets by hand.

### Checking your own folders

`music-cluster check` re-sorts every reference track with itself hidden from its
own group and reports what came back where.

```
Group                             Tracks  Correct  Accuracy
Deep House                           412      381       92%
Tech House                           288      201       70%
Minimal                              156       94       60%

Most easily confused:
    52 tracks from 'Minimal' look like 'Tech House' (33%)
```

That is useful information about your library, not just about the tool: two
folders that overlap this much are either the same folder or need clearer
examples of what separates them.

## Filing

Four modes, all with a dry run first and an undo afterwards:

| Mode | What it does |
| --- | --- |
| `playlist` | Writes one M3U per group. Nothing on disk moves. |
| `copy` | Copies into the group's destination folder. |
| `move` | Moves the file, and follows it in the library. |
| `symlink` | Links it into the folder, original stays put. |

```bash
music-cluster apply 3 --mode move --dry-run   # show the plan and stop
music-cluster apply 3 --mode move             # confirm, then do it
music-cluster undo 3                          # put everything back
```

Name collisions are skipped by default (`--on-conflict rename|overwrite`).

## Commands

| | |
| --- | --- |
| `init` | Create the database and config |
| `analyze PATH` | Extract features without sorting |
| `groups import-tree PARENT` | Import every subfolder as a group |
| `groups add NAME --folder/--playlist/--track` | Build a group from a folder, playlist, or seed tracks |
| `groups list` / `show` / `rename` / `set-destination` / `remove` | Manage groups |
| `groups export` | One playlist per group |
| `fit` | Learn the groups |
| `check` | Per-group accuracy and overlap |
| `sort PATH` | Score new music, then review |
| `review ID` / `sessions` | Resume or list review sessions |
| `apply ID` / `undo ID` | Write decisions to disk, or reverse them |
| `discover PATH` / `candidates ID` | Propose and review candidate groups |
| `similar QUERY` | Find tracks that sound like one you name |
| `search QUERY` | Search the library and show group membership |
| `config --set KEY=VALUE` | Read or change settings |

Add `--collection NAME` to work with more than one sorting scheme.

## Configuration

`~/.music-cluster/config.yaml`, or the Settings screen. Everything is
adjustable; the settings that matter most:

```yaml
sorting:
  auto_accept_confidence: 0.6   # how sure before a track can be auto-filed
  min_margin: 0.08              # how far ahead of the runner-up
  novelty_factor: 3.0           # beyond this, report "matches nothing"
  neighbors: 5                  # references compared against per group
  knn_weight: 0.7               # 1.0 = nearest tracks only, 0 = centroid only
  discriminant_weight: 0.7      # focus on your boundaries vs general character
  feature_weights:              # what "similar" means to you
    timbre: 1.0
    rhythm: 1.0                 # raise if BPM is what your folders follow
    harmony: 1.0
    dynamics: 1.0
  learn_on_commit: true

organize:
  mode: playlist                # default filing mode
  on_conflict: skip

feature_extraction:
  excerpt_seconds: 90           # analyse the middle 90s; 0 for the whole file
```

Changing anything under `sorting` or `feature_extraction` means re-running
`fit` (and re-analysing, for the latter).

### Optional: LLM naming

Discovery can ask an LLM to name candidate groups. It is off by default, and
only track metadata — artist, title, genre tag, tempo — is ever sent; never
audio.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
music-cluster config --set labeling.llm_enabled=true
music-cluster discover ~/Music/Unsorted --llm
```

Without it, names come from genre tags, tempo, and measured audio character.

## Installation

Download an installer for macOS, Windows or Linux from the
[releases page](https://github.com/julesyoungberg/music-cluster/releases) — those
bundle the analysis server, so only FFmpeg is needed alongside them.

From source, Python 3.10+ and FFmpeg. See [INSTALL.md](INSTALL.md) for details.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Performance

- Analysis: roughly a second per track with the default 90-second excerpt,
  across all CPU cores. A 10,000-track library takes well under an hour.
- Fitting: seconds.
- Sorting an already-analysed track: instant. The cost is analysis, once.
- Storage: about 1 KB per track.

Formats: anything FFmpeg decodes — MP3, FLAC, WAV, AIFF, M4A/ALAC, OGG, Opus,
AAC, WMA, APE, WavPack. See [FORMATS.md](FORMATS.md).

## Development

```bash
pip install -r requirements-dev.txt

ruff format . && ruff check --fix . && mypy   # format, lint, type check
pytest tests/unit                             # fast, no audio decoding
pytest tests/e2e tests/integration            # the whole app, real audio

cd ui
npm install
npm run format:check && npm run lint && npm run check
npm test                                      # vitest
npx playwright test                           # the UI against a real API
```

CI runs exactly these, plus a packaging job that produces installers for macOS,
Windows and Linux. See [INSTALL.md](INSTALL.md#continuous-integration).

## License

MIT
