# music-cluster

[![Lint](https://github.com/julesyoungberg/music-cluster/actions/workflows/lint.yml/badge.svg)](https://github.com/julesyoungberg/music-cluster/actions/workflows/lint.yml)
[![Unit tests](https://github.com/julesyoungberg/music-cluster/actions/workflows/test-unit.yml/badge.svg)](https://github.com/julesyoungberg/music-cluster/actions/workflows/test-unit.yml)
[![End-to-end tests](https://github.com/julesyoungberg/music-cluster/actions/workflows/test-e2e.yml/badge.svg)](https://github.com/julesyoungberg/music-cluster/actions/workflows/test-e2e.yml)
[![Build desktop app](https://github.com/julesyoungberg/music-cluster/actions/workflows/build.yml/badge.svg)](https://github.com/julesyoungberg/music-cluster/actions/workflows/build.yml)

Sort new music, and samples, into the folders you already keep.

You buy fifteen tracks and have to decide which crate each one goes in.
music-cluster learns what your existing folders sound like and, for every new
track, says where it thinks it belongs and how sure it is. You decide, and it
files the result.

It's one command, `cratewise`, in four steps:

1. **groups**: tell it which folders you sort into (`groups import-tree`), or
   let it propose piles from an unsorted folder (`discover`).
2. **fit**: learn what each group sounds like, and `check` that your groups can
   actually be told apart.
3. **sort**: score a folder of new music against your groups, then `review` the
   suggestions one track at a time in the terminal.
4. **file**: write your decisions to disk (`apply`, with a dry run and an
   `undo`), or export your groups as playlists for Rekordbox, Traktor or Serato.

Along the way it tags each track (genre, beat type, mood, energy, key), and
it sorts one-shots (kicks, snares, hats, basses, chords) as well as full tracks.
A desktop app runs the same steps with audio previews; see
[Desktop app](#desktop-app).

## Safety first

Your music library is the one thing this tool must not damage, so it is careful
by default.

- **You define the groups.** It never invents a category. Discovery only
  proposes piles; nothing becomes a group until you promote it.
- **Your files only change when you file or export.** Analysing, sorting,
  reviewing and discovering only write to the database in `~/.music-cluster/`.
  Files change only with `apply`, `groups file`, `groups export` and
  `groups serato`.
- **Filing shows its plan first.** `--dry-run` prints the plan and stops;
  without it, `apply` asks before doing anything.
- **Every filing mode has an undo.** `cratewise undo SESSION` reverses a
  session. A file that a later session filed over is left in place, and undo
  names that later session so you can undo it first.
- **No silent overwrites.** A file already at the destination is skipped by
  default (`--on-conflict rename` adds a number instead). With
  `--on-conflict overwrite`, the replaced file is moved to a `backups/` folder
  beside the database and `undo` puts it back.
- **Your Serato crates are left alone.** Only crates cratewise wrote are ever
  replaced or removed.
- **Tags never move a file** and never change a group.
- **Your audio never leaves your computer.** Optional LLM naming is off by
  default and sends only metadata.

One thing it cannot protect: if a DJ app already manages a track, moving the
file breaks the app's link to it. Use `playlist` or `copy` filing for those.

## Prerequisites

- **Python 3.10 to 3.12.** Newer versions may work, but librosa and numba,
  which the analysis uses, tend to support them late.
- **FFmpeg**, to decode audio: `brew install ffmpeg` on macOS,
  `sudo apt-get install ffmpeg` on Debian or Ubuntu, `winget install ffmpeg`
  on Windows.

## Install

```bash
pipx install --python python3.12 git+https://github.com/julesyoungberg/music-cluster
# or: uv tool install --python 3.12 git+https://github.com/julesyoungberg/music-cluster
cratewise --help
```

From a checkout, for development, see [Development](#development).
[INSTALL.md](INSTALL.md) covers the desktop app, optional extras and
troubleshooting.

## Configure

```bash
cratewise init              # creates ~/.music-cluster/library.db and config.yaml
cratewise config            # what's in effect
cratewise info              # what's in the library
```

Every setting has a default, and the [Configuration](#configuration) section
lists the ones worth changing. To keep a separate library per DJ setup, point
the `MUSIC_CLUSTER_DB` and `MUSIC_CLUSTER_CONFIG` environment variables at
another folder.

## Using it

The short version, for a folder of genre folders and a folder of new buys:

```bash
cratewise groups import-tree ~/Music/DJ    # each subfolder becomes a group
cratewise fit                              # learn what each one contains
cratewise check                            # can they be told apart?

cratewise sort ~/Downloads/NewMusic        # score, then review in the terminal
cratewise apply 1 --mode copy --dry-run    # see the plan
cratewise apply 1 --mode copy              # then write it
cratewise undo 1                           # if you change your mind
```

`check` re-sorts every track you already filed, with itself hidden, and reports
how many came back to the right group:

```
Group                             Tracks  Correct  Accuracy
Minimal                              156       94       60%
Tech House                           288      201       70%
Deep House                           412      381       92%

Most easily confused:
    52 tracks from 'Minimal' look like 'Tech House' (33%)
```

That says as much about your folders as about the tool: two folders that
overlap this much are either one folder, or need clearer examples of what
separates them.

`sort` analyses the new music and saves a numbered **session**:

```
Session 1 (NewMusic):
  11 to review
  3 auto-accepted
  1 matched nothing in this collection
```

A track far from every group is reported as matching nothing rather than
forced into the least-bad one. `review` then shows each track's length, BPM,
key and genre tag with up to five ranked groups and a confidence for each.
Press `1` to `5` to file it, `s` to skip, `l` to pick any group, `q` to stop
and come back later.

The [command-line reference](docs/cli.md) has every command and option, with
examples, including building a group from a playlist or a few seed tracks
(`groups add`), finding tracks that sound alike (`similar`), and keeping more
than one sorting scheme (`--collection NAME`).

### Starting from one unsorted pile

Discovery breaks a pile into candidate groups, each with a few representative
tracks to audition. Keep the ones that are real, name them, discard the rest,
then sort against them like any other groups.

```bash
cratewise discover ~/Music/Unsorted        # propose candidate groups
cratewise candidates 1                     # audition, name, keep or discard
cratewise fit
```

## Samples

A kick is not a short track. It is 300 milliseconds long and has no tempo, and
what makes it a kick rather than a clap is the shape of its envelope and where
its energy sits. None of that survives being averaged over ninety seconds the
way a track is.

So a collection declares what it holds, once, when you create it:

```bash
cratewise collection create "Drum library" --profile sample
cratewise groups import-tree ~/Samples/Drums --collection "Drum library"
cratewise fit --collection "Drum library"
cratewise sort ~/Downloads/NewPack --collection "Drum library"
```

From there it is the same tool, with the same review, filing modes and undo.
Underneath, a sample collection:

- **analyses from the first sample and never excerpts**, four times more finely
  through time, so a 60 ms hi-hat still has a measurable shape;
- **adds 29 measurements of the single sound**: attack, decay, sustain and
  release, how long it lasts, energy in eight bands from sub to air, pitch and
  how many notes, percussive or harmonic, one hit or a bar of them;
- **records no BPM for a single hit**, rather than the meaningless number a
  beat tracker returns for a 400 ms kick.

Discovery over samples names its candidates in a producer's words:

```bash
cratewise discover ~/Samples/Unsorted --profile sample
```

```
Run 3: 5 candidate group(s)

  [ 0] Kicks                                 214 tracks  92% Kick
  [ 1] Closed Hats                           186 tracks  88% Closed Hat
  [ 2] Snares                                140 tracks  71% Snare
  [ 3] Basses                                 96 tracks  84% Bass
  [ 4] Chords                                 61 tracks  77% Chord
```

The percentage is how much of the pile looks like its name. A candidate that is
45% anything has found something you don't have a word for yet. The guess uses
the filename (in a sample library `BD_808_01.wav` is a kick, and nothing
measured will know better) and the audio, for every pack named `Sample 04.wav`.

```bash
cratewise samples classify ~/Samples/Unsorted              # what it makes of a folder
cratewise samples classify ~/Samples/Unsorted --no-names   # audio only
cratewise samples categories
```

A collection's profile is fixed when it is created, because its stored
features assume one kind of audio. To sort both music and samples, make two
collections; they share one library.

## Auto-tagging

Groups say which of *your* folders a track belongs in. Tags say what it *is*,
in the words you would use with another DJ.

| Tag | What it is |
| --- | --- |
| **Beat type** | Four to the floor, half-time, 2-step, breaks, baile funk (tamborzão), dembow, backbeat or beatless, and whether it swings |
| **Genre** | House, deep house, tech house, techno, hard techno, trance, UK garage, breaks, drum & bass, jungle, dubstep, trap, baile funk, reggaeton, hip hop, downtempo, ambient |
| **Mood** | One or two of dark, euphoric, melancholic, aggressive, chill, hypnotic, groovy |
| **Energy** | 1 to 10 |
| **Danceability** | 0 to 1 |
| **Key** | With its Camelot code, for mixing in key |

Music is tagged while it is analysed. A library analysed before tagging
existed is tagged with `tags run`.

```bash
cratewise tags find --beat half_time --min-energy 7   # browse by tag
cratewise tags show "roller" --why                    # a track's tags, with the scores behind them
cratewise tags summary                                # the library by genre, beat and mood
cratewise tags vocabulary                             # every tag, and what it means
```

**How beat type is measured.** The spectrum is split into kick, snare and hat
bands; the beat is tracked; each band is sampled twelve times per beat and
folded into one averaged bar: the track's drum pattern, as you would draw it in
a sequencer. That separates a four-to-the-floor from a 2-step at the same
132 BPM (the 2-step has no kick on three). Beat trackers often count a 172 as
86, so a slow reading with busy sixteenths is re-read at double time. Swing is
how late the offbeat hats land.

**What it is not.** Every tag is a readable rule over measured quantities
(`music_cluster/tagging.py`), not a trained model, so each tag can be explained
and is wrong in explainable ways. Genre comes mostly from tempo and beat type,
so two genres that share both look alike; a genre tag already in the file tips
close calls.

## How the sorting works

Your folders are the training data. That is what makes this different from
clustering a library and hoping the clusters mean something.

1. **Features.** Every track gets a 96-number description: timbre (MFCCs and
   spectral shape), rhythm (tempo and onsets), harmony (chroma) and dynamics.
   Artist, title, genre, BPM and key tags are read at the same time. A sample
   collection adds the 29 measurements described under [Samples](#samples).
2. **A space tuned to your boundaries.** The feature space is fitted with LDA,
   which finds the directions that separate *your* groups, blended with PCA,
   which keeps the general shape of the audio. LDA alone would lose the ability
   to notice a track that resembles nothing you own.
3. **Fair comparison across group sizes.** A track's distance to a group blends
   the group's centre (stable for a small group) with its nearest tracks
   (handles a folder that holds several distinct sounds).
4. **A decision, with a confidence.** Scores give a confidence and a margin over
   the runner-up. Clear both thresholds and a track can be accepted
   automatically (`sort --auto-accept`); otherwise it waits for your review.
   Too far from everything and it matches nothing.
5. **Learning.** Tracks you file become references for their group, so the
   sorter follows your taste as it drifts. Turn it off with
   `sorting.learn_on_commit: false`.

## Filing

Four modes, each with a dry run first and an undo after:

| Mode | What it does |
| --- | --- |
| `playlist` | Writes one M3U per group. Nothing on disk moves. |
| `copy` | Copies into the group's destination folder. |
| `move` | Moves the file, and follows it in the library. |
| `symlink` | Links it into the folder; the original stays put. |

```bash
cratewise apply 3 --mode move --dry-run   # show the plan and stop
cratewise apply 3 --mode move             # confirm, then do it
cratewise undo 3                          # put everything back
```

A group whose tracks live elsewhere (one made by discovery, or from a playlist)
can be gathered into its folder the same way: give it a folder with
`cratewise groups set-destination`, then run
`cratewise groups file "Minimal" --mode copy`. Tracks already there are left
alone, and `undo` reverses it.

## Exporting to DJ software

Export every group as a playlist for your DJ app. This only writes a new file;
your music stays where it is. Run it again whenever your groups change.

| Format | For | Import it via |
| --- | --- | --- |
| `rekordbox` | Rekordbox | Preferences → Advanced → rekordbox xml, then the "rekordbox xml" tree in the sidebar |
| `traktor` | Traktor | Right-click Playlists → Import Playlist |
| `m3u8` / `m3u` | Engine DJ, VirtualDJ, others | Drag the playlists in, or the app's playlist import |

```bash
cratewise groups export --format rekordbox --output ~/Music/exports
cratewise groups serato --dry-run          # Serato: crates straight into _Serato_
```

Serato crates are written into Serato's own folder, under a "Cratewise" crate,
so they work like filing: `--dry-run` shows them first, a crate you made is
never replaced, and `--remove` deletes exactly the crates cratewise wrote.
Rekordbox and Traktor exports carry BPM, key and genre; tags already in your
files win, and measured values only fill the gaps.

## Configuration

`~/.music-cluster/config.yaml`, changed by hand, with
`cratewise config --set KEY=VALUE`, or in the desktop app's Settings. The
settings that matter most:

```yaml
sorting:
  auto_accept_confidence: 0.7   # how sure before a track can be auto-accepted
  min_margin: 0.08              # how far ahead of the runner-up
  novelty_factor: 3.0           # group-widths from every group before "matches nothing"
  neighbors: 5                  # nearest tracks compared against, per group
  knn_weight: 0.3               # 1.0 = nearest tracks only, 0 = group centre only
  discriminant_weight: 0.7      # focus on your boundaries vs general character
  feature_weights:              # what "similar" means to you
    timbre: 1.0
    rhythm: 1.0                 # raise if BPM is what your folders follow
    harmony: 1.0
    dynamics: 1.0
  learn_on_commit: true
  feature_set: hand             # or ast: see "Sorting with AST" below

organize:
  mode: playlist                # default filing mode
  on_conflict: skip

tagging:
  on_analyze: true              # tag music while analysing it

feature_extraction:
  excerpt_seconds: 90           # analyse the middle 90 s; 0 for the whole file
```

Settings under `feature_extraction`, `sorting` and `discovery` also take a
`profiles:` block, to change a value for sample collections only:

```yaml
sorting:
  profiles:
    sample:
      neighbors: 9
```

After changing anything under `sorting`, run `fit` again; after
`feature_extraction`, re-analyse (`analyze --update`) and then fit.

### Optional: sorting with AST

AST (the Audio Spectrogram Transformer, a network trained on AudioSet) can
describe each track in place of the 96 hand-made features. On FMA genres it
sorts 65% of tracks into the right folder first, against 45% for the hand
features (see [EVALUATION.md](EVALUATION.md)). It is off by default because it
needs PyTorch and Hugging Face `transformers` (several hundred MB), downloads
the model (about 350 MB) on first use, and takes a few seconds per track on
a laptop CPU.

```bash
pipx inject cratewise torch transformers   # or install with the extra: 'cratewise[ast] @ git+https://...'
cratewise config --set sorting.feature_set=ast
cratewise fit                               # embeds every reference track once, then stores it
```

From then on `analyze` embeds each new track right after measuring it, and
`sort` embeds any track that still has none. Music collections only; sample
collections keep the hand features. The desktop app does not bundle it yet.

### Optional: LLM naming

Discovery can ask an LLM to name candidate groups. It is off by default, and
only metadata is sent (artist, title, genre tag, tempo; for samples, filenames
and measured shape), never audio.

```bash
pipx inject cratewise anthropic   # or install with the extra: 'cratewise[llm] @ git+https://...'
export ANTHROPIC_API_KEY=...
cratewise config --set labeling.llm_enabled=true
cratewise discover ~/Music/Unsorted --llm
```

Without it, names come from genre tags, tempo and measured character.

## Desktop app

The same steps with audio previews and waveforms, for when you need to hear a
track to decide between deep house and tech house. Installers for macOS,
Windows and Linux are on the
[releases page](https://github.com/julesyoungberg/music-cluster/releases); they
bundle the analysis server, so only FFmpeg is needed alongside. From a checkout:

```bash
python start-dev.py          # API + UI at http://localhost:1420
python start-dev.py --tauri  # as a desktop window
```

The [tutorial](docs/tutorial.md) walks through a first sort in the app, screen
by screen. It shares the library and config with the command line.

## Performance

- **Analysis:** about a second per track with the default 90-second excerpt,
  across all CPU cores, so a 10,000-track library takes well under an hour.
  Tagging adds about 60%; `tagging.on_analyze: false` turns it off and
  `tags run` catches up later. One-shots take about a tenth of a second each.
- **Fitting:** seconds. **Sorting an analysed track:** instant.
- **Storage:** about 1 KB per track, per profile.

Formats: anything FFmpeg decodes, including MP3, FLAC, WAV, AIFF, M4A/ALAC, OGG,
Opus and AAC. See [FORMATS.md](FORMATS.md).

## Repository layout

```
music_cluster/   the package: cli.py, api.py (the app's server), sorting, tagging, filing
ui/              the desktop app's interface (SvelteKit)
src-tauri/       the desktop shell (Tauri)
tests/           pytest: unit tests on synthetic data, end-to-end on generated audio
eval/            committed eval baselines; see EVALUATION.md
docs/            the command-line reference and the app tutorial
scripts/         build, screenshot and landing-page scripts
site/            the landing page
```

## Development

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -r requirements-dev.txt && uv pip install -e .

ruff format . && ruff check --fix . && mypy   # format, lint, type check
pytest tests/unit                             # fast, no audio decoding
pytest tests/e2e tests/integration            # the whole tool, real audio

cd ui && npm ci
npm run format:check && npm run lint && npm run check
npm test                                      # vitest
npx playwright test                           # the UI against a real API
```

CI runs these, plus a job that builds installers for macOS, Windows and Linux.
See [INSTALL.md](INSTALL.md#continuous-integration).

### Evaluation

Tests say the code does what it was written to do; `cratewise eval` says how
well that works on music. It scores beat type, tempo and key on audio whose
answers are known, and sorting on simulated libraries built from public
datasets (FMA, GiantSteps), each against simple baselines, and compares the
numbers with a committed baseline:

```bash
cratewise eval run --baseline eval/baselines/synthetic.json
cratewise eval download giantsteps-key && cratewise eval run --suite key-giantsteps
```

See [EVALUATION.md](EVALUATION.md).

## License

MIT, copyright Light Lands. See [LICENSE](LICENSE).
