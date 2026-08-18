# music-cluster

Sort new music into the folders you already keep.

You buy fifteen tracks, and now you have to decide which crate each one goes in.
music-cluster learns what your existing genre folders sound like and, for every
new track, tells you where it thinks it belongs and how sure it is. You listen,
you decide, and it files the results.

It does the same for samples. Point it at your kicks, snares, claps, hats,
basses and chords and it learns those folders too — see
[Samples](#samples).

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

## Samples

A kick is not a short track. It is 300 milliseconds long, it has no tempo, and
what makes it a kick rather than a clap is the shape of its envelope and where
its energy sits — none of which survives being averaged over ninety seconds the
way a track is analysed.

So a collection declares what it holds, once, when you make it:

```bash
music-cluster collection create "Drum library" --profile sample
music-cluster groups import-tree ~/Samples/Drums --collection "Drum library"
music-cluster fit --collection "Drum library"

music-cluster sort ~/Downloads/NewPack --collection "Drum library"
```

From there it is the same tool: the same review screen, the same confidence
bars, the same four filing modes, the same undo. What changes is underneath.

**Analysis starts at the first sample and never excerpts.** A one-shot's attack
is the most identifying thing about it and it is over 20 ms in. Analysis steps
four times more finely through time, so a 60 ms hi-hat still has a measurable
shape.

**Twenty-nine extra measurements, all about a single sonic event.** Attack,
decay, sustain and release; how long the sound actually lasts; energy across
eight bands from sub to air; whether there is a note and how many notes;
whether the sound is percussive or harmonic; whether it is one hit or a bar of
them. This is what separates a clap from a snare when their spectra are nearly
identical, and a bass from a chord when both are just "a low pitched thing".

**Tempo is not invented.** Beat tracking a 400 ms kick returns a number, and
that number is meaningless. Under this profile a single hit records no BPM
rather than a fabricated one.

### Discovering a pack that has no structure

Discovery over samples names its candidates in the vocabulary you would use:

```bash
music-cluster discover ~/Samples/Unsorted --profile sample
```

```
Run 3: 7 candidate group(s)

  [ 0] Kicks                                 214 tracks  92% Kick
  [ 1] Closed Hats                           186 tracks  88% Closed Hat
  [ 2] Snares                                140 tracks  71% Snare
  [ 3] Basses                                 96 tracks  84% Bass
  [ 4] Chords                                 61 tracks  77% Chord
```

The percentage is how much of the pile really looks like its name — a candidate
that is 45% anything is telling you the clustering found something you have not
got a word for yet. Names are still suggestions: nothing becomes a group until
you promote it.

Two things inform the guess. The filename, which in a sample library is usually
the truth (`BD_808_01.wav` is a kick, and nothing measured is going to know
better), and the audio, which carries the half of every pack named
`Sample 04.wav`. To see what it makes of a folder without changing anything:

```bash
music-cluster samples classify ~/Samples/Unsorted
music-cluster samples classify ~/Samples/Unsorted --no-names   # audio only
music-cluster samples categories
```

### Both in one library

Profiles are per collection, and a collection's profile is fixed when it is
created — its stored features, its fitted space and its group references all
assume one kind of audio. Sorting both means two collections, which share one
database and one library; a file analysed both ways keeps both vectors.

```bash
music-cluster collection list
   1  My Folders                   music     8 groups    3104 tracks
   2  Drum library                 sample   11 groups     842 tracks
```

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
   time. A sample collection measures 125 dimensions instead: the same
   ninety-six plus a block describing the single event — envelope, band
   balance, pitch. See [Samples](#samples).

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
| `samples classify PATH` | Report what a folder of one-shots looks like |
| `samples categories` | List the sample categories used for naming |
| `config --set KEY=VALUE` | Read or change settings |

Add `--collection NAME` to work with more than one sorting scheme, and
`--profile sample` when creating or analysing one that holds one-shots.

## Configuration

`~/.music-cluster/config.yaml`, or the Settings screen. Everything is
adjustable; the settings that matter most:

Settings under `feature_extraction`, `sorting` and `discovery` also take a
`profiles:` block, which is how a profile's own defaults get overridden:

```yaml
sorting:
  neighbors: 5
  profiles:
    sample:
      neighbors: 9              # only for sample collections
```

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
  profiles:
    sample:
      hop_size: 256             # analysis resolution for one-shots
      max_seconds: 30           # cap on how much of a long loop is read
```

Changing anything under `sorting` or `feature_extraction` means re-running
`fit` (and re-analysing, for the latter).

### Optional: LLM naming

Discovery can ask an LLM to name candidate groups. It is off by default, and
only metadata — artist, title, genre tag, tempo, or for samples the filenames
and measured shape — is ever sent; never audio.

```bash
pip install anthropic
export ANTHROPIC_API_KEY=...
music-cluster config --set labeling.llm_enabled=true
music-cluster discover ~/Music/Unsorted --llm
```

Without it, names come from genre tags, tempo, and measured audio character.

## Installation

Python 3.10+ and FFmpeg. See [INSTALL.md](INSTALL.md) for details.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## Performance

- Analysis: roughly a second per track with the default 90-second excerpt,
  across all CPU cores. A 10,000-track library takes well under an hour.
  One-shots are quicker — around a tenth of a second each for typical drum
  hits, longer for multi-second loops.
- Fitting: seconds.
- Sorting an already-analysed track: instant. The cost is analysis, once.
- Storage: about 1 KB per track, per profile.

Formats: anything FFmpeg decodes — MP3, FLAC, WAV, AIFF, M4A/ALAC, OGG, Opus,
AAC, WMA, APE, WavPack. See [FORMATS.md](FORMATS.md).

## Development

```bash
pip install -r requirements-dev.txt
pytest

cd ui && npm install && npm run check
```

## License

MIT
