# Command-line reference

Everything the desktop app does, the `cratewise` command does too. This
page lists every command and every option, with an example of each. For a
guided walk through a first sort, read the [tutorial](tutorial.md) instead.

```bash
cratewise --help              # list the commands
cratewise sort --help         # options for one command
cratewise --version
```

**Contents**

- [Conventions](#conventions)
- [Setting up](#setting-up): [`init`](#init), [`info`](#info), [`config`](#config)
- [The library](#the-library): [`analyze`](#analyze), [`search`](#search), [`similar`](#similar), [`prune`](#prune)
- [Collections](#collections): [`collection create`](#collection-create), [`collection list`](#collection-list), [`collection delete`](#collection-delete)
- [Groups](#groups): [`groups import-tree`](#groups-import-tree), [`groups add`](#groups-add), [`groups list`](#groups-list), [`groups show`](#groups-show), [`groups rename`](#groups-rename), [`groups set-destination`](#groups-set-destination), [`groups file`](#groups-file), [`groups remove`](#groups-remove), [`groups export`](#groups-export)
- [Learning the groups](#learning-the-groups): [`fit`](#fit), [`check`](#check)
- [Sorting](#sorting): [`sort`](#sort), [`review`](#review), [`sessions`](#sessions)
- [Filing](#filing): [`apply`](#apply), [`undo`](#undo)
- [Discovery](#discovery): [`discover`](#discover), [`candidates`](#candidates)
- [Auto-tags](#auto-tags): [`tags run`](#tags-run), [`tags find`](#tags-find), [`tags show`](#tags-show), [`tags summary`](#tags-summary), [`tags vocabulary`](#tags-vocabulary)
- [Samples](#samples): [`samples classify`](#samples-classify), [`samples categories`](#samples-categories)
- [Evaluation](#evaluation): [`eval run`](#eval-run), [`eval download`](#eval-download), [`eval compare`](#eval-compare)
- [Environment variables](#environment-variables)

## Conventions

**Collections.** A collection is one sorting scheme: a set of groups that new
music is filed into. Commands that work on a collection take
`--collection NAME`. The value can be a name or a numeric ID (as shown by
`collection list`). Left out, the command uses the collection that was changed
most recently. `groups import-tree` and `groups add` are the exception: they
default to the collection called `My Folders`, and create it if it does not
exist.

**Profiles.** `--profile music` (the default) is for full-length tracks;
`--profile sample` is for one-shots and loops (kicks, snares, hats, basses,
chords). A collection's profile is fixed when it is created. See
[Samples](../README.md#samples) in the README.

**Workers.** `--workers N` sets how many files are analysed in parallel. It
defaults to `performance.num_workers` in the config, which is `-1`, meaning
every CPU core. Lower it if a large library runs out of memory.

**Recursion.** Commands that scan folders (`analyze`, `sort`,
`groups import-tree`) look inside subfolders unless you pass `--no-recursive`.

**Nothing is written without asking.** Analysing, sorting, reviewing and
discovering only change the database in `~/.music-cluster/`. Your audio files
are only touched by `apply` (after a preview and a confirmation) and by `undo`.

---

## Setting up

### `init`

Create the database, the config file and a first collection.

```bash
cratewise init
cratewise init --profile sample             # first collection holds samples
cratewise init --db-path ~/DJ/library.db    # keep the database somewhere else
```

| Option | Meaning |
| --- | --- |
| `--db-path PATH` | Where to put the database. Saved to the config as `database.path`. Default `~/.music-cluster/library.db`. |
| `--profile music\|sample` | Profile of the first collection. `music` creates `My Folders`; `sample` creates `My Samples`. |

Running `init` again is safe: existing data is kept.

### `info`

Show the database path, how many tracks are in the library (and how many were
analysed as music or as samples), and each collection with its group count,
reference track count and self-check accuracy.

```bash
cratewise info
```

```
Database: /Users/you/.music-cluster/library.db
Tracks:   3946 (3104 as music, 842 as sample)

Collections:
  My Folders [music]: 8 groups, 3104 reference tracks (fitted, 81% self-check)
  Drum library [sample]: 11 groups, 842 reference tracks (not fitted)
```

### `config`

Print the configuration, or change one or more settings.

```bash
cratewise config                                   # print as YAML
cratewise config --json                            # print as JSON
cratewise config --set organize.mode=copy
cratewise config --set sorting.neighbors=8 --set sorting.min_margin=0.1
cratewise config --set labeling.llm_enabled=true
```

| Option | Meaning |
| --- | --- |
| `--set KEY=VALUE` | Set a dotted key. Repeatable. `true`/`yes`/`on` and `false`/`no`/`off` become booleans, `none`/`null` or an empty value clears the key, and numbers become numbers. |
| `--json` | Print as JSON instead of YAML. |

The file lives at `~/.music-cluster/config.yaml`. The settings that matter most
are listed in the README under [Configuration](../README.md#configuration).
After changing anything under `sorting`, run [`fit`](#fit) again; after
changing anything under `feature_extraction`, re-analyse
(`analyze --update`) and then fit.

---

## The library

The library is every audio file the tool has analysed. Analysis happens once
per file; everything afterwards (sorting, fitting, searching) reads the stored
result.

### `analyze`

Extract audio features (and, for music, auto-tags) without sorting anything.
`groups import-tree`, `sort` and `discover` analyse what they need on their
own, so you only need this to prepare a large library ahead of time.

```bash
cratewise analyze ~/Music/DJ
cratewise analyze ~/Music/DJ ~/Music/Promos --workers 4
cratewise analyze ~/Samples --profile sample
cratewise analyze ~/Music/DJ --update                # re-analyse known files
cratewise analyze ~/Music/DJ --extensions mp3,flac   # only these formats
```

| Option | Meaning |
| --- | --- |
| `PATHS...` | One or more files or folders. |
| `-r` / `--no-recursive` | Scan subfolders (default) or only the top level. |
| `-u`, `--update` | Re-analyse files already in the library. Without it they are skipped. |
| `--extensions LIST` | Comma-separated extensions to include, e.g. `mp3,flac,wav`. Default: `mp3`, `m4a`, `aac`, `ogg`, `opus`, `wma`, `flac`, `wav`, `aiff`, `aif`, `ape`, `alac`, `wv` (see [FORMATS.md](../FORMATS.md)). |
| `--workers N` | Parallel workers; `-1` uses every core. |
| `--profile music\|sample` | How to analyse: a 90-second excerpt from the middle of each track (`music`), or each file from its first sample (up to 30 seconds) with envelope and pitch measurements (`sample`). |

Prints how many files were analysed, skipped (already known) and failed, and
the first few errors.

### `search`

Find tracks by artist, title, album or genre, and see which groups each one
belongs to.

```bash
cratewise search "burial"
cratewise search "deep house" --limit 50
```

| Option | Meaning |
| --- | --- |
| `QUERY` | Text to match. |
| `--limit N` | Show at most N tracks. Default 20. |

### `similar`

Find tracks that sound like one you name. The first library match for `QUERY`
is the seed. Lower distances are closer.

```bash
cratewise similar "Archangel"
cratewise similar "BD_808_01" --profile sample --limit 40
```

| Option | Meaning |
| --- | --- |
| `QUERY` | Text that identifies the seed track. |
| `--limit N` | How many similar tracks to list. Default 20. |
| `--profile music\|sample` | Compare as music or as samples. |

Useful for building a new group by hand: find one track that belongs, list the
tracks that sound like it, and add the right ones with
`groups add NAME --track ...`.

### `prune`

Remove library entries whose files no longer exist on disk (because you
deleted or moved them outside the tool). It never deletes audio files.

```bash
cratewise prune
```

---

## Collections

### `collection create`

```bash
cratewise collection create "DJ folders"
cratewise collection create "Drum library" --profile sample --description "One-shots"
```

| Option | Meaning |
| --- | --- |
| `NAME` | Collection name. Must be unique. |
| `--description TEXT` | Optional note. |
| `--profile music\|sample` | What the collection holds. Fixed once created: to sort both tracks and samples, make two collections. |

### `collection list`

```bash
cratewise collection list
```

```
   1  My Folders                   music     8 groups    3104 tracks
   2  Drum library                 sample   11 groups     842 tracks
```

The first column is the ID, which any `--collection` option accepts.

### `collection delete`

Delete a collection and all its groups. Tracks stay in the library and on disk.

```bash
cratewise collection delete "Old scheme"
cratewise collection delete "Old scheme" --yes    # no confirmation prompt
```

---

## Groups

A group is one of your folders (or playlists): its tracks are the references
that define what belongs in it, and its destination folder is where new music
sorted into it is filed.

### `groups import-tree`

Turn every subfolder of a parent folder into a group, with the subfolder as the
group's destination. This is the fastest way in when your music is already
organised into genre folders.

```bash
cratewise groups import-tree ~/Music/DJ
cratewise groups import-tree ~/Music/DJ --min-tracks 10
cratewise groups import-tree ~/Samples/Drums --collection "Drum library" --profile sample
```

| Option | Meaning |
| --- | --- |
| `PARENT` | The folder that contains your genre folders. |
| `--collection NAME` | Collection to import into. Default `My Folders`; created if missing. |
| `--min-tracks N` | Skip subfolders with fewer than N audio files. Default 1. |
| `-r` / `--no-recursive` | Include files in nested folders inside each subfolder (default), or only files directly inside it. |
| `--workers N` | Parallel workers for analysis. |
| `--profile music\|sample` | Profile for a collection this command creates. |

Every track in those folders is analysed (about a second per track the first
time). Ends by listing each group and its reference count. Next step:
[`fit`](#fit).

### `groups add`

Create one group, from a folder, a playlist, or individual seed tracks, or
create it empty.

```bash
cratewise groups add "Deep House" --folder ~/Music/DJ/Deep
cratewise groups add "Warm-up" --playlist ~/Playlists/warmup.m3u8 --destination ~/Music/DJ/Warmup
cratewise groups add "Rollers" --track a.flac --track b.mp3 --track c.wav --destination ~/Music/DJ/Rollers
cratewise groups add "Someday"                     # empty group
```

| Option | Meaning |
| --- | --- |
| `NAME` | Group name. |
| `--collection NAME` | Default `My Folders`; created if missing. |
| `--folder DIR` | Use every audio file in this folder as references. Unless `--destination` is given, the folder becomes the destination. |
| `--playlist FILE` | Use the tracks listed in an M3U/M3U8 playlist as references. |
| `--track FILE` | Add one seed track. Repeat for more. |
| `--destination DIR` | Folder where tracks sorted into this group are filed. |
| `--description TEXT` | Optional note (used with `--track` or for an empty group). |
| `--workers N` | Parallel workers for analysis. |
| `--profile music\|sample` | Profile for a collection this command creates. |

Use only one of `--folder`, `--playlist` and `--track`; if you pass more than
one, only the first of them in that order is used. A group without a
destination can still be filed as a playlist, but not copied, moved or
symlinked.

### `groups list`

```bash
cratewise groups list
cratewise groups list --collection "Drum library"
```

```
My Folders:

  ID  Group                           Tracks  Self-check  Destination
   1  Deep House                         412         92%  /Users/you/Music/DJ/Deep House
   2  Tech House                         288         70%  /Users/you/Music/DJ/Tech House
   3  Minimal                            156         60%  (none)
```

Self-check is filled in after [`fit`](#fit).

### `groups show`

List a group's destination and reference tracks.

```bash
cratewise groups show "Deep House"
cratewise groups show "Deep House" --limit 100
```

| Option | Meaning |
| --- | --- |
| `NAME` | Group name or ID. |
| `--collection NAME` | Collection to look in. |
| `--limit N` | Show at most N tracks. Default 25. |

### `groups rename`

```bash
cratewise groups rename "Deep" "Deep House"
cratewise groups rename "Deep" "Deep House" --collection "DJ folders"
```

Like every group command, it takes `--collection NAME`.

### `groups set-destination`

Set the folder new music sorted into this group is filed into. The folder does
not have to exist yet; it is created when something is filed there.

```bash
cratewise groups set-destination "Minimal" ~/Music/DJ/Minimal
cratewise groups set-destination "Kicks" ~/Samples/Kicks --collection "Drum library"
```

### `groups file`

Gather a group's existing tracks into its destination folder. Use it for groups
whose tracks live elsewhere, such as ones made by `discover` or imported from a
playlist. Tracks already anywhere inside the folder are left alone. It shows
the plan first and asks before touching anything; the run is recorded as a
session named "Filing <group>", so `undo` puts everything back.

```bash
cratewise groups file "Minimal" --mode copy --dry-run   # show the plan and stop
cratewise groups file "Minimal" --mode move             # confirm, then do it
cratewise undo 12                                       # the session ID it printed
```

| Option | Meaning |
| --- | --- |
| `--mode copy\|move\|symlink` | Required. `copy` and `symlink` leave the originals where they are; `move` relocates them and the library follows. |
| `--on-conflict skip\|rename` | What to do when the folder already has a file with that name. Default from config (`skip`). Overwriting is not offered: undo could not bring the replaced file back. |
| `--dry-run` | Show the plan and stop. |
| `--yes` | Skip the confirmation prompt. |
| `--collection NAME` | Collection the group is in. |

### `groups remove`

Delete a group from the collection. Its tracks stay in the library and on disk.

```bash
cratewise groups remove "Minimal"
cratewise groups remove "Minimal" --yes    # no confirmation prompt
cratewise groups remove "Minimal" --collection "DJ folders"
```

### `groups export`

Export every group as a playlist for DJ software. Only writes new files; it
refuses to replace a file of the same name that cratewise did not write.

```bash
cratewise groups export
cratewise groups export --output ~/Playlists --format rekordbox
cratewise groups export --output ~/Playlists --format m3u --relative-paths
```

| Option | Meaning |
| --- | --- |
| `--collection NAME` | Collection to export. |
| `--output DIR` | Where to write the files. Default `./playlists`. |
| `--format m3u8\|m3u\|rekordbox\|traktor` | One M3U per group, one Rekordbox XML, or one Traktor NML. Default `m3u8`. |
| `--relative-paths` | Write paths relative to the playlist instead of absolute paths. |

---

## Learning the groups

### `fit`

Learn what each group in a collection sounds like. Needs at least two groups
with reference tracks. Takes seconds.

```bash
cratewise fit
cratewise fit --collection "Drum library"
```

```
Fitting 'My Folders'...
Fitted 8 groups / 3104 reference tracks (lda space, 21 dims)
Self-check accuracy: 81.2%

Run `cratewise check` for a per-group breakdown.
```

Groups that had to be left out (no reference tracks, or tracks not analysed)
are listed as yellow warnings under the accuracy. Run it again whenever you add or remove reference tracks, or change a
`sorting` setting.

### `check`

Report how well your groups can be told apart. Every reference track is
re-sorted with itself hidden from its own group; the report shows how many came
back to where you filed them, and which pairs of groups get mixed up.

```bash
cratewise check
cratewise check --refit            # fit first, then report
```

| Option | Meaning |
| --- | --- |
| `--collection NAME` | Collection to check. |
| `--refit` | Fit before reporting. |

```
Group                             Tracks  Correct  Accuracy
Minimal                              156       94       60%
Tech House                           288      201       70%
Deep House                           412      381       92%

Most easily confused:
    52 tracks from 'Minimal' look like 'Tech House' (33%)
```

Heavy overlap between two groups usually means they really do sound alike:
either merge them, or add reference tracks that show what separates them.

---

## Sorting

### `sort`

Analyse a folder of new music, score every track against the collection's
groups, and save the suggestions as a numbered **session**. By default the
interactive review starts straight away.

```bash
cratewise sort ~/Downloads/NewMusic
cratewise sort ~/Downloads/NewMusic --name "November buys" --auto-accept
cratewise sort ~/Downloads/Pack --collection "Drum library" --no-review
```

| Option | Meaning |
| --- | --- |
| `PATHS...` | Files or folders of new music. |
| `--collection NAME` | Collection to sort against. If it has not been fitted, or its groups changed since, it is fitted first. |
| `--name TEXT` | Session name. Default: the folder name. |
| `-r` / `--no-recursive` | Scan subfolders (default) or only the top level. |
| `--auto-accept` | Accept every suggestion that clears `sorting.auto_accept_confidence` and `sorting.min_margin`, so you only review the uncertain ones. Accepted tracks still go through `apply` before anything is written. |
| `--workers N` | Parallel workers for analysis. |
| `--review` / `--no-review` | Start the interactive review when scoring finishes (default), or just print the session number. |

```
Session 4 (NewMusic):
  11 to review
  3 auto-accepted
  1 matched nothing in this collection
```

A track that is far from every group is marked **unmatched** rather than forced
into the least-bad one.

### `review`

Step through a session's pending tracks one at a time. Each shows its
duration, BPM, key and genre tag, then up to five suggested groups with a
confidence.

```bash
cratewise review 4
```

```
────────────────────────────────────────────────────────────────────────
Artist - Track Title   [11 left]
6:12  124 BPM  8A  Deep House

  1.   82%  Deep House
  2.   41%  Tech House
  3.   12%  Minimal

  [1-5] file it   [s] skip   [l] list all groups   [q] quit
  >
```

| Key | Action |
| --- | --- |
| `1`–`5` (or Enter for `1`) | File into that suggestion. |
| `s` | Skip this track (it is not filed). |
| `l` | List every group with its ID, then type an ID to file into any group. |
| `q` | Stop. Decisions so far are saved; run `review` again to continue. |

Decisions are only recorded in the database. Nothing is written to disk until
[`apply`](#apply).

### `sessions`

List sort sessions with their status and how many tracks are still pending.

```bash
cratewise sessions
cratewise sessions --collection "Drum library"
```

```
   4  November buys                open          15 tracks,   11 pending
   3  October buys                 committed     22 tracks,    0 pending
```

A session is `committed` once it has been applied with no tracks left to
review; until then it is `open`.

---

## Filing

### `apply`

Write a session's accepted decisions to disk. It always prints the plan first.
With `--dry-run` it stops there; otherwise it asks for confirmation.

```bash
cratewise apply 4 --dry-run                 # show the plan, change nothing
cratewise apply 4 --mode playlist           # one playlist per group
cratewise apply 4 --mode copy
cratewise apply 4 --mode move --on-conflict rename
cratewise apply 4 --mode symlink --yes
```

| Option | Meaning |
| --- | --- |
| `SESSION_ID` | The session number from `sort` or `sessions`. |
| `--mode MODE` | `playlist`, `copy`, `move` or `symlink`. Default: `organize.mode` in the config, which is `playlist`. |
| `--playlist-dir DIR` | Where playlist mode writes. Default `organize.playlist_dir` (`~/.music-cluster/playlists`). |
| `--on-conflict POLICY` | When a file of the same name is already in the destination: `skip` (default), `rename` (add a number), or `overwrite`. **`overwrite` cannot be undone**: the replaced file is not kept. |
| `--dry-run` | Print the plan and stop. Nothing is changed. |
| `--yes` | Do not ask for confirmation. |
| `--learn` / `--no-learn` | Add the filed tracks to their groups as new references. Default: `sorting.learn_on_commit` (on). Run `fit` afterwards to fold them in. |

| Mode | What it does |
| --- | --- |
| `playlist` | Writes one M3U8 per group. Nothing on disk moves. |
| `copy` | Copies each track into its group's destination folder. |
| `move` | Moves each track into the destination and updates the library to point at the new path. |
| `symlink` | Creates a link in the destination; the original stays where it is. |

Copy, move and symlink need the group to have a destination folder; tracks in
groups without one are listed as skipped in the plan.

```
Mode: move

  move     01 Track.flac -> /Users/you/Music/DJ/Deep House/01 Track.flac
  move     02 Other.mp3 -> /Users/you/Music/DJ/Tech House/02 Other.mp3
  skipped: group has no destination folder (Minimal)

Proceed to move 2 item(s)? [y/N]:
```

### `undo`

Reverse every file operation a session performed.

Undo does not restore a file that `--on-conflict overwrite` replaced, and it
does not take back the reference tracks that `--learn` added to groups (remove
those with the group screen in the app, or re-import the group). The session's
decisions stay applied, so `apply` will not run them a second time.

```bash
cratewise undo 4
cratewise undo 4 --remove-playlists --yes
```

| Option | Meaning |
| --- | --- |
| `SESSION_ID` | The session to reverse. |
| `--remove-playlists` | Also delete the playlists the session wrote. Without it, playlists are left in place. |
| `--yes` | Do not ask for confirmation. |

What undo does for each mode:

| Mode | Undo |
| --- | --- |
| `move` | Moves each file back to where it was and points the library at it again. |
| `copy` | Deletes the copies. The originals were never touched. |
| `symlink` | Deletes the links. |
| `playlist` | Nothing, unless `--remove-playlists` is given. |

---

## Discovery

For a pile of music that has no folders yet. Discovery clusters the pile into
**candidate** groups; you audition each and decide which become real groups.

### `discover`

```bash
cratewise discover ~/Music/Unsorted
cratewise discover ~/Music/Unsorted --algorithm kmeans --groups 12 --name "Big pile"
cratewise discover ~/Music/Unsorted --algorithm kmeans --granularity fine
cratewise discover ~/Music/Unsorted --min-size 5
cratewise discover ~/Samples/Unsorted --profile sample
cratewise discover ~/Music/Unsorted --llm
```

| Option | Meaning |
| --- | --- |
| `PATHS...` | Files or folders to break down. |
| `--name TEXT` | Run name. Default: the folder name. |
| `--algorithm` | `hdbscan` (density based; finds its own number of piles; the default), `kmeans` or `hierarchical` (a fixed number of piles). |
| `--groups N` | With `kmeans` or `hierarchical`, aim for N piles. Left out, the count is estimated. |
| `--granularity` | With `kmeans` or `hierarchical`, when the count is estimated: `coarse` (half as many piles), `broad`, `normal` (default), `fine`, or `finest` (twice as many). |
| `--min-size N` | Smallest pile to report. Default `discovery.min_group_size`: 8 for music, 5 for samples. |
| `--llm` / `--no-llm` | Ask an LLM to name the piles. Default `labeling.llm_enabled` (off). Only metadata is sent, never audio; see [LLM naming](../README.md#optional-llm-naming). |
| `--workers N` | Parallel workers for analysis. |
| `--profile music\|sample` | `sample` names piles as Kicks, Closed Hats, Basses and so on instead of by genre and tempo. |

```
Run 3: 7 candidate group(s)
12 track(s) did not fit any candidate

  [ 0] Bright Drum & Bass 174 BPM            64 tracks  ~174 BPM
  [ 1] Noisy Techno 132 BPM                  51 tracks  ~132 BPM
  [ 2] Dark Deep House 122 BPM               48 tracks  ~122 BPM
  ...

Next: cratewise candidates 3
```

Nothing becomes a group until you promote it with `candidates`.

### `candidates`

Go through a run's pending candidates one by one. Each shows its size, tempo
and character (or, for samples, its category share, length, attack and pitch),
the genre tags inside it, and the file paths of its representative tracks so
you can audition them. Promoted groups go into the most recently changed
collection, which must have the same profile as the run.

```bash
cratewise candidates 3
```

| Key | Action |
| --- | --- |
| `y` | Make it a group with all its tracks as references. Asks for a name (default: the suggested one) and a destination folder (optional). |
| `s` | Make it a group using only the representative tracks as references. Use this when the pile is roughly right but not every track in it should define the group. |
| `n` (or Enter) | Discard it. |
| `q` | Stop. |

Afterwards, run [`fit`](#fit).

---

## Auto-tags

Genre, beat type, mood, energy (1–10), danceability (0–1) and key, measured
from the audio by readable rules. Music is tagged while it is analysed. See
[Auto-tagging](../README.md#auto-tagging) for how each tag is measured.

### `tags run`

Tag music that has no tags yet.

```bash
cratewise tags run                    # every analysed track without tags
cratewise tags run ~/Music/New        # analyse (and tag) new files, tag older ones
cratewise tags run --update           # re-tag everything
```

| Option | Meaning |
| --- | --- |
| `PATHS...` | Optional. Limit to these files or folders; files not yet in the library are analysed as music first. |
| `-u`, `--update` | Re-tag tracks that already have tags. |
| `--workers N` | Parallel workers. |

### `tags find`

List tracks that match every filter you give.

```bash
cratewise tags find --beat half_time --min-energy 7
cratewise tags find --genre techno --mood hypnotic --camelot 8A
cratewise tags find --beat two_step --swung --min-danceability 0.7 --limit 100
```

```
  Artist A - Roller                     Drum & Bass · Half-time · Dark · E8 · dance 0.81 · 172 BPM · 8A
  Artist B - Weight                     Dubstep · Half-time · Hypnotic · E7 · dance 0.77 · 140 BPM · 5A
```

| Option | Values |
| --- | --- |
| `--genre` | `deep_house`, `house`, `tech_house`, `techno`, `hard_techno`, `trance`, `uk_garage`, `breaks`, `drum_and_bass`, `jungle`, `dubstep`, `trap`, `baile_funk`, `reggaeton`, `hip_hop`, `downtempo`, `ambient` |
| `--beat` | `four_on_floor`, `half_time`, `two_step`, `breakbeat`, `baile_funk`, `dembow`, `backbeat`, `beatless` |
| `--mood` | `dark`, `euphoric`, `melancholic`, `aggressive`, `chill`, `hypnotic`, `groovy` |
| `--min-energy N`, `--max-energy N` | Energy level, 1–10. |
| `--min-danceability X` | 0–1. |
| `--camelot KEY` | Camelot key, e.g. `8A`. |
| `--swung` / `--straight` | Only swung grooves, or only straight ones. |
| `--limit N` | Default 50. |

### `tags show`

Show the tags of tracks matching a search.

```bash
cratewise tags show "roller"
cratewise tags show "roller" --why
```

| Option | Meaning |
| --- | --- |
| `QUERY` | Text to match. |
| `--limit N` | Tracks to show. Default 5. |
| `--why` | Also print the top scores behind the beat, genre and mood tags, and the measured drum pattern (kick, snare and hat across one bar). |

### `tags summary`

How the library breaks down by genre, beat type and mood, and how many music
tracks are still untagged.

```bash
cratewise tags summary
```

### `tags vocabulary`

Every genre, beat type and mood the tagger can emit, with what each means.

```bash
cratewise tags vocabulary
```

---

## Samples

### `samples classify`

Analyse a folder as samples and print what each file appears to be (kick,
snare, clap, hat, bass, chord…), with a confidence and the evidence, then a
tally. Nothing is filed and no groups are created: it is a way to check the
tool hears your samples the way you do before you build a collection.

```bash
cratewise samples classify ~/Samples/Unsorted
cratewise samples classify ~/Samples/Unsorted --no-names --limit 200
```

| Option | Meaning |
| --- | --- |
| `PATHS...` | Files or folders of one-shots. |
| `--limit N` | List at most N files. Default 50. |
| `--names` / `--no-names` | Let filenames such as `BD_808_01.wav` inform the guess (default), or judge by the audio alone. |
| `--workers N` | Parallel workers. |

### `samples categories`

List the sample categories used to name discovered sample groups.

```bash
cratewise samples categories
```

---

## Evaluation

For measuring how well the tool works, not for everyday use. See
[EVALUATION.md](../EVALUATION.md).

### `eval run`

```bash
cratewise eval run                                          # synthetic suites
cratewise eval run --baseline eval/baselines/synthetic.json
cratewise eval run --suite key-giantsteps --suite tempo-giantsteps
cratewise eval run --all --folders ~/Music/DJ
```

| Option | Meaning |
| --- | --- |
| `--suite NAME` | Repeatable: `beats-synthetic`, `keys-synthetic`, `sorting-fma`, `sorting-fma-sub`, `key-giantsteps`, `tempo-giantsteps`. Default: the synthetic suites. |
| `--all` | Every suite whose data has been downloaded. |
| `--folders DIR` | Also simulate sorting into this folder tree (read only). |
| `--data-dir DIR` | Where datasets live. Default `~/.music-cluster-eval/data`. |
| `--trials N` | Simulated libraries per sorting suite. Default 50. |
| `--limit N` | Use at most N files per dataset. |
| `--quick` | Fewer synthetic variations. |
| `--seed N` | Random seed. Default 0. |
| `--workers N` | Default `-1` (every core). |
| `--out FILE` | Report path. Default `eval-report.json`. |
| `--baseline FILE` | Compare against an earlier report; exits non-zero on a regression. |
| `--tolerance X` | Allowed drop before a number counts as a regression. Default 0.02. |

### `eval download`

Fetch a public dataset into the eval data folder (never your library).

```bash
cratewise eval download giantsteps-key
```

Datasets: `fma-small`, `fma-medium`, `giantsteps-key`, `giantsteps-tempo`.
`--data-dir DIR` changes where they go.

### `eval compare`

Diff the headline numbers of two reports; exits 1 on a regression.

```bash
cratewise eval compare eval-report.json eval/baselines/synthetic.json --tolerance 0.02
```

---

## Environment variables

| Variable | Effect |
| --- | --- |
| `MUSIC_CLUSTER_DB` | Use this database file instead of `database.path`. Useful for keeping a separate library per setup. |
| `MUSIC_CLUSTER_CONFIG` | Use this config file instead of `~/.music-cluster/config.yaml`. |
| `ANTHROPIC_API_KEY` | Needed for LLM naming in discovery (with `pip install anthropic`). |
