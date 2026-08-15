"""Command-line interface for music-cluster.

The workflow, in the order you would actually use it:

    music-cluster init
    music-cluster groups import-tree ~/Music/DJ      # your existing folders
    music-cluster fit                                # learn what they contain
    music-cluster check                              # are they distinguishable?
    music-cluster sort ~/Downloads/New               # score the new stuff
    music-cluster review <session>                   # you decide
    music-cluster apply <session> --mode copy        # write it to disk

For an unsorted pile with no structure yet, start with `discover` instead.
"""

import json
import os
import sys
from typing import Optional

import click
from tqdm import tqdm

from .config import Config
from .database import Database
from . import discovery as discovery_mod
from . import groups as groups_mod
from . import organizer, sorting
from .library import analyze_paths, prune_missing
from .metadata import display_name
from .utils import parse_extensions


DEFAULT_COLLECTION = "My Folders"


# ----------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------


def get_context():
    """Config and database, the two things nearly every command needs."""
    config = Config()
    return config, Database(config.get_db_path())


def progress_bar(desc: str):
    """Progress callback backed by tqdm, created lazily on first call."""
    state = {"bar": None}

    def callback(done: int, total: int, _filepath: str) -> None:
        if total == 0:
            return
        if state["bar"] is None:
            state["bar"] = tqdm(total=total, desc=desc, unit="track")
        state["bar"].n = done
        state["bar"].refresh()
        if done >= total:
            state["bar"].close()

    return callback


def resolve_collection_or_exit(db: Database, name: Optional[str]):
    try:
        return groups_mod.resolve_collection(db, name)
    except LookupError as exc:
        raise click.ClickException(str(exc))


def echo_error(message: str) -> None:
    click.echo(click.style(f"Error: {message}", fg="red"), err=True)


def format_confidence(value: Optional[float]) -> str:
    if value is None:
        return "   - "
    percent = value * 100
    color = "green" if percent >= 75 else "yellow" if percent >= 50 else "red"
    return click.style(f"{percent:4.0f}%", fg=color)


@click.group()
@click.version_option(version="2.0.0")
def cli():
    """Sort new music into the folders you already keep."""


# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------


@cli.command()
@click.option("--db-path", type=str, default=None, help="Database path")
def init(db_path):
    """Create the database and config file."""
    config = Config()
    if db_path:
        config.set(["database", "path"], db_path)
    config.save()

    path = config.get_db_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    db = Database(path)
    groups_mod.ensure_collection(db, DEFAULT_COLLECTION, "Default sorting scheme")

    click.echo(f"Database:   {path}")
    click.echo(f"Config:     {config.config_path}")
    click.echo(f"Collection: {DEFAULT_COLLECTION}")
    click.echo("\nNext: music-cluster groups import-tree ~/Music/YourDJFolders")


@cli.command()
def info():
    """Show what is in the library."""
    config, db = get_context()
    click.echo(f"Database: {config.get_db_path()}")
    click.echo(f"Tracks:   {db.count_tracks()} ({db.count_features()} analysed)")

    collections = db.list_collections()
    if not collections:
        click.echo("\nNo collections yet. Run: music-cluster collection create <name>")
        return

    click.echo("\nCollections:")
    for collection in collections:
        model = db.load_model(collection["id"])
        accuracy = model["metrics"].get("accuracy") if model else None
        fitted = (
            f"fitted, {accuracy * 100:.0f}% self-check" if accuracy is not None else "not fitted"
        )
        click.echo(
            f"  {collection['name']}: {collection['group_count']} groups, "
            f"{collection['track_count']} reference tracks ({fitted})"
        )


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-r/--no-recursive", "recursive", default=True, help="Scan subdirectories")
@click.option("-u", "--update", is_flag=True, help="Re-analyse tracks already in the library")
@click.option("--extensions", type=str, default=None, help="Comma-separated extensions")
@click.option("--workers", type=int, default=None, help="Parallel workers (-1 = all CPUs)")
def analyze(paths, recursive, update, extensions, workers):
    """Extract audio features for files, without sorting them."""
    config, db = get_context()
    result = analyze_paths(
        db,
        config,
        list(paths),
        recursive=recursive,
        update=update,
        extensions=parse_extensions(extensions) if extensions else None,
        workers=workers,
        progress=progress_bar("Analysing"),
    )
    click.echo(
        f"\nAnalysed {result.analyzed}, skipped {result.skipped} (already known), "
        f"failed {result.failed}."
    )
    for error in result.errors[:5]:
        click.echo(f"  {os.path.basename(error['filepath'])}: {error['error']}")


@cli.command()
def prune():
    """Remove library entries whose files no longer exist."""
    _, db = get_context()
    removed = prune_missing(db)
    click.echo(f"Removed {len(removed)} missing track(s).")


# ----------------------------------------------------------------------
# Collections
# ----------------------------------------------------------------------


@cli.group()
def collection():
    """Manage sorting schemes."""


@collection.command("create")
@click.argument("name")
@click.option("--description", type=str, default=None)
def collection_create(name, description):
    """Create a collection."""
    _, db = get_context()
    if db.get_collection(name=name):
        raise click.ClickException(f"A collection named {name!r} already exists")
    db.create_collection(name, description)
    click.echo(f"Created collection {name!r}.")


@collection.command("list")
def collection_list():
    """List collections."""
    _, db = get_context()
    collections = db.list_collections()
    if not collections:
        click.echo("No collections yet.")
        return
    for item in collections:
        click.echo(
            f"{item['id']:>4}  {item['name']:<30} {item['group_count']:>3} groups"
            f"  {item['track_count']:>6} tracks"
        )


@collection.command("delete")
@click.argument("name")
@click.confirmation_option(prompt="Delete this collection and all of its groups?")
def collection_delete(name):
    """Delete a collection (tracks stay in the library)."""
    _, db = get_context()
    target = db.get_collection(name=name)
    if not target:
        raise click.ClickException(f"No collection named {name!r}")
    db.delete_collection(target["id"])
    click.echo(f"Deleted {name!r}.")


# ----------------------------------------------------------------------
# Groups
# ----------------------------------------------------------------------


@cli.group()
def groups():
    """Manage the folders you sort into."""


@groups.command("import-tree")
@click.argument("parent", type=click.Path(exists=True, file_okay=False))
@click.option("--collection", "collection_name", default=None, help="Collection name")
@click.option("--min-tracks", type=int, default=1, help="Skip folders with fewer tracks")
@click.option("-r/--no-recursive", "recursive", default=True)
@click.option("--workers", type=int, default=None)
def groups_import_tree(parent, collection_name, min_tracks, recursive, workers):
    """Import every subfolder of PARENT as a group.

    The fast path when your library is already a folder of genre folders.
    """
    config, db = get_context()
    target = groups_mod.ensure_collection(db, collection_name or DEFAULT_COLLECTION)

    reports = groups_mod.import_folders_as_groups(
        db,
        target["id"],
        parent,
        config,
        recursive=recursive,
        min_tracks=min_tracks,
        workers=workers,
        progress=progress_bar("Analysing"),
    )
    if not reports:
        raise click.ClickException(f"No subfolders with at least {min_tracks} track(s) in {parent}")

    click.echo("")
    for report in reports:
        click.echo(f"  {report.group_name:<32} {report.added:>5} reference tracks")
    click.echo(f"\nImported {len(reports)} group(s) into {target['name']!r}.")
    click.echo("Next: music-cluster fit")


@groups.command("add")
@click.argument("name")
@click.option("--collection", "collection_name", default=None)
@click.option("--folder", type=click.Path(exists=True, file_okay=False), help="Import a folder")
@click.option("--playlist", type=click.Path(exists=True, dir_okay=False), help="Import a playlist")
@click.option("--track", "tracks", multiple=True, type=click.Path(exists=True, dir_okay=False),
              help="Add individual seed tracks (repeatable)")
@click.option("--destination", type=click.Path(file_okay=False), default=None,
              help="Folder new music assigned to this group should go to")
@click.option("--description", type=str, default=None)
@click.option("--workers", type=int, default=None)
def groups_add(name, collection_name, folder, playlist, tracks, destination, description, workers):
    """Create a group from a folder, a playlist, or a few seed tracks."""
    config, db = get_context()
    target = groups_mod.ensure_collection(db, collection_name or DEFAULT_COLLECTION)
    callback = progress_bar("Analysing")

    try:
        if folder:
            report = groups_mod.import_folder_as_group(
                db, target["id"], folder, config, name=name,
                set_destination=destination is None, workers=workers, progress=callback,
            )
            if destination:
                db.update_group(report.group_id, destination_path=os.path.abspath(destination))
        elif playlist:
            report = groups_mod.import_playlist_as_group(
                db, target["id"], playlist, config, name=name,
                destination_path=destination, workers=workers, progress=callback,
            )
        elif tracks:
            group = groups_mod.create_group(
                db, target["id"], name, description=description,
                destination_path=destination, source_kind="manual",
            )
            report = groups_mod.add_tracks_to_group(
                db, group, [os.path.abspath(t) for t in tracks], config,
                role="seed", workers=workers, progress=callback,
            )
        else:
            group = groups_mod.create_group(
                db, target["id"], name, description=description, destination_path=destination
            )
            click.echo(f"Created empty group {name!r}. Add tracks with --track/--folder/--playlist.")
            return
    except ValueError as exc:
        raise click.ClickException(str(exc))

    click.echo(
        f"\n{report.group_name}: {report.added} reference track(s) added"
        + (f", {report.already_present} already present" if report.already_present else "")
    )


@groups.command("list")
@click.option("--collection", "collection_name", default=None)
def groups_list(collection_name):
    """List the groups in a collection."""
    _, db = get_context()
    target = resolve_collection_or_exit(db, collection_name)
    rows = db.list_groups(target["id"])
    if not rows:
        click.echo(f"No groups in {target['name']!r} yet.")
        return

    model = db.load_model(target["id"])
    accuracy = {
        entry["group_id"]: entry["accuracy"]
        for entry in (model["metrics"].get("per_group", []) if model else [])
    }

    click.echo(f"{target['name']}:\n")
    click.echo(f"{'ID':>4}  {'Group':<30} {'Tracks':>7}  {'Self-check':>10}  Destination")
    for group in rows:
        score = accuracy.get(group["id"])
        score_text = f"{score * 100:.0f}%" if score is not None else "-"
        destination = group["destination_path"] or click.style("(none)", fg="yellow")
        click.echo(
            f"{group['id']:>4}  {group['name']:<30} {group['size']:>7}  {score_text:>10}  {destination}"
        )


@groups.command("show")
@click.argument("name")
@click.option("--collection", "collection_name", default=None)
@click.option("--limit", type=int, default=25)
def groups_show(name, collection_name, limit):
    """Show a group's reference tracks."""
    _, db = get_context()
    target = resolve_collection_or_exit(db, collection_name)
    try:
        group = groups_mod.resolve_group(db, target["id"], name)
    except LookupError as exc:
        raise click.ClickException(str(exc))

    members = db.get_group_members(group["id"], limit=limit)
    total = len(db.get_group_member_ids(group["id"]))
    click.echo(f"{group['name']} — {total} reference track(s)")
    click.echo(f"Destination: {group['destination_path'] or '(none)'}\n")
    for track in members:
        click.echo(f"  {display_name(track)}")
    if total > limit:
        click.echo(f"  ... and {total - limit} more")


@groups.command("rename")
@click.argument("name")
@click.argument("new_name")
@click.option("--collection", "collection_name", default=None)
def groups_rename(name, new_name, collection_name):
    """Rename a group."""
    _, db = get_context()
    target = resolve_collection_or_exit(db, collection_name)
    group = groups_mod.resolve_group(db, target["id"], name)
    db.update_group(group["id"], name=new_name)
    click.echo(f"Renamed {group['name']!r} to {new_name!r}.")


@groups.command("set-destination")
@click.argument("name")
@click.argument("destination", type=click.Path(file_okay=False))
@click.option("--collection", "collection_name", default=None)
def groups_set_destination(name, destination, collection_name):
    """Set where music sorted into a group should be filed."""
    _, db = get_context()
    target = resolve_collection_or_exit(db, collection_name)
    group = groups_mod.resolve_group(db, target["id"], name)
    db.update_group(group["id"], destination_path=os.path.abspath(os.path.expanduser(destination)))
    click.echo(f"{group['name']} -> {os.path.abspath(os.path.expanduser(destination))}")


@groups.command("remove")
@click.argument("name")
@click.option("--collection", "collection_name", default=None)
@click.confirmation_option(prompt="Remove this group?")
def groups_remove(name, collection_name):
    """Delete a group (its tracks stay in the library)."""
    _, db = get_context()
    target = resolve_collection_or_exit(db, collection_name)
    group = groups_mod.resolve_group(db, target["id"], name)
    db.delete_group(group["id"])
    db.touch_collection(target["id"])
    click.echo(f"Removed {group['name']!r}.")


@groups.command("export")
@click.option("--collection", "collection_name", default=None)
@click.option("--output", type=click.Path(file_okay=False), default="./playlists")
@click.option("--format", "playlist_format", type=click.Choice(["m3u", "m3u8"]), default="m3u8")
@click.option("--relative-paths", is_flag=True)
def groups_export(collection_name, output, playlist_format, relative_paths):
    """Write one playlist per group."""
    _, db = get_context()
    target = resolve_collection_or_exit(db, collection_name)
    written = organizer.export_groups_as_playlists(
        db, db.list_groups(target["id"]), output,
        playlist_format=playlist_format, relative_paths=relative_paths,
    )
    click.echo(f"Wrote {len(written)} playlist(s) to {os.path.abspath(output)}")


# ----------------------------------------------------------------------
# Fit and check
# ----------------------------------------------------------------------


@cli.command()
@click.option("--collection", "collection_name", default=None)
def fit(collection_name):
    """Learn what each group contains."""
    config, db = get_context()
    target = resolve_collection_or_exit(db, collection_name)

    click.echo(f"Fitting {target['name']!r}...")
    try:
        metrics = groups_mod.fit_collection(db, target, config)
    except ValueError as exc:
        raise click.ClickException(str(exc))

    embedding = metrics.get("embedding", {})
    click.echo(
        f"Fitted {metrics['n_groups']} groups / {metrics['n_tracks']} reference tracks "
        f"({embedding.get('projection', '?')} space, {embedding.get('output_dim', '?')} dims)"
    )
    click.echo(f"Self-check accuracy: {metrics['accuracy'] * 100:.1f}%")
    for problem in metrics.get("problems", []):
        click.echo(click.style(f"  warning: {problem['name']}: {problem['issue']}", fg="yellow"))
    click.echo("\nRun `music-cluster check` for a per-group breakdown.")


@cli.command()
@click.option("--collection", "collection_name", default=None)
@click.option("--refit", is_flag=True, help="Refit before checking")
def check(collection_name, refit):
    """Report how distinguishable your groups are from each other."""
    config, db = get_context()
    target = resolve_collection_or_exit(db, collection_name)

    if refit:
        groups_mod.fit_collection(db, target, config)

    model = db.load_model(target["id"])
    if not model:
        raise click.ClickException("Not fitted yet. Run `music-cluster fit` first.")
    metrics = model["metrics"]

    click.echo(f"{target['name']} — self-check\n")
    click.echo(f"Overall accuracy: {metrics.get('accuracy', 0) * 100:.1f}%")
    click.echo(
        f"Reference tracks: {metrics.get('n_tracks', 0)} across {metrics.get('n_groups', 0)} groups\n"
    )

    click.echo(f"{'Group':<32} {'Tracks':>7} {'Correct':>8} {'Accuracy':>9}")
    for entry in sorted(metrics.get("per_group", []), key=lambda e: e["accuracy"]):
        color = "green" if entry["accuracy"] >= 0.8 else "yellow" if entry["accuracy"] >= 0.5 else "red"
        click.echo(
            f"{entry['name']:<32} {entry['size']:>7} {entry['correct']:>8} "
            + click.style(f"{entry['accuracy'] * 100:>8.0f}%", fg=color)
        )

    pairs = metrics.get("confusable_pairs", [])
    if pairs:
        click.echo("\nMost easily confused:")
        for pair in pairs:
            click.echo(
                f"  {pair['count']:>4} tracks from {pair['from_name']!r} "
                f"look like {pair['to_name']!r} ({pair['rate'] * 100:.0f}%)"
            )
        click.echo(
            "\nHigh overlap usually means two groups are genuinely similar — either merge\n"
            "them, or add more reference tracks that show what makes them different."
        )


# ----------------------------------------------------------------------
# Sorting
# ----------------------------------------------------------------------


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--collection", "collection_name", default=None)
@click.option("--name", default=None, help="Name for this sort session")
@click.option("-r/--no-recursive", "recursive", default=True)
@click.option("--auto-accept", is_flag=True, help="Pre-accept high-confidence suggestions")
@click.option("--workers", type=int, default=None)
@click.option("--review/--no-review", default=True, help="Open the review prompt when done")
def sort(paths, collection_name, name, recursive, auto_accept, workers, review):
    """Score new music against your groups."""
    config, db = get_context()
    target = resolve_collection_or_exit(db, collection_name)

    try:
        outcome = sorting.create_session(
            db, config, target, list(paths), name=name, recursive=recursive,
            workers=workers, progress=progress_bar("Analysing"), auto_accept=auto_accept,
        )
    except (ValueError, LookupError) as exc:
        raise click.ClickException(str(exc))

    session = outcome["session"]
    summary = outcome["summary"]
    click.echo(f"\nSession {session['id']} ({session['name']}):")
    click.echo(f"  {summary.get('pending', 0)} to review")
    if summary.get("accepted"):
        click.echo(f"  {summary['accepted']} auto-accepted")
    if summary.get("unmatched"):
        click.echo(f"  {summary['unmatched']} matched nothing in this collection")

    if review and summary.get("pending"):
        review_session(db, config, session["id"])
    elif summary.get("pending"):
        click.echo(f"\nNext: music-cluster review {session['id']}")
    else:
        click.echo(f"\nNext: music-cluster apply {session['id']} --mode copy")


@cli.command()
@click.argument("session_id", type=int)
def review(session_id):
    """Step through a session's suggestions and decide."""
    config, db = get_context()
    review_session(db, config, session_id)


def review_session(db: Database, config: Config, session_id: int) -> None:
    """Interactive review loop: one track at a time, one keystroke per decision."""
    overview = sorting.session_overview(db, session_id)
    groups_by_id = {group["id"]: group for group in overview["groups"]}

    while True:
        items = db.list_sort_items(session_id, status=sorting.STATUS_PENDING, limit=1)
        if not items:
            break
        item = items[0]
        remaining = db.sort_session_summary(session_id).get(sorting.STATUS_PENDING, 0)

        click.echo("\n" + "─" * 72)
        click.echo(click.style(display_name(item), bold=True) + f"   [{remaining} left]")
        details = [
            f"{int(item['duration'] // 60)}:{int(item['duration'] % 60):02d}" if item["duration"] else None,
            f"{item['tag_bpm']:.0f} BPM" if item["tag_bpm"] else None,
            item["tag_key"],
            item["genre"],
        ]
        detail_line = "  ".join(part for part in details if part)
        if detail_line:
            click.echo(click.style(detail_line, dim=True))
        click.echo("")

        ranked = item.get("ranked") or []
        if not ranked:
            click.echo(click.style("  No group came close to this track.", fg="yellow"))
        for index, entry in enumerate(ranked[:5], start=1):
            name = groups_by_id.get(entry["group_id"], {}).get("name", entry["group_name"])
            click.echo(f"  {index}. {format_confidence(entry['score'])}  {name}")

        click.echo("\n  [1-5] file it   [s] skip   [l] list all groups   [q] quit")
        choice = click.prompt("  >", default="1", show_default=False).strip().lower()

        if choice == "q":
            click.echo(f"\nStopped. Resume with: music-cluster review {session_id}")
            return
        if choice == "s":
            sorting.decide(db, item["id"], skip=True)
            continue
        if choice == "l":
            for group in overview["groups"]:
                click.echo(f"    {group['id']:>4}  {group['name']} ({group['size']})")
            group_id = click.prompt("  Group ID", type=int, default=0)
            if group_id in groups_by_id:
                sorting.decide(db, item["id"], group_id=group_id)
            continue
        if choice.isdigit() and 1 <= int(choice) <= len(ranked):
            sorting.decide(db, item["id"], group_id=ranked[int(choice) - 1]["group_id"])
            continue

        click.echo(click.style("  Not a valid choice.", fg="yellow"))

    summary = db.sort_session_summary(session_id)
    click.echo(f"\nReview complete: {summary.get('accepted', 0)} filed, "
               f"{summary.get('skipped', 0)} skipped.")
    click.echo(f"Next: music-cluster apply {session_id} --mode copy")


@cli.command(name="sessions")
@click.option("--collection", "collection_name", default=None)
def sessions_cmd(collection_name):
    """List sort sessions."""
    _, db = get_context()
    target = groups_mod.resolve_collection(db, collection_name, required=False)
    rows = db.list_sort_sessions(target["id"] if target else None)
    if not rows:
        click.echo("No sort sessions yet.")
        return
    for row in rows:
        click.echo(
            f"{row['id']:>4}  {row['name'] or '(unnamed)':<28} {row['status']:<10} "
            f"{row['item_count']:>5} tracks, {row['pending_count']:>4} pending"
        )


@cli.command()
@click.argument("session_id", type=int)
@click.option("--mode", type=click.Choice(list(organizer.MODES)), default=None,
              help="How to file the tracks (default: from config)")
@click.option("--playlist-dir", type=click.Path(file_okay=False), default=None)
@click.option("--on-conflict", type=click.Choice(list(organizer.CONFLICT_POLICIES)), default=None)
@click.option("--dry-run", is_flag=True, help="Show what would happen and stop")
@click.option("--yes", is_flag=True, help="Skip the confirmation prompt")
@click.option("--learn/--no-learn", default=None,
              help="Add filed tracks to their groups as new references")
def apply(session_id, mode, playlist_dir, on_conflict, dry_run, yes, learn):
    """Write a session's decisions to disk."""
    config, db = get_context()

    try:
        plan = sorting.plan_session_commit(
            db, config, session_id, mode=mode, playlist_dir=playlist_dir, on_conflict=on_conflict
        )
    except LookupError as exc:
        raise click.ClickException(str(exc))

    if plan.is_empty and not plan.problems:
        click.echo("Nothing to apply — no accepted decisions are waiting.")
        return

    click.echo(f"Mode: {plan.mode}\n")
    for entry in plan.playlists:
        click.echo(f"  playlist  {entry['track_count']:>4} tracks -> {entry['path']}")
    for action in plan.actions[:20]:
        click.echo(
            f"  {action.action:<8} {os.path.basename(action.source_path)} -> {action.dest_path}"
            + (f"  ({action.note})" if action.note else "")
        )
    if len(plan.actions) > 20:
        click.echo(f"  ... and {len(plan.actions) - 20} more")

    for problem in plan.problems:
        click.echo(click.style(f"  skipped: {problem.get('issue')}"
                               f" ({problem.get('group_name') or problem.get('filepath') or problem.get('track_id')})",
                               fg="yellow"))

    if dry_run:
        click.echo("\nDry run — nothing was changed.")
        return
    if plan.is_empty:
        click.echo("\nNothing left to do.")
        return

    if not yes:
        verb = "move" if plan.mode == "move" else plan.mode
        if not click.confirm(f"\nProceed to {verb} {len(plan.actions) or len(plan.playlists)} item(s)?"):
            click.echo("Cancelled.")
            return

    outcome = sorting.commit_session(
        db, config, session_id, mode=mode, playlist_dir=playlist_dir,
        on_conflict=on_conflict, learn=learn,
    )
    result = outcome["result"]
    click.echo(
        f"\nDone: {result['performed_count']} file operation(s), "
        f"{len(result['playlists'])} playlist(s)."
    )
    if outcome["learned"]:
        click.echo(f"Added {outcome['learned']} track(s) to their groups as new references.")
        click.echo("Run `music-cluster fit` to fold them into the model.")
    for failure in result["failures"][:5]:
        echo_error(f"{failure.get('source_path') or failure.get('path')}: {failure['error']}")

    if plan.mode == "move":
        click.echo(f"\nTo undo: music-cluster undo {session_id}")


@cli.command()
@click.argument("session_id", type=int)
@click.option("--remove-playlists", is_flag=True, help="Also delete generated playlists")
@click.confirmation_option(prompt="Reverse the file operations from this session?")
def undo(session_id, remove_playlists):
    """Reverse a session's file operations."""
    _, db = get_context()
    outcome = organizer.undo_session(db, session_id, remove_playlists=remove_playlists)
    click.echo(f"Reversed {outcome['undone']} operation(s).")
    for failure in outcome["failures"][:5]:
        echo_error(str(failure))


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------


@cli.command()
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("--name", default=None, help="Name for this discovery run")
@click.option("--algorithm", type=click.Choice(["hdbscan", "kmeans", "hierarchical"]), default=None)
@click.option("--groups", "target_groups", type=int, default=None,
              help="Aim for this many candidate groups (kmeans/hierarchical)")
@click.option("--granularity", type=click.Choice(["coarse", "broad", "normal", "fine", "finest"]),
              default=None)
@click.option("--min-size", type=int, default=None, help="Smallest candidate group to report")
@click.option("--llm/--no-llm", default=None, help="Use an LLM to suggest names")
@click.option("--workers", type=int, default=None)
def discover(paths, name, algorithm, target_groups, granularity, min_size, llm, workers):
    """Propose candidate groups from an unsorted pile."""
    config, db = get_context()
    try:
        outcome = discovery_mod.run_discovery(
            db, config, paths=list(paths), name=name, algorithm=algorithm,
            granularity=granularity, target_groups=target_groups, min_group_size=min_size,
            workers=workers, progress=progress_bar("Analysing"), use_llm=llm,
        )
    except (ValueError, ImportError) as exc:
        raise click.ClickException(str(exc))

    run = outcome["run"]
    click.echo(f"\nRun {run['id']}: {len(outcome['candidates'])} candidate group(s)")
    if outcome["unassigned"]:
        click.echo(f"{outcome['unassigned']} track(s) did not fit any candidate")
    click.echo("")
    for candidate in outcome["candidates"]:
        stats = candidate["stats"]
        bpm = stats.get("tag_bpm_median") or stats.get("detected_bpm")
        line = (
            f"  [{candidate['candidate_index']:>2}] {candidate['suggested_name']:<34} "
            f"{candidate['size']:>5} tracks"
        )
        click.echo(f"{line}  ~{bpm:.0f} BPM" if bpm else line)
    click.echo(f"\nNext: music-cluster candidates {run['id']}")


@cli.command(name="candidates")
@click.argument("run_id", type=int)
def candidates_cmd(run_id):
    """Review a discovery run's candidates and promote the good ones."""
    config, db = get_context()
    run = db.get_discovery_run(run_id)
    if not run:
        raise click.ClickException(f"No discovery run with ID {run_id}")

    target = resolve_collection_or_exit(db, None)
    click.echo(f"Run {run_id} ({run['name']}) — promoting into {target['name']!r}\n")

    for candidate in db.list_discovery_candidates(run_id):
        if candidate["status"] != "pending":
            continue
        stats = candidate["stats"]
        click.echo("─" * 72)
        click.echo(click.style(candidate["suggested_name"], bold=True)
                   + f"   {candidate['size']} tracks")
        descriptors = ", ".join(stats.get("descriptors", []))
        bpm = stats.get("tag_bpm_median") or stats.get("detected_bpm")
        tempo = f"~{bpm:.0f} BPM   " if bpm else ""
        click.echo(click.style(f"  {tempo}{descriptors}", dim=True))
        if stats.get("top_genres"):
            tags = ", ".join(f"{g['name']} ({g['count']})" for g in stats["top_genres"])
            click.echo(click.style(f"  tags: {tags}", dim=True))

        click.echo("\n  Representative tracks:")
        for track_id in candidate["exemplars"]:
            track = db.get_track(track_id)
            if track:
                click.echo(f"    {track['filepath']}")

        click.echo("\n  [y] make this a group   [s] seeds only   [n] discard   [q] quit")
        choice = click.prompt("  >", default="n", show_default=False).strip().lower()

        if choice == "q":
            return
        if choice == "n":
            discovery_mod.reject_candidate(db, candidate["id"])
            continue
        if choice in ("y", "s"):
            group_name = click.prompt("  Group name", default=candidate["suggested_name"])
            destination = click.prompt("  Destination folder (blank for none)", default="",
                                       show_default=False).strip()
            outcome = discovery_mod.promote_candidate(
                db, config, target, candidate["id"], name=group_name,
                destination_path=destination or None, seeds_only=(choice == "s"),
            )
            click.echo(f"  Created {group_name!r} with {outcome['added']} reference track(s).\n")

    click.echo("\nDone. Next: music-cluster fit")


@cli.command(name="similar")
@click.argument("query")
@click.option("--limit", type=int, default=20)
def similar_cmd(query, limit):
    """Find tracks that sound like a track matching QUERY."""
    config, db = get_context()
    matches = db.list_tracks(limit=5, query=query)
    if not matches:
        raise click.ClickException(f"No track matching {query!r}")

    seed = matches[0]
    click.echo(f"Tracks similar to {display_name(seed)}:\n")
    results = discovery_mod.expand_seeds(db, config, [seed["id"]], limit=limit)
    for result in results:
        click.echo(f"  {result['relative_distance']:.2f}  {display_name(result['track'])}")


@cli.command()
@click.argument("query")
@click.option("--limit", type=int, default=20)
def search(query, limit):
    """Search the library and show which groups each track belongs to."""
    _, db = get_context()
    tracks = db.list_tracks(limit=limit, query=query)
    if not tracks:
        click.echo("No matches.")
        return
    for track in tracks:
        memberships = db.find_track_groups(track["id"])
        labels = ", ".join(group["name"] for group in memberships) or "unfiled"
        click.echo(f"  {display_name(track):<52} {click.style(labels, dim=True)}")


@cli.command(name="config")
@click.option("--set", "assignments", multiple=True, metavar="KEY=VALUE",
              help="Set a dotted config key, e.g. sorting.neighbors=8")
@click.option("--json", "as_json", is_flag=True, help="Print the config as JSON")
def config_cmd(assignments, as_json):
    """Show or change configuration."""
    config = Config()

    for assignment in assignments:
        if "=" not in assignment:
            raise click.ClickException(f"Expected KEY=VALUE, got {assignment!r}")
        key, raw = assignment.split("=", 1)
        config.set(key.split("."), _coerce(raw))
        click.echo(f"{key} = {_coerce(raw)!r}")

    if assignments:
        config.save()
        click.echo(f"\nSaved to {config.config_path}")
        return

    if as_json:
        click.echo(json.dumps(config.config, indent=2))
    else:
        import yaml

        click.echo(f"# {config.config_path}\n")
        click.echo(yaml.dump(config.config, default_flow_style=False, sort_keys=False))


def _coerce(raw: str):
    """Turn a CLI string into the type the config expects."""
    lowered = raw.strip().lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    if lowered in ("none", "null", ""):
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def main():
    try:
        cli()
    except KeyboardInterrupt:
        click.echo("\nInterrupted.")
        sys.exit(130)


if __name__ == "__main__":
    main()
