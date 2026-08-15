"""Sort sessions: the workflow for filing a batch of new music.

A session takes a folder (or explicit list) of incoming tracks, scores each one
against the groups of a collection, and holds the suggestions until a human has
signed off on them. Committing a session writes the decisions to disk and — by
default — folds the confirmed tracks back into their groups as new references,
so the collection gets better at its owner's taste with every batch.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import Config
from .database import Database
from .groups import load_sorter, resolve_collection
from .library import analyze_files, collect_files
from .metadata import display_name
from .organizer import CommitPlan, CommitResult, execute_plan, plan_commit
from .sorter import STATUS_AUTO, STATUS_REVIEW, STATUS_UNMATCHED


# Item statuses. The first three come from the sorter; the rest are decisions.
STATUS_PENDING = "pending"
STATUS_ACCEPTED = "accepted"
STATUS_SKIPPED = "skipped"
DECIDED_STATUSES = (STATUS_ACCEPTED,)


def create_session(
    db: Database,
    config: Config,
    collection: Dict,
    paths: Sequence[str],
    name: Optional[str] = None,
    recursive: bool = True,
    workers: Optional[int] = None,
    progress: Optional[Callable[[int, int, str], None]] = None,
    auto_accept: bool = False,
) -> Dict[str, Any]:
    """Create a sort session for the audio under ``paths`` and score it.

    Args:
        auto_accept: Pre-accept high-confidence suggestions so the human only
            reviews what the sorter is unsure about. Off by default: the human
            stays in the loop unless they opt out of it.
    """
    files = collect_files(paths, recursive=recursive)
    if not files:
        raise ValueError("No audio files found in the given path(s)")

    analysis = analyze_files(db, config, files, workers=workers, progress=progress)
    if not analysis.track_ids:
        raise ValueError("None of the files could be analysed")

    source = paths[0] if len(paths) == 1 else f"{len(paths)} paths"
    session_id = db.create_sort_session(
        collection_id=collection["id"],
        name=name or _default_session_name(paths),
        source_path=os.path.abspath(os.path.expanduser(source))
        if len(paths) == 1
        else None,
        settings={"auto_accept": auto_accept},
    )

    score_session(db, config, session_id, track_ids=analysis.track_ids, auto_accept=auto_accept)
    return {
        "session": db.get_sort_session(session_id),
        "analysis": analysis.to_dict(),
        "summary": db.sort_session_summary(session_id),
    }


def score_session(
    db: Database,
    config: Config,
    session_id: int,
    track_ids: Optional[Sequence[int]] = None,
    auto_accept: bool = False,
) -> Dict[str, int]:
    """(Re-)score a session's tracks against its collection's current groups.

    Safe to run again after the collection changes: existing human decisions
    are preserved, only pending suggestions are refreshed.
    """
    session = db.get_sort_session(session_id)
    if not session:
        raise LookupError(f"No sort session with ID {session_id}")

    collection = resolve_collection(db, session["collection_id"])
    sorter, _ = load_sorter(db, collection, config)

    if track_ids is None:
        track_ids = [item["track_id"] for item in db.list_sort_items(session_id, limit=10**6)]
    if not track_ids:
        return {}

    matrix, found_ids = db.get_feature_matrix(track_ids)
    if len(found_ids) == 0:
        raise ValueError("None of the session's tracks have features")

    suggestions = sorter.suggest_batch(matrix, found_ids)
    items = []
    for suggestion in suggestions:
        payload = suggestion.to_dict()
        if suggestion.status == STATUS_AUTO and auto_accept:
            payload["status"] = STATUS_ACCEPTED
        elif suggestion.status in (STATUS_AUTO, STATUS_REVIEW):
            # A confident suggestion is still only a suggestion until a human
            # confirms it; the confidence is stored so the UI can rank by it.
            payload["status"] = STATUS_PENDING
        else:
            payload["status"] = STATUS_UNMATCHED
        items.append(payload)

    db.upsert_sort_items(session_id, items)

    if auto_accept:
        for item in db.list_sort_items(session_id, limit=10**6):
            if item["status"] == STATUS_ACCEPTED and item["final_group_id"] is None:
                db.decide_sort_item(item["id"], STATUS_ACCEPTED, item["suggested_group_id"])

    return db.sort_session_summary(session_id)


def decide(
    db: Database,
    item_id: int,
    group_id: Optional[int] = None,
    skip: bool = False,
) -> Dict[str, Any]:
    """Record a human decision for one item.

    ``group_id`` of None with ``skip=False`` resets the item to pending.
    """
    item = db.get_sort_item(item_id)
    if not item:
        raise LookupError(f"No sort item with ID {item_id}")

    if skip:
        db.decide_sort_item(item_id, STATUS_SKIPPED, None)
    elif group_id is None:
        db.decide_sort_item(item_id, STATUS_PENDING, None)
    else:
        db.decide_sort_item(item_id, STATUS_ACCEPTED, group_id)

    return db.get_sort_item(item_id)


def accept_all_confident(db: Database, config: Config, session_id: int) -> int:
    """Accept every pending item whose suggestion cleared the auto threshold."""
    session = db.get_sort_session(session_id)
    collection = resolve_collection(db, session["collection_id"])
    from .groups import sorter_config_for  # local import avoids a cycle at module load

    thresholds = sorter_config_for(collection, config)

    accepted = 0
    for item in db.list_sort_items(session_id, status=STATUS_PENDING, limit=10**6):
        if item["suggested_group_id"] is None:
            continue
        if (
            (item["confidence"] or 0) >= thresholds.auto_accept_confidence
            and (item["margin"] or 0) >= thresholds.min_margin
        ):
            db.decide_sort_item(item["id"], STATUS_ACCEPTED, item["suggested_group_id"])
            accepted += 1
    return accepted


def pending_assignments(db: Database, session_id: int) -> List[Dict[str, Any]]:
    """Accepted-but-not-yet-applied items, ready to be written to disk."""
    assignments = []
    for item in db.list_sort_items(session_id, status=STATUS_ACCEPTED, limit=10**6):
        if item["applied"] or not item["final_group_id"]:
            continue
        assignments.append(
            {
                "item_id": item["id"],
                "track_id": item["track_id"],
                "group_id": item["final_group_id"],
                "filepath": item["filepath"],
                "filename": item["filename"],
                "duration": item["duration"],
                "display": display_name(item),
            }
        )
    return assignments


def plan_session_commit(
    db: Database,
    config: Config,
    session_id: int,
    mode: Optional[str] = None,
    playlist_dir: Optional[str] = None,
    on_conflict: Optional[str] = None,
) -> CommitPlan:
    """Work out what committing a session would do, without touching anything."""
    session = db.get_sort_session(session_id)
    if not session:
        raise LookupError(f"No sort session with ID {session_id}")

    organize = config.section("organize")
    groups = {group["id"]: group for group in db.list_groups(session["collection_id"])}

    return plan_commit(
        db,
        pending_assignments(db, session_id),
        groups,
        mode=mode or organize.get("mode", "playlist"),
        playlist_dir=playlist_dir or organize.get("playlist_dir"),
        playlist_format=organize.get("playlist_format", "m3u8"),
        relative_paths=organize.get("relative_paths", False),
        on_conflict=on_conflict or organize.get("on_conflict", "skip"),
        session_name=session.get("name"),
    )


def commit_session(
    db: Database,
    config: Config,
    session_id: int,
    mode: Optional[str] = None,
    playlist_dir: Optional[str] = None,
    on_conflict: Optional[str] = None,
    learn: Optional[bool] = None,
) -> Dict[str, Any]:
    """Apply a session's accepted decisions.

    Args:
        learn: Add the committed tracks to their groups as new references.
            Defaults to the ``sorting.learn_on_commit`` setting.
    """
    session = db.get_sort_session(session_id)
    if not session:
        raise LookupError(f"No sort session with ID {session_id}")

    plan = plan_session_commit(
        db, config, session_id, mode=mode, playlist_dir=playlist_dir, on_conflict=on_conflict
    )
    assignments = pending_assignments(db, session_id)
    result = execute_plan(
        db, plan, assignments, session_id=session_id, playlist_header=session.get("name")
    )
    db.mark_items_applied(result.applied_item_ids)

    if learn is None:
        learn = config.get("sorting", "learn_on_commit", default=True)

    learned = 0
    if learn:
        learned = _learn_from_commit(db, session, result, assignments)

    summary = db.sort_session_summary(session_id)
    if not summary.get(STATUS_PENDING):
        db.update_sort_session(session_id, status="committed", committed_at=_now())

    return {
        "plan": plan.to_dict(),
        "result": result.to_dict(),
        "learned": learned,
        "summary": summary,
    }


def _learn_from_commit(
    db: Database,
    session: Dict,
    result: CommitResult,
    assignments: Sequence[Dict[str, Any]],
) -> int:
    """Add successfully filed tracks to their groups as confirmed references."""
    applied = {assignment["item_id"] for assignment in assignments} & set(
        result.applied_item_ids
    )
    by_group: Dict[int, List[int]] = {}
    for assignment in assignments:
        if assignment["item_id"] in applied:
            by_group.setdefault(assignment["group_id"], []).append(assignment["track_id"])

    learned = 0
    for group_id, track_ids in by_group.items():
        existing = set(db.get_group_member_ids(group_id))
        fresh = [track_id for track_id in track_ids if track_id not in existing]
        if fresh:
            db.add_group_members(group_id, fresh, role="confirmed")
            learned += len(fresh)

    if learned:
        # Invalidate the fitted sorter: the references it was built from changed.
        db.touch_collection(session["collection_id"])
    return learned


def session_overview(db: Database, session_id: int) -> Dict[str, Any]:
    """Session, its collection's groups, and per-status counts."""
    session = db.get_sort_session(session_id)
    if not session:
        raise LookupError(f"No sort session with ID {session_id}")

    groups = db.list_groups(session["collection_id"])
    summary = db.sort_session_summary(session_id)
    return {
        "session": session,
        "groups": groups,
        "summary": summary,
        "total": sum(summary.values()),
    }


def _default_session_name(paths: Sequence[str]) -> str:
    if len(paths) == 1:
        return os.path.basename(os.path.normpath(os.path.expanduser(paths[0]))) or "Sort"
    return f"{len(paths)} folders"


def _now() -> str:
    from .database import utc_now

    return utc_now()
