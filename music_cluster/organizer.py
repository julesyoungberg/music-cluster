"""Applying sorting decisions to disk.

Four modes, from safest to most committal:

``playlist``  write one playlist per group (nothing on disk moves)
``symlink``   link the file into the group's folder
``copy``      copy the file into the group's folder
``move``      move the file into the group's folder, and follow it in the library

Every mode plans first and can be run as a dry run, and every file operation is
recorded so a commit can be undone.
"""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .database import Database
from .utils import ensure_directory, get_relative_path


MODES = ("playlist", "copy", "move", "symlink")
CONFLICT_POLICIES = ("skip", "rename", "overwrite")


@dataclass
class PlannedAction:
    """One intended file operation."""

    track_id: int
    group_id: int
    group_name: str
    action: str
    source_path: str
    dest_path: Optional[str]
    note: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "group_id": self.group_id,
            "group_name": self.group_name,
            "action": self.action,
            "source_path": self.source_path,
            "dest_path": self.dest_path,
            "note": self.note,
        }


@dataclass
class CommitPlan:
    """The full set of operations a commit would perform."""

    mode: str
    actions: List[PlannedAction] = field(default_factory=list)
    playlists: List[Dict[str, Any]] = field(default_factory=list)
    problems: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.actions and not self.playlists

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "actions": [action.to_dict() for action in self.actions],
            "playlists": self.playlists,
            "problems": self.problems,
            "action_count": len(self.actions),
            "playlist_count": len(self.playlists),
        }


@dataclass
class CommitResult:
    """What actually happened."""

    mode: str
    performed: List[Dict[str, Any]] = field(default_factory=list)
    playlists: List[str] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    applied_item_ids: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "performed": self.performed,
            "performed_count": len(self.performed),
            "playlists": self.playlists,
            "failures": self.failures,
            "applied_item_ids": self.applied_item_ids,
        }


def plan_commit(
    db: Database,
    assignments: Sequence[Dict[str, Any]],
    groups_by_id: Dict[int, Dict],
    mode: str = "playlist",
    playlist_dir: Optional[str] = None,
    playlist_format: str = "m3u8",
    relative_paths: bool = False,
    on_conflict: str = "skip",
    session_name: Optional[str] = None,
) -> CommitPlan:
    """Work out what a commit would do.

    ``assignments`` are dicts with at least ``track_id``, ``group_id`` and
    ``filepath``; extra keys (such as ``item_id``) are carried through.
    """
    if mode not in MODES:
        raise ValueError(f"Unknown mode {mode!r}; choose from {MODES}")
    if on_conflict not in CONFLICT_POLICIES:
        raise ValueError(f"Unknown conflict policy {on_conflict!r}")

    plan = CommitPlan(mode=mode)
    by_group: Dict[int, List[Dict[str, Any]]] = {}

    for assignment in assignments:
        group_id: Optional[int] = assignment.get("group_id")
        group = groups_by_id.get(group_id) if group_id is not None else None
        if group_id is None or group is None:
            plan.problems.append({"track_id": assignment.get("track_id"), "issue": "unknown group"})
            continue
        if not os.path.exists(assignment["filepath"]):
            plan.problems.append(
                {
                    "track_id": assignment.get("track_id"),
                    "issue": "file no longer exists",
                    "filepath": assignment["filepath"],
                }
            )
            continue
        by_group.setdefault(group_id, []).append(assignment)

    if mode == "playlist":
        directory = os.path.abspath(
            os.path.expanduser(playlist_dir or "~/.music-cluster/playlists")
        )
        for group_id, items in by_group.items():
            group = groups_by_id[group_id]
            filename = _safe_filename(group["name"]) + f".{playlist_format}"
            if session_name:
                filename = f"{_safe_filename(session_name)} - {filename}"
            plan.playlists.append(
                {
                    "group_id": group_id,
                    "group_name": group["name"],
                    "path": os.path.join(directory, filename),
                    "track_count": len(items),
                    "relative_paths": relative_paths,
                }
            )
        return plan

    planned_destinations: Dict[str, int] = {}
    for group_id, items in by_group.items():
        group = groups_by_id[group_id]
        destination = group.get("destination_path")
        if not destination:
            plan.problems.append(
                {
                    "group_id": group_id,
                    "group_name": group["name"],
                    "issue": "group has no destination folder",
                    "track_count": len(items),
                }
            )
            continue

        for assignment in items:
            source = assignment["filepath"]
            target = os.path.join(destination, os.path.basename(source))
            note = None

            if os.path.abspath(source) == os.path.abspath(target):
                plan.problems.append(
                    {
                        "track_id": assignment["track_id"],
                        "group_id": group_id,
                        "issue": "already in destination folder",
                    }
                )
                continue

            taken = os.path.exists(target) or target in planned_destinations
            if taken:
                if on_conflict == "skip":
                    plan.problems.append(
                        {
                            "track_id": assignment["track_id"],
                            "group_id": group_id,
                            "issue": "destination already has a file with this name",
                            "dest_path": target,
                        }
                    )
                    continue
                if on_conflict == "rename":
                    target = _unique_path(target, planned_destinations)
                    note = "renamed to avoid collision"
                else:
                    note = "overwrites existing file"

            planned_destinations[target] = assignment["track_id"]
            plan.actions.append(
                PlannedAction(
                    track_id=assignment["track_id"],
                    group_id=group_id,
                    group_name=group["name"],
                    action=mode,
                    source_path=source,
                    dest_path=target,
                    note=note,
                )
            )

    return plan


def execute_plan(
    db: Database,
    plan: CommitPlan,
    assignments: Sequence[Dict[str, Any]],
    session_id: Optional[int] = None,
    playlist_header: Optional[str] = None,
) -> CommitResult:
    """Carry out a plan, logging every operation so it can be undone."""
    result = CommitResult(mode=plan.mode)
    log: List[Dict[str, Any]] = []
    by_track = {assignment["track_id"]: assignment for assignment in assignments}

    for entry in plan.playlists:
        items = [
            assignment
            for assignment in assignments
            if assignment.get("group_id") == entry["group_id"]
        ]
        try:
            write_playlist(
                entry["path"],
                items,
                relative_paths=entry.get("relative_paths", False),
                header=playlist_header or entry["group_name"],
            )
            result.playlists.append(entry["path"])
            log.append(
                {
                    "session_id": session_id,
                    "track_id": None,
                    "group_id": entry["group_id"],
                    "action": "playlist",
                    "source_path": None,
                    "dest_path": entry["path"],
                }
            )
            result.applied_item_ids.extend(
                assignment["item_id"] for assignment in items if assignment.get("item_id")
            )
        except Exception as exc:
            result.failures.append({"path": entry["path"], "error": str(exc)})

    for action in plan.actions:
        try:
            if not action.dest_path:
                raise ValueError(f"Action {action.action!r} has no destination path")
            ensure_directory(os.path.dirname(action.dest_path))
            if action.action == "copy":
                shutil.copy2(action.source_path, action.dest_path)
            elif action.action == "move":
                shutil.move(action.source_path, action.dest_path)
                db.update_track_path(
                    action.track_id, action.dest_path, os.path.basename(action.dest_path)
                )
            elif action.action == "symlink":
                if os.path.lexists(action.dest_path):
                    os.remove(action.dest_path)
                os.symlink(action.source_path, action.dest_path)
            else:
                raise ValueError(f"Cannot execute action {action.action!r}")

            result.performed.append(action.to_dict())
            log.append(
                {
                    "session_id": session_id,
                    "track_id": action.track_id,
                    "group_id": action.group_id,
                    "action": action.action,
                    "source_path": action.source_path,
                    "dest_path": action.dest_path,
                }
            )
            assignment = by_track.get(action.track_id)
            if assignment and assignment.get("item_id"):
                result.applied_item_ids.append(assignment["item_id"])
        except Exception as exc:
            result.failures.append(
                {
                    "track_id": action.track_id,
                    "source_path": action.source_path,
                    "dest_path": action.dest_path,
                    "error": str(exc),
                }
            )

    db.log_file_actions(log)
    return result


def undo_session(db: Database, session_id: int, remove_playlists: bool = False) -> Dict[str, Any]:
    """Reverse the file operations recorded for a session."""
    actions = db.list_file_actions(session_id)
    undone: List[int] = []
    failures: List[Dict[str, Any]] = []

    for action in actions:
        kind = action["action"]
        try:
            if kind == "move":
                if os.path.exists(action["dest_path"]):
                    ensure_directory(os.path.dirname(action["source_path"]))
                    shutil.move(action["dest_path"], action["source_path"])
                if action["track_id"]:
                    db.update_track_path(
                        action["track_id"],
                        action["source_path"],
                        os.path.basename(action["source_path"]),
                    )
            elif kind in ("copy", "symlink"):
                if os.path.lexists(action["dest_path"]):
                    os.remove(action["dest_path"])
            elif kind == "playlist":
                if remove_playlists and os.path.exists(action["dest_path"]):
                    os.remove(action["dest_path"])
                else:
                    continue
            undone.append(action["id"])
        except Exception as exc:
            failures.append({"action_id": action["id"], "error": str(exc)})

    db.mark_actions_undone(undone)
    return {"undone": len(undone), "failures": failures}


def write_playlist(
    path: str,
    tracks: Sequence[Dict[str, Any]],
    relative_paths: bool = False,
    header: Optional[str] = None,
) -> str:
    """Write an extended M3U playlist."""
    ensure_directory(os.path.dirname(path))
    base = os.path.dirname(path)

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("#EXTM3U\n")
        if header:
            handle.write(f"#PLAYLIST:{header}\n")
        for track in tracks:
            filepath = track["filepath"]
            duration = int(track.get("duration") or 0)
            label = track.get("display") or track.get("filename") or Path(filepath).name
            handle.write(f"#EXTINF:{duration},{label}\n")
            handle.write(f"{get_relative_path(filepath, base) if relative_paths else filepath}\n")
    return path


def export_groups_as_playlists(
    db: Database,
    groups: Sequence[Dict],
    output_dir: str,
    playlist_format: str = "m3u8",
    relative_paths: bool = False,
) -> List[str]:
    """Write one playlist per group containing all of its reference tracks."""
    output_dir = os.path.abspath(os.path.expanduser(output_dir))
    written: List[str] = []

    for group in groups:
        members = db.get_group_members(group["id"], limit=10**6)
        if not members:
            continue
        path = os.path.join(output_dir, f"{_safe_filename(group['name'])}.{playlist_format}")
        write_playlist(path, members, relative_paths=relative_paths, header=group["name"])
        written.append(path)

    return written


def _safe_filename(name: str) -> str:
    """Make a group name usable as a filename on any platform."""
    cleaned = "".join("_" if char in '<>:"/\\|?*' else char for char in name).strip(" .")
    return cleaned[:120] or "group"


def _unique_path(path: str, reserved: Dict[str, int]) -> str:
    """Append a counter until the path is free both on disk and in the plan."""
    stem, suffix = os.path.splitext(path)
    counter = 1
    candidate = f"{stem} ({counter}){suffix}"
    while os.path.exists(candidate) or candidate in reserved:
        counter += 1
        candidate = f"{stem} ({counter}){suffix}"
    return candidate
