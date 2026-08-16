"""REST API behind the desktop UI.

Long-running work (analysis, sorting, discovery) runs as a background task and
is polled through ``/api/tasks/{id}`` — everything else is synchronous.

Run with:
    uvicorn music_cluster.api:app --reload --port 8000
"""

import logging
import os
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from . import discovery as discovery_mod
from . import groups as groups_mod
from . import organizer, sorting
from .config import Config
from .database import Database
from .library import analyze_paths, prune_missing
from .media import cached_waveform, extract_artwork, media_type_for, parse_range_header
from .metadata import display_name
from .utils import find_audio_files


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Music Cluster API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db() -> Database:
    return Database(Config().get_db_path())


def get_context():
    config = Config()
    return config, Database(config.get_db_path())


# ----------------------------------------------------------------------
# Background tasks
# ----------------------------------------------------------------------

_tasks: Dict[str, Dict[str, Any]] = {}
_tasks_lock = threading.Lock()


def start_task(kind: str, worker, **kwargs) -> str:
    """Run ``worker(update, **kwargs)`` in the background and return its ID."""
    task_id = uuid.uuid4().hex

    with _tasks_lock:
        _tasks[task_id] = {
            "id": task_id,
            "kind": kind,
            "status": "running",
            "progress": 0.0,
            "current": 0,
            "total": 0,
            "message": "Starting...",
            "started_at": time.time(),
            "result": None,
            "error": None,
        }

    def update(**fields: Any) -> None:
        with _tasks_lock:
            _tasks[task_id].update(fields)

    def run() -> None:
        try:
            result = worker(update, **kwargs)
            update(status="completed", progress=1.0, result=result, finished_at=time.time())
        except Exception as exc:
            logger.exception("Background task %s (%s) failed", task_id, kind)
            update(status="failed", error=str(exc), finished_at=time.time())

    threading.Thread(target=run, daemon=True).start()
    return task_id


def progress_updater(update, label: str):
    """Adapt the library's progress callback to task status fields."""

    def callback(done: int, total: int, filepath: str) -> None:
        update(
            current=done,
            total=total,
            progress=(done / total) if total else 0.0,
            message=f"{label} {done}/{total}: {os.path.basename(filepath)}" if total else label,
        )

    return callback


@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Poll a background task."""
    with _tasks_lock:
        task = _tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/api/tasks")
async def list_tasks(limit: int = 20):
    """Recent background tasks, newest first."""
    with _tasks_lock:
        tasks = sorted(_tasks.values(), key=lambda t: t["started_at"], reverse=True)
    return {"tasks": tasks[:limit]}


# ----------------------------------------------------------------------
# Request models
# ----------------------------------------------------------------------


class CollectionRequest(BaseModel):
    name: str
    description: Optional[str] = None


class CollectionUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None


class GroupRequest(BaseModel):
    name: str
    description: Optional[str] = None
    destination_path: Optional[str] = None
    color: Optional[str] = None


class GroupUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    destination_path: Optional[str] = None
    color: Optional[str] = None
    position: Optional[int] = None


class ImportRequest(BaseModel):
    path: str
    name: Optional[str] = None
    recursive: bool = True
    destination_path: Optional[str] = None
    workers: Optional[int] = None


class ImportTreeRequest(BaseModel):
    path: str
    recursive: bool = True
    min_tracks: int = 1
    workers: Optional[int] = None


class AddMembersRequest(BaseModel):
    track_ids: List[int] = Field(default_factory=list)
    filepaths: List[str] = Field(default_factory=list)
    role: str = "reference"


class AnalyzeRequest(BaseModel):
    paths: List[str]
    recursive: bool = True
    update: bool = False
    workers: Optional[int] = None


class SortRequest(BaseModel):
    collection_id: int
    paths: List[str]
    name: Optional[str] = None
    recursive: bool = True
    auto_accept: bool = False
    workers: Optional[int] = None


class DecisionRequest(BaseModel):
    group_id: Optional[int] = None
    skip: bool = False


class BulkDecisionRequest(BaseModel):
    item_ids: List[int]
    group_id: Optional[int] = None
    skip: bool = False


class CommitRequest(BaseModel):
    mode: Optional[str] = None
    playlist_dir: Optional[str] = None
    on_conflict: Optional[str] = None
    learn: Optional[bool] = None
    dry_run: bool = False


class DiscoverRequest(BaseModel):
    paths: List[str] = Field(default_factory=list)
    track_ids: List[int] = Field(default_factory=list)
    name: Optional[str] = None
    algorithm: Optional[str] = None
    granularity: Optional[str] = None
    target_groups: Optional[int] = None
    min_group_size: Optional[int] = None
    recursive: bool = True
    use_llm: Optional[bool] = None
    workers: Optional[int] = None


class PromoteRequest(BaseModel):
    collection_id: int
    name: Optional[str] = None
    destination_path: Optional[str] = None
    seeds_only: bool = False
    target_group_id: Optional[int] = None


class SeedExpandRequest(BaseModel):
    seed_track_ids: List[int]
    limit: int = 50
    max_distance: Optional[float] = None


class ConfigUpdate(BaseModel):
    values: Dict[str, Any]


# ----------------------------------------------------------------------
# General
# ----------------------------------------------------------------------


@app.get("/")
async def root():
    return {
        "name": "Music Cluster API",
        "version": "2.0.0",
        "workflow": "directed sorting into the groups you already keep",
    }


@app.get("/api/info")
async def get_info():
    """Library-wide summary for the dashboard."""
    config, db = get_context()
    collections = db.list_collections()

    enriched = []
    for collection in collections:
        model = db.load_model(collection["id"])
        enriched.append(
            {
                **collection,
                "fitted": model is not None,
                "accuracy": (model or {}).get("metrics", {}).get("accuracy"),
                "fitted_at": (model or {}).get("fitted_at"),
            }
        )

    return {
        "database_path": config.get_db_path(),
        "track_count": db.count_tracks(),
        "analyzed_count": db.count_features(),
        "collections": enriched,
        "sessions": db.list_sort_sessions()[:10],
        "discovery_runs": db.list_discovery_runs()[:10],
    }


@app.get("/api/browse")
async def browse(path: Optional[str] = None):
    """List subdirectories so the UI can offer a folder picker."""
    target = os.path.abspath(os.path.expanduser(path or "~"))
    if not os.path.isdir(target):
        raise HTTPException(status_code=400, detail=f"Not a directory: {target}")

    entries = []
    try:
        for entry in sorted(os.scandir(target), key=lambda e: e.name.lower()):
            if entry.name.startswith(".") or not entry.is_dir():
                continue
            entries.append({"name": entry.name, "path": entry.path})
    except PermissionError:
        raise HTTPException(status_code=403, detail="Permission denied") from None

    audio_count = len(find_audio_files(target, recursive=False))
    return {
        "path": target,
        "parent": os.path.dirname(target) if target != os.path.dirname(target) else None,
        "directories": entries,
        "audio_file_count": audio_count,
    }


@app.get("/api/config")
async def get_config():
    return Config().config


@app.put("/api/config")
async def update_config(request: ConfigUpdate):
    """Update config values addressed by dotted key paths."""
    config = Config()
    for key, value in request.values.items():
        config.set(key.split("."), value)
    config.save()
    return config.config


# ----------------------------------------------------------------------
# Tracks and media
# ----------------------------------------------------------------------


@app.get("/api/tracks")
async def get_tracks(
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
    query: Optional[str] = None,
    unassigned_in: Optional[int] = Query(None, description="Collection ID"),
):
    """List library tracks."""
    db = get_db()
    tracks = db.list_tracks(
        limit=limit, offset=offset, query=query, unassigned_in_collection=unassigned_in
    )
    return {
        "tracks": [_track_payload(track) for track in tracks],
        "total": db.count_tracks(),
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/tracks/{track_id}")
async def get_track(track_id: int):
    db = get_db()
    track = db.get_track(track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    return {
        **_track_payload(track),
        "groups": db.find_track_groups(track_id),
        "exists": os.path.exists(track["filepath"]),
    }


@app.get("/api/tracks/{track_id}/audio")
async def get_track_audio(track_id: int, request: Request):
    """Stream a track, honouring Range requests so seeking works."""
    db = get_db()
    track = db.get_track(track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")

    filepath = track["filepath"]
    if not os.path.isfile(filepath):
        raise HTTPException(status_code=404, detail="Audio file not found")

    media_type = media_type_for(filepath)
    stat = os.stat(filepath)
    file_size = stat.st_size
    byte_range = parse_range_header(request.headers.get("range"), file_size)

    # Revalidate rather than refetch: scrubbing a track issues a burst of range
    # requests, and re-downloading audio the browser already has is what makes
    # seeking feel slow. The tag changes as soon as the file does.
    etag = f'"{int(stat.st_mtime_ns)}-{file_size}"'

    if byte_range is None:
        return FileResponse(
            filepath,
            media_type=media_type,
            headers={"Accept-Ranges": "bytes", "Cache-Control": "no-cache", "ETag": etag},
        )

    start, end = byte_range
    length = end - start + 1

    def stream(chunk_size: int = 64 * 1024):
        with open(filepath, "rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining > 0:
                chunk = handle.read(min(chunk_size, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(
        stream(),
        status_code=206,
        media_type=media_type,
        headers={
            "Content-Range": f"bytes {start}-{end}/{file_size}",
            "Content-Length": str(length),
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-cache",
            "ETag": etag,
        },
    )


# Decoding audio is CPU-bound. This endpoint is deliberately synchronous so
# FastAPI runs it in a worker thread instead of on the event loop, where a
# single long track used to stall every other request — including the audio
# stream feeding the player. The semaphore then keeps a screenful of waveform
# requests from taking every core at once.
_waveform_slots = threading.Semaphore(max(2, min(4, (os.cpu_count() or 2))))


@app.get("/api/tracks/{track_id}/waveform")
def get_track_waveform(
    track_id: int,
    request: Request,
    samples: int = Query(240, ge=16, le=2000),
):
    """Envelope for drawing a track's waveform, cached between requests."""
    config, db = get_context()
    track = db.get_track(track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if not os.path.isfile(track["filepath"]):
        raise HTTPException(status_code=404, detail="Audio file not found")

    try:
        with _waveform_slots:
            payload, key = cached_waveform(
                track["filepath"], samples=samples, cache_dir=config.get_cache_dir()
            )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Waveform generation failed for track %s", track_id)
        raise HTTPException(status_code=422, detail=f"Could not read audio: {exc}") from exc

    etag = f'"{key}"'
    headers = {"ETag": etag, "Cache-Control": "private, max-age=86400"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return JSONResponse(payload, headers=headers)


@app.get("/api/tracks/{track_id}/artwork")
async def get_track_artwork(track_id: int):
    db = get_db()
    track = db.get_track(track_id)
    if not track:
        raise HTTPException(status_code=404, detail="Track not found")
    if not os.path.isfile(track["filepath"]):
        raise HTTPException(status_code=404, detail="Audio file not found")

    artwork = extract_artwork(track["filepath"])
    if not artwork:
        return Response(status_code=204)
    return artwork


@app.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    """Analyse files without sorting them."""

    def worker(update, **_):
        config, db = get_context()
        result = analyze_paths(
            db,
            config,
            request.paths,
            recursive=request.recursive,
            update=request.update,
            workers=request.workers,
            progress=progress_updater(update, "Analysing"),
        )
        return result.to_dict()

    return {"task_id": start_task("analyze", worker)}


@app.post("/api/prune")
async def prune():
    removed = prune_missing(get_db())
    return {"removed": len(removed), "track_ids": removed}


# ----------------------------------------------------------------------
# Collections
# ----------------------------------------------------------------------


@app.get("/api/collections")
async def list_collections():
    db = get_db()
    return {"collections": db.list_collections()}


@app.post("/api/collections")
async def create_collection(request: CollectionRequest):
    db = get_db()
    if db.get_collection(name=request.name):
        raise HTTPException(status_code=409, detail="A collection with that name already exists")
    collection_id = db.create_collection(request.name, request.description)
    return db.get_collection(collection_id=collection_id)


@app.get("/api/collections/{collection_id}")
async def get_collection(collection_id: int):
    db = get_db()
    collection = _require_collection(db, collection_id)
    model = db.load_model(collection_id)
    return {
        "collection": collection,
        "groups": db.list_groups(collection_id),
        "model": _model_payload(model),
    }


@app.put("/api/collections/{collection_id}")
async def update_collection(collection_id: int, request: CollectionUpdate):
    db = get_db()
    _require_collection(db, collection_id)
    db.update_collection(collection_id, **request.model_dump(exclude_none=True))
    return db.get_collection(collection_id=collection_id)


@app.delete("/api/collections/{collection_id}")
async def delete_collection(collection_id: int):
    db = get_db()
    _require_collection(db, collection_id)
    db.delete_collection(collection_id)
    return {"deleted": True}


@app.post("/api/collections/{collection_id}/fit")
async def fit_collection(collection_id: int):
    """Fit the sorter and report how separable the groups are."""
    config, db = get_context()
    collection = _require_collection(db, collection_id)

    def worker(update, **_):
        update(message="Fitting reference groups...")
        return groups_mod.fit_collection(db, collection, config)

    return {"task_id": start_task("fit", worker)}


@app.get("/api/collections/{collection_id}/check")
async def check_collection(collection_id: int):
    """Latest self-check metrics for a collection."""
    db = get_db()
    _require_collection(db, collection_id)
    model = db.load_model(collection_id)
    if not model:
        raise HTTPException(status_code=404, detail="Collection has not been fitted yet")
    return {"metrics": model["metrics"], "fitted_at": model["fitted_at"], "params": model["params"]}


# ----------------------------------------------------------------------
# Groups
# ----------------------------------------------------------------------


@app.get("/api/collections/{collection_id}/groups")
async def list_groups(collection_id: int):
    db = get_db()
    _require_collection(db, collection_id)
    return {"groups": db.list_groups(collection_id)}


@app.post("/api/collections/{collection_id}/groups")
async def create_group(collection_id: int, request: GroupRequest):
    config, db = get_context()
    collection = _require_collection(db, collection_id)
    group = groups_mod.create_group(
        db,
        collection["id"],
        request.name,
        description=request.description,
        destination_path=request.destination_path,
        color=request.color,
    )
    return group


@app.get("/api/groups/{group_id}")
async def get_group(group_id: int, limit: int = Query(200, ge=1, le=2000), offset: int = 0):
    db = get_db()
    group = db.get_group(group_id=group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    member_ids = db.get_group_member_ids(group_id)
    model = db.load_model(group["collection_id"])
    per_group = {
        entry["group_id"]: entry
        for entry in (model["metrics"].get("per_group", []) if model else [])
    }

    return {
        "group": group,
        "size": len(member_ids),
        "members": [
            _track_payload(track) for track in db.get_group_members(group_id, limit, offset)
        ],
        "quality": per_group.get(group_id),
    }


@app.put("/api/groups/{group_id}")
async def update_group(group_id: int, request: GroupUpdate):
    db = get_db()
    group = db.get_group(group_id=group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    db.update_group(group_id, **request.model_dump(exclude_none=True))
    db.touch_collection(group["collection_id"])
    return db.get_group(group_id=group_id)


@app.delete("/api/groups/{group_id}")
async def delete_group(group_id: int):
    db = get_db()
    group = db.get_group(group_id=group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    db.delete_group(group_id)
    db.touch_collection(group["collection_id"])
    return {"deleted": True}


@app.post("/api/groups/{group_id}/members")
async def add_group_members(group_id: int, request: AddMembersRequest):
    """Add reference tracks by ID (already in the library) or by path."""
    config, db = get_context()
    group = db.get_group(group_id=group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")

    added = 0
    if request.track_ids:
        existing = set(db.get_group_member_ids(group_id))
        fresh = [tid for tid in request.track_ids if tid not in existing]
        added += db.add_group_members(group_id, fresh, role=request.role)

    if request.filepaths:
        report = groups_mod.add_tracks_to_group(
            db, group, request.filepaths, config, role=request.role
        )
        added += report.added

    if added:
        db.touch_collection(group["collection_id"])
    return {"added": added, "size": len(db.get_group_member_ids(group_id))}


@app.delete("/api/groups/{group_id}/members/{track_id}")
async def remove_group_member(group_id: int, track_id: int):
    db = get_db()
    group = db.get_group(group_id=group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Group not found")
    removed = db.remove_group_member(group_id, track_id)
    if removed:
        db.touch_collection(group["collection_id"])
    return {"removed": removed}


@app.post("/api/collections/{collection_id}/import-folder")
async def import_folder(collection_id: int, request: ImportRequest):
    """Import one folder as a group."""
    config, db = get_context()
    collection = _require_collection(db, collection_id)

    def worker(update, **_):
        report = groups_mod.import_folder_as_group(
            db,
            collection["id"],
            request.path,
            config,
            name=request.name,
            recursive=request.recursive,
            set_destination=request.destination_path is None,
            workers=request.workers,
            progress=progress_updater(update, "Analysing"),
        )
        if request.destination_path:
            db.update_group(report.group_id, destination_path=request.destination_path)
        return report.to_dict()

    return {"task_id": start_task("import-folder", worker)}


@app.post("/api/collections/{collection_id}/import-tree")
async def import_tree(collection_id: int, request: ImportTreeRequest):
    """Import every subfolder of a directory as its own group."""
    config, db = get_context()
    collection = _require_collection(db, collection_id)

    def worker(update, **_):
        reports = groups_mod.import_folders_as_groups(
            db,
            collection["id"],
            request.path,
            config,
            recursive=request.recursive,
            min_tracks=request.min_tracks,
            workers=request.workers,
            progress=progress_updater(update, "Analysing"),
        )
        return {"groups": [report.to_dict() for report in reports]}

    return {"task_id": start_task("import-tree", worker)}


@app.post("/api/collections/{collection_id}/import-playlist")
async def import_playlist(collection_id: int, request: ImportRequest):
    """Import a playlist file as a group."""
    config, db = get_context()
    collection = _require_collection(db, collection_id)

    def worker(update, **_):
        report = groups_mod.import_playlist_as_group(
            db,
            collection["id"],
            request.path,
            config,
            name=request.name,
            destination_path=request.destination_path,
            workers=request.workers,
            progress=progress_updater(update, "Analysing"),
        )
        return report.to_dict()

    return {"task_id": start_task("import-playlist", worker)}


@app.post("/api/collections/{collection_id}/export")
async def export_groups(
    collection_id: int, output: str = Query(...), playlist_format: str = "m3u8"
):
    """Write one playlist per group."""
    db = get_db()
    _require_collection(db, collection_id)
    written = organizer.export_groups_as_playlists(
        db, db.list_groups(collection_id), output, playlist_format=playlist_format
    )
    return {"playlists": written}


# ----------------------------------------------------------------------
# Sort sessions
# ----------------------------------------------------------------------


@app.get("/api/sessions")
async def list_sessions(collection_id: Optional[int] = None):
    db = get_db()
    return {"sessions": db.list_sort_sessions(collection_id)}


@app.post("/api/sessions")
async def create_session(request: SortRequest):
    """Analyse a batch of new music and score it against a collection."""
    config, db = get_context()
    collection = _require_collection(db, request.collection_id)

    def worker(update, **_):
        outcome = sorting.create_session(
            db,
            config,
            collection,
            request.paths,
            name=request.name,
            recursive=request.recursive,
            workers=request.workers,
            progress=progress_updater(update, "Analysing"),
            auto_accept=request.auto_accept,
        )
        return outcome

    return {"task_id": start_task("sort", worker)}


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: int):
    db = get_db()
    try:
        return sorting.session_overview(db, session_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: int):
    db = get_db()
    if not db.get_sort_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    db.delete_sort_session(session_id)
    return {"deleted": True}


@app.get("/api/sessions/{session_id}/items")
async def get_session_items(
    session_id: int,
    status: Optional[str] = None,
    limit: int = Query(100, ge=1, le=1000),
    offset: int = Query(0, ge=0),
):
    """Review queue for a session."""
    db = get_db()
    session = db.get_sort_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    items = db.list_sort_items(session_id, status=status, limit=limit, offset=offset)
    for item in items:
        item["display"] = display_name(item)
    return {
        "items": items,
        "summary": db.sort_session_summary(session_id),
        "groups": db.list_groups(session["collection_id"]),
    }


@app.post("/api/sessions/{session_id}/rescore")
async def rescore_session(session_id: int, auto_accept: bool = False):
    """Re-score pending items against the collection's current groups."""
    config, db = get_context()
    if not db.get_sort_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    def worker(update, **_):
        update(message="Re-scoring...")
        return sorting.score_session(db, config, session_id, auto_accept=auto_accept)

    return {"task_id": start_task("rescore", worker)}


@app.put("/api/items/{item_id}")
async def decide_item(item_id: int, request: DecisionRequest):
    """Record a decision for one item."""
    db = get_db()
    try:
        return sorting.decide(db, item_id, group_id=request.group_id, skip=request.skip)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.put("/api/sessions/{session_id}/items")
async def decide_items(session_id: int, request: BulkDecisionRequest):
    """Record the same decision for several items."""
    db = get_db()
    if not db.get_sort_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    for item_id in request.item_ids:
        sorting.decide(db, item_id, group_id=request.group_id, skip=request.skip)
    return {"updated": len(request.item_ids), "summary": db.sort_session_summary(session_id)}


@app.post("/api/sessions/{session_id}/accept-confident")
async def accept_confident(session_id: int):
    """Accept every pending suggestion that cleared the auto threshold."""
    config, db = get_context()
    if not db.get_sort_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    accepted = sorting.accept_all_confident(db, config, session_id)
    return {"accepted": accepted, "summary": db.sort_session_summary(session_id)}


@app.post("/api/sessions/{session_id}/commit")
async def commit_session(session_id: int, request: CommitRequest):
    """Preview or apply a session's decisions."""
    config, db = get_context()
    if not db.get_sort_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    try:
        if request.dry_run:
            plan = sorting.plan_session_commit(
                db,
                config,
                session_id,
                mode=request.mode,
                playlist_dir=request.playlist_dir,
                on_conflict=request.on_conflict,
            )
            return {"dry_run": True, "plan": plan.to_dict()}

        return sorting.commit_session(
            db,
            config,
            session_id,
            mode=request.mode,
            playlist_dir=request.playlist_dir,
            on_conflict=request.on_conflict,
            learn=request.learn,
        )
    except (LookupError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/sessions/{session_id}/undo")
async def undo_session(session_id: int, remove_playlists: bool = False):
    """Reverse a session's file operations."""
    db = get_db()
    if not db.get_sort_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    return organizer.undo_session(db, session_id, remove_playlists=remove_playlists)


# ----------------------------------------------------------------------
# Discovery
# ----------------------------------------------------------------------


@app.get("/api/discovery")
async def list_discovery_runs():
    db = get_db()
    return {"runs": db.list_discovery_runs()}


@app.post("/api/discovery")
async def create_discovery_run(request: DiscoverRequest):
    """Propose candidate groups from an unsorted pile."""
    config, db = get_context()

    def worker(update, **_):
        return discovery_mod.run_discovery(
            db,
            config,
            paths=request.paths or None,
            track_ids=request.track_ids or None,
            name=request.name,
            algorithm=request.algorithm,
            granularity=request.granularity,
            target_groups=request.target_groups,
            min_group_size=request.min_group_size,
            recursive=request.recursive,
            workers=request.workers,
            progress=progress_updater(update, "Analysing"),
            use_llm=request.use_llm,
        )

    return {"task_id": start_task("discover", worker)}


@app.get("/api/discovery/{run_id}")
async def get_discovery_run(run_id: int):
    """A run with its candidates and exemplar tracks."""
    db = get_db()
    run = db.get_discovery_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Discovery run not found")

    candidates = db.list_discovery_candidates(run_id)
    for candidate in candidates:
        tracks = db.get_tracks(candidate["exemplars"])
        candidate["exemplar_tracks"] = [
            _track_payload(tracks[tid]) for tid in candidate["exemplars"] if tid in tracks
        ]
    return {"run": run, "candidates": candidates}


@app.get("/api/discovery/candidates/{candidate_id}/members")
async def get_candidate_members(
    candidate_id: int, limit: int = Query(100, ge=1, le=1000), offset: int = 0
):
    db = get_db()
    candidate = db.get_discovery_candidate(candidate_id)
    if not candidate:
        raise HTTPException(status_code=404, detail="Candidate not found")

    member_ids = db.get_discovery_member_ids(candidate_id)
    page = member_ids[offset : offset + limit]
    tracks = db.get_tracks(page)
    return {
        "total": len(member_ids),
        "tracks": [_track_payload(tracks[tid]) for tid in page if tid in tracks],
    }


@app.post("/api/discovery/candidates/{candidate_id}/promote")
async def promote_candidate(candidate_id: int, request: PromoteRequest):
    """Turn a candidate into a real group, or merge it into an existing one."""
    config, db = get_context()
    collection = _require_collection(db, request.collection_id)
    try:
        return discovery_mod.promote_candidate(
            db,
            config,
            collection,
            candidate_id,
            name=request.name,
            destination_path=request.destination_path,
            seeds_only=request.seeds_only,
            target_group_id=request.target_group_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/api/discovery/candidates/{candidate_id}/reject")
async def reject_candidate(candidate_id: int):
    db = get_db()
    if not db.get_discovery_candidate(candidate_id):
        raise HTTPException(status_code=404, detail="Candidate not found")
    discovery_mod.reject_candidate(db, candidate_id)
    return {"rejected": True}


@app.delete("/api/discovery/{run_id}")
async def delete_discovery_run(run_id: int):
    db = get_db()
    if not db.get_discovery_run(run_id):
        raise HTTPException(status_code=404, detail="Discovery run not found")
    db.delete_discovery_run(run_id)
    return {"deleted": True}


@app.post("/api/similar")
async def find_similar(request: SeedExpandRequest):
    """Find library tracks that sound like the given seeds."""
    config, db = get_context()
    try:
        results = discovery_mod.expand_seeds(
            db,
            config,
            request.seed_track_ids,
            limit=request.limit,
            max_distance=request.max_distance,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {"results": [{**result, "track": _track_payload(result["track"])} for result in results]}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _require_collection(db: Database, collection_id: int) -> Dict:
    collection = db.get_collection(collection_id=collection_id)
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    return collection


def _track_payload(track: Dict) -> Dict:
    return {**track, "display": display_name(track)}


def _model_payload(model: Optional[Dict]) -> Optional[Dict]:
    """Model metadata without the pickled payload."""
    if not model:
        return None
    return {key: value for key, value in model.items() if key != "payload"}
