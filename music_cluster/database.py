"""SQLite persistence for music-cluster.

The schema is built around *directed* sorting: a collection holds the groups a
DJ already sorts into, each group holds reference tracks, and a sort session
records suggestions and human decisions for a batch of incoming music.
"""

import json
import os
import pickle
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


SCHEMA_VERSION = 2


def utc_now() -> str:
    """Timestamp string matching SQLite's own CURRENT_TIMESTAMP format.

    Columns mix defaults (UTC, via CURRENT_TIMESTAMP) with values written
    from Python, and staleness checks compare them as strings, so both sides
    must be UTC in the same layout. The fractional part extends SQLite's own
    format rather than diverging from it, so the two still order correctly
    against each other — and it keeps two changes in the same second apart.
    """
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


TRACK_METADATA_COLUMNS = (
    "title",
    "artist",
    "album",
    "genre",
    "year",
    "tag_bpm",
    "tag_key",
    "comment",
)


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        # The CLI's `init` command creates this directory, but the desktop app
        # has no such step: the first request a fresh install serves would
        # otherwise fail with "unable to open database file".
        directory = os.path.dirname(os.path.abspath(db_path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._init_database()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_database(self) -> None:
        self._migrate_legacy()
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.executescript(
                """
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filepath TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    duration REAL,
                    file_size INTEGER,
                    checksum TEXT,
                    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    analysis_version TEXT,
                    title TEXT,
                    artist TEXT,
                    album TEXT,
                    genre TEXT,
                    year INTEGER,
                    tag_bpm REAL,
                    tag_key TEXT,
                    comment TEXT
                );

                CREATE TABLE IF NOT EXISTS features (
                    track_id INTEGER PRIMARY KEY,
                    feature_vector BLOB NOT NULL,
                    feature_dim INTEGER,
                    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS collections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    description TEXT,
                    settings TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS groups (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    description TEXT,
                    color TEXT,
                    destination_path TEXT,
                    source_kind TEXT DEFAULT 'manual',
                    source_path TEXT,
                    position INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE,
                    UNIQUE(collection_id, name)
                );

                CREATE TABLE IF NOT EXISTS group_members (
                    group_id INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    role TEXT DEFAULT 'reference',
                    weight REAL DEFAULT 1.0,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (group_id, track_id),
                    FOREIGN KEY (group_id) REFERENCES groups(id) ON DELETE CASCADE,
                    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS models (
                    collection_id INTEGER PRIMARY KEY,
                    payload BLOB NOT NULL,
                    params TEXT,
                    metrics TEXT,
                    n_groups INTEGER,
                    n_tracks INTEGER,
                    fitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS sort_sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    collection_id INTEGER NOT NULL,
                    name TEXT,
                    source_path TEXT,
                    status TEXT DEFAULT 'open',
                    settings TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    committed_at TIMESTAMP,
                    FOREIGN KEY (collection_id) REFERENCES collections(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS sort_items (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    suggested_group_id INTEGER,
                    confidence REAL,
                    margin REAL,
                    distance REAL,
                    ranked TEXT,
                    status TEXT DEFAULT 'pending',
                    final_group_id INTEGER,
                    decided_at TIMESTAMP,
                    applied INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES sort_sessions(id) ON DELETE CASCADE,
                    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
                    UNIQUE(session_id, track_id)
                );

                CREATE TABLE IF NOT EXISTS file_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id INTEGER,
                    track_id INTEGER,
                    group_id INTEGER,
                    action TEXT NOT NULL,
                    source_path TEXT,
                    dest_path TEXT,
                    performed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    undone INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS discovery_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    source_path TEXT,
                    algorithm TEXT,
                    params TEXT,
                    metrics TEXT,
                    status TEXT DEFAULT 'open',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS discovery_candidates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER NOT NULL,
                    candidate_index INTEGER NOT NULL,
                    suggested_name TEXT,
                    size INTEGER,
                    centroid BLOB,
                    exemplars TEXT,
                    stats TEXT,
                    status TEXT DEFAULT 'pending',
                    promoted_group_id INTEGER,
                    FOREIGN KEY (run_id) REFERENCES discovery_runs(id) ON DELETE CASCADE,
                    UNIQUE(run_id, candidate_index)
                );

                CREATE TABLE IF NOT EXISTS discovery_members (
                    candidate_id INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    distance REAL,
                    PRIMARY KEY (candidate_id, track_id),
                    FOREIGN KEY (candidate_id) REFERENCES discovery_candidates(id) ON DELETE CASCADE,
                    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS schema_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_tracks_filepath ON tracks(filepath);
                CREATE INDEX IF NOT EXISTS idx_tracks_artist ON tracks(artist);
                CREATE INDEX IF NOT EXISTS idx_group_members_track ON group_members(track_id);
                CREATE INDEX IF NOT EXISTS idx_groups_collection ON groups(collection_id);
                CREATE INDEX IF NOT EXISTS idx_sort_items_session ON sort_items(session_id);
                CREATE INDEX IF NOT EXISTS idx_file_actions_session ON file_actions(session_id);
                """
            )

            cursor.execute(
                "INSERT OR REPLACE INTO schema_meta (key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),),
            )
            conn.commit()

    def _migrate_legacy(self) -> None:
        """Upgrade a v1 (clustering-era) database in place.

        v1 stored feature vectors as pickles and organised everything around
        anonymous clusterings. Analysis is expensive, so the vectors are
        converted rather than discarded; the clustering tables are dropped.
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS schema_meta (key TEXT PRIMARY KEY, value TEXT)"
            )
            row = cursor.execute("SELECT value FROM schema_meta WHERE key = 'version'").fetchone()
            version = int(row[0]) if row else None

            tables = {
                r[0]
                for r in cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            if version is not None or "features" not in tables:
                # Already on the current schema, or a brand new database.
                return

            rows = cursor.execute("SELECT track_id, feature_vector FROM features").fetchall()
            converted = []
            for track_id, blob in rows:
                try:
                    vector = np.asarray(pickle.loads(blob), dtype=np.float64)
                except Exception:
                    continue
                converted.append((vector.tobytes(), len(vector), track_id))
            cursor.executemany(
                "UPDATE features SET feature_vector = ?, feature_dim = ? WHERE track_id = ?",
                converted,
            )

            for legacy in ("cluster_members", "clusters", "clusterings"):
                cursor.execute(f"DROP TABLE IF EXISTS {legacy}")

            # v1 had no tag metadata. These columns must exist before the main
            # schema script runs, because it indexes one of them.
            existing = {row[1] for row in cursor.execute("PRAGMA table_info(tracks)").fetchall()}
            for column, sql_type in (
                ("title", "TEXT"),
                ("artist", "TEXT"),
                ("album", "TEXT"),
                ("genre", "TEXT"),
                ("year", "INTEGER"),
                ("tag_bpm", "REAL"),
                ("tag_key", "TEXT"),
                ("comment", "TEXT"),
            ):
                if column not in existing:
                    cursor.execute(f"ALTER TABLE tracks ADD COLUMN {column} {sql_type}")

            conn.commit()

    @contextmanager
    def connection(self):
        """Connection context manager with foreign keys enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Tracks
    # ------------------------------------------------------------------

    def upsert_track(
        self,
        filepath: str,
        filename: str,
        duration: Optional[float] = None,
        file_size: Optional[int] = None,
        checksum: Optional[str] = None,
        analysis_version: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Insert or update a track, preserving its ID (and group memberships)."""
        metadata = metadata or {}
        values = [metadata.get(column) for column in TRACK_METADATA_COLUMNS]

        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM tracks WHERE filepath = ?", (filepath,))
            row = cursor.fetchone()
            assignments = ", ".join(f"{column} = ?" for column in TRACK_METADATA_COLUMNS)

            if row:
                track_id = row[0]
                cursor.execute(
                    f"""
                    UPDATE tracks SET filename = ?, duration = ?, file_size = ?,
                        checksum = ?, analysis_version = ?, analyzed_at = ?, {assignments}
                    WHERE id = ?
                    """,
                    [
                        filename,
                        duration,
                        file_size,
                        checksum,
                        analysis_version,
                        utc_now(),
                        *values,
                        track_id,
                    ],
                )
            else:
                columns = ", ".join(TRACK_METADATA_COLUMNS)
                placeholders = ", ".join("?" for _ in TRACK_METADATA_COLUMNS)
                cursor.execute(
                    f"""
                    INSERT INTO tracks
                        (filepath, filename, duration, file_size, checksum,
                         analysis_version, analyzed_at, {columns})
                    VALUES (?, ?, ?, ?, ?, ?, ?, {placeholders})
                    """,
                    [
                        filepath,
                        filename,
                        duration,
                        file_size,
                        checksum,
                        analysis_version,
                        utc_now(),
                        *values,
                    ],
                )
                track_id = cursor.lastrowid
            conn.commit()
            return track_id

    def get_track_by_filepath(self, filepath: str) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM tracks WHERE filepath = ?", (filepath,)).fetchone()
            return dict(row) if row else None

    def get_track(self, track_id: int) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM tracks WHERE id = ?", (track_id,)).fetchone()
            return dict(row) if row else None

    def get_tracks(self, track_ids: Sequence[int]) -> Dict[int, Dict]:
        """Fetch many tracks at once, keyed by ID."""
        if not track_ids:
            return {}
        result: Dict[int, Dict] = {}
        ids = list(track_ids)
        with self.connection() as conn:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT * FROM tracks WHERE id IN ({placeholders})", chunk
                ).fetchall()
                for row in rows:
                    result[row["id"]] = dict(row)
        return result

    def list_tracks(
        self,
        limit: int = 100,
        offset: int = 0,
        query: Optional[str] = None,
        unassigned_in_collection: Optional[int] = None,
    ) -> List[Dict]:
        """List tracks, optionally filtered by text or by lack of group membership."""
        clauses: List[str] = []
        params: List[Any] = []

        if query:
            like = f"%{query}%"
            clauses.append(
                "(t.filename LIKE ? OR t.title LIKE ? OR t.artist LIKE ? OR"
                " t.album LIKE ? OR t.genre LIKE ?)"
            )
            params.extend([like] * 5)

        if unassigned_in_collection is not None:
            clauses.append(
                "t.id NOT IN (SELECT gm.track_id FROM group_members gm"
                " JOIN groups g ON gm.group_id = g.id WHERE g.collection_id = ?)"
            )
            params.append(unassigned_in_collection)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])

        with self.connection() as conn:
            rows = conn.execute(
                f"SELECT t.* FROM tracks t {where} ORDER BY t.artist, t.title, t.filename"
                " LIMIT ? OFFSET ?",
                params,
            ).fetchall()
            return [dict(row) for row in rows]

    def count_tracks(self) -> int:
        with self.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM tracks").fetchone()[0]

    def delete_track(self, track_id: int) -> bool:
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM tracks WHERE id = ?", (track_id,))
            conn.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Features
    # ------------------------------------------------------------------

    def add_features(self, track_id: int, feature_vector: np.ndarray) -> None:
        vector = np.asarray(feature_vector, dtype=np.float64)
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO features (track_id, feature_vector, feature_dim)
                VALUES (?, ?, ?)
                """,
                (track_id, vector.tobytes(), len(vector)),
            )
            conn.commit()

    def get_features(self, track_id: int) -> Optional[np.ndarray]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT feature_vector FROM features WHERE track_id = ?", (track_id,)
            ).fetchone()
            return np.frombuffer(row[0], dtype=np.float64).copy() if row else None

    def get_feature_matrix(self, track_ids: Sequence[int]) -> Tuple[np.ndarray, List[int]]:
        """Return features for the given tracks, dropping any without features.

        Row order matches the returned ID list, not the requested order.
        """
        if not track_ids:
            return np.empty((0, 0)), []

        vectors: List[np.ndarray] = []
        found: List[int] = []
        ids = list(track_ids)
        with self.connection() as conn:
            for start in range(0, len(ids), 500):
                chunk = ids[start : start + 500]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT track_id, feature_vector FROM features WHERE track_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    vectors.append(np.frombuffer(row[1], dtype=np.float64))
                    found.append(row[0])

        return _stack(vectors, found)

    def get_all_features(self) -> Tuple[np.ndarray, List[int]]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT track_id, feature_vector FROM features ORDER BY track_id"
            ).fetchall()
        return _stack(
            [np.frombuffer(row[1], dtype=np.float64) for row in rows],
            [row[0] for row in rows],
        )

    def count_features(self) -> int:
        with self.connection() as conn:
            return conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]

    def tracks_without_features(self, track_ids: Sequence[int]) -> List[int]:
        if not track_ids:
            return []
        with self.connection() as conn:
            placeholders = ",".join("?" for _ in track_ids)
            rows = conn.execute(
                f"SELECT track_id FROM features WHERE track_id IN ({placeholders})",
                list(track_ids),
            ).fetchall()
        have = {row[0] for row in rows}
        return [tid for tid in track_ids if tid not in have]

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    def create_collection(
        self,
        name: str,
        description: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                "INSERT INTO collections (name, description, settings) VALUES (?, ?, ?)",
                (name, description, json.dumps(settings or {})),
            )
            conn.commit()
            return cursor.lastrowid

    def get_collection(
        self, collection_id: Optional[int] = None, name: Optional[str] = None
    ) -> Optional[Dict]:
        with self.connection() as conn:
            if collection_id is not None:
                row = conn.execute(
                    "SELECT * FROM collections WHERE id = ?", (collection_id,)
                ).fetchone()
            elif name is not None:
                row = conn.execute("SELECT * FROM collections WHERE name = ?", (name,)).fetchone()
            else:
                row = conn.execute(
                    "SELECT * FROM collections ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()
        return _with_json(row, "settings") if row else None

    def list_collections(self) -> List[Dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT c.*,
                    (SELECT COUNT(*) FROM groups g WHERE g.collection_id = c.id) AS group_count,
                    (SELECT COUNT(*) FROM group_members gm JOIN groups g
                        ON gm.group_id = g.id WHERE g.collection_id = c.id) AS track_count
                FROM collections c ORDER BY c.name
                """
            ).fetchall()
        return [_with_json(row, "settings") for row in rows]

    def update_collection(self, collection_id: int, **fields: Any) -> None:
        allowed = {"name", "description", "settings"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return
        if "settings" in updates and not isinstance(updates["settings"], str):
            updates["settings"] = json.dumps(updates["settings"])
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connection() as conn:
            conn.execute(
                f"UPDATE collections SET {assignments}, updated_at = ? WHERE id = ?",
                [*list(updates.values()), utc_now(), collection_id],
            )
            conn.commit()

    def delete_collection(self, collection_id: int) -> bool:
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM collections WHERE id = ?", (collection_id,))
            conn.commit()
            return cursor.rowcount > 0

    def touch_collection(self, collection_id: int) -> None:
        """Mark a collection as changed so stale fitted models are detectable."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE collections SET updated_at = ? WHERE id = ?",
                (utc_now(), collection_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Groups
    # ------------------------------------------------------------------

    def create_group(
        self,
        collection_id: int,
        name: str,
        description: Optional[str] = None,
        color: Optional[str] = None,
        destination_path: Optional[str] = None,
        source_kind: str = "manual",
        source_path: Optional[str] = None,
        position: Optional[int] = None,
    ) -> int:
        with self.connection() as conn:
            if position is None:
                row = conn.execute(
                    "SELECT COALESCE(MAX(position), -1) + 1 FROM groups WHERE collection_id = ?",
                    (collection_id,),
                ).fetchone()
                position = row[0]
            cursor = conn.execute(
                """
                INSERT INTO groups
                    (collection_id, name, description, color, destination_path,
                     source_kind, source_path, position)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    collection_id,
                    name,
                    description,
                    color,
                    destination_path,
                    source_kind,
                    source_path,
                    position,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_group(
        self,
        group_id: Optional[int] = None,
        collection_id: Optional[int] = None,
        name: Optional[str] = None,
    ) -> Optional[Dict]:
        with self.connection() as conn:
            if group_id is not None:
                row = conn.execute("SELECT * FROM groups WHERE id = ?", (group_id,)).fetchone()
            elif collection_id is not None and name is not None:
                row = conn.execute(
                    "SELECT * FROM groups WHERE collection_id = ? AND name = ?",
                    (collection_id, name),
                ).fetchone()
            else:
                raise ValueError("Provide group_id or (collection_id, name)")
        return dict(row) if row else None

    def list_groups(self, collection_id: int) -> List[Dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT g.*,
                    (SELECT COUNT(*) FROM group_members gm WHERE gm.group_id = g.id) AS size
                FROM groups g WHERE g.collection_id = ?
                ORDER BY g.position, g.name
                """,
                (collection_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def update_group(self, group_id: int, **fields: Any) -> None:
        allowed = {"name", "description", "color", "destination_path", "position", "source_path"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connection() as conn:
            conn.execute(
                f"UPDATE groups SET {assignments}, updated_at = ? WHERE id = ?",
                [*list(updates.values()), utc_now(), group_id],
            )
            conn.commit()

    def delete_group(self, group_id: int) -> bool:
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM groups WHERE id = ?", (group_id,))
            conn.commit()
            return cursor.rowcount > 0

    def add_group_members(
        self, group_id: int, track_ids: Iterable[int], role: str = "reference"
    ) -> int:
        rows = [(group_id, track_id, role) for track_id in track_ids]
        if not rows:
            return 0
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO group_members (group_id, track_id, role)
                VALUES (?, ?, ?)
                ON CONFLICT(group_id, track_id) DO UPDATE SET role = excluded.role
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def remove_group_member(self, group_id: int, track_id: int) -> bool:
        with self.connection() as conn:
            cursor = conn.execute(
                "DELETE FROM group_members WHERE group_id = ? AND track_id = ?",
                (group_id, track_id),
            )
            conn.commit()
            return cursor.rowcount > 0

    def get_group_member_ids(self, group_id: int) -> List[int]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT track_id FROM group_members WHERE group_id = ? ORDER BY added_at",
                (group_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def get_group_members(self, group_id: int, limit: int = 500, offset: int = 0) -> List[Dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT t.*, gm.role, gm.added_at AS member_since
                FROM group_members gm JOIN tracks t ON gm.track_id = t.id
                WHERE gm.group_id = ?
                ORDER BY t.artist, t.title, t.filename
                LIMIT ? OFFSET ?
                """,
                (group_id, limit, offset),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_collection_membership(self, collection_id: int) -> Dict[int, List[int]]:
        """Map ``group_id -> [track_id, ...]`` for a whole collection."""
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT g.id AS group_id, gm.track_id
                FROM groups g LEFT JOIN group_members gm ON gm.group_id = g.id
                WHERE g.collection_id = ?
                ORDER BY g.position, g.id
                """,
                (collection_id,),
            ).fetchall()
        membership: Dict[int, List[int]] = {}
        for row in rows:
            membership.setdefault(row["group_id"], [])
            if row["track_id"] is not None:
                membership[row["group_id"]].append(row["track_id"])
        return membership

    def find_track_groups(self, track_id: int, collection_id: Optional[int] = None) -> List[Dict]:
        """Groups a track already belongs to (used to warn about duplicates)."""
        sql = """
            SELECT g.* FROM group_members gm JOIN groups g ON gm.group_id = g.id
            WHERE gm.track_id = ?
        """
        params: List[Any] = [track_id]
        if collection_id is not None:
            sql += " AND g.collection_id = ?"
            params.append(collection_id)
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Fitted models
    # ------------------------------------------------------------------

    def save_model(
        self,
        collection_id: int,
        payload: bytes,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
        n_groups: int,
        n_tracks: int,
    ) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO models
                    (collection_id, payload, params, metrics, n_groups, n_tracks, fitted_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    collection_id,
                    payload,
                    json.dumps(params),
                    json.dumps(metrics),
                    n_groups,
                    n_tracks,
                    utc_now(),
                ),
            )
            conn.commit()

    def load_model(self, collection_id: int) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM models WHERE collection_id = ?", (collection_id,)
            ).fetchone()
        if not row:
            return None
        model = dict(row)
        model["params"] = json.loads(model["params"] or "{}")
        model["metrics"] = json.loads(model["metrics"] or "{}")
        return model

    def delete_model(self, collection_id: int) -> None:
        with self.connection() as conn:
            conn.execute("DELETE FROM models WHERE collection_id = ?", (collection_id,))
            conn.commit()

    # ------------------------------------------------------------------
    # Sort sessions
    # ------------------------------------------------------------------

    def create_sort_session(
        self,
        collection_id: int,
        name: Optional[str],
        source_path: Optional[str],
        settings: Optional[Dict[str, Any]] = None,
    ) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO sort_sessions (collection_id, name, source_path, settings)
                VALUES (?, ?, ?, ?)
                """,
                (collection_id, name, source_path, json.dumps(settings or {})),
            )
            conn.commit()
            return cursor.lastrowid

    def get_sort_session(self, session_id: int) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM sort_sessions WHERE id = ?", (session_id,)).fetchone()
        return _with_json(row, "settings") if row else None

    def list_sort_sessions(self, collection_id: Optional[int] = None) -> List[Dict]:
        sql = """
            SELECT s.*, c.name AS collection_name,
                (SELECT COUNT(*) FROM sort_items i WHERE i.session_id = s.id) AS item_count,
                (SELECT COUNT(*) FROM sort_items i WHERE i.session_id = s.id
                    AND i.status = 'pending') AS pending_count
            FROM sort_sessions s JOIN collections c ON s.collection_id = c.id
        """
        params: List[Any] = []
        if collection_id is not None:
            sql += " WHERE s.collection_id = ?"
            params.append(collection_id)
        sql += " ORDER BY s.created_at DESC"
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_with_json(row, "settings") for row in rows]

    def update_sort_session(self, session_id: int, **fields: Any) -> None:
        allowed = {"name", "status", "committed_at", "settings"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return
        if "settings" in updates and not isinstance(updates["settings"], str):
            updates["settings"] = json.dumps(updates["settings"])
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connection() as conn:
            conn.execute(
                f"UPDATE sort_sessions SET {assignments} WHERE id = ?",
                [*list(updates.values()), session_id],
            )
            conn.commit()

    def delete_sort_session(self, session_id: int) -> bool:
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM sort_sessions WHERE id = ?", (session_id,))
            conn.commit()
            return cursor.rowcount > 0

    def upsert_sort_items(self, session_id: int, items: List[Dict[str, Any]]) -> None:
        """Write suggestions for a session, replacing any previous suggestion.

        Human decisions are preserved: only pending rows are overwritten.
        """
        if not items:
            return
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO sort_items
                    (session_id, track_id, suggested_group_id, confidence, margin,
                     distance, ranked, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id, track_id) DO UPDATE SET
                    suggested_group_id = excluded.suggested_group_id,
                    confidence = excluded.confidence,
                    margin = excluded.margin,
                    distance = excluded.distance,
                    ranked = excluded.ranked,
                    status = CASE WHEN sort_items.status = 'pending'
                                  THEN excluded.status ELSE sort_items.status END
                """,
                [
                    (
                        session_id,
                        item["track_id"],
                        item.get("suggested_group_id"),
                        item.get("confidence"),
                        item.get("margin"),
                        item.get("distance"),
                        json.dumps(item.get("ranked", [])),
                        item.get("status", "pending"),
                    )
                    for item in items
                ],
            )
            conn.commit()

    def list_sort_items(
        self,
        session_id: int,
        status: Optional[str] = None,
        limit: int = 500,
        offset: int = 0,
    ) -> List[Dict]:
        sql = """
            SELECT i.*, t.filepath, t.filename, t.title, t.artist, t.album, t.genre,
                   t.duration, t.tag_bpm, t.tag_key,
                   gs.name AS suggested_group_name, gf.name AS final_group_name
            FROM sort_items i
            JOIN tracks t ON i.track_id = t.id
            LEFT JOIN groups gs ON i.suggested_group_id = gs.id
            LEFT JOIN groups gf ON i.final_group_id = gf.id
            WHERE i.session_id = ?
        """
        params: List[Any] = [session_id]
        if status:
            sql += " AND i.status = ?"
            params.append(status)
        sql += " ORDER BY i.confidence DESC, t.artist, t.title LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        with self.connection() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_with_json(row, "ranked", default=[]) for row in rows]

    def get_sort_item(self, item_id: int) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM sort_items WHERE id = ?", (item_id,)).fetchone()
        return _with_json(row, "ranked", default=[]) if row else None

    def decide_sort_item(self, item_id: int, status: str, final_group_id: Optional[int]) -> None:
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE sort_items SET status = ?, final_group_id = ?, decided_at = ?
                WHERE id = ?
                """,
                (status, final_group_id, utc_now(), item_id),
            )
            conn.commit()

    def mark_items_applied(self, item_ids: Sequence[int]) -> None:
        if not item_ids:
            return
        with self.connection() as conn:
            conn.executemany(
                "UPDATE sort_items SET applied = 1 WHERE id = ?",
                [(item_id,) for item_id in item_ids],
            )
            conn.commit()

    def sort_session_summary(self, session_id: int) -> Dict[str, int]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM sort_items WHERE session_id = ? GROUP BY status",
                (session_id,),
            ).fetchall()
        return {row["status"]: row["n"] for row in rows}

    # ------------------------------------------------------------------
    # File actions (undo log)
    # ------------------------------------------------------------------

    def log_file_actions(self, actions: List[Dict[str, Any]]) -> None:
        if not actions:
            return
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT INTO file_actions
                    (session_id, track_id, group_id, action, source_path, dest_path)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        action.get("session_id"),
                        action.get("track_id"),
                        action.get("group_id"),
                        action["action"],
                        action.get("source_path"),
                        action.get("dest_path"),
                    )
                    for action in actions
                ],
            )
            conn.commit()

    def list_file_actions(self, session_id: int, include_undone: bool = False) -> List[Dict]:
        sql = "SELECT * FROM file_actions WHERE session_id = ?"
        if not include_undone:
            sql += " AND undone = 0"
        sql += " ORDER BY id DESC"
        with self.connection() as conn:
            rows = conn.execute(sql, (session_id,)).fetchall()
        return [dict(row) for row in rows]

    def mark_actions_undone(self, action_ids: Sequence[int]) -> None:
        if not action_ids:
            return
        with self.connection() as conn:
            conn.executemany(
                "UPDATE file_actions SET undone = 1 WHERE id = ?",
                [(action_id,) for action_id in action_ids],
            )
            conn.commit()

    def update_track_path(self, track_id: int, filepath: str, filename: str) -> None:
        """Follow a track after its file moved on disk."""
        with self.connection() as conn:
            conn.execute(
                "UPDATE tracks SET filepath = ?, filename = ? WHERE id = ?",
                (filepath, filename, track_id),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------

    def create_discovery_run(
        self,
        name: Optional[str],
        source_path: Optional[str],
        algorithm: str,
        params: Dict[str, Any],
        metrics: Dict[str, Any],
    ) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO discovery_runs (name, source_path, algorithm, params, metrics)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, source_path, algorithm, json.dumps(params), json.dumps(metrics)),
            )
            conn.commit()
            return cursor.lastrowid

    def get_discovery_run(self, run_id: int) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute("SELECT * FROM discovery_runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            return None
        run = dict(row)
        run["params"] = json.loads(run["params"] or "{}")
        run["metrics"] = json.loads(run["metrics"] or "{}")
        return run

    def list_discovery_runs(self) -> List[Dict]:
        with self.connection() as conn:
            rows = conn.execute(
                """
                SELECT r.*,
                    (SELECT COUNT(*) FROM discovery_candidates c WHERE c.run_id = r.id) AS candidate_count
                FROM discovery_runs r ORDER BY r.created_at DESC
                """
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_discovery_run(self, run_id: int) -> bool:
        with self.connection() as conn:
            cursor = conn.execute("DELETE FROM discovery_runs WHERE id = ?", (run_id,))
            conn.commit()
            return cursor.rowcount > 0

    def add_discovery_candidate(
        self,
        run_id: int,
        candidate_index: int,
        suggested_name: str,
        size: int,
        centroid: np.ndarray,
        exemplars: List[int],
        stats: Dict[str, Any],
    ) -> int:
        with self.connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO discovery_candidates
                    (run_id, candidate_index, suggested_name, size, centroid, exemplars, stats)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    candidate_index,
                    suggested_name,
                    size,
                    pickle.dumps(np.asarray(centroid)),
                    json.dumps(exemplars),
                    json.dumps(stats),
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def add_discovery_members(self, candidate_id: int, members: List[Tuple[int, float]]) -> None:
        if not members:
            return
        with self.connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO discovery_members (candidate_id, track_id, distance)
                VALUES (?, ?, ?)
                """,
                [(candidate_id, track_id, distance) for track_id, distance in members],
            )
            conn.commit()

    def list_discovery_candidates(self, run_id: int) -> List[Dict]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM discovery_candidates WHERE run_id = ? ORDER BY size DESC",
                (run_id,),
            ).fetchall()
        candidates = []
        for row in rows:
            candidate = dict(row)
            candidate["exemplars"] = json.loads(candidate["exemplars"] or "[]")
            candidate["stats"] = json.loads(candidate["stats"] or "{}")
            candidate.pop("centroid", None)
            candidates.append(candidate)
        return candidates

    def get_discovery_candidate(self, candidate_id: int) -> Optional[Dict]:
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM discovery_candidates WHERE id = ?", (candidate_id,)
            ).fetchone()
        if not row:
            return None
        candidate = dict(row)
        candidate["exemplars"] = json.loads(candidate["exemplars"] or "[]")
        candidate["stats"] = json.loads(candidate["stats"] or "{}")
        if candidate.get("centroid"):
            candidate["centroid"] = pickle.loads(candidate["centroid"])
        return candidate

    def get_discovery_member_ids(self, candidate_id: int) -> List[int]:
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT track_id FROM discovery_members WHERE candidate_id = ? ORDER BY distance",
                (candidate_id,),
            ).fetchall()
        return [row[0] for row in rows]

    def update_discovery_candidate(self, candidate_id: int, **fields: Any) -> None:
        allowed = {"suggested_name", "status", "promoted_group_id"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connection() as conn:
            conn.execute(
                f"UPDATE discovery_candidates SET {assignments} WHERE id = ?",
                [*list(updates.values()), candidate_id],
            )
            conn.commit()

    def update_discovery_run(self, run_id: int, **fields: Any) -> None:
        allowed = {"name", "status"}
        updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
        if not updates:
            return
        assignments = ", ".join(f"{key} = ?" for key in updates)
        with self.connection() as conn:
            conn.execute(
                f"UPDATE discovery_runs SET {assignments} WHERE id = ?",
                [*list(updates.values()), run_id],
            )
            conn.commit()


def _stack(vectors: List[np.ndarray], track_ids: List[int]) -> Tuple[np.ndarray, List[int]]:
    """Stack feature vectors, dropping any whose dimensionality is the odd one out.

    Vectors extracted under different settings cannot share a matrix; the
    majority dimensionality wins and the rest are reported as missing so the
    caller can re-analyse them.
    """
    if not vectors:
        return np.empty((0, 0)), []

    dims = [len(vector) for vector in vectors]
    modal_dim = max(set(dims), key=dims.count)
    keep = [i for i, dim in enumerate(dims) if dim == modal_dim]
    matrix = np.vstack([vectors[i] for i in keep])
    return matrix, [track_ids[i] for i in keep]


def _with_json(row: sqlite3.Row, field: str, default: Any = None) -> Dict:
    """Row to dict, decoding one JSON-encoded column."""
    data = dict(row)
    raw = data.get(field)
    if isinstance(raw, str) and raw:
        try:
            data[field] = json.loads(raw)
        except json.JSONDecodeError:
            data[field] = default if default is not None else {}
    elif raw is None:
        data[field] = default if default is not None else {}
    return data
