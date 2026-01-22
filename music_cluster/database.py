"""Database layer for music-cluster."""

import pickle
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class Database:
    """SQLite database manager for music-cluster."""
    
    def __init__(self, db_path: str):
        """Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self) -> None:
        """Initialize database schema if it doesn't exist."""
        with self.connection() as conn:
            cursor = conn.cursor()
            
            # Create tracks table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filepath TEXT UNIQUE NOT NULL,
                    filename TEXT NOT NULL,
                    duration REAL,
                    file_size INTEGER,
                    checksum TEXT,
                    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    analysis_version TEXT
                )
            """)
            
            # Create features table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS features (
                    track_id INTEGER PRIMARY KEY,
                    feature_vector BLOB NOT NULL,
                    feature_dim INTEGER,
                    normalized BOOLEAN DEFAULT 0,
                    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE
                )
            """)
            
            # Create clusterings table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clusterings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    algorithm TEXT DEFAULT 'kmeans',
                    num_clusters INTEGER,
                    parameters TEXT,
                    silhouette_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create clusters table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS clusters (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    clustering_id INTEGER NOT NULL,
                    cluster_index INTEGER NOT NULL,
                    name TEXT,
                    size INTEGER,
                    representative_track_id INTEGER,
                    centroid BLOB,
                    FOREIGN KEY (clustering_id) REFERENCES clusterings(id) ON DELETE CASCADE,
                    FOREIGN KEY (representative_track_id) REFERENCES tracks(id),
                    UNIQUE(clustering_id, cluster_index)
                )
            """)
            
            # Add name column if it doesn't exist (migration)
            cursor.execute("PRAGMA table_info(clusters)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'name' not in columns:
                cursor.execute("ALTER TABLE clusters ADD COLUMN name TEXT")
            
            # Create cluster_members table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cluster_members (
                    cluster_id INTEGER NOT NULL,
                    track_id INTEGER NOT NULL,
                    distance_to_centroid REAL,
                    FOREIGN KEY (cluster_id) REFERENCES clusters(id) ON DELETE CASCADE,
                    FOREIGN KEY (track_id) REFERENCES tracks(id) ON DELETE CASCADE,
                    PRIMARY KEY (cluster_id, track_id)
                )
            """)
            
            # Create indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_tracks_filepath ON tracks(filepath)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_cluster_members_track ON cluster_members(track_id)")
            
            conn.commit()
    
    @contextmanager
    def connection(self):
        """Context manager for database connection.
        
        Yields:
            sqlite3.Connection
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    # Track operations
    
    def add_track(self, filepath: str, filename: str, duration: Optional[float] = None,
                  file_size: Optional[int] = None, checksum: Optional[str] = None,
                  analysis_version: Optional[str] = None) -> int:
        """Add a track to the database.
        
        Args:
            filepath: Absolute path to the track
            filename: Filename only
            duration: Track duration in seconds
            file_size: File size in bytes
            checksum: MD5 checksum
            analysis_version: Version of analysis used
            
        Returns:
            Track ID
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO tracks 
                (filepath, filename, duration, file_size, checksum, analysis_version, analyzed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (filepath, filename, duration, file_size, checksum, analysis_version, datetime.now()))
            conn.commit()
            return cursor.lastrowid
    
    def get_track_by_filepath(self, filepath: str) -> Optional[Dict]:
        """Get track by filepath.
        
        Args:
            filepath: Track filepath
            
        Returns:
            Track dictionary or None
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE filepath = ?", (filepath,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_track_by_id(self, track_id: int) -> Optional[Dict]:
        """Get track by ID.
        
        Args:
            track_id: Track ID
            
        Returns:
            Track dictionary or None
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks WHERE id = ?", (track_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_all_tracks(self) -> List[Dict]:
        """Get all tracks.
        
        Returns:
            List of track dictionaries
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM tracks ORDER BY filepath")
            return [dict(row) for row in cursor.fetchall()]
    
    def count_tracks(self) -> int:
        """Count total number of tracks.
        
        Returns:
            Number of tracks
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM tracks")
            return cursor.fetchone()[0]
    
    # Feature operations
    
    def add_features(self, track_id: int, feature_vector: np.ndarray, normalized: bool = False) -> None:
        """Add features for a track.
        
        Args:
            track_id: Track ID
            feature_vector: Feature vector as numpy array
            normalized: Whether features are normalized
        """
        feature_blob = pickle.dumps(feature_vector)
        feature_dim = len(feature_vector)
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO features 
                (track_id, feature_vector, feature_dim, normalized)
                VALUES (?, ?, ?, ?)
            """, (track_id, feature_blob, feature_dim, 1 if normalized else 0))
            conn.commit()
    
    def get_features(self, track_id: int) -> Optional[np.ndarray]:
        """Get features for a track.
        
        Args:
            track_id: Track ID
            
        Returns:
            Feature vector or None
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT feature_vector FROM features WHERE track_id = ?", (track_id,))
            row = cursor.fetchone()
            if row:
                return pickle.loads(row[0])
            return None
    
    def get_all_features(self) -> Tuple[np.ndarray, List[int]]:
        """Get all feature vectors and their track IDs.
        
        Returns:
            Tuple of (feature_matrix, track_ids)
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT f.track_id, f.feature_vector 
                FROM features f
                JOIN tracks t ON f.track_id = t.id
                ORDER BY f.track_id
            """)
            rows = cursor.fetchall()
            
            if not rows:
                return np.array([]), []
            
            track_ids = [row[0] for row in rows]
            features = [pickle.loads(row[1]) for row in rows]
            feature_matrix = np.array(features)
            
            return feature_matrix, track_ids
    
    def count_features(self) -> int:
        """Count tracks with features.
        
        Returns:
            Number of tracks with features
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM features")
            return cursor.fetchone()[0]
    
    # Clustering operations
    
    def add_clustering(self, name: Optional[str] = None, algorithm: str = "kmeans",
                      num_clusters: int = 0, parameters: Optional[str] = None,
                      silhouette_score: Optional[float] = None) -> int:
        """Add a clustering.
        
        Args:
            name: Clustering name
            algorithm: Algorithm used
            num_clusters: Number of clusters
            parameters: JSON parameters
            silhouette_score: Quality score
            
        Returns:
            Clustering ID
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO clusterings 
                (name, algorithm, num_clusters, parameters, silhouette_score, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (name, algorithm, num_clusters, parameters, silhouette_score, datetime.now()))
            conn.commit()
            return cursor.lastrowid
    
    def delete_clustering(self, clustering_id: int) -> bool:
        """Delete a clustering and all its clusters (cascade delete).
        
        Args:
            clustering_id: Clustering ID to delete
            
        Returns:
            True if deleted, False if not found
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            # Check if clustering exists
            cursor.execute("SELECT id FROM clusterings WHERE id = ?", (clustering_id,))
            if not cursor.fetchone():
                return False
            
            # Delete clustering (cascade will delete clusters and cluster_members)
            cursor.execute("DELETE FROM clusterings WHERE id = ?", (clustering_id,))
            conn.commit()
            return True
    
    def get_clustering(self, clustering_id: Optional[int] = None, name: Optional[str] = None) -> Optional[Dict]:
        """Get clustering by ID or name.
        
        Args:
            clustering_id: Clustering ID
            name: Clustering name
            
        Returns:
            Clustering dictionary or None
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            if clustering_id is not None:
                cursor.execute("SELECT * FROM clusterings WHERE id = ?", (clustering_id,))
            elif name is not None:
                cursor.execute("SELECT * FROM clusterings WHERE name = ?", (name,))
            else:
                # Get latest
                cursor.execute("SELECT * FROM clusterings ORDER BY created_at DESC LIMIT 1")
            
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def get_all_clusterings(self) -> List[Dict]:
        """Get all clusterings.
        
        Returns:
            List of clustering dictionaries
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clusterings ORDER BY created_at DESC")
            return [dict(row) for row in cursor.fetchall()]
    
    def add_cluster(self, clustering_id: int, cluster_index: int, size: int,
                   representative_track_id: Optional[int] = None,
                   centroid: Optional[np.ndarray] = None, name: Optional[str] = None) -> int:
        """Add a cluster.
        
        Args:
            clustering_id: Clustering ID
            cluster_index: Cluster index (0-based)
            size: Number of tracks in cluster
            representative_track_id: ID of representative track
            centroid: Centroid vector
            name: Optional cluster name
            
        Returns:
            Cluster ID
        """
        centroid_blob = pickle.dumps(centroid) if centroid is not None else None
        
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO clusters 
                (clustering_id, cluster_index, size, representative_track_id, centroid, name)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (clustering_id, cluster_index, size, representative_track_id, centroid_blob, name))
            conn.commit()
            return cursor.lastrowid
    
    def get_clusters_by_clustering(self, clustering_id: int) -> List[Dict]:
        """Get all clusters for a clustering.
        
        Args:
            clustering_id: Clustering ID
            
        Returns:
            List of cluster dictionaries
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM clusters 
                WHERE clustering_id = ?
                ORDER BY cluster_index
            """, (clustering_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_cluster(self, cluster_id: int) -> Optional[Dict]:
        """Get cluster by ID.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            Cluster dictionary or None
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM clusters WHERE id = ?", (cluster_id,))
            row = cursor.fetchone()
            if row:
                cluster = dict(row)
                if cluster['centroid']:
                    cluster['centroid'] = pickle.loads(cluster['centroid'])
                return cluster
            return None
    
    def add_cluster_member(self, cluster_id: int, track_id: int, distance: float) -> None:
        """Add a track to a cluster.
        
        Args:
            cluster_id: Cluster ID
            track_id: Track ID
            distance: Distance to centroid
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO cluster_members 
                (cluster_id, track_id, distance_to_centroid)
                VALUES (?, ?, ?)
            """, (cluster_id, track_id, distance))
            conn.commit()
    
    def get_cluster_members(self, cluster_id: int) -> List[Dict]:
        """Get all tracks in a cluster.
        
        Args:
            cluster_id: Cluster ID
            
        Returns:
            List of track dictionaries with distance info
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT t.*, cm.distance_to_centroid
                FROM cluster_members cm
                JOIN tracks t ON cm.track_id = t.id
                WHERE cm.cluster_id = ?
                ORDER BY cm.distance_to_centroid
            """, (cluster_id,))
            return [dict(row) for row in cursor.fetchall()]
    
    def get_track_cluster(self, track_id: int, clustering_id: int) -> Optional[int]:
        """Get cluster ID for a track in a specific clustering.
        
        Args:
            track_id: Track ID
            clustering_id: Clustering ID
            
        Returns:
            Cluster ID or None
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT cm.cluster_id
                FROM cluster_members cm
                JOIN clusters c ON cm.cluster_id = c.id
                WHERE cm.track_id = ? AND c.clustering_id = ?
            """, (track_id, clustering_id))
            row = cursor.fetchone()
            return row[0] if row else None
    
    def update_cluster_name(self, cluster_id: int, name: str) -> None:
        """Update cluster name.
        
        Args:
            cluster_id: Cluster ID
            name: New cluster name
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("UPDATE clusters SET name = ? WHERE id = ?", (name, cluster_id))
            conn.commit()
    
    def get_cluster_by_index(self, clustering_id: int, cluster_index: int) -> Optional[Dict]:
        """Get cluster by clustering ID and cluster index.
        
        Args:
            clustering_id: Clustering ID
            cluster_index: Cluster index
            
        Returns:
            Cluster dictionary or None
        """
        with self.connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM clusters 
                WHERE clustering_id = ? AND cluster_index = ?
            """, (clustering_id, cluster_index))
            row = cursor.fetchone()
            if row:
                cluster = dict(row)
                if cluster['centroid']:
                    cluster['centroid'] = pickle.loads(cluster['centroid'])
                return cluster
            return None
